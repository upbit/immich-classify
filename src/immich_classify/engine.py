"""Task execution engine with async concurrency, pause/resume, and progress tracking."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from loguru import logger

from immich_classify.config import Config
from immich_classify.database import Database
from immich_classify.immich_client import Asset, ImmichClient
from immich_classify.prompt_base import BasePrompt
from immich_classify.vlm_client import VLMClient, VLMError


class TaskEngine:
    """Async engine for executing classification tasks with concurrency control."""

    def __init__(
        self,
        config: Config,
        database: Database,
        immich_client: ImmichClient,
        vlm_client: VLMClient,
    ) -> None:
        self._config = config
        self._database = database
        self._immich_client = immich_client
        self._vlm_client = vlm_client
        self._cancelled = False

    def request_cancel(self) -> None:
        """Signal the engine to stop processing (for Ctrl+C handling)."""
        self._cancelled = True

    async def create_and_run_task(
        self,
        album_ids: list[str],
        prompt_config: BasePrompt,
        concurrency: int | None = None,
    ) -> str:
        """Create a new task and run it.

        Args:
            album_ids: List of Immich album IDs to classify.
            prompt_config: The prompt and schema configuration.
            concurrency: Override concurrency from config.

        Returns:
            The task ID.
        """
        task_id = str(uuid.uuid4())
        effective_concurrency = concurrency if concurrency is not None else self._config.concurrency

        # Collect all assets from all albums, deduplicating by asset_id
        seen_asset_ids: set[str] = set()
        all_assets: list[Asset] = []
        for album_id in album_ids:
            assets = await self._immich_client.get_album_assets(album_id)
            for asset in assets:
                if asset.asset_id not in seen_asset_ids:
                    seen_asset_ids.add(asset.asset_id)
                    all_assets.append(asset)

        if not all_assets:
            logger.warning("No image assets found in the specified albums")
            return task_id

        # Create task in database
        await self._database.create_task(
            task_id=task_id,
            album_ids=album_ids,
            prompt_config=prompt_config.to_dict(),
            total_count=len(all_assets),
        )

        # Insert pending results for all assets
        asset_ids = [asset.asset_id for asset in all_assets]
        await self._database.insert_pending_results(task_id, asset_ids)

        # Run the task
        await self._run_task(task_id, prompt_config, effective_concurrency)
        return task_id

    async def resume_task(
        self,
        task_id: str,
        concurrency: int | None = None,
    ) -> None:
        """Resume a paused or failed task.

        Args:
            task_id: The task to resume.
            concurrency: Override concurrency from config.
        """
        task = await self._database.get_task(task_id)
        if task is None:
            logger.error("Task {} not found", task_id)
            return

        if task["status"] not in ("paused", "failed", "pending"):
            logger.error(
                "Task {} has status '{}', can only resume paused/failed/pending tasks",
                task_id,
                task["status"],
            )
            return

        prompt_config = BasePrompt.from_dict(json.loads(task["prompt_config"]))
        effective_concurrency = concurrency if concurrency is not None else self._config.concurrency
        self._cancelled = False

        await self._run_task(task_id, prompt_config, effective_concurrency)

    async def _run_task(
        self,
        task_id: str,
        prompt_config: BasePrompt,
        concurrency: int,
    ) -> None:
        """Execute the classification task.

        Args:
            task_id: The task to run.
            prompt_config: Classification prompt configuration.
            concurrency: Maximum number of concurrent image processing workers.
        """
        await self._database.update_task_status(task_id, "running")

        pending_asset_ids = await self._database.get_pending_asset_ids(task_id)
        if not pending_asset_ids:
            logger.info("Task {} has no pending assets, marking as completed", task_id)
            await self._database.update_task_status(task_id, "completed")
            return

        task = await self._database.get_task(task_id)
        assert task is not None
        total = task["total_count"]
        completed_so_far = task["completed_count"] + task["failed_count"]

        logger.info(
            "Starting task {} with {} pending assets (concurrency={})",
            task_id,
            len(pending_asset_ids),
            concurrency,
        )

        semaphore = asyncio.Semaphore(concurrency)
        counter = _Counter(completed_so_far)

        async def process_one(asset_id: str) -> None:
            if self._cancelled:
                return
            async with semaphore:
                if self._cancelled:
                    return
                await self._process_single_asset(task_id, asset_id, prompt_config, counter, total)

        tasks = [asyncio.create_task(process_one(aid)) for aid in pending_asset_ids]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

        if self._cancelled:
            await self._database.update_task_status(task_id, "paused")
            logger.info("Task {} paused ({}/{} completed)", task_id, counter.value, total)
        else:
            await self._database.update_task_status(task_id, "completed")
            logger.info("Task {} completed ({}/{})", task_id, counter.value, total)

    async def _process_single_asset(
        self,
        task_id: str,
        asset_id: str,
        prompt_config: BasePrompt,
        counter: _Counter,
        total: int,
    ) -> None:
        """Process a single image asset: download, classify, save result.

        Args:
            task_id: The owning task.
            asset_id: The Immich asset ID.
            prompt_config: Classification prompt.
            counter: Shared progress counter.
            total: Total number of assets in the task.
        """
        try:
            image_base64, content_type = await self._immich_client.download_image_base64(
                asset_id, self._config.image_size
            )
            result = await self._vlm_client.classify_image(
                image_base64, prompt_config, content_type=content_type
            )

            await self._database.save_result(
                task_id=task_id,
                asset_id=asset_id,
                result_json=result,
                raw_response=json.dumps(result),
            )
            await self._database.increment_task_completed(task_id)
            counter.increment()

            # Build a summary of the result for logging
            summary_parts: list[str] = []
            for key, value in result.items():
                if isinstance(value, list):
                    summary_parts.append(f"{key}: {value}")
                else:
                    summary_parts.append(f"{key}: {value}")
            summary = ", ".join(summary_parts)

            logger.info(
                "[{}/{}] asset_id: {} - {}",
                counter.value,
                total,
                asset_id,
                summary,
            )

        except VLMError as exc:
            await self._database.save_error(
                task_id=task_id,
                asset_id=asset_id,
                error_message=str(exc),
                raw_response=exc.raw_response,
            )
            await self._database.increment_task_failed(task_id)
            counter.increment()
            logger.warning(
                "[{}/{}] asset_id: {} - FAILED: {}",
                counter.value,
                total,
                asset_id,
                exc,
            )

        except Exception as exc:
            await self._database.save_error(
                task_id=task_id,
                asset_id=asset_id,
                error_message=str(exc),
            )
            await self._database.increment_task_failed(task_id)
            counter.increment()
            logger.error(
                "[{}/{}] asset_id: {} - ERROR: {}",
                counter.value,
                total,
                asset_id,
                exc,
            )


async def debug_classify(
    config: Config,
    immich_client: ImmichClient,
    vlm_client: VLMClient,
    album_id: str,
    prompt_config: BasePrompt,
    count: int = 10,
) -> list[dict[str, Any]]:
    """Run a small debug batch of classifications without writing to the database.

    Args:
        config: Application configuration.
        immich_client: Immich API client.
        vlm_client: VLM client.
        album_id: Album to sample from.
        prompt_config: Classification prompt configuration.
        count: Number of images to classify.

    Returns:
        List of classification results with asset IDs.
    """
    import random

    assets = await immich_client.get_album_assets(album_id)
    if not assets:
        logger.warning("No image assets found in album {}", album_id)
        return []

    sample_size = min(count, len(assets))
    sample = random.sample(assets, sample_size)

    results: list[dict[str, Any]] = []

    logger.info("─── Debug Classification ───")
    logger.info("Album: {} | Sample size: {} | Image size: {}", album_id, sample_size, config.image_size)
    logger.info("VLM: {} | Model: {}", config.vlm_api_url, config.vlm_model_name or "(server default)")
    logger.info(
        "Schema fields: {}",
        ", ".join(f"{k} ({v.field_type})" for k, v in prompt_config.schema.items()),
    )
    logger.info("────────────────────────────")

    for i, asset in enumerate(sample, 1):
        try:
            image_base64, content_type = await immich_client.download_image_base64(
                asset.asset_id, config.image_size
            )
            result = await vlm_client.classify_image(image_base64, prompt_config, content_type=content_type)
            results.append({
                "asset_id": asset.asset_id,
                "file_name": asset.original_file_name,
                "status": "success",
                "result": result,
            })
            logger.info(
                "[{}/{}] {} ({}) - {}",
                i,
                sample_size,
                asset.asset_id,
                asset.original_file_name,
                result,
            )
        except VLMError as exc:
            results.append({
                "asset_id": asset.asset_id,
                "file_name": asset.original_file_name,
                "status": "failed",
                "error": str(exc),
                "raw_response": exc.raw_response,
            })
            logger.warning(
                "[{}/{}] {} ({}) - FAILED: {}",
                i,
                sample_size,
                asset.asset_id,
                asset.original_file_name,
                exc,
            )
            if exc.raw_response:
                logger.debug("Raw VLM response:\n{}", exc.raw_response[:3000])
        except Exception as exc:
            results.append({
                "asset_id": asset.asset_id,
                "file_name": asset.original_file_name,
                "status": "error",
                "error": str(exc),
            })
            logger.error(
                "[{}/{}] {} ({}) - ERROR: {}",
                i,
                sample_size,
                asset.asset_id,
                asset.original_file_name,
                exc,
            )

    return results


class _Counter:
    """Simple thread-safe counter for progress tracking."""

    def __init__(self, initial: int = 0) -> None:
        self._value = initial

    @property
    def value(self) -> int:
        return self._value

    def increment(self) -> None:
        self._value += 1
