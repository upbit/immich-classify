"""Task execution engine with async concurrency, pause/resume, and progress tracking."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from typing import Any

from loguru import logger
from tqdm import tqdm

from immich_classify.config import Config
from immich_classify.database import Database
from immich_classify.immich_client import Asset, ImmichClient
from immich_classify.prompt_base import BasePrompt
from immich_classify.vlm_client import VLMClient, VLMError


_tqdm_sink_installed = False


def _install_tqdm_log_sink() -> None:
    """Route loguru output through ``tqdm.write`` so scrolling log lines do
    not clobber the sticky progress bar at the bottom of the terminal.

    Safe to call repeatedly — the default loguru sink is only replaced once.
    Non-TTY stderr (pipes, redirects, CI) is left alone so log files don't end
    up with bare carriage returns from tqdm.
    """
    global _tqdm_sink_installed
    if _tqdm_sink_installed:
        return
    if not sys.stderr.isatty():
        _tqdm_sink_installed = True
        return

    def _sink(message: Any) -> None:  # loguru Message is str-like
        tqdm.write(str(message), end="")

    # Replace the default stderr sink so the two don't fight for the last line.
    logger.remove()
    logger.add(_sink, colorize=True)
    _tqdm_sink_installed = True


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

        # Insert pending results for all assets (with archived/trashed flags)
        asset_ids = [asset.asset_id for asset in all_assets]
        asset_flags = {
            asset.asset_id: (asset.is_archived, asset.is_trashed)
            for asset in all_assets
        }
        await self._database.insert_pending_results(task_id, asset_ids, asset_flags)

        # Run the task
        await self._run_task(task_id, prompt_config, effective_concurrency)
        return task_id

    async def resume_task(
        self,
        task_id: str,
        concurrency: int | None = None,
    ) -> None:
        """Resume a paused, failed, or completed task.

        A "completed" task can still be resumed to retry any assets whose
        per-asset status is not 'success' (i.e. previously failed items).

        Args:
            task_id: The task to resume.
            concurrency: Override concurrency from config.
        """
        task = await self._database.get_task(task_id)
        if task is None:
            logger.error("Task {} not found", task_id)
            return

        if task["status"] not in ("paused", "failed", "pending", "completed"):
            logger.error(
                "Task {} has status '{}', can only resume paused/failed/pending/completed tasks",
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

        # Retry policy: on every run/resume, previously-failed rows are flipped
        # back to 'pending' so they are reprocessed, and the task-level counters
        # are rebuilt from the per-asset rows (the ground truth). Without this
        # step, the counters drift on each resume — a retried failure bumps
        # failed_count again even though the row transitions success→failed→
        # success, producing the "Progress:464/2189, Failed:2124" pattern.
        reset_count = await self._database.reset_failed_results_to_pending(task_id)
        completed_so_far, failed_so_far = await self._database.reset_counts_from_results(task_id)
        if reset_count:
            logger.info(
                "Task {} resuming: retrying {} previously-failed asset(s); "
                "counters reset to completed={}, failed={}",
                task_id,
                reset_count,
                completed_so_far,
                failed_so_far,
            )

        pending_asset_ids = await self._database.get_pending_asset_ids(task_id)
        if not pending_asset_ids:
            logger.info("Task {} has no pending assets, marking as completed", task_id)
            await self._database.update_task_status(task_id, "completed")
            return

        task = await self._database.get_task(task_id)
        assert task is not None
        total = task["total_count"]

        logger.info(
            "Starting task {} with {} pending assets (concurrency={})",
            task_id,
            len(pending_asset_ids),
            concurrency,
        )

        semaphore = asyncio.Semaphore(concurrency)
        counter = _Counter(completed_so_far)

        # Sticky bottom progress bar. ``leave=True`` so the completed bar stays
        # visible after the run; ``dynamic_ncols`` so it adapts to terminal
        # resizes. Initial position is the already-successful count so the bar
        # picks up exactly where the last run left off.
        _install_tqdm_log_sink()
        progress: Any = tqdm(
            total=total,
            initial=completed_so_far,
            unit="img",
            desc="classify",
            dynamic_ncols=True,
            leave=True,
            mininterval=0.2,
            postfix={"ok": completed_so_far, "fail": 0},
        )
        stats: dict[str, int] = {"ok": completed_so_far, "fail": 0}

        async def process_one(asset_id: str) -> None:
            if self._cancelled:
                return
            async with semaphore:
                if self._cancelled:
                    return
                await self._process_single_asset(
                    task_id, asset_id, prompt_config, counter, total, progress, stats
                )

        tasks = [asyncio.create_task(process_one(aid)) for aid in pending_asset_ids]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            progress.close()

        if self._cancelled:
            await self._database.update_task_status(task_id, "paused")
            logger.info(
                "Task {} paused ({}/{} completed, {} failed this run)",
                task_id,
                counter.value,
                total,
                stats["fail"],
            )
        else:
            await self._database.update_task_status(task_id, "completed")
            logger.info(
                "Task {} completed ({}/{}, {} failed this run)",
                task_id,
                counter.value,
                total,
                stats["fail"],
            )

    async def _process_single_asset(
        self,
        task_id: str,
        asset_id: str,
        prompt_config: BasePrompt,
        counter: _Counter,
        total: int,
        progress: Any,
        stats: dict[str, int],
    ) -> None:
        """Process a single image asset: download, classify, save result.

        Args:
            task_id: The owning task.
            asset_id: The Immich asset ID.
            prompt_config: Classification prompt.
            counter: Shared progress counter (successes only).
            total: Total number of assets in the task.
            progress: Sticky tqdm bar at the bottom of the terminal.
            stats: Shared ``{"ok": int, "fail": int}`` counters for the bar
                postfix. ``ok`` mirrors ``counter.value`` but is plumbed
                through so the bar update is a single dict write.
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
            stats["ok"] += 1
            progress.update(1)
            progress.set_postfix(stats, refresh=False)

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
            stats["fail"] += 1
            progress.update(1)
            progress.set_postfix(stats, refresh=False)
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
            stats["fail"] += 1
            progress.update(1)
            progress.set_postfix(stats, refresh=False)
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
