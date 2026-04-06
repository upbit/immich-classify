"""Tests for task execution engine."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from immich_classify.config import Config
from immich_classify.database import Database
from immich_classify.engine import TaskEngine, debug_classify
from immich_classify.immich_client import Asset
from immich_classify.prompt import ClassificationPrompt
from immich_classify.vlm_client import VLMError


def _make_config() -> Config:
    return Config(
        immich_api_url="http://localhost:2283",
        immich_api_key="test-key",
        vlm_api_url="http://localhost:8000/v1",
        vlm_api_key="no-key",
        vlm_model_name="",
        database_path=":memory:",
        concurrency=2,
        timeout=60,
        image_size="thumbnail",
    )


def _make_mock_immich(assets: list[Asset]) -> MagicMock:
    client = MagicMock()
    client.get_album_assets = AsyncMock(return_value=assets)
    client.download_image_base64 = AsyncMock(return_value=("base64data", "image/jpeg"))
    return client


def _make_mock_vlm(result: dict[str, Any] | None = None, error: VLMError | None = None) -> MagicMock:
    client = MagicMock()
    if error is not None:
        client.classify_image = AsyncMock(side_effect=error)
    else:
        client.classify_image = AsyncMock(return_value=result or {"category": "people"})
    return client


@pytest_asyncio.fixture
async def db() -> Database:
    database = Database(":memory:")
    await database.connect()
    yield database  # type: ignore[misc]
    await database.close()


class TestTaskEngine:
    """Tests for TaskEngine."""

    @pytest.mark.asyncio
    async def test_create_and_run_task_success(self, db: Database) -> None:
        assets = [
            Asset(asset_id="img-1", original_file_name="a.jpg", asset_type="IMAGE"),
            Asset(asset_id="img-2", original_file_name="b.jpg", asset_type="IMAGE"),
        ]
        config = _make_config()
        immich = _make_mock_immich(assets)
        vlm = _make_mock_vlm({"category": "landscape", "quality": "high"})

        engine = TaskEngine(config, db, immich, vlm)
        task_id = await engine.create_and_run_task(
            album_ids=["album-1"],
            prompt_config=ClassificationPrompt(),
        )

        task = await db.get_task(task_id)
        assert task is not None
        assert task["status"] == "completed"
        assert task["total_count"] == 2
        assert task["completed_count"] == 2
        assert task["failed_count"] == 0

        results = await db.get_results(task_id)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_vlm_error_continues_processing(self, db: Database) -> None:
        assets = [
            Asset(asset_id="img-1", original_file_name="a.jpg", asset_type="IMAGE"),
            Asset(asset_id="img-2", original_file_name="b.jpg", asset_type="IMAGE"),
        ]
        config = _make_config()
        immich = _make_mock_immich(assets)

        # First call fails, second succeeds
        vlm = MagicMock()
        vlm.classify_image = AsyncMock(
            side_effect=[
                VLMError("timeout"),
                {"category": "food"},
            ]
        )

        engine = TaskEngine(config, db, immich, vlm)
        task_id = await engine.create_and_run_task(
            album_ids=["album-1"],
            prompt_config=ClassificationPrompt(),
        )

        task = await db.get_task(task_id)
        assert task is not None
        assert task["status"] == "completed"
        assert task["completed_count"] == 1
        assert task["failed_count"] == 1

    @pytest.mark.asyncio
    async def test_deduplicates_assets_across_albums(self, db: Database) -> None:
        """Assets appearing in multiple albums should only be processed once."""
        shared_asset = Asset(asset_id="img-shared", original_file_name="shared.jpg", asset_type="IMAGE")
        album1_assets = [
            Asset(asset_id="img-1", original_file_name="a.jpg", asset_type="IMAGE"),
            shared_asset,
        ]
        album2_assets = [
            shared_asset,
            Asset(asset_id="img-2", original_file_name="b.jpg", asset_type="IMAGE"),
        ]
        config = _make_config()
        immich = MagicMock()
        immich.get_album_assets = AsyncMock(side_effect=[album1_assets, album2_assets])
        immich.download_image_base64 = AsyncMock(return_value=("base64data", "image/jpeg"))
        vlm = _make_mock_vlm({"category": "people"})

        engine = TaskEngine(config, db, immich, vlm)
        task_id = await engine.create_and_run_task(
            album_ids=["album-1", "album-2"],
            prompt_config=ClassificationPrompt(),
        )

        task = await db.get_task(task_id)
        assert task is not None
        assert task["total_count"] == 3  # img-1, img-shared, img-2 (deduplicated)
        assert task["completed_count"] == 3

    @pytest.mark.asyncio
    async def test_empty_album_returns_quickly(self, db: Database) -> None:
        config = _make_config()
        immich = _make_mock_immich([])
        vlm = _make_mock_vlm()

        engine = TaskEngine(config, db, immich, vlm)
        task_id = await engine.create_and_run_task(
            album_ids=["empty-album"],
            prompt_config=ClassificationPrompt(),
        )

        # Task should not be created in DB since there are no assets
        task = await db.get_task(task_id)
        assert task is None

    @pytest.mark.asyncio
    async def test_cancel_pauses_task(self, db: Database) -> None:
        assets = [
            Asset(asset_id=f"img-{i}", original_file_name=f"{i}.jpg", asset_type="IMAGE")
            for i in range(20)
        ]
        config = _make_config()
        immich = _make_mock_immich(assets)

        call_count = 0

        async def slow_classify(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                engine.request_cancel()
            return {"category": "people"}

        vlm = MagicMock()
        vlm.classify_image = AsyncMock(side_effect=slow_classify)

        engine = TaskEngine(config, db, immich, vlm)
        task_id = await engine.create_and_run_task(
            album_ids=["album-1"],
            prompt_config=ClassificationPrompt(),
        )

        task = await db.get_task(task_id)
        assert task is not None
        assert task["status"] == "paused"
        # Should have processed some but not all
        assert task["completed_count"] < 20

    @pytest.mark.asyncio
    async def test_resume_task(self, db: Database) -> None:
        assets = [
            Asset(asset_id="img-1", original_file_name="a.jpg", asset_type="IMAGE"),
            Asset(asset_id="img-2", original_file_name="b.jpg", asset_type="IMAGE"),
        ]
        config = _make_config()
        immich = _make_mock_immich(assets)

        # First run: all fail
        vlm_fail = _make_mock_vlm(error=VLMError("timeout"))
        engine = TaskEngine(config, db, immich, vlm_fail)
        task_id = await engine.create_and_run_task(
            album_ids=["album-1"],
            prompt_config=ClassificationPrompt(),
        )

        task = await db.get_task(task_id)
        assert task is not None
        assert task["completed_count"] == 0
        assert task["failed_count"] == 2

        # Mark as paused for resume
        await db.update_task_status(task_id, "paused")

        # Resume with working VLM
        vlm_ok = _make_mock_vlm({"category": "people"})
        engine2 = TaskEngine(config, db, immich, vlm_ok)
        await engine2.resume_task(task_id)

        task = await db.get_task(task_id)
        assert task is not None
        assert task["status"] == "completed"


class TestDebugClassify:
    """Tests for debug_classify function."""

    @pytest.mark.asyncio
    async def test_debug_classify_returns_results(self) -> None:
        config = _make_config()
        assets = [
            Asset(asset_id="img-1", original_file_name="a.jpg", asset_type="IMAGE"),
            Asset(asset_id="img-2", original_file_name="b.jpg", asset_type="IMAGE"),
        ]
        immich = _make_mock_immich(assets)
        vlm = _make_mock_vlm({"category": "food", "quality": "high"})

        results = await debug_classify(
            config, immich, vlm,
            album_id="album-1",
            prompt_config=ClassificationPrompt(),
            count=5,
        )

        assert len(results) == 2  # Only 2 assets available
        assert all(r["status"] == "success" for r in results)

    @pytest.mark.asyncio
    async def test_debug_classify_handles_errors(self) -> None:
        config = _make_config()
        assets = [
            Asset(asset_id="img-1", original_file_name="a.jpg", asset_type="IMAGE"),
        ]
        immich = _make_mock_immich(assets)
        vlm = _make_mock_vlm(error=VLMError("timeout"))

        results = await debug_classify(
            config, immich, vlm,
            album_id="album-1",
            prompt_config=ClassificationPrompt(),
            count=5,
        )

        assert len(results) == 1
        assert results[0]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_debug_classify_empty_album(self) -> None:
        config = _make_config()
        immich = _make_mock_immich([])
        vlm = _make_mock_vlm()

        results = await debug_classify(
            config, immich, vlm,
            album_id="empty",
            prompt_config=ClassificationPrompt(),
        )

        assert len(results) == 0
