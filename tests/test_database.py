"""Tests for database operations."""

from __future__ import annotations

import pytest
import pytest_asyncio

from immich_classify.database import Database


@pytest_asyncio.fixture
async def db() -> Database:
    """Create an in-memory database for testing."""
    database = Database(":memory:")
    await database.connect()
    yield database  # type: ignore[misc]
    await database.close()


@pytest.mark.asyncio
async def test_create_and_get_task(db: Database) -> None:
    await db.create_task(
        task_id="task-001",
        album_ids=["album-a", "album-b"],
        prompt_config={"system_prompt": "test"},
        total_count=100,
    )
    task = await db.get_task("task-001")
    assert task is not None
    assert task["task_id"] == "task-001"
    assert task["status"] == "pending"
    assert task["total_count"] == 100
    assert task["completed_count"] == 0
    assert task["failed_count"] == 0


@pytest.mark.asyncio
async def test_get_task_not_found(db: Database) -> None:
    task = await db.get_task("nonexistent")
    assert task is None


@pytest.mark.asyncio
async def test_get_all_tasks(db: Database) -> None:
    await db.create_task("task-1", ["a"], {}, 10)
    await db.create_task("task-2", ["b"], {}, 20)
    tasks = await db.get_all_tasks()
    assert len(tasks) == 2


@pytest.mark.asyncio
async def test_update_task_status(db: Database) -> None:
    await db.create_task("task-1", ["a"], {}, 10)
    await db.update_task_status("task-1", "running")
    task = await db.get_task("task-1")
    assert task is not None
    assert task["status"] == "running"


@pytest.mark.asyncio
async def test_increment_task_counts(db: Database) -> None:
    await db.create_task("task-1", ["a"], {}, 10)
    await db.increment_task_completed("task-1")
    await db.increment_task_completed("task-1")
    await db.increment_task_failed("task-1")

    task = await db.get_task("task-1")
    assert task is not None
    assert task["completed_count"] == 2
    assert task["failed_count"] == 1


@pytest.mark.asyncio
async def test_insert_pending_and_get_pending(db: Database) -> None:
    await db.create_task("task-1", ["a"], {}, 3)
    await db.insert_pending_results("task-1", ["img-1", "img-2", "img-3"])

    pending = await db.get_pending_asset_ids("task-1")
    assert len(pending) == 3
    assert set(pending) == {"img-1", "img-2", "img-3"}


@pytest.mark.asyncio
async def test_save_result_marks_success(db: Database) -> None:
    await db.create_task("task-1", ["a"], {}, 1)
    await db.insert_pending_results("task-1", ["img-1"])

    await db.save_result(
        task_id="task-1",
        asset_id="img-1",
        result_json={"category": "people", "quality": "high"},
        raw_response='{"category": "people", "quality": "high"}',
    )

    pending = await db.get_pending_asset_ids("task-1")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_save_error_marks_failed(db: Database) -> None:
    await db.create_task("task-1", ["a"], {}, 1)
    await db.insert_pending_results("task-1", ["img-1"])

    await db.save_error(
        task_id="task-1",
        asset_id="img-1",
        error_message="VLM timeout",
        raw_response=None,
    )

    # Failed results are still in pending list (status != 'success')
    pending = await db.get_pending_asset_ids("task-1")
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_get_results_with_filter(db: Database) -> None:
    await db.create_task("task-1", ["a"], {}, 3)
    await db.insert_pending_results("task-1", ["img-1", "img-2", "img-3"])

    await db.save_result("task-1", "img-1", {"category": "people"}, "{}")
    await db.save_result("task-1", "img-2", {"category": "landscape"}, "{}")
    await db.save_result("task-1", "img-3", {"category": "people"}, "{}")

    # No filter
    all_results = await db.get_results("task-1")
    assert len(all_results) == 3

    # Filter by category
    people = await db.get_results("task-1", filters={"category": "people"})
    assert len(people) == 2

    landscape = await db.get_results("task-1", filters={"category": "landscape"})
    assert len(landscape) == 1


@pytest.mark.asyncio
async def test_get_result_summary(db: Database) -> None:
    await db.create_task("task-1", ["a"], {}, 3)
    await db.insert_pending_results("task-1", ["img-1", "img-2", "img-3"])

    await db.save_result("task-1", "img-1", {"category": "people"}, "{}")
    await db.save_error("task-1", "img-2", "timeout")

    summary = await db.get_result_summary("task-1")
    assert summary.get("success") == 1
    assert summary.get("failed") == 1
    assert summary.get("pending") == 1


@pytest.mark.asyncio
async def test_insert_pending_ignores_duplicates(db: Database) -> None:
    await db.create_task("task-1", ["a"], {}, 2)
    await db.insert_pending_results("task-1", ["img-1", "img-2"])
    # Insert again - should not raise
    await db.insert_pending_results("task-1", ["img-1", "img-2", "img-3"])

    pending = await db.get_pending_asset_ids("task-1")
    assert len(pending) == 3
