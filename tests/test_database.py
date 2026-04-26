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
async def test_get_results_filter_numeric_value(db: Database) -> None:
    """CLI passes filter values as strings; numeric JSON fields must still match.

    Regression test for the "no data" bug when filtering on integer fields like
    ``foreground_count=0`` — ``json_extract`` returns an INTEGER, so a bare
    string comparison never matches.
    """
    await db.create_task("task-1", ["a"], {}, 3)
    await db.insert_pending_results("task-1", ["img-1", "img-2", "img-3"])

    await db.save_result("task-1", "img-1", {"foreground_count": 0}, "{}")
    await db.save_result("task-1", "img-2", {"foreground_count": 0}, "{}")
    await db.save_result("task-1", "img-3", {"foreground_count": 2}, "{}")

    # CLI delivers strings — get_results must coerce "0" → 0.
    zero = await db.get_results("task-1", filters={"foreground_count": "0"})
    assert len(zero) == 2

    two = await db.get_results("task-1", filters={"foreground_count": "2"})
    assert len(two) == 1


@pytest.mark.asyncio
async def test_get_results_filter_boolean_value(db: Database) -> None:
    """Boolean JSON fields should match ``true``/``false`` filter strings."""
    await db.create_task("task-1", ["a"], {}, 2)
    await db.insert_pending_results("task-1", ["img-1", "img-2"])

    await db.save_result("task-1", "img-1", {"background_ignored": True}, "{}")
    await db.save_result("task-1", "img-2", {"background_ignored": False}, "{}")

    positive = await db.get_results("task-1", filters={"background_ignored": "true"})
    assert len(positive) == 1
    assert positive[0]["asset_id"] == "img-1"


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


@pytest.mark.asyncio
async def test_insert_pending_with_flags(db: Database) -> None:
    """Asset flags (is_archived, is_trashed) are stored on insert."""
    await db.create_task("task-1", ["a"], {}, 3)
    flags = {
        "img-1": (True, False),   # archived
        "img-2": (False, True),   # trashed
        "img-3": (False, False),  # normal
    }
    await db.insert_pending_results("task-1", ["img-1", "img-2", "img-3"], asset_flags=flags)

    summary = await db.get_asset_flag_summary("task-1")
    assert summary["archived"] == 1
    assert summary["trashed"] == 1


@pytest.mark.asyncio
async def test_get_results_filters_trashed(db: Database) -> None:
    """get_results should exclude trashed assets by default."""
    await db.create_task("task-1", ["a"], {}, 3)
    flags = {
        "img-1": (False, False),
        "img-2": (False, True),   # trashed
        "img-3": (True, False),   # archived (still visible)
    }
    await db.insert_pending_results("task-1", ["img-1", "img-2", "img-3"], asset_flags=flags)
    await db.save_result("task-1", "img-1", {"cat": "a"}, "{}")
    await db.save_result("task-1", "img-2", {"cat": "b"}, "{}")
    await db.save_result("task-1", "img-3", {"cat": "c"}, "{}")

    results = await db.get_results("task-1")
    # img-2 is trashed → filtered out
    assert len(results) == 2
    ids = {r["asset_id"] for r in results}
    assert ids == {"img-1", "img-3"}


@pytest.mark.asyncio
async def test_update_asset_flags(db: Database) -> None:
    """update_asset_flags changes the stored flags."""
    await db.create_task("task-1", ["a"], {}, 1)
    await db.insert_pending_results("task-1", ["img-1"])

    await db.update_asset_flags("task-1", "img-1", is_archived=True, is_trashed=False)
    summary = await db.get_asset_flag_summary("task-1")
    assert summary["archived"] == 1
    assert summary["trashed"] == 0

    await db.update_asset_flags("task-1", "img-1", is_archived=False, is_trashed=True)
    summary = await db.get_asset_flag_summary("task-1")
    assert summary["archived"] == 0
    assert summary["trashed"] == 1


@pytest.mark.asyncio
async def test_get_asset_ids_for_task(db: Database) -> None:
    await db.create_task("task-1", ["a"], {}, 2)
    await db.insert_pending_results("task-1", ["img-1", "img-2"])

    ids = await db.get_asset_ids_for_task("task-1")
    assert set(ids) == {"img-1", "img-2"}


@pytest.mark.asyncio
async def test_get_all_asset_flags(db: Database) -> None:
    await db.create_task("task-1", ["a"], {}, 2)
    flags = {"img-1": (True, False), "img-2": (False, True)}
    await db.insert_pending_results("task-1", ["img-1", "img-2"], asset_flags=flags)

    result = await db.get_all_asset_flags("task-1")
    assert result["img-1"] == (True, False)
    assert result["img-2"] == (False, True)


@pytest.mark.asyncio
async def test_batch_update_asset_flags(db: Database) -> None:
    await db.create_task("task-1", ["a"], {}, 3)
    await db.insert_pending_results("task-1", ["img-1", "img-2", "img-3"])

    # All start as (False, False); batch-update two of them.
    await db.batch_update_asset_flags("task-1", [
        ("img-1", True, False),
        ("img-2", False, True),
    ])

    result = await db.get_all_asset_flags("task-1")
    assert result["img-1"] == (True, False)
    assert result["img-2"] == (False, True)
    assert result["img-3"] == (False, False)
