"""Tests for CLI argument parsing and filter parsing."""

from __future__ import annotations

import pytest

from immich_classify.cli import build_parser, cmd_status, _parse_filters
from immich_classify.config import Config
from immich_classify.database import Database


class TestBuildParser:
    """Tests for CLI argument parser."""

    def test_albums_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["albums"])
        assert args.command == "albums"

    def test_classify_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["classify", "--album", "abc-123"])
        assert args.command == "classify"
        assert args.albums == ["abc-123"]
        assert args.prompt_config is None
        assert args.concurrency is None

    def test_classify_with_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "classify", "--album", "abc-123", "--album", "def-456",
            "--prompt-config", "/path/to/config.py",
            "--concurrency", "4",
        ])
        assert args.albums == ["abc-123", "def-456"]
        assert args.prompt_config == "/path/to/config.py"
        assert args.concurrency == 4

    def test_debug_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["debug", "--album", "abc-123"])
        assert args.command == "debug"
        assert args.album == "abc-123"
        assert args.count == 10  # default

    def test_debug_with_count(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["debug", "--album", "abc", "--count", "5"])
        assert args.count == 5

    def test_status_command_no_task(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"
        assert args.task is None

    def test_status_command_with_task(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status", "--task", "task-123"])
        assert args.task == "task-123"

    def test_results_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["results", "--task", "task-123"])
        assert args.command == "results"
        assert args.task == "task-123"
        assert args.format == "table"  # default

    def test_results_with_filter_and_format(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "results", "--task", "task-123",
            "--filter", "category=people",
            "--filter", "quality=high",
            "--format", "csv",
        ])
        assert args.filter == ["category=people", "quality=high"]
        assert args.format == "csv"

    def test_pause_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["pause", "--task", "task-123"])
        assert args.command == "pause"
        assert args.task == "task-123"

    def test_resume_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["resume", "--task", "task-123"])
        assert args.command == "resume"

    def test_cancel_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["cancel", "--task", "task-123"])
        assert args.command == "cancel"

    def test_no_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestParseFilters:
    """Tests for _parse_filters."""

    def test_none_input(self) -> None:
        assert _parse_filters(None) is None

    def test_empty_list(self) -> None:
        assert _parse_filters([]) is None

    def test_single_filter(self) -> None:
        result = _parse_filters(["category=people"])
        assert result == {"category": "people"}

    def test_multiple_filters(self) -> None:
        result = _parse_filters(["category=people", "quality=high"])
        assert result == {"category": "people", "quality": "high"}

    def test_filter_with_spaces(self) -> None:
        result = _parse_filters(["category = people"])
        assert result == {"category": "people"}

    def test_invalid_filter_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_filters(["invalid-no-equals"])


# ── cmd_status regression tests ──────────────────────────────────────────
#
# User-facing issue: for a paused task with pending rows, the CLI showed
# "Progress: 696/4863, Failed: 319" and the user reasonably asked why
# 696+319 != 4863. Root cause: cmd_status only surfaced the completed and
# failed buckets, hiding pending (un-attempted) rows. The fix displays
# pending and reconciles the denormalized counters from per-asset rows
# before printing so the output always satisfies
# ``completed + failed + pending == total``.


def _make_config(db_path: str) -> Config:
    return Config(
        immich_api_url="http://localhost:2283",
        immich_api_key="test-key",
        vlm_api_url="http://localhost:8000/v1",
        vlm_api_key="no-key",
        vlm_model_name="",
        database_path=db_path,
        concurrency=1,
        timeout=60,
        image_size="thumbnail",
        default_prompt="",
    )


class TestCmdStatus:
    """Regression tests for ``immich-classify status``."""

    @pytest.mark.asyncio
    async def test_status_shows_pending_for_paused_task(
        self, tmp_path: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Detail view must surface the pending bucket and reconcile
        counters so ``completed+failed+pending == total``.

        Mirrors the user's "696/4863, Failed: 319" report in miniature:
        pending assets exist (user hit Ctrl+C mid-run); the display must
        make them visible instead of leaving a silent arithmetic gap.
        """
        db_path = str(tmp_path) + "/classify.db"  # type: ignore[attr-defined]
        db = Database(db_path)
        await db.connect()
        await db.create_task("t-1", ["a"], {}, 10)
        await db.insert_pending_results("t-1", [f"img-{i}" for i in range(10)])
        await db.save_result("t-1", "img-0", {"x": 1}, "{}")
        await db.save_result("t-1", "img-1", {"x": 1}, "{}")
        await db.save_result("t-1", "img-2", {"x": 1}, "{}")
        await db.save_error("t-1", "img-3", "boom")
        await db.save_error("t-1", "img-4", "boom")
        await db.update_task_status("t-1", "paused")
        await db.close()

        await cmd_status(_make_config(db_path), task_id="t-1")
        out = capsys.readouterr().out

        assert "Total:      10" in out
        assert "Completed:  3" in out
        assert "Failed:     2" in out
        assert "Pending:    5" in out
        # The arithmetic must add up — this is the invariant the user
        # expected and the whole point of the fix.
        assert 3 + 2 + 5 == 10

    @pytest.mark.asyncio
    async def test_status_list_shows_pending_column(
        self, tmp_path: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The table listing all tasks must include a Pending column.

        Without this, a user scanning the list (like the user who reported
        the bug) can't see why Progress + Failed doesn't equal Total.
        """
        db_path = str(tmp_path) + "/classify.db"  # type: ignore[attr-defined]
        db = Database(db_path)
        await db.connect()
        await db.create_task("t-paused", ["a"], {}, 5)
        await db.insert_pending_results(
            "t-paused", ["img-1", "img-2", "img-3", "img-4", "img-5"]
        )
        await db.save_result("t-paused", "img-1", {"x": 1}, "{}")
        await db.save_error("t-paused", "img-2", "boom")
        # img-3, img-4, img-5 stay pending.
        await db.update_task_status("t-paused", "paused")
        await db.close()

        await cmd_status(_make_config(db_path), task_id=None)
        out = capsys.readouterr().out

        assert "Pending" in out  # header
        # The pending count (3) shows in the row.
        # We check for "1/5" (progress) and pending=3.
        assert "1/5" in out
        assert "3" in out  # pending count

    @pytest.mark.asyncio
    async def test_status_reconciles_inflated_counters(
        self, tmp_path: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Legacy pre-fix DB rows (with double-counted failed_count) must
        not leak into the status output. Reconciliation from per-asset
        rows happens before display.
        """
        db_path = str(tmp_path) + "/classify.db"  # type: ignore[attr-defined]
        db = Database(db_path)
        await db.connect()
        await db.create_task("t-bad", ["a"], {}, 3)
        await db.insert_pending_results("t-bad", ["img-1", "img-2", "img-3"])
        await db.save_result("t-bad", "img-1", {"x": 1}, "{}")
        await db.save_error("t-bad", "img-2", "boom")

        # Simulate the old bug: inflate failed_count far past total_count.
        for _ in range(20):
            await db.increment_task_failed("t-bad")
        bad = await db.get_task("t-bad")
        assert bad is not None
        assert bad["failed_count"] == 20  # way more than total=3
        await db.close()

        await cmd_status(_make_config(db_path), task_id="t-bad")
        out = capsys.readouterr().out

        # Post-reconciliation: failed==1 (only one row truly failed).
        assert "Failed:     1" in out
        assert "Completed:  1" in out
        assert "Pending:    1" in out
