"""Tests for CLI argument parsing and filter parsing."""

from __future__ import annotations

import pytest

from immich_classify.cli import build_parser, _parse_filters


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
