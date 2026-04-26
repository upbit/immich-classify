"""SQLite database operations using aiosqlite."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import aiosqlite
from loguru import logger

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id         TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'pending',
    album_ids       TEXT NOT NULL,
    prompt_config   TEXT NOT NULL,
    total_count     INTEGER NOT NULL DEFAULT 0,
    completed_count INTEGER NOT NULL DEFAULT 0,
    failed_count    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classification_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT NOT NULL REFERENCES tasks(task_id),
    asset_id        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    result_json     TEXT,
    error_message   TEXT,
    raw_response    TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE(task_id, asset_id)
);

CREATE INDEX IF NOT EXISTS idx_results_task ON classification_results(task_id);
CREATE INDEX IF NOT EXISTS idx_results_status ON classification_results(status);
"""

# Incremental migrations for columns added after the initial schema.
_MIGRATIONS = [
    "ALTER TABLE classification_results ADD COLUMN is_archived INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE classification_results ADD COLUMN is_trashed INTEGER NOT NULL DEFAULT 0",
]


def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _coerce_filter_value(raw: str) -> Any:
    """Coerce a CLI string filter value to its best-match Python/JSON type.

    ``json_extract`` returns the native JSON type (int/float/bool/null/text),
    so comparing against a bare string can silently miss numeric or boolean
    fields. Try JSON-parsing first; fall back to the raw string.
    """
    stripped = raw.strip()
    if stripped == "":
        return raw
    # ``true``/``false``/``null`` are lowercase in JSON; accept common casings.
    lowered = stripped.lower()
    if lowered in ("true", "false"):
        return 1 if lowered == "true" else 0
    if lowered == "null":
        return None
    try:
        return json.loads(stripped)
    except (ValueError, TypeError):
        return raw


class FilterParseError(ValueError):
    """Raised when a ``key=value`` filter string is malformed."""


def parse_filter_strings(filter_args: list[str] | None) -> dict[str, str] | None:
    """Parse a list of ``key=value`` strings into a dict.

    Shared between the CLI (``results`` subcommand) and the WebUI so both use
    identical parsing rules. Splits on the first ``=`` only, so values may
    themselves contain ``=`` characters.

    Args:
        filter_args: List of ``key=value`` strings, or ``None``.

    Returns:
        Dict of ``{key: value}`` pairs, or ``None`` if ``filter_args`` is empty.

    Raises:
        FilterParseError: If any entry lacks a ``=`` separator.
    """
    if not filter_args:
        return None
    filters: dict[str, str] = {}
    for f in filter_args:
        if "=" not in f:
            raise FilterParseError(
                f"Invalid filter format: {f!r}. Use key=value."
            )
        key, value = f.split("=", 1)
        filters[key.strip()] = value.strip()
    return filters


class Database:
    """Async SQLite database manager for classification tasks and results."""

    def __init__(self, database_path: str) -> None:
        self._database_path = database_path
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the database connection and initialize schema."""
        self._connection = await aiosqlite.connect(self._database_path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.executescript(SCHEMA_SQL)
        await self._connection.commit()
        await self._run_migrations()
        logger.debug("Database initialized at {}", self._database_path)

    async def _run_migrations(self) -> None:
        """Apply incremental schema migrations (idempotent)."""
        for sql in _MIGRATIONS:
            try:
                await self.connection.execute(sql)
                await self.connection.commit()
            except Exception:  # noqa: BLE001 — column already exists
                pass

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    @property
    def connection(self) -> aiosqlite.Connection:
        """Get the active database connection."""
        assert self._connection is not None, "Database not connected. Call connect() first."
        return self._connection

    # ── Task operations ──────────────────────────────────────────────

    async def create_task(
        self,
        task_id: str,
        album_ids: list[str],
        prompt_config: dict[str, Any],
        total_count: int,
    ) -> None:
        """Create a new classification task.

        Args:
            task_id: Unique task identifier (UUID).
            album_ids: List of Immich album IDs to process.
            prompt_config: Serialized BasePrompt configuration.
            total_count: Total number of assets to process.
        """
        now = _now_iso()
        await self.connection.execute(
            """
            INSERT INTO tasks (task_id, status, album_ids, prompt_config, total_count, created_at, updated_at)
            VALUES (?, 'pending', ?, ?, ?, ?, ?)
            """,
            (task_id, json.dumps(album_ids), json.dumps(prompt_config), total_count, now, now),
        )
        await self.connection.commit()
        logger.info("Task {} created with {} assets", task_id, total_count)

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get task details by ID.

        Args:
            task_id: The task ID to look up.

        Returns:
            Task row as a dictionary, or None if not found.
        """
        cursor = await self.connection.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def get_all_tasks(self) -> list[dict[str, Any]]:
        """Get all tasks ordered by creation time (newest first).

        Returns:
            List of task rows as dictionaries.
        """
        cursor = await self.connection.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def update_task_status(self, task_id: str, status: str) -> None:
        """Update the status of a task.

        Args:
            task_id: The task to update.
            status: New status (pending/running/paused/completed/cancelled/failed).
        """
        await self.connection.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE task_id = ?",
            (status, _now_iso(), task_id),
        )
        await self.connection.commit()

    async def increment_task_completed(self, task_id: str) -> None:
        """Increment the completed count for a task.

        Args:
            task_id: The task to update.
        """
        await self.connection.execute(
            "UPDATE tasks SET completed_count = completed_count + 1, updated_at = ? WHERE task_id = ?",
            (_now_iso(), task_id),
        )
        await self.connection.commit()

    async def increment_task_failed(self, task_id: str) -> None:
        """Increment the failed count for a task.

        Args:
            task_id: The task to update.
        """
        await self.connection.execute(
            "UPDATE tasks SET failed_count = failed_count + 1, updated_at = ? WHERE task_id = ?",
            (_now_iso(), task_id),
        )
        await self.connection.commit()

    # ── Result operations ────────────────────────────────────────────

    async def insert_pending_results(
        self,
        task_id: str,
        asset_ids: list[str],
        asset_flags: dict[str, tuple[bool, bool]] | None = None,
    ) -> None:
        """Batch-insert pending result placeholders for a task.

        Args:
            task_id: The task these assets belong to.
            asset_ids: List of Immich asset IDs to insert.
            asset_flags: Optional mapping of asset_id → (is_archived, is_trashed).
        """
        now = _now_iso()
        flags = asset_flags or {}
        await self.connection.executemany(
            """
            INSERT OR IGNORE INTO classification_results
                (task_id, asset_id, status, is_archived, is_trashed, created_at)
            VALUES (?, ?, 'pending', ?, ?, ?)
            """,
            [
                (
                    task_id,
                    asset_id,
                    int(flags.get(asset_id, (False, False))[0]),
                    int(flags.get(asset_id, (False, False))[1]),
                    now,
                )
                for asset_id in asset_ids
            ],
        )
        await self.connection.commit()

    async def save_result(
        self,
        task_id: str,
        asset_id: str,
        result_json: dict[str, Any],
        raw_response: str,
    ) -> None:
        """Save a successful classification result.

        Args:
            task_id: The owning task ID.
            asset_id: The Immich asset ID.
            result_json: Parsed classification result.
            raw_response: Raw VLM response text.
        """
        await self.connection.execute(
            """
            UPDATE classification_results
            SET status = 'success', result_json = ?, raw_response = ?
            WHERE task_id = ? AND asset_id = ?
            """,
            (json.dumps(result_json), raw_response, task_id, asset_id),
        )
        await self.connection.commit()

    async def save_error(
        self,
        task_id: str,
        asset_id: str,
        error_message: str,
        raw_response: str | None = None,
    ) -> None:
        """Save a failed classification result.

        Args:
            task_id: The owning task ID.
            asset_id: The Immich asset ID.
            error_message: Description of the error.
            raw_response: Optional raw VLM response for debugging.
        """
        await self.connection.execute(
            """
            UPDATE classification_results
            SET status = 'failed', error_message = ?, raw_response = ?
            WHERE task_id = ? AND asset_id = ?
            """,
            (error_message, raw_response, task_id, asset_id),
        )
        await self.connection.commit()

    async def get_pending_asset_ids(self, task_id: str) -> list[str]:
        """Get asset IDs that haven't been successfully processed yet.

        Args:
            task_id: The task to query.

        Returns:
            List of asset IDs with status != 'success'.
        """
        cursor = await self.connection.execute(
            """
            SELECT asset_id FROM classification_results
            WHERE task_id = ? AND status != 'success'
            ORDER BY id
            """,
            (task_id,),
        )
        rows = await cursor.fetchall()
        return [row["asset_id"] for row in rows]

    async def get_results(
        self,
        task_id: str,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Query classification results, optionally filtered by JSON fields.

        Args:
            task_id: The task to query.
            filters: Optional dict of {json_field: value} to filter by.
                Values are parsed as JSON when possible so that numeric and
                boolean fields compare correctly against SQLite's
                ``json_extract`` output (which preserves JSON types).

        Returns:
            List of result rows as dictionaries.
        """
        # Baseline: how many successful rows exist for this task, before filtering.
        base_cursor = await self.connection.execute(
            "SELECT COUNT(*) AS n FROM classification_results "
            "WHERE task_id = ? AND status = 'success'",
            (task_id,),
        )
        base_row = await base_cursor.fetchone()
        base_count = base_row["n"] if base_row is not None else 0
        logger.debug(
            "get_results: task_id={} success_rows={} filters={}",
            task_id,
            base_count,
            filters,
        )

        query = "SELECT * FROM classification_results WHERE task_id = ? AND status = 'success' AND is_trashed = 0"
        params: list[Any] = [task_id]

        if filters:
            for key, raw_value in filters.items():
                coerced_value = _coerce_filter_value(raw_value)
                query += f" AND json_extract(result_json, '$.{key}') = ?"
                params.append(coerced_value)
                logger.debug(
                    "get_results: filter json_extract($.{})={!r} (raw={!r}, type={})",
                    key,
                    coerced_value,
                    raw_value,
                    type(coerced_value).__name__,
                )

        logger.debug("get_results: SQL={} params={}", query, params)
        cursor = await self.connection.execute(query, params)
        rows = list(await cursor.fetchall())
        logger.debug(
            "get_results: task_id={} matched_rows={}/{}",
            task_id,
            len(rows),
            base_count,
        )

        # If the user supplied filters but nothing matched, surface a sample of
        # the values that actually exist for each filter key to make mismatches
        # (types, spelling, casing) obvious.
        if filters and not rows and base_count > 0:
            for key in filters:
                sample_cursor = await self.connection.execute(
                    f"""
                    SELECT DISTINCT json_extract(result_json, '$.{key}') AS v,
                           typeof(json_extract(result_json, '$.{key}')) AS t
                    FROM classification_results
                    WHERE task_id = ? AND status = 'success'
                    LIMIT 10
                    """,
                    (task_id,),
                )
                sample_rows = await sample_cursor.fetchall()
                observed = [(row["v"], row["t"]) for row in sample_rows]
                logger.warning(
                    "get_results: no matches for filter '{}={}'. "
                    "Observed values (value, sqlite_type) in column: {}",
                    key,
                    filters[key],
                    observed,
                )

        return [dict(row) for row in rows]

    async def get_result_summary(self, task_id: str) -> dict[str, int]:
        """Get a summary of result statuses for a task.

        Args:
            task_id: The task to query.

        Returns:
            Dict mapping status to count (e.g. {"success": 42, "failed": 3, "pending": 5}).
        """
        cursor = await self.connection.execute(
            """
            SELECT status, COUNT(*) as count
            FROM classification_results
            WHERE task_id = ?
            GROUP BY status
            """,
            (task_id,),
        )
        rows = await cursor.fetchall()
        return {row["status"]: row["count"] for row in rows}

    # ── Immich status sync ───────────────────────────────────────────

    async def get_asset_ids_for_task(self, task_id: str) -> list[str]:
        """Return all asset IDs belonging to a task.

        Args:
            task_id: The task to query.

        Returns:
            List of asset IDs.
        """
        cursor = await self.connection.execute(
            "SELECT asset_id FROM classification_results WHERE task_id = ?",
            (task_id,),
        )
        rows = await cursor.fetchall()
        return [row["asset_id"] for row in rows]

    async def update_asset_flags(
        self,
        task_id: str,
        asset_id: str,
        is_archived: bool,
        is_trashed: bool,
    ) -> None:
        """Update the archived/trashed flags for a single asset.

        Args:
            task_id: The owning task.
            asset_id: The Immich asset ID.
            is_archived: Current archived state.
            is_trashed: Current trashed state.
        """
        await self.connection.execute(
            """
            UPDATE classification_results
            SET is_archived = ?, is_trashed = ?
            WHERE task_id = ? AND asset_id = ?
            """,
            (int(is_archived), int(is_trashed), task_id, asset_id),
        )
        await self.connection.commit()

    async def get_all_asset_flags(
        self, task_id: str
    ) -> dict[str, tuple[bool, bool]]:
        """Return current (is_archived, is_trashed) for every asset in a task.

        Args:
            task_id: The task to query.

        Returns:
            Dict mapping asset_id → (is_archived, is_trashed).
        """
        cursor = await self.connection.execute(
            "SELECT asset_id, is_archived, is_trashed FROM classification_results WHERE task_id = ?",
            (task_id,),
        )
        rows = await cursor.fetchall()
        return {
            row["asset_id"]: (bool(row["is_archived"]), bool(row["is_trashed"]))
            for row in rows
        }

    async def batch_update_asset_flags(
        self,
        task_id: str,
        flags: list[tuple[str, bool, bool]],
    ) -> None:
        """Batch-update archived/trashed flags for multiple assets in one commit.

        Args:
            task_id: The owning task.
            flags: List of (asset_id, is_archived, is_trashed) tuples.
        """
        if not flags:
            return
        await self.connection.executemany(
            """
            UPDATE classification_results
            SET is_archived = ?, is_trashed = ?
            WHERE task_id = ? AND asset_id = ?
            """,
            [
                (int(is_archived), int(is_trashed), task_id, asset_id)
                for asset_id, is_archived, is_trashed in flags
            ],
        )
        await self.connection.commit()
        logger.info(
            "[db] batch_update_asset_flags: task={} updated {} rows",
            task_id,
            len(flags),
        )

    async def get_asset_flag_summary(
        self, task_id: str
    ) -> dict[str, int]:
        """Return counts of archived and trashed assets for a task.

        Args:
            task_id: The task to query.

        Returns:
            Dict with keys 'archived' and 'trashed'.
        """
        cursor = await self.connection.execute(
            """
            SELECT
                SUM(is_archived) AS archived,
                SUM(is_trashed)  AS trashed
            FROM classification_results
            WHERE task_id = ?
            """,
            (task_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return {"archived": 0, "trashed": 0}
        return {
            "archived": int(row["archived"] or 0),
            "trashed": int(row["trashed"] or 0),
        }
