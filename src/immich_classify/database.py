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


def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


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
        logger.debug("Database initialized at {}", self._database_path)

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
            prompt_config: Serialized ClassificationPrompt configuration.
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

    async def insert_pending_results(self, task_id: str, asset_ids: list[str]) -> None:
        """Batch-insert pending result placeholders for a task.

        Args:
            task_id: The task these assets belong to.
            asset_ids: List of Immich asset IDs to insert.
        """
        now = _now_iso()
        await self.connection.executemany(
            """
            INSERT OR IGNORE INTO classification_results (task_id, asset_id, status, created_at)
            VALUES (?, ?, 'pending', ?)
            """,
            [(task_id, asset_id, now) for asset_id in asset_ids],
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

        Returns:
            List of result rows as dictionaries.
        """
        query = "SELECT * FROM classification_results WHERE task_id = ? AND status = 'success'"
        params: list[Any] = [task_id]

        if filters:
            for key, value in filters.items():
                query += f" AND json_extract(result_json, '$.{key}') = ?"
                params.append(value)

        cursor = await self.connection.execute(query, params)
        rows = await cursor.fetchall()
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
