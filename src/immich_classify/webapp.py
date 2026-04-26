"""FastAPI web application for interactive filter & browse of classification results.

The ``immich-classify app`` CLI subcommand launches this WebUI. It reuses the
same filter-and-query pipeline as the ``results`` subcommand so that
``results --filter key=value`` and the in-browser filter form always return
the same rows for the same task.

Architecture:

* A single :class:`Database` and :class:`ImmichClient` are created in the
  FastAPI ``lifespan`` context and shared across requests.
* Thumbnails can't be fetched directly from Immich by the browser (the
  ``x-api-key`` header can't be attached to an ``<img>`` element), so
  ``/thumbnail/{asset_id}`` proxies the authenticated fetch through the
  backend and sets a ``Cache-Control`` header to keep grid scrolling snappy.
* The filter form is generated dynamically by reading the task's stored
  ``prompt_config`` and reconstructing the :class:`BasePrompt` via
  :meth:`BasePrompt.from_dict` — enum values, bool flags, and numeric fields
  all get appropriate HTML inputs.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, cast

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from immich_classify.config import Config
from immich_classify.database import (
    Database,
    FilterParseError,
    parse_filter_strings,
)
from immich_classify.immich_client import ImmichClient
from immich_classify.prompt_base import BasePrompt


_PACKAGE_ROOT = Path(__file__).resolve().parent
_TEMPLATES_DIR = _PACKAGE_ROOT / "templates"
_STATIC_DIR = _PACKAGE_ROOT / "static"


def create_app(config: Config) -> FastAPI:
    """Build a FastAPI app backed by the shared :class:`Database` and :class:`ImmichClient`.

    Args:
        config: Application configuration (loaded via ``load_config()``).

    Returns:
        A configured :class:`FastAPI` instance ready to be served.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Open the database and Immich client once for the server's lifetime."""
        database = Database(config.database_path)
        await database.connect()
        immich = ImmichClient(config.immich_api_url, config.immich_api_key, config.timeout)
        app.state.db = database
        app.state.immich = immich
        app.state.config = config
        logger.info(
            "WebUI ready: db={} immich={}",
            config.database_path,
            config.immich_api_url,
        )
        try:
            yield
        finally:
            await immich.close()
            await database.close()

    app = FastAPI(
        title="immich-classify WebUI",
        description="Interactive filter & browse for classification results.",
        version="0.1.0",
        lifespan=lifespan,
    )

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ── Routes ─────────────────────────────────────────────────────
    # Pyright's ``reportUnusedFunction`` can't tell that each handler below is
    # registered via ``@app.get(...)``, so we silence it per-handler. The
    # decorator does the real "using" of each function.

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:  # pyright: ignore[reportUnusedFunction]
        """Render the single-page UI."""
        return cast(
            HTMLResponse,
            templates.TemplateResponse(
                request,
                "index.html",
                {"immich_api_url": config.immich_api_url},
            ),
        )

    @app.get("/api/tasks")
    async def api_tasks(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Return all tasks (newest first), with safe summary fields."""
        db = cast(Database, request.app.state.db)
        rows = await db.get_all_tasks()
        tasks: list[dict[str, Any]] = []
        for r in rows:
            prompt_name = ""
            raw_cfg = r.get("prompt_config")
            if raw_cfg:
                try:
                    parsed: dict[str, Any] = json.loads(raw_cfg)
                    prompt_name = str(parsed.get("name", ""))
                except (ValueError, TypeError):
                    prompt_name = ""
            tasks.append({
                "task_id": r["task_id"],
                "status": r["status"],
                "total_count": r["total_count"],
                "completed_count": r["completed_count"],
                "failed_count": r["failed_count"],
                "created_at": r["created_at"],
                "prompt_name": prompt_name,
            })
        return JSONResponse({"tasks": tasks})

    @app.get("/api/tasks/{task_id}/schema")
    async def api_task_schema(task_id: str, request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Return the classification schema used by this task.

        The frontend uses this to auto-generate the filter form: enum fields
        become ``<select>``, bool fields become checkboxes, numeric fields
        become ``<input type="number">``, strings become text inputs.
        """
        db = cast(Database, request.app.state.db)
        task = await db.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        try:
            prompt_cfg: dict[str, Any] = json.loads(task["prompt_config"] or "{}")
            prompt = BasePrompt.from_dict(prompt_cfg)
        except (ValueError, TypeError, AssertionError, KeyError) as exc:
            logger.warning("Failed to parse prompt_config for task {}: {}", task_id, exc)
            prompt = BasePrompt()

        fields: list[dict[str, Any]] = []
        for field_name, sf in prompt.schema.items():
            fields.append({
                "name": field_name,
                "field_type": sf.field_type,
                "description": sf.description,
                "enum": sf.enum,
                "default": sf.default,
            })

        return JSONResponse({
            "task_id": task_id,
            "prompt_name": prompt.name,
            "fields": fields,
        })

    @app.get("/api/tasks/{task_id}/results")
    async def api_task_results(  # pyright: ignore[reportUnusedFunction]
        task_id: str,
        request: Request,
        filter: list[str] | None = Query(default=None),  # noqa: A002 — matches CLI --filter
    ) -> JSONResponse:
        """Return classification results for a task, optionally filtered.

        Accepts repeated ``?filter=key=value`` query parameters; parsing and
        type-coercion are identical to the ``results`` CLI subcommand.
        """
        db = cast(Database, request.app.state.db)
        task = await db.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        try:
            filters = parse_filter_strings(filter)
        except FilterParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        rows = await db.get_results(task_id, filters)

        results: list[dict[str, Any]] = []
        for r in rows:
            entry: dict[str, Any] = {"asset_id": r["asset_id"]}
            raw_json = r.get("result_json")
            if raw_json:
                try:
                    parsed_fields: dict[str, Any] = json.loads(raw_json)
                    entry["fields"] = parsed_fields
                except (ValueError, TypeError):
                    entry["fields"] = {}
            else:
                entry["fields"] = {}
            results.append(entry)

        summary = await db.get_result_summary(task_id)

        return JSONResponse({
            "task_id": task_id,
            "filters": filters or {},
            "total": len(results),
            "success_total": summary.get("success", 0),
            "results": results,
        })

    @app.get("/thumbnail/{asset_id}")
    async def thumbnail(asset_id: str, request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        """Proxy the authenticated Immich thumbnail fetch for a given asset."""
        immich = cast(ImmichClient, request.app.state.immich)
        try:
            data, content_type = await immich.download_image_bytes(asset_id, "thumbnail")
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.warning("Thumbnail fetch for {} failed: HTTP {}", asset_id, status)
            raise HTTPException(status_code=status, detail="Thumbnail fetch failed") from exc
        except httpx.HTTPError as exc:
            logger.warning("Thumbnail fetch for {} errored: {}", asset_id, exc)
            raise HTTPException(status_code=502, detail="Immich unreachable") from exc

        return Response(
            content=data,
            media_type=content_type,
            headers={"Cache-Control": "max-age=3600"},
        )

    @app.get("/api/immich-url/{asset_id}")
    async def api_immich_url(asset_id: str, request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Return the Immich web UI URL for this asset, so the frontend can link out."""
        cfg = cast(Config, request.app.state.config)
        # Immich's web client uses `/photos/{asset_id}` for single-asset view.
        return JSONResponse({
            "asset_id": asset_id,
            "url": f"{cfg.immich_api_url}/photos/{asset_id}",
        })

    return app
