"""Tests for the FastAPI WebUI (immich-classify app)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from immich_classify.config import Config
from immich_classify.database import Database
from immich_classify.immich_client import ImmichClient
from immich_classify.webapp import create_app


TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xcf"
    b"\xc0\xf0\x1f\x00\x05\x00\x01\xff\xa6\xa0r\x0e\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_config(db_path: str) -> Config:
    return Config(
        immich_api_url="http://immich.test",
        immich_api_key="test-key",
        vlm_api_url="http://vlm.test",
        vlm_api_key="no-key",
        vlm_model_name="",
        database_path=db_path,
        concurrency=1,
        timeout=60,
        image_size="thumbnail",
        default_prompt="",
    )


def _immich_mock_transport(asset_to_bytes: dict[str, bytes]) -> httpx.MockTransport:
    """Mock transport that returns the tiny PNG for registered assets, 404 otherwise."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        for asset_id, content in asset_to_bytes.items():
            if path == f"/api/assets/{asset_id}/thumbnail":
                return httpx.Response(
                    200,
                    content=content,
                    headers={"content-type": "image/png"},
                )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def _seed_task(
    db: Database, task_id: str, schema_fields: dict[str, dict[str, Any]]
) -> None:
    """Create a task whose prompt_config carries the given schema fields."""
    prompt_config = {
        "name": "test_prompt",
        "system_prompt": "sys",
        "user_prompt": "user {schema_description}",
        "schema": schema_fields,
    }
    await db.create_task(task_id, ["album-1"], prompt_config, total_count=3)


async def _seed_results(
    db: Database, task_id: str, asset_to_result: dict[str, dict[str, Any] | None]
) -> None:
    """Insert placeholder rows then save_result for each asset with a dict value."""
    await db.insert_pending_results(task_id, list(asset_to_result.keys()))
    for asset_id, result in asset_to_result.items():
        if result is not None:
            await db.save_result(task_id, asset_id, result, raw_response="{}")


@pytest_asyncio.fixture
async def seeded_app(tmp_path: Path) -> AsyncIterator[tuple[TestClient, str]]:
    """Build a FastAPI TestClient with a seeded SQLite DB and a mocked ImmichClient."""
    db_path = str(tmp_path / "test.db")
    # Seed DB up-front.
    db = Database(db_path)
    await db.connect()
    task_id = "task-abc-123"
    await _seed_task(
        db,
        task_id,
        schema_fields={
            "foreground_count": {"field_type": "int", "description": "how many people"},
            "scene": {
                "field_type": "string",
                "description": "scene",
                "enum": ["indoor", "outdoor"],
            },
            "is_screenshot": {"field_type": "bool", "description": "screenshot flag"},
        },
    )
    await _seed_results(
        db,
        task_id,
        {
            "asset-1": {
                "foreground_count": 1,
                "scene": "indoor",
                "is_screenshot": False,
            },
            "asset-2": {
                "foreground_count": 3,
                "scene": "outdoor",
                "is_screenshot": False,
            },
            "asset-3": {
                "foreground_count": 1,
                "scene": "outdoor",
                "is_screenshot": True,
            },
        },
    )
    await db.close()

    config = _make_config(db_path)
    app = create_app(config)

    # Replace the auto-created ImmichClient (built in lifespan) with one that
    # uses a MockTransport, so /thumbnail/{id} doesn't try to reach a real server.
    transport = _immich_mock_transport({
        "asset-1": TINY_PNG,
        "asset-2": TINY_PNG,
        "asset-3": TINY_PNG,
    })

    with TestClient(app) as client:
        # At this point lifespan has already started — swap the Immich client.
        real_immich = app.state.immich
        await real_immich.close()
        mocked = ImmichClient.__new__(ImmichClient)
        mocked._api_url = config.immich_api_url
        mocked._api_key = config.immich_api_key
        mocked._client = httpx.AsyncClient(
            base_url=config.immich_api_url,
            headers={"x-api-key": config.immich_api_key},
            transport=transport,
        )
        app.state.immich = mocked
        yield client, task_id


@pytest.mark.asyncio
async def test_api_tasks_lists_seeded_task(
    seeded_app: tuple[TestClient, str],
) -> None:
    client, task_id = seeded_app
    resp = client.get("/api/tasks")
    assert resp.status_code == 200
    data = resp.json()
    assert any(t["task_id"] == task_id for t in data["tasks"])


@pytest.mark.asyncio
async def test_api_schema_reflects_prompt_config(
    seeded_app: tuple[TestClient, str],
) -> None:
    client, task_id = seeded_app
    resp = client.get(f"/api/tasks/{task_id}/schema")
    assert resp.status_code == 200
    data = resp.json()
    field_names = {f["name"] for f in data["fields"]}
    assert field_names == {"foreground_count", "scene", "is_screenshot"}
    scene = next(f for f in data["fields"] if f["name"] == "scene")
    assert scene["enum"] == ["indoor", "outdoor"]
    count = next(f for f in data["fields"] if f["name"] == "foreground_count")
    assert count["field_type"] == "int"


@pytest.mark.asyncio
async def test_api_results_filter_matches_cli_semantics(
    seeded_app: tuple[TestClient, str],
) -> None:
    client, task_id = seeded_app
    # foreground_count=1 should match asset-1 and asset-3 (ints, not strings).
    resp = client.get(
        f"/api/tasks/{task_id}/results",
        params=[("filter", "foreground_count=1")],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert {r["asset_id"] for r in data["results"]} == {"asset-1", "asset-3"}
    # is_screenshot=true should only match asset-3.
    resp = client.get(
        f"/api/tasks/{task_id}/results",
        params=[("filter", "is_screenshot=true")],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["results"][0]["asset_id"] == "asset-3"
    assert data["results"][0]["fields"]["is_screenshot"] is True


@pytest.mark.asyncio
async def test_api_results_no_filters_returns_all(
    seeded_app: tuple[TestClient, str],
) -> None:
    client, task_id = seeded_app
    resp = client.get(f"/api/tasks/{task_id}/results")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert data["success_total"] == 3


@pytest.mark.asyncio
async def test_thumbnail_proxies_bytes(
    seeded_app: tuple[TestClient, str],
) -> None:
    client, _ = seeded_app
    resp = client.get("/thumbnail/asset-1")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
    assert resp.content == TINY_PNG
    assert "max-age" in resp.headers.get("cache-control", "")


@pytest.mark.asyncio
async def test_unknown_task_and_bad_filter(
    seeded_app: tuple[TestClient, str],
) -> None:
    client, _ = seeded_app
    # Unknown task → 404 on schema
    resp = client.get("/api/tasks/nope/schema")
    assert resp.status_code == 404
    # Unknown task → 404 on results
    resp = client.get("/api/tasks/nope/results")
    assert resp.status_code == 404
    # Bad filter format (no '=') → 400
    # Need a real task for this; use the seeded one.
    resp_tasks = client.get("/api/tasks").json()
    real_task = resp_tasks["tasks"][0]["task_id"]
    resp = client.get(
        f"/api/tasks/{real_task}/results",
        params=[("filter", "no-equals-sign")],
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_immich_url_endpoint(
    seeded_app: tuple[TestClient, str],
) -> None:
    client, _ = seeded_app
    resp = client.get("/api/immich-url/asset-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["asset_id"] == "asset-1"
    assert data["url"].endswith("/photos/asset-1")


def test_index_page_renders(tmp_path: Path) -> None:
    """Smoke test: the Jinja template renders without error."""
    db_path = str(tmp_path / "empty.db")
    # Make sure the DB file can be opened (lifespan will create tables).
    config = _make_config(db_path)
    app = create_app(config)
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "immich-classify" in resp.text
        # Ensure the Immich API URL landed in the template.
        assert json.dumps(config.immich_api_url) in resp.text
