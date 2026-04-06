"""Tests for Immich API client."""

from __future__ import annotations

import base64

import httpx
import pytest

from immich_classify.immich_client import ImmichClient


def _make_transport(handler: httpx.MockTransport | None = None) -> httpx.MockTransport:
    """Helper to avoid repeating mock transport creation."""
    if handler is not None:
        return handler
    raise ValueError("handler required")


class TestImmichListAlbums:
    """Tests for ImmichClient.list_albums."""

    @pytest.mark.asyncio
    async def test_list_albums_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/albums"
            assert request.headers["x-api-key"] == "test-key"
            return httpx.Response(
                200,
                json=[
                    {"id": "album-1", "albumName": "Vacation", "assetCount": 50},
                    {"id": "album-2", "albumName": "Family", "assetCount": 120},
                ],
            )

        client = ImmichClient.__new__(ImmichClient)
        client._api_url = "http://localhost:2283"
        client._api_key = "test-key"
        client._client = httpx.AsyncClient(
            base_url="http://localhost:2283",
            headers={"x-api-key": "test-key"},
            transport=httpx.MockTransport(handler),
        )

        albums = await client.list_albums()
        assert len(albums) == 2
        assert albums[0].album_id == "album-1"
        assert albums[0].album_name == "Vacation"
        assert albums[0].asset_count == 50
        assert albums[1].album_id == "album-2"
        await client.close()

    @pytest.mark.asyncio
    async def test_list_albums_api_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "Unauthorized"})

        client = ImmichClient.__new__(ImmichClient)
        client._api_url = "http://localhost:2283"
        client._api_key = "bad-key"
        client._client = httpx.AsyncClient(
            base_url="http://localhost:2283",
            headers={"x-api-key": "bad-key"},
            transport=httpx.MockTransport(handler),
        )

        with pytest.raises(httpx.HTTPStatusError):
            await client.list_albums()
        await client.close()


class TestImmichGetAlbumAssets:
    """Tests for ImmichClient.get_album_assets."""

    @pytest.mark.asyncio
    async def test_get_album_assets_filters_images(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "assets": [
                        {"id": "img-1", "originalFileName": "photo.jpg", "type": "IMAGE"},
                        {"id": "vid-1", "originalFileName": "video.mp4", "type": "VIDEO"},
                        {"id": "img-2", "originalFileName": "pic.png", "type": "IMAGE"},
                    ]
                },
            )

        client = ImmichClient.__new__(ImmichClient)
        client._api_url = "http://localhost:2283"
        client._api_key = "test-key"
        client._client = httpx.AsyncClient(
            base_url="http://localhost:2283",
            headers={"x-api-key": "test-key"},
            transport=httpx.MockTransport(handler),
        )

        assets = await client.get_album_assets("album-1")
        assert len(assets) == 2
        assert assets[0].asset_id == "img-1"
        assert assets[1].asset_id == "img-2"
        await client.close()


class TestImmichDownloadImage:
    """Tests for ImmichClient.download_image_base64."""

    @pytest.mark.asyncio
    async def test_download_thumbnail(self) -> None:
        image_data = b"fake-image-bytes"

        def handler(request: httpx.Request) -> httpx.Response:
            assert "/thumbnail" in str(request.url)
            return httpx.Response(
                200, content=image_data,
                headers={"content-type": "image/webp"},
            )

        client = ImmichClient.__new__(ImmichClient)
        client._api_url = "http://localhost:2283"
        client._api_key = "test-key"
        client._client = httpx.AsyncClient(
            base_url="http://localhost:2283",
            headers={"x-api-key": "test-key"},
            transport=httpx.MockTransport(handler),
        )

        encoded, content_type = await client.download_image_base64("img-1", "thumbnail")
        expected = base64.b64encode(image_data).decode("ascii")
        assert encoded == expected
        assert content_type == "image/webp"
        await client.close()

    @pytest.mark.asyncio
    async def test_download_original(self) -> None:
        image_data = b"original-bytes"

        def handler(request: httpx.Request) -> httpx.Response:
            assert "/original" in str(request.url)
            return httpx.Response(200, content=image_data)

        client = ImmichClient.__new__(ImmichClient)
        client._api_url = "http://localhost:2283"
        client._api_key = "test-key"
        client._client = httpx.AsyncClient(
            base_url="http://localhost:2283",
            headers={"x-api-key": "test-key"},
            transport=httpx.MockTransport(handler),
        )

        encoded, content_type = await client.download_image_base64("img-1", "original")
        expected = base64.b64encode(image_data).decode("ascii")
        assert encoded == expected
        assert content_type == "image/jpeg"  # default when no header
        await client.close()
