"""Immich API client for albums, assets, and image downloads."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, cast

import httpx
from loguru import logger


@dataclass(frozen=True)
class Album:
    """Represents an Immich album."""

    album_id: str
    album_name: str
    asset_count: int


@dataclass(frozen=True)
class Asset:
    """Represents an Immich asset (photo/video)."""

    asset_id: str
    original_file_name: str
    asset_type: str  # "IMAGE" or "VIDEO"
    is_archived: bool = False
    is_trashed: bool = False


class ImmichClient:
    """Async client for Immich API interactions."""

    def __init__(self, api_url: str, api_key: str, timeout: int = 60) -> None:
        """Initialize the Immich client.

        Args:
            api_url: Base URL of the Immich server (e.g. http://localhost:2283).
            api_key: Immich API key for authentication.
            timeout: Request timeout in seconds.
        """
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self._api_url,
            headers={"x-api-key": self._api_key},
            timeout=httpx.Timeout(timeout),
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def list_albums(self) -> list[Album]:
        """Fetch all albums from Immich.

        Returns:
            List of Album objects.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
            httpx.ConnectError: If the server is unreachable.
        """
        response = await self._client.get("/api/albums")
        response.raise_for_status()
        data: list[dict[str, Any]] = response.json()
        albums = [
            Album(
                album_id=album["id"],
                album_name=album["albumName"],
                asset_count=album.get("assetCount", 0),
            )
            for album in data
        ]
        logger.debug("Fetched {} albums from Immich", len(albums))
        return albums

    async def get_album_assets(self, album_id: str) -> list[Asset]:
        """Fetch all assets in a specific album.

        Args:
            album_id: The ID of the album to fetch.

        Returns:
            List of Asset objects (only IMAGE type assets).

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        response = await self._client.get(f"/api/albums/{album_id}")
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        raw_assets: Any = data.get("assets", [])
        assert isinstance(raw_assets, list)
        asset_list = cast(list[dict[str, Any]], raw_assets)
        assets: list[Asset] = []
        for item in asset_list:
            asset_type = str(item.get("type", "IMAGE"))
            if asset_type == "IMAGE":
                assets.append(Asset(
                    asset_id=str(item["id"]),
                    original_file_name=str(item.get("originalFileName", "")),
                    asset_type=asset_type,
                    is_archived=bool(item.get("isArchived", False)),
                    is_trashed=bool(item.get("isTrashed", False)),
                ))
        logger.debug(
            "Album {} has {} image assets (out of {} total)",
            album_id,
            len(assets),
            len(asset_list),
        )
        return assets

    async def download_image_bytes(
        self, asset_id: str, image_size: str = "thumbnail"
    ) -> tuple[bytes, str]:
        """Download an image and return the raw bytes with content type.

        Args:
            asset_id: The Immich asset ID.
            image_size: Either "thumbnail" or "original".

        Returns:
            Tuple of (raw_bytes, content_type).

        Raises:
            httpx.HTTPStatusError: If the download fails.
        """
        if image_size == "thumbnail":
            url = f"/api/assets/{asset_id}/thumbnail"
        else:
            url = f"/api/assets/{asset_id}/original"

        response = await self._client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/jpeg")
        logger.debug(
            "Downloaded {} bytes for asset {} ({}, {})",
            len(response.content),
            asset_id,
            image_size,
            content_type,
        )
        return response.content, content_type

    async def download_image_base64(
        self, asset_id: str, image_size: str = "thumbnail"
    ) -> tuple[str, str]:
        """Download an image and return it as a base64-encoded string with content type.

        Args:
            asset_id: The Immich asset ID.
            image_size: Either "thumbnail" or "original".

        Returns:
            Tuple of (base64_encoded_data, content_type).

        Raises:
            httpx.HTTPStatusError: If the download fails.
        """
        raw_bytes, content_type = await self.download_image_bytes(asset_id, image_size)
        encoded = base64.b64encode(raw_bytes).decode("ascii")
        return encoded, content_type

    async def get_asset_info(self, asset_id: str) -> dict[str, Any]:
        """Fetch current asset info from Immich.

        Args:
            asset_id: The Immich asset ID.

        Returns:
            Dict with is_archived and is_trashed status.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        url = f"/api/assets/{asset_id}"
        logger.debug("[get_asset_info] GET {} for asset_id={}", url, asset_id)
        response = await self._client.get(url)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        is_archived = bool(data.get("isArchived", False))
        is_trashed = bool(data.get("isTrashed", False))
        logger.info(
            "[get_asset_info] asset_id={} isArchived={} isTrashed={} (raw: isArchived={!r}, isTrashed={!r})",
            asset_id,
            is_archived,
            is_trashed,
            data.get("isArchived"),
            data.get("isTrashed"),
        )
        return {
            "asset_id": asset_id,
            "is_archived": is_archived,
            "is_trashed": is_trashed,
        }
