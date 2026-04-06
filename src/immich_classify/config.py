"""Configuration loading from .env files and environment variables."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from .env and environment variables."""

    immich_api_url: str
    immich_api_key: str
    vlm_api_url: str
    vlm_api_key: str
    vlm_model_name: str
    database_path: str
    concurrency: int
    timeout: int
    image_size: str

    def __post_init__(self) -> None:
        if self.image_size not in ("thumbnail", "original"):
            raise ValueError(
                f"CLASSIFY_IMAGE_SIZE must be 'thumbnail' or 'original', got '{self.image_size}'"
            )
        if self.concurrency < 1:
            raise ValueError(
                f"CLASSIFY_CONCURRENCY must be >= 1, got {self.concurrency}"
            )
        if self.timeout < 1:
            raise ValueError(
                f"CLASSIFY_TIMEOUT must be >= 1, got {self.timeout}"
            )


def load_config(env_file: str | Path | None = None) -> Config:
    """Load configuration from .env file and environment variables.

    Priority: environment variables override .env file values.

    Args:
        env_file: Path to .env file. If None, searches for .env in current directory.

    Returns:
        Validated Config instance.

    Raises:
        SystemExit: If required configuration is missing.
    """
    # Load .env file first (environment variables take precedence)
    if env_file is not None:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)

    missing_fields: list[str] = []

    immich_api_url = os.environ.get("IMMICH_API_URL", "")
    if not immich_api_url:
        missing_fields.append("IMMICH_API_URL")

    immich_api_key = os.environ.get("IMMICH_API_KEY", "")
    if not immich_api_key:
        missing_fields.append("IMMICH_API_KEY")

    if missing_fields:
        logger.error(
            "Missing required configuration: {}. "
            "Set them in .env file or as environment variables.",
            ", ".join(missing_fields),
        )
        sys.exit(1)

    vlm_api_url = os.environ.get("VLM_API_URL", "http://localhost:8000/v1")
    vlm_api_key = os.environ.get("VLM_API_KEY", "no-key")
    vlm_model_name = os.environ.get("VLM_MODEL_NAME", "")
    database_path = os.environ.get("CLASSIFY_DB_PATH", "./classify.db")

    try:
        concurrency = int(os.environ.get("CLASSIFY_CONCURRENCY", "1"))
    except ValueError:
        logger.error("CLASSIFY_CONCURRENCY must be an integer")
        sys.exit(1)

    try:
        timeout = int(os.environ.get("CLASSIFY_TIMEOUT", "60"))
    except ValueError:
        logger.error("CLASSIFY_TIMEOUT must be an integer")
        sys.exit(1)

    image_size = os.environ.get("CLASSIFY_IMAGE_SIZE", "thumbnail")

    return Config(
        immich_api_url=immich_api_url.rstrip("/"),
        immich_api_key=immich_api_key,
        vlm_api_url=vlm_api_url.rstrip("/"),
        vlm_api_key=vlm_api_key,
        vlm_model_name=vlm_model_name,
        database_path=database_path,
        concurrency=concurrency,
        timeout=timeout,
        image_size=image_size,
    )
