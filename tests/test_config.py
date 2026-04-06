"""Tests for configuration loading."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from immich_classify.config import Config, load_config


class TestConfig:
    """Tests for Config dataclass validation."""

    def test_valid_config(self) -> None:
        config = Config(
            immich_api_url="http://localhost:2283",
            immich_api_key="test-key",
            vlm_api_url="http://localhost:8000/v1",
            vlm_api_key="no-key",
            vlm_model_name="",
            database_path="./classify.db",
            concurrency=1,
            timeout=60,
            image_size="thumbnail",
        )
        assert config.immich_api_url == "http://localhost:2283"

    def test_invalid_image_size(self) -> None:
        with pytest.raises(ValueError, match="CLASSIFY_IMAGE_SIZE"):
            Config(
                immich_api_url="http://localhost:2283",
                immich_api_key="test-key",
                vlm_api_url="http://localhost:8000/v1",
                vlm_api_key="no-key",
                vlm_model_name="",
                database_path="./classify.db",
                concurrency=1,
                timeout=60,
                image_size="invalid",
            )

    def test_invalid_concurrency(self) -> None:
        with pytest.raises(ValueError, match="CLASSIFY_CONCURRENCY"):
            Config(
                immich_api_url="http://localhost:2283",
                immich_api_key="test-key",
                vlm_api_url="http://localhost:8000/v1",
                vlm_api_key="no-key",
                vlm_model_name="",
                database_path="./classify.db",
                concurrency=0,
                timeout=60,
                image_size="thumbnail",
            )

    def test_invalid_timeout(self) -> None:
        with pytest.raises(ValueError, match="CLASSIFY_TIMEOUT"):
            Config(
                immich_api_url="http://localhost:2283",
                immich_api_key="test-key",
                vlm_api_url="http://localhost:8000/v1",
                vlm_api_key="no-key",
                vlm_model_name="",
                database_path="./classify.db",
                concurrency=1,
                timeout=0,
                image_size="thumbnail",
            )


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_env_vars(self) -> None:
        env = {
            "IMMICH_API_URL": "http://immich:2283",
            "IMMICH_API_KEY": "my-key",
            "VLM_API_URL": "http://vlm:8000/v1",
            "VLM_API_KEY": "vlm-key",
            "VLM_MODEL_NAME": "test-model",
            "CLASSIFY_DB_PATH": "/tmp/test.db",
            "CLASSIFY_CONCURRENCY": "4",
            "CLASSIFY_TIMEOUT": "30",
            "CLASSIFY_IMAGE_SIZE": "original",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config(env_file="/dev/null")

        assert config.immich_api_url == "http://immich:2283"
        assert config.immich_api_key == "my-key"
        assert config.vlm_api_url == "http://vlm:8000/v1"
        assert config.vlm_api_key == "vlm-key"
        assert config.vlm_model_name == "test-model"
        assert config.database_path == "/tmp/test.db"
        assert config.concurrency == 4
        assert config.timeout == 30
        assert config.image_size == "original"

    def test_default_values(self) -> None:
        env = {
            "IMMICH_API_URL": "http://immich:2283",
            "IMMICH_API_KEY": "my-key",
        }
        # Clear any existing env vars that might interfere
        clear_vars = [
            "VLM_API_URL", "VLM_API_KEY", "VLM_MODEL_NAME",
            "CLASSIFY_DB_PATH", "CLASSIFY_CONCURRENCY",
            "CLASSIFY_TIMEOUT", "CLASSIFY_IMAGE_SIZE",
        ]
        clean_env = {k: v for k, v in env.items()}
        for var in clear_vars:
            clean_env[var] = ""

        with patch.dict(os.environ, clean_env, clear=False):
            # Remove empty strings so defaults take effect
            for var in clear_vars:
                os.environ.pop(var, None)
            config = load_config(env_file="/dev/null")

        assert config.vlm_api_url == "http://localhost:8000/v1"
        assert config.vlm_api_key == "no-key"
        assert config.vlm_model_name == ""
        assert config.database_path == "./classify.db"
        assert config.concurrency == 1
        assert config.timeout == 60
        assert config.image_size == "thumbnail"

    def test_missing_required_fields(self) -> None:
        # Remove required env vars
        env_to_clear = ["IMMICH_API_URL", "IMMICH_API_KEY"]
        with patch.dict(os.environ, {}, clear=False):
            for var in env_to_clear:
                os.environ.pop(var, None)
            with pytest.raises(SystemExit):
                load_config(env_file="/dev/null")

    def test_url_trailing_slash_stripped(self) -> None:
        env = {
            "IMMICH_API_URL": "http://immich:2283/",
            "IMMICH_API_KEY": "my-key",
            "VLM_API_URL": "http://vlm:8000/v1/",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config(env_file="/dev/null")

        assert config.immich_api_url == "http://immich:2283"
        assert config.vlm_api_url == "http://vlm:8000/v1"
