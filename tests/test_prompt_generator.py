"""Tests for AI-assisted prompt generation."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from immich_classify.prompt_base import BasePrompt, SchemaField
from immich_classify.prompt_generator import (
    PromptGenerator,
    PromptGeneratorError,
    export_as_python,
)


class TestExportAsPython:
    """Tests for export_as_python utility."""

    def test_export_creates_file(self) -> None:
        prompt = BasePrompt()
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name
        try:
            export_as_python(prompt, path)
            assert os.path.exists(path)
            content = open(path).read()
            assert "prompt = BasePrompt(" in content
            assert "SchemaField" in content
        finally:
            os.unlink(path)

    def test_export_contains_all_schema_fields(self) -> None:
        prompt = BasePrompt(
            prompt_type="custom",
            schema={
                "has_smile": SchemaField(
                    field_type="bool",
                    description="Is smiling",
                ),
                "tags": SchemaField(
                    field_type="list[string]",
                    description="Tags",
                    enum=["a", "b"],
                ),
                "count": SchemaField(
                    field_type="int",
                    description="Count",
                    default=0,
                ),
            },
        )
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name
        try:
            export_as_python(prompt, path)
            content = open(path).read()
            assert "'has_smile'" in content
            assert "'tags'" in content
            assert "'count'" in content
            assert "enum=" in content
            assert "default=0" in content
        finally:
            os.unlink(path)

    def test_exported_file_is_loadable(self) -> None:
        """Verify the exported .py file can be exec'd and contains a valid prompt."""
        prompt = BasePrompt(
            prompt_type="test_export",
            system_prompt="Test system prompt",
            user_prompt="Test {schema_description} prompt",
            schema={
                "field_a": SchemaField(field_type="string", description="A field"),
                "field_b": SchemaField(field_type="bool", description="B field"),
            },
        )
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name
        try:
            export_as_python(prompt, path)

            # Load the module
            import importlib.util

            spec = importlib.util.spec_from_file_location("test_module", path)
            assert spec is not None
            assert spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            loaded = getattr(module, "prompt", None)
            assert isinstance(loaded, BasePrompt)
            assert loaded.prompt_type == "test_export"
            assert "field_a" in loaded.schema
            assert "field_b" in loaded.schema
        finally:
            os.unlink(path)


class TestPromptGenerator:
    """Tests for PromptGenerator (mocked API)."""

    @pytest.mark.asyncio
    async def test_generate_parses_valid_response(self) -> None:
        """Test that generate() correctly parses a well-formed API response."""
        generated_config = {
            "prompt_type": "custom",
            "system_prompt": "You detect cats in images.",
            "user_prompt": "Detect cats: {schema_description}",
            "schema": {
                "has_cat": {
                    "field_type": "bool",
                    "description": "Whether the image contains a cat",
                },
                "cat_count": {
                    "field_type": "int",
                    "description": "Number of cats",
                    "default": 0,
                },
            },
        }

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(generated_config),
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        generator = PromptGenerator(
            api_url="http://localhost:8000/v1",
            api_key="test-key",
            model_name="test-model",
        )

        # Monkey-patch the client.post to return our mock
        import httpx

        async def mock_post(url: str, json: dict) -> httpx.Response:  # type: ignore[type-arg]
            return httpx.Response(
                status_code=200,
                json=mock_response,
                request=httpx.Request("POST", url),
            )

        generator._client.post = mock_post  # type: ignore[assignment]

        try:
            result = await generator.generate("find cats in photos")
            assert isinstance(result, BasePrompt)
            assert "has_cat" in result.schema
            assert result.schema["has_cat"].field_type == "bool"
            assert "cat_count" in result.schema
        finally:
            await generator.close()


class TestPromptGeneratorError:
    """Tests for PromptGeneratorError."""

    def test_error_with_raw_response(self) -> None:
        err = PromptGeneratorError("something failed", raw_response='{"error": "bad"}')
        assert str(err) == "something failed"
        assert err.raw_response == '{"error": "bad"}'

    def test_error_without_raw_response(self) -> None:
        err = PromptGeneratorError("something failed")
        assert err.raw_response is None
