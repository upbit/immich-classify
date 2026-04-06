"""Tests for VLM client."""

from __future__ import annotations

import json

import httpx
import pytest

from immich_classify.prompts.classification import ClassificationPrompt
from immich_classify.vlm_client import (
    VLMClient,
    VLMError,
    _extract_json_from_mixed_content,
    _strip_markdown_json,
)


def _make_vlm_client(handler: httpx.MockTransport) -> VLMClient:
    """Create a VLMClient with a mock transport."""
    client = VLMClient.__new__(VLMClient)
    client._api_url = "http://localhost:8000/v1"
    client._api_key = "test-key"
    client._model_name = "test-model"
    client._client = httpx.AsyncClient(
        transport=handler,
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
    )
    return client


class TestVLMClassify:
    """Tests for VLMClient.classify_image."""

    @pytest.mark.asyncio
    async def test_successful_classification(self) -> None:
        classification_result = {
            "category": "people",
            "quality": "high",
            "tags": ["portrait", "outdoor"],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["model"] == "test-model"
            assert body["temperature"] == 0.6
            assert "response_format" in body
            # Verify message structure
            messages = body["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            user_content = messages[1]["content"]
            assert any(item["type"] == "image_url" for item in user_content)

            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(classification_result),
                            }
                        }
                    ]
                },
            )

        client = _make_vlm_client(httpx.MockTransport(handler))
        prompt = ClassificationPrompt()
        result = await client.classify_image("base64data", prompt)
        assert result["category"] == "people"
        assert result["quality"] == "high"
        assert result["tags"] == ["portrait", "outdoor"]
        await client.close()

    @pytest.mark.asyncio
    async def test_no_model_name(self) -> None:
        """When model_name is empty, the 'model' field should be omitted."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert "model" not in body
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"category": "other"}'}}
                    ]
                },
            )

        client = _make_vlm_client(httpx.MockTransport(handler))
        client._model_name = ""
        result = await client.classify_image("base64data", ClassificationPrompt())
        assert result["category"] == "other"
        await client.close()

    @pytest.mark.asyncio
    async def test_api_error_raises_vlm_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        client = _make_vlm_client(httpx.MockTransport(handler))
        with pytest.raises(VLMError, match="VLM API error 500"):
            await client.classify_image("base64data", ClassificationPrompt())
        await client.close()

    @pytest.mark.asyncio
    async def test_invalid_json_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json at all")

        client = _make_vlm_client(httpx.MockTransport(handler))
        with pytest.raises(VLMError, match="invalid JSON"):
            await client.classify_image("base64data", ClassificationPrompt())
        await client.close()

    @pytest.mark.asyncio
    async def test_unexpected_response_structure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"result": "no choices field"})

        client = _make_vlm_client(httpx.MockTransport(handler))
        with pytest.raises(VLMError, match="Unexpected VLM response"):
            await client.classify_image("base64data", ClassificationPrompt())
        await client.close()

    @pytest.mark.asyncio
    async def test_non_json_message_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "I cannot classify this image"}}
                    ]
                },
            )

        client = _make_vlm_client(httpx.MockTransport(handler))
        with pytest.raises(VLMError, match="not valid JSON"):
            await client.classify_image("base64data", ClassificationPrompt())
        await client.close()

    @pytest.mark.asyncio
    async def test_structured_output_format(self) -> None:
        """Verify the response_format field is correctly structured."""

        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"category": "food"}'}}
                    ]
                },
            )

        client = _make_vlm_client(httpx.MockTransport(handler))
        await client.classify_image("base64data", ClassificationPrompt())

        response_format = captured_body["response_format"]
        assert isinstance(response_format, dict)
        assert response_format["type"] == "json_schema"
        json_schema = response_format["json_schema"]
        assert isinstance(json_schema, dict)
        assert json_schema["name"] == "classification"
        assert json_schema["strict"] is True
        schema = json_schema["schema"]
        assert isinstance(schema, dict)
        assert "category" in schema["properties"]
        await client.close()

    @pytest.mark.asyncio
    async def test_markdown_wrapped_json_is_parsed(self) -> None:
        """VLM returns JSON wrapped in ```json ... ``` — should be parsed correctly."""
        wrapped_content = '```json\n{"category": "architecture", "quality": "medium", "tags": ["building"]}\n```'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": wrapped_content}}
                    ]
                },
            )

        client = _make_vlm_client(httpx.MockTransport(handler))
        result = await client.classify_image("base64data", ClassificationPrompt())
        assert result["category"] == "architecture"
        assert result["quality"] == "medium"
        assert result["tags"] == ["building"]
        await client.close()

    @pytest.mark.asyncio
    async def test_markdown_wrapped_no_language_tag(self) -> None:
        """VLM returns JSON wrapped in ``` ... ``` (no language tag)."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": '```\n{"category": "food"}\n```'}}
                    ]
                },
            )

        client = _make_vlm_client(httpx.MockTransport(handler))
        result = await client.classify_image("base64data", ClassificationPrompt())
        assert result["category"] == "food"
        await client.close()


class TestStripMarkdownJson:
    """Tests for _strip_markdown_json utility function."""

    def test_plain_json_unchanged(self) -> None:
        text = '{"category": "people"}'
        assert _strip_markdown_json(text) == text

    def test_strip_json_code_block(self) -> None:
        text = '```json\n{"category": "food"}\n```'
        assert _strip_markdown_json(text) == '{"category": "food"}'

    def test_strip_bare_code_block(self) -> None:
        text = '```\n{"key": "value"}\n```'
        assert _strip_markdown_json(text) == '{"key": "value"}'

    def test_strip_with_extra_whitespace(self) -> None:
        text = '  ```json\n  {"a": 1}  \n```  '
        result = _strip_markdown_json(text)
        assert json.loads(result) == {"a": 1}

    def test_strip_JSON_uppercase_tag(self) -> None:
        text = '```JSON\n{"x": 1}\n```'
        assert _strip_markdown_json(text) == '{"x": 1}'

    def test_multiline_json_preserved(self) -> None:
        text = '```json\n{\n  "category": "landscape",\n  "tags": ["sky", "mountain"]\n}\n```'
        result = _strip_markdown_json(text)
        parsed = json.loads(result)
        assert parsed["category"] == "landscape"
        assert parsed["tags"] == ["sky", "mountain"]

    def test_no_code_block_returns_original(self) -> None:
        text = "This is not JSON at all"
        assert _strip_markdown_json(text) == text

    def test_partial_code_block_returns_original(self) -> None:
        text = '```json\n{"incomplete": true}'
        assert _strip_markdown_json(text) == text


class TestExtractJsonFromMixedContent:
    """Tests for _extract_json_from_mixed_content utility function."""

    def test_reasoning_then_code_block(self) -> None:
        """Model outputs reasoning text followed by a ```json code block."""
        text = (
            "Based on the image provided, I can identify the following people:\n"
            "- One adult woman holding a baby.\n"
            "- One infant being held by the woman.\n"
            "Therefore, the total count of people is 2.\n\n"
            '```json\n{\n  "person_count": 2\n}\n```'
        )
        result = _extract_json_from_mixed_content(text)
        assert json.loads(result) == {"person_count": 2}

    def test_reasoning_then_bare_json(self) -> None:
        """Model outputs reasoning text followed by bare JSON (no code block)."""
        text = (
            "The image shows a landscape with mountains.\n\n"
            '{"category": "landscape", "quality": "high"}'
        )
        result = _extract_json_from_mixed_content(text)
        assert json.loads(result) == {"category": "landscape", "quality": "high"}

    def test_plain_json_returned_as_is(self) -> None:
        """Pure JSON input is returned unchanged."""
        text = '{"category": "food"}'
        result = _extract_json_from_mixed_content(text)
        assert result == text

    def test_no_json_returns_original(self) -> None:
        """Text with no JSON at all is returned unchanged."""
        text = "I cannot classify this image because it is too dark."
        assert _extract_json_from_mixed_content(text) == text

    def test_embedded_code_block_preferred_over_bare_json(self) -> None:
        """When both a code block and bare JSON exist, the code block wins."""
        text = (
            'Some text mentioning {"partial": true} in passing.\n\n'
            '```json\n{"actual": "result"}\n```'
        )
        result = _extract_json_from_mixed_content(text)
        assert json.loads(result) == {"actual": "result"}


class TestMixedContentClassification:
    """Integration tests: VLM returns reasoning + JSON, should still parse."""

    @pytest.mark.asyncio
    async def test_reasoning_before_json_code_block(self) -> None:
        """Qwen-style output: reasoning text then ```json ... ```."""
        mixed_content = (
            "Based on the image provided, I can identify the following people:\n"
            "- One adult woman holding a baby.\n"
            "Therefore, the total count of people is 2.\n\n"
            '```json\n{"person_count": 2}\n```'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": mixed_content}}
                    ]
                },
            )

        client = _make_vlm_client(httpx.MockTransport(handler))
        result = await client.classify_image("base64data", ClassificationPrompt())
        assert result["person_count"] == 2
        await client.close()

    @pytest.mark.asyncio
    async def test_reasoning_before_bare_json(self) -> None:
        """Model outputs reasoning text then bare JSON without code block."""
        mixed_content = (
            "This image shows a plate of sushi.\n\n"
            '{"category": "food", "quality": "high", "tags": ["sushi"]}'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": mixed_content}}
                    ]
                },
            )

        client = _make_vlm_client(httpx.MockTransport(handler))
        result = await client.classify_image("base64data", ClassificationPrompt())
        assert result["category"] == "food"
        assert result["tags"] == ["sushi"]
        await client.close()
