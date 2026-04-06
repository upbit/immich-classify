"""Tests for VLM client."""

from __future__ import annotations

import json

import httpx
import pytest

from immich_classify.prompt import ClassificationPrompt
from immich_classify.vlm_client import VLMClient, VLMError


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
            assert body["temperature"] == 0.1
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
