"""OpenAI-compatible VLM client for image classification."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
from loguru import logger

from immich_classify.prompt import ClassificationPrompt


class VLMError(Exception):
    """Error raised when VLM classification fails."""

    def __init__(self, message: str, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class VLMClient:
    """Async client for OpenAI-compatible Vision Language Model APIs."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "",
        timeout: int = 60,
    ) -> None:
        """Initialize the VLM client.

        Args:
            api_url: Base URL of the OpenAI-compatible API (e.g. http://localhost:8000/v1).
            api_key: API key for authentication.
            model_name: Model name to use. Empty string uses server default.
            timeout: Request timeout in seconds.
        """
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def classify_image(
        self,
        image_base64: str,
        prompt_config: ClassificationPrompt,
        content_type: str = "image/jpeg",
    ) -> dict[str, Any]:
        """Send an image to the VLM for classification.

        Args:
            image_base64: Base64-encoded image data.
            prompt_config: The classification prompt and schema configuration.
            content_type: MIME type of the image.

        Returns:
            Parsed JSON classification result.

        Raises:
            VLMError: If the request fails or the response cannot be parsed.
        """
        data_uri = f"data:{content_type};base64,{image_base64}"
        user_prompt = prompt_config.build_user_prompt()
        json_schema = prompt_config.build_json_schema()

        request_body: dict[str, Any] = {
            "messages": [
                {
                    "role": "system",
                    "content": prompt_config.system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "classification",
                    "strict": True,
                    "schema": json_schema,
                },
            },
        }

        if self._model_name:
            request_body["model"] = self._model_name

        url = f"{self._api_url}/chat/completions"

        try:
            response = await self._client.post(url, json=request_body)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise VLMError(f"VLM request timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raw = exc.response.text
            raise VLMError(
                f"VLM API error {exc.response.status_code}: {raw}",
                raw_response=raw,
            ) from exc
        except httpx.ConnectError as exc:
            raise VLMError(f"Cannot connect to VLM API at {url}: {exc}") from exc

        raw_text = response.text
        try:
            response_data: dict[str, Any] = response.json()
        except json.JSONDecodeError as exc:
            raise VLMError(
                f"VLM returned invalid JSON response: {exc}",
                raw_response=raw_text,
            ) from exc

        # Extract the message content from OpenAI response format
        try:
            raw_choices: Any = response_data["choices"]
            assert isinstance(raw_choices, list)
            choices = cast(list[dict[str, Any]], raw_choices)
            assert len(choices) > 0
            first_choice = choices[0]
            message = first_choice["message"]
            assert isinstance(message, dict)
            message_dict = cast(dict[str, Any], message)
            message_content: str = str(message_dict["content"])
        except (KeyError, IndexError, AssertionError) as exc:
            raise VLMError(
                f"Unexpected VLM response structure: {exc}",
                raw_response=raw_text,
            ) from exc

        # Parse the message content as JSON
        try:
            parsed_result: Any = json.loads(message_content)
        except json.JSONDecodeError as exc:
            raise VLMError(
                f"VLM response content is not valid JSON: {exc}",
                raw_response=raw_text,
            ) from exc

        assert isinstance(parsed_result, dict)
        result = cast(dict[str, Any], parsed_result)
        logger.debug("VLM classification result: {}", result)
        return result
