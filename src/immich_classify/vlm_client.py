"""OpenAI-compatible VLM client for image classification."""

from __future__ import annotations

import json
import re
from typing import Any, cast

import httpx
from loguru import logger

from immich_classify.prompt_base import BasePrompt


class VLMError(Exception):
    """Error raised when VLM classification fails."""

    def __init__(self, message: str, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


# Regex to match ```json ... ``` or ``` ... ``` wrappers (with optional language tag)
_MARKDOWN_CODE_BLOCK_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)

# Regex to find a ```json ... ``` block anywhere in the text (not just at start/end).
# Used as a fallback when models output reasoning text before the JSON block.
_EMBEDDED_CODE_BLOCK_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(\{.*?\})\n?\s*```",
    re.DOTALL,
)

# Regex to find a top-level JSON object anywhere in the text.
# Used as a last-resort fallback when models output reasoning text
# followed by bare JSON (no code block wrapper).
_BARE_JSON_OBJECT_RE = re.compile(
    r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
    re.DOTALL,
)


def _strip_markdown_json(text: str) -> str:
    """Strip markdown code block wrappers from a string.

    Many local models wrap their JSON output in ```json ... ``` even when
    structured output (response_format) is requested. This function extracts
    the inner content.

    Args:
        text: Raw message content from the VLM.

    Returns:
        The inner content if a code block wrapper was found, otherwise the
        original text unchanged.
    """
    match = _MARKDOWN_CODE_BLOCK_RE.match(text.strip())
    if match:
        return match.group(1).strip()
    return text


def _extract_json_from_mixed_content(text: str) -> str:
    """Extract JSON from model output that contains reasoning text before JSON.

    Some models (e.g. Qwen3.5) output a chain-of-thought explanation followed
    by the actual JSON result.  This function attempts multiple strategies to
    locate and extract the JSON object:

    1. Look for an embedded ```json ... ``` code block anywhere in the text.
    2. Look for a bare JSON object (``{ ... }``) in the text.

    Args:
        text: Raw message content that failed direct JSON parsing.

    Returns:
        The extracted JSON string if found, otherwise the original text
        unchanged (so the caller can raise the appropriate error).
    """
    # Strategy 1: embedded markdown code block
    embedded_match = _EMBEDDED_CODE_BLOCK_RE.search(text)
    if embedded_match:
        logger.debug(
            "Extracted JSON from embedded code block in mixed-content response"
        )
        return embedded_match.group(1).strip()

    # Strategy 2: bare JSON object — find the last { ... } which is most
    # likely the actual result (reasoning text may contain partial braces).
    bare_matches = _BARE_JSON_OBJECT_RE.findall(text)
    if bare_matches:
        # Prefer the last match — models typically put the answer at the end.
        candidate: str = bare_matches[-1].strip()
        # Quick sanity check: must parse as valid JSON
        try:
            json.loads(candidate)
            logger.debug(
                "Extracted bare JSON object from mixed-content response"
            )
            return candidate
        except json.JSONDecodeError:
            pass

    return text


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
        prompt_config: BasePrompt,
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
            "temperature": 0.6,
            "top_p": 0.9,
            "max_tokens": 1024,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": prompt_config.name,
                    "strict": True,
                    "schema": json_schema,
                },
            },
        }

        if self._model_name:
            request_body["model"] = self._model_name

        url = f"{self._api_url}/chat/completions"

        # Log request details for debugging
        logger.debug(
            "VLM request: POST {} | model={} | image_size={} chars | content_type={}",
            url,
            self._model_name or "(server default)",
            len(image_base64),
            content_type,
        )
        logger.debug("VLM system_prompt: {}", prompt_config.system_prompt[:200])
        logger.debug("VLM user_prompt: {}", user_prompt[:200])
        logger.debug("VLM json_schema: {}", json.dumps(json_schema, ensure_ascii=False))

        try:
            response = await self._client.post(url, json=request_body)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise VLMError(
                f"VLM request timed out after {self._client.timeout.read}s: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raw = exc.response.text
            logger.debug("VLM HTTP error response body: {}", raw[:2000])
            raise VLMError(
                f"VLM API error {exc.response.status_code}: {raw}",
                raw_response=raw,
            ) from exc
        except httpx.ConnectError as exc:
            raise VLMError(f"Cannot connect to VLM API at {url}: {exc}") from exc

        raw_text = response.text

        # Log full response metadata
        logger.debug(
            "VLM response: HTTP {} | content_length={} | content_type={}",
            response.status_code,
            len(raw_text),
            response.headers.get("content-type", "unknown"),
        )
        logger.debug("VLM raw response (first 2000 chars): {}", raw_text[:2000])

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

            # Log finish_reason — key diagnostic for empty/truncated responses
            finish_reason = first_choice.get("finish_reason", "unknown")
            logger.debug("VLM finish_reason: {}", finish_reason)
            if finish_reason not in ("stop", "eos"):
                logger.warning(
                    "VLM finish_reason is '{}' (expected 'stop'). "
                    "Response may be truncated or the model refused to answer.",
                    finish_reason,
                )

            message = first_choice["message"]
            assert isinstance(message, dict)
            message_dict = cast(dict[str, Any], message)
            message_content: str = str(message_dict["content"])

            # Log the extracted content
            logger.debug("VLM message.content ({} chars): {}", len(message_content), message_content[:1000])

            # Check for empty content — common failure mode
            if not message_content.strip():
                raise VLMError(
                    f"VLM returned empty message content (finish_reason={finish_reason}). "
                    "This often means: (1) the model does not support vision/image input, "
                    "(2) the model does not support structured output (response_format), or "
                    "(3) the image was rejected by the model's safety filter.",
                    raw_response=raw_text,
                )
        except (KeyError, IndexError, AssertionError) as exc:
            raise VLMError(
                f"Unexpected VLM response structure: {exc}",
                raw_response=raw_text,
            ) from exc

        # Strip markdown code block wrappers if present.
        # Many local models (Qwen, LLaMA, etc.) wrap JSON in ```json ... ```
        # even when structured output is requested.
        cleaned_content = _strip_markdown_json(message_content)
        if cleaned_content != message_content:
            logger.debug(
                "Stripped markdown code block wrapper from VLM response ({} -> {} chars)",
                len(message_content),
                len(cleaned_content),
            )

        # Parse the message content as JSON
        try:
            parsed_result: Any = json.loads(cleaned_content)
        except json.JSONDecodeError:
            # Fallback: some models (e.g. Qwen3.5) output reasoning text
            # before the actual JSON.  Try to extract JSON from mixed content.
            logger.debug(
                "Direct JSON parse failed; attempting to extract JSON from mixed-content response"
            )
            extracted_content = _extract_json_from_mixed_content(cleaned_content)
            if extracted_content != cleaned_content:
                logger.info(
                    "Extracted JSON from mixed-content VLM response "
                    "(model likely output reasoning text before JSON)"
                )
            try:
                parsed_result = json.loads(extracted_content)
            except json.JSONDecodeError as exc:
                raise VLMError(
                    f"VLM response content is not valid JSON: {exc}",
                    raw_response=raw_text,
                ) from exc

        assert isinstance(parsed_result, dict)
        result = cast(dict[str, Any], parsed_result)
        logger.debug("VLM classification result: {}", result)
        return result
