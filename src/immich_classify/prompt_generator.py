"""AI-assisted prompt configuration generator.

Uses a strong LLM to generate ``BasePrompt`` configurations from
natural language task descriptions.  The generated configs can be previewed,
tested via the ``debug`` command, and exported as reusable ``.py`` files.
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any, cast

import httpx
from loguru import logger

from immich_classify.prompt_base import BasePrompt

# ── Meta-prompt for the generator LLM ──────────────────────────────

_GENERATOR_SYSTEM_PROMPT = """\
You are an expert at designing image classification schemas for Vision Language Models.

Given a user's task description, generate a BasePrompt configuration as a JSON object with these exact fields:

{
  "prompt_type": "custom",
  "system_prompt": "<system message for the VLM>",
  "user_prompt": "<user message template — MUST contain the literal placeholder {schema_description}>",
  "schema": {
    "<field_name>": {
      "field_type": "<one of: string, int, float, bool, list[string]>",
      "description": "<what this field captures>",
      "enum": ["<optional list of allowed values>"],
      "default": "<optional default value, use null if required>"
    }
  }
}

Rules:
1. The user_prompt MUST contain exactly one occurrence of {schema_description} as a placeholder.
2. Keep schemas focused: 3-8 fields that directly serve the user's goal.
3. Use the most specific field_type for each field (prefer bool for yes/no, int for counts).
4. Include enum constraints when the set of values is well-defined and small.
5. The system_prompt should clearly define the VLM's role for this specific task.
6. Output ONLY the JSON object, no markdown wrappers or extra text.
"""

# Regex to match ```json ... ``` or ``` ... ``` wrappers
_MARKDOWN_CODE_BLOCK_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)


class PromptGeneratorError(Exception):
    """Error raised when prompt generation fails."""

    def __init__(self, message: str, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class PromptGenerator:
    """Generate BasePrompt configs using a strong LLM.

    This class calls an OpenAI-compatible text API (which may differ from
    the VLM used for image classification) to turn natural language
    descriptions into structured prompt configurations.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "",
        timeout: int = 60,
    ) -> None:
        """Initialize the prompt generator.

        Args:
            api_url: Base URL of the OpenAI-compatible API.
            api_key: API key for authentication.
            model_name: Model name to use.  Empty string uses server default.
            timeout: Request timeout in seconds.
        """
        self._api_url = api_url.rstrip("/")
        self._model_name = model_name
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def generate(self, description: str) -> BasePrompt:
        """Generate a BasePrompt from a natural language description.

        Args:
            description: What the user wants to classify or detect
                (e.g. "detect whether people in photos are smiling").

        Returns:
            A BasePrompt instance with generated system/user prompts
            and schema fields.

        Raises:
            PromptGeneratorError: If the API call fails or the response
                cannot be parsed into a valid prompt configuration.
        """
        request_body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": _GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ],
            "temperature": 0.3,
            "max_tokens": 2048,
        }
        if self._model_name:
            request_body["model"] = self._model_name

        url = f"{self._api_url}/chat/completions"

        logger.debug("PromptGenerator request: POST {}", url)
        logger.debug("PromptGenerator description: {}", description)

        try:
            response = await self._client.post(url, json=request_body)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise PromptGeneratorError(
                f"Generator request timed out: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raw = exc.response.text
            raise PromptGeneratorError(
                f"Generator API error {exc.response.status_code}: {raw}",
                raw_response=raw,
            ) from exc
        except httpx.ConnectError as exc:
            raise PromptGeneratorError(
                f"Cannot connect to generator API at {url}: {exc}"
            ) from exc

        raw_text = response.text
        logger.debug("PromptGenerator raw response: {}", raw_text[:2000])

        # Extract content from OpenAI response format
        try:
            response_data: dict[str, Any] = response.json()
            raw_choices: Any = response_data["choices"]
            assert isinstance(raw_choices, list)
            choices = cast(list[dict[str, Any]], raw_choices)
            message = cast(dict[str, Any], choices[0]["message"])
            content: str = str(message["content"])
        except (KeyError, IndexError, AssertionError, json.JSONDecodeError) as exc:
            raise PromptGeneratorError(
                f"Unexpected generator response structure: {exc}",
                raw_response=raw_text,
            ) from exc

        # Strip markdown code block wrappers if present
        match = _MARKDOWN_CODE_BLOCK_RE.match(content.strip())
        if match:
            content = match.group(1).strip()

        # Parse as JSON → BasePrompt
        try:
            parsed: Any = json.loads(content)
        except json.JSONDecodeError as exc:
            raise PromptGeneratorError(
                f"Generator output is not valid JSON: {exc}",
                raw_response=content,
            ) from exc

        if not isinstance(parsed, dict):
            raise PromptGeneratorError(
                f"Generator output is not a JSON object (got {type(parsed).__name__})",
                raw_response=content,
            )

        prompt_data = cast(dict[str, Any], parsed)

        # Validate that user_prompt contains the required placeholder
        user_prompt = str(prompt_data.get("user_prompt", ""))
        if "{schema_description}" not in user_prompt:
            logger.warning(
                "Generated user_prompt is missing {{schema_description}} placeholder; "
                "appending it automatically."
            )
            prompt_data["user_prompt"] = (
                user_prompt + "\n\n{schema_description}"
            )

        try:
            return BasePrompt.from_dict(prompt_data)
        except (KeyError, AssertionError) as exc:
            raise PromptGeneratorError(
                f"Generated config has invalid structure: {exc}",
                raw_response=content,
            ) from exc


def export_as_python(prompt: BasePrompt, path: str) -> None:
    """Export a BasePrompt as a loadable Python file.

    The generated file defines a ``prompt`` variable that can be used
    with ``--prompt-config``.

    Args:
        prompt: The prompt configuration to export.
        path: File path to write to.
    """
    lines: list[str] = [
        '"""Auto-generated prompt configuration.',
        "",
        "Load with: immich-classify classify --prompt-config THIS_FILE.py",
        '"""',
        "",
        "from immich_classify.prompt_base import BasePrompt, SchemaField",
        "",
        "prompt = BasePrompt(",
        f"    prompt_type={prompt.prompt_type!r},",
        f"    system_prompt=(",
    ]

    # Wrap long strings nicely
    for line in textwrap.wrap(prompt.system_prompt, width=72):
        lines.append(f"        {line!r}")
    lines.append("    ),")

    lines.append("    user_prompt=(")
    for line in prompt.user_prompt.split("\n"):
        lines.append(f"        {(line + chr(10))!r}")
    lines.append("    ),")

    lines.append("    schema={")
    for name, sf in prompt.schema.items():
        parts: list[str] = [
            f"        {name!r}: SchemaField(",
            f"            field_type={sf.field_type!r},",
            f"            description={sf.description!r},",
        ]
        if sf.enum is not None:
            parts.append(f"            enum={sf.enum!r},")
        if sf.default is not None:
            parts.append(f"            default={sf.default!r},")
        parts.append("        ),")
        lines.extend(parts)

    lines.extend([
        "    },",
        ")",
        "",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Exported prompt config to {}", path)
