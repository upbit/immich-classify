"""AI-assisted prompt configuration generator.

Uses a strong LLM to generate ``BasePrompt`` configurations from
natural language task descriptions.  The generated configs can be previewed,
tested via the ``debug`` command, and exported as reusable ``.py`` files.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any, cast

import httpx
import jinja2
from loguru import logger

from immich_classify.prompt_base import BasePrompt

# ── Meta-prompt for the generator LLM ──────────────────────────────

_GENERATOR_SYSTEM_PROMPT = """\
You are an expert at designing image classification schemas for Vision Language Models.

Given a user's task description, generate a BasePrompt configuration as a JSON object with these exact fields:

{
  "name": "custom",
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


def _derive_class_name(file_path: str) -> str:
    """Derive a PascalCase class name from a file path.

    Examples:
        ``foreground_people.py`` → ``ForegroundPeoplePrompt``
        ``cat_finder.py``        → ``CatFinderPrompt``

    Args:
        file_path: The output file path.

    Returns:
        A PascalCase class name ending with ``Prompt``.
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    # Convert snake_case to PascalCase
    pascal = "".join(word.capitalize() for word in stem.split("_"))
    # Append "Prompt" if not already present
    if not pascal.endswith("Prompt"):
        pascal += "Prompt"
    return pascal


# ── Jinja2 template for exported prompt files ──────────────────────

_JINJA_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    keep_trailing_newline=True,
)
_JINJA_ENV.filters["repr"] = repr

_EXPORT_TEMPLATE = _JINJA_ENV.from_string(
    '''\
"""Auto-generated prompt configuration.

Load with: immich-classify classify --prompt-config {{ filename }}
"""

from dataclasses import dataclass, field

from immich_classify.prompt_base import BasePrompt, SchemaField, register_prompt


@register_prompt
@dataclass
class {{ class_name }}(BasePrompt):
    """Auto-generated prompt: {{ prompt_name }}."""

    name: str = {{ prompt_name | repr }}

    system_prompt: str = (
{%- for line in system_prompt_lines %}
        {{ line | repr }}
{%- endfor %}
    )

    user_prompt: str = (
{%- for line in user_prompt_lines %}
        {{ line | repr }}
{%- endfor %}
    )

    schema: dict[str, SchemaField] = field(default_factory=lambda: {
{%- for field_name, sf in schema_items %}
        {{ field_name | repr }}: SchemaField(
            field_type={{ sf.field_type | repr }},
            description={{ sf.description | repr }},
{%- if sf.enum is not none %}
            enum={{ sf.enum | repr }},
{%- endif %}
{%- if sf.default is not none %}
            default={{ sf.default | repr }},
{%- endif %}
        ),
{%- endfor %}
    })


# Module-level instance for --prompt-config loading
prompt = {{ class_name }}()
'''
)


def export_as_python(prompt: BasePrompt, path: str) -> None:
    """Export a BasePrompt as a loadable Python file with a class definition.

    The generated file defines a ``BasePrompt`` subclass (with a name
    derived from the file path) decorated with ``@register_prompt`` and
    ``@dataclass``, plus a module-level ``prompt`` instance for backward
    compatibility with ``--prompt-config``.

    Uses a Jinja2 template for clearer structure and easier maintenance.

    Args:
        prompt: The prompt configuration to export.
        path: File path to write to.
    """
    class_name = _derive_class_name(path)
    prompt_name = os.path.splitext(os.path.basename(path))[0]

    # Wrap system_prompt lines to stay within PEP 8 line width
    system_prompt_lines = textwrap.wrap(prompt.system_prompt, width=72)

    # Split user_prompt by newlines, appending \n to each segment
    # so the reconstructed string preserves original line breaks
    user_prompt_lines = [
        line + "\n" for line in prompt.user_prompt.split("\n")
    ]

    rendered = _EXPORT_TEMPLATE.render(
        filename=os.path.basename(path),
        class_name=class_name,
        prompt_name=prompt_name,
        system_prompt_lines=system_prompt_lines,
        user_prompt_lines=user_prompt_lines,
        schema_items=list(prompt.schema.items()),
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(rendered)

    logger.info("Exported prompt config to {}", path)
