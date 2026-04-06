"""Base prompt class, schema field definition, and prompt registry.

Provides ``BasePrompt`` — the base dataclass that all concrete prompt
implementations inherit from — along with ``SchemaField`` for defining
classification fields and ``register_prompt`` for automatic discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

# ── Prompt registry ─────────────────────────────────────────────────
# Maps prompt_type strings to their concrete BasePrompt subclass so that
# ``from_dict()`` can reconstruct the correct type.
PROMPT_REGISTRY: dict[str, type[BasePrompt]] = {}


def register_prompt(cls: type[BasePrompt]) -> type[BasePrompt]:
    """Class decorator that adds a prompt subclass to the registry."""
    prompt_type = cls.__dataclass_fields__["prompt_type"].default
    assert isinstance(prompt_type, str), f"prompt_type must be a str, got {type(prompt_type)}"
    PROMPT_REGISTRY[prompt_type] = cls
    return cls


@dataclass
class SchemaField:
    """Definition of a single field in the classification schema."""

    field_type: str  # "string", "int", "float", "bool", "list[string]"
    description: str  # Field description for the prompt
    enum: list[str] | None = None  # Optional: allowed values
    default: Any = None  # Optional: default value

    def to_schema_description(self, field_name: str) -> str:
        """Generate a human-readable schema description for this field.

        Args:
            field_name: The name of this field in the schema.

        Returns:
            A formatted string describing this field.
        """
        parts = [f"- {field_name} ({self.field_type}): {self.description}"]
        if self.enum is not None:
            parts.append(f"  Allowed values: {self.enum}")
        if self.default is not None:
            parts.append(f"  Default: {self.default}")
        return "\n".join(parts)

    def to_json_schema(self) -> dict[str, Any]:
        """Convert this field to a JSON Schema property definition.

        Returns:
            A dictionary representing the JSON Schema for this field.
        """
        type_mapping: dict[str, dict[str, Any]] = {
            "string": {"type": "string"},
            "int": {"type": "integer"},
            "float": {"type": "number"},
            "bool": {"type": "boolean"},
            "list[string]": {"type": "array", "items": {"type": "string"}},
        }
        schema = type_mapping.get(self.field_type, {"type": "string"}).copy()
        schema["description"] = self.description
        if self.enum is not None:
            schema["enum"] = self.enum
        return schema


@dataclass
class BasePrompt:
    """Base prompt and schema definition for image classification.

    Subclasses override default field values to provide pre-configured
    prompts for specific tasks.  All subclasses share the same interface
    and can be used interchangeably by ``VLMClient`` and ``TaskEngine``.
    """

    prompt_type: str = ""

    system_prompt: str = ""

    user_prompt: str = ""

    schema: dict[str, SchemaField] = field(default_factory=lambda: dict[str, SchemaField]())

    def build_schema_description(self) -> str:
        """Build a human-readable schema description from all fields.

        Returns:
            A formatted string describing the full classification schema.
        """
        lines: list[str] = []
        for name, schema_field in self.schema.items():
            lines.append(schema_field.to_schema_description(name))
        return "\n".join(lines)

    # Appended to every user prompt to discourage models from emitting
    # chain-of-thought reasoning before the JSON payload.
    _JSON_ONLY_SUFFIX: str = (
        "\n\nIMPORTANT: Respond with ONLY the JSON object. "
        "Do NOT include any explanation, reasoning, or additional text "
        "before or after the JSON."
    )

    def build_user_prompt(self) -> str:
        """Build the final user prompt with schema description filled in.

        A ``_JSON_ONLY_SUFFIX`` is automatically appended to reinforce the
        requirement that the model must output **only** valid JSON without
        any preceding chain-of-thought or explanatory text.

        Returns:
            The user prompt with {schema_description} replaced and the
            JSON-only instruction appended.
        """
        base = self.user_prompt.format(
            schema_description=self.build_schema_description()
        )
        return base + self._JSON_ONLY_SUFFIX

    def build_json_schema(self) -> dict[str, Any]:
        """Build a JSON Schema object for structured output (response_format).

        Returns:
            A complete JSON Schema object for the classification output.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, schema_field in self.schema.items():
            properties[name] = schema_field.to_json_schema()
            if schema_field.default is None:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON storage.

        Returns:
            Dictionary representation of this prompt configuration.
        """
        schema_dict: dict[str, Any] = {}
        for name, schema_field in self.schema.items():
            field_dict: dict[str, Any] = {
                "field_type": schema_field.field_type,
                "description": schema_field.description,
            }
            if schema_field.enum is not None:
                field_dict["enum"] = schema_field.enum
            if schema_field.default is not None:
                field_dict["default"] = schema_field.default
            schema_dict[name] = field_dict

        return {
            "prompt_type": self.prompt_type,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "schema": schema_dict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BasePrompt:
        """Deserialize from a dictionary.

        Uses the ``prompt_type`` field to look up the correct subclass in the
        prompt registry.  Falls back to the base ``BasePrompt`` when the type
        is unknown or missing (backward compatibility).

        Args:
            data: Dictionary representation of a prompt configuration.

        Returns:
            A BasePrompt (or subclass) instance.
        """
        prompt_type = str(data.get("prompt_type", ""))
        target_cls = PROMPT_REGISTRY.get(prompt_type, cls)

        schema: dict[str, SchemaField] = {}
        raw_schema: Any = data.get("schema", {})
        assert isinstance(raw_schema, dict)
        schema_dict = cast(dict[str, Any], raw_schema)
        for name, field_data_raw in schema_dict.items():
            assert isinstance(field_data_raw, dict)
            field_data = cast(dict[str, Any], field_data_raw)
            field_type = str(field_data["field_type"])
            description = str(field_data["description"])
            raw_enum: Any = field_data.get("enum")
            enum_list: list[str] | None = None
            if raw_enum is not None:
                assert isinstance(raw_enum, list)
                enum_list = [str(e) for e in cast(list[Any], raw_enum)]
            schema[name] = SchemaField(
                field_type=field_type,
                description=description,
                enum=enum_list,
                default=field_data.get("default"),
            )

        # Use the target subclass defaults for any missing fields
        defaults = target_cls()
        return target_cls(
            prompt_type=prompt_type,
            system_prompt=data.get("system_prompt", defaults.system_prompt),
            user_prompt=data.get("user_prompt", defaults.user_prompt),
            schema=schema if schema else defaults.schema,
        )
