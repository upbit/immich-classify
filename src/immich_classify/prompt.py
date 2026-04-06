"""Classification prompt and schema definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast


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
class ClassificationPrompt:
    """Classification prompt and schema definition."""

    system_prompt: str = (
        "You are an image classification assistant. "
        "Analyze the given image and output a JSON object "
        "following the specified schema. Output ONLY valid JSON, no other text."
    )

    user_prompt: str = (
        "Classify this image according to the following schema:\n"
        "{schema_description}\n\n"
        "Output a JSON object with the specified fields."
    )

    schema: dict[str, SchemaField] = field(default_factory=lambda: {
        "category": SchemaField(
            field_type="string",
            description="Primary category of the image",
            enum=[
                "people", "landscape", "food", "animal", "architecture",
                "document", "screenshot", "meme", "other",
            ],
        ),
        "quality": SchemaField(
            field_type="string",
            description="Image quality assessment",
            enum=["high", "medium", "low", "blurry"],
        ),
        "tags": SchemaField(
            field_type="list[string]",
            description="Descriptive tags for the image content",
        ),
    })

    def build_schema_description(self) -> str:
        """Build a human-readable schema description from all fields.

        Returns:
            A formatted string describing the full classification schema.
        """
        lines: list[str] = []
        for name, schema_field in self.schema.items():
            lines.append(schema_field.to_schema_description(name))
        return "\n".join(lines)

    def build_user_prompt(self) -> str:
        """Build the final user prompt with schema description filled in.

        Returns:
            The user prompt with {schema_description} replaced.
        """
        return self.user_prompt.format(
            schema_description=self.build_schema_description()
        )

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
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "schema": schema_dict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationPrompt:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary representation of a prompt configuration.

        Returns:
            A ClassificationPrompt instance.
        """
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

        # Use a temporary default instance to get field defaults
        defaults = cls()
        return cls(
            system_prompt=data.get("system_prompt", defaults.system_prompt),
            user_prompt=data.get("user_prompt", defaults.user_prompt),
            schema=schema,
        )
