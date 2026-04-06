"""Classification prompt and schema definitions.

Provides a base ``ClassificationPrompt`` class and several pre-configured
subclasses for common tasks (tagging, smile detection, person filtering).
Custom task types can subclass ``ClassificationPrompt`` and will be
automatically discoverable via the prompt registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

# ── Prompt registry ─────────────────────────────────────────────────
# Maps prompt_type strings to their concrete ClassificationPrompt subclass
# so that ``from_dict()`` can reconstruct the correct type.
_PROMPT_REGISTRY: dict[str, type[ClassificationPrompt]] = {}


def _register_prompt(cls: type[ClassificationPrompt]) -> type[ClassificationPrompt]:
    """Class decorator that adds a prompt subclass to the registry."""
    # Resolve the default prompt_type from the dataclass field
    prompt_type = cls.__dataclass_fields__["prompt_type"].default
    assert isinstance(prompt_type, str), f"prompt_type must be a str, got {type(prompt_type)}"
    _PROMPT_REGISTRY[prompt_type] = cls
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


@_register_prompt
@dataclass
class ClassificationPrompt:
    """Base classification prompt and schema definition.

    Subclasses override default field values to provide pre-configured
    prompts for specific tasks.  All subclasses share the same interface
    and can be used interchangeably by ``VLMClient`` and ``TaskEngine``.
    """

    prompt_type: str = "classification"

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
            "prompt_type": self.prompt_type,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "schema": schema_dict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationPrompt:
        """Deserialize from a dictionary.

        Uses the ``prompt_type`` field to look up the correct subclass in the
        prompt registry.  Falls back to the base ``ClassificationPrompt`` when
        the type is unknown or missing (backward compatibility).

        Args:
            data: Dictionary representation of a prompt configuration.

        Returns:
            A ClassificationPrompt (or subclass) instance.
        """
        prompt_type = str(data.get("prompt_type", "classification"))
        target_cls = _PROMPT_REGISTRY.get(prompt_type, cls)

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


# ── Pre-configured prompt subclasses ────────────────────────────────


@_register_prompt
@dataclass
class TaggingPrompt(ClassificationPrompt):
    """Pre-configured prompt for image tagging.

    Focuses on extracting descriptive tags and scene type from images.
    """

    prompt_type: str = "tagging"

    system_prompt: str = (
        "You are an image tagging assistant. "
        "Analyze the given image and output a JSON object with descriptive tags. "
        "Output ONLY valid JSON, no other text."
    )

    user_prompt: str = (
        "Tag this image according to the following schema:\n"
        "{schema_description}\n\n"
        "Provide specific, descriptive tags that capture the key subjects, "
        "actions, setting, and mood of the image. Output a JSON object."
    )

    schema: dict[str, SchemaField] = field(default_factory=lambda: {
        "tags": SchemaField(
            field_type="list[string]",
            description="Descriptive tags for the image content (5-15 tags)",
        ),
        "scene": SchemaField(
            field_type="string",
            description="Primary scene type",
            enum=["indoor", "outdoor", "studio", "screenshot", "other"],
        ),
    })


@_register_prompt
@dataclass
class SmileDetectionPrompt(ClassificationPrompt):
    """Pre-configured prompt for smile and facial expression detection."""

    prompt_type: str = "smile_detection"

    system_prompt: str = (
        "You are a facial expression analysis assistant. "
        "Analyze the given image for people and their facial expressions. "
        "Output ONLY valid JSON, no other text."
    )

    user_prompt: str = (
        "Analyze facial expressions in this image according to the following schema:\n"
        "{schema_description}\n\n"
        "If no people are visible, set has_people to false and use default values "
        "for other fields. Output a JSON object."
    )

    schema: dict[str, SchemaField] = field(default_factory=lambda: {
        "has_people": SchemaField(
            field_type="bool",
            description="Whether the image contains people",
        ),
        "has_smile": SchemaField(
            field_type="bool",
            description="Whether anyone in the image is smiling",
            default=False,
        ),
        "smile_count": SchemaField(
            field_type="int",
            description="Number of people who are smiling",
            default=0,
        ),
        "expression": SchemaField(
            field_type="string",
            description="Dominant facial expression in the image",
            enum=["happy", "neutral", "sad", "surprised", "angry", "other", "no_face"],
            default="no_face",
        ),
    })


@_register_prompt
@dataclass
class PersonFilterPrompt(ClassificationPrompt):
    """Pre-configured prompt for filtering images by person presence."""

    prompt_type: str = "person_filter"

    system_prompt: str = (
        "You are an image analysis assistant specializing in detecting people. "
        "Analyze the given image and determine if it contains people. "
        "Output ONLY valid JSON, no other text."
    )

    user_prompt: str = (
        "Analyze this image for the presence of people according to the following schema:\n"
        "{schema_description}\n\n"
        "Output a JSON object with the specified fields."
    )

    schema: dict[str, SchemaField] = field(default_factory=lambda: {
        "has_person": SchemaField(
            field_type="bool",
            description="Whether the image contains any people",
        ),
        "person_count": SchemaField(
            field_type="int",
            description="Number of people visible in the image",
            default=0,
        ),
        "is_portrait": SchemaField(
            field_type="bool",
            description="Whether the image is a portrait (one person as the main subject)",
            default=False,
        ),
    })
