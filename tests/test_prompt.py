"""Tests for prompt base and classification prompt."""

from __future__ import annotations

import json

from immich_classify.prompt_base import (
    BasePrompt,
    PROMPT_REGISTRY,
    SchemaField,
)
from immich_classify.prompts.classification import ClassificationPrompt


class TestSchemaField:
    """Tests for SchemaField."""

    def test_schema_description_basic(self) -> None:
        field = SchemaField(
            field_type="string",
            description="Primary category",
        )
        desc = field.to_schema_description("category")
        assert "category" in desc
        assert "string" in desc
        assert "Primary category" in desc

    def test_schema_description_with_enum(self) -> None:
        field = SchemaField(
            field_type="string",
            description="Quality level",
            enum=["high", "low"],
        )
        desc = field.to_schema_description("quality")
        assert "high" in desc
        assert "low" in desc

    def test_to_json_schema_string(self) -> None:
        field = SchemaField(
            field_type="string",
            description="A string field",
            enum=["a", "b"],
        )
        schema = field.to_json_schema()
        assert schema["type"] == "string"
        assert schema["enum"] == ["a", "b"]
        assert schema["description"] == "A string field"

    def test_to_json_schema_list_string(self) -> None:
        field = SchemaField(
            field_type="list[string]",
            description="Tags",
        )
        schema = field.to_json_schema()
        assert schema["type"] == "array"
        assert schema["items"] == {"type": "string"}

    def test_to_json_schema_int(self) -> None:
        field = SchemaField(field_type="int", description="Count")
        schema = field.to_json_schema()
        assert schema["type"] == "integer"

    def test_to_json_schema_bool(self) -> None:
        field = SchemaField(field_type="bool", description="Flag")
        schema = field.to_json_schema()
        assert schema["type"] == "boolean"

    def test_to_json_schema_float(self) -> None:
        field = SchemaField(field_type="float", description="Score")
        schema = field.to_json_schema()
        assert schema["type"] == "number"


class TestClassificationPrompt:
    """Tests for ClassificationPrompt."""

    def test_default_schema_has_expected_fields(self) -> None:
        prompt = ClassificationPrompt()
        assert "category" in prompt.schema
        assert "quality" in prompt.schema
        assert "tags" in prompt.schema

    def test_default_name(self) -> None:
        prompt = ClassificationPrompt()
        assert prompt.name == "classification"

    def test_build_schema_description(self) -> None:
        prompt = ClassificationPrompt()
        desc = prompt.build_schema_description()
        assert "category" in desc
        assert "quality" in desc
        assert "tags" in desc

    def test_build_user_prompt(self) -> None:
        prompt = ClassificationPrompt()
        user_prompt = prompt.build_user_prompt()
        # Schema description should be filled in (no remaining placeholder)
        assert "{schema_description}" not in user_prompt
        assert "category" in user_prompt

    def test_build_json_schema(self) -> None:
        prompt = ClassificationPrompt()
        schema = prompt.build_json_schema()
        assert schema["type"] == "object"
        assert "category" in schema["properties"]
        assert "quality" in schema["properties"]
        assert "tags" in schema["properties"]
        # All default fields have no default, so all required
        assert "category" in schema["required"]

    def test_to_dict_includes_name(self) -> None:
        prompt = ClassificationPrompt()
        data = prompt.to_dict()
        assert data["name"] == "classification"

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        original = ClassificationPrompt()
        data = original.to_dict()
        # Verify it's JSON-serializable
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = BasePrompt.from_dict(restored_data)

        assert restored.system_prompt == original.system_prompt
        assert restored.user_prompt == original.user_prompt
        assert restored.name == original.name
        assert set(restored.schema.keys()) == set(original.schema.keys())
        for name in original.schema:
            assert restored.schema[name].field_type == original.schema[name].field_type
            assert restored.schema[name].description == original.schema[name].description
            assert restored.schema[name].enum == original.schema[name].enum

    def test_from_dict_backward_compat_no_name(self) -> None:
        """Old data without name should fall back to BasePrompt."""
        data = {
            "system_prompt": "test",
            "user_prompt": "test {schema_description}",
            "schema": {
                "foo": {"field_type": "string", "description": "a field"},
            },
        }
        prompt = BasePrompt.from_dict(data)
        assert isinstance(prompt, BasePrompt)

    def test_custom_schema(self) -> None:
        custom = ClassificationPrompt(
            schema={
                "is_nsfw": SchemaField(
                    field_type="bool",
                    description="Whether the image is NSFW",
                ),
                "confidence": SchemaField(
                    field_type="float",
                    description="Confidence score",
                ),
            }
        )
        schema = custom.build_json_schema()
        assert "is_nsfw" in schema["properties"]
        assert schema["properties"]["is_nsfw"]["type"] == "boolean"
        assert schema["properties"]["confidence"]["type"] == "number"


class TestPromptRegistry:
    """Tests for the prompt registry mechanism."""

    def test_classification_registered(self) -> None:
        assert "classification" in PROMPT_REGISTRY

    def test_registry_maps_to_correct_class(self) -> None:
        assert PROMPT_REGISTRY["classification"] is ClassificationPrompt

    def test_from_dict_dispatches_to_subclass(self) -> None:
        data = ClassificationPrompt().to_dict()
        restored = BasePrompt.from_dict(data)
        assert isinstance(restored, ClassificationPrompt)
        assert restored.name == "classification"

    def test_from_dict_unknown_type_falls_back_to_base(self) -> None:
        data = {
            "name": "unknown_custom_type",
            "system_prompt": "custom",
            "user_prompt": "custom {schema_description}",
            "schema": {
                "x": {"field_type": "bool", "description": "test"},
            },
        }
        prompt = BasePrompt.from_dict(data)
        assert isinstance(prompt, BasePrompt)
        assert prompt.name == "unknown_custom_type"
