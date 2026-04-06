"""Built-in classification prompt.

Provides ``ClassificationPrompt`` — a general-purpose image classification
prompt with category, quality, and tags fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from immich_classify.prompt_base import BasePrompt, SchemaField, register_prompt


@register_prompt
@dataclass
class ClassificationPrompt(BasePrompt):
    """General-purpose image classification prompt.

    Fields:
        category: High-level image category (people, landscape, food, etc.)
        quality: Subjective quality rating (high, medium, low)
        tags: Descriptive tags for the image
    """

    name: str = "classification"

    system_prompt: str = (
        "You are a photo organizer. Classify the image into the given schema. "
        "Output ONLY valid JSON."
    )

    user_prompt: str = (
        "Classify this image according to the following schema:\n"
        "{schema_description}\n\n"
        "Output a JSON object with the specified fields."
    )

    schema: dict[str, SchemaField] = field(default_factory=lambda: {
        "category": SchemaField(
            field_type="string",
            description="High-level image category",
            enum=["people", "landscape", "food", "animal", "architecture",
                  "document", "screenshot", "other"],
        ),
        "quality": SchemaField(
            field_type="string",
            description="Subjective quality rating",
            enum=["high", "medium", "low"],
        ),
        "tags": SchemaField(
            field_type="list[string]",
            description="Descriptive tags for the image",
        ),
    })


# Module-level instance for backward compatibility with --prompt-config loading
prompt = ClassificationPrompt()
