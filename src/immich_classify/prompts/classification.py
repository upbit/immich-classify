"""General-purpose image classification prompt.

Classifies images by category, quality, and descriptive tags.
This is the default prompt used when no ``--prompt-config`` is specified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from immich_classify.prompt_base import BasePrompt, SchemaField, register_prompt


@register_prompt
@dataclass
class ClassificationPrompt(BasePrompt):
    """Default prompt for general image classification."""

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
