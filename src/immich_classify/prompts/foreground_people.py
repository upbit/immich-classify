"""Auto-generated prompt configuration: Foreground people detection.

Load with: immich-classify classify --prompt-config foreground_people.py
"""

from dataclasses import dataclass, field

from immich_classify.prompt_base import BasePrompt, SchemaField, register_prompt


@register_prompt
@dataclass
class ForegroundPeoplePrompt(BasePrompt):
    """Count foreground people while ignoring background figures."""

    prompt_type: str = "foreground_people"

    system_prompt: str = (
        "You are an expert visual analyst specializing in crowd density and "
        "foreground detection. Your task is to count only the people who are "
        "clearly visible in the foreground (primary subjects) and explicitly "
        "ignore any people present in the background, crowd, or distant areas. "
        "Focus on individuals who are the main focus of the image."
    )

    user_prompt: str = (
        "Please analyze the provided image and return the count of foreground "
        "people, excluding any background figures. Here is the schema "
        "description to guide your output: {schema_description}\n"
    )

    schema: dict[str, SchemaField] = field(default_factory=lambda: {
        "foreground_count": SchemaField(
            field_type="int",
            description="The total number of people identified as primary subjects in the foreground.",
            default=0,
        ),
        "background_ignored": SchemaField(
            field_type="bool",
            description="A boolean flag indicating whether background figures were successfully excluded from the count.",
            default=True,
        ),
        "detection_confidence": SchemaField(
            field_type="float",
            description="A confidence score (0.0 to 1.0) representing the certainty of the foreground detection.",
            default=0.9,
        ),
        "foreground_details": SchemaField(
            field_type="list[string]",
            description="A list of brief descriptions for each detected foreground person (e.g., 'man in blue shirt', 'woman holding bag').",
        ),
        "background_note": SchemaField(
            field_type="string",
            description="A short note describing the nature of the background figures that were ignored (e.g., 'crowd in distance', 'people behind fence').",
            default="None",
        ),
        "occlusion_status": SchemaField(
            field_type="string",
            description="The occlusion status of the foreground subjects ('fully_visible', 'partially_occluded', 'heavily_occluded').",
            enum=["fully_visible", "partially_occluded", "heavily_occluded"],
            default="fully_visible",
        ),
    })


# Module-level instance for backward compatibility with --prompt-config loading
prompt = ForegroundPeoplePrompt()
