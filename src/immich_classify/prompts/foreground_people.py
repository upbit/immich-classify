"""Auto-generated prompt configuration: Foreground people detection.

Load with: immich-classify classify --prompt-config foreground_people.py
"""

from dataclasses import dataclass, field

from immich_classify.prompt_base import BasePrompt, SchemaField, register_prompt


@register_prompt
@dataclass
class ForegroundPeoplePrompt(BasePrompt):
    """Count foreground people while ignoring background figures."""

    name: str = "foreground_people"

    system_prompt: str = (
        '你是一个高精度的视觉分析模型，专门负责对图像进行人物计数。你的核心任务是检测并统计输入图像中“真实且处于主视角内”的人数。你必须严格遵守一套复杂'
        '的排除规则，包括忽略背景微小目标、不完整遮挡的人体、屏幕截图中的非核心主体、以及所有非现实的图像元素（如海报、模型、影子等）。你的输出必须包含一'
        '个详细的分析过程和一个最终的整数计数。'
    )

    user_prompt: str = (
        '请根据以下规则对图像进行人物计数，并严格按照指定的JSON格式输出结果：\n'
        '\n'
        '核心任务：检测并统计输入图像中“真实且处于主视角内”的人数。\n'
        '\n'
        '排除规则摘要：\n'
        '1. 忽略背景微小/失焦目标。\n'
        '2. 忽略不完整或重度遮挡的人体（无法辨认完整主体）。\n'
        '3. 屏幕截图/嵌套图像限制：仅统计屏幕内容中的“核心主体人物”，忽略UI头像、缩略图人物、现实环境中的2D/3D模型。\n'
        '4. 忽略镜像、影子和非现实元素。\n'
        '\n'
        '{schema_description}\n'
    )

    schema: dict[str, SchemaField] = field(default_factory=lambda: {
        'analysis': SchemaField(
            field_type='string',
            description='简要的分析过程，说明哪些被计入，哪些触发了排除规则被忽略. 触发模型思维链分析过程。',
        ),
        'count': SchemaField(
            field_type='int',
            description='最终统计的、符合所有规则的人数。',
        ),
    })


# Module-level instance for backward compatibility with --prompt-config loading
prompt = ForegroundPeoplePrompt()
