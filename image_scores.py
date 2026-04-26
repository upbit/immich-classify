"""Auto-generated prompt configuration.

Load with: immich-classify classify --prompt-config image_scores.py
"""

from dataclasses import dataclass, field

from immich_classify.prompt_base import BasePrompt, SchemaField, register_prompt


@register_prompt
@dataclass
class ImageScoresPrompt(BasePrompt):
    """Auto-generated prompt: image_scores."""

    name: str = 'image_scores'

    system_prompt: str = (
        '你是一位专业的图像美学评估专家，专注于分析图像构图质量。请根据构图规则、视觉平衡、美学原则等维度进行客观评分，评分范围1-'
        '10分（10分为最优构图）。'
    )

    user_prompt: str = (
        '请分析以下图像的构图美学质量：{schema_description}\n'
    )

    schema: dict[str, SchemaField] = field(default_factory=lambda: {
        'composition_type': SchemaField(
            field_type='string',
            description='图像使用的构图方式（如三分法/对称构图/对角线构图等）',
            enum=['三分法', '对称构图', '对角线构图', '框架式构图', '中心构图', '引导线构图', '留白构图'],
        ),
        'aesthetic_score': SchemaField(
            field_type='int',
            description='美学质量评分（1-10分）',
        ),
        'justification': SchemaField(
            field_type='list[string]',
            description='评分依据的构图分析要点列表',
        ),
        'is_balanced': SchemaField(
            field_type='bool',
            description='画面元素分布是否达到视觉平衡',
        ),
        'has_guiding_lines': SchemaField(
            field_type='bool',
            description='是否存在有效的视觉引导线',
        ),
        'negative_space_ratio': SchemaField(
            field_type='float',
            description='留白区域占比（0.0-1.0）',
        ),
    })


# Module-level instance for --prompt-config loading
prompt = ImageScoresPrompt()
