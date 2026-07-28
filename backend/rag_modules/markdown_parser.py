"""
Markdown 菜谱解析器

解析用户上传的 Markdown 格式菜谱文件，提取菜名、难度、食材列表、用量、步骤、附加内容等结构化信息。

支持的 Markdown 格式：
    # 菜名（可选"的做法"后缀）
    ![图片](path)                # 可选
    预估烹饪难度：★★★             # 可选
    ## 必备原料和工具
    - 食材A
    - 食材B（备注）
    ## 计算
    每份：
    - 食材A：500 克
    - 食材B：3 片
    ## 操作
    - 步骤1
    - 步骤2
    ## 附加内容
    - 备注1
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParsedIngredient:
    """食材条目（仅名称，无用量）"""
    name: str


@dataclass
class ParsedIngredientAmount:
    """食材用量条目"""
    name: str
    amount: Optional[float] = None
    unit: Optional[str] = None


@dataclass
class ParsedRecipe:
    """解析后的菜谱结构化数据"""
    name: str
    difficulty: int = 0
    ingredients: List[ParsedIngredient] = field(default_factory=list)
    ingredient_amounts: List[ParsedIngredientAmount] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    additional_notes: List[str] = field(default_factory=list)
    image_path: Optional[str] = None


def _clean_ingredient_name(text: str) -> str:
    """清洗食材名：去掉括号备注，去除首尾空白。"""
    # 去掉中英文括号及其内容
    text = re.sub(r'[（(].*?[)）]', '', text)
    return text.strip()


def _extract_section(text: str, heading: str) -> str:
    """提取 Markdown 中指定二级标题（## xxx）到下一个二级标题之间的内容。

    Args:
        text: 完整 Markdown 文本
        heading: 二级标题文字（不含 ## 前缀）

    Returns:
        该章节的内容文本（不含标题行），如果章节不存在返回空字符串。
    """
    # 匹配 ## heading ... 直到下一个 ## 或文档结尾
    pattern = rf'^##\s+{re.escape(heading)}\s*$([\s\S]*?)(?=^##\s|\Z)'
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else ''


def _extract_bullets(section_text: str) -> List[str]:
    """从章节文本中提取所有以 - 或 * 开头的列表项。"""
    items = re.findall(r'^[-*]\s+(.+)$', section_text, re.MULTILINE)
    return [item.strip() for item in items if item.strip()]


def parse_markdown_recipe(text: str) -> ParsedRecipe:
    """解析 Markdown 菜谱文本，返回结构化的 ParsedRecipe。

    Args:
        text: Markdown 格式的菜谱文本

    Returns:
        ParsedRecipe 结构化数据

    Raises:
        ValueError: 当缺少必要字段（标题、操作步骤）时抛出
    """
    # 归一化：统一换行符，去除 BOM
    text = text.replace('\r\n', '\n').replace('\r', '\n').lstrip('﻿')

    # 1. 菜名（第一个 # 标题）
    title_match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
    if not title_match:
        raise ValueError("Markdown文件缺少菜谱标题（# 标题）")
    name = title_match.group(1).strip()
    # 去掉尾部"的做法"
    name = re.sub(r'的做法\s*$', '', name).strip()
    if not name:
        raise ValueError("Markdown文件菜谱标题为空")

    # 2. 难度（数 ★ 个数）
    difficulty = 0
    diff_match = re.search(r'预估烹饪难度[：:]\s*(.+)', text)
    if diff_match:
        difficulty = diff_match.group(1).count('★')

    # 3. 图片路径
    image_path = None
    img_match = re.search(r'!\[([^\]]*)\]\(([^)]+)\)', text)
    if img_match:
        image_path = img_match.group(2).strip()

    # 4. 必备原料和工具
    ingredients: List[ParsedIngredient] = []
    ing_section = _extract_section(text, '必备原料和工具')
    for item in _extract_bullets(ing_section):
        ing_name = _clean_ingredient_name(item)
        if ing_name:
            ingredients.append(ParsedIngredient(name=ing_name))

    # 5. 计算（用量）
    ingredient_amounts: List[ParsedIngredientAmount] = []
    calc_section = _extract_section(text, '计算')
    for item in _extract_bullets(calc_section):
        # 尝试匹配 "名称：数字 单位" 格式
        amt_match = re.match(r'^(.+?)[：:]\s*([\d.]+)\s*(.*)$', item)
        if amt_match:
            ing_name = _clean_ingredient_name(amt_match.group(1))
            amount = float(amt_match.group(2))
            unit = amt_match.group(3).strip() or None
            ingredient_amounts.append(ParsedIngredientAmount(name=ing_name, amount=amount, unit=unit))
        else:
            # 无数字的情况（如"适量"），尝试提取名称和剩余文本作为 unit
            colon_match = re.match(r'^(.+?)[：:]\s*(.*)$', item)
            if colon_match:
                ing_name = _clean_ingredient_name(colon_match.group(1))
                unit = colon_match.group(2).strip() or None
                ingredient_amounts.append(ParsedIngredientAmount(name=ing_name, amount=None, unit=unit))

    # 6. 操作步骤
    steps_section = _extract_section(text, '操作')
    steps = _extract_bullets(steps_section)
    if not steps:
        raise ValueError("Markdown文件缺少操作步骤（## 操作）")

    # 7. 附加内容
    notes_section = _extract_section(text, '附加内容')
    additional_notes = _extract_bullets(notes_section)

    return ParsedRecipe(
        name=name,
        difficulty=difficulty,
        ingredients=ingredients,
        ingredient_amounts=ingredient_amounts,
        steps=steps,
        additional_notes=additional_notes,
        image_path=image_path,
    )
