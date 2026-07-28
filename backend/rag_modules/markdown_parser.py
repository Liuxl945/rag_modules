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


def _parse_amount_line(item: str) -> Optional[ParsedIngredientAmount]:
    """解析用量行，支持多种格式。

    支持格式：
        - "鸡蛋：550 克"          （冒号分隔，空格分隔数字和单位）
        - "鸡蛋 400g（约 8 颗）"  （空格分隔，单位紧跟数字，括号备注自动剥离）
        - "老抽：2ml"             （冒号 + 单位无空格）
        - "盐：适量"              （无数字，全部作为 unit）

    Returns:
        ParsedIngredientAmount 或 None（行内容为空时）
    """
    item = item.strip()
    if not item:
        return None

    # 先尝试冒号格式："名称：数字 单位" 或 "名称：其他文本"
    colon_match = re.match(r'^(.+?)[：:]\s*(.*)$', item)
    if colon_match:
        raw_name = colon_match.group(1)
        rest = colon_match.group(2).strip()
        # 剥离括号备注
        rest_clean = re.sub(r'[（(].*?[)）]', '', rest).strip()
        # 范围格式："5-8 克" 或 "0.5-1g"
        range_match = re.match(r'^([\d.]+)\s*[-~–—]\s*[\d.]+\s*(.*)$', rest_clean)
        if range_match:
            try:
                amount = float(range_match.group(1))
            except ValueError:
                amount = None
            unit = range_match.group(2).strip() or None
        else:
            # 单值格式
            num_match = re.match(r'^([\d.]+)\s*(.*)$', rest_clean)
            if num_match:
                try:
                    amount = float(num_match.group(1))
                except ValueError:
                    amount = None
                unit = num_match.group(2).strip() or None
            else:
                amount = None
                unit = rest or None
        return ParsedIngredientAmount(
            name=_clean_ingredient_name(raw_name),
            amount=amount,
            unit=unit,
        )

    # 空格格式："名称 数字单位（备注）" 如 "鸡蛋 400g（约 8 颗）"、"香叶 0.5-1g"
    # 先剥离括号内容，再找名称和数字
    cleaned = re.sub(r'[（(].*?[)）]', '', item).strip()
    # 匹配范围格式 "数字-数字单位"（如 0.5-1g），取范围起始值作为 amount
    range_match = re.match(r'^(.+?)\s+([\d.]+)\s*[-~–—]\s*[\d.]+\s*([^\s]*)\s*$', cleaned)
    if range_match:
        raw_name = range_match.group(1)
        try:
            amount = float(range_match.group(2))
        except ValueError:
            amount = None
        unit = range_match.group(3).strip() or None
        return ParsedIngredientAmount(
            name=_clean_ingredient_name(raw_name),
            amount=amount,
            unit=unit,
        )
    # 匹配单值格式 "数字 + 可选空格 + 单位"
    space_match = re.match(r'^(.+?)\s+([\d.]+)\s*([^\s]*)\s*$', cleaned)
    if space_match:
        raw_name = space_match.group(1)
        try:
            amount = float(space_match.group(2))
        except ValueError:
            amount = None
        unit = space_match.group(3).strip() or None
        return ParsedIngredientAmount(
            name=_clean_ingredient_name(raw_name),
            amount=amount,
            unit=unit,
        )

    # 无数字无冒号：整行作为名称（兜底）
    return ParsedIngredientAmount(
        name=_clean_ingredient_name(item),
        amount=None,
        unit=None,
    )


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
        parsed_amount = _parse_amount_line(item)
        if parsed_amount:
            ingredient_amounts.append(parsed_amount)

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
