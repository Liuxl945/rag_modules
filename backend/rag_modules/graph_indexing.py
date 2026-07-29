"""
图索引模块 — 基于 LightRAG 的键值对 (K,V) 检索机制

实现思路：
- 将图数据库中的实体（Recipe / Ingredient / CookingStep）以「名称」作为唯一索引键，
  将其属性、分类等信息打包为 value_content，存入 entity_kv_store。
- 将图中的关系（REQUIRES / HAS_STEP / BELONGS_TO_CATEGORY）以「(源, 目标, 类型)」
  为签名建索引键，通过 LLM 可进一步生成全局主题关键词做增强检索。
- 支持「按索引键反查实体 / 关系」，供 hybrid_retrieval 模块在检索阶段调用。
"""

import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class EntityKeyValue:
    """实体键值对：以「名称」为唯一索引键，value 存储详细描述。

    Attributes:
        entity_name: 实体的显示名称（如菜品名「宫保鸡丁」、食材名「鸡肉」）
        index_keys: 用于检索的索引键列表（通常至少包含 entity_name）
        value_content: 该实体的完整信息描述文本，供 LLM 生成答案时引用
        entity_type: 实体类别：Recipe / Ingredient / CookingStep
        metadata: 原始属性字典，保留 nodeId、category 等结构化字段
    """

    entity_name: str
    index_keys: List[str]                    # 索引键列表（供检索匹配）
    value_content: str                       # 详细描述内容（供 LLM 读取）
    entity_type: str                         # 实体类型 (Recipe, Ingredient, CookingStep)
    metadata: Dict[str, Any]


@dataclass
class RelationKeyValue:
    """关系键值对：以「源实体 → 关系类型 → 目标实体」为签名，value 存储关系描述。

    Attributes:
        relation_id: 关系的唯一标识（格式: rel_{index}_{source_id}_{target_id}）
        index_keys: 多个索引键（包含关系类型 + LLM 生成的全局主题关键词）
        value_content: 该关系的完整描述文本
        relation_type: 关系类型（REQUIRES / HAS_STEP / BELONGS_TO_CATEGORY）
        source_entity: 源实体在 entity_kv_store 中的 key（即 nodeId）
        target_entity: 目标实体在 entity_kv_store 中的 key（即 nodeId）
        metadata: 源/目标实体的名称快照，以及来源标记
    """

    relation_id: str
    index_keys: List[str]  # 多个索引键（可包含全局主题）
    value_content: str     # 关系描述内容
    relation_type: str     # 关系类型（REQUIRES / HAS_STEP / BELONGS_TO_CATEGORY）
    source_entity: str     # 源实体 ID（entity_kv_store 的 key）
    target_entity: str     # 目标实体 ID（entity_kv_store 的 key）
    metadata: Dict[str, Any]


class GraphIndexingModule:
    """图索引模块

    负责将 Neo4j 中存储的菜谱知识图谱，转换为适合「键值对检索」的结构。

    核心流程：
        1. create_entity_key_values   → 将 Recipe / Ingredient / CookingStep 存入 entity_kv_store
        2. create_relation_key_values → 将关系三元组 (source, type, target) 存入 relation_kv_store
        3. deduplicate_entities_and_relations → 去重冗余实体和关系

    Public API:
        - get_entities_by_key(key)       → 按索引键查实体列表
        - get_relations_by_key(key)      → 按索引键查关系列表
        - get_statistics()               → 获取统计信息（供展示用）

    Note:
        index_keys 中的「全局主题关键词」来自 _generate_relation_index_keys，
        根据关系类型预生成（如 REQUIRES 对应「食材搭配」，BELONGS_TO_CATEGORY 对应「菜品分类」），
        可选扩展 LLM 增强。这使得主题级检索无需遍历整张图，只需查哈希表即可命中相关实体/关系。
    """

    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client

        # 键值对存储：entity_id → EntityKeyValue / relation_id → RelationKeyValue
        self.entity_kv_store: Dict[str, EntityKeyValue] = {}
        self.relation_kv_store: Dict[str, RelationKeyValue] = {}

        # 反向索引：索引键 → 对应的实体/关系 ID 列表（默认值 defaultdict）
        self.key_to_entities: Dict[str, List[str]] = defaultdict(list)
        self.key_to_relations: Dict[str, List[str]] = defaultdict(list)

    def create_entity_key_values(self, recipes: List[Any], ingredients: List[Any],
                                cooking_steps: List[Any]) -> Dict[str, EntityKeyValue]:
        """为图中所有实体创建键值对。

        每个实体的「名称」作为唯一索引键（index_keys = [entity_name]），
        将其属性、分类等结构化信息拼成自然语言描述存入 value_content。

        Args:
            recipes: 菜谱节点列表（GraphNode）
            ingredients: 食材节点列表（GraphNode）
            cooking_steps: 烹饪步骤节点列表（GraphNode）

        Returns:
            entity_kv_store 字典（entity_id → EntityKeyValue）
        """
        logger.info("开始创建实体键值对...")

        # 1. 处理菜谱实体 (Recipe): 将菜品的属性（名称、描述、分类、难度等）
        #    拼接成自然语言文本存入 value_content，供 LLM 生成回答时直接使用。
        for recipe in recipes:
            entity_id = recipe.node_id
            entity_name = recipe.name or f"菜谱_{entity_id}"

            # 构建详细内容
            content_parts = [f"菜品名称: {entity_name}"]

            if hasattr(recipe, 'properties'):
                props = recipe.properties
                if props.get('description'):
                    content_parts.append(f"描述: {props['description']}")
                if props.get('category'):
                    content_parts.append(f"分类: {props['category']}")
                if props.get('cuisineType'):
                    content_parts.append(f"菜系: {props['cuisineType']}")
                if props.get('difficulty'):
                    content_parts.append(f"难度: {props['difficulty']}")
                if props.get('cookingTime'):
                    content_parts.append(f"制作时间: {props['cookingTime']}")

            # 创建键值对
            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],  # 使用名称作为唯一索引键
                value_content='\n'.join(content_parts),
                entity_type="Recipe",
                metadata={
                    "node_id": entity_id,
                    "properties": getattr(recipe, 'properties', {})
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        # 2. 处理食材实体 (Ingredient): 存储类别、营养信息、储存方式等属性。
        for ingredient in ingredients:
            entity_id = ingredient.node_id
            entity_name = ingredient.name or f"食材_{entity_id}"

            content_parts = [f"食材名称: {entity_name}"]

            if hasattr(ingredient, 'properties'):
                props = ingredient.properties
                if props.get('category'):
                    content_parts.append(f"类别: {props['category']}")
                if props.get('nutrition'):
                    content_parts.append(f"营养信息: {props['nutrition']}")
                if props.get('storage'):
                    content_parts.append(f"储存方式: {props['storage']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="Ingredient",
                metadata={
                    "node_id": entity_id,
                    "properties": getattr(ingredient, 'properties', {})
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        # 3. 处理烹饪步骤实体 (CookingStep): 存储步骤描述、顺序、技巧和时长。
        for step in cooking_steps:
            entity_id = step.node_id
            entity_name = f"步骤_{entity_id}"

            content_parts = [f"烹饪步骤: {entity_name}"]

            if hasattr(step, 'properties'):
                props = step.properties
                if props.get('description'):
                    content_parts.append(f"步骤描述: {props['description']}")
                if props.get('order'):
                    content_parts.append(f"步骤顺序: {props['order']}")
                if props.get('technique'):
                    content_parts.append(f"技巧: {props['technique']}")
                if props.get('time'):
                    content_parts.append(f"时间: {props['time']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="CookingStep",
                metadata={
                    "node_id": entity_id,
                    "properties": getattr(step, 'properties', {})
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        logger.info(f"实体键值对创建完成，共 {len(self.entity_kv_store)} 个实体")
        return self.entity_kv_store

    def create_relation_key_values(self, relationships: List[Tuple[str, str, str]]) -> Dict[str, RelationKeyValue]:
        """为图中所有关系创建键值对。

        每条关系 (source_entity, relation_type, target_entity) 生成一个
        RelationKeyValue，其中 index_keys 包含关系类型 + LLM 生成的全局主题关键词。

        Args:
            relationships: 三元组列表，每个元素为 (source_id, relation_type, target_id)

        Returns:
            relation_kv_store 字典（relation_id → RelationKeyValue）
        """
        logger.info("开始创建关系键值对...")

        for i, (source_id, relation_type, target_id) in enumerate(relationships):
            relation_id = f"rel_{i}_{source_id}_{target_id}"

            # 获取源实体和目标实体信息，若任一方不存在则跳过。
            source_entity = self.entity_kv_store.get(source_id)
            target_entity = self.entity_kv_store.get(target_id)

            if not source_entity or not target_entity:
                continue

            # 构建关系描述：包含关系类型、源/目标实体的名称和类别。
            content_parts = [
                f"关系类型: {relation_type}",
                f"源实体: {source_entity.entity_name} ({source_entity.entity_type})",
                f"目标实体: {target_entity.entity_name} ({target_entity.entity_type})"
            ]

            # 生成多个索引键（包含全局主题）：同一关系可被多种关键词命中。
            index_keys = self._generate_relation_index_keys(
                source_entity, target_entity, relation_type
            )

            # 创建关系键值对
            relation_kv = RelationKeyValue(
                relation_id=relation_id,
                index_keys=index_keys,
                value_content='\n'.join(content_parts),
                relation_type=relation_type,
                source_entity=source_id,
                target_entity=target_id,
                metadata={
                    "source_name": source_entity.entity_name,
                    "target_name": target_entity.entity_name,
                    "created_from_graph": True
                }
            )

            self.relation_kv_store[relation_id] = relation_kv

            # 为每个索引键建立映射，使「按关键词反查关系」成为 O(1) 操作。
            for key in index_keys:
                self.key_to_relations[key].append(relation_id)

        logger.info(f"关系键值对创建完成，共 {len(self.relation_kv_store)} 个关系")
        return self.relation_kv_store

    def _generate_relation_index_keys(self, source_entity: EntityKeyValue,
                                    target_entity: EntityKeyValue,
                                    relation_type: str) -> List[str]:
        """为关系生成多个索引键，包含全局主题关键词。

        根据预定义的关系类型映射表，为不同关系生成对应的领域关键词，
        使同一关系可被多种语义相近的查询命中。

        Args:
            source_entity: 源实体 (EntityKeyValue)
            target_entity: 目标实体 (EntityKeyValue)
            relation_type: 关系类型字符串

        Returns:
            去重后的索引键列表（至少包含 relation_type）
        """
        keys = [relation_type]  # 基础关系类型键

        # 根据关系类型和实体类型生成主题键
        if relation_type == "REQUIRES":
            # 菜谱-食材关系的主题键：「这道菜需要什么？」相关问题可命中。
            keys.extend([
                "食材搭配",
                "烹饪原料",
                f"{source_entity.entity_name}_食材",
                target_entity.entity_name
            ])
        elif relation_type == "HAS_STEP":
            # 菜谱-步骤关系的主题键：「这道菜怎么做？」相关问题可命中。
            keys.extend([
                "制作步骤",
                "烹饪过程",
                f"{source_entity.entity_name}_步骤",
                "制作方法"
            ])
        elif relation_type == "BELONGS_TO_CATEGORY":
            # 分类关系的主题键：「川菜里有什么？」相关问题可命中。
            keys.extend([
                "菜品分类",
                "美食类别",
                target_entity.entity_name
            ])

        # 使用 LLM 增强关系索引键（可选，需配置 enable_llm_relation_keys=True）
        if getattr(self.config, 'enable_llm_relation_keys', False):
            enhanced_keys = self._llm_enhance_relation_keys(source_entity, target_entity, relation_type)
            keys.extend(enhanced_keys)

        # 去重并返回
        return list(set(keys))

    def _llm_enhance_relation_keys(self, source_entity: EntityKeyValue,
                                 target_entity: EntityKeyValue,
                                 relation_type: str) -> List[str]:
        """使用 LLM 增强关系索引键，生成全局主题关键词。

        通过调用 LLM 的语义理解能力，为每条关系生成候选「主题关键词」，
        扩展检索时的匹配覆盖面。

        Args:
            source_entity: 源实体 (EntityKeyValue)
            target_entity: 目标实体 (EntityKeyValue)
            relation_type: 关系类型字符串

        Returns:
            LLM 生成的主题关键词列表（失败时返回空列表）
        """
        prompt = f"""
        分析以下实体关系，生成相关的主题关键词：

        源实体: {source_entity.entity_name} ({source_entity.entity_type})
        目标实体: {target_entity.entity_name} ({target_entity.entity_type})
        关系类型: {relation_type}

        请生成3-5个相关的主题关键词，用于索引和检索。
        返回JSON格式：{ { "keywords": ["关键词1", "关键词2", "关键词3"] } }
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )

            result = json.loads(response.choices[0].message.content.strip())
            return result.get("keywords", [])

        except Exception as e:
            logger.error(f"LLM增强关系索引键失败: {e}")
            return []

    def deduplicate_entities_and_relations(self):
        """对实体和关系执行去重，合并冗余条目。

        去重策略：
          - 实体：基于 entity_name 合并，保留第一个实体的 value_content 并追加其他实体内容。
          - 关系：基于 (source, target, type) 签名合并，仅保留第一个。

        去重后重建 key_to_entities / key_to_relations 映射，确保一致性。
        """
        logger.info("开始去重实体和关系...")

        # 实体去重：基于名称
        name_to_entities = defaultdict(list)
        for entity_id, entity_kv in self.entity_kv_store.items():
            name_to_entities[entity_kv.entity_name].append(entity_id)

        # 合并重复实体：保留第一个，仅合并「主实体中尚未出现」的新信息。
        entities_to_remove = []
        for name, entity_ids in name_to_entities.items():
            if len(entity_ids) > 1:
                # 保留第一个，合并其他的内容
                primary_id = entity_ids[0]
                primary_entity = self.entity_kv_store[primary_id]

                # 收集主实体已有的内容行，用于跳过完全相同的重复信息。
                # 同名实体（如初始导入产生的多个「鸡蛋」节点）的 value_content 往往逐字相同，
                # 旧实现会把每个副本整段追加，导致出现几十个重复的
                # 「补充信息: 食材名称: 鸡蛋 / 类别: 蛋白质」块，污染检索上下文。
                #
                # 注意：primary_entity.value_content 本身可能已经包含「补充信息:」前缀行
                # （如嵌套合并场景），splitlines 后这些前缀行也加入 seen_lines，
                # 但新副本的正文行（"食材名称: 鸡蛋"）已在 seen_lines 中，不会重复追加。
                seen_lines = set()
                for line in primary_entity.value_content.splitlines():
                    stripped = line.strip()
                    if stripped:
                        seen_lines.add(stripped)

                for entity_id in entity_ids[1:]:
                    duplicate_entity = self.entity_kv_store[entity_id]
                    # 仅保留主实体中尚未出现的非空行（标题行「食材名称: ...」会因已存在被过滤），
                    # 只有真正带来新信息（如不同的营养/储存字段）时才追加，避免噪声。
                    new_lines = []
                    for line in duplicate_entity.value_content.splitlines():
                        stripped = line.strip()
                        # 跳过空行和已存在的行；也跳过"补充信息:"标记行本身（它不是有效信息）
                        if stripped and stripped not in seen_lines and not stripped.startswith("补充信息:"):
                            new_lines.append(stripped)
                    if new_lines:
                        supplement = "\n".join(new_lines)
                        primary_entity.value_content += f"\n\n补充信息: {supplement}"
                        seen_lines.update(new_lines)
                    # 标记删除（即使无新信息也要移除冗余节点）
                    entities_to_remove.append(entity_id)

                # 安全上限：合并后若内容过长（>1000 字符），截断到首个完整块
                if len(primary_entity.value_content) > 1000:
                    logger.warning(
                        f"实体「{name}」去重后内容过长（{len(primary_entity.value_content)}字符），截断到1000字符"
                    )
                    truncated = primary_entity.value_content[:1000]
                    last_nl = truncated.rfind('\n')
                    if last_nl > 200:
                        truncated = truncated[:last_nl]
                    primary_entity.value_content = truncated + "…"

        # 删除重复实体
        for entity_id in entities_to_remove:
            del self.entity_kv_store[entity_id]

        # 关系去重：基于源-目标-类型三元组签名
        relation_signature_to_ids = defaultdict(list)
        for relation_id, relation_kv in self.relation_kv_store.items():
            signature = f"{relation_kv.source_entity}_{relation_kv.target_entity}_{relation_kv.relation_type}"
            relation_signature_to_ids[signature].append(relation_id)

        # 合并重复关系（保留第一个，删除其余）
        relations_to_remove = []
        for signature, relation_ids in relation_signature_to_ids.items():
            if len(relation_ids) > 1:
                # 保留第一个，删除其他
                for relation_id in relation_ids[1:]:
                    relations_to_remove.append(relation_id)

        # 删除重复关系
        for relation_id in relations_to_remove:
            del self.relation_kv_store[relation_id]

        # 重建索引映射，确保去重后的结果与反向索引一致。
        self._rebuild_key_mappings()

        logger.info(f"去重完成 - 删除了 {len(entities_to_remove)} 个重复实体，{len(relations_to_remove)} 个重复关系")

    def _rebuild_key_mappings(self):
        """重建键到实体/关系的映射。

        清空现有的 key_to_entities / key_to_relations，
        然后从 entity_kv_store / relation_kv_store 中重新构建反向索引。
        """
        self.key_to_entities.clear()
        self.key_to_relations.clear()

        # 重建实体映射：每个 index_key → 对应的 entity_id 列表
        for entity_id, entity_kv in self.entity_kv_store.items():
            for key in entity_kv.index_keys:
                self.key_to_entities[key].append(entity_id)

        # 重建关系映射：每个 index_key → 对应的 relation_id 列表
        for relation_id, relation_kv in self.relation_kv_store.items():
            for key in relation_kv.index_keys:
                self.key_to_relations[key].append(relation_id)

    def get_entities_by_key(self, key: str) -> List[EntityKeyValue]:
        """根据索引键获取匹配的实体列表。

        Args:
            key: 检索用的索引关键词（如菜品名「宫保鸡丁」）

        Returns:
            匹配到的 EntityKeyValue 列表（按 key 匹配到所有 entity_id，再查 store）
        """
        entity_ids = self.key_to_entities.get(key, [])
        return [self.entity_kv_store[eid] for eid in entity_ids if eid in self.entity_kv_store]

    def get_relations_by_key(self, key: str) -> List[RelationKeyValue]:
        """根据索引键获取匹配的关系列表。

        Args:
            key: 检索用的索引关键词（如「食材搭配」「川菜」等）

        Returns:
            匹配到的 RelationKeyValue 列表
        """
        relation_ids = self.key_to_relations.get(key, [])
        return [self.relation_kv_store[rid] for rid in relation_ids if rid in self.relation_kv_store]

    def get_statistics(self) -> Dict[str, Any]:
        """获取键值对存储统计信息。

        Returns:
            包含实体总数、关系总数、各类实体计数等信息的字典。
        """
        return {
            "total_entities": len(self.entity_kv_store),
            "total_relations": len(self.relation_kv_store),
            "total_entity_keys": sum(len(kv.index_keys) for kv in self.entity_kv_store.values()),
            "total_relation_keys": sum(len(kv.index_keys) for kv in self.relation_kv_store.values()),
            "entity_types": {
                "Recipe": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Recipe"]),
                "Ingredient": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Ingredient"]),
                "CookingStep": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "CookingStep"])
            }
        }