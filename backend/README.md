# 图 RAG 智能烹饪助手 -- 项目学习与转行指南 V2

> 本文档基于**当前最新代码**重新梳理，相对 V1 更新了行号、关系类型、模型配置，并新增了「父文档检索」「RRF 融合完善」「停用词表」「图索引预热」「组合策略 Round-robin」等内容。
> 目标：帮你系统理解这个 RAG 项目，从架构到面试，一步步把它变成你自己的项目。

---

## 第 1 步：先理解 RAG 到底在解决什么问题

RAG（Retrieval-Augmented Generation，检索增强生成）解决的是 LLM 的两个硬伤：
- **知识截止**：模型训练后发生的事它不知道
- **幻觉**：模型会一本正经地胡说

核心思路就一句话：**先检索相关资料，再让 LLM 基于资料回答**。

```
用户问题 -> 检索相关文档 -> 把文档塞进 prompt -> LLM 生成回答
```

最朴素的 RAG（baseline）长这样：
```
文档 -> 切块 -> 向量化 -> 存向量库 -> 查询时算相似度取 top-k -> 喂给 LLM
```

这个项目**不是朴素 RAG**，它在 baseline 之上做了 3 层升级，这正是它的含金量所在：

| 升级点 | 朴素 RAG | 本项目 |
|--------|----------|--------|
| 数据源 | 纯文本 | **Neo4j 知识图谱**（菜谱-食材-步骤有结构化关系） |
| 检索方式 | 单一向量检索 | **三路混合检索 + RRF 融合** |
| 推理能力 | 关键词/语义匹配 | **图多跳推理 + 子图提取** |
| 策略选择 | 固定流程 | **LLM 智能路由** |

**记住这张对比表**，面试时这就是你的"项目亮点"开场。

---

## 第 2 步：架构总览--6 个模块怎么协作

打开 `main.py`，看 `AdvancedGraphRAGSystem` 这个类（[main.py:60](code/C9/main.py#L60)），它就是整个系统的"指挥官"。系统由 6 个模块组成，分三类：

```
┌──────────────────── 数据层（离线构建一次）────────────────────┐
│  ① GraphDataPreparationModule   Neo4j -> Document 文档         │
│  ② MilvusIndexConstructionModule Document -> 向量索引           │
└──────────────────────────────────────────────────────────────┘
┌──────────────────── 检索层（每次查询都跑）────────────────────┐
│  ③ GraphIndexingModule          图键值索引（LightRAG 风格）    │
│  ④ HybridRetrievalModule        传统混合检索（三路 RRF + 父文档）│
│  ⑤ GraphRAGRetrieval            图 RAG 检索（多跳推理）        │
└──────────────────────────────────────────────────────────────┘
┌──────────────────── 决策/生成层 ─────────────────────────────┐
│  ⑥ IntelligentQueryRouter       智能路由（选哪个检索引擎）    │
│  ⑦ GenerationIntegrationModule  LLM 答案生成（流式 + 重试）    │
└──────────────────────────────────────────────────────────────┘
```

**依赖关系**（这点很重要，决定初始化顺序，看 `main.py` 的 `initialize_system` [main.py:100](code/C9/main.py#L100)）：
- `④` 依赖 `①②③`（需要数据、向量库、图索引）+ LLM 客户端
- `⑤` 只依赖 LLM 客户端（运行时自己连 Neo4j）
- `⑥` 依赖 `④⑤`（要同时持有两个检索引擎才能路由）

**关键洞察**：`③` GraphIndexingModule 不是独立模块，它被 `④` HybridRetrievalModule 内部持有（看 [hybrid_retrieval.py:107](code/C9/rag_modules/hybrid_retrieval.py#L107)）。所以实际是 6 个模块。LLM 客户端由 `⑦` 生成模块创建后，复用给 `④⑤⑥`，避免重复建连。

### 各模块文件对照表

| 模块 | 文件路径 | 核心职责 |
|------|----------|----------|
| ① 数据准备 | [graph_data_preparation.py](code/C9/rag_modules/graph_data_preparation.py) | Neo4j -> Document |
| ② 向量索引 | [milvus_index_construction.py](code/C9/rag_modules/milvus_index_construction.py) | Document -> Milvus 向量索引 |
| ③ 图键值索引 | [graph_indexing.py](code/C9/rag_modules/graph_indexing.py) | LightRAG 风格 KV 索引 |
| ④ 混合检索 | [hybrid_retrieval.py](code/C9/rag_modules/hybrid_retrieval.py) | 三路召回 + RRF 融合 + 父文档回填 |
| ⑤ 图 RAG | [graph_rag_retrieval.py](code/C9/rag_modules/graph_rag_retrieval.py) | 多跳遍历 + 子图推理 |
| ⑥ 智能路由 | [intelligent_query_router.py](code/C9/rag_modules/intelligent_query_router.py) | LLM 决策选引擎 + 组合策略 |
| ⑦ 答案生成 | [generation_integration.py](code/C9/rag_modules/generation_integration.py) | LLM 生成（流式 + 重试降级） |
| 编排层 | [main.py](code/C9/main.py) | 串联所有模块 |
| 配置 | [config.py](code/C9/config.py) | 统一配置管理 |

---

## 第 3 步：数据流--一条查询从输入到输出的完整旅程

这是面试必答题。以用户问 **"鸡肉配什么蔬菜好？"** 为例：

### 离线阶段（构建知识库，只跑一次）
```
Neo4j 图数据库
  │  load_graph_data()  ①加载 Recipe/Ingredient/CookingStep 节点
  │     （Cypher 过滤 nodeId >= '200000000'，并 JOIN Category 关系）
  ↓
GraphNode 列表（内存中的图节点）
  │  build_recipe_documents()  ②JOIN REQUIRES(食材)/CONTAINS_STEP(步骤)，拼成自然语言文档
  ↓
Document 列表（"# 宫保鸡丁\n## 所需食材\n1. 鸡肉..."）
  │  chunk_documents()  ③优先按 ## 二级标题分块，否则按 500 字滑动切（overlap 50）
  ↓
chunk 列表（每个 chunk 带 parent_id / chunk_index）
  │  build_vector_index()  ④BGE 向量化 -> Milvus 批量插入 -> 建 HNSW 索引 -> load 到内存
  ↓
Milvus 向量库（就绪）
  │  initialize()  ⑤BM25 建索引 + 图键值索引构建 + 父文档映射懒建
  ↓
检索器就绪
```

> 注意：即使从已存在的 Milvus 集合加载（路径 A），也必须重新跑 `load_graph_data`，
> 因为图键值索引在内存中构建，依赖 `data_module` 里的节点数据。看 [main.py:175](code/C9/main.py#L175) 的 `build_knowledge_base`。

### 在线阶段（每次查询都跑）
```
"鸡肉配什么蔬菜好？"
  │
  ↓ IntelligentQueryRouter.analyze_query()  ①LLM 四维分析查询特征
  │  -> 复杂度 0.6, 关系密集度 0.8, 推荐策略: graph_rag
  ↓ IntelligentQueryRouter.route_query()  ②根据策略分派
  │
  ├─[hybrid_traditional]-> HybridRetrieval.hybrid_search()
  │     ├─ extract_query_keywords (LLM 提取 实体级+主题级 双层关键词)
  │     ├─ dual_level_retrieval (图键值双层检索，未命中走 Neo4j fulltext 降级)
  │     ├─ vector_search_enhanced (Milvus + 一跳邻居扩展)
  │     ├─ bm25_search (jieba 分词 + 停用词过滤 + BM25)
  │     ├─ _rrf_merge (三路 RRF 融合, k=60) -> 候选池（rerank_candidate_k=20）
  │     ├─ [可选] rerank (BAAI/bge-reranker-v2-m3 cross-encoder 精排) -> Top-K
  │     └─ [可选] _attach_parent_documents (父文档回填前 N 条)
  │
  └─[graph_rag]-> GraphRAGRetrieval.graph_rag_search()
        ├─ understand_graph_query (LLM 转图查询计划 + 5 种 query_type)
        ├─ multi_hop_traversal (Cypher 变长路径 + 路径评分)
        └─ extract_knowledge_subgraph + graph_structure_reasoning (子图 + 推理链)
  │
  ↓ GenerationIntegrationModule.generate_adaptive_answer[_stream]()
  │  组装上下文 -> LLM 生成（流式带重试，失败降级非流式）
  ↓
最终回答
```

### 精确率提升：top_k 收敛 + Rerank 精排（V2.1）

两阶段检索（two-stage retrieval）的标准做法：三路 RRF 先融合出较大候选池（`rerank_candidate_k=20`），再用 cross-encoder 精排，最后只留 top_k=3 喂给 LLM。

- **为什么有效**：bi-encoder（向量检索）query/doc 各自编码，速度快但细粒度匹配弱；cross-encoder（重排器）把 (query, doc) 一起输入，捕捉更细的语义相关性，排在前面的结果质量显著更高。
- **模型**：`BAAI/bge-reranker-v2-m3`（中文 cross-encoder，~568MB）。
- **懒加载 + 降级**：首次查询才加载（启动零成本）；模型未缓存/加载失败时自动跳过重排，返回 RRF 原顺序（不影响主流程）。
- **首次下载**（仅需一次，之后离线可用）：
  ```bash
  # 默认从 HuggingFace 下载
  HF_HUB_OFFLINE=0 python scripts/download_reranker.py
  # 国内镜像（推荐）
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=0 python scripts/download_reranker.py
  ```
- **配置**：`config.py` 中 `enable_rerank`（默认 True）、`rerank_candidate_k`（候选池大小）、`top_k`（默认 3，V2 为 5）。

**建议**：拿张纸自己画一遍这个流程图，能默画出来就算掌握 60% 了。

---

## 第 4 步：深挖每个模块的核心创新点

这部分是面试深挖区，挑最值得讲的 4 个。

### 4.1 GraphIndexingModule -- LightRAG 风格的 KV 索引

文件：[graph_indexing.py](code/C9/rag_modules/graph_indexing.py)

**它在做什么**：把图数据库的实体和关系，转成"键值对"结构，让检索变成 O(1) 的哈希查找，而不是遍历图。

```
实体 KV:  "宫保鸡丁" -> EntityKeyValue(详情描述)
关系 KV:  "食材搭配" -> [所有 REQUIRES 关系列表]   ← 注意这个"主题键"是精髓
```

**精髓在 `_generate_relation_index_keys`**（[graph_indexing.py:277](code/C9/rag_modules/graph_indexing.py#L277)）：一条 `宫保鸡丁-REQUIRES-鸡肉` 的关系，除了用关系类型 `REQUIRES` 做键，还额外生成"食材搭配""烹饪原料"等**主题键**。这样用户问"有什么食材搭配推荐"时，不用语义匹配，直接查哈希表就能命中。

三种关系类型的主题键映射：
| 关系类型 | 生成的主题键 | 命中的问题 |
|----------|-------------|-----------|
| `REQUIRES` | 食材搭配 / 烹饪原料 / `{菜名}_食材` / 目标食材名 | "这道菜需要什么食材？" |
| `HAS_STEP` | 制作步骤 / 烹饪过程 / `{菜名}_步骤` / 制作方法 | "这道菜怎么做？" |
| `BELONGS_TO_CATEGORY` | 菜品分类 / 美食类别 / 目标分类名 | "川菜里有什么？" |

**为什么这么做**：图遍历慢且贵，把高频查询模式预计算成 KV，用空间换时间。

> ⚠️ **代码一致性提醒（面试可作"找茬"亮点）**：
> `graph_indexing.py` 在 `_generate_relation_index_keys` 里用的是 **`HAS_STEP`**，
> 但 `graph_data_preparation.py` 和 `graph_rag_retrieval.py` 的 Cypher/prompt 用的是 **`CONTAINS_STEP`**。
> 这意味着"步骤"关系的主题键可能匹配不上实际关系类型——是个潜在 bug，也是改造点（见第 6 步改造 5）。

**去重优化**（[graph_indexing.py:370](code/C9/rag_modules/graph_indexing.py#L370)）：`deduplicate_entities_and_relations` 会按 `entity_name` 合并重复实体（内容追加到主条目），按 `(source, target, type)` 三元组签名去重关系，然后 `_rebuild_key_mappings` 重建反向索引。

### 4.2 HybridRetrievalModule -- 三路 RRF 融合 + 父文档回填（最硬核）

文件：[hybrid_retrieval.py](code/C9/rag_modules/hybrid_retrieval.py)

**三路召回为什么都要**：
- **BM25**：精确关键词匹配（用户说"鸡胸肉"，就该命中含"鸡胸肉"的文档）
- **向量检索**：语义相似（用户说"减肥肉"，能命中"低脂高蛋白"的鸡胸肉文档）
- **图键值双层**：结构化关系（实体级精确 + 主题级抽象）
- 单路都有盲区，三路互补

#### 4.2.1 双层关键词提取（LLM，[hybrid_retrieval.py:237](code/C9/rag_modules/hybrid_retrieval.py#L237)）
`extract_query_keywords` 把查询拆成两层：
- `entity_keywords`：具体实体（鸡胸肉、西兰花、平底锅）
- `topic_keywords`：抽象主题（减肥、川菜、低热量）

分别喂给 `entity_level_retrieval`（精确匹配，0.9 分）和 `topic_level_retrieval`（关系级 0.95 / 分类级 0.85 分）。

#### 4.2.2 Neo4j 降级补充检索
图键值索引没召回够时，有两层降级：
- `_neo4j_entity_level_search`（[hybrid_retrieval.py:361](code/C9/rag_modules/hybrid_retrieval.py#L361)）：调 `db.index.fulltext.queryNodes` 全文索引，得分 ×0.7
- `_neo4j_topic_level_search`（[hybrid_retrieval.py:516](code/C9/rag_modules/hybrid_retrieval.py#L516)）：按 `category/cuisineType/tags CONTAINS keyword` 匹配，固定 0.75 分

> 这就是"多层兜底"工程思维：精确匹配 -> 图键值 -> Neo4j fulltext -> Cypher 模糊，逐级降级保召回。

#### 4.2.3 BM25 + 中文停用词（V2 新增）
`_CHINESE_STOPWORDS`（[hybrid_retrieval.py:48](code/C9/rag_modules/hybrid_retrieval.py#L48)）手动维护了一份烹饪场景停用词表（助词/疑问词/人称代词/语气词），不依赖第三方停用词包。`_tokenize_chinese`（[hybrid_retrieval.py:149](code/C9/rag_modules/hybrid_retrieval.py#L149)）用 jieba 分词后过滤停用词、空白、单字符，降低 BM25 噪声。BM25 分数 ≤0 的文档直接丢弃。

#### 4.2.4 RRF 融合公式（[hybrid_retrieval.py:797](code/C9/rag_modules/hybrid_retrieval.py#L797)，必背）
```
score(d) = Σ_i 1 / (k + best_rank_i(d))    # k=60
```
- 不关心各路原始分数（BM25 分数和向量分数根本不在一个量纲），**只看排名**
- 一篇文档在多路都靠前 -> 分数累加 -> 排名上升
- 这解决了"不同检索器分数不可比"的经典问题

**V2 完善的几个细节**（面试官最爱问）：
1. **同 source 内去重**：一道菜的多个 chunk 共享同一个 `node_id`，RRF 只取该 source 内**最佳排名**算一次分（`best_rank_per_source`），避免重复加分；命中 chunk 次数另存到 `rrf_chunk_hits` 供分析。
2. **canonical doc 选择**：最终展示给 LLM 的 `page_content`，选全局最小 rank 对应的 chunk；rank 相同时按 `ranked_lists` 顺序（dual_level > vector > bm25）优先。
3. **去重 key 兜底**：优先用 `node_id`，缺失时回退到 `page_content[:200]` 的 MD5 hash。
4. **防 mutate**：返回的 Document 是新对象，浅拷贝 metadata，不污染上游检索器返回的原始 Document。

#### 4.2.5 父文档回填（V2 新增，[hybrid_retrieval.py:912](code/C9/rag_modules/hybrid_retrieval.py#L912)）
RRF 融合后返回的是 chunk，内容可能不完整（只含某步骤）。`_attach_parent_documents` 用**完整菜谱文档**替换前 `parent_doc_top_n`（默认 3）条的 `page_content`，给 LLM 更完整上下文。

- 默认**关闭**（`enable_parent_doc_retrieval=False`），直接用 chunk 当上下文
- 只换内容、**不改排名/数量**；超过 `parent_doc_max_chars`（默认 4000）截断
- 父文档映射 `_build_parent_doc_map` 在 `initialize` 时懒建一次：`{node_id: 完整菜谱 Document}`

> 配置开关在 [config.py:31-34](code/C9/config.py#L31)：`enable_parent_doc_retrieval` / `parent_doc_top_n` / `parent_doc_max_chars`。

### 4.3 GraphRAGRetrieval -- 图多跳推理（最大卖点）

文件：[graph_rag_retrieval.py](code/C9/rag_modules/graph_rag_retrieval.py)

**它和混合检索的本质区别**：
- 混合检索：基于**预建索引**匹配，找"长得像"的
- 图 RAG：基于**图拓扑**遍历，找"有关系的"，能发现隐含关联

#### 4.3.1 图索引预热（V2 新增，[graph_rag_retrieval.py:181](code/C9/rag_modules/graph_rag_retrieval.py#L181)）
`initialize` 时 `_build_graph_index` 把 Neo4j 中**按度数降序 Top 1000** 的高频实体预加载到 `entity_cache`，并把所有关系类型频次存入 `relation_cache`。后续查询优先命中本地缓存，减少数据库访问。度数（degree）在这里是衡量节点重要性的指标。

#### 4.3.2 查询意图理解（[graph_rag_retrieval.py:233](code/C9/rag_modules/graph_rag_retrieval.py#L233)）
`understand_graph_query` 用 LLM 把"鸡肉配什么蔬菜"翻译成 `GraphQuery`，含 5 种 `QueryType`：

| 类型 | 场景 | 例子 |
|------|------|------|
| `ENTITY_RELATION` | 一跳直接关系 | 鸡肉和胡萝卜能一起做菜吗？ |
| `MULTI_HOP` | 2-3 跳推理 | 鸡肉配什么蔬菜？（鸡肉->菜品->食材->蔬菜） |
| `SUBGRAPH` | 完整知识网络 | 川菜有什么特色？ |
| `PATH_FINDING` | 最短路径 | 从食材到成品菜的制作路径 |
| `CLUSTERING` | 相似聚类 | 和宫保鸡丁类似的菜有哪些？ |

prompt 里明确区分"图内实体"（川菜、鸡肉）和"属性级约束"（糖尿病、30分钟），后者放进 `constraints` 而非 `source_entities`——这是防止 LLM 把抽象约束当节点查的关键设计。LLM 失败时降级为默认 subgraph 查询。

#### 4.3.3 多跳遍历 + 路径评分（[graph_rag_retrieval.py:368](code/C9/rag_modules/graph_rag_retrieval.py#L368)）
用 Cypher `(source)-[*1..N]-(target)` 做变长路径遍历，**路径评分公式直接写进 Cypher**（[graph_rag_retrieval.py:376](code/C9/rag_modules/graph_rag_retrieval.py#L376)）：
```
relevance = (1.0 / path_len)                           # 短路径得分高
          + (avg_degree / 10.0)                        # 高度数节点得分高
          + (0.3 if relation_type 匹配 else 0.0)       # 关系类型匹配加分
```
短路径 + 高度数节点 + 关系类型匹配 = 高分。取 Top 20。

#### 4.3.4 子图提取 + 图结构推理（[graph_rag_retrieval.py:470](code/C9/rag_modules/graph_rag_retrieval.py#L470)）
`extract_knowledge_subgraph` 用原生 Cypher（不依赖 APOC 插件）提取核心实体周围 `max_depth` 跳的局部子图，并计算图谱指标：节点数、关系数、**密度**（`density = 边数 / (节点数×(节点数-1)/2)`）。`graph_structure_reasoning` 基于子图拓扑生成推理链（因果/组成/相似）。

> **这就是 GraphRAG 论文的核心思想**：传统 RAG 检索孤立的文本块，GraphRAG 检索知识网络。

### 4.4 IntelligentQueryRouter -- 智能路由 + 组合策略

文件：[intelligent_query_router.py](code/C9/rag_modules/intelligent_query_router.py)

**设计哲学**：不是所有问题都需要图推理。"红烧肉怎么做"用混合检索又快又好，"川菜和鲁菜的关系"才需要图 RAG。用 LLM 分析查询特征，自动选引擎。

**四维分析**（[intelligent_query_router.py:126](code/C9/rag_modules/intelligent_query_router.py#L126)）：
- 查询复杂度（0-1）、关系密集度（0-1）、推理需求、实体数量 -> 推荐策略

**三种策略**：
| 策略 | 触发场景 | 执行 |
|------|---------|------|
| `HYBRID_TRADITIONAL` | 简单信息查找 | `hybrid_search` |
| `GRAPH_RAG` | 复杂关系推理 | `graph_rag_search` |
| `COMBINED` | 两者结合 | `_combined_search`（Round-robin） |

**组合策略 Round-robin**（[intelligent_query_router.py:319](code/C9/rag_modules/intelligent_query_router.py#L319)，V2 亮点）：把 `top_k` 平分给两种引擎，**交替**添加结果（图 RAG 优先，通常质量更高），基于 `page_content[:100]` 的 MD5 去重。这是 LightRAG "Round-robin 策略"的体现。

**降级思维**（[intelligent_query_router.py:221](code/C9/rag_modules/intelligent_query_router.py#L221)）：LLM 挂了？用 `_rule_based_analysis` 基于关键词匹配估算复杂度/关系密集度（置信度 0.6）。任何检索引擎失败也降级到传统混合检索。这种"优雅降级"是工程素养的体现，面试加分项。

---

## 第 5 步：技术栈画像

面试官常问"用了哪些技术"，你要能报得清楚：

| 层 | 技术 | 用途 |
|----|------|------|
| 图数据库 | **Neo4j** | 存储菜谱知识图谱（菜谱-食材-步骤-分类） |
| 向量数据库 | **Milvus** | 向量存储 + HNSW 近似最近邻检索 |
| 嵌入模型 | **BGE-small-zh-v1.5** | 中文文本向量化（512 维，CPU 推理，归一化） |
| 向量索引 | **HNSW**（M=16, efConstruction=200, 查询 ef=64） | 近似最近邻，COSINE 度量 |
| LLM | **DeepSeek**（OpenAI 兼容接口） | 关键词提取、查询分析、答案生成 |
| 关键词检索 | **jieba + rank_bm25** | 中文分词 + 停用词过滤 + BM25 |
| 框架 | LangChain | Document 抽象、HuggingFaceEmbeddings |
| 融合算法 | **RRF**（k=60） | 多路检索结果融合 |

**HNSW 关键参数**（[milvus_index_construction.py:208](code/C9/rag_modules/milvus_index_construction.py#L208)）：
- `M=16`：每节点最大邻居数（越大越精确，构建越慢）
- `efConstruction=200`：构建时搜索宽度（越大索引质量越高）
- `ef=64`：查询时搜索宽度（越大越精确，越慢）

**两篇必读论文**（面试会被问"参考了什么"）：
- **LightRAG**（微软，2024）：KV 索引 + 双层检索 + Round-robin 策略的思想来源
- **GraphRAG**（微软，2024）：图结构推理的思想来源

---

## 第 6 步：如何把它变成"你自己的项目"

光看懂不够，要动手改，才是你的。建议按这个顺序改造：

### 改造 1（最简单，验证你跑通了）：换数据源
把菜谱换成你熟悉的领域（比如电影、招聘、法律条文）。改 `graph_data_preparation.py` 的 Cypher 查询和 `build_recipe_documents` 的文档组装逻辑。这一步能逼你真正理解数据层。

### 改造 2（中等，体现你懂检索）：调 RRF / 父文档参数
- 改 `_RRF_K`（[hybrid_retrieval.py:56](code/C9/rag_modules/hybrid_retrieval.py#L56)）看效果变化（k 越大，排名差异越平滑）
- 打开父文档回填：`enable_parent_doc_retrieval=True`，对比开/关的答案完整度
- 给三路加权重（当前是无权重的纯 RRF，你可以试试加权 RRF）
- 加一路新的检索（比如重排器 reranker）

### 改造 3（进阶，体现你懂架构）：加评估体系
当前项目**没有评估**，这是最大的短板。加一个：
- 用 RAGAS 评估答案质量（faithfulness、answer_relevancy）
- 构造测试集，对比 hybrid vs graph_rag 在不同查询上的表现
- 这能让你的项目从"玩具"变成"工程"

### 改造 4（高阶，真正的差异化）：补全占位实现
`graph_rag_retrieval.py` 里这几个是占位实现（返回空或固定值），把它们真正实现，你的图 RAG 就完整了：
- `_find_entity_relations`（[graph_rag_retrieval.py:947](code/C9/rag_modules/graph_rag_retrieval.py#L947)）：实体一跳关系查询，当前返回 `[]`
- `_find_shortest_paths`（[graph_rag_retrieval.py:959](code/C9/rag_modules/graph_rag_retrieval.py#L959)）：最短路径查找，当前返回 `[]`
- `graph_structure_reasoning` / `_build_reasoning_chain` / `_validate_reasoning_chains`：推理链当前是固定字符串占位

### 改造 5（修 bug，体现你读懂了细节）：统一关系类型
`graph_indexing.py` 用 `HAS_STEP`，而数据层和图 RAG 用 `CONTAINS_STEP`（见 4.1 的提醒）。统一成 `CONTAINS_STEP`，否则步骤关系的主题键索引匹配不上实际边。能发现并修复这个，面试时非常加分。

---

## 第 7 步：面试高频问题预演

提前想好这几个问题的答案：

1. **"你这个项目和普通 RAG 有什么区别？"** -> 第 1 步的对比表
2. **"为什么用图数据库而不是直接存文档？"** -> 菜谱有天然的结构化关系（食材-步骤-分类），图能做关系推理，纯文档做不到
3. **"RRF 为什么比直接加权分数好？"** -> 不同检索器分数量纲不同，不可加；RRF 只用排名，鲁棒。再讲同 source 内只取最佳 rank 避免重复加分、canonical doc 选择
4. **"图 RAG 什么场景下比混合检索好？"** -> 多跳关系推理（A 配什么 B）、子图级问题（X 有什么特色）
5. **"路由器怎么决定用哪个引擎？"** -> LLM 四维分析 + 规则降级兜底；组合策略用 Round-robin 交替融合
6. **"瓶颈在哪？怎么优化？"** -> LLM 调用太多（关键词提取+查询分析+图意图理解+生成至少 4 次），可缓存/用小模型替代中间步骤；图索引预热 Top 1000 实体已是优化手段
7. **"如果数据量增大 100 倍怎么办？"** -> 图索引内存放不下，要改成分片/外部索引；Milvus 要分区；BM25 索引可换 Elasticsearch
8. **"父文档检索解决什么问题？"** -> chunk 切碎后上下文不全，用完整菜谱文档回填前 N 条，给 LLM 更完整上下文（可控开关、不改排名）
9. **"系统怎么保证可用性？"** -> 多层降级：LLM 挂了走规则分析；图键值没召回走 Neo4j fulltext；流式失败降级非流式；任一引擎失败降级传统检索

---

## 配置与运行（V2 新增章节）

### 环境变量（[.env.example](code/C9/.env.example)）
```bash
MOONSHOT_API_KEY=your_api_key_here     # LLM API key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=all-in-rag
MILVUS_HOST=localhost
MILVUS_PORT=19530
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
LLM_MODEL=deepseek-v4-flash
LOG_LEVEL=INFO
```

### ⚠️ 当前配置的几个不一致点（务必知晓）
通读代码后发现，**配置尚未完全接入环境变量**，存在几处不一致，运行前需要对齐：

1. **LLM 模型名不一致**：
   - `config.py` 默认 `llm_model = "deepseek-v4-flash"`（[config.py:26](code/C9/config.py#L26)）
   - `generation_integration.py` 默认 `model_name = "deepseek-v4-flash"`（[generation_integration.py:50](code/C9/rag_modules/generation_integration.py#L50)）
   - `.env.example` 示例 `LLM_MODEL=deepseek-v4-flash`
   - 实际生效的是 `config.py` 的值（main.py 用 `self.config.llm_model` 传入），即 **deepseek-v4-flash**

2. **API key 名不一致**：
   - `generation_integration.py` 读的是 `os.getenv("DEEPSEEK_API_KEY")`，base_url 是 `https://api.deepseek.com`（[generation_integration.py:68-75](code/C9/rag_modules/generation_integration.py#L68)）
   - 但 `.env.example` 写的是 `MOONSHOT_API_KEY`
   - **运行时必须设置 `DEEPSEEK_API_KEY`**，否则生成模块初始化直接抛 ValueError

3. **Neo4j 端口不一致**：
   - `config.py` 默认 `neo4j_uri = "bolt://localhost:17687"`（[config.py:13](code/C9/config.py#L13)）
   - `.env.example` 示例 `bolt://localhost:7687`
   - 实际生效的是 `config.py` 的 **17687**

4. **config.py 未读 .env**：`GraphRAGConfig` 全是 dataclass 硬编码默认值，没有 `os.getenv`。`main.py` 虽 `load_dotenv()`，但只有 `generation_integration.py` 的 `DEEPSEEK_API_KEY` 真正用上了 .env。

> **结论**：要么按 `config.py` 的默认值准备环境（DeepSeek key + Neo4j 17687），要么改造 `config.py` 让它从环境变量读取（见改造 6）。这也是一个很好的"配置工程化"改造练习。

### 改造 6（配置工程化）：让 config.py 接入环境变量
把 `GraphRAGConfig` 的字段默认值改成 `os.getenv("XXX", default)`，让 `.env` 真正生效。这是把项目从"能跑"推向"可部署"的关键一步。

### 启动流程（[main.py:537](code/C9/main.py#L537)）
```bash
python main.py
```
```
main() -> AdvancedGraphRAGSystem()
       -> initialize_system()     # 初始化 6 大核心模块
       -> build_knowledge_base()  # 构建或加载知识库
       -> run_interactive()       # 进入交互式问答循环
```
交互命令：`stats`（系统统计）/ `rebuild`（重建知识库）/ `quit`（退出）/ 其他输入作为问题（默认流式输出）。

---

## 建议的学习路径

1. **本周**：跑通项目（先按上面"配置不一致"对齐环境），对着第 3 步的流程图，在每个关键函数打断点，看数据长什么样
2. **下周**：完成改造 1（换数据源）+ 改造 6（配置接入环境变量），这两步最磨人但也最涨功力
3. **第三周**：完成改造 3（加评估），这是工程化的关键
4. **面试前**：能把第 3 步流程图默画出来，第 7 步问题对答如流，能主动说出"配置不一致""HAS_STEP/CONTAINS_STEP"这类细节

---

## 附 A：关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| RAG | 检索增强生成，先检索再让 LLM 基于资料回答 |
| Embedding | 把文本转成向量，语义相近的文本向量也相近 |
| HNSW | 层次导航小世界算法，Milvus 用的近似最近邻索引 |
| COSINE | 余弦相似度，衡量两个向量方向的一致性 |
| BM25 | 基于词频-逆文档频率的关键词检索算法 |
| RRF | 倒数排名融合，用排名而非分数融合多路检索结果（k=60） |
| canonical doc | RRF 融合后选出的"代表文档"，取全局最小 rank 的 chunk |
| GraphRAG | 用知识图谱的结构化关系做检索和推理的 RAG |
| LightRAG | 微软提出的轻量图 RAG 框架，KV 索引 + 双层检索 + Round-robin |
| 多跳推理 | 通过图的多条边连接，发现间接关系（A->B->C） |
| 子图提取 | 围绕核心实体提取局部知识网络（含密度等图谱指标） |
| 父文档检索 | 检索到 chunk 后，用其所属的完整文档替换，补全上下文 |
| 双层检索 | 实体级（精确匹配）+ 主题级（关系/分类抽象）两层召回 |
| Reranker | 对初检结果做精排，提升 top-k 质量 |
| RAGAS | RAG 系统的评估框架，评估生成质量和检索质量 |
| Round-robin | 轮询交替融合，组合策略里图 RAG 与传统检索交替取结果 |

## 附 B：V2 相对 V1 的主要变化

| 变化点 | V1 | V2（当前代码） |
|--------|----|----|
| 父文档检索 | 无 | 新增 `_attach_parent_documents`，默认关闭，前 N 条回填完整菜谱 |
| RRF 融合 | 基础去重 | 新增 canonical doc 选择、`rrf_chunk_hits`、浅拷贝防 mutate |
| BM25 分词 | jieba 分词 | 新增 `_CHINESE_STOPWORDS` 停用词表 + 单字符过滤 |
| Neo4j 降级 | 未提 | 实体级 fulltext + 主题级 Cypher 两层降级补充检索 |
| 图 RAG 预热 | 未提 | 新增 `_build_graph_index`，Top 1000 高度数实体缓存 |
| 查询类型 | 未详列 | 5 种 QueryType 明确分派（multi_hop/subgraph/...） |
| 组合策略 | 未详述 | `_combined_search` Round-robin 交替融合，图 RAG 优先 |
| 生成模块 | 标准生成 | 流式 + 重试（3 次，2/4/6s）+ 降级非流式 |
| 行号引用 | 旧行号 | 全部更新为当前行号 |
| 已知问题 | 未列 | 新增 HAS_STEP/CONTAINS_STEP 不一致、配置未接入 .env 等提醒 |

## 附 C：V2.1 精确率提升变更（top_k 收敛 + Rerank 精排）

| 变化点 | V2 | V2.1（当前代码） |
|--------|----|----|
| 默认 top_k | 5 | 3（收敛返回块数以降低噪声） |
| 两阶段精排 | 无 | 三路 RRF 融合出候选池（20）→ `bge-reranker-v2-m3` cross-encoder 精排 → 取 top-3 |
| 重排模块 | 无 | 新增 `rag_modules/reranker.py`（`RerankerModule`，懒加载 + 可降级） |
| 前端展示 | 仅通道命中 | 新增 🔁 重排 tag + 重排得分行 |
| 模型下载 | 无 | `scripts/download_reranker.py` 一次性下载（支持 hf-mirror） |
| 降级行为 | - | 模型未缓存/加载失败时自动跳过重排，返回 RRF 原顺序 |
