"""
生成集成模块 - 负责基于检索结果生成最终答案

核心职责：
    将检索到的文档（context）和用户问题（question）组装为 LLM 提示词，
    调用 LLM 生成烹饪领域的专业回答。

提供两种生成模式：
    1. generate_adaptive_answer       -> 标准模式（一次性返回完整答案）
    2. generate_adaptive_answer_stream -> 流式模式（逐字输出，带重试机制）

外部依赖：
    - OpenAI SDK（兼容 DeepSeek API）：通过 DEEPSEEK_API_KEY 环境变量认证
    - langchain_core.documents.Document：检索结果的统一格式

Note:
    本模块使用 LightRAG 风格的统一提示词，根据问题性质自适应输出格式
    （列表 / 详细步骤 / 综合回答），无需预先分类查询类型。
"""

import logging
import os
import time
from typing import List

from openai import OpenAI
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """生成集成模块 - 负责答案生成

    职责：
        - 初始化 LLM 客户端（兼容 OpenAI 接口的 DeepSeek API）
        - 将检索到的文档组装为上下文（context）
        - 调用 LLM 生成烹饪领域的专业回答

    设计要点：
        - 使用 LightRAG 风格的统一提示词，根据问题性质自适应输出格式
        - 流式模式带重试机制（max_retries=3，递增等待时间）
        - 流式失败时降级为非流式生成，确保系统可用性

    Public API:
        - generate_adaptive_answer(question, documents)        -> 标准模式生成
        - generate_adaptive_answer_stream(question, documents) -> 流式模式生成（生成器）
    """

    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 2048):
        """初始化生成集成模块。

        从环境变量读取 DEEPSEEK_API_KEY，创建 OpenAI 兼容客户端连接 DeepSeek API。

        Args:
            model_name: LLM 模型名称（默认 "kimi-k2-0711-preview"）
            temperature: 生成温度（默认 0.1，低温度确保回答稳定）
            max_tokens: 单次回答最大输出 token 数（默认 2048）

        Raises:
            ValueError: 当 DEEPSEEK_API_KEY 环境变量未设置时
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 初始化 OpenAI 客户端（使用 DeepSeek API 端点）
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"  # DeepSeek API 端点（兼容 OpenAI 接口）
        )

        logger.info(f"生成模块初始化完成，模型: {model_name}")

    def generate_adaptive_answer(self, question: str, documents: List[Document]) -> str:
        """智能统一答案生成（标准模式，一次性返回完整答案）。

        自动适应不同类型的查询，无需预先分类：
            - 询问多个菜品 -> 提供清晰的列表
            - 询问具体制作方法 -> 提供详细步骤
            - 一般性咨询 -> 提供综合性回答

        Args:
            question: 用户的问题
            documents: 检索到的相关文档列表（page_content 作为上下文）

        Returns:
            LLM 生成的完整回答字符串（生成失败时返回错误提示）
        """
        # 构建上下文：将检索到的文档内容拼接，保留检索层级标记
        context_parts = []

        for doc in documents:
            content = doc.page_content.strip()
            if content:
                # 添加检索层级信息（如果有的话，如 [ENTITY] / [TOPIC]）
                level = doc.metadata.get('retrieval_level', '')
                if level:
                    context_parts.append(f"[{level.upper()}] {content}")
                else:
                    context_parts.append(content)

        context = "\n\n".join(context_parts)

        # LightRAG 风格的统一提示词（根据问题性质自适应输出格式）
        prompt = f"""
        作为一位专业的烹饪助手，请基于以下信息回答用户的问题。

        检索到的相关信息：
        {context}

        用户问题：{question}

        请提供准确、实用的回答。根据问题的性质：
        - 如果是询问多个菜品，请提供清晰的列表
        - 如果是询问具体制作方法，请提供详细步骤
        - 如果是一般性咨询，请提供综合性回答

        回答：
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LightRAG答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def generate_adaptive_answer_stream(self, question: str, documents: List[Document], max_retries: int = 3):
        """LightRAG 风格的流式答案生成（带重试机制）。

        逐字输出 LLM 生成的回答，适用于实时交互场景。
        内置重试机制（max_retries=3，递增等待时间），流式失败时降级为非流式生成。

        Args:
            question: 用户的问题
            documents: 检索到的相关文档列表
            max_retries: 最大重试次数（默认 3）

        Yields:
            str: LLM 生成的文本片段（逐字输出）

        重试策略：
            - 每次重试等待时间递增（(attempt+1)*2 秒）
            - 所有重试失败后降级为非流式生成（generate_adaptive_answer）
            - 非流式也失败时返回错误提示
        """
        # 构建上下文（与 generate_adaptive_answer 逻辑一致）
        context_parts = []

        for doc in documents:
            content = doc.page_content.strip()
            if content:
                level = doc.metadata.get('retrieval_level', '')
                if level:
                    context_parts.append(f"[{level.upper()}] {content}")
                else:
                    context_parts.append(content)

        context = "\n\n".join(context_parts)

        # LightRAG 风格的统一提示词
        prompt = f"""
        作为一位专业的烹饪助手，请基于以下信息回答用户的问题。

        检索到的相关信息：
        {context}

        用户问题：{question}

        请提供准确、实用的回答。根据问题的性质：
        - 如果是询问多个菜品，请提供清晰的列表
        - 如果是询问具体制作方法，请提供详细步骤
        - 如果是一般性咨询，请提供综合性回答

        回答：
        """

        # 重试循环：最多尝试 max_retries 次
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,    # 启用流式输出
                    timeout=60      # 增加超时设置（60秒，避免长回答中断）
                )

                # 首次尝试和重试时输出不同的提示信息
                if attempt == 0:
                    print("开始流式生成回答...\n")
                else:
                    print(f"第{attempt + 1}次尝试流式生成...\n")

                # 逐字读取流式响应并 yield 输出
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content  # 使用 yield 返回流式内容（生成器模式）

                # 如果成功完成，退出重试循环
                return

            except Exception as e:
                logger.warning(f"流式生成第{attempt + 1}次尝试失败: {e}")

                if attempt < max_retries - 1:
                    # 还有重试机会：递增等待时间后重试
                    wait_time = (attempt + 1) * 2  # 递增等待时间（2s / 4s / 6s）
                    print(f"⚠️ 连接中断，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 所有重试都失败：降级为非流式生成
                    logger.error(f"流式生成完全失败，尝试非流式后备方案")
                    print("⚠️ 流式生成失败，切换到标准模式...")

                    try:
                        # 调用非流式生成作为后备方案
                        fallback_response = self.generate_adaptive_answer(question, documents)
                        yield fallback_response
                        return
                    except Exception as fallback_error:
                        # 非流式也失败：返回错误提示
                        logger.error(f"后备生成也失败: {fallback_error}")
                        error_msg = f"抱歉，生成回答时出现网络错误，请稍后重试。错误信息：{str(e)}"
                        yield error_msg
                        return
