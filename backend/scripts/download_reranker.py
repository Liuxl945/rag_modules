"""一次性下载重排模型到本地 HuggingFace 缓存。

首次启用重排需要联网下载一次 BAAI/bge-reranker-v2-m3（~568MB），下载后永久缓存，
之后即使 HF_HUB_OFFLINE=1（离线模式）也能用。此脚本不修改运行时配置，只负责触发下载。

用法（任选其一，从 backend/ 目录运行）：

  # 1. 默认从 HuggingFace 下载（需能访问 huggingface.co）
     HF_HUB_OFFLINE=0 python scripts/download_reranker.py

  # 2. 国内镜像（推荐，下载更快）
     HF_ENDPOINT=https://hf-mirror.com python scripts/download_reranker.py

  # 3. 自定义模型名
     RERANK_MODEL=BAAI/bge-reranker-base python scripts/download_reranker.py

下载完成后，重新启动后端即可自动启用重排（config.enable_rerank 默认 True）。
"""

import os
import sys


def main():
    model_name = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    # 确保允许联网下载（若 HF_HUB_OFFLINE=1，sentence-transformers 不会尝试下载）
    offline = os.getenv("HF_HUB_OFFLINE", "").strip().lower()
    if offline in ("1", "true", "yes"):
        print(
            f"⚠️  当前 HF_HUB_OFFLINE={offline}，无法下载。\n"
            f"   请运行：HF_HUB_OFFLINE=0 python scripts/download_reranker.py\n"
            f"   国内用户：HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=0 python scripts/download_reranker.py"
        )
        sys.exit(1)

    max_length = int(os.getenv("RERANK_MAX_LENGTH", "512"))

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("❌ 缺少 sentence-transformers，请先：pip install sentence-transformers")
        sys.exit(1)

    print(f"开始下载重排模型: {model_name}（max_length={max_length}）...")
    print("（首次下载约 568MB，耐心等待，完成后会永久缓存）")

    try:
        # 实例化即触发下载；device=cpu 与主应用一致
        model = CrossEncoder(model_name, max_length=max_length, device="cpu")
        # 用一个最小样例验证模型可用
        score = model.predict([("鸡胸肉怎么做", "鸡胸肉洗净切块，热锅冷油下锅煎至两面金黄")])[0]
        print(f"✅ 模型下载并加载成功，测试打分: {float(score):.4f}")
        print("现在可以重新启动后端，重排将自动启用（config.enable_rerank=True）。")
    except Exception as e:
        print(f"❌ 模型下载/加载失败: {e}")
        print("如果是网络问题，尝试国内镜像：")
        print("  HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=0 python scripts/download_reranker.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
