from __future__ import annotations

from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GENERATED = DOCS / "generated"

ARTICLES = [
    {"path": "papers/A/01-foundations/A001-backpropagation.md", "title": "反向传播：多层网络为什么能够端到端学习", "paper": "Rumelhart, Hinton & Williams, Learning representations by back-propagating errors", "url": "https://doi.org/10.1038/323533a0", "legacy": True},
    {"path": "papers/A/01-foundations/A002-adam.md", "title": "Adam：一阶矩、二阶矩与偏差校正到底在做什么", "paper": "Adam: A Method for Stochastic Optimization", "url": "https://arxiv.org/abs/1412.6980", "legacy": True},
    {"path": "papers/A/01-foundations/A003-adamw.md", "title": "AdamW：为什么 Weight Decay 必须从自适应梯度里解耦", "paper": "Decoupled Weight Decay Regularization", "url": "https://arxiv.org/abs/1711.05101", "legacy": True},
    {"path": "papers/A/01-foundations/A004-batch-normalization.md", "title": "Batch Normalization：算法、反向传播与机制解释边界", "paper": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", "url": "https://arxiv.org/abs/1502.03167", "legacy": True},
    {"path": "papers/A/01-foundations/A005-resnet.md", "title": "ResNet：残差学习为什么缓解深层网络的退化问题", "paper": "Deep Residual Learning for Image Recognition", "url": "https://arxiv.org/abs/1512.03385", "legacy": True},
    {"path": "papers/A/01-foundations/A006-scalable-muon.md", "title": "Scalable Muon：矩阵优化器怎样扩展到大模型训练", "paper": "Muon is Scalable for LLM Training", "url": "https://arxiv.org/abs/2502.16982", "legacy": True},
    {"path": "papers/A/03-transformer/A013-transformer-attention-is-all-you-need.md", "title": "Transformer：Attention 如何替代循环序列建模", "paper": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762", "legacy": True},
    {"path": "papers/A/02-text-representation/A009-kaplan-scaling-laws.md", "title": "Kaplan Scaling Laws", "paper": "Scaling Laws for Neural Language Models", "url": "https://arxiv.org/abs/2001.08361", "legacy": False},
    {"path": "papers/A/02-text-representation/A010-chinchilla-compute-optimal.md", "title": "Chinchilla", "paper": "Training Compute-Optimal Large Language Models", "url": "https://arxiv.org/abs/2203.15556", "legacy": False},
    {"path": "papers/A/03-transformer/A018-yarn.md", "title": "YaRN", "paper": "YaRN: Efficient Context Window Extension of Large Language Models", "url": "https://arxiv.org/abs/2309.00071", "legacy": False},
    {"path": "papers/A/03-transformer/A014-roformer-rope.md", "title": "RoPE：把相对位置信息写进旋转内积", "paper": "RoFormer: Enhanced Transformer with Rotary Position Embedding", "url": "https://arxiv.org/abs/2104.09864", "legacy": True},
    {"path": "papers/A/03-transformer/A016-rmsnorm.md", "title": "RMSNorm：为什么只做均方根缩放也能稳定 Transformer", "paper": "Root Mean Square Layer Normalization", "url": "https://arxiv.org/abs/1910.07467", "legacy": True},
    {"path": "papers/A/04-efficient-attention/A019-mqa.md", "title": "MQA：为什么共享 K/V 能显著降低 Decode Cache", "paper": "Fast Transformer Decoding: One Write-Head is All You Need", "url": "https://arxiv.org/abs/1911.02150", "legacy": True},
    {"path": "papers/A/04-efficient-attention/A020-gqa.md", "title": "GQA：在 MHA 表达能力与 MQA 推理效率之间折中", "paper": "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", "url": "https://arxiv.org/abs/2305.13245", "legacy": True},
    {"path": "papers/A/04-efficient-attention/A053-dual-chunk-attention.md", "title": "Dual Chunk Attention", "paper": "Training-Free Long-Context Scaling of Large Language Models", "url": "https://arxiv.org/abs/2402.17463", "legacy": False},
    {"path": "papers/A/04-efficient-attention/A054-minference.md", "title": "MInference 1.0", "paper": "MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention", "url": "https://arxiv.org/abs/2407.02490", "legacy": False},
    {"path": "papers/A/10-gpu-operators/A055-flashattention.md", "title": "FlashAttention", "paper": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "url": "https://arxiv.org/abs/2205.14135", "legacy": False},
    {"path": "papers/A/10-gpu-operators/A056-flashattention2.md", "title": "FlashAttention-2", "paper": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", "url": "https://arxiv.org/abs/2307.08691", "legacy": False},
    {"path": "papers/C/01-distributed-training/C001-4d-parallelism.md", "title": "4D Parallelism", "paper": "System synthesis: Llama 3 / Megatron-LM / GPipe / ZeRO / Ring Attention", "url": "https://arxiv.org/abs/2407.21783", "legacy": False},
    {"path": "papers/A/05-moe/A032-sparsely-gated-moe.md", "title": "Sparsely-Gated MoE：条件计算怎样扩大模型容量", "paper": "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", "url": "https://arxiv.org/abs/1701.06538", "legacy": True},
    {"path": "papers/A/05-moe/A033-switch-transformer.md", "title": "Switch Transformer：把 Top-K MoE 简化成 Top-1 路由", "paper": "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", "url": "https://arxiv.org/abs/2101.03961", "legacy": True},
    {"path": "papers/A/05-moe/A034-deepseekmoe.md", "title": "DeepSeekMoE", "paper": "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models", "url": "https://arxiv.org/abs/2401.06066", "legacy": False},
    {"path": "papers/B/05-moe-complete-llm/B008-deepseek-v2.md", "title": "DeepSeek-V2", "paper": "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", "url": "https://arxiv.org/abs/2405.04434", "legacy": False},
    {"path": "papers/B/05-moe-complete-llm/B009-deepseek-v3.md", "title": "DeepSeek-V3", "paper": "DeepSeek-V3 Technical Report", "url": "https://arxiv.org/abs/2412.19437", "legacy": False},
    {"path": "papers/B/05-moe-complete-llm/B010-deepseek-r1.md", "title": "DeepSeek-R1", "paper": "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", "url": "https://arxiv.org/abs/2501.12948", "legacy": False},
    {"path": "papers/B/05-moe-complete-llm/B011-llama3.md", "title": "Llama 3", "paper": "The Llama 3 Herd of Models", "url": "https://arxiv.org/abs/2407.21783", "legacy": False},
    {"path": "papers/B/05-moe-complete-llm/B012-qwen2.5.md", "title": "Qwen2.5", "paper": "Qwen2.5 Technical Report", "url": "https://arxiv.org/abs/2412.15115", "legacy": False},
    {"path": "papers/A/08-post-training/A051-deepseekmath-grpo.md", "title": "DeepSeekMath / GRPO", "paper": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", "url": "https://arxiv.org/abs/2402.03300", "legacy": False},
    {"path": "papers/A/08-post-training/A052-dpo.md", "title": "DPO", "paper": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "url": "https://arxiv.org/abs/2305.18290", "legacy": False},
]

BANNED_QUOTE_FRAGMENTS = (
    "V2 重写",
    "重制版 A",
    "Paperlist",
    "figure_manifests/",
    "公开发布前",
    "本项目研究批注",
)

TEXT_REPLACEMENTS = {
    "当前项目中": "在这条技术主线中",
    "当前项目真实进度": "这条技术主线",
    "从**当前真实进度**继续": "沿这条技术链继续",
    "Paperlist 的下一编号": "线性论文编号中的下一项",
    "按 Paperlist 顺序": "按线性论文清单顺序",
    "Paperlist": "论文资源清单",
    "当前项目下一批": "后续专题",
}

def demote_headings(text: str) -> str:
    out = []
    in_fence = False
    fence = None
    backtick_fence = chr(96) * 3
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith((backtick_fence, "~~~")):
            token = stripped[:3]
            if not in_fence:
                in_fence = True
                fence = token
            elif token == fence:
                in_fence = False
                fence = None
            out.append(line)
            continue
        if not in_fence and re.match(r"^#{1,5} ", line):
            line = "#" + line
        out.append(line)
    return "\n".join(out).rstrip() + "\n"

def strip_internal_quotes(text: str) -> str:
    out = []
    for line in text.splitlines():
        if line.startswith(">") and any(fragment in line for fragment in BANNED_QUOTE_FRAGMENTS):
            continue
        if "figure_manifests/" in line:
            continue
        out.append(line)
    return "\n".join(out)

def sanitize(article: dict, raw: str) -> str:
    text = raw.replace("\r\n", "\n")
    # Normalize legacy Markdown syntax for MyST/Sphinx without touching source drafts.
    text = text.replace("\n\\[\n", "\n$$\n").replace("\n\\]\n", "\n$$\n")
    backtick_fence = chr(96) * 3
    text = text.replace(backtick_fence + "mermaid", backtick_fence + "{mermaid}")
    text = text.replace("~~~mermaid", "~~~{mermaid}")
    for old, new in TEXT_REPLACEMENTS.items():
        text = text.replace(old, new)

    if article["legacy"]:
        lines = text.splitlines()
        start = next((i for i, line in enumerate(lines) if line.startswith("# ")), 0)
        body = "\n".join(lines[start:])
        body = strip_internal_quotes(body)
        body = demote_headings(body)
        header = (
            f"# {article['title']}\n\n"
            f"> **原论文**：[{article['paper']}]({article['url']})。  \n"
            f"> 本文在原论文论证链基础上补充必要数学推导、工程直觉与现代实现对照。\n\n"
        )
        return header + body

    text = strip_internal_quotes(text)
    text = text.replace(
        "而从**这条技术主线**看，V3 与 R1 也已经完成第一轮主干拆解，因此",
        "沿这条技术链继续，",
    )
    text = text.replace("已经完成第一轮主干拆解", "已经形成完整技术连接")
    text = text.replace("## 14.2 现在应该往哪里走？", "## 14.2 从这里应该往哪里走？")
    return text.rstrip() + "\n"

def main() -> None:
    if GENERATED.exists():
        shutil.rmtree(GENERATED)
    (GENERATED / "papers").mkdir(parents=True, exist_ok=True)

    for article in ARTICLES:
        src = ROOT / article["path"]
        if not src.is_file():
            raise FileNotFoundError(src)
        rel = Path(article["path"])
        dst = GENERATED / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(sanitize(article, src.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"ARTICLE {rel.as_posix()}")

    src_figures = ROOT / "figures"
    dst_figures = GENERATED / "figures"
    if src_figures.exists():
        shutil.copytree(src_figures, dst_figures)
        print("FIGURES copied")

    md_link = re.compile(r"\[[^\]]*\]\(([^)]+\.md(?:#[^)]+)?)\)")
    broken = []
    for md in (GENERATED / "papers").rglob("*.md"):
        content = md.read_text(encoding="utf-8")
        for raw_target in md_link.findall(content):
            target = raw_target.split("#", 1)[0]
            resolved = (md.parent / target).resolve()
            if not resolved.exists():
                broken.append((md.relative_to(GENERATED), raw_target))
    if broken:
        joined = "\n".join(f"{src}: {target}" for src, target in broken)
        raise RuntimeError("Generated article links point to unpublished pages:\n" + joined)

    print(f"PUBLIC_ARTICLES={len(ARTICLES)}")

if __name__ == "__main__":
    main()
