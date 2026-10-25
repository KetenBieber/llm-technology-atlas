# SnapKV：为什么生成前的 Prompt 后缀能提示哪些 KV 应该保留？

> **原始论文**：[SnapKV: LLM Knows What You are Looking for Before Generation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html)，NeurIPS 2024。 [官方/作者 PDF](https://arxiv.org/pdf/2404.14469)。


前置：[H2O 的动态重度命中统计](C036-h2o.md) 和 [PagedAttention 的块式分配](../01-distributed-training/C012-vllm-pagedattention.md)。后续：[KIVI 保留位置但减少每个 KV 的比特](C038-kivi.md)。

## 如果 Prompt 很长，能否在第一个 Decode Token 之前先压缩它？

传统 KV 缓存 $K,V$ 每层保留全部历史长度 $L$，当用户一次性提交几十万 Token 文档时，完整 Prefill 完成后 Decode 每步仍需读大量历史 Key。H2O 可以边生成边统计过去哪些 Token 获得高注意力，但 SnapKV 问了一个更具体的问题：

**模型即将开始回答时，Prompt 末尾的几个 Query 是否已经体现了它接下来会重点关注哪些历史内容？**

论文从每个 Attention Head 的实证可视化观察到，Prompt 尾部 Observation Window 的注意力偏好，与该 Head 后续生成时聚焦的历史位置在多种任务上有稳定相关性。于是可以在真正开始 Decode 之前，先用末尾窗口的 Attention 作为 Proxy 选择缓存位置。

## 按每个 Head 压缩，不是所有层共用一张“重要 Token 表”

设第 $h$ 个 Head 的 Prompt Attention 为 $A^{(h)}\in\mathbb R^{L\times L}$。取尾部窗口长度 $w$，对历史位置 $j\le L-w$ 定义示意分数

$$
s_j^{(h)}=\frac1w\sum_{i=L-w+1}^{L}A_{ij}^{(h)}.
$$

真正实现还包括 **Pooling/Clustering**：避免每个 Head 的重要位置因局部微小抖动而产生无用的孤立点。它在第 $h$ 个 Head 上选取约 $k$ 个高分位置，其他 Head 的选择可以不同：

$$
I_h=\operatorname{TopK}_j
\big(\operatorname{Pool}(s^{(h)})_j,\ k\big).
$$

这里 $k$ 为压缩预算，Pool 是对应局部观测汇聚的抽象。与把整条 Prompt 截成最近 $k$ 个 Token 不同，$I_h$ 可以包含远处的代码定义、相关背景或长文中的锚点；与 H2O 不同，它是在 Prompt 末尾“预测未来重要位置”，而不依赖生成期间持续累积得分。

~~~{mermaid}
flowchart LR
 P[长 Prompt Prefill] --> A[末尾 Observation Window]
 A --> H[每个 Head 汇总历史注意力]
 H --> C[局部 Pooling + TopK]
 C --> S[按 Head 压缩历史 KV]
 S --> D[更小的 Decode KV]
~~~

*自制算法结构图，非原论文 Figure。原论文关注 Prediction Consistency、Observation Window 对应的分数分布、Pool/TopK 和生成性能。*

## 概念实验：一个 Head 保留索引与另一个 Head 不同

假设 Prompt 有 10000 个 Token，某 Head 为了追踪实体引用更关注索引 300、1024、8500，另一个 Head 倾向对近期语法位置 9400–9900 赋较高分数。SnapKV 并非必须让所有 Head 都只保留这三段，也不把 Head 得分相加为全层唯一 TopK；它允许按 Head 分别选址，在总预算下保存该 Head 所需的 Key/Value。

注意原论文并不保证“未来 Query 总会关注同样位置”。用户接着要求模型对长文其他主题作答案，预观察窗口可能漏掉其证据；检索精度下降风险是真实的。**KV 驱逐是有损近似**，不是 [FlashAttention](../../A/10-gpu-operators/A055-flashattention.md) 那样不改变 Attention 数学结果的 IO 优化。

## 原文实验结论为何要看平台与序列设置？

[NeurIPS 2024 正式论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html) 报告在 **16K Prompt** 条件下相对基线约 **3.6× 生成速度**、**8.2× 内存效率**，在 16 项长序列数据集上保持可比效果；并有单张 A100 80GB + HuggingFace 实现运行更长 Prompt 的演示。这里“可运行更长 Context”并不保证模型的训练/位置外推能力同步改善。

复现实验要固定 Observation Window 大小、每 Head 的 KV 容量、模型层数/头数和 Prompt 长度；比较等额缓存预算下的 Keep-Recent、[H2O](C036-h2o.md) 与 SnapKV。一定要同时测 long-context QA 或 Needle Retrieval，单看延迟不能发现关键证据被错误驱逐。

下一步 [KIVI](C038-kivi.md) 采取另一个独立轴：不根据重要性删除所有低分 Token，而对要保留的 Key/Value 使用不同维度的量化，减少每个元素的字节数。
