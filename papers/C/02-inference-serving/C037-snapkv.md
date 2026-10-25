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

## 先把 Score 聚合、局部 Pool、TopK 按顺序串起来

设输入 Prompt 长 $L$，Observation Window 长 $w$，在某层某个 Head 中有 Query–Key Attention 矩阵 $A\in\mathbb R^{L\times L}$（实际实现无需永久保留这个完整矩阵）。对历史位置 $j<L-w$ 先累积末端 Query 注意力：

$$
s_j=\sum_{i=L-w+1}^{L}A_{ij}.
$$

再对沿**序列轴**邻近的 $s_j$ 进行局部 Pooling，例如使用奇数核宽 $k$ 的最大池化：

$$
\widetilde s_j=\max_{r\in[j-\lfloor k/2\rfloor,j+\lfloor k/2\rfloor]}s_r.
$$

最终为每个 Head 选出固定预算内的历史位置 $I_h=\operatorname{TopK}(\widetilde s,K)$，再把最近 Observation Window 的 KV 与选中历史 KV 组合。这段演示刻意区分**时间窗口**和**Pool Kernel**：前者决定观察哪些 Query，后者决定历史重要性索引的局部聚合；把两者都叫一个 Window 会导致实现时维度错误。

### 数字例子：为什么没有 Pooling 时可能错过连续结构？

假设某 Head 对历史六个位置的 Score 为 §[0.02,0.45,0.05,0.40,0.03,0.05]§。直接 Top-2 选择位置 2、4；若 $k=3$ 的 Max-Pool，位置 1–3 可能都被“位置 2 很重要”带动而得较高分，位置 3–5 则被位置 4 带动。此时 TopK 会倾向保留一段局部相关位置附近的 Key。它既可能有利于保留完整知识片段，也可能把某个尖峰膨胀成冗余占用，因此实际需消融 Kernel Size 和 K。

上面仅是示意值，真实作者实现的 Pool Mode、Tie-breaking 和 Keep-Recent 规则应按代码/论文版本核对，不能把教学核假定成唯一的具体实现。

## Prefill/Decode 必须分开测

SnapKV 的压缩动作发生在 Prompt Prefill 后、真正的生成前；它减少的是**随后 Decode 反复读取的 KV 数量**。这并不自动降低为了第一次处理长 Prompt 所需的 Attention 计算，也不能让远超过模型位置能力的 Prompt 突然变得完全可理解。

[NeurIPS 2024 SnapKV 正式摘要](https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html) 明确写出 16K Context 下最高 3.6× 生成速度与 8.2× 内存效率，以及 A100 80GB 上 380K Token 输入的演示。读作者针对每 Head 的 Figure 时要关心：保留位置是否在不同 Head 重合？Window $w$ 改变后能否覆盖答案相关段落？Needle Retrieval 的效果与 Long-Context QA 是否一致？不能只看“最大长度”这一数字。

## 原文实验结论为何要看平台与序列设置？

[NeurIPS 2024 正式论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html) 报告在 **16K Prompt** 条件下相对基线约 **3.6× 生成速度**、**8.2× 内存效率**，在 16 项长序列数据集上保持可比效果；并有单张 A100 80GB + HuggingFace 实现运行更长 Prompt 的演示。这里“可运行更长 Context”并不保证模型的训练/位置外推能力同步改善。

复现实验要固定 Observation Window 大小、每 Head 的 KV 容量、模型层数/头数和 Prompt 长度；比较等额缓存预算下的 Keep-Recent、[H2O](C036-h2o.md) 与 SnapKV。一定要同时测 long-context QA 或 Needle Retrieval，单看延迟不能发现关键证据被错误驱逐。

下一步 [KIVI](C038-kivi.md) 采取另一个独立轴：不根据重要性删除所有低分 Token，而对要保留的 Key/Value 使用不同维度的量化，减少每个元素的字节数。
