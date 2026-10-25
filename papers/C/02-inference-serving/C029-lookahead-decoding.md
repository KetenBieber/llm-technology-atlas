# Lookahead Decoding：如何不训练草稿模型也能并行找后续 Token？

> **原始论文**：[Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://proceedings.mlr.press/v235/fu24a.html)，ICML 2024。**类型**：C 类推理机制与系统。 [原文 PDF](https://proceedings.mlr.press/v235/fu24a/fu24a.pdf)。

前置：[严格投机生成](C020-speculative-decoding.md)；下一篇：[REST](C030-rest.md) 与 [Ouroboros](C031-ouroboros.md)。

## 不训练 Draft，谁来预先猜未来？

普通自回归模型每次给出 $p(x_{t+1}\mid x_{\le t})$，似乎只能等待下一个 Token 真正生成后再向前。但现代 GPU 能一次处理多个 Token 的位置。如果额外的并行计算不太增加墙钟时间，是否可以用同一个 Target 同时尝试一批未来位置？Lookahead Decoding 的答案是：利用 **Jacobi 式并行猜测的迭代轨迹**，从中持续提取可验证的短 n-gram。

假设要填三个未知位置，先放任意候选 $z_1^{(0)},z_2^{(0)},z_3^{(0)}$。在下一轮，让模型同时更新每个位置：

$$
z_i^{(k+1)}=
\operatorname{argmax}_x
p(x\mid x_{\le t},z_{1:i-1}^{(k)}).
$$

这是解释贪心 Jacobi 更新的教学简写；注意每个位置在第 $k+1$ 轮读的是**上一轮**其左侧的猜测，所以一次计算可以并行。第一位置很快正确，随后正确前缀可能逐轮向右传播。并非每轮全窗口都是真实可提交的结果。

## 两条分支为什么必须协作？

~~~{mermaid}
flowchart LR
 P[已有确认前缀] --> L[Lookahead：Jacobi 窗口并行迭代]
 L --> N[缓存若干 n-gram]
 N --> Q[按前缀匹配取候选]
 Q --> V[Target 验证候选]
 V --> A[提交连续匹配部分]
 A --> P
~~~

*自制解释图。原论文的 Lookahead Branch 与 Verification Branch 通过一个组合的 Attention Mask 共同执行；[官方实现](https://github.com/hao-ai-lab/LookaheadDecoding) 展示窗口 W、n-gram 长度 N 与验证候选的组织方式。*

- **Lookahead Branch**：维护宽度为 $W$ 的未来位置窗口，并回看 $N$ 轮迭代结果，收集可能的 n-gram；它负责生产候选。
- **Verification Branch**：把 n-gram 的起始 Token 与当前已确认位置匹配，再用真实 Target 条件分布验证后续 Token；它负责消耗候选，并不能把所有 Jacobi 猜测直接当作输出。

举例某次 Jacobi 轨迹提取出“是 / 一个 / 例子”，当前真实前缀末尾出现“是”时可尝试后续两个 Token；但 Target 若验证第二个 Token 不吻合，则不能提交整段候选。检索的 n-gram 是机会，而非事实。

## 为什么叫“额外算 FLOPs 换更少的解码轮次”？

标准自回归一个 Token 做一次 Target Forward，其固定权重搬运成本很高；Lookahead 在一次 Forward 内增加可并行的 Token 位置，提高 GPU 利用率，同时期望缩短需要重复读取整套权重的轮数。可用

$$
T_{\rm total}\approx
\text{Forward 轮次}\times
\text{每轮真实延迟}
$$

建立直觉：候选窗口变大可能减少 Forward 轮次，却增大每轮计算；因此 $W,N$ 不是越大越好，GPU 算力、KV 占用和序列长度会改变平衡。

## 严格正确性的范围

Lookahead 没有独立 Draft，也没有外部检索文档，其候选从**同一模型迭代轨迹**中产生。论文对精确性保证建立在验证逻辑上：在原始 Target 的真实前缀条件下，只接受经检查的连续后继；不能把单轮 Jacobi 的所有位置无条件一起输出。具体贪心/采样实现须按原文不同实验协议解读，不要由“Exact”一词推断任何修改采样策略都自动等价。

## 原论文报告、工程限制

[ICML 2024 原文](https://proceedings.mlr.press/v235/fu24a.html) 报告 MT-Bench 等任务的单机加速最高约 **1.8×**，代码补全场景在多 GPU strong scaling 下最高约 **4×**；不同并行资源和 benchmark 不能直接对比。需要查看任务是否具备易重复 n-gram、可用的 GPU 算力冗余以及 Prompt 中相似短语的比例。

[REST](C030-rest.md) 不通过 Jacobi 生成短语，而从外部数据存储中检索；[Ouroboros](C031-ouroboros.md) 则利用短语池补充较轻的 Draft 生成流程。这三个技术名称都涉及 n-gram，却在**候选来源、是否有 Draft 网络、Token 验证时间分配**三个维度明显不同。
