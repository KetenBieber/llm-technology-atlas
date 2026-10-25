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

## Jacobi 迭代为什么不等于直接并行生成多个正确 Token？

设一个玩具贪心语言模型规则：序列的三个未来位置应为 §A,B,C§，每个位置的正确 Token 取决于前一个位置是否已经正确。例如 $f_1(\text{前缀})=A$、$f_2(A)=B$、$f_3(A,B)=C$，但给错误的前缀时暂时输出 §X§。

初始并行候选为 $z^{(0)}=[X,X,X]$。第 1 轮更新每个位置时，使用的是**前一轮**的左侧输入：第一位置变为 A，第二和第三可能仍是 X，得到 §[A,X,X]§；第 2 轮可能为 §[A,B,X]§；第 3 轮变为 §[A,B,C]§。每轮的所有位置可并行执行，但**正确前缀仍沿位置轴逐轮传播**。如果把同一轮全部猜测直接提交，便违反自回归语义。

Lookahead 的关键是不用等整个 Jacobi Window 全部收敛，而是保留不同轮次、不同位置出现的 n-gram，将它们提供给 **Verification Branch** 尝试跳跃。这样在本轮第一个真正要生成的 Token 尚未全部确定时，旧轨迹里已经可能藏有一些未来会有用的续接片段。

## 真正的并行执行怎么维持因果历史？

如果一个窗口宽 $W=4$，回看 $N=3$ 轮 Jacobi 轨迹，那么候选张量概念上带有“迭代轮数 × 未来位置”两个索引。用于验证的候选有另一条轴：已确认 Token 前缀 + 若干待验证位置。**两条支路虽然同一次大模型前向计算，但并不能无条件共享整张 Attention 上三角矩阵**。论文把 Lookahead、Verification 位置交错排列，用定制因果 Mask 保证候选只依赖其合法前缀。

若采用 $W$、$N$ 更大，单 Forward 会计算更多 Token 位置；其目标不是降低总 FLOPs，而是在 GPU 空闲算力仍较大、参数权重读入占主导时，增加一点 Arithmetic Work 以换取更少的 Decoder Weight Loads。因此应同时报告 $T_{\rm forward}(W,N)$ 与实际减少的 Forward 轮次，而不是只看候选个数。

**实验判断**：[ICML 2024 正式论文](https://proceedings.mlr.press/v235/fu24a.html) 的单卡 MT-Bench 与多卡 Code Completion 强扩展是不同设置。其最高 1.8× / 4× 都有相应 GPU、任务与窗口参数。若模型运行在已经 Saturated 的大 Batch GEMM，额外 Jacobi 计算可能比减少的一次权重读入更贵；这正好反驳“算得越多一定越快”的误解。

## 原论文报告、工程限制

[ICML 2024 原文](https://proceedings.mlr.press/v235/fu24a.html) 报告 MT-Bench 等任务的单机加速最高约 **1.8×**，代码补全场景在多 GPU strong scaling 下最高约 **4×**；不同并行资源和 benchmark 不能直接对比。需要查看任务是否具备易重复 n-gram、可用的 GPU 算力冗余以及 Prompt 中相似短语的比例。

[REST](C030-rest.md) 不通过 Jacobi 生成短语，而从外部数据存储中检索；[Ouroboros](C031-ouroboros.md) 则利用短语池补充较轻的 Draft 生成流程。这三个技术名称都涉及 n-gram，却在**候选来源、是否有 Draft 网络、Token 验证时间分配**三个维度明显不同。
