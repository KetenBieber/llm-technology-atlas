# Ouroboros：为何以短语为单位推进草稿更划算？

> **原始论文**：[Ouroboros: Generating Longer Drafts Phrase by Phrase for Faster Speculative Decoding](https://aclanthology.org/2024.emnlp-main.742/)，EMNLP 2024。**类型**：C 类推理机制与系统。 [原文 PDF](https://aclanthology.org/2024.emnlp-main.742.pdf)。

前置：[REST 的候选检索](C030-rest.md) / [Lookahead](C029-lookahead-decoding.md)；核心数学前置：[严格推测采样](C020-speculative-decoding.md)。

## Target 一次能验证多个 Token，Draft 却一枚枚生成——瓶颈转移了

普通推测解码每轮由小型 Draft 自回归生成 $\gamma$ 个 Token，然后 Target 一次并行验证。随着 Target Forward 次数减少，**Draft 自己连跑 $\gamma$ 步的成本**逐渐成为瓶颈。Ouroboros 认为既然文本经常包含可复用的短语，草稿阶段也可以试图按短语推进，而不总是一枚枚生成。

它同时利用已有对话、已生成上下文和候选短语池，并尝试增加草稿长度而不等比例增加小模型 Forward 数。与 [REST](C030-rest.md) 只依赖文本检索取代 Draft 模型不同，Ouroboros **保留一个小型草稿模型**，短语用于辅助该模型草稿过程。

~~~{mermaid}
flowchart TD
 P[目标模型确认的前缀] --> S[小草稿模型]
 H[历史对话 / 当前候选短语池] --> M[短语匹配与候选扩展]
 S --> M
 M --> D[更长的草稿序列]
 D --> V[Target 一次并行验证]
 V --> R[合规接受或校正]
 R --> H
 R --> P
~~~

*自制解释图。原论文 [Figure 2（PDF 第 3 页）](https://aclanthology.org/2024.emnlp-main.742.pdf) 用橙色标草稿模型 Forward：常规推测需要逐步生成一整条序列，Ouroboros 借短语并行构造和已有短语延长候选。*

## 从时延公式判断为什么更长草稿不一定更慢

设 $t_s$ 为小模型一次 Forward 时间、$t_T$ 为 Target 验证时间、$A(\gamma)$ 为长度 $\gamma$ 草稿被确认的平均 Token 数。忽略极小的调度项，普通推测的每轮时间近似为

$$
T_{\rm round}\approx\gamma t_s+t_T.
$$

若一次验证还能够额外确定下一个 Token，原论文的速度模型可以写成

$$
S_{\rm vanilla}\approx
\frac{(A(\gamma)+1)t_T}{\gamma t_s+t_T}.
$$

Ouroboros 设草稿过程的有效节约系数为 $c$，再通过短语延长草稿 $\beta$ 个 Token，则示意性对应为

$$
S_{\rm phrase}\approx
\frac{(A(\gamma+\beta)+1)t_T}
{(\gamma/c)t_s+t_T}.
$$

这个模型刻意**忽略检索与拼接时间**，是分析思路而非精确硬件运行式。只要 $\beta$ 增大不带来相应接受长度收益，Target 仍可能花更多时间验证无用 Token；若候选短语来自错误主题，速度甚至下降。

## 为什么不能直接复制短语当成最终结果？

检索到的短语只是候选，它们不一定属于当前任务的正确输出。如果被模型草稿阶段引用，也必须交给 Target 在对应祖先前缀下验证。对于贪心生成，不能提交与 Target 自身 argmax 不同的 Token；随机采样还必须严格处理 Target 与候选分布的概率校正。

因此“训练免费”在此指**不需要重新训练小模型与大模型**，不是“无计算成本”或“生成的文本无条件正确”。作者会从验证结果中继续复用可能有价值的短语——但哪些策略真实实现、短语池是否动态更新，应以论文第 3 节描述为准。

## 草稿阶段如何做到“少跑模型，却延长候选”？

标准 Draft 连续产生 $K$ 个 Token，需要小模型前向 $K$ 次（开启 KV 缓存后是逐步 Decode，不是每次重算所有历史）。Ouroboros 的核心是利用已有可用短语作为候选续写，使模型一次产生若干锚点 Token 后，可以**在不逐个走完整小模型 Forward 的情况下**插入更多候选位置。

设标准草稿每步用时 $t_s$，Target 验证为 $t_T$，产生 $K$ 枚草稿；期望接受的连续 Token 数为 $A(K)$。若每轮成功还可得到一个额外 Target Token，则平均代价近似

$$
t_{\rm vanilla}=\frac{K t_s+t_T}{A(K)+1}.
$$

如果有可复用短语，一轮只做 $m<K$ 次小模型推进，插入 $r$ 个短语 Token，并花 $t_{\rm phrase}$ 检索和组合，则

$$
t_{\rm phrase-mode}=\frac{m t_s+t_{\rm phrase}+t_T(K+r)}{A(K+r)+1}.
$$

此处 $t_T(K+r)$ 强调更长的验证序列可能增加 Target 计算，不能无条件设为常数。**少跑 $K-m$ 次草稿 Forward** 与 **多提交 $A(K+r)-A(K)$ 个 Token** 是两种可能同时发生、也可能相互抵消的收益。

### 一个反例：延长三枚候选并不一定有利

假设草稿已给 §def f (§，短语池里常见延续 §x ) :§；但当前真实代码是 §def f (self, x):§。如果短语在第二个候选 Token 就被 Target 否决，那么插入的后三枚候选不能帮助提交更多 Token，却增加了构造和验证成本。短语匹配命中率很高也不意味着 Target 认可的连续前缀更长；**短语池优化不能只用检索 Top-K Hit Rate 评价**。

## 原论文 Figure 2 的颜色和执行轴

[EMNLP 2024 Figure 2](https://aclanthology.org/2024.emnlp-main.742/) 的目的，是让每一次小模型 Forward 的数量和插入短语得到的草稿长度能够分别被观察。要追踪图中的草稿 Token、来自已有短语的 Token，以及 Target 大模型真正接受的区间，而不应把所有草稿方框当成最终输出。

正文 §3 讨论 Phrase Candidate Generation、Phrase Matching 与 Phrase Pool Management。若模型输出不断回流短语池，要考虑缓存增长和错误草稿被持续复用的问题；这些属于近似检索策略的系统代价，不是正确性保证。原文最高速度数据必须与相同的 Task、Batch、GPU、温度和草稿长度比较，否则“快 3.9×”没有可复现意义。

## 对照实验证据与版本差异

[EMNLP 2024 正式论文](https://aclanthology.org/2024.emnlp-main.742/) 摘要报告相比常规推测解码最高约 **2.4×**、相比普通解码最高约 **3.9×**，而早期 arXiv 版本的摘要对第一项指标可能出现不同值。网站引用优先使用正式 EMNLP 版本，避免混用版本数字。

研究者需要把收益拆成两项：缩短 Draft Forward 链的时间，以及因更长候选可能多提交的 Token。原论文 Figure 2 是结构解释，具体有效性应结合各任务的消融与实际生成时延。对于短句、陌生话题或候选短语池命中率低的场景，这两项收益可能显著缩小。

与 [Lookahead](C029-lookahead-decoding.md) 对照的关键是候选短语从哪里产生：Lookahead 的 Jacobi Window 本身耗费同一个 Target 的并行计算，Ouroboros 则主要让较轻量 Draft 的生成过程利用短语。和 [REST](C030-rest.md) 比，前者不是完全移除 Draft 网络。
