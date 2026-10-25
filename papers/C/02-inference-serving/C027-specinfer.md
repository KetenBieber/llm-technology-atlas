# SpecInfer：从单条草稿验证到可共享前缀的 Token Tree

> **原始论文**：[SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification](https://arxiv.org/abs/2305.09781)，ASPLOS 2024。**类型**：C 类推理机制与系统。 [原文 PDF](https://arxiv.org/pdf/2305.09781)。

前置：[严格推测采样](C020-speculative-decoding.md) / [ReDrafter](C026-redrafter.md)，后续：[Sequoia](C028-sequoia.md)。

## 为什么不是把所有候选序列组成一个普通 Batch？

传统 [Speculative Decoding](C020-speculative-decoding.md) 在一个 Target Forward 里验证某条 Draft 序列。但如果有多条共享历史的分支，分别送 Target 会重复计算相同 Token。SpecInfer 不只是增加候选的数量，而是改变 **Target 的执行对象**：把多条候选重构成共享前缀的 Token Tree，利用一次 Tree-Based Parallel Decoding 计算各节点条件概率。

~~~{mermaid}
flowchart LR
 S[小型推测模型] --> C[多条候选]
 C --> T[Trie 合并共享前缀]
 T --> M[构造祖先可见 Attention Mask]
 M --> V[Target 并行验证各节点]
 V --> A[符合目标规则的前缀提交]
~~~

*自制解释图。原文 [Figure 1（PDF 第 3 页）](https://arxiv.org/pdf/2305.09781) 分别画增量解码、序列推测和树形推测的时间线，并不是“同样计算量下任意场景恒定倍数加速”的证明。*

假设有两条候选 A→B 和 A→C。朴素分别验证包含 A、B、A、C 四个位置；共享树仅 A、B、C 三个位置。对于 B，允许注意力读取真实前缀、A 和 B；对于 C，只能读取真实前缀、A 和 C。

按节点 $u_1=A,u_2=B,u_3=C$ 排列，忽略真实 Prompt，候选内部允许访问矩阵为

$$
\begin{pmatrix}
1&0&0\\
1&1&0\\
1&0&1
\end{pmatrix}.
$$

其中 1 是允许访问的布尔状态，非 Softmax 权重。对任意两个节点，Mask 可表示为

$$
M_{ij}=\begin{cases}
0,&u_j\in\operatorname{Ancestors}(u_i)\cup\{u_i\},\\
-\infty,&\text{其他}.
\end{cases}
$$

**科学与工程上最重要的性质**：每个分支在 Target 中看到的 Token 历史，必须等于单独执行该分支时看到的历史。若平行节点彼此可见，便不再是同一个 Target 条件分布；若相同前缀被重复计算，则失去 Token Tree 的计算复用价值。

## 树的来源与树的验证是两个独立模块

SpecInfer 从较小的推测模型创建候选，通过 **expansion 与 merge** 利用同一个或多个小模型生成不同路径的多样性。候选质量的好坏决定树能有多长的有效前缀；而怎样执行树形 Attention 解决的是 Target 计算冗余。这两个问题不能只用“更高接受率”概括。

$K$ 条长度为 $L_i$ 的候选若单独验证，需要 $\sum_i L_i$ 个候选位置。Trie 的实际不同节点数 $U$ 满足

$$
\max_iL_i\le U\le\sum_iL_i.
$$

完全重合的候选共享最多；互不相同的候选计算复用少。越大、越广的树并不必然高效，因为 Target Tree Forward 与 KV/Mask 准备开销也会增加。

## 为什么“条件分布算对了”还不是“采样无损”？

贪心路径验证只需沿 Target 自身 $\arg\max$ 连续匹配；随机采样却必须正确处理概率质量。SpecInfer 提出针对树形候选的验证/采样方案，以保留与原 Target 增量解码一致的输出分布。树形 Mask 保证 **$p(x\mid\text{祖先})$ 的条件输入正确**，而正确的接受–残差规则保证 **按这个 $p$ 进行采样**；两者缺一不可。

单链版本的接受概率可以写成

$$
a(x)=\min(1,p(x)/q(x)).
$$

但若有多个子节点，不能无条件把所有兄弟节点独立套这条式子；还涉及候选是否重复、抽样顺序和剩余分布的正确更新。实现应直接依据论文 Tree Verification 算法，或参考 [标准证明](C020-speculative-decoding.md)，而非靠 Mask 图猜测采样算法。

## 原始实验证据

[ASPLOS 2024 原论文](https://arxiv.org/pdf/2305.09781) 报告分布式推理服务相对既有系统 **1.5–2.8×**、参数卸载推理 **2.6–3.5×**。这些是特定部署与基线的系统指标，不能转写成“树形方法减少 3.5 倍 FLOPs”。论文 [Figure 1](https://arxiv.org/pdf/2305.09781) 展示执行时间线；性能数字须回到测试模型、设备配置、候选策略和对照系统上解释。

要独立复现，至少测总候选位置数、共享树节点数 $U$、Mask 构造时延、Target Forward 时延、可确认 Token 数及随机采样分布一致性。只记录树节点变少，无法排除小 Batch/kernel overhead 抵消收益的情况。

**技术定位**：SpecInfer 解决**如何对树正确并行验证**；[Sequoia](C028-sequoia.md) 研究有限预算里怎样选择树的形状；[EAGLE-2](C023-eagle2.md) 进一步研究树怎样随当前上下文调整。
