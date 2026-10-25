# BiLD：小模型前进、大模型回滚与严格无损推测的根本区别

> **原始论文**：[Speculative Decoding with Big Little Decoder](https://papers.neurips.cc/paper_files/paper/2023/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html)，NeurIPS 2023。**类型**：C 类推理机制与系统。 [原文 PDF](https://papers.nips.cc/paper_files/paper/2023/file/7b97adeafa1c51cf65263459ca9d0d7c-Paper-Conference.pdf)。

前置：[严格推测采样](C020-speculative-decoding.md)；横向对照：[Draft & Verify](C032-draft-and-verify.md)、[LayerSkip](C033-layerskip.md)。

## 小模型生成后，大模型只是“偶尔看看”能否达到严格无损？

BiLD（Big Little Decoder）用一个小模型快速自回归生成主要文本，大模型只在需要时检查并纠正。这与现代 Draft/Target 的形式相似，但其**正确性目标不同**：BiLD 主要权衡延迟和最终任务质量，不是一般性地数学保证输出分布精确等于每步都由大模型采样的分布。

[NeurIPS 2023 原论文](https://papers.neurips.cc/paper_files/paper/2023/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html) 用机器翻译、摘要任务展示这种策略。原文 [Figure 1（PDF 第 2 页）](https://papers.nips.cc/paper_files/paper/2023/file/7b97adeafa1c51cf65263459ca9d0d7c-Paper-Conference.pdf) 对照普通大模型逐步生成和 BiLD 的“小模型自回归 + 大模型并行检查”。

## 第一项策略：什么时候让大模型接管？

假设小模型在当前前缀下给出分布 $q_s(x\mid h)$。如果最大概率太低，说明其不够自信，触发 **Fallback**：

$$
\max_xq_s(x\mid h)<\tau_{\rm fb}
\quad\Rightarrow\quad
\text{让大模型生成下一 Token}.
$$

这比每一步都执行大模型省时，但“自信”不等于“正确”。如果小模型以 0.95 的概率自信地产生错误词，Fallback 就可能完全不启动。这是为什么还需要第二项策略。

## 第二项策略：大模型发现旧 Token 不匹配怎么办？

**Rollback**：大模型检查最近的小模型生成序列，找到第一个偏差超过阈值的位置，废弃那里及其之后的候选，再由大模型改写该处。

设 $d(p_L,q_s)$ 衡量大小模型的差异，定义

$$
m=\min\{i:\ d(p_{L,i},q_{s,i})>\tau_{\rm rb}\}.
$$

一旦存在这样的 $m$，丢弃 $m$ 之后由错误前缀生成的 Token，使用大模型在位置 $m$ 的决定重新推进。论文还讨论通过大小模型预测对齐改善策略的效率。

~~~{mermaid}
flowchart TD
 S[小模型自回归生成] --> C{最大概率是否低于阈值}
 C -->|是| L[调用大模型]
 C -->|否| S
 L --> D{已有序列是否出现高差异点}
 D -->|是| R[回滚到首次偏差处并修改]
 D -->|否| A[保留已生成前缀并继续]
 R --> S
 A --> S
~~~

*自制解释图。主要对照原论文 §3.3 Fallback、§3.4 Rollback、Algorithm 1；这里只表达分支判定，不代表完整实现中的每次大模型缓存复用细节。*

## 为什么这不是 C020 所证明的严格分布等价？

[严格推测采样](C020-speculative-decoding.md) 的随机候选接受–拒绝使用

$$
a(x)=\min(1,p_L(x)/q_s(x))
$$

并在拒绝后对剩余概率 $[p_L-q_s]_+$ 归一化采样。这是目标分布不变的数学条件。而 BiLD 的 Fallback 取决于 $\max q_s$ 和人工阈值，Rollback 取决于差异评分超过另一阈值。

作一个反例：词表仅 A/B，大模型 $p_L(A)=0.6$，小模型 $q_s(A)=0.9$。若当前小模型预测 A 的置信度很高且尚未触发 Rollback，BiLD 就更容易输出 A；一般不能等价于每步按照大模型 $p_L$ 采样。**“质量下降很少”与“随机生成分布严格一致”是两条不同命题。**

## 原始成绩必须携带质量代价

[NeurIPS 2023 正式论文](https://papers.neurips.cc/paper_files/paper/2023/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html) 的详细正文把最高 **2.12×** 明确放在允许约 1 个任务评分点下降的设置，且在另一种无质量降级实验设置报告最高约 **1.85×**。这比只引用摘要的“2.12×、质量下降很少”更能避免误解。实验使用 NVIDIA T4 的翻译与摘要任务，不可无条件推广到现代大规模 LLM。

这也是为什么 BiLD 虽然论文题目包含 Speculative Decoding，却应放在教程的**严格推测采样比较**里：它更接近一个阈值控制的大小模型协作策略。研究者可调节 $\tau_{\rm fb}$、$\tau_{\rm rb}$ 画出延迟—任务质量 Pareto 曲线，再与 C020 在目标分布不变约束下的延迟比较；若不对齐质量条件，直接按倍数排名并无意义。
