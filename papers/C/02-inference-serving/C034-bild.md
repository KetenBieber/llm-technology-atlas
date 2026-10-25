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

## Fallback 不是“阈值越严格就越接近严格无损”

假设小模型对 A/B 给出的分布是 $q(A)=0.95,q(B)=0.05$，大模型为 $p(A)=0.55,p(B)=0.45$。若 Fallback 阈值 $\tau_{\rm fb}=0.9$，则小模型非常自信，§max q=0.95§ 不触发检查；但它与大模型的预测差异很大。**仅靠小模型自信度无法发现“自信错误”**。即使增加定期大模型回滚检查，只要偏差阈值或调用时机不是按精确接受–残差采样构造，通常也不能保持 Target 原采样分布。

反过来，如果令 $\tau_{\rm fb}\to1$，小模型每步几乎都触发大模型，可以逼近原模型质量，但系统又趋向普通大模型推理成本，失去 BiLD 的加速收益。阈值是延迟—质量的 **Pareto 控制量**，不是精确概率学定理里的可调拒绝界。

## 回滚后为什么必须丢弃后续 Token？

已确认前缀为 $s$，小模型输出 §a,b,c§，大模型发现 §b§ 应替换为 §d§。那么草稿 §c§ 是在 $s,a,b$ 下生成的，不能直接作为 $s,a,d$ 的合法后继保留下来。若直接保留，模型实际生成历史与用于产生 §c§ 的条件历史不一致。正确的回滚至少需要废弃位置 §b§ 及后面依赖它的所有 Token，并以更新的前缀重新推进。

若累计生成 $K$ 枚 Token、因 Rollback 保留的平均比例为 $\rho$，并设小模型成本为 $t_s$、大模型触发成本为 $t_L$，可写成教学成本式

$$
\bar t=\frac{K t_s+t_L}{\rho K}.
$$

例如 $K=10,t_s=1$ ms、$t_L=20$ ms、$\rho=0.5$，每个保留 Token 平均花 $(10+20)/5=6$ ms；若 $\rho=0.2$ 则为 15 ms。这里没有把重新生成时的额外上下文成本单独列出，真实系统还有 Cache/Mask 维护。这个例子说明**回滚太频繁会吞掉小模型的全部优势**。

### Figure 1 与速度表分别承担什么证据责任？

[NeurIPS 2023 官方论文](https://papers.neurips.cc/paper_files/paper/2023/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html) 中的 Figure 1 用时间轴对比普通大模型逐 Token 计算和 BiLD 的间歇性调用；它没有画出与严格分布采样方法一样的校正概率证明。论文的 Fallback/Rollback 阈值消融和翻译/摘要的任务指标，才支持“在允许某种精度退化时快多少”的结论。

因此比较 BiLD 与 [严格推测采样](C020-speculative-decoding.md)，至少固定目标任务、GPU、Batch、最终质量要求：前者的 Fallback/Rollback 阈值允许在质量—延迟曲线选择点；后者要求在同一目标采样定义下概率分布不变。若仅挑两篇最高倍数排列，实际上是在比较**不同约束下的优化问题**。

## 原始成绩必须携带质量代价

[NeurIPS 2023 正式论文](https://papers.neurips.cc/paper_files/paper/2023/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html) 的详细正文把最高 **2.12×** 明确放在允许约 1 个任务评分点下降的设置，且在另一种无质量降级实验设置报告最高约 **1.85×**。这比只引用摘要的“2.12×、质量下降很少”更能避免误解。实验使用 NVIDIA T4 的翻译与摘要任务，不可无条件推广到现代大规模 LLM。

这也是为什么 BiLD 虽然论文题目包含 Speculative Decoding，却应放在教程的**严格推测采样比较**里：它更接近一个阈值控制的大小模型协作策略。研究者可调节 $\tau_{\rm fb}$、$\tau_{\rm rb}$ 画出延迟—任务质量 Pareto 曲线，再与 C020 在目标分布不变约束下的延迟比较；若不对齐质量条件，直接按倍数排名并无意义。
