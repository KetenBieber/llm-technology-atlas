# REST：检索如何代替神经草稿模型？

> **原始论文**：[REST: Retrieval-Based Speculative Decoding](https://aclanthology.org/2024.naacl-long.88/)，NAACL 2024。**类型**：C 类推理机制与系统。 [原文 PDF](https://aclanthology.org/2024.naacl-long.88.pdf)。

前置：[推测采样的验证与校正](C020-speculative-decoding.md)。横向对照：[Lookahead Decoding](C029-lookahead-decoding.md)、[Ouroboros](C031-ouroboros.md)。

## “合适的草稿”一定要由模型生成吗？

当待生成内容经常沿着已有语料中出现过的局部模式继续时，神经网络逐 Token 预测不一定是获取候选的唯一办法。例如模型正在补全 Python 代码的“for i in”，大量代码片段里都有“range(”之类的延续。这些片段未必真是 Target 的下一步，却**足以作为低成本等待验证的候选**。

REST 的第一性出发点是将 Draft 网络完全替换为检索：使用已确认前缀作为 Query，在现有文档语料形成的数据存储中找相似连续片段，再把片段中后续的 Token 拿来作为猜测。它既不是微调大模型，也不是训练一个弱小的 Token 生成器。

~~~{mermaid}
flowchart LR
 P[已确认前缀的尾部] --> Q[检索最长匹配后缀]
 DB[已有语料数据存储] --> Q
 Q --> R[取匹配处的后续 Token]
 R --> T[合并成 Trie 候选树]
 T --> V[Target 树形验证]
 V --> P
~~~

*自制解释图，非原论文 Figure。[REST Figure 1（PDF 第 3 页）](https://aclanthology.org/2024.naacl-long.88.pdf) 展示 Query、检索结果、Trie 构建和树形验证四步。*

## 检索出的结果如何变成一棵候选树？

假设输入末尾恰好为某组 Token $[a,b]$，文档存储中有：

~~~text
... a b c d e ...
... a b c f ...
... a b g h ...
~~~

按 $[a,b]$ 匹配，取延续 $[c,d,e]$、$[c,f]$、$[g,h]$。如果全部逐条送 Target，候选 c 会重复计算；REST 把共享的 c 合并成一个节点：

~~~text
root
 ├─ c ┬─ d ─ e
 │    └─ f
 └─ g ─ h
~~~

这棵树对应路径集合 $\{cde,cf,gh\}$ ，其中 f 与 d 都是 c 的直接子节点。每个节点只能看到已确认前缀和自己的祖先 Token。查询召回多少结果、匹配后缀多长、树的分支与深度怎样裁剪，都会影响 Target 的验证成本。

如果检索不准确，Target 会很早否决候选；因此检索“召回率高”与“最终连续接受长度高”不是同一个指标。系统优化的对象是**检索时间 + 构树时间 + Target 验证时间**除以被确认的 Token 数，而非只最大化检索结果数量。

## 检索是候选提议，不能直接修改目标概率

设 Target 的下一 Token 分布为 $p(x\mid s)$，检索器返回的候选构成集合 $C(s)$。即使 $C$ 根据历史语料强烈偏好某 Token，最终输出仍应由 Target 在真实条件下判断。正确的树形贪心验证，或与目标采样分布兼容的严格随机校正，是方法不降低目标质量的基础。

例如 Token “range”在资料库里很频繁，并不意味着可以直接将其概率赋值为 0.9；REST 返回的是**候选**，不是 Token 目标分布估计。原论文 §3 的 trie pruning、候选选择与 Tree Attention 应连起来理解。与 [Lookahead](C029-lookahead-decoding.md) 的区别是：REST 有外部检索存储，Lookahead 用 Jacobi 迭代从正在计算的模型产生 n-gram。

## 原论文结果、边界和数据结构选择

[NAACL 2024 原论文](https://aclanthology.org/2024.naacl-long.88/) 在 7B、13B 模型、batch=1 等实验下报告约 **1.62–2.36×** 的代码或文本生成加速，强调不额外训练草稿模型。这个范围不等于“任意语料、任意场景都能加速”：结构化代码更容易存在重复片段，创作性回答的局部延续可能缺少相似样本。

作者的 Figure 1 是系统数据流证据；为了判断速度因果，需要进一步检查检索存储的来源和规模、Trie 裁剪和 Target 运行基线。离线存储占用内存、文本索引更新和检索延迟，在高并发系统里不可视为零。

**实践建议**：对同一 Target 分别测检索耗时、索引命中率、可用于验证的路径数、平均接受长度和最终 Token/s；再与无外部存储的 [Lookahead](C029-lookahead-decoding.md) 对照。[Ouroboros](C031-ouroboros.md) 进一步研究如何把已有短语直接插入草稿生成过程。
