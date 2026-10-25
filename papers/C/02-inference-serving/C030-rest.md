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

## 一个检索表究竟怎样将文本前缀映射到候选？

假设历史存储包含四段 Token：

~~~text
a b c d e
x b c d y
a b f
a g c d
~~~

当前已确认前缀结尾是 §a,b§，对后缀 §a,b§ 作精确匹配可找到第一、第三行，候选延续为 §c,d,e§ 与 §f§。如果仅以 §b§ 为 Query，则也会命中第二行，增加 §c,d,y§ 的候选，检索召回上升，但很可能引入在不同语境里不合适的路径。这说明 **匹配后缀越短，召回越广、噪声越大**；越长则可能找不到匹配。它是检索的偏差—方差式折中，而不是调整 Neural Draft 的 Temperature。

检索的数据结构必须支持由最后几个 Token 查询存储的续接位置；可以用带边的 Trie、n-gram Index 或 Hash-based Prefix Map 存储。REST 的核心路径是 Retriever 返回具体 Token 序列，然后按树节点预算截断和合并，并交给 Target 验证。没有匹配时可退化为普通 Target 一步解码，不能无限等待数据库或凭空产生草稿。

### 用候选集合大小估算系统开销

若检索耗时为 $t_r$、构树为 $t_b$、Target 验证耗时为 $t_v$、每轮平均提交 $A$ 个 Token，则

$$
\bar t=\frac{t_r+t_b+t_v}{A}.
$$

假设原本一次 Target Decode 12 ms，而 REST 一轮检索+构树 3 ms、Target Tree Verification 16 ms、平均提交 2 Token，则每个 Token 平均 9.5 ms，速度约 1.26×。当资料库与当前任务域不匹配而平均提交仅 1 Token 时，平均 19 ms，反而比 12 ms 慢。这是教学估算；实际的 Trie/Mask 操作和 Warm Cache 也有时间。

## Figure 1 后面还要关注数据的来源

[NAACL 2024 原论文](https://aclanthology.org/2024.naacl-long.88/) 的 Figure 1 展示 Prefix Query → Retriever → Speculative Continuations → Token Tree Verification。它不是一个已经验证的加速因果实验。为检验方法应该比较：同一个 Target + 相同生成数据，在储存库只保留目标领域片段、混入大量域外片段以及完全无命中三种工况下的检索耗时和平均接受长度。

还要问一个工程问题：**索引更新从哪里来？** 如果上线后会不断把当前生成结果写回索引，可能引入请求之间的数据混用、隐私和质量污染；若只使用离线公开语料则没有这种回流，但覆盖新知识的能力有限。REST 论文的加速结果不能不附数据存储来源便推断到所有用户会话。

## 原论文结果、边界和数据结构选择

[NAACL 2024 原论文](https://aclanthology.org/2024.naacl-long.88/) 在 7B、13B 模型、batch=1 等实验下报告约 **1.62–2.36×** 的代码或文本生成加速，强调不额外训练草稿模型。这个范围不等于“任意语料、任意场景都能加速”：结构化代码更容易存在重复片段，创作性回答的局部延续可能缺少相似样本。

作者的 Figure 1 是系统数据流证据；为了判断速度因果，需要进一步检查检索存储的来源和规模、Trie 裁剪和 Target 运行基线。离线存储占用内存、文本索引更新和检索延迟，在高并发系统里不可视为零。

**实践建议**：对同一 Target 分别测检索耗时、索引命中率、可用于验证的路径数、平均接受长度和最终 Token/s；再与无外部存储的 [Lookahead](C029-lookahead-decoding.md) 对照。[Ouroboros](C031-ouroboros.md) 进一步研究如何把已有短语直接插入草稿生成过程。
