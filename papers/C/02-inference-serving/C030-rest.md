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

### 从原论文 §3.2 重建 Datastore 与 Trie 的真正排序规则

在 [REST NAACL 2024 正式论文 §3.2、Figure 1（印刷第 1584 页）](https://aclanthology.org/2024.naacl-long.88.pdf)，存储单元不是一条“输入问题到标准答案”的监督样本，而是 **Context–Continuation Pair**，记为

$$
D=\{(c_i,t_i)\}_{i=1}^M,\qquad
S(s)=\{t_i\mid (c_i,t_i)\in \operatorname{Retrieve}(D,s)\}.
$$

$c_i$ 是数据集中某个出现位置之前的 Token 上下文，$t_i$ 是紧随其后的 Token 片段；当前输入 $s$ 用来做后缀匹配。作者选择快速 **Exact Matching**，而不是推理时先加载一个 Embedding 模型做 Dense Retrieval，因为检索器本身的额外 Forward 就可能吃掉目标模型省下的计算。

Figure 1 的 Trie Node 不是纯粹的“有没有这条路径”。当多条召回片段共享某个前缀，节点会累计对应的**出现频率/权重**，后续按照节点频率进行剪枝，只保留预算内的高频支路。把这个机制漏掉，系统就可能在相同 Target Tree Budget 下把大量预算浪费在偶然出现的罕见候选上。

**手算例子**：召回结果是 §a b c§ 出现 4 次、§a b d§ 出现 2 次、§a e§ 出现 1 次。候选树的 $a$ 节点频率为 7，$a\to b$ 为 6，$a\to e$ 为 1；节点 §c§ 为 4，§d§ 为 2。如果最多允许保留 4 个非根节点，必须结合祖先闭包，不能只选频率最大的四个分散节点——否则拿到 §c§ 却没有 §a,b§ 就失去了合法的条件前缀。此处的频率计数是解释原理的教学数据，不是论文实际数据存储统计。

### 为什么命中数高，还可能跑得更慢？

设每轮树最多 $U$ 个候选节点，而真正从根开始连续确认的 Token 数为 $L$。若候选权重集中在某条非常常见、但与当前 Prompt 含义不符的句式上，可能 $U$ 很大而 $L=1$。在这种情况下，验证仍承担树 Attention 的计算量，而没有足够多的 Token 摊薄成本。

$$
\bar T_{\rm token}\simeq
\frac{T_{\rm lookup}+T_{\rm trie}(U)+T_{\rm target}(U)}
{\mathbb E[L]}.
$$

进一步还要把是否从 CPU RAM 读取索引、是否从 GPU 侧共享 Trie 结构、目标模型当前 Batch 的竞争考虑进去。REST 的静态语料存储并不是免费的模型外部记忆；其安全边界是**检索来的 Token 只作为候选**，不能直接拿库中的出现频率替代 Target 的概率分布。

### 对应原始 Figure 1 与实验的三个分离结论

原论文的 Figure 1（PDF 第 3 页，印刷 1584）描绘的是“Exact-Match Retrieval → 根据频率建 Trie 并剪枝 → Tree Mask 验证”。它说明实现路线；其中的数字代表 Trie Branch 出现次数，不是 Target 给每个 Token 的 Softmax 概率。论文 §4 另外给出了不同数据存储来源与模型的加速结果：CodeLlama/Code 数据、Vicuna/UltraChat 数据分别属于不同实验配置，不能用一组最高倍数证明任何语料都能形成同样的高价值 Trie。

如果要完整复现，除了 Token/s，还必须固定检索存储规模并报告 **后缀匹配长度分布、命中率、Trie 剪枝前后节点数、平均连续接受长度**。这些可观察量一一对应 REST 的“检索质量、结构压缩、Target 受益”三个机制，而不是单靠一句“检索很快”。

[NAACL 2024 原论文](https://aclanthology.org/2024.naacl-long.88/) 在 7B、13B 模型、batch=1 等实验下报告约 **1.62–2.36×** 的代码或文本生成加速，强调不额外训练草稿模型。这个范围不等于“任意语料、任意场景都能加速”：结构化代码更容易存在重复片段，创作性回答的局部延续可能缺少相似样本。

作者的 Figure 1 是系统数据流证据；为了判断速度因果，需要进一步检查检索存储的来源和规模、Trie 裁剪和 Target 运行基线。离线存储占用内存、文本索引更新和检索延迟，在高并发系统里不可视为零。

**实践建议**：对同一 Target 分别测检索耗时、索引命中率、可用于验证的路径数、平均接受长度和最终 Token/s；再与无外部存储的 [Lookahead](C029-lookahead-decoding.md) 对照。[Ouroboros](C031-ouroboros.md) 进一步研究如何把已有短语直接插入草稿生成过程。
