# StreamingLLM：为什么滑动窗口丢掉开头 Token 会让注意力失稳？

> **原始论文**：[Efficient Streaming Language Models with Attention Sinks](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html)，ICLR 2024。 [官方/作者 PDF](https://arxiv.org/pdf/2309.17453)。


**研究入口**：[RoPE 的位置编码](../../A/03-transformer/A014-roformer-rope.md) → [PagedAttention 的 KV 管理](../01-distributed-training/C012-vllm-pagedattention.md)；下一篇 [H2O 的缓存驱逐](C036-h2o.md)。

## 如果只缓存最近 4096 个 Token，为什么模型会突然失常？

设已有上下文 $x_1,\ldots,x_t$，自回归 Decoder 每层存储历史 Token 对应的 Key 和 Value，以便生成 $x_{t+1}$ 时不重算旧隐状态。最直观的低内存方案是：容量 $W$ 满时删除最早的 Key/Value，仅保留 $x_{t-W+1:t}$。但原论文发现，简单 Window Attention 在长度跨过缓存阈值后，困惑度会急剧恶化。关键原因不是“没看到旧事实所以答不出旧问题”这么简单，而是某些很早的 Token 即使不承载任务信息，仍会积累显著的 Attention 权重。

对一个 Query $q_t$，标准自注意力概率为

$$
a_{t,j}=\frac{\exp(q_t^\top k_j/\sqrt{d_h})}
{\sum_{r\le t}\exp(q_t^\top k_r/\sqrt{d_h})}.
$$

当模型学到向首批 Token 分配一部分概率质量时，哪怕那些 Token 语义无关，硬删除也会改变归一化分母和其他位置相对权重。不同层/头的注意力分布不完全相同，不能理解成**所有头只关注第一枚 Token**。

## Attention Sink 并不等于“最重要的语义记忆”

一类 Token 被频繁 Attention，与“它能帮助回答当前问题”是两种不同性质。论文将不一定具有语义重要性、却稳定吸收注意力分数的初始 Token 称为 **Attention Sink**。直观地看，Softmax 归一化要求各位置概率之和为 1，Head 不想从已有内容提取有用信息时也仍需要把概率分配给某个位置；开头的稳定锚点可能承担这一角色。这是论文给出的经验分析与机制解释，不是“数学保证训练一定学出 Sink”。

### StreamingLLM 的缓存集合

若初始保留 $n_s$ 枚 Sink Token，最近窗口保留 $n_w$ 枚 Token，那么解码时使用的缓存位置集合可表达为

$$
S_t=\{1,\ldots,n_s\}\cup
\{\max(n_s+1,t-n_w+1),\ldots,t\}.
$$

因此 $|S_t|\le n_s+n_w$，在无限进行的流式生成中，**单次前向的 KV 缓存规模不再随全历史长度线性增长**。这并不表示模型能在第 400 万 Token 之后检索到第 100 万 Token 的具体事实：被逐出的语义信息已经不再可见。论文中的“无限序列”强调连续流式处理的稳定性，不等于无限有效推理上下文。

~~~{mermaid}
flowchart LR
 P[旧式全量缓存: 1..t] --> O[KV 随 t 增长]
 W[朴素滑窗: 仅最近 W 个] --> F[移除 Attention Sink，可能失稳]
 S[StreamingLLM] --> K[保留初始 Sink]
 S --> R[保留最近 Token]
 K --> V[有界缓存 + 稳定生成]
 R --> V
~~~

*自制机制图，非原论文 Figure。对应原论文 Figure 1 的 Full Attention、Window Attention、StreamingLLM 对照。*

## 为什么位置索引还要特殊处理？

在采用 RoPE 等相对位置相关编码时，只把旧 KV 从显存删除，而继续让实际逻辑位置无限递增，可能让仍可见的 Sink 与近期 Token 间隔越来越大，超出训练时位置分布。StreamingLLM 将**缓存中的 Token 位置重新映射为连续的逻辑位置**，以保留对训练中更熟悉的相对位置范围。这与延长 RoPE 可处理的真正完整文本长度不同：位置外推方法参考 [YaRN](../../A/03-transformer/A018-yarn.md)，StreamingLLM 的重点是受限缓存的稳定流式解码。

需要区分两种开销：初次 Prefill 一个超长 Prompt 仍有原本的计算和输入要求，StreamingLLM 主要优化持续生成中 KV 保留与注意力工作量。系统集成还需考虑 Attention Mask、跨层位置一致性及 KV 组织。

## 三种窗口方案究竟差在哪个计算量？

设每次模型实际可见窗口大小为 $W$，持续生成长度为 $T\gg W$。朴素窗口直接丢弃所有早期 KV，只保留最后 $W$ 个位置，**每次 Decode 可以是 $\mathcal O(W)$ 的 KV 读取**，但丢掉 Attention Sink 导致模型实际输出分布崩坏。

另一种思路是每前进一个 Token 就**重新 Prefill 最近 $W$ 枚 Token**，让原位置语义自洽，避免缓存前缀的错位状态；但单轮重算所有层时，每层还可能带来 $\mathcal O(W^2)$ 的 Attention 计算，整个长度 $T$ 的生成就需要重复执行 $T$ 次上下文计算。这也是作者用 Sliding Window with Re-computation 作速度参照时最重要的条件。

StreamingLLM 保留 $s$ 个最初 Sink 和 $W-s$ 个近期 Token 的 KV，并在解码中直接读取缓存。它**没有重算所有被丢弃 Token 的隐藏状态**。在固定 $W$ 下，每步只需处理大小受限的 KV；长序列持续运行的额外解码成本近似随 $T W$ 增长，而不是每次都生成一整段新窗口的 Key/Value。

### 当位置 0 的 Token 很远时，为何要重新编号？

以 RoPE 为例，某个通道的 Key、Query 采用旋转角度 $\omega_k m$、$\omega_k n$，两者内积带有相对角度 $\omega_k(m-n)$。如果最初 Token 的逻辑位置仍为 0，新生成 Token 的位置已到 100 万，Sink 与当前 Token 的角度差极大，超出训练时所见范围。

StreamingLLM 的重映射不是把整个 100 万长文本无损压缩到 4096，而是把**仍保留的缓存位置**分配给短的连续逻辑位置集合，使模型使用接近训练长度的位置差。这有助于减轻极端相对位置问题，但并不保证保留历史全文的语义检索能力。

## Figure 1 的四组对照恰好解释方法的两个目标

### 从一个 8-Token 缓存手算四种保留方式

假设当前已经读入 $t=20$ 枚 Token，Attention 预算最多只能缓存 $W=8$ 个位置。**全量 KV** 的长度为 20；**最近窗口**为 $\{13,\ldots,20\}$，开头位置 1 被清除；**StreamingLLM** 如果保留 $n_s=2$ 个 Sink 和最近 $n_w=6$ 个位置，则

$$
S_{20}=\{1,2\}\cup\{15,16,17,18,19,20\},
\qquad |S_{20}|=8.
$$

这两个长度为 8 的缓存占用一样，区别是**Softmax 归一化所能读到的 Key 集合不同**。当某些 Head 在后续几十步持续给 Token 1 非零且稳定的 Attention Mass 时，StreamingLLM 能保留这个概率锚点，而简单滑窗无法做到。此结论并非“Token 1 保存了所有旧文本语义”；例如 Token 8 的具体内容已经被驱逐，除非别的仍可见 Token 编码了相关信息，它不能凭空被完整恢复。

对单 Decoder 层、$H_{\rm kv}$ 个 KV Head、Head Dim 为 $d_h$、每个元素 $b$ Byte 的缓存，理论占用为

$$
M_{\rm KV,layer}=2H_{\rm kv}d_h b\,|S_t|.
$$

若 $H_{\rm kv}=8,d_h=128,b=2$，保留 8 个 Token 仅占该层 $32768$ Byte；全量保留 20 个则需 $81920$ Byte。这个例子把“有界缓存”和“语义检索仍可能丢失”分离为**两个独立判断维度**，不冒充论文任何一组真实实验设定。

[ICLR 2024 正式论文 Figure 1](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html) 分为 Dense Attention、Window Attention、Sliding Window with Re-computation 和 StreamingLLM 四种路径。必须同时看两列信息：**计算/缓存复杂度**与**长文本 PPL**。Dense Attention 保留所有位置但无法维持固定内存；普通 Window Attention 内存少但 PPL 失稳；重计算滑窗质量稳定但慢；StreamingLLM 才同时追求有界内存与稳定 PPL。

为理解其边界，可把上下文人为设为“最前面一条必须回复的账号代码，其后重复无意义内容十万 Token”。缓存最前 Sink 并不会保存所有被驱逐的中间信息。*稳定预测下一个自然语言 Token* 与 *能回答任意很久以前的问题* 是两种不同指标；后者需额外测长程 Retrieval，不由 Figure 1 单独证明。

## 原文证据及结果可适用到哪里？

[ICLR 2024 论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html) 报告 Llama-2、MPT、Falcon、Pythia 在流式语言建模中可稳定持续处理百万级 Token；相对滑窗重复计算基线最高 **22.2×** 的流式吞吐速度提升。这不是与优化过的全量 KV 连续 Decode 在所有 Batch 下比较的统一倍数。原文还验证训练时引入专用 Sink Token 的可能性，但“预训练额外 Sink Token”并非部署已有模型时必做前提。

阅读 Figure 1 时，先看训练长度和缓存长度，再看横轴延长后哪条曲线困惑度失稳；读系统实验时应找清楚比较的是从头重复计算滑窗还是缓存复用。对需要回溯远古文档事实的任务应另测 Needle Retrieval，不能以流式困惑度稳定推出历史知识不会丢失。

**延伸问题**：既然保留所有初始 Token 只是一个固定规则，能否直接统计哪些历史 Token 真正获得过高注意力，再动态驱逐低价值项？[H2O](C036-h2o.md) 给出另一种回答；[SnapKV](C037-snapkv.md) 则在 Prompt 尾部观察窗口中预测重要 Key。
