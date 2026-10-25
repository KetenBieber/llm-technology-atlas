# ReDrafter：循环草稿与动态树如何联合降低推理开销？

> **原始论文**：[Recurrent Drafter for Fast Speculative Decoding in Large Language Models](https://machinelearning.apple.com/research/recurrent-drafter)，Apple Research / arXiv:2403.09919。**类型**：C 类推理机制与系统。 [原文 PDF](https://arxiv.org/pdf/2403.09919)。

前置：[Hydra](C025-hydra.md) / [EAGLE](C022-eagle.md)，相关执行基础：[SpecInfer](C027-specinfer.md)。

## 为什么廉价的草稿也有可能拖慢推理？

Target 自回归每一步都可能受到权重读取带宽限制；推测采样希望用轻量 Draft 先生成多个 Token，再由 Target 并行确认。但是 Draft 仍需逐 Token 前向，若每次都执行完整的小 Transformer，多个 Draft Forward 的总成本很可能吞掉 Target 少调用几次节省的时间。

ReDrafter 的核心选择是**用循环草稿网络推进已生成状态**，并以 Target 已经计算出的隐藏表征作为条件。用教学模型写为

$$
s_{i+1}=F_\phi(s_i,E(\hat x_i),h_t),\qquad
q_{i+1}=\operatorname{softmax}(W_ds_{i+1}).
$$

其中 $h_t$ 是已确认前缀的 Target 状态，$s_i$ 是小型循环草稿的状态。公式强调“Target 特征作条件 + 新 Token + 循环状态”，不等同于声明论文只用了最普通的单层 $tanh$ RNN。

## 单条循环序列为什么还不够？

一个循环模型仍会犯错。如果只输出一条序列，Target 在前几个 Token 处拒绝就丢掉了其余候选；ReDrafter 利用 **Beam Search** 扩展多条替代路径，再把它们相同的前缀合并成树。

~~~{mermaid}
flowchart TD
 H[Target 前缀隐藏状态] --> R[循环草稿网络]
 R --> B[Beam 候选续写]
 B --> T[动态树：共享相同前缀]
 T --> V[Target 树形并行验证]
 V --> P[确认前缀，继续下一轮]
~~~

*自制机制图。原论文的三项联合设计是 Recurrent Drafter、Dynamic Tree Attention 和 Target 蒸馏；并非单纯把 Draft Transformer 换成 RNN。*

例如草稿产生 A-B-C、A-B-D、A-E-F 三条候选。逐路径需要 9 个候选位置，Trie 里仅 A、B、C、D、E、F 六个不同位置。一般地有

$$
U=\left|\bigcup_j \operatorname{prefixes}(\mathrm{beam}_j)\right|
\le \sum_j L_j.
$$

这是**减少重复候选节点数量**的组合学不等式，并不直接给出 GPU 时间加速，因为树形 Mask、Kernel、KV 索引也有执行开销。平行分支不能互相读取 Token，否则会破坏每个 Target 条件概率的语义。

## 蒸馏在这里到底监督什么？

ReDrafter 还使用来自 Target 的训练信号，让小循环网络在已确认前缀下给出更有用的概率。可将蒸馏动机概括为让 $q_\phi(\cdot\mid x_{\le t})$ 接近 $p_{\text{Target}}(\cdot\mid x_{\le t})$，例如常见的 KL 形式

$$
\mathcal L_{\rm KD}=
\mathbb E_{x_{\le t}}\!
D_{\rm KL}\left(p_{\rm Target}\,\|\,q_\phi\right).
$$

此处仅解释蒸馏目标，不把这个抽象式冒充作者完整的实际 Loss 或温度设定；精确训练配方需查阅 [原文](https://arxiv.org/abs/2403.09919) 的训练章节。由此可知 ReDrafter **不是**不训练的候选检索方法，[REST](C030-rest.md) 才是另一条路线。

## 原论文报告了什么，不能推出什么？

[Apple 研究介绍](https://machinelearning.apple.com/research/recurrent-drafter) 报告：NVIDIA H100 / PyTorch 上对 Vicuna 的 MT-Bench 实验最高 **2.8×** 加速；Apple Silicon / MLX/Metal 最高 **2.3×**。这两条对应不同平台和运行环境，不能把倍数互相比较成 GPU 性能排名。

应分别检验循环网络成本、Beam 数、树去重率、蒸馏对候选质量的影响。若希望复现所谓“更快”，还要报告 Target 验证一次花多久、每轮平均确认几枚 Token、草稿生成与树组织各自耗时；接受率与端到端延迟不能互换。

**关键区分**：[Hydra](C025-hydra.md) 通过顺序依赖 Heads 预测不同未来位置；ReDrafter 用一个可循环推进状态的候选网络并执行 Beam Search；[SpecInfer](C027-specinfer.md) 研究多分支共享前缀后，Target 究竟如何一次正确验证。
