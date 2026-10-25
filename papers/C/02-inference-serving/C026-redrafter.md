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

## 为什么 Recurrent State 比多个独立 Head 更容易表达长草稿？

采用形状明确的表示：Target 最后确认位置状态 $h_t\in\mathbb R^d$，草稿当前隐状态 $s_i\in\mathbb R^{d_s}$，Token Embedding $e(\hat x_i)\in\mathbb R^{d_e}$。一个教学版 Recurrent Transition 是

$$
s_{i+1}=\tanh\left(W_s s_i+W_e e(\hat x_i)+W_h h_t+b\right),
\quad W_s\in\mathbb R^{d_s\times d_s}.
$$

从 $s_{i+1}$ 投影至词表 $V$ 得到 $q_{i+1}=\mathrm{softmax}(U s_{i+1})$，其中 $U:[V,d_s]$。同一组循环参数 $W_s,W_e,W_h$ 跨草稿步共享，而 [Hydra](C025-hydra.md) 的各 Head 面向不同未来深度，这解释了两者结构上的差异。**这不是原文 RNN Cell 的逐字等式**；实际结构、输入融合与 KD 路线必须对照 [Apple 原始论文](https://arxiv.org/abs/2403.09919) 的 §3。

### Beam Search 与树合并必须分开

假设 Beam Search 返回长度均为 4 的三个候选：

~~~text
a b c d
a b c e
a f g h
~~~

逐条执行 Target 的候选位置数为 $3\times4=12$。以 Trie 合并后，唯一节点是 §a,b,c,d,e,f,g,h§，仅 8 个节点。理论候选节点减少 $1-8/12=1/3$，但每个 Token Node 必须携带正确的 Parent Index 与树 Mask。树里 §d§ 不能看到兄弟 §e§，§g§ 不能看到来自另一支的 §b§；若完全改成普通 Causal Mask，将使注意力读到非祖先 Token，验证的条件分布就错了。

更重要的是，Beam Search 最大化的是草稿概率或评分，**不是无条件确定 Target 采样结果**。如果使用随机采样，必须显式实现正确的 Tree Acceptance 与残差规则。Trie 去重只保证推理计算图的重复前缀减少，不保证严格采样分布。

## 从执行时序拆出真正的三个创新点

ReDrafter 将不同阶段交替执行：（1）Target 已确认前缀输出特征；（2）小循环 Drafter 在此条件下执行 Beam Search；（3）将重叠的 Beam Prefix 合并并组织 Dynamic Tree Attention；（4）Target 并行校验候选；（5）接受并推进至下一轮。这使一次 Target Forward 可以覆盖多个候选 Token，却也引入 $T_{\rm recurrent}+T_{\rm beam}+T_{\rm mask}$。

从硬件视角看，假设 Target 一次单 Token Forward 要 15 ms，而一轮草稿+构树合计 5 ms，树验证 18 ms；若平均确认 3 个 Token，平均成本为 $(5+18)/3\approx7.67$ ms/token，相对普通 15 ms/token 约 1.96 倍。若树候选质量下降使确认数仅 1，成本反而是 23 ms/token。这是教学成本模型，说明需要的指标是 **每轮 Target 提交数与额外草稿开销的联合分布**。

原论文 [Apple Research 摘要](https://machinelearning.apple.com/research/recurrent-drafter) 明确将 RNN、Dynamic Tree Attention、KD 列为三项驱动因素。不能仅从 2.8 倍的 H100 结果判断究竟是其中哪个模块贡献更多，应针对移除 Dynamic Tree、移除 KD 和不同 Beam Width 的消融测量。

## 原论文报告了什么，不能推出什么？

### 对照原论文 Figure 2、Figure 3 和真正的 RNN 更新式

在 [ReDrafter arXiv:2403.09919v5 第 2 页 Figure 2](https://arxiv.org/pdf/2403.09919)，作者展示了一个完整的执行轮次：LLM 先确定绿色 Token，Drafter 以它与 Target 最后一层隐藏状态作条件执行 Beam Search；LLM 对分支进行一次验证，蓝色的最长可接受前缀追加到输出。图中灰色词序列代表原始 LLM 顺序生成结果，用于对照输出一致性。这张图的证据作用是**定义执行语义**，不是 GPU 加速测量。

再看同一 PDF 第 3 页 **Figure 3**。作者画出前缀隐藏状态 $h$ 和每个候选 Token Embedding $e_t$ 如何被递归输入 Draft。与前文的教学式不同，论文 §3.1 给出了更精确的 RNN 关系：

$$
s_t=f(U s_{t-1}+W e_t+b),\qquad
g_t=[s_t,h].
$$

其中 $s_t$ 是共享参数的循环状态，$g_t$ 拼接了 Target 的 $h$，随后送入带 Skip Connection 的 MLP 和 Softmax 预测未来 Token。这说明“持续记住历史 Token”的部分由 $s_t$ 负责，而 $h$ 作为固定条件参与每个草稿步的输出 Head。把它写成 §tanh(W_s s_i+W_e e_i+W_h h)§ 只能是教学变种，不能误认为原论文 Eq.。

**Shape 检查**：若 $s_t\in\mathbb R^{d_s}$、$h\in\mathbb R^{d_h}$，则拼接向量 $g_t\in\mathbb R^{d_s+d_h}$；任何接收 $g_t$ 的首个线性层都应以 $d_s+d_h$ 为输入维度。若把 $h$ 不经投影直接加到 $s_t$，只有 $d_s=d_h$ 时才合法。由此也解释为什么必须区分“输入拼接”和“隐状态递推”两条算子路径。

### 由 Beam Trie 推导动态树 Attention Mask

设 Beam 候选为 §a-b-c§ 与 §a-d-e§，构树后节点为 $a,b,c,d,e$。若节点按照拓扑顺序排列，定义 $M_{ij}=0$ 当且仅当 $j$ 是 $i$ 的祖先或 $i=j$，其他位置为 $-\infty$。那么 $c$ 可访问 §a,b,c§，而不能读到 §d,e§；$e$ 可读 §a,d,e§ 而不可读 §b,c§。如果直接套“索引更早就能访问”的普通下三角 Mask，后一分支就可能偷看兄弟分支，Target 校验到的将是**不存在的自回归前缀**，即使输出 Token 恰好相同，也不再具备作者声称的验证语义。

注意这里的 Mask 规则是基于 Figure 2/3 的教学重建，树节点的物理 Packing 和 Dynamic Tree Attention 的执行实现还需结合原文 §3.3。

原论文 **Figure 1（第 1 页）** 才包含实测速度图：左侧是 H100/PyTorch，右侧是 M2 Ultra/MLX；对比 EAGLE、Medusa 和自回归，不能跨不同硬件直接比较柱高。Figure 2/3 解释机制，Figure 1 体现系统效果，三者承担完全不同的证据职责。

[Apple 研究介绍](https://machinelearning.apple.com/research/recurrent-drafter) 报告：NVIDIA H100 / PyTorch 上对 Vicuna 的 MT-Bench 实验最高 **2.8×** 加速；Apple Silicon / MLX/Metal 最高 **2.3×**。这两条对应不同平台和运行环境，不能把倍数互相比较成 GPU 性能排名。

应分别检验循环网络成本、Beam 数、树去重率、蒸馏对候选质量的影响。若希望复现所谓“更快”，还要报告 Target 验证一次花多久、每轮平均确认几枚 Token、草稿生成与树组织各自耗时；接受率与端到端延迟不能互换。

**关键区分**：[Hydra](C025-hydra.md) 通过顺序依赖 Heads 预测不同未来位置；ReDrafter 用一个可循环推进状态的候选网络并执行 Beam Search；[SpecInfer](C027-specinfer.md) 研究多分支共享前缀后，Target 究竟如何一次正确验证。
