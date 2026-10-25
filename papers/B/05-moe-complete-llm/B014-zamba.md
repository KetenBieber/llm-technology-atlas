# Zamba：为什么把一组 Attention/MLP 权重重复调用 13 次，能得到一个有全局读取能力的紧凑语言模型？

> **原始报告**：Paolo Glorioso 等，*Zamba: A Compact 7B SSM Hybrid Model*，Zyphra，2024。[原文 arXiv:2405.16712 PDF（12 页）](https://arxiv.org/pdf/2405.16712) · [官方项目](https://www.zyphra.com/our-work/zamba) · [官方 Hugging Face 模型说明](https://huggingface.co/docs/transformers/model_doc/zamba)。
>
> **研究对象**：Zamba-7B-v1，约 7B 参数、1T Token 的双阶段预训练。本文不把后续 Zamba-2 或其它型号的层数、架构和结果倒灌入 v1。相关对照：[Mamba](../../A/09-state-space/A059-mamba.md)、[Mamba-2](../../A/09-state-space/A060-mamba2-ssd.md)、[Griffin](../../A/09-state-space/A064-griffin.md)、[Jamba](B013-jamba.md)。

## 问题从哪里来：Attention 可以只做很少次，却必须每次都有一套独立权重吗？

在常规 Decoder 中，若模型有 $L$ 个 Block，每个 Block 含独立 Attention 与 FFN，那么每加一层都会增加一套 $W_Q,W_K,W_V,W_O$ 以及 FFN Weight。纯 Mamba 不需要全局 Attention KV Cache，但它将历史压到状态后，某些精确检索及上下文学习能力可能受限。

已知可以混合 Attention 与状态模型后，出现两个完全不同的问题：

- **计算视角**：某个 Token 穿过多少次 Attention？这决定它能接受多少次“对历史位置重新按当前 Query 赋权”的处理。
- **参数视角**：这些 Attention 计算是否每次都要购买一份全新的 Weight？这决定模型可用参数容量及存储成本。

[Zamba Figure 2（原始 PDF 第 3 页）](https://arxiv.org/pdf/2405.16712) 的重要之处恰在于把这两个维度解耦：模型**保留多次 Attention 调用，但多处调用复用同一套 Weight**。这与简单减少 Attention 层数量的 [Jamba](B013-jamba.md) 不同。也与同一个 Transformer Block 的多轮迭代方法有关，但 Zamba 不是纯粹的通用循环 Transformer：大部分深度仍由**独立的 Mamba Block**组成。

## Figure 2 的真实结构：6 层 Mamba 不是 Attention 的残差附庸

原论文 Figure 2（第 3 页）画出三组连续的 6× Mamba，之间插入粉色共享 Attention/MLP。文字 §II 更准确地描述完整网络：

1. 基础 Backbone 是大量普通 Mamba Block，各个 Mamba Block 有自己独立参数和自己的层状态。
2. 每隔 $N$ 个 Mamba Block，插入一处 **Global Shared Attention（GSA）** 调用，包含 Self-Attention 和 MLP。Zamba-v1 实际采用每 6 个 Mamba Block 重用一次。
3. 多次 GSA 调用的 Attention Weight 和 MLP Weight **共享同一组参数**，但每次调用作用在**不同深度处的输入表示**上。
4. GSA 输出还要经过**每处调用专属的可学习线性映射**（原文说 un-shared learnable Linear），再送回主 Residual Stream。因此不是“模型中只有一处可学习的 Attention 输出投影”。

最容易读错的是：**共享权重 ≠ 共享激活 ≠ 共享 KV Cache**。不同调用层看到的是不同的隐藏表示 $x^{(\ell)}$，其 Key/Value 张量也是不同深度变换产生的。原论文 §IV 明确指出，Zamba 的这一个共享 Attention Block 在整体模型中执行 13 次，**每次调用都具有独立的 Activations 与 KV Cache Entries**。

因此 Zamba 的 KV Cache 并不是只存一层的 Key/Value，也不会因“权重文件只有一个”就压成一份缓存。它依旧随 Context 长度线性增长，只是相对于每层都有 Attention 的传统 Transformer，**Attention 调用层数所占比例较低**。

## 为什么每次 GSA 调用还要拼上最初的 Embedding？

普通纯 Mamba Residual 经过许多 Block 更新，状态表征倾向于保留当前任务有用的压缩信息，但**最初 Token 的可分辨细节不一定一直在同样容易检索的坐标中**。作者提供另一条较“原始”的信息路径：

$$
u_\ell=[x_\ell;x_0]\in\mathbb R^{2d},
$$

其中 $x_0$ 是最初的 Token Embedding，$x_\ell$ 是当前深度主 Residual；中括号表示在 Feature 轴 Concatenate，不是 Token 轴插入第二段文字。

GSA 对这个包含历史输入特征的组合做归一化、全局 Attention、MLP，再经 Linear Projection 回到适配残差流的维度。原文 Eq.(5)(6) 的结构化表达可以写作

$$
\begin{aligned}
z_\ell &= \operatorname{Linear}_\ell
\big(\operatorname{MLP}(\operatorname{LN}(
\operatorname{Attn}(\operatorname{LN}([x_\ell;x_0]))))\big),\\
x_{\ell+1}&=x_\ell+\operatorname{Mamba}\big(\operatorname{LN}(x_\ell+z_\ell)\big).
\end{aligned}
$$

这是为了讲解的一条**示意层级表达**，并不是所有 Norm/Residual 的逐算子复现代码；真实的 GSA 自身残差/MLP 处理需要以官方代码和原论文结构为准。特别注意 $z_\ell$ 的输出维度需要回到 $d$，否则无法与 $x_\ell$ 相加。

### 用一个 Shape 例子证伪“Concatenate 不增加成本”

假设 $B=2,T=1024,d=2048$。原本 Residual Tensor $x_\ell$ 是 $[2,1024,2048]$，原始 Embedding 同 Shape；Concatenate 后进入 GSA 的是 $[2,1024,4096]$。仅这个输入的 FP16 Payload 由约 8 MiB 变成 16 MiB，且 Q/K/V Projection 的输入维度也增加。

因此“共享 Attention 降低参数”与“每次 GSA 不增加任何 FLOPs”完全不同。Zamba 的设计是**在同样的总参数/模型 Weight Budget 下重复利用已付费权重，以额外的 Activation/Compute 换取更强的特征处理**。这一取舍需要从运行时 FLOPs 和质量两条轴同时衡量。

## 参数共享的第一性：同一矩阵在不同深度调用，表达是否等价于只调用一次？

设共享模块是 $F_\theta$，$G_1,G_2$ 是其前后两个独立 Mamba 子网络，分别有参数 $\phi_1,\phi_2$。两次调用得到

$$
y=G_2\big(F_\theta(G_1(F_\theta(x)))\big).
$$

与只调用一次

$$
\tilde y=G_2(G_1(F_\theta(x)))
$$

显然不同，除非这些函数满足特殊恒等条件。**参数共享只意味着多处反向梯度更新同一个 $\theta$，并不是多次调用产生完全相同的隐藏表示**。共享模块在不同深度接收到的是不同 $x_\ell$，每次的 Attention MatMul/Softmax 仍要真实执行。

假设单个 GSA 参数大小为 $P_A$，每调用一处额外的输出适配投影参数为 $P_O$，重复 $r$ 次时：

$$
P_{\rm shared}\approx P_A+rP_O,\qquad
P_{\rm independent}\approx r(P_A+P_O).
$$

差值 $(r-1)P_A$ 是理想意义上的参数节约。**执行 FLOPs** 却不是 $1/r$：$r$ 次共享 GSA 仍需要 $r$ 次相应 Attention/MLP 计算，并生成各自的 KV。模型通过增加相同参数的使用频率，改变了**参数使用效率与每 Token 计算量的关系**，不是压缩了所有运行成本。

### 为什么是同一模块共享，不是直接在所有 Mamba 层后面插入一个小 Attention Head？

增加一个轻量 Attention Head 并不等价于完整 GSA。原文 GSA 是 Attention + MLP，此外还有原始 Embedding 拼接和每次调用专属 Linear。它不仅为模型提供位置级检索，也借共享 FFN 多次处理来自不同深度的特征。

换句话说，Zamba 同时测试了两种归纳偏置：**重复使用同一套全局检索能力**、**通过不同深度的 Mamba 残差流改变检索上下文**。其目标不是节省一份 Attention Kernel 的启动开销，而是探索不同 Layer 能否共享所需的查询/特征变换。

~~~{mermaid}
flowchart TD
 E["最初 Embedding x0"] --> C1["concat [x_layer,x0]"]
 X["残差 x_layer"] --> C1
 C1 --> G["共享 GSA: Attention + MLP 参数 θ"]
 G --> L1["第 ℓ 处独立 Linear_ℓ"]
 L1 --> A1["回加 Residual"]
 A1 --> M["6 × 独立 Mamba Blocks"]
 M --> C2["下一处 concat [x_new,x0]"]
 E --> C2
 C2 --> G2["同一 GSA 参数 θ"]
 G2 --> L2["下一处独立 Linear"]
 L2 --> Y["下一轮 Residual"]
~~~

*自制结构解释图，非论文 Figure 2。两处 GSA 使用相同 $\theta$，但它们的输入、线性适配器和 KV Cache 不共享。*

## Figure 2 到 Eq.(5)(6) 的图文对应究竟在哪里？

[Figure 2（PDF 第 3 页）](https://arxiv.org/pdf/2405.16712) 每处 Shared Attention 旁边都画一条来自 Input Embedding 的跳跃线，接入带 “cat” 的节点；中间灰色 Shared Box 包含 Attention 与 MLP；每次 Shared Box 输出后单独写着 Linear，再返回蓝色 Mamba Block 的 Residual。这些关键标记恰好对应公式中的三个必须复原的结构：

- $[x_\ell;x_0]$：原始嵌入特征反复被访问，不是只有模型第一层能读取；
- $\theta$ 共享：GSA 权重重复使用，但作用在不同深度表征；
- $\operatorname{Linear}_\ell$ 不共享：可以学习当前调用位置特有的变换，以缓解把同一 Attention 模块强加给所有层造成的分布差异。

作者把这个方案类比为大脑皮层多个区域访问共享的海马记忆；这个比喻只能作为架构启发，**不构成模型具有真实人类认知记忆结构的实验或理论证据**。

## 两阶段预训练为什么不能被误认为“架构本身变强了”的证明？

[原报告 §III 与 Table I（PDF 第 5 页）](https://arxiv.org/pdf/2405.16712) 中，第一阶段使用开源 Web 文本，训练约 950B Tokens；随后做较短的 Annealing Phase，用更高质量的 Instruction/Synthetic 数据，并更快衰减学习率，合计约 1T Tokens。原文 Table I 列出数据组成，例如 RefinedWeb 约 62.35%、The Pile 15.48%、C4 12.85%、PeS2o 4.89%、Cosmopedia 3.55%、arXiv 0.90%。

因此最终的语言模型质量包含两个贡献：

1. 以 Mamba Backbone + Shared GSA 建立的**模型结构**；
2. 大量训练 Token 的内容选择、去重与高质量数据的 Annealing。

如果 [Table III（PDF 第 6 页）](https://arxiv.org/pdf/2405.16712) 中 Zamba Annealed 版本比 Phase-1 更好，**不能将这份增量直接归给 GSA 的共享机制**，因为架构未必改变，而训练数据与学习率分布已经改变。

### Table III 可直接复核的模型质量差异

| 同一 Zamba 7B | PIQA | HellaSwag | WinoGrande | MMLU（5-shot） |
|---|---:|---:|---:|---:|
| Phase 1 | 79.00 | 77.50 | 70.80 | 50.82 |
| Annealed | 81.37 | 83.60 | 76.40 | 57.72 |
| 增量（百分点） | +2.37 | +6.10 | +5.60 | +6.90 |

原文 Table III 的其他 Benchmark 多数是 Zero-Shot，只有 MMLU 是 5-Shot；所以“所有测试都是 0-shot”是不准确的。该表说明经 Phase-2 数据/优化后的整体模型表现改变，**不是单独的 GSA-on/GSA-off 控制实验**。

与外部 Llama2、Mistral、Gemma 等模型比较时同样需要留意 Token Budget、Data Mix 和是否经过 Annealing。作者公开模型在代码相关任务表现相对较弱，原文将其可能原因归于训练数据中 Code 比例不足；不能只凭结果就认定“共享注意力天生不会写代码”。

## Figure 3 与 Figure 4 的作用不同，不能把成功率和速度合成一个分数

[Figure 3（PDF 第 6 页）](https://arxiv.org/pdf/2405.16712) 是对若干**开源 Checkpoint 模型**的评估综合图，比较 Zamba Phase 1/Annealed 与 OLMo、Amber-7B、Pythia 等。图注明确 Aggregate Score 的组成是 PIQA、ARC-Easy、ARC-Challenge、HellaSwag、WinoGrande、BoolQ、OpenBookQA 的**均值**；它不是 PPL，也不是由 7 个任务的样本数加权得到的总准确率。

[Figure 4（PDF 第 7 页）](https://arxiv.org/pdf/2405.16712) 则是前向执行时间与 KV Cache Memory 的系统对照，在不同模型规模与上下文长度下比较推理成本。原文强调 Zamba 虽然共享 GSA 权重，**实际每层调用仍需要额外计算**；相对 Transformer 的收益更多来自较少 Attention 调用位置以及 Mamba 的长期状态压缩。因此 Figure 4 可以用来验证运行资源，不能替代 Figure 3 的任务质量。

作者原文 §IV 特别指出 Zamba 7B 有**13 次 Shared Attention 调用**，每次需要不同的 KV Cache Entries。若用最朴素的总量公式，设每次调用 K/V 都是 $H_{kv}$ Heads，每 Head $d_h$、每 Element $b$ Byte、历史长度为 $T$，则

$$
M_{\rm KV,total}=2\,T\,(\text{13 次调用})\,H_{kv}d_hb
$$

（此处忽略 Batch、Page Metadata 和实际 Head 配置）。Weight 仍然只有一份，不意味着 KV Cache 的 13 项可合并成 1 项。Mamba 各层的内部 Recurrent State 则另计，且不随 $T$ 线性增长。

## 研究上哪三个边界最值得保留？

**第一，权重共享不会让所有 Attention Layer 的功能相同。** 相同 $\theta$ 由于输入表征、Layer Adapter、Context 不同，产生不同的 Q/K/V 和输出，且反向传播的梯度来自所有调用位置。它是一种 Multi-Depth Parameter Tying，而非简单删除重复层。

**第二，多次调用 Attention 仍在产生大量历史状态。** 13 次调用分别保存 KV，不能把“单个共享 Attention Weight”误写成“单层 KV Cache”。与只保留四个 Attention Layer 的 [Jamba](B013-jamba.md) 对比时，若不对齐 Hidden Width、KV Head Dim、Layer Count、Dtype、Context，就不能由参数共享形式简单判断谁的 KV 更小。

**第三，模型得分必须与训练策略分开归因。** Zamba 的 1T Token、两阶段数据 Annealing 明显影响结果；若要检验共享 GSA 的单独价值，需要固定同一 Token Budget、优化配置和参数预算，分别比较“共享 GSA”“各层独立 Attention/MLP”和“纯 Mamba”，不能只拿最终大表中与其它公司模型的成绩替代消融。

**技术路线总结**：Griffin 先设计新的 **RG-LRU 遗忘单元**再加少量 Local Attention；Jamba 沿 **Mamba/Global Attention/MoE** 的三维组合空间权衡；Zamba 采用**独立 Mamba Backbone + 多深度共享的 Global Attention/MLP**，用更高的参数复用次数换取紧凑模型的表达力。三者都不是“Attention 被 SSM 证明淘汰”的证据，而是在回答不同的问题：**什么时候压缩历史，什么时候显式取回历史，以及如何在不同深度分配模型容量**。
