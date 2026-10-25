# Jamba：Mamba 负责压缩历史，Attention 负责显式读取，MoE 又为什么必须是第三个独立维度？

> **原始模型报告**：Opher Lieber 等，*Jamba: A Hybrid Transformer-Mamba Language Model*（AI21 Labs，arXiv:2403.19887v2，2024）。[原始报告 PDF](https://arxiv.org/pdf/2403.19887) · [原始 Jamba-v0.1 权重](https://huggingface.co/ai21labs/Jamba-v0.1)。
>
> **版本注意**：本文解读 2024 年开源的 **Jamba-v0.1 Base，52B 总参数、12B 活跃参数、1:7 Attention:Mamba**。2025 年 ICLR 正式发表版本标题为 [Jamba: Hybrid Transformer-Mamba Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a9ed43fa31dc8b4a7d7a673d713dcb5f-Abstract-Conference.html)，报告 Jamba-1.5 另有不同模型规模和训练配方；**本文的 Table/Figure 均以原始 arXiv v2 为准，不把两个版本的指标混写**。
>
> **前置**：[Transformer](../../A/03-transformer/A013-transformer-attention-is-all-you-need.md)、[Mamba](../../A/09-state-space/A059-mamba.md)、[Sparsely-Gated MoE](../../A/05-moe/A032-sparsely-gated-moe.md)；比较 [Griffin](../../A/09-state-space/A064-griffin.md) 的局部注意力/新递推单元与 [Zamba](B014-zamba.md) 的共享 Attention。

## 从完整模型开始：Jamba 不是一个新的 Mamba Cell

纯 Transformer 与纯 Mamba 分别体现两种历史读取哲学。前者把每一枚 Token 的 Key/Value 留在历史缓存，未来 Query 可显式重新打分；后者依靠输入相关的动态 SSM，把过去压缩成固定大小的隐状态，Decode 时无须为每个 Token 生成新的 KV 缓存项。

两者各自有明确局限：

- 全局 Attention 能基于当前 Query 选择精确位置，但 KV Cache 随输入长度 $T$ 增长。
- Mamba 的状态尺寸与 $T$ 无关，但有限状态里的内容会聚合/干扰，某些精确检索和 In-Context Learning 能力可能不如 Full Attention。
- 只把 Attention 变少不一定有足够的模型容量；只靠增加 Dense FFN 参数又会增加每次 Token 推理所激活的计算量。

Jamba 的选择不是另外发明一套 SSM 方程，而是用**三个独立可调的轴**建立混合 Decoder：**哪几层执行序列混合用 Mamba/Attention、哪几层 FFN 使用 MoE、每次 Token 路由到几个 Expert**。这份技术报告的独创性在**大规模组合架构与量化实验取舍**，而不是把作者用到的 Mamba、MoE 称为 Jamba 首创。

## Figure 1 的一条 8-Layer 路径代表什么？

[原始 v2 Figure 1(a)(b)，PDF 第 3 页](https://arxiv.org/pdf/2403.19887) 清楚分开：

- 左图 (a) 是一个 **Jamba Block**，由 $l=8$ 层组成：**1 个 Transformer Temporal Mixer + 7 个 Mamba Temporal Mixer**。
- 图中每层仍有一个 FFN 子层；每隔 $e=2$ 层把 Dense MLP 换成 MoE。因为两项选择互相独立，所以可以有 Attention+MLP、Attention+MoE、Mamba+MLP、Mamba+MoE **四种 Layer**。
- 右图 (b) 把这四类 Layer 的 Pre-Norm、Residual、Sequence Mixer 和 FFN 画清楚。也就是说，**MoE 不负责跨 Token 的 Attention**；它在 Token 的 Channel Mixing 阶段按 Router 分配专家。

假设示意 Block 的前 8 层从下到上：

| 层号 | Temporal Mixing | Feedforward |
|---|---|---|
| 1 | Mamba | Dense MLP |
| 2 | Mamba | MoE |
| 3 | Mamba | Dense MLP |
| 4 | Mamba | MoE |
| 5 | Transformer Attention | Dense MLP |
| 6 | Mamba | MoE |
| 7 | Mamba | Dense MLP |
| 8 | Mamba | MoE |

具体 Attention 与 MoE 的位置按原论文 Figure 1 排列，表格强调两条选择轴可正交组合。**不能把“每 8 层一次 Attention”理解为“每层的 1/8 Head 使用 Attention”**：前者是层级执行频率，后者是完全不同的混合架构。

~~~{mermaid}
flowchart TD
 X["Block 输入 X:[B,T,d]"] --> L1["Layer1：Mamba + MLP"]
 L1 --> L2["Layer2：Mamba + MoE"]
 L2 --> L3["Layer3：Mamba + MLP"]
 L3 --> L4["Layer4：Attention + MoE"]
 L4 --> L5["Layer5：Mamba + MLP"]
 L5 --> L6["Layer6：Mamba + MoE"]
 L6 --> L7["Layer7：Mamba + MLP"]
 L7 --> L8["Layer8：Mamba + MoE"]
 L8 --> Y["下一 Block / 输出"]
~~~

*自制组合示意，非原论文 Figure；特意分开 Mamba/Attention 的序列混合轴与 MLP/MoE 的通道混合轴。*

## 为什么说 Attention 比例、MoE 密度和专家数量是三种不同的预算？

[原始 v2 §2、PDF 第 4 页](https://arxiv.org/pdf/2403.19887) 给出的架构自由度为

$$
(l,\ a:m,\ e,\ n,\ K),
$$

分别是：Block 层数；Attention/Mamba 层比；每隔多少层应用 MoE；每个 MoE Layer 的可选 Expert 数；每 Token 选取几个活跃 Expert。任何一个变量改变，都可能同时影响 Model Quality 与执行预算，但影响路径不同。

**Attention 比例 $a:m$** 直接决定需要保存多少份全局 KV Cache。Mamba 层依旧需要自己的 SSM State，但是其尺寸不随上下文历史长度增长。

**MoE 密度 $1/e$** 决定了多少个 FFN 属于稀疏专家层。MoE 层中的 $n$ 是**总可选专家数**，$K$ 是每个 Token 的**激活专家数**。

设一个 FFN Expert 有 $p$ 个参数，某层有 $n=16$ 个专家，每 Token 只访问 $K=2$ 个：

$$
P_{\text{available,experts}}=16p,\quad
P_{\text{active,experts/token}}\approx2p.
$$

如果相同前向使用均匀 Router、忽略 Gate/Norm 的小开销，FFN 部分每 Token 的主计算量随 2 个活跃专家而非 16 个专家增长。但 **16 个 Expert Weight 仍要存储在总设备内存/网络中**。因此“Jamba 12B Active、52B Total”不是两个模型规模，也不能将 52B 全部乘进每步 FFN FLOPs。

注意 Top-$K$ 专家同时输出，按 Router 权重加权组合；这并不等价于“选择训练数据子集”，也不等价于 Attention 的历史 Key 检索。

## 1:7 的 Attention/Mamba 比例为什么会改变 KV Cache 的增长系数？

如果纯 Decoder 有 $L$ 个 Attention Layer、Batch Size 为 $B$、序列长度为 $T$，每层存 $H_{kv}$ 个 Key/Value Head、每 Head Dim 为 $d_h$、每元素 $b$ 字节：

$$
M_{\rm KV}=2\,B\,T\,L\,H_{kv}\,d_h\,b.
$$

若替换其中 $7/8$ 的 Attention Layer 为 Mamba，保持余下 Attention Layer 的 Head、Dim、Dtype 不变，KV Cache 的理想变动为

$$
M_{\rm KV,hybrid}\approx
2BT\frac{L}{8}H_{kv}d_hb
=\frac18M_{\rm KV,all-attn}.
$$

这是**条件性的架构算式**，不是整个模型显存缩小 8 倍。Mamba 的递推状态、Conv Buffer、专家权重、Activation 与 System Workspace 仍占内存；且基线纯 Transformer 可能使用不同 GQA/MQA 头数。

### 一个可复算的 256K Context 例子

设 $B=1,T=262144,L=32,H_{kv}=8,d_h=128,b=2$。每个 Attention Layer 的缓存：

$$
2\times262144\times8\times128\times2
=1{,}073{,}741{,}824\text{ Byte}=1\text{ GiB}.
$$

所以 32 个全局 Attention Layer 为 32 GiB；仅 4 个 Attention Layer 为 4 GiB。这与原始 Jamba 的 Figure/Table 报告同量级，但这组 Shape 是用于解释的**假设例子**，不是直接引用 Jamba 每个 Layer 的实际头数配置。

**原始 Table 1（PDF 第 4 页）** 明确报告相同 256K、16-bit KV 条件下 Jamba 4GB、Mistral 32GB、Mixtral 32GB、Llama-2 128GB。**不能把这张表当成模型结构相同、只有 Attention 层数量变化的控制实验**：其它模型的 GQA、KV Head 数、Layer 数和 Width 并不相同。

## 为什么只保留少量 Attention 并不是“SSM 已经学到了全部历史细节”？

一条 Mamba 递推

$$
h_t=A_t h_{t-1}+B_t x_t,\qquad y_t=C_t h_t
$$

把历史变为固定大小 $h_t$。若训练任务是“某个特殊事件发生后持续跟踪状态”，它很适合压缩；若任务是“在二十万 Token 以前找出某个随机 Key 的 Value”，纯 SSM 必须将许多独立信息编码到有限精度状态中，可能出现容量冲突。

在 Jamba 的混合架构中，某个 Attention Layer 仍能存储该层对每个历史 Token 的 KV，并在未来 Query 中按内容重新排序，**而其它 Mamba Layer 可以先提取与任务有关的特征或顺序信息**。Attention 提供一条显式位置级读取通道，Mamba 提供经选择的状态传播。因而两种计算机制互补；它们不是可以相互替换的同一个矩阵算子。

但要避免过度叙述：即使有少量全局 Attention，模型也**不保证**在所有实际检索任务里无损找回任意 Token。能否检索取决于 Key/Value 表征、位置处理、训练目标和 Input Distribution。

## 哪组原始实验真正比较了“混合”相对于两个纯模型的收益？

[原论文 Table 4 与 Figure 6（PDF 第 9–10 页）](https://arxiv.org/pdf/2403.19887) 是架构判断的核心证据，不是系统吞吐图。作者固定 1.3B 模型规模、250B Training Tokens，分别训练 Pure Attention、Pure Mamba、Attention/Mamba=1:3 与 1:7 的混合模型，**全部不使用 MoE**。

| 架构 | HellaSwag | WinoGrande | Open LLM | NQ |
|---|---:|---:|---:|---:|
| Pure Attention | 62.4 | 59.6 | 36.4 | 14.5 |
| Pure Mamba | 62.6 | 59.4 | 36.1 | 14.5 |
| Hybrid 1:3 | **65.1** | **61.7** | **37.2** | **16.5** |
| Hybrid 1:7 | **65.1** | **61.7** | **37.2** | 16.0 |

Figure 6 的横轴是 Training Steps（至约 60K）、纵轴是 Loss，曲线分别为两纯架构与两种混合比例；混合曲线在这组实验中 Loss 较低，而 **1:3 与 1:7 差异不显著**。作者因 1:7 的成本收益选择它作为大模型结构；这并不意味着任何 Attention/Mamba 比例在所有任务都几乎一样，更不能因此声称 Attention 已经没用。

随后 [Table 5（PDF 第 10 页）](https://arxiv.org/pdf/2403.19887) 对 7B、50B Training Tokens 的 **无 MoE 模型**再次比较三种架构，混合模型在几项学术任务优于两个单一骨干。这相当于跨两个模型规模/训练预算的重复证据，仍不代表无限扩展性定理。

## 那 MoE 的收益是如何与混合骨干区分开的？

[原论文 Table 7（PDF 第 12 页）](https://arxiv.org/pdf/2403.19887) 单独比较同一 Attention/Mamba Hybrid Backbone 的 **无 MoE** 与 **有 MoE**：

| 同一类混合架构 | OLLM | HellaSwag | WinoGrande | NQ |
|---|---:|---:|---:|---:|
| No MoE | 36.6 | 62.5 | 58.8 | 15.4 |
| With MoE | 38.1 | 66.0 | 61.2 | 18.9 |

这说明增大可选专家容量会进一步改善对应任务；但 MoE 版本的**总可用参数量也更大**，它不是一个“增加路由层的开销为零”的等参数因果试验。不要将 Table 7 的收益归到 Attention 比例，也不要将 Figure 6 无 MoE 的架构结论与表格差值混成同一个实验。

## 位置编码的反直觉发现：有 Mamba 是否就不需要 RoPE？

作者在 [Table 8（PDF 第 13 页）](https://arxiv.org/pdf/2403.19887) 上比较 1.3B、250B Token 的 Hybrid+MoE，有/无 Attention RoPE 时很多指标相近、少量指标各有升降。作者提出一种**可能的解释**：Mamba 层在 Attention 之前已将顺序特征编码进表示，后续 Attention 可以依靠隐含的位置信息。

这不能变成“Attention 数学上不需要位置编码”的普遍结论。两个模型的归纳偏置可能提供部分位置信号，但需要改变 Layer Ordering、Mamba Depth、任务依赖和长度外推设置，才能检验这一解释。论文报告的是特定实验观察，不是严格定理。

### 稳定训练是第三个有证据的架构关口

原论文 §6.4 与 Figure 9 显示，把混合模型扩展到 12B Active/52B Total 时遇到 Loss Spikes，定位到 Mamba 内部激活幅值较大。作者在这些位置加入 RMSNorm 后训练变稳定。这里新增 Norm **改变了真正实现的 Block**，不能在复现时直接用原始纯 Mamba Definition，却声称实现完整的 Jamba-v0.1。

这些细节说明混合架构不是“把网络配置文件的 Attention 换成 Mamba 就能自动训练”；跨模块的归一化、初始化和 Loss Stability 都是实际架构设计的一部分。

## 原始 Jamba-v0.1 的能力与实验边界

[原始 Figure 4（PDF 第 7 页）](https://arxiv.org/pdf/2403.19887) 在 Needle-in-a-Haystack 合成检索中报告长上下文表现良好，已公开版本支持至 256K Context；作者另提及能训练至 1M Context 的模型，并非说公开权重正式支持 1M。该图验证特定简单陈述检索，不代表 256K 内任意复杂推理任务都成功。

[原论文 Table 2（PDF 第 7 页）](https://arxiv.org/pdf/2403.19887) 将 Jamba Base 与 Llama-2、Gemma、Mixtral 在若干常用学术任务上比较，但既有参数差异也有训练语料、Token Budget 差异，因此只能谈**相应设置下具有竞争力的性能**，不能将差值严格解释为 MoE 或 Mamba 的单独增益。

原论文 Figure 1 下方还明确提示：Jamba-v0.1 是**未做 Alignment 与 Instruction Tuning 的 Base Model**，不应和另一个聊天对齐模型在直接对话体验上机械比较。训练配方与后训练对模型服务质量的影响需要单独分析。

**总结**：Jamba 的核心架构创新是让“显式历史读取的 Attention 层比例”“固定状态传播的 Mamba 层比例”“通道容量的 MoE 密度与 Top-K”成为可独立调节的设计自由度。它和 [Griffin](../../A/09-state-space/A064-griffin.md) 的区别是**混合已存在的全局 Attention/Mamba 机制**，而不是另起炉灶设计递推 Cell。与接下来的 [Zamba](B014-zamba.md) 的区别，则是 Zamba 更强调**跨深度重复使用同一个 Attention/MLP 参数模块**来提高每个新增参数获得的表达能力。
