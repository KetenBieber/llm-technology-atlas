# QLoRA：冻结 4-bit 主干后，梯度为何仍能训练 LoRA？

> **原始论文**：[QLoRA: Efficient Finetuning of Quantized LLMs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)，NeurIPS 2023。


前置：[Weight-Only 低比特](C041-gptq.md) / [AQLM](C050-aqlm.md)。后续：[LoftQ 的量化感知初始化](C052-loftq.md)。

## 4-bit 权重不可微，怎么可能做低显存梯度微调？

先分清两种操作：**对冻结的量化主干做前向与反向传播**，与**训练量化整数本身**不同。QLoRA 让预训练主干权重 $W_0$ 冻结在 4-bit 存储中，并增加可训练的低秩适配器 $BA$。一个 Linear 层可以抽象为

$$
y=X\operatorname{Dequant}(Q_4(W_0))
+\frac{\alpha}{r}XBA,
$$

其中 $A\in\mathbb R^{d_{\rm in}\times r}$，$B\in\mathbb R^{r\times d_{\rm out}}$（矩阵乘法按 $XBA$ 的右乘约定）。$r$ 通常远小于隐藏维度。主干 $W_0$ 不参与更新，反向传播可以穿过固定的 Dequant 和 Linear 运算，得到前面层激活所需梯度；要更新的参数只有 $A,B$。

即使 $Q_4(W_0)$ 本身的 Round 不可微，也不妨碍计算 $\partial \mathcal L/\partial A$ 和 $\partial \mathcal L/\partial B$。这就是“量化模型上训练适配器”与“所有 Weight 都训练成 4-bit”之间的根本差别。

## 梯度究竟经过哪里？画出冻结主干的反向路径

考虑一个前向为

$$
Y=X\bar W+\frac{\alpha}{r}XAB,\qquad
\bar W=\operatorname{Dequant}(Q_4(W_0)).
$$

这里统一采用 $X:[B T,d_{\rm in}]$、$A:[d_{\rm in},r]$、$B:[r,d_{\rm out}]$、$\bar W:[d_{\rm in},d_{\rm out}]$ 的**行向量右乘**约定。

设 $\partial\mathcal L/\partial Y=G$，那么对 LoRA 参数：

$$
\frac{\partial\mathcal L}{\partial B}
=\frac{\alpha}{r}(XA)^\top G,\qquad
\frac{\partial\mathcal L}{\partial A}
=\frac{\alpha}{r}X^\top(GB^\top).
$$

对输入的梯度还要包含冻结主干的作用：

$$
\frac{\partial\mathcal L}{\partial X}
=G\bar W^\top+\frac{\alpha}{r}GB^\top A^\top.
$$

因此，即使 $\bar W$ 是量化后固定的，**反向仍必须通过它把梯度传回上一层激活**，才能训练更早的 Adapter。冻结的是主干的**参数更新**，不是切断整张网络的 Backward Graph。如果在实现中对整个 Transformer Block 使用 `torch.no_grad()`，早期 Adapter 往往拿不到正确的梯度；这是一个典型实战错误。

### 最小数值例子：为什么梯度不需要穿过 Round？

设只有一个输入和一个输出，冻结主干权重的 4-bit 反量化结果为 $\bar w=2$；可训练 Adapter 标量 $a=0.5,b=0.2$，输入 $x=3$。若 $\alpha/r=1$：

$$
y=x(\bar w+ab)=3(2+0.1)=6.3.
$$

以 $\mathcal L=\tfrac12(y-6)^2$，输出梯度为 $0.3$，所以

$$
\frac{\partial\mathcal L}{\partial a}
=0.3\,xb=0.18,\quad
\frac{\partial\mathcal L}{\partial b}
=0.3\,xa=0.45.
$$

反传并不要求计算 $\partial Q_4(w)/\partial w$，因为 $w$ 根本没有参与更新。和 [BitNet](C053-bitnet.md) 从头量化感知训练需要 STE 的情形完全不同。这个例子是可复算数学，不是原论文实验值。

## 为什么只是把权重变成 INT4 还不够？

真正 Full Fine-Tuning 的 Adam 通常需要主干参数、梯度、一二阶优化器状态及额外 Master Weight；冻结主干可移除对主干梯度与优化器状态的存储，但**激活、KV/Attention 临时张量、适配器参数与优化器**仍消耗显存。QLoRA 配套三项工程选择：

### NF4：NormalFloat 不是普通等间隔整数

如果预训练权重近似零均值、某种归一化后的分布接近正态，将 16 个编码点集中在正态概率密度较高的区域，可能比将 16 个码点均匀放到 $[-1,1]$ 更有效。NF4 是为近似正态权重分布设计的非均匀 4-bit 表示；它的最优性有分布与量化假设，不能泛化成任意类型的实数数据都最优。

### Double Quantization：连 Scale 本身也压缩

若每 $g$ 个权重额外存一个 32-bit Scale，元数据摊销为 $32/g$ bit/weight；用第二级量化编码第一层的 Scale，能够继续减少元数据空间。但第二层 Scale 仍然需要存储，不能说“Double Quant 完全没有 Scale 开销”。

### Paged Optimizer：处理峰值而非凭空减少 FLOPs

长序列和 Batch 的激活峰值会与优化器状态发生显存竞争。QLoRA 的 Paged Optimizer 借助 Unified Memory 等机制将暂时不需要的状态按页迁移，从而避免部分瞬时 OOM；这可能引入 Host/Device 传输，吞吐并不一定单调改善。

~~~{mermaid}
flowchart LR
 W[FP16 Pretrained W] --> Q[NF4 + Double Quant 主干冻结]
 Q --> F[Dequant + Linear]
 A[Trainable LoRA A、B] --> L[Low-rank Residual]
 F --> Y[合并输出]
 L --> Y
 Y --> LOSS[下游任务 Loss]
 LOSS --> A
~~~

*自制数据流图，非论文 Figure。QLoRA 的主要梯度只更新 Adapter，图中没有从 Loss 回到冻结 Q(W) 的参数更新路径。*

## NF4 的码点为什么不是等距整数？

对在 Scale 归一化后近似服从 $N(0,1)$ 的权重，若希望每个码点附近承载类似的概率质量，可用正态分布的分位数函数 $F^{-1}$ 构造量化中心。例如理论性地写作

$$
c_i\approx \frac{F^{-1}(i/(2^b+1))+F^{-1}((i+1)/(2^b+1))}{2}.
$$

实际 NF4 并非简单直接使用上述单个近似式就结束：作者还考虑**零码点**、正负区间的分位数分配、将有码点归一化到 $[-1,1]$。参见 [QLoRA 正式 PDF Appendix Figure 7](https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)。要理解 Figure 7，应从“为什么标准等间距 4-bit 格点浪费正态分布中心的密度”开始，而不是把 NF4 误认为“FP4 只是名字不同”。

## Double Quant 再把 Scale 压缩了多少？

假设每 64 个权重存一个 32-bit FP Scale，则元数据成本 $32/64=0.5$ bit/weight。若把第一层 Scale 改成每个 8-bit 表示，每 256 个第一层 Scale 再用 32-bit 第二层 Scale：

$$
b_{\rm meta}=\frac8{64}+\frac{32}{64\cdot256}
=0.126953125\text{ bit/weight}.
$$

与 0.5 对比，理论节约约 0.373 bit/weight。**这就解释了原论文报道的平均约 0.37 bit/parameter**（具体实现还有分布中心化等处理）。对 65B 参数，节约约 $65\cdot10^9\times0.37/8\approx3.0$ GB 的量级。不是“Scale 全部消失”，而是让 Scale 再被第二层编码。

### 峰值显存模型不能只算权重位宽

训练占用可概括为

$$
M_{\rm peak}=M_{Q(W)}+M_{\rm scale}
+M_{A,B,\mathrm{grad,opt}}
+M_{\rm activations}
+M_{\rm runtime}.
$$

QLoRA 减少前两项并仅更新小规模 Adapter，但长 Context 的激活和运行时临时 Tensor 仍可能使训练 OOM。Paged Optimizer 主要处理这样的峰值状态迁移，并不保证梯度 Checkpointing 或 Host/Device IO 没有代价。

## 论文究竟证明了“和 FP16 一样好吗”？

[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html) 报告使用单张 **48GB GPU** 微调 65B 模型的配置，并在测试任务上达到接近全精度微调的表现。具体性能与任务、数据集、LoRA Rank、量化数据类型和训练步数有关；这不是对任何 65B 模型“训练后数学上完全等价”作出保证。

**独立验证点**：固定相同 LoRA Rank 与数据，比较 NF4/FP4、Double Quant 开关、Optimizer Paged/Non-Paged 的峰值显存和任务指标。小 Batch 训练中，实际算力可能仍受反量化 Kernel 与 Activation Checkpointing 限制。下一篇 [LoftQ](C052-loftq.md) 专门问：主干量化误差可以在训练开始之前先被 LoRA 初始化吸收多少？

## 原论文 Figure 与结果应怎样配对解释？

[NeurIPS 2023 原论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract.html) 证明了所测模型/任务中冻结 4-bit 主干与 Adapter 能做到竞争性的微调效果，并报告了单张 48GB GPU 微调 65B 的设置；**这不是每个模型在任何数据分布上都能严格匹配 Full Fine-Tuning 的数学定理**。

要把原论文不同证据分开：Figure 7 解释 NF4 Codebook 形成；对比表/消融检验 NF4、Double Quant、Paged Optimizer、LoRA Target Modules 对质量和资源的影响；Guanaco 聊天评测还受到训练数据与模型选择的影响，不能当成“NF4 单组件一定提高对话智力”的因果证据。

如果自己编程验证，应在相同模型、数据、Rank 下做 `NF4 vs FP4`、`DoubleQuant on/off`、`Paged on/off`，记录 Peak VRAM、训练损失、最终任务指标和每步耗时。只有这四维同时观察，才能分开**存储成功**、**可训练**、**收敛速度**和**部署质量**。
