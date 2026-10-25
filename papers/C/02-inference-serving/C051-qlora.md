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

## 论文究竟证明了“和 FP16 一样好吗”？

[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html) 报告使用单张 **48GB GPU** 微调 65B 模型的配置，并在测试任务上达到接近全精度微调的表现。具体性能与任务、数据集、LoRA Rank、量化数据类型和训练步数有关；这不是对任何 65B 模型“训练后数学上完全等价”作出保证。

**独立验证点**：固定相同 LoRA Rank 与数据，比较 NF4/FP4、Double Quant 开关、Optimizer Paged/Non-Paged 的峰值显存和任务指标。小 Batch 训练中，实际算力可能仍受反量化 Kernel 与 Activation Checkpointing 限制。下一篇 [LoftQ](C052-loftq.md) 专门问：主干量化误差可以在训练开始之前先被 LoRA 初始化吸收多少？
