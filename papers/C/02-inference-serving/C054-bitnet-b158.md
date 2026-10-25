# BitNet b1.58：三值权重为什么对应 1.58 bit，而不是 2 bit？

> **原始论文**：[The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)，2024 原始技术报告；亦可参照 JMLR 2025 BitNet 统一论文。


前置：[原始 BitNet 的二值训练](C053-bitnet.md)、[AWQ 的 Weight-Only 量化](C042-awq.md)。

## 为什么增加一个 0 会改变模型表达能力？

原始 BitNet 使用近似 1-bit 二值权重；b1.58 将主要 BitLinear 权重变成三值

$$
Q(W)\in\{-1,0,+1\}.
$$

这不是对 FP16 权重训练完成后简单套用一次 Round，而是**从模型预训练阶段引入三值结构**。$0$ 使本来接近零的连接可直接消除，而不是被迫变成 $+1/-1$；同时更低数值复杂度有利于简单整数乘加实现。

三种可能状态的信息理论理想编码下界为

$$
\log_2 3\approx 1.58496\ \text{bit/weight}.
$$

这是在每种状态可等概率出现、采用理想熵编码或接近信息下界编码时的状态数所需位数；如果实际用一个 Byte 存每个三值参数，现实存储就是 8 bit/weight，而不是 1.58 bit。实际打包通常需要把多个三值权重编码成若干字节，还要考虑张量对齐、缩放元信息与硬件 Load。

### 一个可手算的长度为 2 的例子

若 $x=[2,3]$，三值权重 $w=[1,0]$，则

$$
x^\top w=2\cdot1+3\cdot0=2.
$$

计算可用复制 + 加法实现，无需完整通用浮点乘法。然而这只展示算术级潜力，并不自动等于 GPU 实际 GEMM 比 FP16 快；Packed Ternary Decode、数据准备、Scale、Kernel 融合都可能占据运行时间。

## AbsMean 与训练为什么不能省略？

原始报告以权重矩阵的绝对均值为基础建立量化尺度，使用 Round/Clip 把归一化权重映射到 $\{-1,0,1\}$。可用概念式写成

$$
s=\frac{\|W\|_1}{mn},\qquad
\widehat W=\operatorname{clip}_{[-1,1]}
\left(\operatorname{round}(W/(s+\epsilon))\right).
$$

此式只表达三值的量化形状，原文对 STE、激活量化、Sub-LayerNorm、训练参数及 Scaling 的精确形式应以官方正文/代码为准。$s$ 是保证不同层具有合理动态范围的统计量；若忽略它直接把所有 FP16 参数整数化，会明显混淆 Quantized Code 和可执行网络的完整参数化。

~~~{mermaid}
flowchart TD
 W[训练中的连续潜在权重] --> S[AbsMean/归一化]
 S --> T[三值 Round / Clip]
 T --> F[BitLinear：整数/低比特 Forward]
 F --> LOSS[下一个 Token 的训练损失]
 LOSS --> STE[直通估计传回连续权重]
 STE --> W
~~~

*自制解释图，非论文 Figure。3 个编码状态和部署位打包需要明确区分。*

## 为什么不能用它直接替代 GPTQ、AWQ 或 QLoRA？

[GPTQ](C041-gptq.md)、[AWQ](C042-awq.md) 对训练好的 FP16 模型做 Post-Training Quantization；[QLoRA](C051-qlora.md) 保留预训练的冻结量化主干，只训练低秩适配器。而 BitNet b1.58 是**预训练阶段的结构化低比特设计**。这些方法的初始条件不同：一个需要重新从头训练，一个希望复用完整预训练模型，一个只更新 Adapter，不能混称为同一种模型压缩工具。

## 实证证据与公平比较

[原始技术报告](https://arxiv.org/abs/2402.17764) 报告 b1.58 在所测模型规模和同等训练 Token 条件下接近 FP16 Transformer 的语言建模/下游表现，并分析内存、延迟、吞吐和能耗的潜力。要注意其结果是指定模型架构与训练预算；不能证明“所有现成 LLaMA 权重随手转成三值而不损质量”。

后续 [Microsoft 官方 BitNet 实现](https://github.com/microsoft/BitNet) 包含 bitnet.cpp 等服务算子、CPU/GPU 内核及新的模型工作；这些推理结果不可被当成 2024 报告已经测量过的性能数字。即使原理上 1.58 bit，比真正部署时的 Padding、Activation 精度、Attention KV、Embedding 与缓存带宽更重要的瓶颈仍可能限制总吞吐。

**工程建议**：对三值算术先使用小矩阵实现 Packed Encode/Decode 并与 FP32 计算数值核对，然后在现成的官方 b1.58 模型上对比 Kernel Token/s、Batch 和能耗。不要用 PTQ 精度表直接替代从头训练时需要的 Scaling 和稳定性分析。
