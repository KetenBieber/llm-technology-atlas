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

## 为什么 1.58-bit 不只是一句 $\log_2 3$：真正的打包例子

5 个三值参数可以取的组合数是 $3^5=243$，小于一个 Byte 能容纳的 $2^8=256$ 种编码。因此在**理想的固定分组打包**中，5 个三值恰好可存进一个 Byte，平均 $8/5=1.6$ bit/weight，已经接近 $\log_2 3\approx1.585$。但是直接把每个三值存为一个 8-bit 有符号整数，物理开销仍然是 8 bit/weight；编码下界本身不保证任何框架真的使用五个三值打包的存储布局。

实际 Weight 的 Scale、对齐、Padding、Kernel 解码与额外元数据必须计入总显存。如果多次回写 Host/GPU 中间 Buffer，理论 Pack Rate 也不等于端到端内存节省倍数。

### 三值对 Layer 准确度有什么结构性改善？

二值集合 $\{-1,+1\}$ 无法表示零。假如量化前一行权重是 $[0.02,-0.01,1.0]$，将三者分别强制投影到 $-1/+1$，两个接近零的连接会被显著放大；三值投影可将前两个连接映为 0、最后一个映为 +1，较容易保留“只靠第三个输入通道”的结构。但三值编码还需要按 Layer Scale 恢复真实量级，不能把这个示例当成对任意数据都更准的严密证明。

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

## 用官方方程把 Forward 和 Backward 连接起来

[2024 年 BitNet b1.58 原始报告](https://arxiv.org/pdf/2402.17764) 以三值 RoundClip 和基于绝对均值的尺度构造 Weight Quantization。训练流程是

$$
W_{\rm latent}\xrightarrow{\rm absmean+roundclip}
W_{\rm ternary}\xrightarrow{\rm BitLinear\ forward}
\mathcal L,
$$

而反向仍对连续 $W_{\rm latent}$ 执行 STE 近似，并维持较高精度的训练优化器状态。注意：这和 [QLoRA](C051-qlora.md) **固定量化主干，只训练 LoRA Adapter** 完全不同——QLoRA 不需要把量化 Round 的梯度传回其冻结主干，BitNet 则必须处理训练过程中参数如何跨越整数状态阈值。

b1.58 论文相较原始 BitNet 的重要架构实现差异是采用更接近 LLaMA 的组件：RMSNorm、SwiGLU、RoPE 并去掉 Bias（原文 `2）；所以在模型性能比较时不可以只说“唯一变量从二值改成三值”，论文实际选择的骨干部件也有差异。

## 为什么不能用它直接替代 GPTQ、AWQ 或 QLoRA？

[GPTQ](C041-gptq.md)、[AWQ](C042-awq.md) 对训练好的 FP16 模型做 Post-Training Quantization；[QLoRA](C051-qlora.md) 保留预训练的冻结量化主干，只训练低秩适配器。而 BitNet b1.58 是**预训练阶段的结构化低比特设计**。这些方法的初始条件不同：一个需要重新从头训练，一个希望复用完整预训练模型，一个只更新 Adapter，不能混称为同一种模型压缩工具。

## Table 1 的数字要逐格读，而不是只写“快了几倍”

原始 PDF 印刷第 3 页 [Table 1](https://arxiv.org/pdf/2402.17764) 在作者所用 RedPajama 100B Token 训练和 FasterTransformer 等执行设置下，报告：

| 规模 | FP16 LLaMA 参照 | BitNet b1.58 | 实际能支持的说法 |
|---|---|---|---|
| 1.3B 运行显存 | 3.34 GB | 1.14 GB | 此配置下内存约 2.93 倍更省 |
| 1.3B 每输出 Token 延迟 | 1.62 ms | 0.97 ms | 延迟约 1.67 倍加速 |
| 3B 运行显存 | 7.89 GB | 2.22 GB | 该模型与 Kernel 配对下约 3.55 倍 |
| 3B 每输出 Token 延迟 | 5.07 ms | 1.87 ms | 该执行设置下约 2.71 倍 |
| 3B WikiText2 PPL | 10.04 | 9.91 | 所测验证集 b1.58 的 PPL 略低 |

比值是同一行 FP16 基线除以 b1.58；不同规模的参照不能交叉计算性能倍数。原文采用作者复现的 LLaMA-like FP16 基线，不能自动等价于外部某个已经训练好的官方 LLaMA checkpoint。

紧接着的 [Table 2](https://arxiv.org/pdf/2402.17764) 比较 ARC-Easy/Challenge、HellaSwag、BoolQ 等多个任务的 Zero-Shot 指标。比如 3B LLaMA-like 的平均得分为 49.7，3B b1.58 为 50.2；但 700M 和 1.3B 中 b1.58 结果略低。故合适表述是**随所测规模增加，三值模型在作者协议下变得更有竞争力**，而非任意大小都严格优于 FP16。

## 与论文 Figure 1 的关系是什么？

Figure 1 把 FP16 和三值网络的成本—性能位置画在同一示意中，说明作者**提出的研究方向**；真正支持峰值显存与推理延迟的是 Table 1 的数值和硬件执行设置，支持任务能力的是 Table 2 的 Zero-Shot Benchmark。不能仅依据一个概念图就宣称某种第三方 NPU/GPU 上能耗下降了固定倍数。

**工程复验**应该先把训练/推理分开：训练关注 STE、Norm 与潜变量精度；推理关注 Packed Ternary Layout、Scale 解码和整数 Kernel。若直接拿一套 FP16 参数强制量化成三值，会失去从头训练的适应过程，和作者结论不在相同假设下。

## 实证证据与公平比较

[原始技术报告](https://arxiv.org/abs/2402.17764) 报告 b1.58 在所测模型规模和同等训练 Token 条件下接近 FP16 Transformer 的语言建模/下游表现，并分析内存、延迟、吞吐和能耗的潜力。要注意其结果是指定模型架构与训练预算；不能证明“所有现成 LLaMA 权重随手转成三值而不损质量”。

后续 [Microsoft 官方 BitNet 实现](https://github.com/microsoft/BitNet) 包含 bitnet.cpp 等服务算子、CPU/GPU 内核及新的模型工作；这些推理结果不可被当成 2024 报告已经测量过的性能数字。即使原理上 1.58 bit，比真正部署时的 Padding、Activation 精度、Attention KV、Embedding 与缓存带宽更重要的瓶颈仍可能限制总吞吐。

**工程建议**：对三值算术先使用小矩阵实现 Packed Encode/Decode 并与 FP32 计算数值核对，然后在现成的官方 b1.58 模型上对比 Kernel Token/s、Batch 和能耗。不要用 PTQ 精度表直接替代从头训练时需要的 Scaling 和稳定性分析。
