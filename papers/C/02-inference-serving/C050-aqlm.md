# AQLM：多个码本相加，为什么能比独立整数舍入更接近低比特权重？

> **原始论文**：Egiazarian 等，[Extreme Compression of Large Language Models via Additive Quantization](https://proceedings.mlr.press/v235/egiazarian24a.html)，ICML 2024。[作者论文 PDF：Eq.(1)–(8)、Figure 3、Algorithm 1](https://arxiv.org/pdf/2401.06118)。


前置：[SpQR 的异常值隔离](C048-spqr.md)、[QuIP# 的向量码本](C049-quip-sharp.md)。后续：[QLoRA 的 NF4 与低显存微调](C051-qlora.md)。

## 一个坐标选择 4 个格点，为何不如多维码字？

常规标量量化在每个 Weight 上独立选择整数格点，其编码复杂度低、GEMM 也易适配，但当目标只有 2 bit/weight 时，每个标量只剩约四个可表达等级。若一小段权重向量沿特定相关方向分布，逐维独立 Round 不一定能利用它们的相关性。

向量量化把长度为 $g$ 的一组权重看作 $w\in\mathbb R^g$，选择码本里的一个向量 $c_k$；Additive Quantization (AQ) 更进一步：从 $M$ 个码本各选一个向量并相加，

$$
\widehat w=\sum_{m=1}^{M}c_{m,i_m},\qquad
i_m\in\{1,\ldots,K\}.
$$

每个 Group 只需存一组整数索引 $(i_1,\ldots,i_M)$，共享一批学得的码本。这不是“对每个 Weight 在 M 个密集矩阵上分别加一个 Float 残差”，而是通过索引产生一条短的和式。

### 表示能力如何提升？为什么搜索又变困难？

若每个码本有 $K$ 条码字，组合上最多可产生 $K^M$ 种索引序列（不同序列可能给出相同的向量和）；要编码则需要约 $M\log_2K$ bit/Group，另有码本摊销的固定存储。

例如 $K=16$、$M=2$，可选择的索引组合有 $256$ 种，但存储只需 $2\times4=8$ bit/Group。这里的 256 只是组合计数，**不是**保证能精确逼近任意 256 个目标向量。寻找最优组合

$$
(i_1^\star,\ldots,i_M^\star)=
\arg\min_{i_1,\ldots,i_M}
\left\|w-\sum_mc_{m,i_m}\right\|_2^2
$$

涉及耦合的离散组合搜索，简单逐码本贪心可能错过全局较好组合；极低 bit 优势正来自用更多编码与校准计算换更高的表达效率。

## 加法码本的参数预算要连“码本本体”一起算

设每组维度为 $g=8$，使用 $M=2$ 个码本，每个码本包含 $K=16$ 个候选向量，索引需要 $2\log_2 16=8$ bit，因此**仅索引**成本为 $8/8=1$ bit/weight。但如果两个码本各保存 $16\times8$ 个 FP16 分量，码本本体共需

$$
M_{\rm codebook}=2\times16\times8\times16
=4096\text{ bit}.
$$

如果只有 32 组 Weight，那么码本开销摊到每个 Weight 为 $4096/(32\times8)=16$ bit/weight，反而不划算。如果共服务 100 万组，其额外摊销约 $4096/(10^6\times8)=0.000512$ bit/weight，几乎可忽略。这里的数值只是为了演示参数摊销，并非 AQLM 官方设定的 Group Size 和 Codebook 规模。

因此**Codebook 的共享范围**对极低 bit 的有效码率至关重要。论文标题里的 2–3 bit/weight 不能只计算索引而完全忽略码本、Scale、Metadata 与对齐填充。

## 为什么 AQLM 还要训练码本参数和跨层误差？

仅以 Weight 本身的欧氏误差优化，可能并不符合神经网络的输出敏感性。AQLM 研究 **输入自适应的权重量化**，并将码本联合优化扩展到 Transformer Block 范围，以减少多个量化层之间的误差传播。可把其动机写为

$$
\min_{\{C_m,i_m\}}\,
\mathbb E_{X\sim\mathcal D_{\rm cal}}
\|F_{\rm full}(X)-F_{\rm compressed}(X)\|_2^2.
$$

这是概念性 Block 重建目标，具体优化顺序、Teacher 输出、Codebook 梯度与离散索引更新以原文算法为准。

### 先把层级目标写对：优化的并非每个原始 Weight 的绝对值

原论文 [§3.1 Eq.(1)–(3)](https://arxiv.org/pdf/2401.06118) 用**输出维度为行**的约定：$W\in\mathbb R^{d_{\rm out}\times d_{\rm in}}$、校准输入 $X\in\mathbb R^{d_{\rm in}\times N}$。设 $W_c$ 为从码本和索引重建的矩阵，逐层主要目标是

$$
\mathcal L_{\rm layer}=\|WX-W_cX\|_F^2
=\operatorname{tr}\big[(W-W_c)XX^\top(W-W_c)^\top\big].
$$

$XX^\top:[d_{\rm in},d_{\rm in}]$ 捕获输入通道之间的相关性；它让一个在权重坐标上看似很大的误差，在几乎不被校准样本访问的方向上代价较低。若从这里直接换成 $\|W-W_c\|_F^2$，就丢掉 AQLM 的输入感知核心。反之，这个二次型也只度量当前样本的**层输出**，未必能直接保证长序列生成质量。

作者以**每个输出行中相邻 $g$ 个输入权重为一组**；每组从 $M$ 个码本（每本 $2^B$ 个 $g$ 维向量）各选一项相加，并额外学习每个输出行的 Scale $s_i$。因此索引部分的名义码率是

$$
b_{\rm indices}=\frac{MB}{g}\quad\text{bit/weight},
$$

完整码率还要加**共享码本 $M\,2^B g$ 个浮点数、每行一个 Scale、Padding/索引对齐**。这是为什么论文 Figure 3 的最后一步是按行乘 Scale，不能只看到中间的多码本相加就忽略掉输出 Scale。

### 两个码本的反例：为什么“依次贪心选最近项”不等于联合搜索？

换一个没有 Tie 的最小教学问题。令目标标量 $w=4$，第一码本 $\mathcal C_1=\{0,5\}$，第二码本 $\mathcal C_2=\{0,4\}$。**逐码本贪心**先在 $\mathcal C_1$ 里选离 4 最近的 5；此时残差为 -1，只能在 $\mathcal C_2$ 里选 0，最终重建为 5，误差平方 1。但是联合比较四种组合，选择 $(0,4)$ 就能精确重建 4，误差 0。离散码本的选择互相制约，正是 Beam Search 有意义的地方。

实际 AQLM 还要处理多个输出行、多个输入 Group 和输入相关矩阵 $XX^\top$，远比这个标量例子复杂。作者 [§3.2 Eq.(4)–(7)](https://arxiv.org/pdf/2401.06118) 把目标展开为码本选择的 **Unary Term + Pairwise Term**，它等价于一个离散 Markov Random Field 的能量函数：选择其中一个码字不仅改变它自己的误差，也改变它与其它被选码字之间的交叉项。Beam Search 保留多个部分候选，而不是把第一次贪心结果永久固定。

~~~{mermaid}
flowchart LR
 W[Weight Group w] --> I[搜索多个 Codebook Index]
 CB[Learnable Codebooks] --> S[码字向量加和]
 I --> S
 S --> R[重建近似权重]
 R --> E[校准输入加权的误差]
 E --> U[联合优化 Codebook/Index]
 U --> CB
~~~

*自制解释图，非论文原图。需分别考察码字检索、解码后的 GEMM 格式与跨 Block 重建训练。*

## 为何多个 Codebook 的选择必须联合优化？

前面的无 Tie 反例已经解释了为何局部贪心可能失败。需要进一步注意，AQLM 的加法不是“各个码本分别选离原始 $w$ 最近的向量后再相加”；后一个码本能否修正前一个码本产生的残差，取决于两者**所有可用码字组合**。选择更宽的 Beam 可以探索更多组合，但仍不是全局最优性的数学保证。

对多维向量，把目标改为

$$
\min_{i_1,\dots,i_M}
(w-\sum_m c_{m,i_m})^\top
H_g\,(w-\sum_m c_{m,i_m}),
$$

其中 $H_g$ 是可由输入统计构建的组内误差权重。这里是**输入加权重建目标的教学式**，原论文还有更具体的 Codebook 学习与跨层联合优化，不能照式宣称就是其精确离散求解器。

## 为什么要超越单层的误差重建？

Transformer Block 含 QKV、输出投影与 FFN，某一层 Weight 的最小欧氏误差未必是最终 Block Output 的最小误差。比如第一层的微小系统偏差可能被第二层放大，也可能被第二层抵消。AQLM 将**输入感知量化**与**跨同一 Transformer Block 的 Codebook 联合优化**结合，针对完整块的中间表征保持精度，这也是它相较经典搜索检索中的 Additive Quantization 的模型专属创新。

### 从 Figure 3 接到 Algorithm 1：三个阶段的更新变量根本不同

请对照原论文 [Figure 3、Algorithm 1（PDF 第 6 页）](https://arxiv.org/pdf/2401.06118)。Figure 3 从左到右依次画的是：整数索引 Codes → 依据索引从多份 Codebook 取向量 → 在同一 Group 上求和 → 拼成每个输出行 → 乘每行的 Scale。**图中的蓝色方块是离散索引，不是存放 FP16 权重的多个矩阵**；绿色部分才是共享的码本向量。没有最后的 Scale，便少了一项原论文实际优化的连续变量。

Algorithm 1 还把训练分成三个不能混为一谈的阶段：

1. **初始化。** 在当前线性层，把权重分成 $g$ 维小组，先对原 Weight 执行 K-Means，再在上一步的**残差**上执行下一轮 K-Means，初始化多份码本及各组索引；收集校准输入 $X$，形成 $XX^\top$。这叫 Residual K-Means，不是随意从标准正态初始化码本。
2. **层内交替优化。** 固定当前整数索引，使用 Adam 优化连续 Codebook 与每输出行 Scale；然后固定码本，对索引做 Beam Search。作者在 §3.2–3.3 明确这是在同一个 $\|(W-W_c)X\|_F^2$ 目标上交替优化，直到改进低于阈值；Beam Search 每次替换一枚 Code 时只需更新受影响的 Unary/Pairwise 能量，不能把 $K^M$ 个组合全部物化再逐个评分。
3. **块内联合微调。** 一旦当前 Transformer Block 中各线性层都压缩，固定**离散索引**，对码本、Scale 以及 Block 中原本未量化的可训练参数（如 Norm Scale、Bias）做 Block Output Reconstruction；目标是压缩前的 $Y_{\rm block}$ 与压缩后同一输入的输出之差。再把量化后的 Block Output 传到下一个 Block，防止只用永远干净的原始隐状态掩盖跨块误差传播。

~~~{mermaid}
flowchart TD
 X["当前 Block 校准输入"] --> FP["保存原始 Block 输出 Y"]
 X --> L["对每个 Linear 收集 X 与 W"]
 L --> INIT["Residual K-Means 初始化索引/码本"]
 INIT --> CB["固定索引 → Adam 更新码本/行 Scale"]
 CB --> BEAM["固定码本 → Beam Search 调整离散索引"]
 BEAM -->|损失仍改善| CB
 BEAM --> DONE["当前 Block 的全部 Linear 量化"]
 DONE --> FT["冻结索引；共同优化可训练参数"]
 FP --> FT
 FT --> NEXT["输出压缩后的 Block 状态给下一层"]
~~~

*自制执行图，非原论文 Algorithm 1；作者的源码会具体决定轮次、Beam Width、初始化和停止阈值，图中不把这些实现超参误写成理论常数。*

上述过程同时回答两个容易混淆的问题：**AQLM 的 PTQ 允许用校准数据做小规模优化，不等于重训语言模型；“跨层联合”指一个 Transformer Block 内多个量化 Linear 的联合重建，不等于全模型所有参数同时反向传播。** 此外，部署时恢复的 $W_c$ 仍是多个码字加和（再乘 Scale），不是因为训练过程中使用 Adam 就变成了一个新的 FP16 Dense Checkpoint。

[ICML 2024 正式论文](https://proceedings.mlr.press/v235/egiazarian24a.html) 直接把贡献概括为这两条，并报告在约 2 bit/parameter 的极低压缩段建立更好的质量—模型体积 Pareto 前沿，还给出 CPU/GPU Token Generation 实现。其端到端速度结论应仅用于论文给定模型、Kernel 和执行设置；不能据此假定 AQ 索引查表普遍比标准 INT4 GEMM 更快。

尤其应该看 [AQLM 原始论文 Figure 1（PDF 第 1 页）](https://arxiv.org/pdf/2401.06118)：横轴是 LLaMA 2 模型规模（7B、13B、70B），纵轴是 WikiText-2 PPL，图中比较 FP16、2-bit QuIP# 与 2-bit AQLM。图意不是“2-bit 模型数学无损”，而是在**作者所用压缩率与 Benchmark 协议**下评估 AQLM 处于哪个精度—体积前沿。若两条 2-bit 方法的实际元数据开销不同，必须进一步按有效 Bits/Weight 对齐，才算严格的同一预算比较。

**复现评估应同时记录**：真实含 Codebook 的 bits/weight、原始 PPL、Block 重建误差、Prefill/Decode 时延。如果只有一个 2-bit PPL 结果，无法判断向量码本的精度收益究竟是否抵消解码开销。与 [QuIP#](C049-quip-sharp.md) 的关键差别是：QuIP# 借特殊结构格码本+Hadamard 变换，AQLM 主打**可训练的多个 Codebook 加法组合以及 Block 联合重建**。

## 为什么“码本压缩更极致”仍不等于 GPU 一定更快？

普通 INT4 Kernel 可一次从 Packed Weight 中还原整数并执行矩阵乘；AQ 需要按索引加载多个码字向量并相加。如果码本驻留 Cache 且查表对齐合适，可以高效；否则随机 Lookup 和多个码本反而增加访存与指令。可将单次延迟粗略拆为 $T_{\rm decode}+T_{\rm matmul}$，而真实吞吐还受 Batch 和 KV 大小影响。

[ICML 2024 正式论文](https://proceedings.mlr.press/v235/egiazarian24a.html) 将**极低比特（约 2–3 bit/parameter）**作为主要挑战，报告输入自适应和跨 Block 优化在超低位宽下提升量化保真。其贡献不等于声称在各种算子上绝对超过 GPTQ/AWQ 的速度。比较应报告有效 bits/weight（含码本、索引、Scale）、PPL 与实际 GPU 延迟，不能仅用 nominal “2-bit” 排名。

如果目标是低显存下**继续微调**，即使压缩权重很成功，也还要解决梯度与优化器状态问题。下一篇 [QLoRA](C051-qlora.md) 从这个问题出发，而不是单纯继续减小 PTQ bit 数。
