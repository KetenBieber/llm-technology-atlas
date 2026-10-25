# AQLM：多个码本相加，为什么能比独立整数舍入更接近低比特权重？

> **原始论文**：[Extreme Compression of Large Language Models via Additive Quantization](https://proceedings.mlr.press/v235/egiazarian24a.html)，ICML 2024。


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

## 为什么 AQLM 还要训练码本参数和跨层误差？

仅以 Weight 本身的欧氏误差优化，可能并不符合神经网络的输出敏感性。AQLM 研究 **输入自适应的权重量化**，并将码本联合优化扩展到 Transformer Block 范围，以减少多个量化层之间的误差传播。可把其动机写为

$$
\min_{\{C_m,i_m\}}\,
\mathbb E_{X\sim\mathcal D_{\rm cal}}
\|F_{\rm full}(X)-F_{\rm compressed}(X)\|_2^2.
$$

这是概念性 Block 重建目标，具体优化顺序、Teacher 输出、Codebook 梯度与离散索引更新以原文算法为准。

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

## 为什么“码本压缩更极致”仍不等于 GPU 一定更快？

普通 INT4 Kernel 可一次从 Packed Weight 中还原整数并执行矩阵乘；AQ 需要按索引加载多个码字向量并相加。如果码本驻留 Cache 且查表对齐合适，可以高效；否则随机 Lookup 和多个码本反而增加访存与指令。可将单次延迟粗略拆为 $T_{\rm decode}+T_{\rm matmul}$，而真实吞吐还受 Batch 和 KV 大小影响。

[ICML 2024 正式论文](https://proceedings.mlr.press/v235/egiazarian24a.html) 将**极低比特（约 2–3 bit/parameter）**作为主要挑战，报告输入自适应和跨 Block 优化在超低位宽下提升量化保真。其贡献不等于声称在各种算子上绝对超过 GPTQ/AWQ 的速度。比较应报告有效 bits/weight（含码本、索引、Scale）、PPL 与实际 GPU 延迟，不能仅用 nominal “2-bit” 排名。

如果目标是低显存下**继续微调**，即使压缩权重很成功，也还要解决梯度与优化器状态问题。下一篇 [QLoRA](C051-qlora.md) 从这个问题出发，而不是单纯继续减小 PTQ bit 数。
