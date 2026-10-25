# QuIP#：Hadamard 为什么能帮助极低比特量化，E8 格码本又解决什么问题？

> **原始论文**：[QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://proceedings.mlr.press/v235/tseng24a.html)，ICML 2024。


前置：[GPTQ 的 Layer 重建代价](C041-gptq.md) / [SpQR 的异常权重隔离](C048-spqr.md)。后续：[AQLM 的多码本残差相加](C050-aqlm.md)。

## 为什么同样能量的向量，其坐标分布决定了量化难度？

对 $w=(10,0,0,0)$，如果每个坐标都只能用相同 Scale 和很少的整数刻度表示，第一维的极值决定了量化范围，剩余坐标通常分辨率浪费。设随机的正交矩阵 $H$ 将 $w$ 分散到多个坐标：

$$
\tilde w=Hw,\qquad H^\top H=I.
$$

虽然 $\|\tilde w\|_2=\|w\|_2$ 完全不变，但能量不再集中于一个标准坐标方向。对随机符号修正的 Hadamard 变换，许多向量的巨大单坐标会被近似摊散。注意“随机旋转后完全没有异常值”不是严格定理；论文用 **Incoherence Processing** 降低权重/输入矩阵中的坐标集中性。

## 变换怎么不改变 Transformer 的线性层？

令输入行为 $x$，Weight 为 $W$，$U,V$ 是维度适当的正交矩阵，则

$$
xW=(xU^\top)(UWV^\top)V.
$$

可以在中间的 $UWV^\top$ 上量化：浮点代数中左右两个正交变换及其逆变化抵消，输出仍是 $xW$。如果变换后量化得到 $Q(UWV^\top)$，则误差变成

$$
\widehat y-y=
(xU^\top)\left[Q(UWV^\top)-UWV^\top\right]V.
$$

因为正交变换不改变欧氏范数，若忽略浮点和量化副作用，目标主要是让**变换后的误差在所用量化器中更小**，并与校准 Hessian 的方向协调。QuIP# 使用 **Randomized Hadamard Transform** 替代旧 QuIP 的部分变换设计，强调更好并更快的混合性。

### Hadamard 不是单纯“把绝对值除以根号 n”

取 2 维 $H=(1/\sqrt2)\begin{bmatrix}1&1\\1&-1\end{bmatrix}$。对 $w=[a,0]^\top$，

$$
Hw=[a/\sqrt2,a/\sqrt2]^\top,
$$

确实把集中于第一个坐标的能量分摊。但 $w=[a,a]^\top$ 反而被映成 $[\sqrt2 a,0]^\top$。因此需要随机符号等机制与统计性分析，而不是说每个确定向量经 Hadamard 都会减少最大幅值。

## 为什么从逐元素 Round 走向 8 维格码本？

如果变换后的向量近似各向同性、形状像球，那么独立标量量化相当于用笛卡尔网格的小正方体格点覆盖分布，浪费球状概率密度结构。**向量量化**把一组 $d$ 个数作为整体，选择码本向量：

$$
k^\star=\arg\min_{k}\;\|z-c_k\|_2^2,\quad
z\in\mathbb R^d,\ c_k\in\mathcal C.
$$

QuIP# 设计了与 8 维 $E_8$ 格结构有关的高效码本（E8P），利用球堆积性能，同时考虑 GPU/Cache 解码成本。不能把“$E_8$ 是最优球堆积”直接推出“所有 Transformer 参数量化误差必然最小”，因为真实权重分布、Hessian 权重和量化位宽都不是完全均匀球形。其优势来自 **变换+码本+块自适应舍入+额外微调** 的联合设计。

~~~{mermaid}
flowchart TD
 W[原始权重] --> R[Randomized Hadamard Incoherence]
 R --> B[BlockLDLQ 自适应 Round]
 E[E8P 向量码本] --> B
 B --> FT[可选微调改善保真]
 FT --> K[可供 GPU 快速解码的低比特布局]
~~~

*自制原理图，非论文 Figure。完整管线还有 Hessian 处理与正交变换的权重/输入配对。*

[ICML 2024 原论文](https://proceedings.mlr.press/v235/tseng24a.html) 将 QuIP# 描述为 4 bits/weight 以下的高保真 Weight-Only PTQ，并对无 Fine-Tuning、无格码本和旧 QuIP 的变体作消融。研究中可以固定 **真实有效 bit 数**（包括 Codebook 和元数据），再比较 PPL、逐层 MSE、预处理耗时以及 Kernel Token/s。

下一篇 [AQLM](C050-aqlm.md) 仍使用码本思想，但不只从一个 Codebook 里选一个向量，而让多个 Codebook 的向量**相加共同逼近**高维权重分组。
