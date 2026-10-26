# QuIP#：Hadamard 为什么能帮助极低比特量化，E8 格码本又解决什么问题？

> **原始论文**：Tseng 等，[QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://proceedings.mlr.press/v235/tseng24a.html)，ICML 2024。[原论文 Algorithm 1/2、§4.1、Table 1（arXiv PDF）](https://arxiv.org/pdf/2402.04396)。


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

## 从原论文 Algorithm 1 重建整条数据流

原论文的 QuIP#-NoFT 并不直接把 $W$ 丢给一个 E8 Codebook：它同时取得 $W\in\mathbb R^{m\times n}$ 和由校准输入产生的 $H\in\mathbb R^{n\times n}$，执行 **Incoherence Processing with Randomized Hadamard Transform（IP-RHT）**，随后调用 **BlockLDLQ** 在变换后的权重和 Hessian 上做自适应分块舍入。

用步骤表达：

1. 校准阶段得到原始 $W$ 和一个度量输入方向敏感性的 $H$；二者维度必须匹配同一个乘法求和轴。
2. 取带随机符号的左右 Hadamard 变换 $U,V$；将坐标表示变为 $\widetilde W=UWV^\top$，同时以相应的正交变换更新误差度量（具体左/右方向和转置取决于作者的 $W$ 布局）。
3. 在**变换后的坐标系**运行 BlockLDLQ，选择 Codebook 中的块向量，利用 Hessian 相关性对尚未处理的块进行误差反馈。
4. 存储低比特码本索引与复原所需变换信息，推理时执行与量化表示匹配的矩阵计算/恢复。

这比“先打散 Outlier，再查一个 Codebook”多了一项很重要的约束：**量化误差代价受输入分布影响，不只靠未加权的欧氏距离评估**。

### 为什么 BlockLDLQ 需要上一块的舍入误差？

对输出行排列的权重 $W\in\mathbb R^{m\times n}$，设校准样本按列堆叠为 $X\in\mathbb R^{n\times N}$。量化后层输出的平方误差不是简单的 $\|W-\widehat W\|_F^2$，而是

$$
\|(W-\widehat W)X\|_F^2
=\operatorname{tr}\!\left((W-\widehat W)H(W-\widehat W)^\top\right),\quad H=XX^\top.
$$

权重误差来自哪个**输入方向**，会改变最终代价。取 $H=\mathrm{diag}(100,1)$，两个相同长度的教学误差 $e_1=[0.1,0]$ 与 $e_2=[0,0.1]$，其欧氏平方都为 $0.01$，却分别产生 $e_1He_1^\top=1$ 和 $e_2He_2^\top=0.01$。因此“码本内欧氏最近的向量”未必是对实际 Layer Output 最好的选择。

原论文 [§4.1](https://arxiv.org/pdf/2402.04396) 沿用 LDLQ 的反馈思想，把权重的**一批相邻输入列**作为一个 $g$ 维块。对对称正定的代理 Hessian 做 Block-LDL 分解

$$
H=L^\top D L,\qquad U=L^\top-I,
$$

其中 $L$ 的对角 $g\times g$ 子块是单位阵，$D$ 为 Block-Diagonal。记已经处理完的所有列块为 $W_{<k},\widehat W_{<k}$，当前块为 $W_k:[m,g]$，$A_k$ 是 $U$ 对应的已处理列到本块的反馈子矩阵，则

$$
\boxed{\widehat W_k=Q_{\mathcal C}\!\left(W_k+
(W_{<k}-\widehat W_{<k})A_k\right).}
$$

$Q_{\mathcal C}$ 是**逐个 $g$ 维码字**投向码本 $\mathcal C$ 的向量量化器；不能把它换回普通逐元素 Round 而仍称为 BlockLDLQ。右边乘法 Shape 为 $[m,\text{已处理列数}]\,[\text{已处理列数},g]=[m,g]$。当 $H$ 没有跨块相关性、$A_k=0$ 时，公式退化为独立块量化；有相关性时，上一块的舍入残差会改变下一块实际被量化的目标。**这就是 BlockLDLQ 比单独选择码本索引多做的一步。**

上面的数学反馈不是作者“保证每条输出都最优”的声明。原文 Theorem 4.1 的误差界还要求 Hessian 的 Incoherence 与向量量化误差协方差上界，不能把定理写成无条件的 PPL 最优。训练/预处理阶段需处理数值病态和 Block Size 整除条件，推理时不再重复求 Hessian 或 LDL 分解。

### Algorithm 2：量化后的权重为什么仍需两次 Hadamard？

把 QuIP# 的坐标改变当作单纯对 Weight 做预处理是不完整的。原始 [Algorithm 1/2（PDF 第 2 页）](https://arxiv.org/pdf/2402.04396) 分别是离线压缩与**实际 Linear 推理**。其 Algorithm 2 用符号向量 $S_V,S_U$ 实现左右随机化 Hadamard：

~~~text
压缩前（离线）：变换 W/H → BlockLDLQ 生成低比特码字索引及符号信息
推理时：
  y = Had(S_V ⊙ x)                    # 输入到量化坐标系
  y = decompress_multiply(W_hat, C, y) # 查 E8P 码字、乘法
  y = Had(S_U ⊙ y)                    # 还原输出坐标系
~~~

*自制逐算子解释，执行顺序取自原论文 Algorithm 2；$x$ 在此按列向量表示，不能与前文为方便说明所用的行向量 $xW$ 直接混用。*

若省掉最后一步，输出的数值坐标仍被随机旋转；若省掉第一步，量化矩阵接收的输入坐标错误。**两次变换是为了保持整层函数语义，不是免费的精度增强**，它们会产生额外计算。研究者应同时考察压缩 Weight 读取减少的 HBM 负载与两次 Hadamard 的算术/实现成本。

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

如果想把本篇用于 Weight-Only 的随机正交打散推广到**激活和 KV Cache 也采用 4-bit**，下一条分支是 [QuaRot](C055-quarot.md)；随后 [SpinQuant](C056-spinquant.md) 继续解决随机旋转在不同初始化下精度波动的问题，而 [FlatQuant](C057-flatquant.md) 将变换限制放宽到可逆仿射并控制算子开销。

## 为什么 E8P 不是一个不能部署的 65,536 行码本？

### 用编码位数手算码本压缩率

假设一个 Codeword 有 8 个浮点坐标，直接以 FP16 保存需要 $8\times16=128$ bit。若采用 16-bit 的离散索引，每个 Codeword 在**不计共享码本本体**时的表示仅占 16 bit，对应 2 bit/weight。这个数字解释了为什么 8 维码本会与极低比特权重压缩联系起来，却没有说明 GPU 查表有多快。

如果朴素地保存 $2^{16}=65536$ 个码字，每个码字都有 8 个 FP16 坐标，完整查找表需要 $2^{16}\times8\times2=1{,}048{,}576$ Byte，即约 1 MiB。QuIP# 利用格码字的对称结构，不按这种朴素方式存储所有向量；它可以从 Base Table、Sign 等紧凑编码现场构建码字。**索引空间有 65536 个组合**与**存储 65536 行 FP16 Codebook**是两件不同的工程选择。

8 维向量如果以 2 bit/weight 编码，需要 16-bit 的索引，共有 $2^{16}=65{,}536$ 种可能码字。若朴素保存每种码字的 8 个 FP16 分量，存储就要 $65{,}536\times8\times2\approx1$ MB；还要考虑 GPU Cache 与 Bank Conflict 的重复缓存，访问成本可能不划算。

作者的 E8P 将码本拆成**256 个基础向量查找 + 符号翻转 + 小幅平移**的结构：16-bit 编码可以分解为 8 bit Base Entry、7 bit Sign 和 1 bit Shift/Parity 相关信息。原论文 §4.2 进一步明确 256 个基础条目由 **227 个规定范数范围内的候选 + 29 个 Padding 条目**构成；查表尺寸为 $256\times8$，可用约 1KiB 的紧凑表表示。这里的 1KiB 是 E8P **Base Table** 的量级，并不包括模型 Weight 索引、缩放、旋转信息或完整模型元数据。

[QuIP# 正式论文与 Algorithm 1/2](https://proceedings.mlr.press/v235/tseng24a.html) 把**可表达性**和**可以装进 GPU L1、解码效率高**同时当作设计约束。$E_8$ 是 8D 球堆积方面的最优结构，不意味着它在真实 Transformer 上的所有 Weighted Reconstruction Objective 都数学全局最优。

### 从一个两维 Hadamard 反例辨析“打散”

$$
H_2=\frac1{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}.
$$

对于集中向量 $w=[10,0]^\top$，有 $H_2w=[7.071,7.071]^\top$，最大坐标变小；但是 $w'=[10,10]^\top$ 被变换成 $[14.142,0]^\top$，最大坐标反而变大。**单个确定 Hadamard 不保证每个输入最大坐标都减少**。随机符号加 Hadamard 的统计打散是作者的策略，不应把这种概率性质改写成逐样本严格上界。

## Figure 3 真正在对比什么？

论文 [Figure 3](https://proceedings.mlr.press/v235/tseng24a.html) 用高斯源数据展示不同维度与结构码本的**元素级 MSE**。看横轴与码本 bit 后，应回答两件事：球形分布为什么更契合格向量量化？为何高维空间的码本覆盖效率更好？但该图不是完整 LLM PPL/Token/s 图，不能直接把高斯向量的 MSE 排名当成 LLaMA 下游性能排名。

还有一项更直接且有控制变量的证据：[原文 Table 1（PDF 第 5 页）](https://arxiv.org/pdf/2402.04396) 固定为 **2-bit QuIP#、不做 Fine-Tuning、WikiText2、Context 4096**，仅比较使用 Hadamard 或 Fourier 做 Incoherence Processing。在 LLaMA-2 7B / 13B / 70B 上，Hadamard 的 PPL 分别是 **8.22 / 6.06 / 4.16**，Fourier 是 **8.30 / 6.08 / 4.17**。这支持作者在该配置下选 RHT 的经验结果，但**不是“有 RHT”与“无任何 Incoherence”**的消融，也不能把这组差值直接归因到 E8P 或 Fine-Tuning。

[QuIP# 正式论文 Figure 1（PDF 第 1 页）](https://arxiv.org/pdf/2402.04396) 还有另一层证据：横轴是**模型的压缩后总 Bits**，纵轴是 WikiText2 PPL，标注不同 bit 的 QuIP#、旧 QuIP 以及 AQLM 等对照。它说明实际 Model Size 与量化 PPL 的关系，不是在固定一个模型下单独证明 E8P 优于其他码本；不同 Model Size 的点可能还改变了参数规模。**Figure 2（PDF 第 2 页）** 只给 Hadamard → Lattice Codebook → Fine-Tuning 的方法流程，不可拿它代替 Runtime 或精度实验。

因此最终需要三层证据：Figure 3 的码本表示力；消融中去掉 RHT、Codebook 或 Fine-Tuning 的误差/任务质量；以及 RTX 4090 等硬件上的实际解码/带宽利用率。作者报告原型 CUDA 内核达到约峰值显存带宽的一半以上，那是具体 Kernel 测量而不是所有完整模型的一般吞吐承诺。
