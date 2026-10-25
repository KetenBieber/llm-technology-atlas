# FlatQuant：为什么量化需要“平坦化”，以及 Kronecker 分解怎样让可学习仿射变换真正可部署？

> **原论文**：Sun 等，[FlatQuant: Flatness Matters for LLM Quantization](https://proceedings.mlr.press/v267/sun25l.html)，ICML 2025 正式版，PMLR 267:57587–57613。[可逐式查看的原始论文 HTML](https://arxiv.org/html/2410.09426) · [作者代码](https://github.com/ruikangliu/FlatQuant)。本文以正式 ICML 的公开实验结论与 arXiv HTML 公式、Figure 为依据；如版本表格细节不同，须按各自图表来源区分。
>
> **接续关系**：[SmoothQuant](C040-smoothquant.md)（Diagonal Scaling）→ [QuaRot](C055-quarot.md)（随机正交 Rotation）→ [SpinQuant](C056-spinquant.md)（可学习正交 Rotation）→ **本篇：逐层可学习的一般可逆仿射变换 + Kronecker 低开销结构**。另见 [OmniQuant](C047-omniquant.md) 的可学习 Clipping、[GPTQ](C041-gptq.md) 的固定校准输入下近似二阶误差补偿。

## 为什么压低最大通道值还不等于真正“适合量化”？

对一个单 Token Activation $x\in\mathbb R^n$，逐 Token INT4 采用同一个 Scale，由最大绝对值决定步长：

$$
\Delta(x)=\frac{\max_i|x_i|}{7},\qquad
\hat x_i=\Delta\cdot\operatorname{clip}\!\left(\operatorname{round}(x_i/\Delta),-7,7\right).
$$

如果把一个通道的绝对值从 100 降到 10，$\Delta$ 会缩小，显然有帮助。但设经过某次 Rotation 后，通道幅值从 $[100,1,1,\ldots]$ 变成 $[10,10,0.001,0.001,\ldots]$，虽然最大值大幅下降，绝大多数接近零的通道仍可能被 Round 到同一个点。**“没有极端 Outlier”并不保证权重/激活在量化格点上分布均匀**。

原论文 [Figure 1（arXiv HTML 的 `2）](https://arxiv.org/html/2410.09426) 将 LLaMA-3-8B / 70B 的 Output Projection 与 FFN Gate Projection 的 Weight/Activation 沿 Channel Magnitude 降序排列：横轴是通道排序，纵轴是该通道的 Frobenius Magnitude；Per-channel Scaling 和 Hadamard 在去掉最醒目的峰值之后，曲线仍会有陡峭斜率或长尾，而 FlatQuant 变换后的曲线更平。**Figure 1 只显示变换后的幅值分布，不直接测量端到端速度**。

原论文 [Figure 2（`2.2）](https://arxiv.org/html/2410.09426) 比较三种量化方案的 Error Surface：横向包含输入序列中的 Token 位置，纵向是 Transformer Layer。较早的 Pivot Tokens 与更深的层更容易有显著量化误差；FlatQuant 降低了局部峰值和沿层传播的误差。它说明更小的激活/权重峰值并非全部目标，**最后还要控制每层输出实际失真如何跨残差网络累积**。

因此问题升级：不是找一把统一的正交旋转，使每层所有数据看起来差不多，而是**为每个 Linear Layer 寻找更符合本层输入统计与量化误差的可逆坐标变换**。

## 从线性代数直接推出：为什么正交不是保持输出的必要条件？

沿用行向量约定，$X:[B T,n]$、$W:[n,m]$、$Y=XW$。对任何**可逆**矩阵 $P\in\mathbb R^{n\times n}$，

$$
Y=(XP)(P^{-1}W).
$$

这里 $P$ 完全不要求 $P^\top P=I$。只要它与逆矩阵在同一条线性计算路径中恰当配对，未量化的层输出严格不变。所以 FlatQuant 追求的是

$$
\min_{P\,\mathrm{invertible}}\big\|
XW-\mathcal D\big(\mathcal Q_a(XP),\mathcal Q_w(P^{-1}W)\big)
\big\|_F^2,
$$

其中 $\mathcal D$ 表示将整数 GEMM 的累加结果按各自 Scale 还原；如果省略这个解量化过程，就会把“两个整数代码矩阵的乘积”和物理层输出错误地等同。

**注意误差方向的配对**：$XP$ 改善 Activation 量化不一定意味着 $P^{-1}W$ 同时改善 Weight 量化。若 $P=\operatorname{diag}(c)$ 使某个通道的 Activation 缩小，其 Weight 一侧会按反向比例放大；这就是 [SmoothQuant](C040-smoothquant.md) 的基本取舍。非对角 $P$ 能在通道之间混合信息，表达空间更大，却更贵。

### 一个 2D 可复算例子

取

$$
X=[2,1],\qquad W=\begin{bmatrix}3\\4\end{bmatrix},\qquad
P=\begin{bmatrix}2&1\\0&1\end{bmatrix}.
$$

原输出 $XW=2\times3+1\times4=10$。另一方面：

$$
XP=[4,3],\quad
P^{-1}=\begin{bmatrix}1/2&-1/2\\0&1\end{bmatrix},\quad
P^{-1}W=\begin{bmatrix}-1/2\\4\end{bmatrix}.
$$

所以 $(XP)(P^{-1}W)=4(-1/2)+3(4)=10$。但它的坐标幅度从 $[2,1]$ 变成 $[4,3]$、Weight 从 $[3,4]$ 变成 $[-0.5,4]$；它仍然完全合法，却不保证量化误差更小。**作者要优化的是“量化后的误差”，而不是单纯找任何可逆矩阵**。

## 为什么直接学习一个完整 $n\times n$ 仿射矩阵不可行？

假设 $n=4096$。如果每个待量化 Linear Layer 都存一套完整 $P$，单个矩阵需要 $4096^2=16{,}777{,}216$ 个参数；FP16 约 32 MiB。除了要离线计算 $P^{-1}W$，前向仍必须运行 $XP$，每个 Token 约有 $n^2$ 量级的额外乘加。若一个 Transformer Block 有若干个 Linear Layer，每层都保留如此巨大的在线变换，可能把 INT4 GEMM 省下的时间重新花在大量 FP16 Matmul 上。

这正是 FlatQuant 相比“每层学一套任意仿射矩阵”多出的一项核心系统贡献：**让表达力较强的坐标变换具备低存储、低开销的可实现结构**。

## Kronecker Product：两个小矩阵如何代表一个大矩阵？

令 $n=n_1n_2$，限制

$$
P=P_1\otimes P_2,\quad
P_1\in\mathbb R^{n_1\times n_1},\;
P_2\in\mathbb R^{n_2\times n_2}.
$$

这是一个受约束的 $n\times n$ 可逆矩阵。若两个小矩阵都可逆，那么

$$
(P_1\otimes P_2)^{-1}=P_1^{-1}\otimes P_2^{-1}.
$$

关键并非“Kronecker 减少 Full Linear 的输出维度”，而是它允许把一次长度 $n$ 的 Transform 改写成**两次小规模矩阵乘法**。

把一个 Token 的长度 $n$ 向量按行主序重新排列为 $V:[n_1,n_2]$。使用行向量展平约定，有恒等式

$$
\operatorname{vec}_{row}(V)\;(P_1\otimes P_2)
=\operatorname{vec}_{row}(P_1^\top V P_2).
$$

右边先用 $P_1^\top$ 作用于第一个小轴，再用 $P_2$ 作用于第二个小轴。实现时要对齐 Tensor Memory Layout 与 Vec 方向，否则误把 Column-Major 的 Kronecker 恒等式用于 Row-Major，可能构造出数值不等价的 Kernel。论文 `3.1 Eq. (3) 用 $\times_1,\times_2$ 张量乘法记号表达了相同思想。

~~~{mermaid}
flowchart LR
 X["Token activation: [T, n]"] --> V["reshape: [T, n1, n2]"]
 V --> A["小矩阵乘 P1 转置"]
 A --> B["小矩阵乘 P2"]
 B --> Q["INT4 quantize / pack"]
 W["FP Weight [n, m]"] --> WI["离线 P inverse 变换"]
 WI --> WQ["量化权重"]
 Q --> GEMM["原生 W4A4 GEMM"]
 WQ --> GEMM
 GEMM --> OUT["还原 Scale 后输出"]
~~~

*自制 Tensor/Kernel 示意，非原论文 Figure；小矩阵按不同轴的乘法方向以正文的 $\operatorname{vec}_{row}$ 约定为准。*

### 4096 维的成本到底差多少？不靠“轻量”两个字

当 $n=4096$ 时，取 $n_1=n_2=64$：

$$
N_{\rm dense}=n^2=16{,}777{,}216,\qquad
N_{\rm kron}=n_1^2+n_2^2=8192.
$$

只存变换矩阵参数时，体积减少 $16{,}777{,}216/8192=2048$ 倍；它并不表示整体模型压缩 2048 倍，因为这只是**原本打算新增的 Transform 元数据**。对一个 Token，完整 $P$ 应用约 $n^2$ 次标量乘加；两次小矩阵相乘的代价约 $n(n_1+n_2)=4096\times128=524{,}288$ 次，理想算术规模减少 32 倍。

更一般地

$$
\frac{n^2}{n_1^2+n_2^2}
\le\frac n2,\qquad
\frac{n^2}{n(n_1+n_2)}
\le\frac{\sqrt n}{2},
$$

两个上界都在 $n_1=n_2=\sqrt n$ 达到。为什么？AM–GM 给出 $n_1^2+n_2^2\ge2n_1n_2=2n$，以及 $n_1+n_2\ge2\sqrt n$。公式直接解释了“尽量选两个接近平方根的因子”这一 Kernel 设计偏好。对于 $n=8192$ 无法恰好平方拆为两个整数幂二，所以论文 `3.1 给出 $64\times128$ 的方案。

**但** $\mathcal O$ 数量下降不等于 GPU 延迟严格按 32 倍下降：两个小矩阵乘可能是 Memory-Bound，额外 Transpose/Reshape 可能破坏连续访问，因此需要本论文的 Kernel Fusion 才能将成本转化为实际 Speedup。

## 为何除 Kronecker，还要通道 Scale 与学习 Clip？

FlatQuant `3.1 不只学 $P_1,P_2$，还组合了两种独立自由度：

**Learnable Per-channel Scaling**：$\operatorname{diag}(c)$ 允许各 Channel 的幅值作不同缩放，相应反向缩放并入其它权重。这样不必期待一个 Kronecker 结构独自适配所有 Weight/Activation 的范围差异。

**Learnable Clipping Thresholds**：在有限离散 INT4 格点上，适当压掉极少数最大值，可以降低绝大多数通道的量化步长。设量化阈值 $t$，实际误差可写成

$$
E(t)=\sum_i\big(x_i-\Delta(t)\,
\operatorname{round}(\operatorname{clip}(x_i,-t,t)/\Delta(t))\big)^2
$$

其中 $\Delta(t)=t/7$ 是教学上以对称 INT4 为例的形式。减少 $t$ 会使正常范围的量化误差下降，但极端大值的 Clip 偏差变大；$E(t)$ 通常**不是单调函数**，应在校准集上优化而不是固定“一律 Clip 90%”。论文同时为 Weight 和 Activation 训练 Clip 参数，并联合学习 $P,c$。

原文 `3.1 Eq. (4) 最终的训练目标不是把“各通道方差差距”直接作为唯一 Loss，而是针对第 $\ell$ 个 Transformer Block：

$$
\min_{\Theta_\ell}
\left\|F_\ell(X)-\widehat F_\ell(X;\Theta_\ell)\right\|_F^2,
\quad \Theta_\ell=\{P_1,P_2,c,\alpha_w,\alpha_a\}_{\rm layer}.
$$

真实模型的原精度 Block $F_\ell(X)$ 作为教师、量化后的 Block $\widehat F_\ell(X;\Theta_\ell)$ 作为学生；优化的是**量化后 Block 输出重建损失**。Layer 之间误差会传播，这就是为什么只有局部权重 MSE 小而实际语言模型 PPL 仍可能很差。

## 非正交变换为什么不能像 QuaRot 一样穿过每个 Norm 和 SwiGLU？

对于任意可逆 $P$，一般有 $\|xP\|_2\ne\|x\|_2$，所以

$$
N(xP)\ne N(x)P
$$

通常成立。若像 QuaRot 那样把任意 $P$ 放入全网络 Residual Stream 并指望每个 RMSNorm 可交换，就会改变函数。**FlatQuant 的不变性来自适当位置的 $P,P^{-1}$ 成对嵌入单个线性映射路径，而不是正交 Norm 等价性**。跨层重参数化时只有论文 `3.2 指明的特殊 Weight/Norm/投影配对与可融合条件才可以合法搬动。

FFN 的 $\operatorname{SiLU}$ 与 Hadamard Product 也不能一般地和 $P$ 交换；Position Encoding 的 RoPE 不能无条件对一般非正交 $P$ 保持相同 Attention Score。作者 Figure 3(b) 将 Attention、FFN、Norm 中能合并的部分标成红色，Online Transformation/Quantization 标成蓝色，通道 Scale 合并标成绿色。**这张图与一条简单的“X→P→INT4→P逆”流程图不是同一细节层次**：只有遵循实际融合位置，才能保证未量化网络的相应层是数值等价的。

## 为什么还要把 Transformation 与 Quantization 融合到单个 GPU Kernel？

设原始实现分别执行 `reshape → P1 matmul → P2 matmul → INT4 round → INT4 pack`，每步把中间结果写回 HBM，再让下一步从 HBM 读回来。虽然浮点 FLOPs 变少了，这些额外 Global Memory Round-trip 仍可能成为主要时间。

论文 `3.3 的 **Fused FlatQuant Kernel** 将按两个轴的小矩阵计算与 Quantization 放进一次 Launch/数据流中，减少中间结果的显式 HBM 写入，并兼容 PreFill 和 Decode。对于形状 `[B,T,n]`，Prefill 时 $BT$ 通常大，有机会在并行 Thread Blocks 间分摊 Transform；Decode 时当前 Token 少，Launch Overhead 与数据访问 Pattern 更容易主导。**权重 W4A4 的 GEMM 是另一项成本**，不能因变换 Kernel 快就省略整个模型的矩阵乘测量。

论文 Figure 4（`4.3）在 **RTX 3090** 上比较 FP16 与 INT4 的 Prefill/Decode 速度，Prefill 长度 2048、Decode 256 Token，在不同 Batch 下最高分别约 **2.30× 与 1.76×**（有 Kernel Fusion，Batch 64）。同样模型、同样 Batch 下才能评价；也不能把单独的变换 Kernel Speedup 当成完整系统吞吐。

## Table 1 / Table 2 / Table 3：三种完全不同的证据

[原始 Table 1（arXiv HTML `4.2）](https://arxiv.org/html/2410.09426) 比较不同模型在 W4A4 上的 WikiText-2 与 C4 PPL；为避免把“量化器更新方式”与“仿射变换”混在一起，作者分别报告 **RTN 与 GPTQ**。

例如 LLaMA-3-8B 的 WikiText-2 PPL：

| 方法 | 权重量化器 | PPL |
|---|---|---:|
| FP16 | — | 6.14 |
| QuaRot | RTN | 10.60 |
| SpinQuant | RTN | 7.96 |
| FlatQuant | RTN | 6.98 |
| QuaRot | GPTQ | 8.16 |
| SpinQuant | GPTQ | 7.39 |
| FlatQuant | GPTQ | 6.90 |

这张表表明 FlatQuant 即使使用便宜的 RTN 也可能非常接近使用 GPTQ 的结果，但**并不证明 FlatQuant 在任何模型、任何量化设置都可以取消 GPTQ**。

**Table 2（Zero-Shot QA）** 给一个更严苛的例子：LLaMA-3-70B 的六任务平均，FP16 为 79.95；QuaRot RTN 35.36、SpinQuant RTN 65.66、FlatQuant RTN **79.01**。这里 FP16 与 FlatQuant 的差距仅 0.94 个百分点，但不能把它混成 Table 1 的 PPL 变化。作者重新运行 QuaRot/SpinQuant 的官方实现并在统一 `lm-eval-harness` 设置下评估，以减少不同论文测评协议导致的不可比性。

**Table 3（LLaMA-3-8B 消融）** 从不加有效方法的 W4A4 RTN 开始：WikiText-2 PPL **1266.60**；只打开 Learnable Transform (LT) 后变成 **8.50**；再加 Per-channel Scaling (PS) 为 **7.95**；LT 与 Learnable Clipping Thresholds (LCT) 配合为 **7.11**；三者同时启用为 **6.98**。可以直接看到哪项设计在这一极端条件下避免了量化网络崩溃，哪项只是进一步修补最后小段精度差。消融不等于数学证明，也不保证相同组件在每个模型上的收益顺序不变。

## 校准预算、复杂度与失败条件

论文 `4.1 的典型校准是 **128 条 WikiText-2 句子，每条采样 2048 Token，15 个 Epoch，Batch 4，AdamW 初始学习率 $5\times10^{-3}$**；LLaMA-3-8B 的默认流程在单 GPU 上约需 26GB 显存和 0.9 小时。与 QuaRot 的基本离线随机重参数化相比，这里新增了需要优化的仿射参数与逆矩阵；与 SpinQuant 不同，它不受全局正交约束，因此对**可逆性、病态数值与逆矩阵计算**更敏感。

工程上至少有四类失效模式：

- **可逆性与条件数**：$P_1,P_2$ 接近奇异时，$P^{-1}W$ 可能产生巨大数值，反而放大量化误差；原论文 Appendix B.1 用矩阵分解/训练技巧控制这一问题，不能直接把随机矩阵当作已训练的 FlatQuant。
- **结构代价**：$P_1\otimes P_2$ 只是整个 $n\times n$ 可逆空间的很小子集，便宜不代表任意目标都可表达。实际最优比还受 GPU 的 Tile、Transpose、Alignment 影响。
- **输入分布迁移**：校准集中对某类 Pivot Tokens 与 Layer 的误差优化，不能确保在别的长上下文任务中获得完全相同的 PPL/Zero-Shot 差距。
- **核函数条件**：如果不能使用作者的 Fused Transformation/Quantization Kernel，而使用普通框架分步计算，运行时额外的读写可能抵消 INT4 GEMM 的收益。

因此这三篇形成了完整但互不替代的探索路线：**QuaRot 回答“能否旋转到更好量化的坐标系”，SpinQuant 回答“如何在正交约束下搜索更好的一套旋转”，FlatQuant 则进一步回答“能否放宽到低成本、逐层可学习的可逆仿射，把局部量化失真与真实 Runtime 一起考虑”。** 三者都需同时核查全精度等价、低比特误差与端到端速度，不可只比较三篇摘要里的最大 Speedup。
