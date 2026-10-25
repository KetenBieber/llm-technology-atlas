# QuaRot：为什么不删离群通道，而要旋转整个 Transformer 的坐标系？

> **原论文**：Ashkboos 等，[QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://papers.nips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html)，NeurIPS 2024 正式版。[正式 PDF](https://papers.nips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf) · [官方代码](https://github.com/spcl/QuaRot)。
>
> **技术位置**：继 [LLM.int8()](C039-llm-int8.md) 的高精度异常支路、[SmoothQuant](C040-smoothquant.md) 的通道缩放、[Atom](C044-atom.md) 的混合位宽之后，研究一种不额外保留 FP16 离群通道的 W4A4KV4 路线。与 [QuIP#](C049-quip-sharp.md) 的旋转相似，但后者主要是 Weight-Only + 格码本；QuaRot 还要真实处理 Activation 和 KV Cache。后续：[SpinQuant](C056-spinquant.md) → [FlatQuant](C057-flatquant.md)。

## 同样是 4 bit，为什么一个离群值会让其余 4095 个通道丢掉精度？

假设一个 Token 的四维 Activation 为 $x=[0.1,-0.2,0.3,8.0]$。最常见的**逐 Token 对称 INT4** 只能使用 $[-7,7]$ 的有符号整数，并用一把 Scale 描述所有通道：

$$
s=\frac{\max_i|x_i|}{7}=\frac87,\qquad
q_i=\operatorname{clip}_{[-7,7]}\!\left(\operatorname{round}(x_i/s)\right),\quad
\hat x_i=sq_i.
$$

于是前三个小值全被舍成零，最后一个被编码为 7；量化前后三个正常通道的信息全部消失。如果使用单独的 FP16 异常通道，保住了精度，却引入额外的小矩阵计算、数据拆分与低精度 Kernel 混合。问题不是“4 bit 不能表示小数”，而是**同一组共享 Scale 的动态范围被少数异常坐标占用了**。

[QuaRot Figure 1（PDF 第 2 页）](https://papers.nips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf) 正是在 LLaMA2-7B 第十层 FFN 输入上测这件事：横轴是 Hidden Dimension Index（约 4096 通道），纵轴是 Activation Value；蓝色 Min/Max、粉色 1/99 Percentile、橙色 25/75 Percentile。左侧部分通道的极值明显超出多数通道，右侧 QuaRot 之后极值变得更均匀。**这是一张坐标分布图，不是 PPL 或速度图**，只能直接支持“坐标范围更适合低 bit 均匀格点”。

真正的第一性问题变成：能否让同一个向量在另一组**完全等价的坐标**下表现得更均匀？

## 如果对输入做旋转，线性层的输出还会相同吗？

统一使用行向量右乘的约定：$X\in\mathbb R^{T\times d}$、$W\in\mathbb R^{d\times m}$，层输出 $Y=XW$。任选正交矩阵 $R\in\mathbb R^{d\times d}$，满足 $R^\top R=I$，构造

$$
X'=XR,\qquad W'=R^\top W.
$$

直接代入：

$$
X'W'=(XR)(R^\top W)=X(RR^\top)W=XW=Y.
$$

这里唯一需要的是 $R$ 是方阵正交、乘法关联和维度一致；**根本不需要模型重新学习语义**。如果后面要让 Residual Stream 始终以旋转坐标传播，则还需要在 Block 的输出端相应地右乘 $R$。例如原网络残差更新 $X_{\mathrm{next}}=X+F(X)$，旋转网络要实现

$$
X'_{\mathrm{next}}=X'+F'(X'),\qquad
F'(XR)=F(X)R,
$$

才能得到 $X'_{\mathrm{next}}=(X+F(X))R$。只给某一条支路随意加一个 $R$，却忘记最后的输出/残差坐标一致性，并不会保持等价。

### 二维反例：正交不会改变能量，却会改变哪个通道是“大数”

使用归一化 Hadamard 矩阵

$$
H_2=\frac1{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix},
\quad H_2^\top H_2=I.
$$

当 $x=[8,0]$ 时，$xH_2=[4\sqrt2,4\sqrt2]\approx[5.657,5.657]$。前向量范数都是 8，但最大坐标从 8 降为 5.657。**同一向量的 L2 范数没变，INT4 共用 Scale 的最大绝对值却可能减小**；这正是坐标系改变能改善量化，而不是消除信息的关键。

反过来，$x=[8,8]$ 变成 $[8\sqrt2,0]$，最大坐标反而变大。故“任何 Hadamard 都会消除任何离群值”是错误的。QuaRot 使用随机化符号的 Hadamard（RHT），依赖高维统计打散效果，而不是逐样本单调减少 $\ell_\infty$ 的定理。

## 为什么 RMSNorm 和 Residual 不是旋转的障碍，而 SwiGLU 却需要特殊处理？

LLaMA 使用 Pre-RMSNorm。先把 Norm 的可学习 Scale $\gamma$ 吸收到后面的 Linear，使 Norm 的非线性部分成为

$$
N(x)=\frac{x}{\sqrt{\frac1d\|x\|_2^2+\epsilon}}.
$$

对正交 $R$ 有 $\|xR\|_2=\|x\|_2$，所以

$$
N(xR)=N(x)R.
$$

原论文 `3.4 Eq. (3) 的转置约定与本文的行向量定义可能相反，但**可交换的根本条件就是 $\ell_2$ 范数正交不变**。未经吸收的逐通道 $\gamma$ 通常不与任意旋转交换，所以不能漏掉预处理这一步；LayerNorm 中的减均值也不能未经条件检查就照搬 RMSNorm 的证明。

看一个门控 FFN：

$$
u=N(x)W_{\rm up},\quad
g=\operatorname{SiLU}(N(x)W_{\rm gate}),\quad
z=g\odot u,\quad y=zW_{\rm down}.
$$

输入侧可以同时将 $W_{\rm up}$ 和 $W_{\rm gate}$ 左乘适当的逆旋转，且保持它们得到的 $u,g$ 不变；但**不能把任意正交矩阵强行穿过 $\operatorname{SiLU}(\cdot)$ 或 $\odot$**：

$$
\operatorname{SiLU}(zH)\neq\operatorname{SiLU}(z)H,\qquad
(aH)\odot(bH)\neq(a\odot b)H
$$

一般成立。QuaRot 的正确做法是在 **$z$ 计算完成之后**做在线 Hadamard，并把逆映射吸收到 $W_{\rm down}$。这在原论文 **Figure 2（原网络）、Figure 3（QuaRot FFN，PDF 第 4–5 页）** 形成清晰的一对对照。

~~~{mermaid}
flowchart LR
 X["Rotated residual X·R"] --> N["RMSNorm: norm invariant"]
 N --> UP["Up / Gate: absorb inverse R"]
 UP --> G["SiLU ⊙ Up"]
 G --> H["在线 Hadamard H"]
 H --> Q["逐 Token INT4"]
 Q --> D["Down: 吸收 H 的逆与输出 R"]
 D --> O["Rotated residual Y·R"]
~~~

*自制数据流示意，不是原论文 Figure。这里的“逆旋转吸收”指矩阵等价重参数化；实际 Weight 行列方向按对应源码 Layout 决定。*

## Attention 内为什么有两套不同的“旋转”，KV Cache 又怎样量化？

Attention 中 $\operatorname{softmax}(QK^\top/\sqrt{d_h})V$ 的 Q/K 与 V 扮演不同角色。对每个 Head 的 Q、K 同时右乘相同正交 $H$：

$$
(QH)(KH)^\top=QHH^\top K^\top=QK^\top.
$$

所以其 **Attention Scores 完全不变**。与此不同，将 $V$ 右乘 $H$ 会把 Head 的输出整体右乘 $H$；必须在 $W_O$ 里吸收 $H^\top$ 才能让最终输出回到原空间。

但是 [RoPE](../../A/03-transformer/A014-roformer-rope.md) 在 Q/K 投影之后施加随位置变化的旋转 $P_t$。通常 $P_tH\neq HP_t$，所以不能任意把该 $H$ 离线吸收到 $W_Q,W_K$，否则会改变位置关系。QuaRot 选择 **Post-RoPE Head-wise Rotation**：先完成 RoPE，再在运行时对 Q/K 做共同的 Hadamard。原论文 `4 Eq. (13)–(14) 指出，Decode 只有一个当前 Query，而 Key 有大量历史 Token；把 Key 缓存成**已应用位置编码和 Hadamard 的形式**，避免每步回头旋转全部旧 Key，是更合理的执行顺序。

对 Value，先从 $W_V/W_O$ 对吸收 Head-wise 旋转，再在需要时通过跨 Head Hadamard 帮助矩阵输入去异常。**某些变换可被离线吸收，另一些在线算子必须保留**，不能概括成“模型只做一次 Hadamard 就够了”。

原论文 `4 把精度路径区分得很清楚：线性层的输入可以按每 Token INT4 量化，存储的 Weight 为 INT4，在 CUTLASS TensorCore 做整数 GEMM（INT32 Accumulation）；RMSNorm、RoPE、Attention Scores/Softmax 和部分在线 Hadamard/缩放继续使用较高精度。KV Cache 使用 4-bit 存储，读取后可在 FP16 路径计算注意力。因此 **W4A4KV4 不等于整个网络每个算子都是 INT4**。

## 用 Shape 与实际字节数区分“KV 省内存”和“INT4 算得快”

设 $B$ 个 Request，每个可见历史长度 $S$，$L$ 层，$H_{kv}$ 个 KV Head，Head Dim $d_h$，每 KV Element 使用 $b$ bit。忽略 Scale/Index/Alignment：

$$
M_{\rm KV}=\frac{2BSLH_{kv}d_hb}{8}\ \mathrm{Byte}.
$$

假设 $B=2,S=2048,L=32,H_{kv}=8,d_h=128$。FP16 的 $b=16$ 需 $268{,}435{,}456$ Byte（256 MiB）；INT4 理想值为 64 MiB。这是**仅 KV payload 的 4 倍比例**，不是整机显存 4 倍下降；Weight、激活 Buffer、Kernel Workspace 与 Scale 仍在。

另外，Cache 压缩通常帮助 Memory-Bound Decode，而 W4A4 的 INT TensorCore GEMM 可以帮助计算量大的 Prefill。这两种收益来自不同资源瓶颈，需分开评价。[FlashInfer](C018-flashinfer.md) 的分页 KV、Block Size 与读取布局对真实效果有直接影响。

## 核对正式论文 Table 1/2 与 Figure 4，不靠摘要最大倍数

[正式 PDF Table 1（第 8 页）](https://papers.nips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf) 采用 WikiText-2、Sequence Length 2048；“Baseline”是不量化的 PPL，“QuaRot GPTQ”是 W4A4KV4 且无保留高精度异常通道。

| LLaMA-2 | FP16 PPL | QuaRot W4A4KV4 PPL | PPL 增量 |
|---|---:|---:|---:|
| 7B | 5.47 | 6.10 | +0.63 |
| 13B | 4.88 | 5.40 | +0.52 |
| 70B | 3.32 | 3.79 | +0.47 |

原论文 Table 1 另将 QuaRot-128G 与 Atom-128G 比较：7B 分别为 5.93 与 6.03；13B 都为 5.26。它只支持**这些 Group-128 配置的 PPL**，不能单凭两列数据宣称所有情形 QuaRot 更快。

**Table 2（同页）** 看的是六种 Zero-Shot 任务的 Avg，而非 PPL：70B 的 FP16 Avg 77.07，QuaRot Avg 75.98，绝对差为 1.09 分（百分比点口径）；7B 由 69.82 降至 65.64，相差 4.18 分。论文称保留约 99% 70B Zero-Shot 性能，与“任何模型准确率只下降 1%”不是同一命题。

**Figure 4（同页）** 才是 Runtime/Memory 证据，且只测单个 LLaMA-2 Transformer Block、RTX 3090：左图长度 2048，横轴 Batch 1、4、16、64，测 Prefill Speedup；右图 Batch 16、Decode 50 Token，横轴为不同输入长度，测 Peak Memory Saving Factor。文中最多 3.33× Prefill 加速、3.89× Decode 内存节省对应特定 Block/硬件设置，不可套用到 70B 整模型所有端到端场景。

## 哪里真正会失败？

- 对特定向量，Hadamard 反而可能放大某个坐标；“输出等价”并不能保证“量化误差减小”。后续 [SpinQuant](C056-spinquant.md) 研究不同随机旋转导致的量化性能方差。
- RMSNorm 的 $\gamma$、Output Projection、Attention 的 RoPE 前后顺序、Residual 坐标若漏掉某处吸收，所谓“等价网络”实际上已变。
- 低比特 Quantization 是非线性 Round；**旋转前后的全精度函数等价，不意味着量化后的输出逐位一致**。
- Kernel 未实现原生 INT4×INT4、Online Hadamard 访存代价太高或序列较短时，墙钟时间可能达不到论文的速度。
- 表中 PPL 与任务精度受样本、GPTQ、Clip、Weight Grouping 影响；QuaRot `5 用 Activation Clip 0.9、KV Clip 0.95、KV Group 128 等配置，比较时不能漏掉这些条件。

这一论文的核心贡献不是“Hadamard 能乘以权重”，而是**把原本只在单个 Weight Matrix 中工作的随机正交变换，做成跨 Residual、FFN、Attention、KV 的计算等价重参数化，再和真实 4-bit Kernel 连接起来**。 [SpinQuant](C056-spinquant.md) 正好追问：既然无穷多个正交基都保持原模型，为什么还要靠随机抽取其中一个？
