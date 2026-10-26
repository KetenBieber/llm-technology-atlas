# SpQR：稀疏高精度异常权重为什么比全局提高位宽更划算？

> **原始论文**：Dettmers 等，[SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1787533e171dcc8549cc2eb5a4840eec-Abstract-Conference.html)，ICLR 2024。[作者 PDF：Figure 2、Algorithm 1、Figure 3、Table 1](https://spcl.inf.ethz.ch/Publications/.pdf/2024_iclr_spqr.pdf)。


前置：[GPTQ 的二阶误差补偿](C041-gptq.md)、[LLM.int8 的异常通道分流](C039-llm-int8.md)。下一篇：[QuIP# 的正交变换与码本](C049-quip-sharp.md)。

## 为什么“只有极少数值难以量化”能够改变存储设计？

如果一个 Weight Matrix 有 100 万个标量，仅 0.1% 造成显著量化重建误差，那么把所有数从 3-bit 升级到 4-bit，新增约 100 万 bit；而让这 1000 个特别敏感的元素采用高精度保存，其原始额外值数据近似只有 $1000\times(16-3)$ bit，但还要存索引与附加元信息。

因此真正的比较是

$$
M_{\mathrm{SpQR}}=
M_{\mathrm{dense\ lowbit}}+
M_{\mathrm{outlier\ values}}+
M_{\mathrm{indices}}+
M_{\mathrm{group\ metadata}}.
$$

*如果异常项足够稀少*，这种表示既可以保留主要低比特权重压缩收益，又能显著抑制少数高损失权重的误差。这解释了论文名字中的 **Sparse-Quantized Representation**：不是把整个模型剪枝成 Sparse Transformer，而是把**稀疏例外放进高精度支路、主体保持低比特密集存储**。

## 先算一笔账：稀疏例外何时真的比统一提升 Bit 更省？

设一层有 $N=1{,}000{,}000$ 个权重。统一从 3-bit 提升到 4-bit，需要额外 $N$ bit，约 125,000 byte。若仅把 $0.1\%$ 即 1000 个例外从 3-bit 改为 16-bit，高精度值本身新增 $1000(16-3)=13{,}000$ bit；如果还为每个例外保存 20-bit 索引，则索引又新增 20,000 bit，总计约 33,000 bit，即约 4125 byte。**同样的主体位宽下，稀疏例外可能便宜很多**。

但这是教学成本估算：真实 SpQR 的压缩格式可能还需存 Offset/Group Scale、Index 编码、排列元信息和对齐 Padding，而且部分例外可能与低比特密集值共存成附加残差。若例外率提高到 5%，上面的值与索引总成本变为 $50{,}000\times33=1{,}650{,}000$ bit，已超过统一多加 1 bit。于是**究竟隔离多少元素**本身是内存—精度的优化问题，不是“异常项越多越保险”。

### 只有权重绝对值大才算 Outlier 吗？

不一定。对单一输出位置

$$
y_n=\sum_kx_kw_{kn},\qquad
\Delta y_n=\sum_kx_k(w_{kn}-\hat w_{kn}).
$$

若输入通道 $k$ 的典型 $|x_k|$ 很小，即便 $w_{kn}$ 自身数值巨大，某些请求的输出误差也未必大。相反，在高敏感输入通道，即使原权重不特别巨大，量化后残差 $\delta_{kn}=w_{kn}-\hat w_{kn}$ 也可能伤害输出。论文所谓**量化 Outlier** 更接近造成严重量化失真的权重，而非任意大绝对值元素。要区分 Weight Outlier、Activation Outlier 和影响下游任务的真正重要权重——这三者不是同一个集合。

## 异常值为什么不只看绝对大小？

假设某个权重 $w_{ij}$ 量化残差 $e_{ij}=w_{ij}-\hat w_{ij}$ 很大，但对应输入通道很少被激活，它对 $XW$ 的损失可能仍较小。反之，绝对值不大的权重如果其错误发生在输入高敏感方向，输出偏差可能更大。SpQR 研究**导致大重建误差的量化离群项**并存储为例外，不能将它简化成“把最大的那 1% 权重 FP16 保存”。

给定 $W$ 和输入 $X$，如果量化得到 $\hat W$，输出误差近似

$$
\mathcal E=\|X(W-\hat W)\|_F^2.
$$

SpQR 的选择机制要考虑量化误差、校准信息和高精度例外的成本。若将所选位置集合写作 $\Omega$，可概念性表示

$$
W\approx Q_b(W_{\overline\Omega})+R_\Omega,
$$

其中 $R_\Omega$ 是仅在少数位置非零的高精度残差；这只表达**数据结构**，不是论文整个量化求解程序。所有例外都放回原坐标后，Linear 输出可以被实现为低比特密集矩阵运算加少量稀疏贡献。

### 两级量化：为什么 Group 很小，元数据却没有膨胀到不可接受？

先不考虑稀疏异常项。若一组只有 $\beta_1=16$ 个 Weight、每个 Weight 3 bit，那么每 16 个值要另存一组 Scale 和 Zero Point。若两者都直接用 FP16，元数据就是 $32/16=2$ bit/weight，等于把名义的 3-bit 提升到 5-bit，失去了许多压缩收益。**小 Group 提高保真度与量化元数据暴涨，原本是一对冲突目标。**

SpQR 的第二项机制是：**连 Scale 和 Zero Point 本身也再次分组量化**。作者 §4.1 与 Algorithm 1 先为每 $\beta_1$ 个普通权重拟合一次量化参数，再对连续 $\beta_2$ 组的 Scale / Zero Point 分别拟合第二级参数。第一级的异常权重须先从统计量拟合中排除；而第二级参数的误差尽可能由之后的 GPTQ 式权重补偿吸收。不能简单地把第一级元数据保留 FP16，然后宣称还实现了 SpQR 的压缩率。

举一个**只核算存储、不代表完整格式**的例子：取 $b_w=b_q=3,\ \beta_1=\beta_2=16$。256 个基础权重编码为 $256\times3=768$ bit；16 组第一级 Scale/Zero Point 各 3 bit，需要 $16\times2\times3=96$ bit；为这些量化后的统计量分别保存二级 Scale/Zero Point，各 2 个 FP16，即再用 $4\times16=64$ bit。于是

$$
b_{\rm dense+stats}=\frac{768+96+64}{256}=3.625\ {\rm bit/weight}.
$$

这与原文 **Figure 3** 的分层存储逻辑一致，但还**没有**加 CSR 异常值/索引、对齐和封装开销，所以不是发表论文里某个完整模型的实际平均 Bit Rate。选择更小 Group 或更高异常率，仍要重新核算真实成本。

~~~{mermaid}
flowchart TD
 W["256 个权重"] --> G["16 组：每组 16 个权重"]
 G --> B["主体：每值 3-bit"]
 G --> P["每组 Scale/Zero"]
 P --> Q["16 组统计量 → 再量化为 3-bit"]
 Q --> M["第二级 Scale/Zero：4 个 FP16 数"]
 B --> PACK["打包 Dense 主体 + 元数据"]
 M --> PACK
 E["少量高敏感 Weight"] --> CSR["CSR 高精度例外"]
 CSR --> OUT["推理时稀疏贡献"]
 PACK --> OUT
~~~

*自制二级量化数据图，非原论文 Figure 3；图中展示一组 256 个权重的理想化统计量层级，CSR 例外单独计费。*

### 例外究竟按什么判定：权重值大，还是“移除后误差下降”？

作者的 [Algorithm 1（PDF 第 7 页）](https://spcl.inf.ethz.ch/Publications/.pdf/2024_iclr_spqr.pdf) 以输出通道为行，令 $W:[m,n]$、校准激活 $X:[n,d]$，构造 $H=2XX^\top$ 并利用 $H+\lambda I$ 的逆 Cholesky 因子评估量化代价。对一个候选位置，比较**该位置参与低比特量化**与**暂时把它从当前组中移除、重估剩余项的量化参数**两种方案；只有带来的误差下降超过阈值 $\tau$，才把它记录成例外。

因此异常项是一种**对反事实量化损失的选择**，而不是单纯筛出最大 $|w|$。随后低比特量化会继续修改还未量化的权重；最终存储的稀疏高精度值可能已被 GPTQ 式反馈更新过，不能保证等于原始权重的 FP16 字节序列。

在实际布局中，作者 §4.2 采用 **CSR 按行存例外 + 密集低位主体**；为了使 Dense Kernel 连续访问，还可保留异常位置原来的低位码，并让稀疏高精度项做相应修正。因此前面的 $Q_b(W_{\overline\Omega})+R_\Omega$ 应读作**等效输出分解**，而非“异常位置在主数组里一律没有任何编码”的内存承诺。

~~~{mermaid}
flowchart TD
 W[FP16 Weight] --> E[估计量化后的敏感例外]
 E --> D[多数权重：分组低比特]
 E --> S[少数 Outlier：高精度稀疏例外]
 D --> L[高效低比特权重读取]
 S --> R[稀疏纠错访问]
 L --> Y[Linear 输出合并]
 R --> Y
~~~

*自制机制图，非原论文原图。需同时读论文存储格式的元信息编码与内核对两种表示的融合方式。*

## 压缩格式为什么决定 Kernel 成败？

把一层分成 Dense Low-Bit 主体和稀疏例外以后，计算变成近似

$$
XW\simeq X\widehat W_{\mathrm{dense}}+XR_{\mathrm{sparse}}.
$$

如果实际的例外采用“替换低比特值”而不是“附加误差”，实现须注意不要把同一个位置算两次。典型高效流程是在低比特密集 GEMV/GEMM 中展开 Weight Pack，再对例外位置作小量高精度贡献并合并结果；索引访问如果高度离散，会造成非合并访存和 Warp 分歧。这个困难属于系统问题，不能被“只保留 0.1% 异常值”一句话掩盖。

~~~{mermaid}
flowchart LR
 D[3–4 bit 密集权重块] --> Q[Packed Low-bit GEMV]
 I[稀疏索引与 FP16 例外] --> S[Gather / Sparse Accumulate]
 X[Activation X] --> Q
 X --> S
 Q --> A[输出融合]
 S --> A
~~~

*自制数据布局图，不是原论文 Figure；需要根据实际稀疏格式决定例外是替换或修正路径。*

## 性能为什么要同时考虑压缩比与例外查找？

低 bit 存储节省的 HBM 带宽，可能被稀疏索引随机访存、分支和解包破坏。如果每个 Outlier 都使 GPU Thread 需要单独跳转地址，局部 Memory Coalescing 会很差，甚至比纯 FP16 Kernel 慢。作者因此不仅提出量化表示，还提供能有效解码与推理的 GPU 方案；如果只复现量化权重的 PPL，不能声称复现了整个 SpQR 系统。

[ICLR 2024 正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1787533e171dcc8549cc2eb5a4840eec-Abstract-Conference.html) 报告在 LLaMA/Falcon 等模型的**特定完整压缩配置**中取得低于约 1% 的相对 PPL 损失，展示 33B 模型在 24GB 消费级 GPU 上运行的案例，并报告特定配置相比 16-bit 基线约 15% 加速。它不意味着**仅用 3-bit 基础码且不计统计量、索引时**也能获得该精度；作者 Table 1 的接近无损设置实际约为 **4.6–4.71 有效 bit/weight**。

**研究问题**：在相同有效 bits/weight（包括索引和 Scale）下，和 [GPTQ](C041-gptq.md) 或 [AWQ](C042-awq.md) 比差多少？若异常项比例从 0.1% 升到 10%，稀疏格式是否仍有意义？[QuIP#](C049-quip-sharp.md) 则尝试通过随机正交变换把误差与异常值在坐标轴上摊平，而非额外存储例外。

## 论文 Figure/结果应支持什么，不能支持什么？

[ICLR 2024 正式版](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1787533e171dcc8549cc2eb5a4840eec-Abstract-Conference.html) 声称的是在若干 LLaMA/Falcon 模型中以约 3–4 bit/weight 获得相对于高精度基线 **小于约 1% 的相对 PPL 损失**，并提供专用 GPU 解码内核。在 24GB GPU 上运行 33B 且报告约 15% 加速，是**特定硬件、量化配置和基线**，不能推广到所有小模型、Batch 与解码长度。

重现实验应画两条而非一条曲线：横轴均使用**包含稀疏索引与 Scale 的有效 bits/weight**；第一条纵轴是 PPL 或任务损失，第二条是同硬件 Token/s。若一项方法在 PPL 上更好但 Kernel 比 FP16 更慢，它只证明更强压缩表示而未完成系统性能目标。

### 原论文 Figure 2 与 Algorithm 1 应怎样连起来？

在 [SpQR 作者 PDF 第 5 页 Figure 2](https://spcl.inf.ethz.ch/Publications/.pdf/2024_iclr_spqr.pdf)，作者把 LLaMA-65B 最后一个 Attention 层输出投影的 **Weight Log-Sensitivities** 画成热力图：横轴是输入维度，纵轴是输出维度；局部窗口展示 Attention Head 的竖向敏感带、Row/Column Outlier 和与 RoPE 相关的周期模式。**深蓝色**代表更高敏感性，不能把这些色带直接当成 Weight 绝对值或 Token Attention Map。不同缩放的局部图也不能直接拿色块宽度当作全模型异常比例。

接着读同一 PDF 的 **Algorithm 1（PDF 第 7 页）**：输入有权重 $W$、校准激活 $X$、位宽 $b$、两个 Group Size、异常项灵敏度阈值 $\tau$ 和 Hessian Regularizer $\lambda$。其 $X:[n,d]$ 是**列为校准样本**的布局，因此误差曲率应是 $XX^\top$，不是未经转置的 $X^\top X$。算法经 Cholesky 相关变换计算敏感性，再寻找例外项、执行分块低 bit 量化与误差补偿，最后返回稀疏高精度项以及多级 Scale/Zero Point。

Figure 2 为**异常项存在复杂结构**提供观察，Algorithm 1 说明**怎样识别并压缩**；只有独立的 PPL、占用和 Kernel Benchmark 才能支持“接近无损且更快”这一最终效果。

**决策点**：如果难量化元素特别少，用 SpQR 单独存例外；如果重建误差遍布整个权重矩阵，则要转向 [QuIP#](C049-quip-sharp.md) 的坐标混合，试图改变误差的集中性。它们不是先后必然继承，而是针对不同误差结构的替代策略。
