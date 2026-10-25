# Mamba-2：从状态递推严格推导出类 Attention 矩阵，Structured State-Space Duality 究竟证明了什么？

> **原论文**：Tri Dao、Albert Gu，**Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**，ICML 2024，PMLR 235:10041–10071。[正式出版页](https://proceedings.mlr.press/v235/dao24a.html) · [公开 PDF：arXiv:2405.21060v1](https://arxiv.org/pdf/2405.21060) · [作者代码](https://github.com/state-spaces/mamba)。
>
> **前置**：[S4：连续 SSM、离散化与固定卷积](A058-s4.md) → [Mamba：输入决定状态传播](A059-mamba.md) → **本文：特殊选择性 SSM 与结构化 Masked Attention 的严格对偶及 Mamba-2 架构**。对照：[Transformer](../03-transformer/A013-transformer-attention-is-all-you-need.md)。

## 1. 论文标题说“Transformers are SSMs”，是宣称 Softmax Attention 与 Mamba 完全相同吗？

不是。原文 Introduction 脚注专门提示：这些连接涉及**某些特定形式的 Attention**，而非所有 Softmax Transformer；标题是对早期 “Transformers are RNNs” 的致意。

为避免把一个漂亮标题误认为全模型等价，先把问题分三层：

1. **表达式**：输入序列 $x_0,\ldots,x_{T-1}$ 到输出 $y_i$ 的线性/条件线性映射，可否同时写成递推与矩阵乘法？
2. **计算方法**：同一个映射能否在 Decode 用递推、在训练时把局部块改写为矩阵乘，以便充分使用 GPU？
3. **架构**：是否可以利用这套数学联系，设计实际更高效、表达能力不降的语言模型层？

[Mamba](A059-mamba.md) 已解决了输入选择导致的时间变化 SSM 如何训练；Mamba-2 的重要创新不只是优化它的 CUDA 实现，而是提出 **Structured State-Space Duality（SSD）**，建立上述三层之间可复用的严格对应关系。

[原论文 Figure 1（PDF 第 2 页）](https://arxiv.org/pdf/2405.21060) 将 SSM、Attention、Structured Matrices 放在三角关系图里：Semiseparable Matrices 将 SSM 变成结构矩阵问题；Structured Masked Attention 将 Attention 也连接进来；二者的交集构成 SSD；再往上设计高效算法，往下得到 Mamba-2。**它不是 Venn Diagram 上 SSM 与所有 Attention 完全重合**。

## 2. 从一个具有输入选择的状态方程推导完整历史依赖

沿 [Mamba](A059-mamba.md) 的记号，设每步线性状态更新

$$
h_t=A_th_{t-1}+B_tx_t,\qquad y_t=C_t^\top h_t.
$$

其中 $h_t\in\mathbb R^N$，$A_t\in\mathbb R^{N\times N}$，$B_t,C_t\in\mathbb R^N$，先考虑标量输入 $x_t$。当 $h_{-1}=0$：

$$
h_0=B_0x_0,
$$

$$
h_1=A_1B_0x_0+B_1x_1,
$$

$$
h_2=A_2A_1B_0x_0+A_2B_1x_1+B_2x_2.
$$

因此一般有

$$
y_i=\sum_{j=0}^{i}
C_i^\top A_iA_{i-1}\cdots A_{j+1}B_j\,x_j,
$$

其中 $j=i$ 时空乘积是 $I$。

定义 **Token-to-Token 变换矩阵**

$$
M_{ij}=\begin{cases}
C_i^\top (A_iA_{i-1}\cdots A_{j+1})B_j,&i\ge j,\\
0,&i<j,
\end{cases}
\qquad Y=MX.
$$

这一步已经证明：**给定这一输入样本产生的 $A_t,B_t,C_t$，状态递推可展开为一个 Causal Lower-Triangular 矩阵**。它与 [S4](A058-s4.md) 的固定卷积核不同，$M_{ij}$ 的系数允许同时依赖绝对位置与内容；当这些参数由 $X$ 生成时，整个神经层作为 $X\mapsto Y$ 的映射一般并非线性的，但**条件于参数序列**可以写成矩阵乘。

不应把“对每条输入可以定义一个 $M(X)$”误说成“训练完后有一个固定的 $M$ 能对任何输入复用”。

## 3. 什么特殊结构让矩阵 $M$ 成为“类似 Attention”的形式？

关键不是强行让所有矩阵 $A_t$ 对角化，而是考虑一个更受约束、但仍能学习输入选择的子类：

$$
A_t=a_tI_N,\qquad a_t\in\mathbb R.
$$

现在状态传播乘积恰好等于

$$
A_i\cdots A_{j+1}
=\left(\prod_{k=j+1}^{i}a_k\right)I_N.
$$

于是

$$
\boxed{M_{ij}=
\mathbf1[i\ge j]\,(C_i^\top B_j)
\prod_{k=j+1}^{i}a_k.}
$$

把这条公式拆成三部分：

- $C_i^\top B_j$：当前输出向量 $C_i$ 与历史输入向量 $B_j$ 的匹配，形式上相当于一个 **Query–Key Inner Product**；
- $\prod_{k=j+1}^ia_k$：从 $j$ 到 $i$ 的沿途保留/遗忘系数，依赖每一步的时间变化状态；
- $\mathbf1[i\ge j]$：只能看见过去和现在的 Causal Mask。

令

$$
L_{ij}=\mathbf1[i\ge j]\prod_{k=j+1}^{i}a_k,
$$

则

$$
\boxed{M=L\odot(CB^\top).}
$$

这就是 **Structured Masked Attention** 的一种特例：不是先经 Softmax($QK^\top$) 归一化，而是用具有特殊乘积结构的 $L$ 对点积分数施加逐元素权重。原论文 §4–5、[Figure 4（PDF 第 18 页）](https://arxiv.org/pdf/2405.21060) 将这种结构称作 SSD 的核心交集。

### 重点纠错：Softmax 不能被一个符号“≈”偷偷抹掉

标准 Softmax Attention 有

$$
A^{\rm softmax}_{ij}
=\frac{\mathbf1[i\ge j]\exp(q_i^\top k_j/\sqrt d)}
{\sum_{r\le i}\exp(q_i^\top k_r/\sqrt d)}.
$$

而 SSD 的上述系数是 $L_{ij}(C_i^\top B_j)$，**没有这个指数函数以及依赖整个前缀的分母**。因此它们的分数可负、行和不一定为 1，不能用同一个 Query–Key 符号就宣称二者数学等价。作者探索的是**特定 Structured Masked Attention 与状态模型的对偶**，不是推翻 Softmax 非线性。

## 4. Semiseparable 到底是什么意思？用一个四步矩阵看清楚

一个矩阵的**某些严格下三角子块可以表示成低秩乘积**，就是 Semiseparable 结构的关键。对前文的 Scalar-Identity SSM，任意隔开的左下矩形 $M_{i:j,\;0:k}$ 都会拥有通过其公共时间边界传播的低维状态，因此它可以因式分解为“从旧输入压缩到边界状态”乘“从边界状态传到未来输出”。

当使用纯标量保留 $a_t$ 时，最基本的掩码 $L$ 的这些子块具有 **Rank≤1**。因为对任何边界 $b$、过去 $j<b$、未来 $i\ge b$：

$$
L_{ij}
=\left(\prod_{k=b}^{i}a_k\right)
\left(\prod_{k=j+1}^{b-1}a_k\right),
$$

前一项只依赖行 $i$，后一项只依赖列 $j$。这就严格因式分解成一个 Column Vector 乘一个 Row Vector。对于 $M=L\odot(CB^\top)$，若状态维度为 $N$，跨边界的子块秩上界一般是 $N$，不能仍然说整体 Token Mixing Matrix 必然秩 1。

**原文 Theorem 3.5（PDF 第 9 页）** 把有限维状态模型和一种 Sequentially Semiseparable（SSS）矩阵表示联系起来；[Figure 4](https://arxiv.org/pdf/2405.21060) 右边画出 SSM、Diagonal SSM、Scalar-Identity SSM 与 Structured Masked Attention 的不同交集，**颜色区域表示模型类别包含与重叠**，不是模型准确率越高颜色越亮。

### 用三步数据分别从递推和 Attention-like 矩阵复算

设 $N=1$，$a_1=1/2,\;a_2=1/4$，$B_j=C_i=1$，输入 $x=[2,3,4]$，初始状态为零。递推：

$$
h_0=2,\quad h_1=\tfrac12(2)+3=4,\quad
h_2=\tfrac14(4)+4=5.
$$

对应的 $L=M$ 为

$$
L=
\begin{bmatrix}
1&0&0\\
1/2&1&0\\
1/8&1/4&1
\end{bmatrix}.
$$

直接矩阵乘：

$$
L\begin{bmatrix}2\\3\\4\end{bmatrix}
=\begin{bmatrix}2\\4\\5\end{bmatrix}.
$$

这严格证明了在固定 $a_t,B_t,C_t$ 时，**递推视角与结构 Masked Attention 视角计算的是同一个结果**。它没有说原始 Token 必须缓存；矩阵形式只是分析工具，在 Decode 可以直接保存一个标量状态。

## 5. 为什么有了二次型矩阵形式，反而有可能算得比纯递推快？

直观上，显式形成 $T\times T$ 的 $M$，再执行 $Y=MX$ 是 $O(T^2)$，似乎比线性扫描更慢。SSD 并不是走这条最朴素的全矩阵路线，而是用矩阵的 Semiseparable 低秩子块做 **Block Decomposition**。

把序列划分成大小为 $Q$ 的 Chunk，令 $T=KQ$（最后不足 $Q$ 时可做特殊处理）。全矩阵 $M$ 的各块 $M^{(i,j)}$，有两种完全不同的情况：

- **对角块 $i=j$**：输入输出在同一个 Chunk 内，直接使用局部 Structured Attention 形式计算；内部只有 $Q$ 个位置，可用矩阵乘处理。
- **严格下三角块 $i>j$**：将早期 Chunk 的输入效果压缩为一个尺寸受状态维度限制的**边界状态**，沿 Chunk 之间做状态递推，再映射到目标 Chunk 的每个位置。无需逐块物化所有 $Q\times Q$ 的矩形。

~~~{mermaid}
flowchart LR
 X["输入 X:[T,P]"] --> C["按 Q 个 Token 切 Chunk"]
 C --> LOC["Chunk 内：小尺寸 Attention-like MatMul"]
 C --> ENC["各 Chunk 汇总：边界状态"]
 ENC --> SCAN["Chunk 间：低秩状态递推"]
 SCAN --> DEC["状态传播到当前 Chunk"]
 LOC --> SUM["两部分相加"]
 DEC --> SUM
 SUM --> Y["输出 Y:[T,P]"]
~~~

*自制 SSD Block-Decomposition 示意，非原论文 Figure；横向两条并行路径分别计算同 Chunk 与更早 Chunk 的贡献。*

[原文 §6、Theorem 6.1（PDF 第 17 页）](https://arxiv.org/pdf/2405.21060) 说明在设 Head Dimension $P=N$、State Expansion Factor $N$ 的条件下，一条 SSD 序列的训练 FLOPs 为 $O(TN^2)$、自回归推理总 FLOPs 为 $O(TN)$、推理状态内存为 $O(N^2)$，核心计算能够转化为矩阵乘。**这里说的是在指定 Head/State 维度下的总体复杂度，不是每个 Token 的 Cost 为 $O(1)$ 与 $N$ 无关**。

为什么能兼顾算力？小矩阵乘虽然在每个 Chunk 内名义上会重复一部分运算，但 GPU 对大块 MatMul 的峰值利用率常比沿 Token 串行推进标量操作更好；非对角块则凭低秩状态避免全局二次展开。论文报告 SSD 核心层相较已优化的 Mamba Selective Scan **2–8×**，属于对应模型维度、序列与硬件的实验结果，不是纯大 O 推导。

## 6. $H_t$ 的真实 Shape 是什么？为什么这能连接到 Multi-Head Attention？

取一个 Head 的 Token Feature 为 $x_t\in\mathbb R^P$，State Expansion 为 $N$，输入/输出门控 $B_t,C_t\in\mathbb R^N$。用一个真正保存每个 Feature Channel 历史贡献的状态矩阵：

$$
H_t=a_tH_{t-1}+B_tx_t^\top,\qquad
H_t\in\mathbb R^{N\times P}.
$$

输出为

$$
y_t=C_t^\top H_t\in\mathbb R^P.
$$

$B_tx_t^\top$ 是 $[N,1]\cdot[1,P]=[N,P]$ 的 Outer Product。跨历史展开：

$$
y_i=\sum_{j\le i}
\left(C_i^\top B_j\right)
\left(\prod_{k=j+1}^{i}a_k\right)x_j.
$$

于是点积 $C_i^\top B_j$ 形式对应 Q/K，而 $x_j$ 形式对应 V——**不是 Attention Score 乘另一个查询**。这种参数化让 SSD 与 Multi-Head / Grouped-Value 的抽象可直接比较，但并不意味着所有 SSM Head 都能任意共享状态且不损失表达力。

$H_t$ 的单 Head Shape 是 $[N,P]$。若 $N=P=64$，一个 Head 的状态有 $4096$ 个元素；按 FP16 存是约 8 KiB，**随历史 Token 数不增长**。但数值精度、总 Head 数、额外 Conv Buffer 和投影激活都应计入完整模型状态；不能把“单头状态 8 KiB”冒充整层的总显存。

## 7. Mamba-2 架构为什么不只是一块更快的 Mamba-1？

[原论文 Figure 6（arXiv v1 PDF 第 23 页）](https://arxiv.org/pdf/2405.21060) 对照 **Sequential Mamba Block** 与 **Parallel Mamba Block**：Mamba-1 的输入先经过投影/局部卷积、得到 SSM 输入，再根据这个输入计算相关状态参数；Mamba-2 则把 $A,B,C,X$ 的数据相关投影尽可能移至 Block 输入的**平行分支**，减少串联的 Linear Operations，并引入额外归一化稳定训练。这是架构上的数据依赖变化，而非同一个 Layer 改了启动配置。

为什么这种改动合理？在 SSD 视角里，Sequence Layer 可看作 $(A,B,C,X)\mapsto Y$，四者是同一模型计算的输入，类似 Attention 将 Q/K/V 并行投影得到后再计算注意力。该结构还允许不同 Head 的状态扩展参数组织得更规则、减少多设备训练时的同步点，但本篇重点是**模型表达和算子张量关系**，非 Tensor Parallel 拆卡实现。

原文 §7 还比较 Multi-Input State Space（MIS）、Multi-Head State Space 与 Grouped-Value Attention 类比结构。真正的表达性问题是：**共享哪些 $B,C$、哪些 Feature/Head 拥有不同的 $a$，会影响相同参数预算下能保留多少独立的长期模式**，而不仅是把 $H$ 个 Head 改成某个好看的整数。

## 8. Figure 4 的理论界和 Figure 5/6 的模型质量不是一回事

[原论文 Figure 4（PDF 第 18 页）](https://arxiv.org/pdf/2405.21060) 左侧给出 SSM 与 SMA 两套运算要素的对应表，右侧是模型族的集合交集：一般 SSM 与一般 Structured Masked Attention 并不互相完全包含，它们的交集是作者定义的 SSD。这个图是**理论结构关系**，不是实验主表。

性能方面，[原论文 §9](https://arxiv.org/pdf/2405.21060) 分开评估：

- **MQAR：Multi-Query Associative Recall**。输入包含许多 Key/Value Pair，并在不同位置查询先前出现的 Key，对模型的精确关联记忆提出挑战。论文显示 Mamba-2 的相关结构改进有优势，但有限状态容量仍是边界，不能由几道合成题推广到无限字典精确查找。
- **语言模型 Scaling**：将 Mamba-2 与 Mamba-1、训练配方接近的 Transformer++ 对比，作者报告在同等 Pile 等数据协议下具备更好的 Loss/Wall-Clock Trade-off。一个模型在相同 Token 数达到较小 Loss，与“每秒吞吐快”是两个不同因子。
- **生成模型评估**：2.7B Mamba-2 在作者指定的 300B Pile Token 训练环境下超过同数据配置的一些 Mamba / Pythia 对照。不能把它读成“在任意数据、任意任务一定超过 6.9B Softmax Transformer”。
- **SSD Kernel Benchmark**：2–8× 是核心层速度区间；整模型还有 Input/Output Projections、Normalization、Embedding 等，因此不能拿这个数直接乘全模型 Token/s。

尤其重要的是，对比必须固定或至少记录：模型宽度、状态维数 $N$、Head Count、训练 Token、Context、Batch、硬件、是否使用 Fused Scan、训练实际 Wall-Clock 与任务质量。

## 9. 与 Softmax Attention 的最终比较：丢了什么，换来了什么？

**得到的能力**：因时间变化的 $a_t,B_t,C_t$，可以做输入相关的记忆与遗忘；其半可分矩阵结构允许同一个算子既有 Matrix Attention View、也有固定状态大小的 Recurrent View；利用 Chunk 分解，可更自然使用大矩阵计算单元训练。

**保留的限制**：$a_tI$ 是强结构约束，不是通用 $N\times N$ 动态状态矩阵；与 Softmax Attention 的完整内容选择能力不同，过去历史经有限状态压缩后可能发生冲突；模型状态复杂度仍随 $N,P$ 增长，$P=N$ 时就是 $N^2$。

最关键的科学边界是：**SSD 证明了 Scalar-Identity SSM 与一类 1-Semiseparable Structured Masked Attention 的条件等价；没有证明标准 Softmax Attention 的每个算子都能由一个固定维度状态准确模拟**。它通过“结构化矩阵”发现新的可训练模型族，再借此改造 Mamba 的架构与算法。

从技术进化关系看：

- [S4](A058-s4.md)：用固定参数 LTI 把递推、卷积、稳定长期记忆连起来；
- [Mamba](A059-mamba.md)：引入输入相关参数，使状态更新按内容选择；
- **Mamba-2**：为特定 Selective SSM 建立与 Structured Masked Attention 的严格对偶，结合 Chunk 内矩阵计算与 Chunk 间状态传播，再据此重排模型 Block。

这条路线真正值得掌握的，不是“谁更快”这句结论，而是**每一步引入的新表达自由度、失去的旧计算恒等式，以及作者怎样找到另一条数学等价路径重新获得训练可并行性**。
