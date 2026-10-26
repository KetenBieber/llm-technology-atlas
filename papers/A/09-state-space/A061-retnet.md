# RetNet：为什么同一套 Retention 能在训练时像 Attention、解码时像 RNN？

> **原论文**：Sun 等，*Retentive Network: A Successor to Transformer for Large Language Models*（2023）。[arXiv:2307.08621v4（14 页，Figure 3 / Figure 4 / Table 2）](https://arxiv.org/pdf/2307.08621) · [论文页面](https://arxiv.org/abs/2307.08621) · [Microsoft 官方介绍](https://www.microsoft.com/en-us/research/?p=958491)。
>
> **先看**：[Transformer](../03-transformer/A013-transformer-attention-is-all-you-need.md) 的 Softmax Attention、[RoPE](../03-transformer/A014-roformer-rope.md) 的旋转位置编码、[S4](A058-s4.md) 的递推/卷积等价。**后续**：[RWKV](A062-rwkv.md)（归一化的指数加权 WKV）、[xLSTM](A063-xlstm.md)（指数门控与矩阵记忆）、[Mamba-2](A060-mamba2-ssd.md)（SSM/结构化 Attention 对偶）。本文是**架构方法论文**；计算开销只是架构约束，不按 GPU Infra 论文展开。

## 为什么“每步只保存固定大小状态”往往比发明新位置编码更困难？

Transformer 的输出可写为

$$
y_t=\sum_{j\le t}\underbrace{\frac{\exp(q_t^\top k_j/\sqrt d)}{\sum_{r\le t}\exp(q_t^\top k_r/\sqrt d)}}_{\alpha_{tj}}v_j.
$$

每出现一个新查询 $q_t$，这些权重可能重新分配：昨天被忽略的 Token 今天可以成为最重要的证据。代价是历史 Key/Value 至少要保留某种可检索表示。RetNet 想要的不是一个“把 KV 存到更便宜内存”的策略，而是**在每步只保留固定尺寸状态的情况下，训练阶段仍然能够批量计算整段序列**。

朴素的历史平均 $s_t=\gamma s_{t-1}+(1-\gamma)x_t$ 当然容易递推，但它只按时间衰减，无法表达查询、键和值之间的关联。RetNet 于是把记忆对象从一个 Value 向量，升级为 Key–Value 的**外积矩阵**。这一处结构选择，决定了后面整个方法可以从 RNN 改写成类 Attention 矩阵。

## 从原论文 Eq.(6) 反推：记忆矩阵究竟保存了什么？

统一使用行向量：$q_t,k_t\in\mathbb R^{1\times d_k}$，$v_t\in\mathbb R^{1\times d_v}$。每个 Head 的记忆状态为

$$
S_t=\gamma S_{t-1}+k_t^\top v_t,\qquad
S_t\in\mathbb R^{d_k\times d_v},\quad 0<\gamma<1,
$$

输出是

$$
o_t=q_tS_t\in\mathbb R^{1\times d_v}.
$$

注意 $k_t^\top v_t$ 的 Shape 是 $[d_k,1]\,[1,d_v]=[d_k,d_v]$；这不是“把 Token 的 Value 简单相加”。如果查询与某个历史 Key 对齐，它会读取那条 Key–Value 外积贡献中的相应 Value 方向。

展开前三步，取 $S_0=0$：

$$
\begin{aligned}
S_1&=k_1^\top v_1,\\
S_2&=\gamma k_1^\top v_1+k_2^\top v_2,\\
S_3&=\gamma^2 k_1^\top v_1+\gamma k_2^\top v_2+k_3^\top v_3.
\end{aligned}
$$

于是

$$
o_t=\sum_{j=1}^{t}\gamma^{t-j}(q_tk_j^\top)v_j.
$$

这就是 Retention 的原始逻辑：每个历史 Value 的读出，既受 Query–Key 匹配影响，又被时间距离乘上指数遗忘系数。**状态大小不随 $t$ 增长**，但不同 Key 的贡献可能在有限尺寸矩阵里互相干扰：该式不能直接推出“无损回忆任何历史”。

### 用三个 Token 手算，防止把外积误当成普通求和

令 $d_k=2,d_v=1,\gamma=1/2$，三个 Key 分别为 $k_1=[1,0],k_2=[0,1],k_3=[1,1]$，Value 为 $v_1=2,v_2=4,v_3=1$。状态依次为

$$
S_1=\begin{bmatrix}2\\0\end{bmatrix},\quad
S_2=\begin{bmatrix}1\\4\end{bmatrix},\quad
S_3=\begin{bmatrix}1.5\\3\end{bmatrix}.
$$

若当前查询 $q_3=[1,2]$，则 $o_3=[1,2][1.5,3]^\top=7.5$。从历史直接展开：

$$
(1/4)(q_3k_1^\top)2+
(1/2)(q_3k_2^\top)4+
(q_3k_3^\top)1
=0.5+4+3=7.5.
$$

这是为验证两种表达严格一致的**教学算例**，不是论文模型的训练结果。这里出现负值输出也是可能的，因为此机制没有 Softmax 的正概率约束。

## 为什么固定步长递推会自动对应 Figure 3(a) 的平行矩阵？

把所有 $q_t,k_t,v_t$ 按时间堆成 $Q:[T,d_k]$、$K:[T,d_k]$、$V:[T,d_v]$。定义下三角衰减矩阵

$$
D_{tj}=
\begin{cases}
\gamma^{t-j},&j\le t,\\
0,&j>t.
\end{cases}
$$

那么

$$
\boxed{O=(QK^\top\odot D)V}.
$$

其第 $t$ 行就是刚才递推展开的 $\sum_{j\le t}\gamma^{t-j}(q_tk_j^\top)v_j$。**这在相同输入、相同精度、相同边界条件下属于代数恒等式，不是训练两套模型之后才经验上相近**。

[原文 Figure 3（PDF 第 4 页）](https://arxiv.org/pdf/2307.08621) 左图的 $QK^\top\odot D$ 代表一次并行 Retention 计算，右图则是“旧状态乘 $\gamma$、当前 $K^\top V$ 写入、用 $Q$ 读出”。两幅图不是两种不同的参数网络；**GroupNorm（GN）也画在两端的输出侧**。这种输出归一化应该发生在同一 Head 的完整 Retention 输出之后，不能把 GN 任意插入 $S_t$ 更新而宣称保持相同等式。

这两种视角和 [S4](A058-s4.md) 的“递推 vs 固定卷积”相关但不同：S4 的固定核只取决于时差；RetNet 的系数还含 $q_tk_j^\top$，**由 Token 内容决定**，因此不是一个固定的标量卷积核。

### 为什么“Retention 是 Softmax Attention 的等价实现”仍然是错的？

标准 Softmax Attention 使用 $\exp(qk^\top)$ 和整行归一化，输出是 Value 的正权重凸组合（忽略进一步投影）；Retention 直接使用 $qk^\top$，再乘时间衰减，没有 Softmax。这两者一般不同。

例如 $q=[1,0],k_1=[-1,0],k_2=[1,0]$，标准 Softmax 在两个 Value 上的权重均为正，而 Retention 对 $v_1$ 的原始系数是负的。这一**表达范围上的差异**意味着不能以公式外观都包含 $QK^\top V$，就宣布二者对所有输入的输出分布不变。

## 原论文 Eq.(5) 中的复数位置相位为什么可以与递推兼容？

论文原始表达还加入位置相位：

$$
Q=(XW_Q)\odot\Theta,\qquad
K=(XW_K)\odot\bar\Theta,\qquad
\Theta_t=e^{it\theta}.
$$

这里 $\Theta$ 的共轭方向取决于 $K$ 与矩阵乘是否使用复共轭转置。关键是相位乘积只取决于 $t-j$，于是既可把相位写在每个 Token 的 $Q,K$ 中，也可以写入递推状态的等价坐标。与 [RoPE](../03-transformer/A014-roformer-rope.md) 相似，这是一类**相对位置信息由绝对相位差产生**的结构，但不能把 RetNet 的位置处理直接等同于旋转后的 Softmax Attention。

原文在 Figure 3、Eq.(5)(6) 说明 Query/Key 都包含相同的位置信息；为与实现代码一致，实际重建时要使用原文给定的共轭约定，避免一个不必要的复共轭错误引入了相反方向的相位。

## 三种计算形式并非“把整段 Attention 全算一次”与“直接 For 循环”两种而已

[原论文 Figure 4（PDF 第 5 页）](https://arxiv.org/pdf/2307.08621) 是三种计算范式的**伪代码对照**：

| 形式 | 输入与状态 | 主要计算 | 适合情境 |
|---|---|---|---|
| Parallel | 整段 $Q,K,V$ | $(QK^\top\odot D)V$ | 给定较短序列的并行训练 |
| Recurrent | 保存 $S_{t-1}$ | $\gamma S_{t-1}+k_t^\top v_t$ | 单步自回归解码 |
| Chunkwise Recurrent | Chunk 内完整 $Q,K,V$，Chunk 间仅传 $S$ | Chunk 内并行 + Chunk 之间传边界状态 | 长序列训练 |

### Chunkwise 的推导核心是“区分当前 Chunk 内与过去 Chunk 的贡献”

设 Chunk 长度 $B=2$，进入 Chunk 前历史状态为 $S_{\rm in}$。这个 Chunk 的两步为

$$
S_1=\gamma S_{\rm in}+k_1^\top v_1,\qquad
S_2=\gamma^2S_{\rm in}+\gamma k_1^\top v_1+k_2^\top v_2.
$$

于是输出为

$$
\begin{aligned}
o_1 &=q_1(\gamma S_{\rm in})+q_1k_1^\top v_1,\\
o_2 &=q_2(\gamma^2 S_{\rm in})
+\gamma q_2k_1^\top v_1+q_2k_2^\top v_2.
\end{aligned}
$$

可以看到两项：**过去 Chunk 提供的 $S_{\rm in}$**，系数随 Chunk 内位置衰减；**本 Chunk 自己的 Token**，可用内部下三角矩阵并行计算。作者原文 Eq.(7) 中的 $\xi$ 和 $\zeta$ 就分别负责输入状态向 Chunk 内传播与当前 Chunk 对输出边界状态的贡献。两边权重需要使用**正确的指数 $B-j-1$ 与 $j+1$**，否则跨 Chunk 的输出不会等于逐步递推。

~~~{mermaid}
flowchart LR
 A["历史状态 S_in"] --> C["乘每位置 γ^j，提供跨 Chunk 贡献"]
 B["当前 Chunk Q,K,V"] --> D["Chunk 内 Retention（因果衰减）"]
 B --> E["更新边界状态 S_out"]
 A --> E
 C --> F["输出 O"]
 D --> F
 E --> G["下一个 Chunk"]
~~~

*自制结构示意，不是论文 Figure 4；连接线解释同一 Retention 在 Chunk 内与 Chunk 间的两条贡献路径。*

### 用数值检验跨 Chunk 状态是否真的等价，而不是只看 Figure 4 的箭头

取 $d_k=2,d_v=1,\gamma=1/2$、四个 Key：

$$
k_1=[1,0],\quad k_2=[0,1],\quad
k_3=[1,1],\quad k_4=[1,-1],
$$

Value 为 $v=[2,4,1,3]$，Query 全取 $q=[1,2]$。从零状态顺序递推，四个时刻的状态列向量依次是

$$
S_1=\begin{bmatrix}2\\0\end{bmatrix},\quad
S_2=\begin{bmatrix}1\\4\end{bmatrix},\quad
S_3=\begin{bmatrix}1.5\\3\end{bmatrix},\quad
S_4=\begin{bmatrix}3.75\\-1.5\end{bmatrix}.
$$

因此 $o_4=qS_4=3.75-3=0.75$。把前两个 Token 看作第一个 Chunk，它的输出边界为 $S_{\rm in}=S_2=[1,4]^\top$。第二 Chunk 的最后一个输出按“过去 Chunk + 本 Chunk”拆成

$$
\begin{aligned}
o_4&=
q\underbrace{\gamma^2S_{\rm in}}_{\text{过去两步累计遗忘}}
+q\underbrace{\big(\gamma k_3^\top v_3+k_4^\top v_4\big)}_{\text{本 Chunk 的写入}}\\
&=\tfrac14(1+8)+\tfrac12(3)+(3-6)
=2.25+1.5-3=\boxed{0.75}.
\end{aligned}
$$

如果某个实现忘记跨 Chunk 应当衰减**两步**，错误地只乘 $\gamma$，得到 $q(0.5S_2)=4.5$、总输出变成 3，这就和整段 Parallel Retention 不再一致。这个错误很隐蔽：Shape 完全匹配、程序也能运行，但跨 Chunk 的长程信息权重变了。

~~~python
import numpy as np

gamma = .5
K = np.array([[1., 0.], [0., 1.], [1., 1.], [1., -1.]])
V = np.array([2., 4., 1., 3.])
Q = np.tile(np.array([1., 2.]), (4, 1))
state = np.zeros(2)
recurrent = []
for k, v, q in zip(K, V, Q):
    state = gamma * state + k * v
    recurrent.append(q @ state)

D = np.tril(gamma ** np.abs(np.subtract.outer(np.arange(4), np.arange(4))))
parallel = (Q @ K.T * D) @ V
cross_old = gamma**2 * np.array([1., 4.])
local = gamma * K[2] * V[2] + K[3] * V[3]
chunk_last = Q[3] @ (cross_old + local)
assert np.allclose(parallel, recurrent)
assert np.isclose(chunk_last, .75)
print("recurrent", recurrent, "parallel", parallel)
~~~

*自制 NumPy 复算，未运行任何完整 RetNet 模型。为突出 Retention Kernel 的数学恒等，代码没有包含多尺度 Head、复数相位、GroupNorm、Swish Gate。实际复现需将这些层恢复到论文 Figure 3 / Eq.(8) 指定的位置；把 GroupNorm 加在隐藏状态递推内部不会保持上述线性等价。*

## Multi-Scale 为什么不是给所有 Head 共享一个衰减率？

对单个 Head，$D_{tj}=\gamma^{t-j}$；$\gamma$ 越靠近 1，保留远历史的能力越强，越小则越偏向局部。只取一把 $\gamma$ 相当于所有 Head 都有相似的时间长度偏好。原论文 §2.2 Eq.(8) 使用不同 Head 的 $\gamma_h$，默认构造为 $1-2^{-5-\mathrm{arange}(h)}$，并配合 Swish Gate：

$$
\begin{aligned}
H_h&=\operatorname{Retention}(X,\gamma_h),\\
Y&=\operatorname{GroupNorm}_{h}([H_1;\ldots;H_h]),\\
\operatorname{MSR}(X)&=[\operatorname{swish}(XW_G)\odot Y]W_O.
\end{aligned}
$$

为什么 Head-specific GroupNorm 必须保留？不同时间尺度的未归一化 Retention 状态方差不同，直接 Concat 可能使长衰减 Head 在数值上压过短衰减 Head。GroupNorm 调整输出量级，但**不会凭空恢复已经压缩状态丢掉的历史信息**。

要注意论文 §3.1 的模型比较并没有机械使用默认的 $\gamma$ 列表，而是改成在 $1/32$ 到 $1/512$ 之间的衰减尺度以保证不同规模模型的公平设定。不要把默认公式与实验公式不加区分地用作复现参数。

## Figure 1、Figure 5 分别支持什么，不支持什么？

[原论文 Figure 1（PDF 第 1 页）](https://arxiv.org/pdf/2307.08621) 把 GPU Memory、Throughput、Latency 与模型规模下的 PPL 放在两类图里：作者标出 8K 输入条件下内存最多 3.4× 降低、吞吐最多 8.4× 提升、延迟最多 15.6× 改善。**它们分别对应不同统计对象/极值，不是同一个运行点可直接相乘的三个倍数**。题目所说低成本来自固定大小 Retention State，而不是训练时平行表示的矩阵永远只占 $O(1)$ 空间。

[Figure 5（PDF 第 7 页）](https://arxiv.org/pdf/2307.08621) 的横轴为 1.3B、2.7B、6.7B 参数规模，纵轴为 Validation PPL（越低越好），RetNet 与 Transformer 各有一条曲线。实验在 **100B Token、最大长度 2048、包含 The Pile/C4/The Stack 的混合语料**上进行（见 Table 2）。曲线展示在指定配方和规模范围内的趋势，并不能证明 RetNet 对任何更大模型、任意数据集都必然胜过 Softmax Transformer。

原文 Table 2 的 1.3B/2.7B/6.7B 使用不同 Layer/Hidden Dim 和学习率；另外 Table 1 是理论性质对照，不是质量主结果。复现实验至少同时控制参数量、Token 预算、Data Mix、位置编码、Head 尺寸及是否在 Chunk 模式使用相同损失计算。

## 在今天应该怎样看 RetNet 的价值和边界？

它最有价值的理论洞察是：**把 Query–Key 条件相关的历史 Value 压成固定大小的外积状态，同时仍可展开为一个按时间衰减的结构化 Token Mixing Matrix**。这种“并行/递推对偶”直接为后续线性 Attention、状态空间及矩阵记忆架构提供了可比较对象。

限制同样明确：没有 Softmax 归一化的正概率语义；固定状态会受 Key 冲突与有限精度影响；$\gamma_h$ 虽能覆盖多个尺度，但其衰减通常是预设的，不像 [Mamba](A059-mamba.md) 的 $\Delta_t$ 可直接随 Token 内容改变记忆寿命；长序列的 Chunk 算法还要处理输出归一化和正确的边界状态。

与 [RWKV](A062-rwkv.md) 的差别：RWKV 在每个 Channel 上保留独立的加权分子/分母，并为**当前 Token 单独增加 Bonus**；RetNet 直接保持 $d_k\times d_v$ 的外积矩阵与 GroupNorm。与 [xLSTM](A063-xlstm.md) 的差别：mLSTM 的矩阵记忆更像 RetNet 的状态，但它将输入相关的 Forget/Input Gate 与额外 Normalizer State 引入到记忆读取中。三者都能避免随序列长度线性增长的 KV Cache，却**绝不自动具有完全相同的表达能力**。
