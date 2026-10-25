# xLSTM：LSTM 真的过时了吗？指数门控、归一化记忆与矩阵关联检索是怎样重新拼起来的？

> **原始论文**：Beck 等，*xLSTM: Extended Long Short-Term Memory*，NeurIPS 2024。[NeurIPS 正式页面](https://papers.nips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html) · [公开 PDF：arXiv:2405.04517](https://arxiv.org/pdf/2405.04517) · [作者实现](https://github.com/NX-AI/xlstm)。本文以公开版 §2、Figure 1、Table 1/2、Appendix 中的数学定义与实验为主。
>
> **前置**：[S4](A058-s4.md)、[RetNet](A061-retnet.md)、[RWKV](A062-rwkv.md)、[Mamba](A059-mamba.md)。本文属于**记忆架构与训练框架**，而非“重新写一个更快的 LSTM CUDA Kernel”。

## LSTM 的哪一个限制真正挡住了它在大模型时代继续扩展？

常见解释说“LSTM 是串行的，Transformer 是并行的”。这只解释了训练时的执行方式，没有回答**记忆单元如何表示内容**，更无法解释为什么现代大型 LSTM 即使加深加宽仍难达到一些注意力模型的质量。

一个标准 LSTM Cell 的核心递推是

$$
c_t=f_t\odot c_{t-1}+i_t\odot z_t,\qquad
h_t=o_t\odot\tanh(c_t).
$$

其中 $i_t,f_t,o_t=\sigma(\cdot)\in(0,1)$，记忆是按通道的 $c_t$。这本来是为解决普通 RNN 的梯度衰减设计的：路径 $\partial c_t/\partial c_{t-1}$ 的直达项是 $f_t$，当 $f_t\approx1$，历史信息可持续传播。

但两处隐藏的设计取舍值得重新讨论：

1. Sigmoid Input/Forget Gate 的数值在 $0$ 与 $1$ 之间。**旧记忆与新记忆的相对权重受限于门控如何同时配合**，而长期积累与近期重写往往需要不同动态范围。
2. $c_t$ 是按通道标量记忆。若要存储“Key A 对应 Value 7、Key B 对应 Value 9”，普通向量记忆必须把多个关联编码进一组坐标里，读出时不容易根据任意 Query 恢复。

xLSTM 的论文提出两个有意不同的路线：**sLSTM：保留门控和 Cell 之间的 Recurrent Mixing，改写记忆归一化；mLSTM：将单个 Cell 变成可存 Key–Value 关联的矩阵，放弃 Hidden-to-Hidden Mixing 来换取并行训练**。

[原文 Figure 1（arXiv PDF 第 1 页）](https://arxiv.org/pdf/2405.04517) 的左侧对照传统 LSTM、sLSTM、mLSTM，右侧指向 xLSTM Block 和 Residual Stacking。这个图讲的是**研究方案的分支**，并不是“一个 Cell 同时具备 sLSTM 所有 Recurrent Mixing 和 mLSTM 所有并行性质”。

真正的研究动机见 **Figure 2（arXiv PDF 第 2 页）**：左图是 Nearest Neighbor Search（给定参考向量、逐步扫描候选，结尾预测与最匹配候选附带的 Value），纵轴为 MSE；右图将 WikiText-103 按 Token 出现频次分组，统计每组的 PPL。两侧分别揭示“难以在扫描中修订旧存储决定”和“稀有 Token 的记忆容量不足”。这支持指数门控与矩阵记忆的设计动机，却不能只凭图中差异把性能提升全部归给单一组件；需要结合下文 Table 2 消融。

## 从 Sigmoid 到指数门控：$e^{\tilde i_t}$ 为什么需要新 Normalizer？

原论文 §2.2 Eq.(8)–(14) 的 sLSTM 首先替换 Input Gate：

$$
\begin{aligned}
c_t&=f_tc_{t-1}+i_tz_t,\\
n_t&=f_tn_{t-1}+i_t,\\
\tilde h_t&=\frac{c_t}{n_t},\qquad h_t=o_t\tilde h_t,\\
i_t&=\exp(\tilde i_t),\qquad
f_t=\sigma(\tilde f_t)\;\text{或}\;\exp(\tilde f_t).
\end{aligned}
$$

为什么多了一份 $n_t$？把递推展开：

$$
c_t=\sum_{j=1}^{t}i_jz_j\prod_{k=j+1}^tf_k,
\qquad
n_t=\sum_{j=1}^{t}i_j\prod_{k=j+1}^tf_k
$$

（为直观起见取 $c_0=n_0=0$）。因此

$$
\boxed{\frac{c_t}{n_t}=
\frac{\sum_{j\le t}w_{tj}z_j}{\sum_{j\le t}w_{tj}}},
\quad w_{tj}=i_j\prod_{k=j+1}^tf_k>0.
$$

这是一种**可递推维护的、正权重归一化记忆**，其数值可以反映不同历史写入的重要性，而不会因为所有 $i_t$ 都超过 1 就让输出简单线性增长。注意，$n_t$ 归一化的是**历史写入权重**，不是用于训练 Loss 的全局 Token Count。

### 三步手算：一个较晚写入可以真实覆盖早期信息

取初始 $c_0=n_0=0$，三个输入 $z_1=2,z_2=100,z_3=4$，忘记门均为 $f=1/2$，输入门依次 $i=[1,0.01,5]$。则

$$
(c_1,n_1)=(2,1),\quad
(c_2,n_2)=(2,\;0.51),\quad
(c_3,n_3)=(21,\;5.255).
$$

第 2 步计算：
$c_2=0.5(2)+0.01(100)=2$，$n_2=0.5(1)+0.01=0.51$，$\tilde h_2\approx3.9216$；第 3 步 $c_3=0.5(2)+5(4)=21$，所以 $\tilde h_3=21/5.255\approx3.9962$。

三步状态与归一化输出汇总如下：

| 时刻 | $c_t$ | $n_t$ | $c_t/n_t$ |
|---|---:|---:|---:|
| 1 | 2 | 1 | 2.0000 |
| 2 | 2 | 0.51 | 3.9216 |
| 3 | **21** | 5.255 | 3.9962 |

这一数值例子是教学构造，说明一个输入的数值大小（第 2 步的 100）并不等于它最后被记住的强度；真实权重是门控积。还要记住 $i_t=\exp(\tilde i_t)$ 是正数，$0.01$ 也是合法的指数门输出。

### 为什么 $n_t$ 不一定让原始 $c_t$ 的数值安全？

如果 $\tilde i_t=1000$，直接计算 $\exp(1000)$ 早在进入 $c_t/n_t$ 之前就已经溢出。因此作者的 [Eq.(15)–(17)，PDF 第 4 页](https://arxiv.org/pdf/2405.04517) 引入类似 Log-Sum-Exp 的额外 Max State：

$$
\begin{aligned}
m_t&=\max(\log f_t+m_{t-1},\log i_t),\\
i'_t&=\exp(\log i_t-m_t),\\
f'_t&=\exp(\log f_t+m_{t-1}-m_t).
\end{aligned}
$$

更新的是**重标度后的** $c'_t=f'_tc'_{t-1}+i'_tz_t$、$n'_t=f'_tn'_{t-1}+i'_t$。因为同一步对原来的 $c_t,n_t$ 乘相同的 $e^{-m_t}$，二者的比值不变。若实现只把输入门改成 $i'_t$ 却忘记对历史记忆乘 $f'_t$，就不是等价的稳定化公式。

**为什么这不是无意义的工程细节？** 这个额外状态使指数门控能够真的训练，而不只是纸面上“给 Sigmoid 换个激活函数”。作者还在 Appendix 给出对前向和梯度不变性的推导，正常情况下仅存在浮点舍入误差。

## sLSTM 的 Memory Mixing 为什么会牺牲沿时间的完全并行？

原论文 §2.2 中 $\tilde i_t=w_i^\top x_t+r_i h_{t-1}+b_i$，$z_t$、Forget、Output Gate 也可以含 $r_\bullet h_{t-1}$。因此**当前 Gate 的系数本身依赖上一时刻的输出**，而上一输出又受更早状态影响。

这与只要先对全部 Token 一次性投影 $i_t,f_t,z_t$、然后使用 Parallel Scan 的情况不同。即使 Cell Update 本身是仿射函数复合，$i_t$ 的计算仍需拿到 $h_{t-1}$，存在无法仅靠固定系数 Scan 消去的时间依赖。

在向量 sLSTM 中，$R_i,R_f,R_o,R_z\in\mathbb R^{d\times d}$ 可以混合多个 Cell 的信息；作者给 sLSTM 加入 Multi-Head 结构并限制 **Head 内部允许 Memory Mixing，Head 之间不做这种 Mixing**。这增加表示灵活性，同时保留了真正递推 RNN 的时间依赖。**不能因为 xLSTM 家族中有“完全可并行”的 mLSTM，就说 sLSTM 也被证明沿时间完全并行。**

## 如果要存 Key–Value 对，为什么不能简单把向量 Cell 变长一些？

把向量 Cell 维数从 $d$ 扩大到 $2d$，只是增加自由度，并没有在结构上规定“哪一部分是 Key、哪一部分是 Value，如何依当前 Query 取回”。mLSTM 明确把记忆改成矩阵：

$$
C_t\in\mathbb R^{d_v\times d_k},\qquad
C_t=f_tC_{t-1}+i_tv_tk_t^\top.
$$

外积 $v_tk_t^\top$ 是一条关联记录：**在 Key 方向 $k_t$ 写入 Value 方向 $v_t$**。当前 Query $q_t$ 则读出

$$
C_tq_t=
\sum_{j\le t}i_j\prod_{r=j+1}^tf_r\;
v_j(k_j^\top q_t).
$$

它和 Attention 的 $K^\top Q$ 关联读出形式很像，但系数未经 Softmax，可能为负且有相互干扰。mLSTM 再保存

$$
n_t=f_tn_{t-1}+i_tk_t,\qquad
\tilde h_t=\frac{C_tq_t}{\max(|n_t^\top q_t|,1)},\quad h_t=o_t\odot\tilde h_t.
$$

[原论文 Eq.(18)–(26)，PDF 第 4–5 页](https://arxiv.org/pdf/2405.04517) 中还包含 Key 的 $1/\sqrt{d_k}$ 缩放、输出门和 Input/Forget Gates。分母的绝对值与阈值 1 是重要设计：当 $n_t^\top q_t$ 很小、为负或发生抵消，直接除以它可能产生极端数值；$\max(|n^\top q|,1)$ 使分母不会低于 1，但因此**它不是 Softmax 的概率归一化，也不保证 $h_t$ 落在历史 Value 的凸包内**。

### 两维矩阵记忆手算

假设 $f_t=1$、$i_t=1$、$o_t=1$。第 1 条信息 $(k_1,v_1)=([1,0]^\top,[3,0]^\top)$，第 2 条信息 $(k_2,v_2)=([0,1]^\top,[0,5]^\top)$。则

$$
C_2=v_1k_1^\top+v_2k_2^\top=
\begin{bmatrix}3&0\\0&5\end{bmatrix},\qquad n_2=\begin{bmatrix}1\\1\end{bmatrix}.
$$

若 $q=[1,0]^\top$，输出 $C_2q/\max(|n_2^\top q|,1)=[3,0]^\top$；若 $q=[0,1]^\top$，输出 $[0,5]^\top$，实现了两条相互正交 Key 的独立读取。

若第三条信息再次使用 $k_3=[1,0]^\top$、但 $v_3=[9,0]^\top$，状态对应位置会积累多个 Value，输出成为按门控加权的组合，**不会无条件自动覆写旧值**。这说明 mLSTM 的矩阵存储仍然存在 Key 冲突与状态容量问题，而不是变成了无限大的 Key–Value Database。

~~~{mermaid}
flowchart TD
 X["输入 x_t"] --> P["Query/Key/Value 投影 + Input/Forget Gate"]
 P --> O["写入外积 v_t k_t^T"]
 OLD["旧矩阵记忆 C_{t-1}"] --> FORGET["Forget：f_t C_{t-1}"]
 FORGET --> SUM["更新矩阵 C_t"]
 O --> SUM
 P --> NORMAL["更新 normalizer n_t"]
 SUM --> READ["当前 Query q_t 读出 C_t q_t"]
 NORMAL --> DIV["max(abs(n_t^T q_t), 1)"]
 READ --> RES["归一化 + Output Gate"]
 DIV --> RES
~~~

*自制矩阵记忆示意，非论文 Figure；$n_t$ 由过去 $n_{t-1}$ 与本次 $i_tk_t$ 共同递推，图中简化了该状态箭头。*

## 为什么 mLSTM 是可并行的，而 sLSTM 不是？

关键不是记忆从标量变成矩阵，而是 **mLSTM 的 Input/Forget Gate、$q_t,k_t,v_t$ 只依赖当前输入 $x_t$**，不使用 sLSTM 那样的 $h_{t-1}$ 回馈投影。于是，在给定整个序列的投影结果后，递推可以展开为

$$
C_t=\sum_{j=1}^{t}
\left(i_j\prod_{r=j+1}^{t}f_r\right)
v_jk_j^\top.
$$

这些权重全由当前序列的各 Token Projection 决定，训练时可以转换为**下三角权重矩阵 + 一系列 Matrix Multiply**；解码时继续只维护 $C_t,n_t,m_t$ 状态。此关系与 [RetNet](A061-retnet.md) 类似，但 RetNet 的 Head 时间衰减通常是固定 $\gamma$，mLSTM 的门控随输入改变。

与 [Mamba-2](A060-mamba2-ssd.md) 的区别：mLSTM 特别保留**归一化 Key–Value Matrix Memory**；Mamba-2 SSD 以 Scalar-Identity SSM 和 Semiseparable Mask 结构建立更一般的矩阵/递推视角。不能只凭都可写成 $S_t=fS_{t-1}+kv^\top$ 就说两者所有 Norm/Gate/Block 完全相同。

## xLSTM Block 是不是单纯把传统 LSTM 串上 48 次？

不是。原论文 §2.4 与 Appendix Figure 6 展示了一套现代语言模型的 Residual Backbone：Block 前后包括 LayerNorm、Up-Projection、Causal Convolution、Down-Projection 与 Residual Connection，再在合适位置放入 sLSTM/mLSTM Cell。

作者还用符号 **xLSTM[a:b]** 表示 **mLSTM Block : sLSTM Block** 的比值，而不是某两个模型的训练 Epoch。比如 xLSTM[7:1] 在 48 Block 的例子里，42 个使用 mLSTM、6 个使用 sLSTM；xLSTM[1:0] 则都是 mLSTM。它们的理论并行性、Recurrent Mixing 与状态内存会因此不同，不能把一组吞吐与另一组 PPL 无条件拼接。

这组结构设计的意义是：**既保留传统 LSTM 在状态跟踪、动态门控上的优势，又利用当前语言模型成熟的残差网络和高维特征处理方法来提高训练可扩展性**。

## Table 1 / Table 2 的因果证据比摘要里“超越 Transformer”更有价值

[原论文 Table 1（arXiv PDF 第 9 页）](https://arxiv.org/pdf/2405.04517) 使用约 **350–450M 参数级别、15B SlimPajama Token**，比较 Validation PPL：

| 模型 | 参数量约（M） | Validation PPL ↓ |
|---|---:|---:|
| GPT-3 风格 Transformer | 356 | 14.26 |
| Llama 风格 Transformer | 407 | 14.25 |
| Mamba | 423 | 13.70 |
| RWKV-4 | 430 | 15.62 |
| RetNet | 431 | 16.23 |
| **xLSTM[1:0]** | **409** | **13.43** |
| xLSTM[7:1] | 408 | 13.48 |

所以可陈述的结果是：**在作者统一的一组数据、训练 Token 与大致模型预算下，xLSTM 得到最低 Validation PPL**。不应将这张表视为在任意规模、任何 2026 年模型 Recipe 下的架构必然排名。尤其 400M 级别实验的结果不能直接外推到 70B LLM 的综合能力。

更值得仔细读的是 [原论文 Table 2（PDF 第 10 页）](https://arxiv.org/pdf/2405.04517)。作者不是简单对比“有 xLSTM”和“无 xLSTM”：

| 改动阶段 | PPL ↓ |
|---|---:|
| Vanilla Multi-Layer LSTM（不易训练） | 2417.86 |
| 加 Residual/Norm Backbone | 35.46 |
| 加 Up-Projection Block | 26.01 |
| 再加 Exponential Gating（sLSTM） | 17.70 |
| 再加入 Matrix Memory（xLSTM[7:1]） | **13.48** |

这份消融证明最终效果**至少涉及训练架构、指数门控、矩阵存储三类因素**，而不能把 2417→13.48 全部归因到 $e^{\tilde i}$ 一个指数函数上。Table 2 底部还比较输入相关的 Forget/Input Gate，与只使用固定 Bias 的门控；作者发现输入相关和可学习门控共同有益。这直接支持“选择记忆强度”比“给模型加一个固定指数衰减率”更重要。

### Figure 6/8：请同时查看数据量和模型是否足够训练

原文 Figure 6（PDF 第 10 页）在 15B SlimPajama Token 下比较不同参数规模的模型 PPL；xLSTM[7:1] 的 2.7B 参数点表现较差，作者明确提及**收敛更慢、训练不足**。Figure 8 则是 300B Token 的较大训练预算下的 Scaling 对比；这两张图不可以直接当作在完全相同训练条件下复现出来的同一条学习曲线。

论文生成速度/吞吐另有独立实验，它们衡量不同的系统指标；不能从固定状态内存的理论分析直接得出某个 8K Context 下的固定 Token/s，尤其 sLSTM 含 Hidden-to-Hidden Recurrence，其训练时序与 mLSTM 不一样。

## 如果在未来模型里采用这些想法，怎样判断是真的架构贡献？

首先必须明确用途：**内容选择、持续更新状态、细粒度键值关联、状态跟踪**分别对记忆机制提出不同要求。RWKV-4 的逐通道 WKV 偏向便宜的加权汇总；RetNet 的矩阵外积配合固定衰减；mLSTM 的矩阵外积配合输入相关的动态门控和 Normalizer；sLSTM 保留额外的 Hidden-to-Hidden Mixing；Mamba 则从状态空间参数 $\Delta,B,C$ 的输入相关选择入手。

这些差异决定了模型会在不同任务上各有优势。一个真正能检验它们的实验，应把不同长度的 Copy、Selective Copy、Multi-Query Associative Recall、状态跟踪与自然语言模型 Loss 放在一起，并报告真实模型 State Budget，而不是只看大模型静态参数量。

数学边界也必须明确：mLSTM 的 Matrix Memory 大小随 $d_kd_v$ 增长，尽管**不随上下文 Token 数增长**；同 Key 多条记录会发生干扰；指数 Gate 需要稳定的 Log-Max；xLSTM 变体中 sLSTM 的 Recurrent Mixing 使它不具备 mLSTM 的全时间并行性。

xLSTM 为架构研究留下的最重要思路，是**把“记什么、怎样归一化、用什么状态形状、能否并行”拆成可独立选择的设计维度，再用真实任务和消融决定组合**。这一点与 [S4 → Mamba → Mamba-2](A058-s4.md) 的研究路线相通，但没有理由把它们视为同一种网络换了名字。
