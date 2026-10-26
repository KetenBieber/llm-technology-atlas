# Mamba：为什么 S4 的固定卷积核无法按内容选记忆？Selective SSM 如何从一阶门控走到可训练语言模型？

> **原论文**：Albert Gu、Tri Dao，**Mamba: Linear-Time Sequence Modeling with Selective State Spaces**，COLM 2024。[原始论文 PDF：arXiv:2312.00752v2](https://arxiv.org/pdf/2312.00752) · [COLM 2024 论文](https://openreview.net/forum?id=tEYskw1VY2) · [官方代码/模型](https://github.com/state-spaces/mamba)。
>
> **路线**：[S4：固定参数状态空间、递推与卷积同构](A058-s4.md) → **本篇：令 $\Delta,B,C$ 依赖当前 Token，实现选择性记忆** → [Mamba-2：Semiseparable Duality / SSD](A060-mamba2-ssd.md)。对照：[Transformer 自注意力](../03-transformer/A013-transformer-attention-is-all-you-need.md)。

## 1. 为什么“我有一个能记一万步的状态”不等于“我能按内容挑出该记住的事情”？

[S4](A058-s4.md) 证明，只要状态矩阵在时间上固定，递推

$$
h_t=\bar Ah_{t-1}+\bar Bu_t,\qquad y_t=Ch_t
$$

可以展开成核 $K_j=C\bar A^j\bar B$，也就是 $y=K*u$。这是一个非常高效的设计，却也暴露了内容选择性问题：**同一个相对时距 $j$ 的输入，不论是关键事实还是无关干扰，它经过同一个线性核系数**。

例如下面两段序列：

- 固定 Copy：§A B C D · · · 复制 A B C D§，所有关键 Token 总出现在固定偏移；
- Selective Copy：§· A · · B · C · · D · 复制 A B C D§，有意义 Token 的间距随机改变。

对第一段，固定长度卷积只要学到正确时间位置就可以；第二段中，模型需要**看到 Token 内容再决定当前这步是否写入状态，以及多久保留它**。普通深层卷积可以依靠不同层的非线性与通道混合提高能力，但固定的单层 LTI 转移并没有在每一步重新选择卷积核的自由度。

[原论文 Figure 2（PDF 第 6 页）](https://arxiv.org/pdf/2312.00752) 左侧画标准 Copying，右上画 Selective Copying（彩色方块之间的空白长度随机），右下则是 **Induction Heads** 形式的关联检索：碰到一个曾经出现过的 Key，要找到与该 Key 相连的 Value。图中的彩色方块与箭头代表输入输出的**内容关系**，不是 GPU Cache 命中情况。该图用于说明“为什么固定时间规则不够”，后续实验才检验算法是否学会这些关系。

## 2. 先从一阶状态推导出门控，而不是把 Selection 当成一个黑盒网络

连续时间系统先写成极简单的低通记忆：

$$
\dot h(t)=-h(t)+u(t).
$$

设当前 Token 经过网络得到可变步长 $\Delta_t>0$，在这一小区间内作 ZOH，精确离散化为

$$
h_t=e^{-\Delta_t}h_{t-1}+(1-e^{-\Delta_t})u_t.
$$

现在让

$$
\Delta_t=\operatorname{softplus}(z_t)
=\log(1+e^{z_t}),\quad z_t=w^\top u_t+b.
$$

利用恒等式

$$
e^{-\operatorname{softplus}(z)}
=\frac1{1+e^z}
=1-\sigma(z),
$$

得到

$$
\boxed{h_t=(1-g_t)h_{t-1}+g_tu_t,\qquad g_t=\sigma(z_t).}
$$

这正是 **Mamba 论文 Theorem 1（§3.5.1，PDF 第 8 页）** 的关键联系：Selective SSM 在 $N=1,A=-1,B=1$ 时，数学上就包含了常见的 RNN Gate。

什么时候忽略 Token？当 $g_t\approx0$，状态几乎不变，相当于绕过无关 Token；什么时候覆写旧状态？当 $g_t\approx1$，新输入几乎取代过去。注意“记忆”和“覆写”本身是一个由输入决定的连续取舍，**不是有了门控便必然实现精确 Key–Value 检索**。网络能否学出正确 Gate 与状态容量仍需实验。

### 四步手算：为什么这能跨越任意数量的空白 Token？

令初始 $h_0=0$，四个输入为 $[5,100,-30,8]$。现在假设模型学出 $g=[1,0,0,1]$（极限值，实际 Sigmoid 只能逼近）。逐步更新：

$$
h_1=5,\quad h_2=5,\quad h_3=5,\quad h_4=8.
$$

第二、三步输入即使很大也不污染记忆，因为当前内容让模型选择了忽略；第四步输入才真正写入状态。若 $g_t$ 是固定常数，例如 0.5，则 $h_2=52.5$、$h_3=11.25$，模型做不到“某些 Token 对状态完全透明”。

这里用的是教学极限 Gate 和一维状态，不是声称真实 Mamba 在训练数据里会输出精确 0/1。

## 3. S6 算法真正改变了哪几个 Tensor Shape？

经典结构化 SSM 的输入 $X:[B,L,D]$，每个 Feature Channel 有一个 $N$ 维状态，静态参数 $\mathbf A,\mathbf B,\mathbf C$ 的基本形状类似 $[D,N]$，时间步长 $\Delta:[D]$。不论输入是哪枚 Token，它们对所有位置复用，因此可预先计算卷积核。

[原论文 Algorithm 1 / Algorithm 2（PDF 第 6 页）](https://arxiv.org/pdf/2312.00752) 的一张并排伪代码表直接对比 S4 与 **S6**：

| 参数 | 原先的结构化 SSM | S6（原文 Algorithm 2） |
|---|---|---|
| $\mathbf A$ | $[D,N]$ 训练参数 | $[D,N]$，仍是训练参数 |
| $\mathbf B$ | $[D,N]$ 或共享结构参数 | $[B,L,N]$，来自 $s_B(X)$ |
| $\mathbf C$ | $[D,N]$ 或共享结构参数 | $[B,L,N]$，来自 $s_C(X)$ |
| $\Delta$ | $[D]$，静态离散步长 | $[B,L,D]$，由输入产生并施加 softplus |
| 隐藏状态 | $[B,D,N]$（逐 Token） | $[B,D,N]$（逐 Token） |

注意 **S6 的 $B_t,C_t$ 不带完整的 $D$ 维展开**，而是对不同 Feature Channel 共享按 Token 产生的 $N$ 维向量，实际计算时广播到 $[B,L,D,N]$。完整隐藏状态若全序列显式存储，将具有 $[B,L,D,N]$ 大小；这是论文特意规避的中间张量。

论文的算法是：

$$
\begin{aligned}
B_t&=s_B(x_t),\\
C_t&=s_C(x_t),\\
\Delta_t&=\operatorname{softplus}(\Delta_{\rm bias}+s_\Delta(x_t)),\\
(\bar A_t,\bar B_t)&=\operatorname{discretize}(\Delta_t,A,B_t),\\
h_t&=\bar A_th_{t-1}+\bar B_tx_t,\quad y_t=C_th_t.
\end{aligned}
$$

其中 $\bar A_t,\bar B_t$ 需要注意按 Feature Channel 和 State Index 广播。**把 $A$ 也写成每 Token 由复杂网络输出的大型密集矩阵不是原文的 S6 实现**；作者正是利用结构化共享参数 $A$，让高维状态的成本可控。

### 理论公式与官方 Reference 实现到底是否逐项相同？

这里有一个会直接影响复现结果的区别。论文 §2 Eq.(4) 给出的 **ZOH 精确离散化**是

$$
\bar A_t=e^{\Delta_t A},\qquad
\bar B_t=\int_0^{\Delta_t}e^{A\tau}B_t\,d\tau.
$$

对 $A=-1,B_t=1$，它给出 $\bar B_t=1-e^{-\Delta_t}$，所以才严格推出前文 Theorem 1 的凸组合 Gate。**但作者的 [selective_scan_ref 代码](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py) 实际采用了另一种离散输入系数计算**：

~~~python
import numpy as np

# 只比较单通道单状态的一个 Token；不是训练 Mamba 模型
delta, A, B, u, h_prev = np.log(2), -1.0, 1.0, 2.0, 0.0
deltaA = np.exp(delta * A)
deltaB_u = delta * B * u               # 官方 Reference 的输入项近似
h_reference = deltaA * h_prev + deltaB_u

zoh_B = np.expm1(delta * A) / A * B    # expm1 避免微小 Δ 时相减失精
h_zoh = deltaA * h_prev + zoh_B * u
assert np.isclose(h_zoh, 1.)
assert np.isclose(h_reference, 2*np.log(2))
print("ZOH", h_zoh, "reference", h_reference)
~~~

这不是可随意忽略的“小数点实现差”。它的两个式子在有限 $\Delta$ 下**通常不严格相同**：

$$
\underbrace{h_t^{\rm ZOH}
=e^{-\Delta_t}h_{t-1}+(1-e^{-\Delta_t})u_t}_{\text{Theorem 1 使用的精确 ZOH 模型}}
\quad\text{vs}\quad
\underbrace{h_t^{\rm ref}
=e^{-\Delta_t}h_{t-1}+\Delta_tu_t}_{\text{Reference Scan 中的输入项近似}}.
$$

取 $h_{t-1}=0,u_t=2,\Delta_t=\ln2$，前者 $h_t=1$，后者是 $2\ln2\approx1.38629$。两者的一阶泰勒项一致，因为 $1-e^{-\Delta}=\Delta+O(\Delta^2)$，但后者在较大步长时并非同一个严格的 Gate 插值。因此**论文的门控联系应解释为所述 ZOH 理想模型的数学性质**，不能直接承诺作者的所有 CUDA/Reference 实现都逐元素满足 $h_t=(1-g_t)h_{t-1}+g_tu_t$。

这一差异也说明，审查序列模型必须同时看论文方程与官方实现：共享 $A$、输入相关的 $B_t,C_t,\Delta_t$ 是架构选择；怎样数值离散 $\bar B_t$ 则属于具体的算子实现。不能用较理想的公式证明某个近似实现具有它并未严格保持的性质。

### 一个能照着复写的最小形状一致性实验

令 $B_{\rm batch}=1,L=3,D=2,N=2$，为便于手算固定两通道的状态矩阵

$$
A=\begin{bmatrix}-1&-2\\-3&-4\end{bmatrix}\in\mathbb R^{D\times N},
\quad \Delta_{t,d}>0,
\quad B_{t,n},C_{t,n}\in\mathbb R.
$$

**这里的 $A$ 不是一个 $2\times2$ 密集的状态转移矩阵**，而是形状为 $[D,N]$ 的逐通道对角谱：$A_{d,n}$ 各自作用在 $h_{d,n}$ 上。真正展开后，每个 Token 对应状态 $h_t:[1,2,2]$，而非普通 Dense Matrix Multipy $A@h$。

按照上面的 Reference 离散约定，正确的广播形式是

$$
\begin{aligned}
\bar A_{t,d,n}&=\exp(\Delta_{t,d}A_{d,n}),\\
b_{t,d,n}&=\Delta_{t,d}B_{t,n}x_{t,d},\\
h_{t,d,n}&=\bar A_{t,d,n}h_{t-1,d,n}+b_{t,d,n},\\
y_{t,d}&=\sum_{n=1}^N C_{t,n}h_{t,d,n}.
\end{aligned}
$$

这四行精确说明了 Algorithm 2 为什么需要 $[B,L,D,N]$ 的**逻辑**中间系数，却不应该为每一步都把这个四维张量物化到 HBM。预测网络产生的是 $[B,L,N]$ 的 $B,C$ 与 $[B,L,D]$ 的 $\Delta$，随后在 Kernel 内逐 Feature/State 广播。

## 4. 为什么增加输入选择后，S4 的全局卷积捷径立刻消失？

设简化一维系统

$$
h_t=a_th_{t-1}+b_tu_t.
$$

连续展开前四项：

$$
h_3=a_3a_2a_1h_0+a_3a_2b_1u_1+a_3b_2u_2+b_3u_3.
$$

从第 $j$ 步输入传到第 $t$ 步的权重是 $b_j\prod_{k=j+1}^ta_k$。如果 $a_k$ 依赖每一步输入，**即便 $t-j$ 相同，不同上下文下的乘积也不相同**，因此不存在一个统一可重用的 $K_{t-j}$。

所以 Mamba 论文不是“用同一个静态卷积核优化得更快”，恰好相反，**它主动牺牲了固定核卷积这条高效路径，换取数据相关的状态更新**。原论文 Figure 1（PDF 第 3 页）蓝色 Projection 生成 $B_t,C_t,\Delta_t$，橙色状态空间更新发生在 GPU SRAM/HBM 的结构；Figure 2 / Algorithm 2 则解释内容选择的原因与符号变化。前者是方法概览，后者是真正的输入选择定义。

但是失去卷积不等于训练只能逐 Token 串行。接下来需要一个与卷积不同的并行化数学工具：**Prefix Scan**。

## 5. Selective Scan 的第一性推导：仿射递推为什么有结合律？

对每个时间位置，把递推写成作用在状态上的仿射变换

$$
T_t(h)=a_th+b_t,
$$

先执行 $T_1$ 再执行 $T_2$：

$$
T_2(T_1(h))=a_2(a_1h+b_1)+b_2
=(a_2a_1)h+(a_2b_1+b_2).
$$

因此定义两个仿射对的合并：

$$
(a_2,b_2)\circ(a_1,b_1)
=(a_2a_1,\;a_2b_1+b_2).
$$

函数复合天然满足结合律，因此

$$
(T_3\circ T_2)\circ T_1=T_3\circ(T_2\circ T_1).
$$

只要运算的顺序不变，括号可以任意重新组织。这意味着我们可以用**并行 Prefix Scan** 求出所有前缀的组合，而不是把整条链固定在一个单线程循环上。不能任意交换两步：对于矩阵 $A_t$，合成一般不满足交换律，尤其 $A_1A_2\ne A_2A_1$；即便标量乘法可交换，仿射部分也依赖顺序。

### 手算三步，把 Scan 的中间量与真实状态对齐

令 $(a_1,b_1)=(0.5,2)$、$(a_2,b_2)=(0.25,3)$、$(a_3,b_3)=(0.1,-1)$，初始 $h_0=4$。

直接递推：

$$
h_1=0.5(4)+2=4,\quad h_2=0.25(4)+3=4,\quad
h_3=0.1(4)-1=-0.6.
$$

合并前两步得到 $(0.125,0.25\cdot2+3)=(0.125,3.5)$；再和第三步合并得到 $(0.0125,0.1\cdot3.5-1)=(0.0125,-0.65)$。作用于初始状态 $h_0=4$：

$$
0.0125\cdot4-0.65=-0.6,
$$

完全一致。Scan 不是数值近似或训练时插入的梯度技巧，而是**同一个动态系统的代数重排**。对于 S6 的多通道/多状态，$\bar A_t$ 的结构（如逐元素对角化）让相应合成算子可以进一步高效实施。

## 6. “硬件感知”出现在论文里，但它不是本篇的算法动机

原论文 §3.3 解释纯 Scan 仍不够：如果先显式展开 $[B,L,D,N]$ 的隐藏状态到 HBM，再完整读取它做 Projection，状态维度 $N$ 会让 Memory Traffic 变大。Mamba 选择在 SRAM/寄存器的块级执行中融合 Discretize、Scan 和收缩，再将最后结果写回 HBM；训练的 Backward 可以重计算局部 Scan 信息，避免保留整份状态轨迹。

这是一项**把内容选择机制变成可训练网络**必需的实现手段，不应据此将 Mamba 归类成“主要发明一个 CUDA Kernel”。模型本身最大的改变是：**输入决定当前如何写入、衰减、读取隐藏状态**。若只复制这个优化内核而保留静态 $\Delta,B,C$，本质上没有实现 Selective SSM 的新表达力。

原论文 Figure 3（PDF 第 8 页）进一步对比 H3、Gated MLP 与 Mamba：

- 投影后一支做 Short Convolution → Selective SSM，提取局部模式并更新长状态。
- 另一支作为非线性门控，用 SiLU/Swish 后逐元素调制第一支的输出。
- 输出再投影回 Model Dimension，多个 Block 通过残差堆叠。

因此 Mamba Block 不需要再交替放置一个单独的 Attention Block 和一个 FFN Block，但“没有 MLP Block”不等于“里面一个线性投影和元素级非线性都没有”。在真实结构里，Projection、Depthwise Conv、SiLU 和 Gate 都存在。

~~~{mermaid}
flowchart LR
 X["输入 X:[B,L,d]"] --> IN["输入投影与分支"]
 IN --> V["主分支：Depthwise Conv + SiLU"]
 IN --> Z["门控支路 z"]
 V --> P["产生 B_t、C_t、Δ_t"]
 P --> S["Selective Scan：逐 Token 状态更新"]
 V --> S
 S --> G["与 SiLU(z) 逐元素相乘"]
 Z --> G
 G --> O["Output Projection + Residual"]
~~~

*自制模型机制图，非原论文 Figure；省略低层 Kernel 的参数布局与归一化细节。*

## 7. Attention 和 Mamba 的差别到底是计算复杂度，还是信息访问模式？

两个方面都不同。给定长度 $L$，标准 Causal Attention 显式在 Token Pair 上计算 Query/Key 相似性，最直接计算是 $O(L^2)$ 级别的 Token 对关系；它在生成时还需保存随 Context 增长的历史 KV。Mamba 的一个层只维护**固定形状的递推状态**，在其状态维度/通道固定时，单次 Decode 的历史状态大小不随 $L$ 增长。

但“固定状态占用”并不等价于“可以无限无损回忆历史任一精确 Token”。比如若输入包含 $M$ 个独立随机 Key–Value Pair，要在最后对任意 Key 正确取回其 Value，模型的固定容量必须承载足够信息。现实中有限精度隐藏状态存在信息容量限制；增大状态维数或层数能缓解但不能消除这个基本权衡。

[论文 Figure 2 右下](https://arxiv.org/pdf/2312.00752) 的 Induction Heads 实验验证在其有限规模测试里，Selection 足以学会某些内容相关检索；它不能据此证明 Mamba 与带任意长 KV Cache 的 Softmax Attention 在表达能力上严格等价。

如果有人说“Mamba 一定比 Transformer 更好”，要继续追问：模型大小、训练 Token、下游任务、Prompt 长度与所要求的**精确关联召回**是否一致。

## 8. 实证部分如何验证“有选择的状态更新”确实改善能力？

论文 [§4、Figure 4 与 Figure 5](https://arxiv.org/pdf/2312.00752) 包含多个不同问题：

**合成选择/复制任务**：比较普通 Copying 与 Selective Copying。前者可由固定核的时间规律解决，后者需要识别哪些 Token 有效；论文展示 Selective SSM 能解决后者并外推到训练时未覆盖的更长长度。这里才与前文的选择机制构成相对直接的因果证据。

**语言模型对照**：预训练损失、模型规模与训练 Token 数需要保持同协议。原文 Mamba-3B 在其训练配置下超过相近参数规模的 Transformer，并可与约 2 倍大小的部分模型在某些指标接近；不能从该比较推断“所有 2026 年 Transformer 都被取代”，因为 Data Recipe、Architecture、Scaling 和 Evaluation 均会变。

**吞吐**：论文摘要提出约 5× 自回归生成吞吐，是在作者选定模型/HW/配置的实测系统口径；这并不意味着隐状态更新的理论 FLOPs 恰好缩小五倍。

**多模态序列**：论文还研究了 Audio Waveform 与 DNA Sequence，说明作者希望 Selective SSM 是通用 Sequence Backbone；这不是仅凭文本 PPL 的一个离题补充，而是在检验**输入选择**对于不同离散/连续信号形态是否通用。

论文的基准并不单独证明哪一个门控公式必然是最优选择。$s_\Delta,s_B,s_C$ 的网络形状与初始化均属于作者选择；理论上同样能够选择内容的其他模型不一定要采用这三个确切函数。

## 9. 应用边界：从 S4 到 Mamba，改进了什么，又失去了什么？

改进的有：固定卷积的内容不敏感问题、Selective Copy 的输入间隔不确定问题、离散 Token 的上下文过滤能力；并且在 Scan 实现里仍能实现全序列训练与常量大小的单步状态。

代价与边界则是：动态 $A_t,B_t,C_t$ 使**固定卷积核不再存在**；训练依赖有效 Scan 实现；常量大小记忆意味着历史信息必然压缩；在高度依赖严格检索的大规模任务上不保证胜过 Full Attention；可变 $\Delta$ 也会影响状态衰减尺度，需要足够稳定的参数化和优化训练。

所以 Mamba 真正改变了**记忆规则**，而不只是用新的并行技巧计算 S4。下一篇 [Mamba-2](A060-mamba2-ssd.md) 不再主要问“怎样增添选择”，而是问：**给定一条时间变化的状态递推，是否存在一个与之严格等价的 Attention-like 矩阵？我们能否在保持线性状态表达的同时重新利用矩阵计算的优势？**
