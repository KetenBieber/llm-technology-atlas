# SpinQuant：随机旋转已能压低离群值，为什么还要在正交流形上学习旋转？

> **原论文**：Liu 等，[SpinQuant: LLM Quantization with Learned Rotations](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html)，ICLR 2025 正式版。[正式 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/e5b1c0d4866f72393c522c8a00eed4eb-Paper-Conference.pdf) · [作者代码](https://github.com/facebookresearch/SpinQuant)。
>
> **技术谱系**：[QuaRot 的无损正交重参数化](C055-quarot.md) → **本篇：在保持正交约束下学习旋转** → [FlatQuant：更一般的可学习低成本仿射](C057-flatquant.md)。前置：[GPTQ](C041-gptq.md) 的量化误差补偿、[RMSNorm](../../A/03-transformer/A016-rmsnorm.md) 的范数几何。

## 旋转不是为了改变模型语义，而是改变 Round 看到的坐标

从最朴素的线性层 $Y=XW$ 开始。对于任何正交矩阵 $R$，可以用 $X'=XR$、$W'=R^\top W$ 表达同一个全精度线性映射：

$$
X'W'=XR R^\top W=XW.
$$

如果两者最后都进入有限位宽的量化器 $\mathcal Q_b$，真实执行的是

$$
\widehat Y_R=\operatorname{Dequant}\!\left(
\mathcal Q_b(XR)\,\mathcal Q_b(R^\top W)
\right),
$$

而不是 $XRR^\top W$。因为 $\mathcal Q_b$ 包含 Round/Clip/Scale，通常

$$
\mathcal Q_b(XR)\mathcal Q_b(R^\top W)\ne
\mathcal Q_b(X)\mathcal Q_b(W).
$$

**正交变换保护的是全精度模型函数，不保护离散量化误差。** 因此存在一个被旧论文忽略的自由度：同一个网络有无数等价的旋转参数化，但量化之后它们的精度并不一样。QuaRot 的随机 Hadamard 已是一种很好的统计选择，却没有以最终任务损失来“挑选最好的旋转”。

### 一个 2×2 例子：同一全精度函数，量化失真为什么不同？

设 $x=[4,0]$、$W=[1,0]^\top$，目标输出 $y=4$。若使用单位旋转 $R=I$，在对称 INT2（整数码点 $-1,0,1$）的逐张量 Scale 下，两个矩阵都只有一个非零坐标，量化可恰好表示它们。若采用 $R=H_2$，则 $x'=[2\sqrt2,2\sqrt2]$、$W'=[1/\sqrt2,1/\sqrt2]^\top$；采用各自 MaxAbs Scale、相同 $R$ 时仍可恰好重建。**这说明“旋转必然改善量化”的命题不能由正交性推出，甚至存在旋转前后都完全正确的反例**。

换成一个有极端值并有相互抵消分量的权重矩阵，在整张量共用 Scale 和粗格点时，换一套 $R$ 可能使原本有效的细小坐标被 Round 为零，也可能让它们远离阈值，因而改变误差。这种差异来自 $\ell_\infty$ 动态范围、通道相关性与输出敏感性，而不是 $R$ 改变了 $\ell_2$ 范数。为了决定哪个 $R$ 更适合实际模型，需要让训练样本的 Loss 发出信号。

## Figure 4 给出了学习旋转的真正理由：随机试 100 次仍有 13 分落差

[ICLR 2025 正式 PDF Figure 4（第 4 页）](https://proceedings.iclr.cc/paper_files/paper/2025/file/e5b1c0d4866f72393c522c8a00eed4eb-Paper-Conference.pdf) 横轴是 LLaMA-2 7B 的 **8 个 Zero-Shot Reasoning Task 平均准确率**，纵向是不同随机种子下结果的分布；三组分别为 Random Floating-Point Orthogonal Rotation、Randomized Hadamard 与 Cayley-optimized Rotation。作者重复了 **100 个随机初始化**：不同随机旋转导致最高与最低相差约 13 分，随机 Hadamard 本身也有约 6 分的性能跨度；优化后的分布更集中、中心更高。

这是一个关键证据区分：**Figure 2** 展示旋转前后的激活异常值分布；**Figure 3** 展示异常值比例与量化误差；**Figure 4** 则表明“异常值总体下降”并不唯一决定任务质量。同一类随机正交变换中，哪些误差落在下游敏感方向上依然重要。Figure 4 不是原始 FP16 模型不同初始化的波动，而是**固定同一模型参数，仅改变量化前的旋转**后得到的任务表现差异。

这也解释作者为什么不能只用“观察到旋转后 MaxAbs 变小”作为损失函数：更合理的目标是直接衡量量化网络在校准样本上的任务/重建 Loss。

## 先把四个旋转矩阵在 Transformer 中的作用划清

论文 [Figure 1（PDF 第 2 页）](https://proceedings.iclr.cc/paper_files/paper/2025/file/e5b1c0d4866f72393c522c8a00eed4eb-Paper-Conference.pdf) 分为三个区域：

- 图 1(a) 的 **$R_1$** 是共享 Residual Stream 坐标旋转；它在 Token Embedding 输出侧进入，所有读取 Residual 的 Q/K/V/Up/Gate Projection 需要与之配对吸收，而 Block 输出端把旋转坐标继续传到下一层。$R_1\in\mathbb R^{d\times d}$，是模型宽度的旋转。
- 图 1(b) 的 **$R_2$** 是 Attention Value 与 Output Projection 之间的 Head-wise 旋转；$R_2\in\mathbb R^{d_h\times d_h}$，每层可不同，在 $V$ 支路与对应 $W_O$ 上成对吸收。
- 图 1(b)(c) 的 **$R_3,R_4$** 是用于消除 Attention KV 与 MLP 中间 Activation 离群值的**在线 Hadamard**。它们不能全部消失在 Linear Weight 里，运行时需要实际执行；论文选择将其保持为计算高效的 Hadamard，而不是同样学习全矩阵。

用 Head 维度举例，$V:[B,T,H,d_h]$，如果每个 Head 右乘同一个 $R_2$，那么 Attention 聚合后的 $PV$ 也右乘 $R_2$；只需在 $W_O$ 的对应 Head 子块左侧使用 $R_2^\top$，整体输出恢复原值。这里利用的是 $P(VR_2)=(PV)R_2$，**不能把 Head 旋转当成对所有 Q/K/RoPE 也随意适用的变换**。

~~~{mermaid}
flowchart LR
 X["残差流 X"] --> R1["R1：全层正交变换，离线吸收"]
 R1 --> ATT["注意力: Q / K / V"]
 ATT --> R2["R2：Value 与 Output 成对吸收"]
 R2 --> R3["R3：在线 Hadamard，KV 侧"]
 R1 --> MLP["FFN：Gate + Up + SwiGLU"]
 MLP --> R4["R4：在线 Hadamard，Down 之前"]
 R3 --> T["量化输入 / KV"]
 R4 --> T
 T --> VERIFY["真实量化网络 Loss"]
 VERIFY --> OPT["只优化 R1/R2；R3/R4 固定结构"]
 OPT --> R1
~~~

*自制解释图，非原论文 Figure；环路仅代表离线校准迭代，并非每次 Decode 在线反传。实际 R3/R4 应结合作者 Figure 1 的 Attention / FFN 位置解读。*

作者因此提供两个工作模式：**SpinQuant-no-had** 只使用可离线吸收的 $R_1,R_2$，适合 W4A8KV8 等没有极端 Activation 压缩的工况；**SpinQuant-had** 再加入 $R_3,R_4$ 在线变换，适合 W4A4KV4 这样的更困难工况。不能把它们两个模式的精度和 Runtime 数据混成一套配置。

## 关键数学：为什么直接训练正交矩阵会破坏其不变性？

如果把 $R$ 当作普通可学习矩阵，直接 SGD：

$$
R_{\rm naive}=R-\eta\nabla_R\mathcal L_Q
$$

一般使 $R_{\rm naive}^\top R_{\rm naive}\ne I$。一旦不正交，之前基于 RMSNorm 范数不变、$RR^\top=I$ 的模型等价论证便失效。**不能只在最初初始化为 Hadamard，就以为更新 100 步后它还自动正交**。

约束集合

$$
\mathcal M=\{R\in\mathbb R^{d\times d}:R^\top R=I\}
$$

是正交矩阵构成的流形。SpinQuant 优化的是

$$
\min_{R_1,R_2\in\mathcal M}
\mathcal L_Q(R_1,R_2\,|\,W_{\rm frozen},X_{\rm calib}),
$$

在校准期间**冻结原模型权重**。这个损失是 Quantized Network 的 Loss；如果不在优化路径里包含 Quantizer，仅训练全精度网络，梯度不会指出哪种 $R$ 更有利于 Round。

### Cayley 变换怎样在每一步精确保留正交约束？

取任意反对称矩阵 $A^\top=-A$，令 $c=\eta/2$，定义

$$
C(A)=(I-cA)^{-1}(I+cA),\qquad R'=C(A)R.
$$

因为 $(I-cA)$ 与 $(I+cA)$ 都是同一矩阵 $A$ 的多项式，可以交换。对反对称矩阵有

$$
C(A)^\top=(I-cA)(I+cA)^{-1}=C(A)^{-1},
$$

因此 $C(A)^\top C(A)=I$，进而

$$
R'^\top R'=R^\top C(A)^\top C(A)R=I.
$$

这就严格解释了为什么 **Cayley SGD 不是随意更新加上“正交惩罚项”**：无论取多大的一个有限合法步，只要按该有理变换精确计算并从正交 $R$ 出发，旋转仍在正交集合内。数值 FP16/近似逆会产生有限精度偏差，实际使用稳定迭代核。

[正式 PDF Eq. (2)–(4)，第 5–6 页](https://proceedings.iclr.cc/paper_files/paper/2025/file/e5b1c0d4866f72393c522c8a00eed4eb-Paper-Conference.pdf) 还给出了如何将普通 Loss Gradient $G=\nabla_R\mathcal L_Q$ 投影到符合约束的方向、构造反对称 $A$ 的办法，以及 Cayley 迭代计算。不能仅复制普通 $\nabla_R\mathcal L$，就说实现了作者的 Manifold Optimizer。

### 用 2×2 的小例子直接检查

取

$$
A=\begin{bmatrix}0&-1\\1&0\end{bmatrix},\quad c=0.2,\quad R=I.
$$

则

$$
C(A)=\frac1{1+c^2}
\begin{bmatrix}
1-c^2&-2c\\2c&1-c^2
\end{bmatrix}
=\begin{bmatrix}
12/13&-5/13\\5/13&12/13
\end{bmatrix}.
$$

每列长度是 $(12^2+5^2)/13^2=1$，两列内积为零；这是一个可直接验证的二维旋转。相反，若误用 $I+cA$ 当成 $C(A)$，列范数平方为 $1+c^2=1.04$，**一步就不再正交**。这组数据为教学构造，不是论文实际优化迭代数值。

## “只训练旋转”究竟要多少样本与成本？

论文 `4.1 记录了可复核的校准流程：使用 **800 条 WikiText-2 校准样本，100 次 Cayley 优化迭代**；$R_1$ 是全网络共享宽度矩阵、$R_2$ 每层一个 Head 旋转，其他预训练权重始终固定。主表中旋转学习主要围绕**激活量化、权重仍为 FP16** 的训练网络进行；旋转训练完成之后，再对旋转后的 Weight 使用 GPTQ，后者另外使用 128 条、长度 2048 的 WikiText-2 样本。这两段不能简称成“用 128 条数据训练旋转”或“对所有原权重做反向微调”。

论文给出的耗时也是不同模型的：LLaMA-2 7B 约 25 分钟、13B 约 30 分钟、70B 约 3.5 小时；这些是作者环境下的校准成本，不是运行一次单 Token 推理的开销。$R_1,R_2$ 约占模型权重规模的 0.26%，且可被折叠到最终的量化 Weight 中；在线 $R_3,R_4$ 在作者执行条件下引入约 8% 的网络延迟开销。没有在线 Hadamard 的配置不能无条件复现 W4A4 的精度。

## 把 Table 1 真正读成一条决策规则

[ICLR 2025 正式 PDF Table 1（第 7 页）](https://proceedings.iclr.cc/paper_files/paper/2025/file/e5b1c0d4866f72393c522c8a00eed4eb-Paper-Conference.pdf) 纵向分成 4-8-16、4-8-8、4-4-16、4-4-4 四组，含义为 Weight–Activation–KV 各自 bit，不要把数字顺序当成 W-KV-A。

在 LLaMA-2 7B **4-4-4** 的 8 项任务平均准确率与 WikiText-2 PPL：

| 方案 | 8-Task Avg（越大越好） | Wiki PPL（越小越好） |
|---|---:|---:|
| FP16 16-16-16 | 66.9 | 5.5 |
| SpinQuant-no-had | 56.0 | 9.2 |
| **SpinQuant-had** | **64.0** | **5.9** |
| LLM-QAT（同列） | 44.9 | 14.9 |

这里 64.0 相对 66.9 只差 **2.9 个百分点**；no-had 在极端 A4 下落差为 10.9。它既支持在线 Hadamard 的**必要性（在所测 A4 场景）**，也证明“学习 $R_1,R_2$ 就够了”的说法不正确。

同一 Table 1 的 LLaMA-3 8B 4-4-4 配置，FP16 为 69.6，no-had 为 52.6，had 为 65.5，仍存在剩余差距；这提醒我们所谓“降低 FP16 Gap”并不是所有模型恢复完全精度。

**Table 1 的部分 Baseline 是作者重新运行代码得到，部分 AWQ/OmniQuant/QuIP# 数值引用原论文**，原文 Table Caption 明确标注来源。不同表格、不同官方 Repository、不同 6-task/8-task 协议，不应随意跨文计算“比 QuaRot 高几分”。

## Figure 4 的因果边界与现实工程取舍

这项工作的贡献不是“有个 Cayley 公式”，而是将**量化之后才显现的坐标敏感性**转化为一个可优化的问题，并确保优化不改变全精度网络的函数。它仍有几个边界：

- 训练只使用有限校准样本，若部署分布改变、出现新的大幅离群 Token，未必能维持相同收益。
- 旋转矩阵的离线学习产生额外一次性校准成本，不能和零训练成本的 QuaRot 完全等价比较。
- 有 RoPE、Norm、Residual 与 Gated FFN 的真实模型，吸收位置不满足等价条件就会导致**未量化时也已改变网络**；必须先执行浮点一致性验证。
- 学到的旋转改善低位宽 Accuracy，但生成速度需要匹配 GPU Kernel、KV Cache 精度与在线 Hadamard；若只复现 PPL 不等于已验证加速。
- Cayley 保正交而不保证优化问题的全局最优，也不保证对每种 Quantizer/Layer 的误差都单调下降。

下一篇 [FlatQuant](C057-flatquant.md) 将进一步挑战“必须保持全网络正交”这一条件：只要一个 Linear Layer 前后成对插入互逆的**可逆变换**，全精度输出同样可以不变；那为什么不学习更一般的非正交变换，且把它拆成两个小矩阵来减少 Runtime？
