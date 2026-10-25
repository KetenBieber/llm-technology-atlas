# BitNet：既然事后量化有损失，为什么不从训练一开始就限制权重为 1-bit？

> **原始论文**：[BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)，2023 原始论文；后续综述发表于 JMLR 2025。


前置：[GPTQ 的事后误差补偿](C041-gptq.md)、[QLoRA 的冻结低比特主干](C051-qlora.md)；下一篇：[BitNet b1.58](C054-bitnet-b158.md)。

## Post-Training Quantization 无论多聪明，都存在一个先后矛盾

训练一个 FP16 Transformer 时，优化器可以把参数选择到连续实数空间里的任意位置；训练完成后再把参数投影到少数格点，很可能偏离训练好的解。表示为

$$
W^\star=\arg\min_{W\in\mathbb R^n}\mathcal L(W),
\qquad
\widehat W=Q(W^\star).
$$

而若最终部署只能使用离散权重集合 $\mathcal Q$，理想目标其实是

$$
W^\star_{\mathcal Q}=\arg\min_{W\in\mathcal Q}\mathcal L(W).
$$

通常 $Q(W^\star)\ne W^\star_{\mathcal Q}$。因此**先做连续优化再投影**与**在低比特限制下寻找最优参数**不等价，这是从头训练低比特网络的第一性理由。

## 1-bit Weight 的神经网络怎么更新？

二值化可示意为 $Q(w)=\operatorname{sign}(w)$（实际 BitNet 量化还含归一化/缩放）。每次前向只使用二值化参数进行计算，但优化器维持连续的潜在权重变量 $W$ 进行更新。这就出现不可微问题：$\operatorname{sign}$ 的导数除阈值点外为 0。

常见低比特训练做法是采用 Straight-Through Estimator (STE)，在反传中用一个可导近似代替真实的量化导数：

$$
\frac{\partial\mathcal L}{\partial W}
\approx \frac{\partial\mathcal L}{\partial Q(W)}
\cdot \mathbf 1_{\{|W|\le c\}}.
$$

这里的截断直通估计是**解释机制的通用形式**，不是声称作者每个 BitLinear 都采用完全相同的 $c$ 和导数定义。重要的是 Forward 的离散路径与 Optimizer 的连续状态可以共存。

BitNet 将 Transformer 中主要 Linear 替换为 **BitLinear**，对权重采用 1-bit 量化训练，并对激活使用相应低精度量化和归一化。其它结构是否仍保持高精度（Embedding、某些算子或输出模块），必须依据论文实现；“1-bit Transformer”不意味着运行时全部状态只有单个位。

~~~{mermaid}
flowchart LR
 FP[潜在浮点参数 W] --> Q[BitLinear：前向 1-bit 量化]
 X[激活 X] --> ACT[归一化/低精度处理]
 Q --> G[低比特 Linear 计算]
 ACT --> G
 G --> L[语言模型 Loss]
 L --> STE[STE/近似梯度]
 STE --> FP
~~~

*自制训练计算图，非论文 Figure。注意 Backward 回到的是潜在训练参数，而不是直接更新不可微的整数编码。*

## 还原 BitLinear 的一次前向：从浮点潜变量到整数乘加

原始 [2023 年 BitNet PDF `2](https://arxiv.org/pdf/2310.11453) 的训练中保存浮点潜变量 $W$，前向通过量化规则得到二值 $\widetilde W$。激活不是原样 FP16 乘以一个 `sign(W)` 就结束，而是经过 Sub-LayerNorm 与量化。其实际计算包含**归一化、二值 Weight、8-bit Activation、输出缩放**多个阶段。

忽略一些符号细节，可将数据流概括为

$$
\widetilde W=Q_1(W;\alpha,\beta),\qquad
\widetilde x=Q_8(\mathrm{LN}(x);\gamma,\eta),
$$

$$
y \approx \frac{\beta\gamma}{Q_b}\,\widetilde W\widetilde x.
$$

原论文 Eq. (11) 给出含 Scale 的 BitLinear，$\beta=\|W\|_1/(nm)$ 这样的均值幅度因子不能省。这里的 $Q_b$、$\gamma$ 和偏移 $\eta$ 由激活量化的精确规则决定，实际复现必须按原 PDF Eq. (2)、(5)、(6)、(11) 编写，而非只凭上面的概念式训练。

### 为什么用 LayerNorm 还不够，要考虑方差？

设归一化后的每个输入维度满足零均值、单位方差，二值权重独立、同分布且取 $\pm\beta$。在一层宽度为 $n$ 的点积里，若不附加 Scale，输出方差数量级是

$$
\operatorname{Var}\left(\sum_{i=1}^n w_ix_i\right)\approx n\beta^2.
$$

对大模型的几千维宽度，这个量会随 $n$ 改变；而残差流与 Norm 需要稳定的激活尺度。原论文引入规范化/量化因子并结合 SubLN 来控制激活方差。**“二值化只省去乘法”没讲到 BitLinear 真正能够堆几十层训练的必要条件**。

## 二值参数和浮点潜变量如何跨越符号阈值？

在前向 $Q_1(w)\in\{-1,+1\}$，当潜变量 $w$ 从 $0.10$ 变到 $0.09$，前向二值结果完全不变。只有当它穿过 0 时，符号才可能变化。若直接对硬 `sign` 求导，区间内部梯度几乎处处为零，于是使用 STE 让梯度可以传回浮点潜变量。

例如优化器以学习率 $\eta=0.03$ 对 $w=0.02$ 做一次真实潜变量更新，若估计梯度 $g=1$，则 $w'=0.02-0.03=-0.01$，二值值才会由 +1 翻成 -1。这个例子展示了**离散前向对小步更新可能不敏感**，也解释原论文 `2.2 为什么特别研究较大学习率的收敛效果；不能由一个玩具例子断言学习率越大就越好。

论文还明确：梯度、优化器状态和潜在训练权重以较高精度存储，以维持训练稳定性。故即使 inference Weight 是 1 bit，**训练期间绝不是所有参数状态都能压到 1 bit**。

## 模型并行时的同步为何会成为隐藏代价？

BitNet 如果对完整矩阵计算统一的 $\alpha,\beta$ 或 Activation $\gamma,\eta$，在 Tensor Parallel 切分后不同 GPU 未必能访问整个 Tensor 的统计量。每层对这些尺度执行全局 All-Reduce，通信和同步可能成为额外瓶颈。原论文 `2.1 提出 **Group Quantization / Group Normalization**，使每个 Partition 对自己的局部 Group 估计 Scale，从而减少跨设备统计所需的额外通信。

这里有精度—并行代价折中：Group 越小，Scale 更贴合本地分布，但更多的 Group Metadata 与 Kernel 复杂度也可能提高。应在相同模型 TP 配置下比较量化质量、All-Reduce 时间和 GPU 利用率，而不能只比较模型权重的 nominal bit 数。

## “只有加减法”是数学性质还是部署现实？

若二值权重属于 $\{-1,+1\}$，乘法 $xw$ 确实可改写为 $+x$ 或 $-x$ 的符号选择，但实际加速依赖设备上是否有高效的二值/低比特矩阵乘 Kernel。若将 1-bit 值解包成 FP16 后送普通 GEMM，能耗和延迟收益会被解码开销侵蚀。模型训练时也仍需较高精度的优化器、梯度等状态，因此单看权重 bit-width 不等于训练全程显存降低 16 倍。

[BitNet 原始论文](https://arxiv.org/abs/2310.11453) 报告从头训练的低比特 Transformer 在所测语言建模规模下可与 FP16 参照取得有竞争力的质量；后来 [JMLR 2025 的统一研究](https://www.jmlr.org/beta/papers/v26/24-2050.html) 进一步覆盖 BitNet b1 和 b1.58。不可以把 2025 的新模型规模和实测速度倒填成 2023 原始论文的成绩。

## 原文 Figure 1 和 Table 1–2：图表的命题并不相同

[2023 年原始 PDF Figure 1](https://arxiv.org/pdf/2310.11453) 同时表达 PTQ 与 BitNet 从头训练的区别，以及在作者计算能耗模型下的理论/经验路线；图上估算的节能随模型规模变化，不能把它当作所有实际部署硬件的电费账单。

原文的 `2.3、Table 1–2 使用特定工艺节点的 ADD/MUL 单操作能耗估计，并用矩阵乘法算术次数推导能耗；这是**模型化估计**，不是对不同服务器供电接口进行普遍实测。可以自己根据 $m\times n$ 与 $n\times p$ 的乘法，复算标准 GEMM 约有 $mnp$ 次乘法、$mp(n-1)$ 次加法，再比使用二值符号选择可减少多少一般乘法。真正端到端，还需加入 Attention、Embedding、Activation 量化以及 Cache/访存。

[BitNet b1.58](C054-bitnet-b158.md) 增加零状态，虽然两条方法路线相近，其实验模型、训练 Token、归一化与推理 Kernel 并不完全相同，2024 新工作的数据不能回填成原始 2023 BitNet 的观察结果。

## 下一步：二值为何还不如三值？

二元 $\{-1,+1\}$ 的每个权重只有符号，不能直接表达零。对许多稀疏或接近零的通道而言，把接近零的权重强制推为 $\pm1$ 可能增加量化误差。[BitNet b1.58](C054-bitnet-b158.md) 在几乎相同的极低位宽下增加了 $0$，允许通过整数值 $\{-1,0,1\}$ 提供更精细的表征。
