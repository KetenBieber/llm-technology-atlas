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

## “只有加减法”是数学性质还是部署现实？

若二值权重属于 $\{-1,+1\}$，乘法 $xw$ 确实可改写为 $+x$ 或 $-x$ 的符号选择，但实际加速依赖设备上是否有高效的二值/低比特矩阵乘 Kernel。若将 1-bit 值解包成 FP16 后送普通 GEMM，能耗和延迟收益会被解码开销侵蚀。模型训练时也仍需较高精度的优化器、梯度等状态，因此单看权重 bit-width 不等于训练全程显存降低 16 倍。

[BitNet 原始论文](https://arxiv.org/abs/2310.11453) 报告从头训练的低比特 Transformer 在所测语言建模规模下可与 FP16 参照取得有竞争力的质量；后来 [JMLR 2025 的统一研究](https://www.jmlr.org/beta/papers/v26/24-2050.html) 进一步覆盖 BitNet b1 和 b1.58。不可以把 2025 的新模型规模和实测速度倒填成 2023 原始论文的成绩。

## 下一步：二值为何还不如三值？

二元 $\{-1,+1\}$ 的每个权重只有符号，不能直接表达零。对许多稀疏或接近零的通道而言，把接近零的权重强制推为 $\pm1$ 可能增加量化误差。[BitNet b1.58](C054-bitnet-b158.md) 在几乎相同的极低位宽下增加了 $0$，允许通过整数值 $\{-1,0,1\}$ 提供更精细的表征。
