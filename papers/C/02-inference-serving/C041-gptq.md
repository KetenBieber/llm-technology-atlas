# GPTQ：为什么近似二阶误差补偿能支持 3–4 bit 权重？

> **原始论文**：[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://openreview.net/forum?id=tcbBPnfwxS)，ICLR 2023（会议正式版本亦使用 OPTQ 题名）。 [官方/作者 PDF](https://spcl.ethz.ch/Publications/.pdf/2023_iclr_gptq.pdf)。


技术前置：[SmoothQuant 的 W8A8](C040-smoothquant.md)、[LLM.int8 的权重/激活分路](C039-llm-int8.md)。对照：[AWQ 的无梯度显著通道保护](C042-awq.md)。

## 仅把 FP16 权重四舍五入到 4-bit，为何有时很伤模型？

设一层权重为 $W$，用 $Q(W)$ 表示量化后的权重，输入校准样本堆成 $X$。我们真正关心的不只是每个 Weight 的数值差 $|W_{ij}-Q(W_{ij})|$，而是量化后 Linear 层的输出偏差：

$$
\mathcal E(Q(W))=\|XW-XQ(W)\|_F^2.
$$

展开得到

$$
\mathcal E=\operatorname{tr}\Big((W-Q(W))^\top X^\top X(W-Q(W))\Big).
$$

令 $E=W-Q(W)$，出现了 $H=X^\top X$ 这个正半定的二阶 Gram 矩阵。$X^\top X$ 度量“沿哪个输入方向的误差会被校准数据放大”。如果某个 Weight 的误差位于频繁激活的相关方向，输出偏差可能远高于同等数值误差的其他 Weight。这说明按 Weight 自身的绝对值最小化量化误差，不等于最小化 Layer 输出误差。

## 核心不是聪明地 Round 一次，而是补偿后续尚未量化的权重

GPTQ 沿用 Optimal Brain Quantization 的思想：当某个权重决定落在有限 bit 格点上时，已经引入的输出误差部分可以通过调整**剩余尚未量化的相关权重**来抵消。它将这类近似二阶更新组织成一列一列的矩阵运算，使用校准 Gram/Hessian 的逆相关信息传播修正，而不是为每个 Weight 单独做昂贵的训练重建。

假设只讨论一个输出列参数向量 $w$，一次量化固定某个分量 $w_i$，误差为 $\delta$。在二次型近似下，若其他分量可调整 $\Delta w_{-i}$，可以解

$$
\min_{\Delta w_{-i}}\;\|X(\delta e_i+\Delta w_{-i})\|_2^2.
$$

写出一阶驻点条件，便得到按 $H$ 的相关结构补偿的解。这个推导说明“为什么二阶信息对量化有帮助”；论文实际的 Batch/Block 更新与 Cholesky 数值实现要按 §3 的算法来看，不能把上面这个教学式当成整个 GPTQ 实现。

~~~{mermaid}
flowchart TD
 X[校准激活 X] --> H[估计 Hessian/Gram]
 W[FP 权重 W] --> Q[选择列并量化]
 H --> C[计算误差补偿方向]
 Q --> C
 C --> N[更新尚未量化的后续列]
 N --> Q
 Q --> O[低比特权重 + Scale]
~~~

*自制机制图，非原论文 Figure。原论文 Figure 1 比较不同模型规模与量化方案的精度，§3 解释为什么需要快的批量更新而不是原 OBQ 的逐权重更新。*

## 把“利用 Hessian 补偿”真正推导成更新量

对某个输出通道的权重向量 $w\in\mathbb R^d$，量化后误差 $e=w-\hat w$。给定输入校准矩阵 $X\in\mathbb R^{N\times d}$，优化目标可写为

$$
\min_e\;e^\top H e,\qquad H=X^\top X+\lambda I.
$$

为了保证量化位置 $i$ 的误差固定为 $e_i=\delta$，使用 Lagrange 乘子 $\nu$：

$$
\mathcal J(e,\nu)=e^\top H e+\nu(e_i-\delta).
$$

一阶条件是 $2He+\nu e_i^{(\mathrm{basis})}=0$，其中 $e_i^{(\mathrm{basis})}$ 是第 $i$ 个单位基向量。将 $H$ 视为正定，得到

$$
e^\star=
\frac{\delta}{(H^{-1})_{ii}}H^{-1}e_i^{(\mathrm{basis})},
\qquad
\min\mathcal E=\frac{\delta^2}{(H^{-1})_{ii}}.
$$

因此**被固定量化的一个分量，会沿 $H^{-1}$ 的相关方向调整其他未固定参数**。符号约定要注意：上式 $e=w-\hat w$ 表示原权重与新权重之差；若实现以 $\Delta w=\hat w-w$ 为变量，则更新符号恰好相反。论文采用 Cholesky 分解和分块更新完成批量修正，而不是为每个 Weight 从头求一次完整矩阵逆。

### 为什么 $H$ 很相关时补偿收益更大？

用一个二维 Gram 例子：

$$
H=\begin{bmatrix}1&0.9\\0.9&1\end{bmatrix}.
$$

假设固定 $e_1=\delta=0.1$，若完全不调整第 2 维，则误差为 $[0.1,0]H[0.1,0]^\top=0.01$。把第 2 维误差调为 $e_2=-0.9\delta=-0.09$，则误差减为

$$
0.1^2(1-0.9^2)=0.0019.
$$

这说明为何“量化第 1 个 Weight 时调整其他 Weight”比各自独立 Round 更好。但若 $H$ 近奇异，不做数值 Damping 会导致逆矩阵噪声巨大；若校准数据缺少某些输入方向，也可能把不正确的相关性带入量化误差更新。

[GPTQ 原文 ICLR 2023](https://openreview.net/forum?id=tcbBPnfwxS) 的 `3.1–3.2 主要解释成块更新、任意列序和 Cholesky 数值稳定的实现取舍；因此推导只说明**一次补偿为什么有效**，不代表本文已完全复现作者的分块量化 Algorithm。读原文速度表时应把“一次性离线 3-bit 量化 175B 所需 GPU 小时”与“部署后每 Token 延迟”分成两个测量。

## 为什么 Cholesky/重排/Group Size 不能当成实现小事？

理论上用 $H^{-1}$ 会遇到 $H$ 奇异或病态的情况，必须引入适当的数值稳定措施，如 Damping；大矩阵的补偿若逐元素处理也会有严重的计算与访存开销。GPTQ 用成块更新和矩阵计算减少串行循环的成本，在数百亿/千亿参数时才有可实现性。

量化到 4-bit 时每个 Weight 不是独立存 0.5 字节：需要 Pack Scale/Zero Point，Group Size 越小量化精度可能越好，但元信息占比、GPU Kernel 访问复杂度也越高。计算 $X^\top X$ 的校准数据代表性决定二阶近似是不是当前部署分布的良好 Proxy。

## 低比特带来的是权重存储收益，速度还需要 Kernel

以参数量 $P$ 的 Decoder 为例，FP16 权重理想存储约 $2P$ 字节；4-bit 权重理想存储为 $P/2$ 字节，理论权重压缩 4 倍，但端到端显存还包括 KV/激活与量化元信息。GPU 在 FP16 激活乘 4-bit 压缩权重时可能需要先反量化到更高精度；实际速度取决于 Weight-Only GEMM/GEMV Kernel，而不是某篇论文的 PPL 表格。

[ICLR 2023](https://openreview.net/forum?id=tcbBPnfwxS) 论文与作者版本 [GPTQ](https://arxiv.org/abs/2210.17323) 的命名存在 OPTQ/GPTQ 差异，应按同一原始工作理解。作者报告将约 175B 的权重在约四 GPU 小时内量化至 3–4 bit，结合专用 Kernel 在所测设备上实现端到端加速；不同硬件和 Weight Layout 的速度不能作为现代所有平台的默认承诺。

**反例意识**：若任务强依赖某些罕见输入方向，但 Calibration $X$ 几乎没有这些样本，$H$ 的估计就可能低估关键误差。对照 [AWQ](C042-awq.md) 时，要特别注意两者都用 Activation 信息，但 AWQ 以简单显著性/缩放规避量化敏感，不执行同样的二阶逐列误差补偿。
