# GPTQ：为什么近似二阶误差补偿能支持 3–4 bit 权重？

> **原始论文**：Frantar 等，[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://openreview.net/forum?id=tcbBPnfwxS)，ICLR 2023。[arXiv v2／带 ICLR 2023 首页的论文 PDF](https://arxiv.org/pdf/2210.17323)；以下的 Figure、Algorithm 与公式均以该版本为准。


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

*自制机制图，非原论文 Figure。原论文 Figure 1 比较不同模型规模与量化方案的精度；Figure 2 和 Algorithm 1 才画出／写出真正的分块量化执行路径。*

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

[GPTQ §4](https://arxiv.org/pdf/2210.17323) 进一步解决**任意固定列序、延迟分块更新、Cholesky 稳定化**三个问题：上述二次型推导说明一次补偿为什么有效，却尚未解释大模型量化为何能在几小时内完成。下节把数学更新与 Algorithm 1 的执行路径接起来。

## 从一次补偿到 Algorithm 1：为什么一行一行量化太慢？

为直接对应原文 Algorithm 1，这里换用输出通道为**行**的惯例：$W\in\mathbb R^{d_{out}\times d_{in}}$，校准激活 $X\in\mathbb R^{d_{in}\times N}$，因此 $WX$ 的 Shape 为 $[d_{out},N]$，正则化后的 Hessian 是

$$
H=2XX^\top+\lambda I\in\mathbb R^{d_{in}\times d_{in}}.
$$

本文开头使用的是 $Y=XW$、输入通道在**列**的记号；两者通过转置对应，不能直接把一个布局中的 $X^\top X$ 与另一个布局中的 $XX^\top$ 混用。因子 2 不改变理想最优补偿的方向，但实现要统一自己的 $H$ 定义和 Damping。

假设当前待处理的第 $j$ 列为 $W_{:,j}\in\mathbb R^{d_{out}}$，对各输出行同时执行同一个列顺序。作者的关键观察是：OBQ 原先逐输出行寻找最优量化位置，带来大量重复求逆更新；**固定所有行的列顺序**只需要维护一套与输入统计相关的 Hessian 信息。这是原文 §4 Step 1 的经验性效率取舍，不是声称固定顺序对所有矩阵都是精度最优。

更进一步，把 $d_{in}$ 列按 $B$ 列分块。块内仍依次做量化和局部误差校正，块外远端列的更新先累计，待本块结束后一次性矩阵乘法施加。这不会凭空减少所有算术量，却把大量低算术强度的细粒度访存，合并成 GPU 更擅长执行的矩阵运算。原文 §4 Step 2 的一个具体设置是 $B=128$，这是论文实现选择，不是算法要求所有层必须取 128。

~~~text
输入：浮点权重 W:[d_out,d_in]，校准 X:[d_in,N]，位宽和分组量化器
建立 H = 2 XX^T + damping，并用 Cholesky 预计算稳定的逆曲率信息
for 每一列块 [i,i+B):
    for 块内列 j 从左到右:
        q_j = quantize(W[:,j])                 # 使用当前已补偿的权重
        e_j = (W[:,j]-q_j) / U[j,j]            # U 为原 Algorithm 1 的三角形式
        W[:,j:i+B] -= outer(e_j,U[j,j:i+B])    # 只立即改动本块
        保存 q_j 与 e_j
    W[:,i+B:] -= E_block @ U[i:i+B,i+B:]       # 一次更新剩余列
输出：Q、各组量化元信息
~~~

*教学伪代码，按论文 Algorithm 1 的行/列布局及分块顺序改写，未在用户机器上执行；其中 $U$ 是论文写作 $\operatorname{Cholesky}(H^{-1})^\top$ 的上三角因子，不是原始 $H^{-1}$。*

请对照 [原论文 Figure 2 与 Algorithm 1（PDF 第 4–5 页）](https://arxiv.org/pdf/2210.17323)：Figure 2 中白色列为当前处理列、粗框是本块、蓝色为块结束后统一更新的未量化区域；Algorithm 1 的两个循环分别对应上述两级更新。它们支持的是**实现步骤与访存组织**，不是量化后准确率本身。准确率要回 Figure 1：左侧 OPT 家族是 **4-bit**、右侧 BLOOM 家族是 **3-bit**，横轴为参数规模（十亿），纵轴为 WikiText2 PPL（越低越好），三条曲线比较 FP16、RTN 和 GPTQ；不同家族与位宽不能合成一项统一的“提升百分比”。

### Cholesky 不是一个只为了跑得快的可选装饰

作者 §4 Step 3 指出，反复用矩阵逆的消元式更新，会因有限精度误差使计算出的“逆 Hessian”不再正定，进而沿错误方向大幅修改尚未量化的 Weight。给 $H$ 加 Damping，并预先构造稳定的 Cholesky 因子，可避免在循环中反复进行不稳定的小矩阵求逆。这不等于数学上量化误差始终为零，也不保证很小或偏置严重的校准集一定适用。

原论文的三项改进可总结为：**统一列序减少重复工作 → Lazy Batch 增大算术强度 → Cholesky 避免反复更新逆矩阵积累数值误差**。只抄一个 $H^{-1}$ 补偿公式而漏掉后三项，就无法解释 GPTQ 为什么能用于千亿参数权重。

## 为什么 Cholesky/重排/Group Size 不能当成实现小事？

理论上用 $H^{-1}$ 会遇到 $H$ 奇异或病态的情况，必须引入适当的数值稳定措施，如 Damping；大矩阵的补偿若逐元素处理也会有严重的计算与访存开销。GPTQ 用成块更新和矩阵计算减少串行循环的成本，在数百亿/千亿参数时才有可实现性。

量化到 4-bit 时每个 Weight 不是独立存 0.5 字节：需要 Pack Scale/Zero Point，Group Size 越小量化精度可能越好，但元信息占比、GPU Kernel 访问复杂度也越高。计算 $X^\top X$ 的校准数据代表性决定二阶近似是不是当前部署分布的良好 Proxy。

## 低比特带来的是权重存储收益，速度还需要 Kernel

以参数量 $P$ 的 Decoder 为例，FP16 权重理想存储约 $2P$ 字节；4-bit 权重理想存储为 $P/2$ 字节，理论权重压缩 4 倍，但端到端显存还包括 KV/激活与量化元信息。GPU 在 FP16 激活乘 4-bit 压缩权重时可能需要先反量化到更高精度；实际速度取决于 Weight-Only GEMM/GEMV Kernel，而不是某篇论文的 PPL 表格。

[ICLR 2023 正式发表的论文与 arXiv v2](https://arxiv.org/pdf/2210.17323) 标题均为 **GPTQ**，而非“会议正式版改叫 OPTQ”；旧稿在此处混淆了名字。作者报告将约 175B 的权重在约四 GPU 小时内量化至 3–4 bit，结合专用 Kernel 在所测设备上实现端到端加速；不同硬件和 Weight Layout 的速度不能作为现代所有平台的默认承诺。

**反例意识**：若任务强依赖某些罕见输入方向，但 Calibration $X$ 几乎没有这些样本，$H$ 的估计就可能低估关键误差。对照 [AWQ](C042-awq.md) 时，要特别注意两者都用 Activation 信息，但 AWQ 以简单显著性/缩放规避量化敏感，不执行同样的二阶逐列误差补偿。
