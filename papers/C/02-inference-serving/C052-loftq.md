# LoftQ：既然 LoRA 要训练，为什么不让初始化主动补偿量化误差？

> **原始论文**：[LoftQ: LoRA-Fine-Tuning-aware Quantization for Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/39ec972afab01e0d8ddc6834a9d12ac1-Abstract-Conference.html)，ICLR 2024。


前置：[QLoRA 的冻结主干与低秩残差](C051-qlora.md) / [GPTQ](C041-gptq.md)。后续：[BitNet 的量化感知预训练](C053-bitnet.md)。

## 把主干压成 4-bit 后，再随机初始化 LoRA，为什么有点浪费？

假设原模型参数为 $W$，量化后权重为 $\hat W=Q(W)$。那么初次前向就存在一个固定差值

$$
E=W-\hat W.
$$

常见的 Adapter 初始化让附加修正项 $BA$ 在开始时近似为 0，以免破坏训练起点；但这里已知主干量化必然产生 $E$。如果能直接让 $BA$ 近似抵消它，下游训练就不需要把最初几个优化阶段完全用于修复主干压缩损失。

LoftQ 将**主干量化**与**LoRA 初始化**绑定求解，而不是量化完成后才随便初始化低秩矩阵。

## 由 QLoRA 的零初始化推到 LoftQ：为什么有一步本来就能提前完成？

以相同的右乘约定，原始线性层 $Y=XW$，主干量化和 LoRA 后的层输出为

$$
\widehat Y=X(Q+AB).
$$

令固定量化误差 $E=W-Q$，则

$$
Y-\widehat Y=X(E-AB).
$$

如果希望 Adapter 一开始就缩小这个差异，最朴素的目标是 $\|E-AB\|_F^2$；比起 Adapter 在训练第一步为零，这至少让它先解释一部分已知的压缩误差。

### 从 SVD 精确推导最佳 Rank-r 初始化

设 $E=U\Sigma V^\top$，奇异值 $\sigma_1\ge\sigma_2\ge\cdots\ge0$。Eckart–Young–Mirsky 定理给出最佳 Rank-$r$ 的 Frobenius 近似为

$$
E_r=U_r\Sigma_rV_r^\top.
$$

取

$$
A=U_r\Sigma_r^{1/2},\qquad
B=\Sigma_r^{1/2}V_r^\top,
$$

就有 $AB=E_r$。因而初始恢复不可能超过的残余为

$$
\min_{\operatorname{rank}(R)\le r}\|E-R\|_F^2=\sum_{i>r}\sigma_i^2.
$$

一个二维手算示例：假设 $E=\operatorname{diag}(0.3,0.1)$，Rank-$1$ 可以令 $A=[\sqrt{0.3},0]^\top$、$B=[\sqrt{0.3},0]$；这样重建误差从未经补偿的 $0.3^2+0.1^2=0.10$ 降至 $0.1^2=0.01$。但如果主要量化误差分散在所有高 Rank 方向，仅用一个低秩 Adapter 很难全部抵消。这说明 Rank 并不只是减少训练参数，它限制了可以修复的量化误差空间。

## 从矩阵分解得到一个直观的目标

对一个 Linear 层 $y=XW$，LoRA 增量为 $BA$，可考虑最小化

$$
\min_{Q,A,B}
\|W-Q(W)-BA\|_F^2.
$$

这里按右乘表达式 $BA\in\mathbb R^{d_{\rm in}\times d_{\rm out}}$。这是解释性联合重建目标；论文在实现中采用交替式量化与低秩近似更新，不是把离散整数编码随便当连续变量求梯度。

如果量化结果 $\hat W$ 暂时固定，那么经典 Eckart–Young–Mirsky 最佳低秩近似告诉我们，对 $E=U\Sigma V^\top$ 的奇异值分解，截取前 $r$ 项可使

$$
E_r=U_r\Sigma_r V_r^\top
$$

成为 Frobenius 范数下的最优 Rank-$r$ 近似。可令 $B=U_r\Sigma_r^{1/2}$、$A=\Sigma_r^{1/2}V_r^\top$，获得能直接抵消部分量化误差的适配器初始化。真正的 LoftQ 还需要让后续量化选择与 Adapter 共同迭代改善。

## 一个无法忽略的 Rank 限制

即使知道完整误差矩阵 $E$，若 Rank 很高但训练允许 $r=8$，LoRA 也无法完全抵消。其理论不可消除的 Frobenius 残差为

$$
\min_{\operatorname{rank}(R)\le r}\|E-R\|_F^2
=\sum_{i>r}\sigma_i(E)^2.
$$

这说明 **量化误差的奇异值谱**决定低秩补偿能达到的上限。如果极低 bit 量化引入高秩、分散的误差，增加 Rank 也许有帮助，但会消耗显存、优化器状态和计算量；并非任何量化错误都能靠低秩补偿解决。

~~~{mermaid}
flowchart TD
 W[原始 FP16 Weight] --> Q[候选低比特量化 Q]
 Q --> E[误差 E = W - Q]
 W --> E
 E --> S[截断 SVD / Rank-r 近似]
 S --> L[初始化 LoRA A 与 B]
 L --> Y[量化主干 + Adapter]
 Y --> U[下游 Fine-Tuning]
~~~

*自制初始化与训练路径图，非论文 Figure。原文在量化与残差低秩近似之间交替执行以改善总重建误差。*

## 为什么 LoftQ 不是只对量化残差做一次 SVD？

原论文 [ICLR 2024 `3、Eq. (6)–(9) 与 Algorithm 1](https://proceedings.iclr.cc/paper_files/paper/2024/file/39ec972afab01e0d8ddc6834a9d12ac1-Paper-Conference.pdf) 把量化权重 $Q$ 与低秩 $AB$ **交替更新**：

1. 以当前低秩修正 $AB$ 为已知，目标相当于压缩 $W-AB$，重新求一个可存储的低比特 $Q$。
2. 固定 $Q$，对残差 $W-Q$ 做 SVD，取 Rank-$r$ 近似更新 $A,B$。
3. 重复若干轮，将新得到的 $Q,A,B$ 作为 LoRA Fine-Tuning 的起点。

这是一种交替近似，不是说离散 $Q$ 的量化算子可以被简单求导求得全局联合最优。论文明确说明 $T=1$ 是只量化一次主干再用 SVD 初始化的特例；继续交替往往可以获得更小的预训练权重重建误差，但要付出额外离线计算。

**为什么这么做比只有 SVD 更有意义？** 如果固定量化器一次舍入在方向 $u_1$ 上产生很大误差，SVD 用 Adapter 消除它；下一轮量化器就可以把一部分剩余误差重新分配到 Adapter 容量更适合解释的低秩方向。这是两个相互耦合的设计变量，而不是相互独立的一次优化。

## 权重重建并不等于所有输入都重建得同样好

真正的层输出误差是

$$
\mathcal L_{\rm out}
=\mathbb E_X\|X(W-Q-AB)\|_F^2
=\operatorname{tr}\left((W-Q-AB)^\top H(W-Q-AB)\right),
$$

其中 $H=\mathbb E[X^\top X]$。LoftQ 原始的 Frobenius 权重残差近似，相当于以 $H\propto I$ 作为一种理想情形。实际部署输入若高度集中在某些特征方向，**最小 Frobenius Weight Error** 不一定等于最小语言建模 Loss。这解释了后续任务 Fine-Tuning 为什么仍不可省略。

论文的训练前第 0 步指标与下游微调后指标需单独读。前者体现量化+LoRA 初始化是否重建原模型，后者才反映最终任务能力。尤其不同 Adapter Rank、量化位宽、SVD 初始化迭代数同时改变时，不能用一张下游总表把收益全部归因于“采用了 QLoRA”。

可以用一个最小复现实验验证定理：在一个 $8\times8$ 矩阵上人为构造残差，计算不同 $r$ 的奇异值尾部平方和，与 `numpy.linalg.svd` 截断后的 Frobenius 残差对照；然后针对同一原始权重比较“只量化”“量化+一次 SVD”“量化和 SVD 交替”的误差曲线。这个小实验只需普通 CPU 线性代数，不需要运行大型 LLM。

## 为什么不能仅通过训练后的下游指标评价“初始化更好”？

把最终精度提高全部归因于 LoRA 初始化会混入训练步数、学习率、不同量化器、Rank 和校准语料差异。更严格的实验应该在训练**第 0 步**记录：

1. 原 FP16 层与 QLoRA、LoftQ 的输出重建误差；
2. 初始 Adapter 的有效 Rank 和参数量；
3. 相同数据和步数下 Fine-Tuning 后表现。

[ICLR 2024 正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/39ec972afab01e0d8ddc6834a9d12ac1-Abstract-Conference.html) 特别关注 2-bit、混合 2/4-bit 配置中量化与适配器一起优化的意义。其收益不是“LoRA 从此不需要训练”，而是在压缩主干与微调目标之间构建更合理的初始状态。

## 与 BitNet 的关系：后训练修补与预训练约束

LoftQ、QLoRA 都从已经训练好的 FP16 权重出发，想在低显存下微调或部署。对于从头训练新模型的情况，可以在整个训练过程中就强制权重接近低比特集合，参数最优点可能与事后 Round 一个 FP16 结果不同。这个问题由 [BitNet](C053-bitnet.md) 进入。
