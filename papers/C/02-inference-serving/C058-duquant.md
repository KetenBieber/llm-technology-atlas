# DuQuant：普通离群值能靠缩放处理，为什么极端离群值还需要两次旋转和 Zigzag 重排？

> **原始论文**：[DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9febda1c8344cc5f2d51713964864e93-Abstract-Conference.html)，Lin 等，NeurIPS 2024。[官方论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/9febda1c8344cc5f2d51713964864e93-Paper-Conference.pdf)；[可逐节检索的论文 HTML](https://arxiv.org/html/2406.01721)；[作者实现](https://github.com/Hsu1023/DuQuant)。本文采用 NeurIPS 正式发表的工作和 arXiv HTML 方法、数值表，所有实验读数均保留模型与量化条件。
>
> **路线**：先读 [SmoothQuant：幅度从激活迁移到权重](C040-smoothquant.md)、[RPTQ：通道范围分组](C046-rptq.md)、[QuaRot：随机 Hadamard 正交变换](C055-quarot.md)；本文给出另一条**利用异常通道位置先验构造局部旋转 + 全局排列**的路线。它与 [SpinQuant](C056-spinquant.md) 的 Cayley 优化不是同一种搜索策略。

## 看见一次 1400 的激活峰值，为什么就该怀疑整层 W4A4 量化？

Uniform INT4 在一组张量上只有 16 个整数码点。以非对称量化器为例，若组内最大最小值分别是 1400 和 -2，则间隔为

$$
\Delta=\frac{x_{\max}-x_{\min}}{2^4-1}
=\frac{1400-(-2)}{15}\approx93.47.
$$

在这组共用 Scale 的情况下，正常的 $-1,0,1,2$ 都挤在同一两个码点附近，甚至很可能被 Round 成同一个整数。真正糟糕的不只是有异常值，而是**低 bit 的量化格点被一个局部异常事件拉到了不适合其他数据的范围**。

[原文 Figure 1、论文第 2 节](https://arxiv.org/html/2406.01721) 将问题区分为两个现象：

1. **Normal Outliers（常规异常通道）**：某些 Channel 在大量 Token 上一直呈现比其余 Channel 大得多的幅值，是长期存在的 Channel 统计差异。
2. **Massive Outliers（极端异常值）**：通常集中在极少数 Token，绝对数值又远大于普通激活。作者在 LLaMA2-7B **第一个 Block 的 FFN Down-Projection 输入**中观察到近似 1400 的单点峰值，其他位置分布相对正常。

这两个问题不能只按“哪个值更大”处理。若只有长期偏大的 Channel，类似 SmoothQuant 的逐通道 Scale 可能足够；但当极端异常值只发生在一个 Token、一个通道时，按通道统计求出的 Scale 会被这个单点决定，其作用却横跨所有 Token。

### SmoothQuant 能否直接把 1400 缩小？可以，但代价会转移到权重

仍用行向量 $X:[T,d]$、$W:[d,m]$ 的约定。取对角矩阵 $\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_d)$：

$$
XW=(X\Lambda^{-1})(\Lambda W).
$$

若某通道通过 $\lambda_i=100$ 从幅值 1400 变成 14，原权重的对应输入行会被放大 100 倍。对 FP16 浮点层输出是严格等价的，但当 **$X\Lambda^{-1}$ 与 $\Lambda W$ 同时经过 INT4 量化**，一边的异常值可能只是被搬到了另一边。原论文 Figure 1(c)(d) 就专门展示 SmoothQuant 后激活与权重的新峰值。

于是 DuQuant 需要的是**同时改变异常值的坐标位置与分布，而不依赖把一个 Channel 的困难线性地交给 Weight**。

## 数学保证从哪里来：三步变换的逆必须严格成对

DuQuant 第 3.2 节 Eq.(5) 给出了全方法的骨架。仍按右乘约定，定义

$$
G=\Lambda^{-1} R_1 P R_2,\qquad
G^{-1}=R_2^\top P^\top R_1^\top\Lambda,
$$

其中 $R_1,R_2$ 是块对角正交矩阵，$P$ 是只排列 Channel 的置换矩阵，$\Lambda$ 是可逆逐通道缩放。因为 $R_i^\top R_i=I$、$PP^\top=I$，直接代入：

$$
(XG)(G^{-1}W)=
X\Lambda^{-1} R_1PR_2R_2^\top P^\top R_1^\top\Lambda W
=XW.
$$

**这个等式才是可信赖的“无损”部分。** 量化器插入后，$Q_4(XG)Q_4(G^{-1}W)$ 一般不再等于原输出。所谓 DuQuant 改善质量，是降低这种有损重建误差；**不是说通过旋转能严格消除所有 Quantization Error**。

对于 FFN 的 SwiGLU，须在 **Down-Projection 的线性输入之后**插入上述配对，不能任意让 $R_1,P,R_2$ 通过 $\mathrm{SiLU}$ 或 Hadamard Product，因为非线性通常不与一般旋转交换。这也是 [QuaRot](C055-quarot.md) 和本文实现上都必须区分可被离线吸收的等价变换与需要在线实际计算的变换的原因。

## 第一次旋转：为什么要先找到最异常的通道，再旋转局部 Block？

如果对 $d=4096$ 维 Activation 直接使用完整 $4096\times4096$ 矩阵，变换的参数和计算成本都是 $O(d^2)$；大量低 bit GEMM 节省的延迟可能重新花在变换中。因此作者限制为块对角矩阵：

$$
R_1=\operatorname{BlockDiag}(R_{1,1},\ldots,R_{1,K}),
\qquad R_{1,k}\in\mathbb R^{b\times b},\quad K=d/b,
$$

并选择 $b=2^n$ 这样的结构，配合分块旋转。对一个校准激活矩阵 $X_b:[T,b]$，最异常位置先由

$$
j^\star=\arg\max_{1\le j\le b}\max_{1\le t\le T}|X_{t,j}|
$$

确定。**这个下标不是要求“训练一个异常检测神经网络”**，而是先用已有校准统计找到哪一列最需要旋转。DuQuant 第 3.2 节 Eq.(2) 通过 Swap Matrix $E_{j^\star}$ 把该位置移到第一个轴，再乘一个其特定行均匀分布的正交矩阵 $\widetilde R$；最后用一个保持已处理轴的局部正交矩阵 $Q'$ 增加未处理轴的随机性。

作者的贪心过程不是把所有通道无脑应用同一个 Hadamard：每轮重新搜索剩余较大的异常通道，尝试更新旋转，对若干次候选取 **最大绝对激活最小的一步**。原文第 3.2 节 Eq.(2)、Appendix G Algorithm 1 可对应每个动作与停止条件。算法的性质是有针对性的异常值削减，不是所有正交矩阵都有该性质。

### 一个可计算的小例子

把一个四维异常输入设为 $x=[8,0,0,0]$，以 $H_4=\frac12\begin{bmatrix}1&1&1&1\\1&-1&1&-1\\1&1&-1&-1\\1&-1&-1&1\end{bmatrix}$ 右乘，得到

$$
xH_4=[4,4,4,4].
$$

原始最大坐标为 8，旋转后为 4，但 $\|x\|_2=\|xH_4\|_2=8$。等价地说：**能量没消失，而是从一个极大坐标转移到四个较小坐标**。这为 INT4 的共享 Scale 释放了动态范围。

这个玩具输入只证明集中向量在这一次旋转下降低了 $\ell_\infty$，不是 DuQuant 作者在 4096 维真实 LLaMA Activation 上的定量实验。若输入变成 $[8,8,8,8]$，同一个 $H_4$ 会把幅度汇集成一个 16，反而更坏；原论文为什么要用异常位置先验和贪心构造而不满足于“随机旋转”就有了数学上的理由。

## Zigzag 排列：为什么第一次局部旋转之后还需要跨 Block 再平衡？

块对角矩阵虽然便宜，但它无法跨 Block 调配峰值：假设四个相邻通道的异常幅度已经被旋转压低，另一组还包含特别大的 Channel，那么两个 Block 可能分别“过度容易量化”和“特别难量化”。[DuQuant Figure 2(a)–(c)](https://arxiv.org/html/2406.01721) 画的正是**旋转 → 全局 Zigzag 重排 → 再旋转**，而不是两次旋转相互重复做同一件事。

令每个 Channel 的校准最大幅度为 $O_j=\max_t|X_{tj}|$。对 $K$ 个大小为 $b$ 的 Block，定义组内均值

$$
M_k=\frac1b\sum_{j\in\mathcal B_k}O_j.
$$

排列优化的局部目标是减少这些均值的 Block 之间方差：

$$
V=\operatorname{Var}(M_1,\ldots,M_K).
$$

**Zigzag 不重新估计网络参数，它只是找通道索引排序。** 先按 $O_j$ 从大到小排列，把 Channel 依次分配到 Block 1、2、…、K；走到最后一个 Block 再倒序分配回 K、…、1，如此往复。第二次 Block Rotation 在已经更加均衡的各 Block 内重新分散峰值。

### 手算 Zigzag 到底改了什么

假设有 8 个 Channel 的异常幅度按从大到小为 $[100,80,60,40,30,20,10,0]$，需要放入两个大小为 4 的 Block。

普通连续分组：

- Block A：$[100,80,60,40]$，均值 $70$；
- Block B：$[30,20,10,0]$，均值 $15$。

Zigzag 前进—折返后，可以得到：

- Block A：$[100,40,30,0]$，均值 $42.5$；
- Block B：$[80,60,20,10]$，均值 $42.5$。

两组均值的方差由正值降为 0。但是注意：**最大值仍然是 100，并没有靠排列自动降到 42.5**。重排的任务是为第二次局部 Rotation 制造平衡输入，而不是直接把 MaxAbs 改成组均值。这是经常被忽略的一层因果关系。

[论文 Theorem 1/2、Eq.(6)–(7)](https://arxiv.org/html/2406.01721) 分别为作者构造下的局部旋转峰值、Zigzag 组间异常统计给出界。不能把定理误写成“所有可能分组、任意正交矩阵都能单调减小任意矩阵的所有异常值”。所研究的是作者指定构造与统计量。

~~~{mermaid}
flowchart TD
 A["校准 Activation X:[T,d]"] --> S["Smooth：Diagonal Scale Λ"]
 S --> R1["R1：找最异常通道，块内贪心旋转"]
 R1 --> Z["P：按幅度排序并 Zigzag 分配到各 Block"]
 Z --> R2["R2：均衡 Block 上再次局部旋转"]
 R2 --> Q["对 XG 与 G⁻¹W 分别进行低 bit Quant"]
 Q --> O["W4A4 的真正误差/任务质量"]
~~~

*自制算法机理图，非原论文 Figure；箭头描述等价变换的 Activation 支路，配对的 Weight 支路必须使用 $G^{-1}$ 才可保持浮点函数。*

## 三个真实执行成本：统计、重排与 Runtime

DuQuant 的一部分操作属于离线 Calibration：计算 $O_j$、贪心搜索、产生 $P$ 和块旋转。逐通道 Scale 可以吸收到 Weight 中；**Permutation 对 Weight 侧应使用逆排列，不能每次为每个 Token 做一次昂贵的全 Tensor Transpose 而不报告成本**。

Block Rotation 既可能采用离线权重吸收，也可能需要在 Activation 动态路径上发生。若一个 Block 维度为 $b$，对全部 $d$ 维 Activation 做局部变换的朴素乘加成本约 $K b^2=d\,b$；与完整密集旋转的 $d^2$ 相比约少 $d/b$ 倍。但这**不等于端到端服务延迟同比例下降**，因为 GEMM、访存、Kernel Launch 以及量化参数更新仍在。

作者 `3.2 Remark 2 还指出：可用包含最大异常值的 Block 构造出的单个旋转矩阵共享给其余 Block，减少旋转元信息以及离线搜索开销；这是一种额外结构化选择，而不是每个 Block 始终保持完全独立的任意矩阵。

在论文第 4 节 的主设置中，Activation 采用 Per-Token 量化、Weight 采用 Per-Channel 量化，并使用非对称 Uniform Quantization；DuQuant RTN 采用激活 0.9、权重 0.8 的 Clip Ratio，DuQuant+LWC 则额外结合 [OmniQuant](C047-omniquant.md) 的可学习 Weight Clipping。此处既有旋转，又有 Clip，不能将两版本得分差全部归因到第二次旋转。

## 用原论文 Table 1、5、6 验证“每一步到底改善了多少”

[原论文 Table 1（正式论文第 4.1 节）](https://arxiv.org/html/2406.01721) 比较 W4A4 在 WikiText-2 上的 PPL（越小越好）：

| 方法 | LLaMA2-7B | LLaMA2-13B | LLaMA2-70B |
|---|---:|---:|---:|
| FP16 | 5.47 | 4.88 | 3.31 |
| SmoothQuant | 83.12 | 35.88 | 26.01 |
| Atom | 8.40 | 6.96 | 未给可比数据 |
| DuQuant（RTN） | **6.28** | **5.42** | **3.79** |
| DuQuant + LWC | 6.08 | 5.33 | 3.76 |

说明：作者明确标注 LLaMA2-70B 上的 Atom、OmniQuant 未处理 GQA，因此其空项**不是零分，更不能作同一模型直接对比**。这份 PPL 结果证明在这些条件下同时平滑 Weight 与 Activation 可以保留更多语言建模质量；却不说明实际 CUDA INT4 GEMM Token/s 有几倍改善。

**Table 5 的 LLaMA3-8B W4A4** 证明换了模型家族之后仍困难：WikiText-2 FP16 PPL 为 6.14；Atom 为 22.14；DuQuant 为 8.56；DuQuant+LWC 为 8.06。即便 DuQuant 有较明显改善，6.14→8.06 仍是非零质量损失。任务 Zero-Shot Avg 从 FP16 74.22 降至 DuQuant 66.21 / DuQuant+LWC 67.72，不能用 PPL 比率直接替换 QA 分数。

更具因果力的是 **Table 6（消融）**。固定 LLaMA2-7B W4A4，作者报告：

| 开启的组件 | WikiText-2 PPL |
|---|---:|
| Smooth 单独 | 原表为 NaN（失稳，不能视作数值零） |
| Rotation 1 单独 | 8.48 |
| Smooth + Rotation 1 | 7.92 |
| Rotation 1 + P + Rotation 2（无 Smooth） | 6.79 |
| Smooth + Rotation 1 + P + Rotation 2 | **6.28** |

它能支持“第一旋转以及 Zigzag+第二旋转逐步改善”的解释；但两行组合并非仅改变一个变量时，不应声称从 8.48→6.28 的全部收益都来自某一组件。

### Figure 2 不是速度基准，真实 Kernel 仍另需确认

[Figure 2](https://arxiv.org/html/2406.01721) 的三段热图用于展示不同重排/旋转阶段激活极值如何分散，不能据此推出 GPU Latency 下降固定倍率。DuQuant 的主要评估焦点是 W4A4 的 PPL、常识问答、MMLU 和 LongBench；如果缺少与真实 INT Kernel 相同环境下的端到端性能，应该把**精度改善**与**服务吞吐证明**分开。

最终可按具体工程目标选择：[QuaRot](C055-quarot.md) 侧重随机正交与真实 W4A4KV4 路线；本篇 DuQuant 用异常位置先验与 Zigzag 显式改善 W4A4 的输入统计；[QServe](C059-qserve.md) 则转向同等低 bit 思路在批量云端部署里遭遇的 CUDA Core 开销问题，解释“算法低 bit”为什么不是“GPU 天然更快”。
