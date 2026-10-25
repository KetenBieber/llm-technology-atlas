# LLM.int8：为什么少量异常特征要走 FP16 路径？

> **原始论文**：[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)，NeurIPS 2022（会议条目亦称 GPT3.int8）。 [官方/作者 PDF](https://arxiv.org/pdf/2208.07339)。


前置：[Transformer 线性层](../../A/03-transformer/A013-transformer-attention-is-all-you-need.md)，后续：[SmoothQuant](C040-smoothquant.md) → [Atom](C044-atom.md)。

## 为什么 CNN 上奏效的 INT8，遇到超大 Transformer 就突然变难？

在 Transformer 中，线性层 $Y=XW$ 的输入激活 $X$ 随着模型规模增大出现少量、但数值非常极端的特征维度（outlier features）。对称线性量化把浮点值映射到有限整数刻度：

$$
Q(x)=\operatorname{clip}\left(\operatorname{round}\frac{x}{s},-127,127\right),\qquad
s=\frac{\max|x|}{127}.
$$

若一组数据绝大部分在 $[-1,1]$，但有一个异常激活为 $80$，量化步长为 $80/127\approx0.63$，普通值只能表示在少数几个整数格点上。**最大值决定了整组值的有效分辨率**；不是简单地再多训几轮校准数据就一定能解决。

在 Decoder 的某些层中，这些异常维度对输出尤其敏感。统一低比特导致的噪声可在深层堆叠中积累，损害原模型的下一个 Token 分布。因此“模型 99.9% 的值都普通”并不代表可以无差别全部 INT8。

## 二路分解：不是整个层一半 FP16、一半 INT8

设 $X\in\mathbb R^{T\times d_{\rm in}}$，$W\in\mathbb R^{d_{\rm in}\times d_{\rm out}}$。对输入通道维度建立异常集合 $O$，正常集合 $N$，有精确的代数恒等式

$$
XW=X_{:,N}W_{N,:}+X_{:,O}W_{O,:}.
$$

这是沿矩阵乘法“求和维度”分拆，因而两项相加得到相同结果。LLM.int8 的选择是：
- 对占绝大多数的正常通道 $N$，使用 vector-wise INT8 量化矩阵乘法；
- 对数量极少且敏感的 $O$，保留 FP16 计算；
- 将两个路径的输出加起来。

~~~{mermaid}
flowchart TD
 X[浮点激活 X] --> A[按异常阈值区分通道]
 A --> N[普通通道 N：向量级 INT8]
 A --> O[异常通道 O：FP16]
 W[权重 W] --> N
 W --> O
 N --> S[两路结果累加]
 O --> S
~~~

*自制解释图，非原论文 Figure。应结合作者论文关于 Emergent Outlier Features 和 mixed-precision decomposition 的主图理解。*

注意这里的**精确等式属于未量化的代数分解**；量化之后 $Q(X_N)Q(W_N)$ 与 $X_NW_N$ 仍有误差。所谓“无精度损失”是论文实验中报告的模型性能基本保持，而不是数学上量化后每个输出元素与 FP16 严格相等。

## 为什么需要 Vector-Wise，而不是全 Tensor 共用一个 Scale？

一个矩阵按整张 Tensor 缩放，所有 Token 与通道共享量化刻度；只要局部出现离群值，整个矩阵的有效动态范围都被挤占。Vector-Wise Quantization 为矩阵乘法的不同向量指定更细粒度的 Scale，使多数非异常通道获得更准确的表示。

例如 $X\in\mathbb R^{T\times d}$ 的每个 Token 行设一个激活 Scale，$W$ 相应输出列设 Scale，INT8 GEMM 后对浮点输出做反量化。这种沿外部维度的 Scaling 可以相对高效地与 GEMM 配合；若沿求和维度细分，Scale 应用位置会更复杂，也可能妨碍标准高效整数 Kernel。

## 这与 SmoothQuant 看起来相反，为什么都成立？

LLM.int8 选择**承认异常值不可忽略**，将它们分配到 FP16 通路；[SmoothQuant](C040-smoothquant.md) 则通过数学等价的通道重缩放，将异常激活的量化难度转移给更耐量化的权重，努力消除混精度小矩阵路径。两者不是“一个准确一个不准确”，而是面对异常分布与硬件 Kernel 的不同工程方案。

对同一个模型比较它们时，至少保持 bit-width、精度误差度量、硬件和 Batch 相同。带有 FP16 Outlier 支路的推理实际加速取决于数据分派、两次 GEMM 的大小与二次融合开销；**显存减半不意味着每次 Forward 必然快两倍**。

## 结果、版本与边界

[作者版本](https://arxiv.org/abs/2208.07339) 的标题是 *LLM.int8()*，NeurIPS 2022 正式会议条目也使用 *GPT3.int8()* 作为题名。作者报告把 OPT/BLOOM 等模型推到 175B 参数级，正常路径绝大多数元素使用 INT8 而异常部分保留 FP16，整体模型在原测试任务上基本不降精度；混精度处理不是零开销。需要分开报告任务指标、模型规模、储存降低和真正的 Token/s，不能从“175B 可以部署”直接推出更高并发服务能力。

研究延伸：为什么 Scale 细粒度从数学上改善误差？为什么异常通道数量与模型规模相关？若异常分布随输入域变化，固定阈值如何退化？[SmoothQuant](C040-smoothquant.md) 给出另一个严谨可推导的解法。
