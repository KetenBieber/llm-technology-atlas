# SpQR：稀疏高精度异常权重为什么比全局提高位宽更划算？

> **原始论文**：[SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1787533e171dcc8549cc2eb5a4840eec-Abstract-Conference.html)，ICLR 2024。


前置：[GPTQ 的二阶误差补偿](C041-gptq.md)、[LLM.int8 的异常通道分流](C039-llm-int8.md)。下一篇：[QuIP# 的正交变换与码本](C049-quip-sharp.md)。

## 为什么“只有极少数值难以量化”能够改变存储设计？

如果一个 Weight Matrix 有 100 万个标量，仅 0.1% 造成显著量化重建误差，那么把所有数从 3-bit 升级到 4-bit，新增约 100 万 bit；而让这 1000 个特别敏感的元素采用高精度保存，其原始额外值数据近似只有 $1000\times(16-3)$ bit，但还要存索引与附加元信息。

因此真正的比较是

$$
M_{\mathrm{SpQR}}=
M_{\mathrm{dense\ lowbit}}+
M_{\mathrm{outlier\ values}}+
M_{\mathrm{indices}}+
M_{\mathrm{group\ metadata}}.
$$

*如果异常项足够稀少*，这种表示既可以保留主要低比特权重压缩收益，又能显著抑制少数高损失权重的误差。这解释了论文名字中的 **Sparse-Quantized Representation**：不是把整个模型剪枝成 Sparse Transformer，而是把**稀疏例外放进高精度支路、主体保持低比特密集存储**。

## 异常值为什么不只看绝对大小？

假设某个权重 $w_{ij}$ 量化残差 $e_{ij}=w_{ij}-\hat w_{ij}$ 很大，但对应输入通道很少被激活，它对 $XW$ 的损失可能仍较小。反之，绝对值不大的权重如果其错误发生在输入高敏感方向，输出偏差可能更大。SpQR 研究**导致大重建误差的量化离群项**并存储为例外，不能将它简化成“把最大的那 1% 权重 FP16 保存”。

给定 $W$ 和输入 $X$，如果量化得到 $\hat W$，输出误差近似

$$
\mathcal E=\|X(W-\hat W)\|_F^2.
$$

SpQR 的选择机制要考虑量化误差、校准信息和高精度例外的成本。若将所选位置集合写作 $\Omega$，可概念性表示

$$
W\approx Q_b(W_{\overline\Omega})+R_\Omega,
$$

其中 $R_\Omega$ 是仅在少数位置非零的高精度残差；这只表达**数据结构**，不是论文整个量化求解程序。所有例外都放回原坐标后，Linear 输出可以被实现为低比特密集矩阵运算加少量稀疏贡献。

~~~{mermaid}
flowchart TD
 W[FP16 Weight] --> E[估计量化后的敏感例外]
 E --> D[多数权重：分组低比特]
 E --> S[少数 Outlier：高精度稀疏例外]
 D --> L[高效低比特权重读取]
 S --> R[稀疏纠错访问]
 L --> Y[Linear 输出合并]
 R --> Y
~~~

*自制机制图，非原论文原图。需同时读论文存储格式的元信息编码与内核对两种表示的融合方式。*

## 性能为什么要同时考虑压缩比与例外查找？

低 bit 存储节省的 HBM 带宽，可能被稀疏索引随机访存、分支和解包破坏。如果每个 Outlier 都使 GPU Thread 需要单独跳转地址，局部 Memory Coalescing 会很差，甚至比纯 FP16 Kernel 慢。作者因此不仅提出量化表示，还提供能有效解码与推理的 GPU 方案；如果只复现量化权重的 PPL，不能声称复现了整个 SpQR 系统。

[ICLR 2024 正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1787533e171dcc8549cc2eb5a4840eec-Abstract-Conference.html) 报告在 LLaMA/Falcon 等模型 3–4 bit 设置中取得低于约 1% 的相对 PPL 损失，展示 33B 模型在 24GB 消费级 GPU 上运行的案例，并报告特定配置相比 16-bit 基线约 15% 加速。这是指定模型/硬件的实证，不是**全部 3-bit 模型普遍无损且加速**的证明。

**研究问题**：在相同有效 bits/weight（包括索引和 Scale）下，和 [GPTQ](C041-gptq.md) 或 [AWQ](C042-awq.md) 比差多少？若异常项比例从 0.1% 升到 10%，稀疏格式是否仍有意义？[QuIP#](C049-quip-sharp.md) 则尝试通过随机正交变换把误差与异常值在坐标轴上摊平，而非额外存储例外。
