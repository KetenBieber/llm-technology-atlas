# SmoothQuant：把激活异常值转移到权重，为什么数学上完全等价？

> **原始论文**：[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html)，ICML 2023。 [官方/作者 PDF](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf)。


建议先读 [LLM.int8 的异常值双路径](C039-llm-int8.md)；后续对照 [GPTQ](C041-gptq.md)、[AWQ](C042-awq.md) 与 [Atom](C044-atom.md)。

## 问题不是参数的动态范围，而是 INT8 激活范围被个别通道拉爆了

考虑一个 Transformer 线性层

$$
Y=XW,\quad X\in\mathbb R^{T\times d_{\rm in}},\quad
W\in\mathbb R^{d_{\rm in}\times d_{\rm out}}.
$$

浮点输入 $X$ 某个通道可能长期出现远超其他通道的值。如果为整个张量定一个 8-bit Scale，少量异常值将扩大 $s=\max|X|/127$，大部分普通值只能落在少数整数级上。单纯裁剪异常值会丢失可能影响下一个 Token 的信号；[LLM.int8](C039-llm-int8.md) 选择对这些异常通道额外运行 FP16 分支，但产生混精度分支处理开销。

例如某次校准激活中极端值为 $100$，大部分非零值只有 $0.1$，若直接用对称 INT8 的统一步长 $\Delta=100/127\approx0.7874$，则 $0.1/\Delta\approx0.127$，Round 后变成 0。注意“量化之后的 0”不是浮点模型的判断，而是**少数异常值扩大公共量化区间**造成的大量小激活被抹掉。Per-Token Quantization 能处理不同 Token 的动态范围，却不能自动消除同一 Token 不同通道之间极不均匀的幅度。

SmoothQuant 的第一性观察是：**激活难量化，权重相对容易量化；能否在完全不改变浮点层输出的条件下调换二者的量级？**

## 只在共享的“输入通道”上乘除相反尺度

选取正数 $s_j>0$，构造对角矩阵 $D=\operatorname{diag}(s_1,\ldots,s_{d_{\rm in}})$。于是

$$
Y=XW=(XD^{-1})(DW)
=\underbrace{\widehat X}_{X D^{-1}}
\underbrace{\widehat W}_{D W}.
$$

这个等式对任何可逆 $D$ 都严格成立，因为中间的 $D^{-1}D=I$。重点是 $D$ 与 $D^{-1}$ 必须放在矩阵乘法**正确的一侧**：$X$ 的第 $j$ 个输入通道除以 $s_j$，$W$ 的第 $j$ 行乘以 $s_j$。如果把 $D$ 放在 $W$ 右侧，除非存在额外特殊结构，通常不是同一函数。

原论文提出基于离线校准统计选择

$$
s_j=\frac{\left(\max_{t}|X_{t,j}|\right)^\alpha}
{\left(\max_{k}|W_{j,k}|\right)^{1-\alpha}},
\qquad \alpha\in[0,1].
$$

$\alpha$ 是控制“把多少动态范围从激活转移到权重”的超参数；当 $\alpha$ 较大，激活更平滑、权重可能更难量化。当 $\alpha$ 较小，权重变化更保守、激活中的离群更大。注意 $\max_t$ 是离线校准样本的统计，不代表真实未来所有 Token 的严格上界。

### 一个能够手算的两通道例子

设 $X=[100,\;1]$，$W=\begin{bmatrix}0.01\\1\end{bmatrix}$，那么

$$
XW=100(0.01)+1(1)=2.
$$

如果取 $D=\operatorname{diag}(100,1)$，则

$$
\widehat X=[1,1],\qquad
\widehat W=\begin{bmatrix}1\\1\end{bmatrix},\qquad
\widehat X\widehat W=2.
$$

分布由激活侧被平滑到了权重侧。此例专门展示浮点恒等式，不意味着 8-bit 量化后误差永远为零：量化后的 $\widehat X_q,\widehat W_q$ 仍带 Round/Clip 误差；真正目标是**两个张量都更适合 W8A8**。

~~~{mermaid}
flowchart LR
 X[原始 X：激活异常值] --> M[输入通道除以 s_j]
 W[原始 W：量化容易] --> N[对应权重行乘以 s_j]
 M --> A[Smoothed X：W8A8 激活]
 N --> B[Smoothed W：W8A8 权重]
 A --> G[INT8 GEMM]
 B --> G
~~~

*自制重参数化图，非论文 Figure。请对照 [SmoothQuant 正式论文 Figure 2（PDF 第 2 页）](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf)：上半显示原始激活 $X$ 的 Outlier 拉大量化范围；下半展示 $\widehat X$ 平滑以后将动态范围压力迁移到 $\widehat W$。原图是机制示意，纵轴“quant. levels”与激活相对幅度说明量化有效级数，并非模型 Benchmark Accuracy。*

## 用原公式算出 Migration Strength 的真正作用

取通道校准激活最大值 $a=100$，该通道权重最大值 $w=0.01$。原论文按

$$
s=\frac{a^\alpha}{w^{1-\alpha}}
$$

定义迁移尺度。若 $\alpha=0$，则 $s=w^{-1}=100$；若 $\alpha=1/2$，得到 $s=\sqrt{a/w}=100$；若 $\alpha=1$，则 $s=a=100$。这个特殊例子**三个取值巧合相同**，根本不适合直觉上看 $\alpha$ 的调节效果。换成 $a=100,w=1$：

$$
s(0)=1,\qquad s(1/2)=10,\qquad s(1)=100.
$$

于是变换后激活最大幅度为 $a/s$，权重最大幅度为 $w\,s$：

| $\alpha$ | 迁移尺度 $s$ | 新激活最大值 | 新权重最大值 |
|---|---:|---:|---:|
| 0 | 1 | 100 | 1 |
| 0.5 | 10 | 10 | 10 |
| 1 | 100 | 1 | 100 |

表格是由原公式手算的单通道教学例子，不表示实际最佳 $\alpha=0.5$。它说明随着 $\alpha$ 增加，激活更容易量化，而权重可能变难量化。真实的误差还取决于其他通道分布、Group Scale 与非线性层，所以最佳迁移程度不一定是两者 Max 完全相等。

## 先分清 Figure 1、Figure 2、Figure 6 分别在证明什么

[ICML 2023 正式论文](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf) 的 **Figure 1 是模型参数规模增长与 GPU 显存容量增长之间的差距图**，并不是异常激活分布；真正的异常值机制图是 **Figure 2**。这处图号差异会直接影响论文证据链：Figure 1 只能解释为什么需要压缩模型，不会独立证明 SmoothQuant 正确。

Figure 6（PDF 第 5 页）进一步把 INT8/FP16 标在 Transformer Block 的具体算子上：Linear 的 GEMM 和 Attention 中的 BMM 可以采用 INT8，而 LayerNorm、Softmax、Residual 等轻量或敏感操作仍保留 FP16。它说明论文的 **W8A8 是计算密集型矩阵乘的权重/激活精度策略**，不是“所有中间算子全部纯 INT8”。

即便 Figure 2 说明量化输入的分布更均匀，也尚不能证明推理一定更快。速度需要结合 INT8 Tensor Core、Kernel Layout、Batch 和 End-to-end Wall Clock 的实验。

一个关键工程反例：若模型只在 Decode 的 Batch=1 下执行 GEMV，权重读取受 HBM 限制，离线平滑可能让 Weight 更好表示但不一定使 INT8 运算获得论文大 Batch GEMM 的加速。**相同 W8A8 的位宽对不同 Shape 会有不同的 Kernel 价值**，因此必须至少分开测 Prefill 和 Decode。

## 什么时候能把 Runtime 开销提前挪到 Offline？

为了运行时不需要对每个 Activation 额外执行高成本通道缩放，可以把 $D^{-1}$ 与前一层可融合的归一化参数或 Linear Bias 等结构合并；把 $DW$ 离线预处理成量化友好权重。这里依赖模型 LayerNorm/RMSNorm 与线性层的位置和实现，**不是任意两层都能无代价融合**。

量化粒度同样关键：per-token activation、per-channel weight 能与高效 INT8 GEMM 的输出 rescale 协调，而不是随意在求和维度上设置 Scale 后仍预期获得同样 Kernel 性能。

### 为什么原论文一定要区分 O1 / O2 / O3？

原文 [Table 2（PDF 第 5 页）](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf) 并没有把所有 SmoothQuant 实现都写成单一的“静态 W8A8”。在该表作者所列的三个效率等级中，量化方案实际为：

| 方案 | Weight INT8 Scale | Activation INT8 Scale | 工程取舍 |
|---|---|---|---|
| SmoothQuant-O1 | Per-Tensor | Per-Token Dynamic | 对每个 Token 动态观测幅度，开销较高、精度通常较稳 |
| SmoothQuant-O2 | Per-Tensor | Per-Tensor Dynamic | 一次动态幅度统计覆盖更大张量，运行更直接 |
| SmoothQuant-O3 | Per-Tensor | Per-Tensor Static | 离线定步长，减少运行时统计和缩放工作 |

三者的区别不是重新训练了三个浮点模型，而是**对相同平滑思想采用不同量化粒度/运行时统计方式**。Table 2 标的是实现设置；它本身不报告每种方案的实际 Accuracy。离线静态量化在部署输入明显偏离 Calibration 分布时可能失真，因而 O3 的最低开销也不应被解释为对任何模型都无精度损失。

真正需要读数时，接着看 [Table 3（PDF 第 6 页）](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf)：论文对 OPT-175B 用多项 Zero-Shot Accuracy 和 WikiText Perplexity 联合比较 FP16、朴素 W8A8、ZeroQuant、LLM.int8 与 SmoothQuant 的不同设置。它能验证指定评测条件下的**量化后任务保真**，而不是证明某个 $\alpha$ 在所有模型、所有数据分布上都最优。

## 原论文结果应该如何引用？

[ICML 2023 正式论文](https://proceedings.mlr.press/v202/xiao23c.html) 报告多种 LLM（OPT、BLOOM、GLM、LLaMA）W8A8 部署，最高约 **1.56× 推理加速**与**2× 内存降低**，实验条件为作者所用的 FasterTransformer 等基线。Figure 2 的作用是解释重新分配离群范围的机制，不是独立的运行时间证据；必须配合量化误差/任务准确率和 Hardware Kernel 的测量。

实践中要分别比较 $XD^{-1}DW$ 的浮点等价误差、量化后 $Q(XD^{-1})Q(DW)$ 的输出误差、任务 perplexity 和 GPU Tensor Core 的实际时间。若以等价重参数化推导出“量化无损”，就是把前后两个不同命题混为一谈。

下一问题：SmoothQuant 重点让**权重和激活**都适应 W8A8；若核心约束只是在低显存设备里把**权重**压到 3–4 bit，是否可以利用 Hessian 近似做更小的误差补偿？进入 [GPTQ](C041-gptq.md)。
