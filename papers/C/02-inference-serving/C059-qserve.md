# QServe：INT4 的理论算力更高，为什么实际批量推理会慢于 W8A8？

> **原论文**：[QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://proceedings.mlsys.org/paper_files/paper/2025/hash/fbe2b2f74a2ece8070d8fb073717bda6-Abstract-Conference.html)，Lin 等，MLSys 2025。[官方正式 PDF](https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf)、[MIT HAN Lab 项目](https://hanlab.mit.edu/projects/qserve)、[作者代码](https://github.com/mit-han-lab/omniserve)。图表和页码依据 MLSys 正式版。
>
> **依赖路线**：[Atom 的 W4A4 服务](C044-atom.md) 与 [QuaRot 的 W4A4KV4](C055-quarot.md) → [DuQuant 的低 bit 精度](C058-duquant.md) → **本篇：QoQ 精度方案 W4A8KV4 + QServe GPU 系统**。前置阅读还有 [LLM.int8()](C039-llm-int8.md)、[SmoothQuant](C040-smoothquant.md)、[PagedAttention](../01-distributed-training/C012-vllm-pagedattention.md)。

## 理论峰值为什么骗过了我们？

相同 A100 GPU 上，INT4 Tensor Core 理论峰值计算吞吐可以高于 INT8，于是直觉认为 W4A4 会比 W8A8 快；但实际执行的对象不是一条 Tensor Core 指令，而是整套 GEMM 的 Decode、Group Scale、内存读取、寄存器转换和写回。

[论文 Figure 2(b)（PDF 第 3 页）](https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf) 的 A100 LLaMA2-7B 批量服务对照：

| 实际系统 | Token/s |
|---|---:|
| TensorRT-LLM FP16 | 1468 |
| TensorRT-LLM W4A16 | 1474 |
| TensorRT-LLM W8A8 | **2104** |
| Atom W4A4 | 986 |
| QuaRot W4A4 | 817 |

可见**标称计算 bit 更小**与**真实 Token/s 更高**是不同命题；不能把后两者慢全归因于“4 bit 量化本身无价值”，也不能把前三个 TensorRT 的速度当成独立 Kernel 的绝对算力。

Figure 2(a) 进一步表明 Batch 增长后 Attention 与 GEMM 都在争夺执行时间。作者 §2 在 1024 Prompt、512 Output 的代表性运行下观察到 Decode 耗时显著更高。因而 QServe 要同时研究 Weight–Activation GEMM 与长期访问的 KV Cache。

## Batch 为什么决定该优先省 Weight 带宽，还是提高 Tensor Core 利用率？

对 $X\in\mathbb R^{m\times k}$、$W\in\mathbb R^{k\times n}$，一次 GEMM 约有 $mnk$ 次 MAC。如果 $n,k$ 远大于 $m$，Weight 的主导读取量为 $kn$，因此粗略的 Weight-only 算术强度是

$$
\mathcal I_{\mathrm{MAC/weight\ element}}\approx\frac{mnk}{kn}=m.
$$

Decode 时 $m$ 近似 Batch Size。**低 Batch** 时经常受 HBM Weight 读取制约，此时 W4A16 可能优于 W8A8；**较高 Batch** 时 Weight 被多个 Token 重用，GEMM 趋向算力瓶颈，INT8 Tensor Core 的高吞吐更重要。

[原论文 Figure 3 的 A100 Roofline](https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf) 在其带宽和峰值算力假设下指出 W4A16 与 W8A8 的理论拐点约在 $m=78$ 左右。**78 不是 CUDA API 的批量限制**，换 GPU、矩阵 Shape、Kernel 和外部 Memory Pipeline 后都会变化。论文提出 W4A8 作为两者折中：4-bit Weight 少读内存，8-bit Activation 仍可进入原生 INT8 Tensor Core。

但仍有一个前提：如果每次矩阵乘都要在 $k$ 轴内反复做 FP32 Group Dequant，理论 Roofline 完全无法兑现。

## 为什么沿 Reduction 维度的 Scale 会把整数 GEMM 拆碎？

将长度为 $k$ 的内积按 Group $g$ 拆开。一般的 Group Quantization 在同一 Group 使用

$$
x_i\simeq s_{x,g}(q_{x,i}-z_{x,g}),\qquad
w_i\simeq s_{w,g}(q_{w,i}-z_{w,g}).
$$

那么

$$
x^\top w\simeq
\sum_g s_{x,g}s_{w,g}
\underbrace{\sum_{i\in\mathcal I_g}(q_{x,i}-z_{x,g})(q_{w,i}-z_{w,g})}_{\text{INT32 partial sum}}.
$$

因为 $s_{x,g}s_{w,g}$ 随 $g$ 改变，无法提到整个外层求和之外。于是 Kernels 在 $k$ 的 Main Loop 中不得不反复将 INT32 Partial Sum 转到浮点、乘 Scale、积累输出。GPU 的较慢 CUDA Core 参与每一轮 Main Loop，可能让标称很快的 INT4 Tensor Core 等待。

相反，如果 Activation 每 Token 一把 Scale，Weight 每输出通道一把 Scale，两个 Scale 均**不依赖 Reduction 索引**，就可以先完成一个标准 INT GEMM，再在最后的 Epilogue 还原

$$
y_{t,o}\simeq s_{x,t}s_{w,o}\sum_k q_{x,t,k}q_{w,k,o}.
$$

[Figure 4/5（PDF 第 3–4 页）](https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf) 解释 $m,n$ 两个 Tile 维度能并行，$k$ 需主循环串行 Reduction；被忽略的“每轮反量化指令”正落在最不应该增加开销的位置。

## QoQ 的 Progressive Group Quantization：为何先 INT8，再压成 INT4？

作者 §3.1 并不先做普通 INT4 Weight、运行时反量化回 FP16。它专门构造**两级整数表示**：

1. 对 FP 权重做 Per-Output-Channel 对称量化，得到 $q_8^{(0)}$，使用浮点 $s_0$；
2. 对这些 INT8 整数码，再按 Group 做非对称 INT4，得到 $q_4$，每组附带整数 Scale $s_1$、Zero Point $z_1$；
3. GPU 从 HBM 读取 Packed INT4，只解码到中间 INT8 表示，然后执行 **INT8×INT8 Tensor Core GEMM**，最后用 $s_0$ 等外层 Scale 还原输出。

$$
W\simeq s_0q_8^{(0)},\qquad
q_8^{(0)}\simeq s_1(q_4-z_1),\quad
\widehat W=s_0s_1(q_4-z_1).
$$

**计算输入类型是 INT8，存储 Weight 类型是 INT4**。所以 W4A8KV4 不能读成“真正以原生 INT4 Tensor Core 执行所有 GEMM”。

### Figure 6 真正给出了一个可复算的实例

[正式论文 Figure 6（PDF 第 5 页）](https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf) 使用一个 $w=-0.20$ 的位置，第一层 $s_0\approx0.07$，故 $q_8^{(0)}=\mathrm{round}(-0.20/0.07)\approx-3$。第二层的 Group 参数示例为 $s_1=2,z_1=8$：

$$
q_4=\mathrm{round}(-3/2)+8\approx7,\quad
\widehat q_8^{(0)}=(7-8)\cdot2=-2.
$$

重建到 FP 则约为 $-0.14$。这里 $-3/2=-1.5$ 恰在整数舍入的 Tie 位置，图中选择的 $q_4=7$ 对应一种向零处理 Tie 的约定；若直接使用 Python 的 Round-to-Even，得到的整数码可能不同。**实际复现必须遵守论文 Kernel 的 Round/Clip 语义**，而不是仅照抄语言运行时默认的四舍五入。这个例子说明两级量化并非逐 Weight 零误差；它放弃部分重建精度，以换取一个能够真正使用 **INT8 Tensor Core** 的表示。

### 为什么输入 INT8 的最大值必须限制在 ±119？

若第一层允许 $[-127,127]$，第二次 Group Round 可能把重建后的中间整数推到 128。原文 §3.1 的例子是某 Group 的码值在 $[-113,120]$，第二层可能得到 $s_1=16,z_1=7$，则最大的 $q_4=15$ 重建为 $(15-7)\cdot16=128$，已超过有符号 INT8 的 127。

作者据最大 Group Scale 与量化舍入误差的上界，在初次编码时留下安全余量，选择**保护范围 $[-119,119]$**。这是服务于无饱和指令的特定整数 Pipeline，而不是 INT8 物理类型突然变成只能表示 119。论文指出若依赖 Saturation 指令避免溢出，特定实现的计算吞吐会受到显著损害。

## “先乘后减”与 Register Parallelism 为什么能比“先减后乘”快？

普通公式看起来是 $(q_4-z_1)s_1$，但在固定宽度 SIMD、已打包的四个 INT4 数值以及 Register Live Range 条件下，采用

$$
(q_4-z_1)s_1=q_4s_1-z_1s_1
$$

可能改变依赖链和可向量化性。QServe §4.2.2 用**Subtraction After Multiplication**，并配合四路 Register-Level Parallelism 解码四个 INT4 Weight，尽量让较慢的 CUDA Core 计算不阻塞 Tensor Core 主循环。必须依赖前节保护范围，才能避开不合法整数溢出。

还有一项独立成本：[Figure 9](https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf) 展示线程需要的 Weight 布局与 Packed INT4 在 Shared Memory 中的物理布局不一致。直接套用按 INT8 设计的加载会增加 Pointer Arithmetic；QServe 在**离线**执行 Compute-Aware Weight Reorder，使运行时的线程加载排列更匹配 Tensor Core 所需数据。它与 [RPTQ](C046-rptq.md) 面向量化误差的通道重排是完全不同的设计目标。

~~~{mermaid}
flowchart LR
 W["浮点 W"] --> C["离线 per-channel INT8 / ±119"]
 C --> P["按 Group 再打包 INT4"]
 P --> L["静态 Weight 重排"]
 L --> D["寄存器 4-way INT4→INT8 / 乘后减"]
 D --> M["INT8×INT8 Tensor Core"]
 X["FP16 Activation"] --> Q["在 Norm/Activation Kernel 中融合 INT8 量化"]
 Q --> M
 M --> O["INT32 Accum + 外层 Scale"]
 O --> Y["FP16 输出"]
~~~

*自制算子执行图，非原论文 Figure；最终 Block 输入输出仍可以是 FP16，不代表所有算子、缓存和 Kernel 均采用相同 bit。*

## SmoothAttention：不量化 Query，为什么还要修改 Query 的数值？

KV Cache 的 Key 出现固定异常 Channel 时，直接使用 Per-Head INT4 容易损失大量正常 Token 的分辨率。[原论文 Figure 7（PDF 第 5 页）](https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf) 显示 Post-RoPE Key 的异常幅值约大于普通激活十倍，而 Value 没有同等明显的异常结构。

设 $\Lambda=\mathrm{diag}(\lambda)$，在同一 Head 内

$$
QK^\top=(Q\Lambda)(K\Lambda^{-1})^\top.
$$

因此可以**把长期缓存的 K 缩小**，同时在不进行 KV4 量化的 Q 侧承受对称放大，保证浮点 Attention Score 不变。作者在 §3.2 Eq.(4) 使用 $\lambda_i=(\max|K_i|)^\alpha$，实践中选择 $\alpha=0.5$；这是经验强度，不是所有分布的解析最优解。

这里最容易遗漏的是 RoPE 位置：每个 Head 内成对位置维度会旋转，任意 $\lambda_i\ne\lambda_j$ 的 Channel Scaling 不与 RoPE 交换。论文 §3.2 Eq.(5) 要求**相应 RoPE 旋转配对维度使用相同 $\lambda$**，才允许在 Q/K Projection Weight 里吸收 Scale。若没有满足条件却在 RoPE 前后随意调整顺序，浮点 Attention Score 都可能被改坏。

这与 [SmoothQuant](C040-smoothquant.md) 的不同，在于 SmoothAttention 的 Q 侧不做 4-bit KV 存储，不必在两侧均匀分担同一量化误差；代价则是 Q 侧数值范围/类型需要保持足够精度。压缩 KV4 并不是免费获得 4-bit Attention Compute，论文 Figure 8 明确映射：GEMM W4A8→FP16 Output，Attention 仍在较高精度的 CUDA Core 路径运行。

## 一个足以推翻“KV4 一定比 KV8 快两倍”的正式实验

[正式论文 Table 1（PDF 第 8 页）](https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf) 使用 A100 的 KV Attention Kernel：

| KV 序列长度 | KV8 | 朴素 KV4 | 优化后 QServe KV4 |
|---|---:|---:|---:|
| 128 | 0.09 ms | 0.10 ms | **0.07 ms** |
| 512 | 0.23 ms | 0.27 ms | **0.16 ms** |
| 1024 | 0.42 ms | 0.48 ms | **0.28 ms** |
| 1536 | 0.62 ms | 0.69 ms | **0.41 ms** |

当 $S=1024$，KV Payload 理论减半，直接替换实现却从 0.42ms 变成 0.48ms；经过降低解包 ALU 数、改进 Scale/Zero Point 预取和调整 FP16 运算路径后，才到 0.28ms，约 **1.5×**。论文解释这是 A100 的 CUDA Core Roofline 过早被解码/转换运算饱和。换成 L40S，计算/带宽比不同，朴素 KV4 本身也可能有收益。

因此部署 KV Cache 量化至少要画两个东西：HBM 读取字节随 Context 的变化，以及每个 Element 解包需要的非 Tensor Core ALU 指令数。若后一项超过前者省下的等待，就会发生 Table 1 所展示的性能反转。

## Figure 11 与 Table 5：最大系统吞吐包含“更大的 Batch”

论文在 A100 80GB SXM4、L40S 48GB、TensorRT-LLM v0.9.0 的设置下评测；输入 1024 Token、输出 512 Token，**统计内存约束允许的最大生成吞吐**，不是固定 Batch 的单请求 TTFT/TPOT。

[Appendix Table 5（PDF 第 17 页）](https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf) 报告：

| 同 GPU 的最优 TensorRT-LLM | TensorRT-LLM Token/s | QServe Token/s | 比例 |
|---|---:|---:|---:|
| A100 / Llama-3-8B | 2503 | 3005 | 1.20× |
| A100 / Qwen1.5-72B | 143 | 340 | 2.38× |
| L40S / Llama-3-8B | 2634 | 3656 | 1.39× |
| L40S / Qwen1.5-72B | 17 | 59 | 3.47× |

这些比值以每个 GPU、每个模型最好的 **TensorRT-LLM 可运行配置**为参照，某些大模型的 FP16 或 W8A8 对照会 OOM，不能把 OOM 当成 0 Token/s 去除。QServe 相比 TensorRT 的增益包含**更高单 Batch 效率和更大可承载 Batch**两部分，论文 Appendix Figure 18 单独研究固定 Batch 的相对吞吐。Figure 1 的 L40S/A100 采购成本比较是论文当年的设备经济性条件，不能当作2026年的云租赁报价。

**Figure 11** 对多个模型与两类 GPU 给整体吞吐曲线，**Figure 12** 分解 KV4、SmoothAttention、Progressive Group Quant、Weight Reorder 对成本和精度的贡献。要检验每个模块，必须固定硬件、模型、Quant Config 和 Batch，不可把其中某个系统最大 Speedup 说成一条算子指令独立提升了同等比例。

## 结论应当怎样指导真实工程？

- 对低 Batch，先量化 Weight 减少 HBM 字节，并评估 Packed Decode 固定开销；对高 Batch，优先确认 Tensor Core 的输入类型和利用率。
- 对长 Context，KV4 是否更快取决于 Decode Kernel 的解包 ALU、Scale MetaData、Page Layout，而不仅是“HBM 少一半”。
- 对完整模型，W4A8KV4 的 W/A/KV 三组位宽不是同一个计算精度：GEMM 真正以 INT8 Tensor Core 运算，Attention 主计算依旧有 FP16 路线。
- 对精度，浮点 QK 等价不等于量化后的 K 误差为零；特别是 RoPE 配对 Scale 错位会让未经量化的网络就失去等价性。
- 对上线目标，应将 **TTFT、TPOT、Max Token/s、显存、任务 PPL/Accuracy** 五类指标分别复核；没有 SLO 的最大吞吐不代表每个交互请求都能更快。

[DuQuant](C058-duquant.md) 致力于改善 Quantizer 面前的数据几何，[FlatQuant](C057-flatquant.md) 扩展了可学习的等价参数化，**QServe 则把精度方案再向前推进一步：如何设计真正顺着 GPU 执行路径流动的 Packed Data 和 Main Loop。** 相邻研究解决不同层次的瓶颈，不应只根据摘要里的加速倍数排序。
