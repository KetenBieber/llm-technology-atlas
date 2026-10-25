# DistFlashAttn：分布式 FlashAttention 为什么不能简单复制 Ring？

> **论文**：[DistFlashAttn: Distributed Memory-efficient Attention for Long-context LLMs Training](https://arxiv.org/abs/2310.03294)，Li 等，COLM 2024；[正式会议论文](https://openreview.net/forum?id=pUEDkZyPDl)。
> **类型**：C · 长上下文训练系统。核心问题：环形传送 KV 已经让训练能使用多张卡，为什么设备仍会不均衡、通信阻塞和反向重计算放大？

## 阅读导航：从 Ring 的正确性走到性能

### 把原论文 Figure 1 的“等待一半 GPU”写成时间线和负载量

[DistFlashAttn COLM 2024 原始论文 Figure 1（PDF 第 2 页）](https://arxiv.org/pdf/2310.03294) 按图注明确比较 **8 个 Worker** 的两种方案：(a) 朴素 Ring 使较早 Token 所属 Worker 的有效因果 Attention 量更少，(b) 作者的 Workload Balancing 把后续 Token 多出来的 Query/Key 块工作分给原本空闲的 Worker。图注明示普通 Ring 在设备数 $P$ 增大时空闲比例趋近 $1/2$，优化后当 $P$ 为偶数时空闲比例可近似为 $1/(2P)$，$P$ 为奇数时理论上可为 0。图的两个“Finish”时间分别用于对比调度完工的时隙，不是所有 GPU 实测运行时间的统一绝对值。

这与前文 $S=8$、双 Worker 的 $W_0=10,W_1=26$ 示意一致：分给早期连续 Token 的 Worker 不能因为“也有四个 Query”就假装它和后期 Worker 的乘法工作量完全相同。在严格 Causal Attention 下，**可见 Key 的数量**才决定每个 Query 的有效计算工作。负载均衡需要重新分配有效 Token 块并维护 Owner/通信和 Softmax 统计，不能通过简单的“前半张卡算完直接休息”解决。

原文 **Figure 2（PDF 第 3 页）** 用第 7 号 Worker 的任务展示 **Overlap of KV Communication and Attention Compute**：一边 Fetch 后续 KV Block，一边对当前可用 KV 执行 Attention。它说明时间重叠设计；如果 PCIe/网络通信耗时始终大于可并行的矩阵计算，这些传输仍可能留在关键路径。**Figure 3（PDF 第 5 页）** 则比较 HuggingFace Checkpointing 与 Rematerialization-aware Checkpointing 的重新计算路径；性能收益还需实际 Checkpointing 消融，不能只凭画出的箭头少就认定总耗时必然降低。

### 原论文 §3 的三条设计与 Ring 基线究竟差在哪？

[DistFlashAttn COLM 2024 原文](https://arxiv.org/pdf/2310.03294) 的三项贡献是 Token-level Workload Balancing、Overlapping KV Communication 与 Rematerialization-aware Gradient Checkpointing。第一项处理因果 Attention 的**非对称有效工作量**：由于位置越靠后的 Query 能读到的合法 Key 越多，若各设备持有同样长度的连续 Query Token、都执行完整的 Causal Attention，后半段设备的有效计算量会更多，前半段可能空转。不是每个拥有 $S/P$ 个 Query 的 Rank 都有相同的有效点积数。

设 $S=8$、两个设备，设备 0 持有位置 1–4，设备 1 持有 5–8。每个 Query $i$ 可见的合法 Key 数在因果掩码下为 $i$，两台设备工作量比例是

$$
W_0=1+2+3+4=10,\qquad
W_1=5+6+7+8=26.
$$

即使 Sequence 切分长度都是 4，真实有效计算也不是 1:1。若改成让每台设备同时负责早期和后期的 Query，则可以在保持正确因果 Mask 的同时更均匀地分摊工作。这是负载均衡的动机示例，不冒充原文实际的 Token 排列方案。

第二项通过 **KV 接收与当前 Block 的 Attention Kernel 重叠**来隐藏通信，受限于 $T_{\rm compute}\ge T_{\rm communication}$ 的资源条件。第三项关注 Backward 的 Rematerialization：若训练用普通 Activation Checkpointing，在 Backward 重新计算一个 Transformer Block 时可能再次触发跨设备 KV 传输，增加网络成本；DistFlashAttn 的 Checkpointing 需要在保存多少中间状态与额外 KV 通信之间重新权衡。

本文真正要验证的是原文关于不同 Context、GPU 数下的通信/计算时间消融，不是仅凭作者报告的最高 Speedup 认定三个改动都同等重要。特别是 Ring 和 Ulysses 采用了不同 Head/Sequence 分片与通信路径，必须在同一模型、Context、设备条件下比较端到端训练时长。

先读 [Ring Attention](C005-ring-attention.md) 的按块 online softmax，再读 [DeepSpeed-Ulysses](C010-deepspeed-ulysses.md) 的 All-to-all 维度交换。三者都力图让**精确长上下文训练**能跨设备执行，但它们组织每个 rank 的 Query ownership 与 KV transport 的方式不同。

![自制解释图：DistFlashAttn 的三项系统补偿](../../../figures/explainers/C011/01-mechanism.svg)

*自制解释图，非原论文 Figure。原作者图和性能图见 [COLM 正式论文](https://openreview.net/forum?id=pUEDkZyPDl)。*

## 一、Causal Attention 使“各 GPU 分到一样多 token”不等于计算均衡

因果自注意力的一个 Query 位置 $i$ 只允许看到 Key 位置 $j\le i$。完整密集注意力矩阵的有效部分是三角形而非正方形：

$$
M_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

假设总序列 $S$，分 $P$ 个连续 token blocks。若 rank 0 负责最前面的 Query，它只需访问很短的历史；rank $P-1$ 负责最末尾的 Query，则需要看全序列的大部分 KV。

在等长切分下，rank $r$ 的有效 QK pairs 近似为：

$$
C_r\approx
\left(\frac SP\right)^2
\left(r+\frac12\right).
$$

这里忽略 block 边界整数离散性，假定从 $r=0$ 编号、每块等长。这个式子来自：当前 $S/P$ 个 Query 都看到前面 $r$ 个完整块，外加本地 block 的下三角约半个平方。于是后面的设备拥有更多有效计算任务。

### 一个四 GPU 的直观例子

令每 block 有 $c$ 个 tokens，$P=4$：

$$
C_0:C_1:C_2:C_3
\approx0.5:1.5:2.5:3.5.
$$

最大负载约为最小的七倍。即使 Ring KV 分布方式完全正确，**最忙 GPU**也可能决定一轮结束时间。仅“把 tokens 平均分配给 GPU”并不是足够好的 workload balancing。

这也解释了为何本文的第一项设计是 **token-level workload balancing**，而不仅仅是简单地加快网络带宽。

## 二、重新安排 Query 的责任，不等于改变 Causal Mask

DistFlashAttn 让不同设备承接更均衡的有效 Query/Key interactions。可以把靠前、计算量轻的 query positions 与靠后、计算量重的位置以合适映射组合，在每 GPU 的工作量中平衡上下三角区域。

但数学约束仍然是：

$$
O_i=\sum_{j\le i}\alpha_{ij}V_j,
\qquad
\alpha_{ij}=
\frac{e^{q_i k_j^\top/\sqrt{d_h}}}
{\sum_{l\le i}e^{q_i k_l^\top/\sqrt{d_h}}}.
$$

改变执行位置不能改变 $j\le i$ 的语义。如果系统把 Query 重新排列，必须依靠原始绝对位置的 mask、恢复索引和梯度回写来保持输出与预设 attention 一致。

这也是为什么**token-level balancing** 的正确性要求高于简单把 GPU 任务表画得均匀：要确保没有漏算有效 pair，也没有误算未来 Key。

## 三、Overlap KV communication：什么时候发送不阻塞 GEMM？

对于一个 KV block，发送时间：

$$
T_{\rm send}\simeq V_{\rm KV}/B_{\rm net}.
$$

与本地 Query block 计算时间：

$$
T_{\rm compute}\simeq W_{\rm QK/PV}/F_{\rm effective}.
$$

如果发送下一块与处理当前块能异步交叠：

$$
T_{\rm stage}\gtrsim
\max(T_{\rm compute},T_{\rm send})
$$

可比串行相加更短。但若 block 小、causal mask 删除大量计算、网络带宽低，就可能出现：

$$
T_{\rm send}>T_{\rm compute},
$$

这时通信会暴露在关键路径上。尤其**负载均衡之后**部分 GPU 的 compute time 发生改变，调度应与通信重叠一起优化，不能分别假定二者独立达到最优。

## 四、为什么 Gradient Checkpointing 会影响远程 KV 的网络成本？

常见 activation recomputation 方案仅在 forward 保存少量 checkpoints，在 backward 前重新执行 forward 恢复中间状态。单 GPU 上这主要交换 GPU FLOPs 与 HBM bytes。

分布式 exact Attention 情况不同。某个 Q block 的 forward 重算，可能又要经过远程 KV blocks：

$$
\text{重新计算 Attention}
\Rightarrow
\text{再次访问远程 K/V}
\Rightarrow
\text{额外网络通信}.
$$

因此常规的 activation checkpoint policy 若只算“少保留多少激活”，会忽略重计算造成的第二轮跨设备 KV 访问。

本文提出的**rematerialization-aware gradient checkpointing** 正是在训练执行和远程数据生命周期层面重新权衡：保存哪个统计量或边界状态、什么时候重算、是否可以重用跨卡访问，必须放进同一份调度成本模型。

不能把“FlashAttention 自身 backward 会做 recomputation”与“分布式 checkpoint 必须重新传 KV”混为一件事；前者研究 GPU SRAM/HBM，后者还牵涉 network communication。

## 五、原始实验支持到什么程度？

论文在 Llama-7B 及变体上评估约 **32K–512K** 长度的训练，作者报告在特定系统对照下相对 Ring Self-Attention、Megatron-LM+FlashAttention、Ring Attention 和 Ulysses 获得不同程度性能优势。

应特别注意其摘要里的对比不是同一 baseline：对不同系统分别报告了**可训练序列长度倍率**与**速度倍率**。不要从若干上限数字挑最大值，拼接成“同一条件下同时提升 8 倍长度与 5.6 倍速度”。

原论文对 Ring Attention 的性能对比还涉及不同实现细节、block 大小与 checkpoint 方案；如要做正式硬件 benchmark，需要锁定相同模型、global batch、设备、dtype 和 FLOPs 统计口径。

## 六、这篇论文对系统研究真正有价值的地方

它让我们明白：**算法正确性**、**静态工作量分配**、**跨卡通信**和**反向重算**是四个不同层次。

- Ring Attention：首先建立 blockwise exact attention 与 KV circulate 的可执行机制。
- Ulysses：用 sequence/head sharding 交换实现低通信量的另一设计点。
- DistFlashAttn：针对 causal 矩阵的不均衡，进一步优化 load balancing、overlap 与 checkpoint-aware transfer。

全文价值不是多一条速度排名，而是教会我们问：

> 谁负责哪些有效的 $QK^\top$ tiles？那些 tiles 需要从谁那里取得 K/V？如果反向再执行，数据还要再搬一次吗？

下一站会切换**训练系统**到**在线推理系统**：[PagedAttention / vLLM](C012-vllm-pagedattention.md) 研究动态请求生成时 KV cache 的存储碎片，而不是跨 GPU 训练的 KV transmission。

**原始资料**：[COLM 2024 论文](https://openreview.net/forum?id=pUEDkZyPDl)、[arXiv 2310.03294](https://arxiv.org/abs/2310.03294)。
