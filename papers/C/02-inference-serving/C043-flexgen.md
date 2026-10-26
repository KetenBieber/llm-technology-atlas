# FlexGen：单张小显存 GPU 为什么还可以运行超大模型？

> **原始论文**：Sheng 等，[FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://proceedings.mlr.press/v202/sheng23a.html)，ICML 2023。[正式 PDF（重点见 Figure 3、Algorithm 1、Eq.(1)、Table 1）](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf)。本文讨论的是**离线高吞吐服务**，不能将其等同于交互式低延迟引擎。


基础：[ZeRO-Offload](../01-distributed-training/C008-zero-offload.md)；相关：[H2O 的缓存缩减](C036-h2o.md)、[GPTQ 的权重压缩](C041-gptq.md)；后续：[Atom 的 W4A4 服务吞吐](C044-atom.md)。

## GPU 显存根本放不下权重，还谈什么推理性能？

大模型自回归推理至少同时占用三类存储：模型权重 $W$、每层历史 KV 缓存 $K/V$、中间激活 $A$。最简单设权重均为 FP16，参数量 $P$，权重就需约 $2P$ 字节；若 $P=175$B，仅权重约 350GB，显然无法全部存入一张消费级 GPU。

直觉方案是将权重放在 CPU 内存，使用前拷到 GPU；如果主机内存也不够，就通过 SSD 作为第三级存储。问题转化为：

**哪些 Tensor 留 GPU、哪些换到 CPU、哪些放在磁盘，以及什么时候搬运，才能让 GPU 在有限显存下尽量不等待？**

这不再是改变 Transformer 的概率表达式，而是针对 GPU/CPU/SSD 三级存储建立调度策略。

## 为什么优化目标不是“单 Token 延迟最低”？

从 SSD 逐层读权重可能导致一次生成极其缓慢，但批量生成多请求时，可以把一次权重搬运的成本在多个 Token/序列上均摊。FlexGen 主要面向 **Latency-Insensitive 的高吞吐 Batch 推理**，如离线批量摘要，而不是把单用户交互式 TTFT 压到极低。

设某层权重为 $B_W$ 字节，主机到 GPU 有效带宽 $b_{HG}$，加载耗时下界近似

$$
T_{\rm transfer}\ge B_W/b_{HG}.
$$

若对 Batch $B$ 个序列分别逐次拷权重，则搬运会重复 $B$ 次；若先拷一层权重，再用它处理多个请求相关计算，带宽成本可被多个 Token 共同摊薄。但 Batch 增大又占用更多 KV 和激活，形成资源约束。

FlexGen 将决定权重、KV、激活在 GPU/CPU/磁盘的放置比例和传输成本形式化为 **Linear Programming（线性规划）**，寻找不同设备容量约束下的可行高吞吐策略。可以用示意性容量约束说明：

$$
M^{W}_{GPU}+M^{KV}_{GPU}+M^{A}_{GPU}\le C_{GPU},
$$

$$
M^{W}_{CPU}+M^{KV}_{CPU}+M^{A}_{CPU}\le C_{CPU}.
$$

除此之外，原论文还有 SSD 容量、传输、算子执行时序、重叠计算等约束；这不是完整 LP 目标。

但这份模型并不直接把“某张模型放得下”作为优化目标。作者定义了两个不同的 Batch 粒度：

- **GPU Batch Size** $b$：一次在 GPU 上完成的请求数，主要受显存和当轮 Activation/KV 约束。
- **Number of GPU Batches per Block** $n_b$：让同一层权重停留在 GPU、连续服务的 GPU Batch 数；**Effective Batch Size** 是 $b\,n_b$。

增加 $n_b$ 可以在更少权重搬运次数里生成更多 Token，但 CPU/SSD 必须容纳更多请求的 KV/Activation，因此不会无限增大。原文 §4.2 与 Algorithm 1 中，真正用于搜索的是两类调度参数和三类张量的位置策略，不是单一“调大 Batch”。

## 为什么要选 Zig-Zag Block Schedule？

普通 Transformer 的计算沿 Layer 顺序进行，若每一步都在 GPU/CPU/SSD 之间频繁拷入拷出，设备会大量等待。FlexGen 用 **Zig-Zag Block Schedule** 在多个 Layer、多个 Request/Block 中重新安排执行与搬运顺序，使权重重用、KV 访问和设备数据传输更均衡。它不改变每个 Token 的因果 Attention 约束，只重排合法计算与 IO 的时机。

~~~{mermaid}
flowchart LR
 SSD[SSD: 冷权重/KV] <--> CPU[CPU RAM: 分层缓存]
 CPU <--> GPU[GPU: 当前层计算]
 GPU --> B[分块处理多请求]
 B --> O[复用已搬运的权重]
 O --> GPU
~~~

*自制硬件数据流图，非原论文 Figure。此图只说明三层存储可能的搬运方向，不代表这些箭头都同时工作。真正的调度与重叠应对照原论文 Figure 3 和 Algorithm 1。*

### Figure 3 与 Algorithm 1：Zig-Zag 究竟在什么轴上“折返”？

[Figure 3（论文 PDF 第 5 页）](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf) 的横轴沿 **Token × Layer** 形成计算格子，纵向堆着不同 GPU Batches。上半 (a) **Row-by-row** 是先完成一个小 Batch 的各层各 Token，再处理另一 Batch；好处是能更快结束单个请求，坏处是同一层 Weight 必须反复从慢存储读取。下半 (b) **Zig-Zag Block** 则把多个 GPU Batch 的相同 Layer 靠在一起执行，按 Layer 在这些 Batch 间折返，让权重驻留期间被重复使用；但不能把整个数据集一口气遍历到同一列末尾，否则未完成请求的 Activation/KV 会耗尽 CPU/SSD 容量。

把某次生成第 $i$ 枚 Token、当前第 $j$ 层、第 $k$ 个 GPU Batch 的计算记作 $C_{i,j,k}$。一段最小教学时间线是：

~~~text
普通逐 Batch:
  载入 Layer 1 → 算 Batch 1 → 载入 Layer 2 → 算 Batch 1 ...
  重来：再次载入 Layer 1 → 算 Batch 2 → 再载入 Layer 2 ...

Block Schedule 的同一层局部:
  载入 Layer 1
     算 Batch 1 → 存 B1 的中间值、载入 B2 的中间值
     算 Batch 2 → 存 B2 的中间值、载入 B3 的中间值
     ...
  再进入 Layer 2：重复上述多个 Batch
~~~

*自制时间线，非作者源代码；只展示一个 Layer 的权重被多个 Batch 复用，略去了实际双缓冲和依赖边。*

原文 **Algorithm 1** 用三层循环 $i$（生成步）、$j$（Layer）、$k$（GPU Batch）组织上述执行。在最内层同时发起下一层 Weight Load、上一 Batch 的 Activation/KV Store、下一 Batch 的 Activation/KV Load、当前 Batch Compute，然后同步。**可以重叠的是满足依赖、并且硬件资源允许的操作**，不是宣称 PCIe/SSD 传输会被 GPU 计算 100% 遮蔽。

原论文 **Theorem 4.1** 进一步给出 Zig-Zag Block Schedule 的理论 I/O 复杂度与最优调度相差不超过 $2\times$。这个结论针对附录 A.2 规定的计算格图和 I/O 模型，**不能改写成真实 GPU 的延迟不超过最优算法两倍**：实际系统还有 Kernel、SSD 抖动、Pinned Memory、PCIe 拥塞和同步。作者提出了理论 I/O 更优的 Diagonal Block Schedule，但由于动态 KV Buffer/非连续访问的实现难度，实际工程只采用较容易落地的 Zig-Zag。

## 计算–搬运时间线：Offloading 里的“快”究竟来自哪里？

设单张 GPU 显存 $C_G=16$ GiB，需要运行一个 30B FP16 模型，仅权重就约 60GB，明显放不下。若其中每层权重被分批从 CPU/SSD 搬入 GPU，单纯逐 Token 每层搬一次就会重复读取极大量的数据。做 Batch 吞吐优化时可以复用已经搬入的 Layer Weight，在相同权重驻留期间执行多个 Sequence 的计算。

为建立下界，假设一轮需要跨 PCIe 搬运 10 GB 权重，有效带宽为 20 GB/s，那么**仅 Host→GPU 搬运**至少需

$$
T_{\rm H2D}\ge 10/20=0.5\ {\rm s}.
$$

如果单 Token 只获得一轮的计算成果，这会造成巨大交互延迟；若一次权重搬运在 Block Schedule 中让 32 条 Sequence 各推进一枚 Token，忽略其他开销，摊销权重搬运仅为 $0.5/32\approx15.6$ ms/Token。它不是每条请求都在 15.6 ms 后得到响应，**吞吐的均摊成本与单请求时延仍是两种量**。

### 为什么不能简单让所有东西都 Offload 到 SSD？

若读取频繁的历史 KV 也全部存磁盘，每次生成时会增加大量往返流量；若激活也存 SSD，计算块之间需频繁传输。FlexGen 把 Weight、KV、Activation 的放置比例作为不同变量，在 GPU、CPU、SSD 容量与带宽限制下同时优化。对于某一个 Tensor $X$，三处驻留比例满足

$$
g_X+c_X+d_X=1,\qquad g_X,c_X,d_X\ge0.
$$

各存储层还有占用约束 $\sum_XM_Xg_X\le C_G$、$\sum_XM_Xc_X\le C_C$，以及按时间线得到的有效带宽和执行重叠条件。注意这是用于说明放置决策的 LP 简化，并非原论文完整线性规划公式。

### 为什么论文说“线性规划搜索”，却不能直接对所有超参一次求最优？

论文 [§4.3 Eq.(1)](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf) 具体采用 **9 个位置比例变量**：

$$
p=(w_g,w_c,w_d,\ h_g,h_c,h_d,\ c_g,c_c,c_d).
$$

$w$ 代表 Weight，$h$ 代表 Activation，$c$ 代表 Attention KV Cache；下标 $g,c,d$ 分别是 GPU、CPU、Disk。每种 Tensor 的比例和为 1，例如 $w_g+w_c+w_d=1$。这三个角色的访问频率、搬运体积及生命周期完全不同，因此不能合成“所有 Tensor 用一个 Offload 百分比”。

对**先固定的** GPU Batch Size $b$ 和 Block Size $b\,n_b$，才求解

$$
\min_p\ \frac{T(p;b,n_b)}{b\,n_b}
\quad\text{s.t. GPU/CPU/Disk 容量、传输与其它可行约束。}
$$

上式是忠于原文 Eq.(1) 的简写：$T$ 表示对一个 Block 的解析执行时间预测，目标是它的**单位输出工作量时间**，不是单个用户的端到端完成时延。实现枚举候选 Batch 组合，对每组解一个关于位置比例的 LP，再比较模型预测吞吐。本文前面写的 $\sum M_Xg_X\le C_G$ 仅是**容量不等式的教学局部**，不是完整代价模型。

作者还提出了 **Computation Delegation** 这条重要分支：KV Cache 在 CPU 时，把历史 KV 全量搬回 GPU 计算 Attention 可能比在 CPU 算分数更费 I/O。原论文 §4.2 举例：FP16 下长度 $s$ 的 Cache 搬运量与隐藏维度乘 $s$ 成正比，而只把当前 Activation 送去 CPU 的传输与历史长度无关；当 $s$ 足够大（论文用 $s\ge512$ 举例），**即便 CPU 算力更低，也可能因节省 PCIe I/O 而有利**。这是成本模型下的决策，不是“CPU Attention 永远更快”。

## Zig-Zag Schedule 为什么不同于把 Batch 简单叠大？

若 Batch 增加，GPU 同时保留的 KV/Activation 也会上涨，因此根本无法无限扩 Batch。Zig-Zag 的思想是在层顺序、生成块与设备数据搬运之间重排计算，使已经驻留的某层 Weight 被多个 Microbatch 复用，同时数据传输可与其它可并行操作重叠。硬件要支持合法的计算依赖；若 Microbatch 下一层需要上一层输出但该 Tensor 还在 SSD，仍然必须等数据就绪。

[ICML 2023 官方论文](https://proceedings.mlr.press/v202/sheng23a.html) 的性能结论主要服务于吞吐优先的离线情境。要对比它与现代 [PagedAttention](../01-distributed-training/C012-vllm-pagedattention.md) 的在线服务，不仅要统一模型和 GPU，还要统一 TTFT、TPOT、Batch 和 Host/SSD 带宽，不能把“同一 GPU 可运行更大模型”误说成更低交互延迟。

## 为什么 Quantization 是让 Offloading 可行的另一轴？

如果每次从 SSD/CPU 搬 $B_W$，4-bit 体积理论上只有 FP16 的四分之一，带宽受限时可能显著降低搬运时间。但低比特权重必须解码，且 KV 和 Activation 仍占空间。FlexGen 使用其压缩与 offloading 策略联合优化；[GPTQ](C041-gptq.md) 更侧重权重保真压缩，[Atom](C044-atom.md) 更侧重服务器侧能实际使用的 4-bit 计算算子。**压缩能让模型跑起来**与**算得更快**是两个独立条件。

## 看数字必须理解基线任务性质

[ICML 2023 论文](https://proceedings.mlr.press/v202/sheng23a.html) 的目标是在有限 GPU 资源下执行更大的 LLM，并在离线高吞吐情境下比较其他 Offloading/基线系统。作者报告过在单张 16GB T4 的特定 OPT-175B 离线设置达到约 **1 token/s** 量级的吞吐，以当时同类有限资源基线比较取得显著加速；这种数值不代表 2026 年各种本地设备实际速度，亦不等于“单用户交互每秒 1 Token 一定能接受”。

真正解释速度主张的是 [Figure 1（论文 PDF 第 1 页）](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf)：实验用 **NVIDIA T4 16GB + 208GB CPU DRAM + 1.5TB SSD**，左图 OPT-175B、右图 OPT-30B；横纵轴分别涉及块延迟与生成吞吐的折中，**不同点是不同 Batch/调度策略**。Figure 1 的 Pareto Frontier 说明有限硬件上的最大吞吐可以随 Batch 变大而提高，但作者也同时在图上呈现了随之增加的延迟；不能只取吞吐最大的点来代表交互体验。论文摘要特别给出 OPT-175B 的 **1 token/s、Effective Batch Size=144**，这是大批量工作量的**聚合生成吞吐**，而非单条请求必然有 1 token/s 的响应服务保证。

自己搭建对照实验要固定模型、Prompt/输出长度、Batch、SSD 读带宽、PCIe/NVLink 带宽及 GPU 内存；分别记录数据传输、算术时间、峰值显存与吞吐。若用户主要目标是实时回答，FlexGen 的低延迟适用性需要单独评估，不应被吞吐倍数误导。
