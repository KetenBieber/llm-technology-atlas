# FlexGen：单张小显存 GPU 为什么还可以运行超大模型？

> **原始论文**：[FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://proceedings.mlr.press/v202/sheng23a.html)，ICML 2023。 [官方/作者 PDF](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf)。


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

*自制硬件数据流图，非原论文 Figure。原文 Figure 的重点还包括按层/Batch 的执行时间线与显存峰值，不能只凭三级存储示意断言 I/O 已隐藏。*

## 为什么 Quantization 是让 Offloading 可行的另一轴？

如果每次从 SSD/CPU 搬 $B_W$，4-bit 体积理论上只有 FP16 的四分之一，带宽受限时可能显著降低搬运时间。但低比特权重必须解码，且 KV 和 Activation 仍占空间。FlexGen 使用其压缩与 offloading 策略联合优化；[GPTQ](C041-gptq.md) 更侧重权重保真压缩，[Atom](C044-atom.md) 更侧重服务器侧能实际使用的 4-bit 计算算子。**压缩能让模型跑起来**与**算得更快**是两个独立条件。

## 看数字必须理解基线任务性质

[ICML 2023 论文](https://proceedings.mlr.press/v202/sheng23a.html) 的目标是在有限 GPU 资源下执行更大的 LLM，并在离线高吞吐情境下比较其他 Offloading/基线系统。作者报告过在单张 16GB T4 的特定 OPT-175B 离线设置达到约 **1 token/s** 量级的吞吐，以当时同类有限资源基线比较取得显著加速；这种数值不代表 2026 年各种本地设备实际速度，亦不等于“单用户交互每秒 1 Token 一定能接受”。

自己搭建对照实验要固定模型、Prompt/输出长度、Batch、SSD 读带宽、PCIe/NVLink 带宽及 GPU 内存；分别记录数据传输、算术时间、峰值显存与吞吐。若用户主要目标是实时回答，FlexGen 的低延迟适用性需要单独评估，不应被吞吐倍数误导。
