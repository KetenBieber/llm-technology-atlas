# 分布式训练系统

这一专题研究：当模型、训练状态、长上下文 activation 已经无法由单张 GPU 承担时，怎样把一个训练任务拆到多张 GPU 上，同时控制显存、通信、pipeline bubble 与网络拓扑成本。

入口从 **4D Parallelism** 开始。这里不把 TP / PP / CP / FSDP 当作四个 API 记忆，而是分别追问它们切的是哪一个逻辑维度、产生什么 collective / point-to-point 通信，以及这些通信为什么必须映射到不同层级的物理互联。

从 4D 的 DP/FSDP 分支继续到 **ZeRO 原始论文**：模型状态内存为什么常以 16 bytes/parameter 估算？optimizer state、gradient、parameter 分别在何时能分片？为什么 Stage 3 是“按需收集”，而不是改变模型前向计算？它与 activation、临时 buffer 和通信开销之间存在什么交换？

随后沿模型层深度进入 **GPipe**（同步 micro-batch 流水线）与 **Megatron-LM**（TP/PP/DP 联合配置及 interleaved 1F1B），最后进入 **Ring Attention**，进一步分析固定 Query ownership、环形 KV 传输与长上下文计算通信重叠。四篇论文分别讨论 storage、depth、hidden、sequence 的不同系统瓶颈，彼此不可混作一种模型并行。

沿系统优化继续下钻：**PipeDream / 2BW** 对比同步 flush 与权重版本管理，**ZeRO-Offload / Infinity** 探索 CPU/NVMe 内存层级，**Ulysses / DistFlashAttn** 则与 Ring Attention 构成跨设备长上下文 Attention 的比较。最后阅读 **PagedAttention**，明确区分“训练阶段的模型状态/通信”与“推理阶段按请求增长的 KV Cache”。

```{toctree}
:maxdepth: 2

../generated/papers/C/01-distributed-training/C001-4d-parallelism
../generated/papers/C/01-distributed-training/C002-zero
../generated/papers/C/01-distributed-training/C003-gpipe
../generated/papers/C/01-distributed-training/C004-megatron-lm
../generated/papers/C/01-distributed-training/C005-ring-attention
../generated/papers/C/01-distributed-training/C006-pipedream
../generated/papers/C/01-distributed-training/C007-pipedream-2bw
../generated/papers/C/01-distributed-training/C008-zero-offload
../generated/papers/C/01-distributed-training/C009-zero-infinity
../generated/papers/C/01-distributed-training/C010-deepspeed-ulysses
../generated/papers/C/01-distributed-training/C011-distflashattn
../generated/papers/C/01-distributed-training/C012-vllm-pagedattention
```
