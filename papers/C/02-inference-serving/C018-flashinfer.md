# FlashInfer：KV Cache 形状每轮都变，GPU Kernel 为什么还能用 CUDA Graph？

> **原始论文**：Ye 等，[FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html)，MLSys 2025。**类型**：C · GPU Kernel 与动态服务执行引擎。

## 一、FlashAttention 解决的是哪一层问题？

[FlashAttention](../../A/10-gpu-operators/A055-flashattention.md) 通过分块在线 Softmax 避免写出 $S^2$ Attention Matrix。但在线服务还有一组专属难点：请求数每步变化、各请求上下文长度不同、KV Cache 可能采用 [PagedAttention](../01-distributed-training/C012-vllm-pagedattention.md) 的非连续物理 Page，Prefill 与 Decode 的矩阵形状更截然不同。

![FlashInfer 运行时的布局、负载平衡与可复用 Kernel](../../../figures/explainers/C018/01-overview.svg)

*自制解释图，非论文原图。原论文的 KV Layout、CUDA Graph 与 Kernel Scheduling 设计见 [MLSys 2025 正式版](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html)。*

连续 KV 的第 i 条请求可以表示为 $K_i\in\mathbb R^{L_i\times n_{kv}\times d_h}$。分页后，取第 j 个 Token 的物理地址要先查：

$$
p_i(j)=\operatorname{block\_table}_i[\lfloor j/B\rfloor],
\qquad o(j)=j\bmod B.
$$

其中 B 是每个 KV Page 中的 Token 容量。逻辑 Attention 没变，Kernel 却需要处理 **Page 索引读取与长度不规则**。如果简单地给每个请求一组固定 SMs，长请求可能占住 GPU，其余 SMs 却空闲。

## 二、为什么 CUDA Graph 和动态调度会冲突？

CUDA Graph 倾向重用已经捕获的 Launch 与执行图，以减少 CPU 为每轮 Kernel Launch 付出的固定开销。然而下一轮可能从 8 条请求变成 37 条，每条有效 Page 数也不同。若每轮根据任务重新生成复杂 Graph，CPU 开销会重新出现；若用最大尺寸填充，则会做大量无效工作。

因此需要分开两个对象：

- **可预先捕获/复用的执行形态**；
- **每轮变化的请求长度、Page Table 与实际分片任务元数据**。

FlashInfer 设计可与 CUDA Graph 兼容的负载均衡调度，让每轮通过动态元数据分配 Query/KV 工作，又尽量重用静态执行基础。这并非宣称所有 Memory Addresses 永远静态，更不是 CUDA Graph 自动解决请求长短不均衡。

## 三、可组合 KV Format 与 JIT 的价值是什么？

普通连续布局、Paged KV、不同 Attention masks、Head 数、不同位置编码与精度，对内存访问和 Kernel Specialization 有不同要求。FlashInfer 通过 **block-sparse/composable KV 表示与 customizable attention template** 将布局适配和 Kernel 算术结合，并允许通过 JIT 编译专用 Kernel。

从理解角度可划分三层：

$$
\underbrace{\text{Attention 数学定义}}_{\text{QK / mask / Softmax}}
\quad\big|\quad
\underbrace{\text{KV 物理布局}}_{\text{连续 / Paged / 稀疏索引}}
\quad\big|\quad
\underbrace{\text{GPU 执行安排}}_{\text{CTA / warp / split-K / Graph}}.
$$

只改变第二层不应改变第一层的数学值，但可能要求第三层完全不同。如何在运行时选策略，是这篇系统论文超出 FlashAttention 原始 tile 证明的核心。

### 为什么 kernel 更快不代表每个请求更快？

假设一次 decode step，Attention kernel 只占总时间 30%，其余 70% 是 GEMM/MLP/Sampling/Runtime。假设 Attention 快 2 倍，Amdahl Law 预测的总体加速至多：

$$
S=\frac1{0.7+0.3/2}\approx1.176.
$$

这是**教学假设**，非原论文硬件读数。它说明我们需要分别测 kernel-level latency、TPOT、端到端请求吞吐与显存占用。

## 四、原论文实证边界

论文在所选 inference engine 与 compiler baseline 上报告约 **29–69% 的 Inter-token latency 降低**，某些 Long-context 测试约 **28–30% 延迟降低**，Parallel Generation 工作负载约 **13–17% 加速**。这三个区间是不同实验工况，不能直接相加或认为任何 kernel 都有相同改进。

若研究 C++/CUDA 工程实现，应特别检查 Kernel 按 Q-length、KV-length、head 数如何选择 CTAs、是否复用 CUDA Graph、request 长度变化时 split-KV 的 balance。原文完整 Figure 及测试条件见上面的 MLSys 入口。

下篇 [Punica](C019-punica.md) 再加入一个新变化：同一个 Batch 内请求共享基础模型，却拥有**不同 LoRA Adapter**。这不是 KV 布局能独立解决的问题。

**依赖**：[FlashAttention](../../A/10-gpu-operators/A055-flashattention.md) → [vLLM/PagedAttention](../01-distributed-training/C012-vllm-pagedattention.md) → [SGLang](C017-sglang.md) → [FlashInfer](C018-flashinfer.md)。
