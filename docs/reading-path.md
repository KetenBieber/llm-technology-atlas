# 学习路线

这条路线优先保证“后面的文章不会突然出现一个完全陌生的关键概念”。不要求一次性读完所有论文；遇到已经掌握的节点可以直接跳过。

## 第一阶段：训练与网络基础

1. {doc}`反向传播 <generated/papers/A/01-foundations/A001-backpropagation>`
2. {doc}`Adam <generated/papers/A/01-foundations/A002-adam>`
3. {doc}`AdamW <generated/papers/A/01-foundations/A003-adamw>`
4. {doc}`Batch Normalization <generated/papers/A/01-foundations/A004-batch-normalization>`
5. {doc}`ResNet <generated/papers/A/01-foundations/A005-resnet>`
6. {doc}`Scalable Muon <generated/papers/A/01-foundations/A006-scalable-muon>`

## 第二阶段：Transformer 与高效注意力

7. {doc}`Transformer <generated/papers/A/03-transformer/A013-transformer-attention-is-all-you-need>`
8. {doc}`RoPE <generated/papers/A/03-transformer/A014-roformer-rope>`
9. {doc}`RMSNorm <generated/papers/A/03-transformer/A016-rmsnorm>`
10. {doc}`MQA <generated/papers/A/04-efficient-attention/A019-mqa>`
11. {doc}`GQA <generated/papers/A/04-efficient-attention/A020-gqa>`

## 第三阶段：稀疏 MoE

12. {doc}`Sparsely-Gated MoE <generated/papers/A/05-moe/A032-sparsely-gated-moe>`
13. {doc}`Switch Transformer <generated/papers/A/05-moe/A033-switch-transformer>`
14. {doc}`DeepSeekMoE <generated/papers/A/05-moe/A034-deepseekmoe>`

## 第四阶段：现代大模型主线

15. {doc}`DeepSeek-V2 / MLA <generated/papers/B/05-moe-complete-llm/B008-deepseek-v2>`
16. {doc}`DeepSeek-V3 <generated/papers/B/05-moe-complete-llm/B009-deepseek-v3>`
17. {doc}`DeepSeek-R1 <generated/papers/B/05-moe-complete-llm/B010-deepseek-r1>`

## 第五阶段：Reasoning Post-training

18. {doc}`DeepSeekMath / GRPO <generated/papers/A/08-post-training/A051-deepseekmath-grpo>`

## 第六阶段：第二条现代模型主线

19. {doc}`Llama 3 <generated/papers/B/05-moe-complete-llm/B011-llama3>`
20. {doc}`Qwen2.5 <generated/papers/B/05-moe-complete-llm/B012-qwen2.5>`

这一阶段开始做“模型之间的系统设计对照”：DeepSeek 主线重点看 MLA / MoE / reasoning RL；Llama 3 看 dense Transformer 上的数据工程、scaling law、128K continued pre-training、4D parallelism 与 RS/SFT/DPO；Qwen2.5 再把视角扩展到多尺寸模型家族、18T 数据、Offline DPO → Online GRPO 与 1M 长上下文系统。

## 第七阶段：回到 Preference Learning 方法层

21. {doc}`Direct Preference Optimization <generated/papers/A/08-post-training/A052-dpo>`

连续读完 Llama 3 与 Qwen2.5 后再回到 DPO，可以把“为什么实际模型频繁采用 DPO”与它的 Bradley–Terry、KL-regularized RL、implicit reward、reference policy 和 gradient weighting 一次性对应起来。

## 第八阶段：回溯 Scaling Laws 与训练规划

22. {doc}`Kaplan Scaling Laws <generated/papers/A/02-text-representation/A009-kaplan-scaling-laws>`
23. {doc}`Chinchilla / Compute-Optimal Training <generated/papers/A/02-text-representation/A010-chinchilla-compute-optimal>`

这一阶段回头解释 Llama 3 与 Qwen2.5 里已经出现的 IsoFLOPs、compute-optimal、small-run extrapolation 与 overtraining：Kaplan 建立可预测 scaling 范式，Chinchilla 再用三套受控实验重新估计固定 FLOPs 下参数量与训练 token 的分配。

## 第九阶段：长上下文位置外推

24. {doc}`YaRN <generated/papers/A/03-transformer/A018-yarn>`

这一阶段把 Qwen2.5 中的一句“YaRN + DCA”拆开：先从 RoPE 的相对位置与外推失败出发，理解 Position Interpolation、NTK-aware / NTK-by-parts，再到完整 YaRN 的 frequency-selective interpolation 与 attention magnitude scaling。

## 第十阶段：长上下文 Attention Relation

25. {doc}`Dual Chunk Attention <generated/papers/A/04-efficient-attention/A053-dual-chunk-attention>`

这一阶段不再只改 RoPE frequency，而是直接研究 relative-position matrix：同一个 query 对 current chunk、previous chunk 和更早 chunks 为什么需要三种位置映射，怎样在不训练权重的情况下保留 local precision 与 global content access，以及为什么这仍然没有解决 O(L²) prefill。

## 第十一阶段：百万上下文 Prefill Runtime

26. {doc}`MInference 1.0 <generated/papers/A/04-efficient-attention/A054-minference>`

这一阶段把问题从 positional geometry 推到系统执行：为什么 1M prompt 的 dense prefill 会让 TTFT 达到分钟级，attention sparsity 为什么既高度稀疏又 input-dynamic，以及怎样通过 per-head offline pattern search、online sparse-index approximation 与 GPU-friendly sparse kernels 真正减少 QK pair 计算。读完后，下一项系统基础依赖是 FlashAttention。

## 第十二阶段：GPU IO-aware Attention

27. {doc}`FlashAttention <generated/papers/A/10-gpu-operators/A055-flashattention>`

这里从“少算哪些 QK pair”回到更底层的执行问题：即使 dense Attention 的数学工作完全不减少，为什么仍能通过 SRAM tiling、online softmax、kernel fusion 与 backward recomputation 大幅降低 HBM traffic；并理解“更多 FLOPs 反而更快”在 memory-bound GPU kernel 中为什么成立。下一节点是 FlashAttention-2 的并行分解与 work partition。

## 第十三阶段：GPU Parallelism 与 Work Partition

28. {doc}`FlashAttention-2 <generated/papers/A/10-gpu-operators/A056-flashattention2>`

FA1 把 HBM traffic 压下去以后，新的瓶颈变成 GPU utilization。本阶段重点理解 thread block / warp / SM、sequence-level parallelism、split-K 与 split-Q 的通信差异，以及为什么 Big-O 与 IO complexity 都不变，仍然可以再获得约 2× 的 kernel speedup。完成后转向 Llama 3 已经明确暴露的 TP / PP / CP / FSDP 多 GPU 并行主线。

## 第十四阶段：分布式训练系统

29. {doc}`4D Parallelism：TP / PP / CP / FSDP <generated/papers/C/01-distributed-training/C001-4d-parallelism>`

这一阶段把 Llama 3 中已经出现的 4D parallelism 真正展开：从训练显存账本出发，分别理解 TP 切 layer 内 tensor、PP 切 layer depth、CP 切 sequence、DP/FSDP 切 data replica 与 model states；再推导 pipeline bubble、global batch、process group、all-gather / reduce-scatter 与 topology-aware placement。读完后，就可以继续进入 Hopper 上的 FlashAttention-3，或继续下钻通信 overlap 与大规模训练 runtime。


## 第十五阶段：Hopper Asynchrony 与低精度 Attention

30. {doc}`FlashAttention-3 <generated/papers/A/10-gpu-operators/A057-flashattention3>`

FA2 解决 work partition 后，H100 上的新瓶颈转向 heterogeneous execution units 的同步等待。本阶段重点理解 TMA / WGMMA、warpgroup、producer-consumer specialization、circular SMEM buffer、ping-pong 与 2/3-stage GEMM-softmax pipeline，并进一步理解 FP8 的 k-major layout、in-kernel transpose、block quantization 与 incoherent processing。到这里，FlashAttention 主线形成 IO → Parallelism → Asynchrony → Low Precision 的完整硬件演化链。

读到 V3 / R1 时不需要把所有方法重新推一遍；直接使用文章中的站内链接回到 MLA、MoE 或 GRPO 专题，再返回模型主线即可。
