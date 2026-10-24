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

## 第十六阶段：ZeRO 与模型训练状态的存储所有权

31. {doc}`ZeRO：为什么数据并行不需要每卡都永久复制全部训练状态 <generated/papers/C/01-distributed-training/C002-zero>`

沿第十四阶段的 4D Parallelism 继续深入 DP/FSDP 分支：先用混合精度 Adam 的 16 bytes/parameter 建立训练内存账本，再从参数、梯度和优化器状态的不同生命周期推导 ZeRO-1/2/3。重点解释为什么 reduce-scatter + all-gather 在 ZeRO-2 的理论通信体积仍约为 $2\Psi$，而 ZeRO-3 必须在 forward/backward 各自临时收集权重，使体积约为 $3\Psi$；最后讨论 ZeRO-R 的 activation partitioning、临时 buffer、碎片管理，以及原论文已实现的百亿/千亿级训练和万亿参数理论容量之间的证据边界。

## 第十七阶段：GPipe——层深度切分和同步 micro-batch

32. {doc}`GPipe：把网络分到多张卡后，为什么还必须把一个 batch 切得更碎 <generated/papers/C/01-distributed-training/C003-gpipe>`

这一阶段把 PP 从一个名字还原成可执行时间线。重点推导为什么 layer partition 本身无法提高单样本的串行依赖吞吐、为什么 micro-batches 可以填满流水线、为何整批梯度聚合后更新仍保留同步 SGD 语义；再区分相对 makespan 的 bubble fraction $(K-1)/(M+K-1)$ 与相对 ideal compute 的 overhead $(K-1)/M$，讨论 activation recomputation 与 BatchNorm 的边界。

## 第十八阶段：Megatron-LM——TP/PP/DP 联合权衡

33. {doc}`Megatron-LM：为什么组合并行维度仍然是一道优化题 <generated/papers/C/01-distributed-training/C004-megatron-lm>`

先从 MLP 的列切与行切推导 GeLU 局部计算和归约语义，再把 layer 内高频 TP collective、stage 边界的 PP p2p 和 DP 梯度聚合放在统一成本模型里。比较 GPipe、PipeDream-Flush 1F1B 与论文的 interleaved chunks，推导 bubble overhead 下降约 $1/v$、通信量却上升的条件，并回到大模型集群实测与理论完整训练时长估计的区别。

## 第十九阶段：Ring Attention——精确全局 Attention 的跨设备数据流

34. {doc}`Ring Attention：固定 Query、环形传递 KV 的第一性推导 <generated/papers/C/01-distributed-training/C005-ring-attention>`

在 FlashAttention 已避免保存完整 $S^2$ 矩阵之后，为什么超长序列的中间激活仍受单卡 HBM 约束？这一阶段推导 online softmax block statistics 的正确合并，解释 KV ring 传输的因果逻辑和 $c\gtrsim F/B$ 的通信隐藏门槛，并明确区分“容量随着设备数量扩展”与“总 Attention FLOPs 被减少”。

## 第二十阶段：PipeDream 与 2BW——流水线中权重版本的正确性

35. {doc}`PipeDream：为什么消除 flush 需要 Weight Stashing <generated/papers/C/01-distributed-training/C006-pipedream>`
36. {doc}`PipeDream-2BW：限制两个权重版本的训练调度 <generated/papers/C/01-distributed-training/C007-pipedream-2bw>`

先证明 backward 必须使用对应 forward 的权重，再区分梯度局部正确与同步 SGD 完全等价两种条件；理解 PipeDream 的 1F1B 无 flush/权重 stashing，以及 2BW 的梯度合并、双缓冲与更新滞后取舍。

## 第二十一阶段：ZeRO-Offload 与 Infinity——从 GPU HBM 扩展到 CPU/NVMe

37. {doc}`ZeRO-Offload：将 Adam 状态与 CPU 更新从 GPU 主干解耦 <generated/papers/C/01-distributed-training/C008-zero-offload>`
38. {doc}`ZeRO-Infinity：NVMe、带宽分区、预取和异步传输 <generated/papers/C/01-distributed-training/C009-zero-infinity>`

在 ZeRO-1/2/3 基础上，继续区分分片所有权、驻留层级和计算位置；推算 CPU/GPU 数据搬运量，分析为什么 Infinity 必须聚合多条 PCIe 链路并重叠 NVMe/PCIe/GPU 通信。

## 第二十二阶段：Ulysses 与 DistFlashAttn——长上下文计算的跨设备资源平衡

39. {doc}`DeepSpeed-Ulysses：两次 All-to-all 交换 sequence/head 分片 <generated/papers/C/01-distributed-training/C010-deepspeed-ulysses>`
40. {doc}`DistFlashAttn：Causal workload balancing 与 checkpoint-aware KV 通信 <generated/papers/C/01-distributed-training/C011-distflashattn>`

与 Ring Attention 比较三种解法：固定 Query/轮转 KV，按 heads 独立重分片，或针对 Causal Attention 的有效 pair 分配负载；分别证明计算语义并评估通信/重计算。

## 第二十三阶段：vLLM / PagedAttention——推理服务的动态 KV Cache 管理

41. {doc}`PagedAttention：逻辑连续而物理离散的 KV Cache <generated/papers/C/01-distributed-training/C012-vllm-pagedattention>`

将训练系统的状态管理与 decode 服务阶段分开：推导每 token KV 容量与 block 内部碎片上界，解释 block table、按需分配、prefix sharing、Copy-on-write 与 continuous batching 如何影响请求吞吐。

## 第二十四阶段：Orca 与 Sarathi-Serve——动态请求与 Prefill 时间切分

42. {doc}`Orca：按 Decode Iteration 调度与 Selective Batching <generated/papers/C/02-inference-serving/C013-orca>`
43. {doc}`Sarathi-Serve：Decode 优先的 Chunked Prefill <generated/papers/C/02-inference-serving/C014-sarathi-serve>`

沿 vLLM 的动态 KV 管理进入在线调度：先弄清请求为何不能当作一次静态推理，再研究怎样限制每轮 Prefill Token Budget，避免长 Prompt 造成生成 Token 的等待抖动。

## 第二十五阶段：DistServe 与 Splitwise——阶段分离的 SLO 与硬件成本

44. {doc}`DistServe：Goodput 导向的 Prefill/Decode GPU 分离 <generated/papers/C/02-inference-serving/C015-distserve>`
45. {doc}`Splitwise：为何 Prefill/Decode 应匹配不同类型 GPU <generated/papers/C/02-inference-serving/C016-splitwise>`

在同池调度之外引入网络状态交接，推导 KV Migration 的字节数量，区分吞吐与满足 TTFT/TPOT 条件的 Goodput；最后用 Roofline、硬件功率与预算说明分阶段异构部署的取舍。

## 第二十六阶段：SGLang、FlashInfer、Punica——推理运行时的数据复用

46. {doc}`SGLang：RadixAttention、公共 Prefix 与 LLM Program 执行 <generated/papers/C/02-inference-serving/C017-sglang>`
47. {doc}`FlashInfer：KV Layout、负载均衡与 CUDA Graph <generated/papers/C/02-inference-serving/C018-flashinfer>`
48. {doc}`Punica：共享 Base Weight 与多租户 LoRA Batching <generated/papers/C/02-inference-serving/C019-punica>`

把逻辑 KV 与物理地址的关系延伸到跨请求前缀共享，再进入动态 Attention GPU Kernel，最后讨论不同 Adapter 的低秩路径如何与公共基础模型一起 Batch。

## 第二十七阶段：Speculative Decoding、Medusa 与 EAGLE——严格采样与草稿模型

49. {doc}`Speculative Decoding：接受–拒绝与残差分布的严格证明 <generated/papers/C/02-inference-serving/C020-speculative-decoding>`
50. {doc}`Medusa：多头草稿候选树与 Tree Attention <generated/papers/C/02-inference-serving/C021-medusa>`
51. {doc}`EAGLE：Feature Autoregression、Shifted Token 与 Target 验证 <generated/papers/C/02-inference-serving/C022-eagle>`

由“每轮只产生一个 Token”推出 Speculation，证明目标分布不变性，再研究如何改造候选生成方式：独立 Draft、额外未来 Token Heads 与第二倒数层 Feature Predictor。严格区分原始模型分布不变、联合微调后的 Target 分布、Typical Acceptance 等不同正确性语义。

读到 V3 / R1 时不需要把所有方法重新推一遍；直接使用文章中的站内链接回到 MLA、MoE 或 GRPO 专题，再返回模型主线即可。
