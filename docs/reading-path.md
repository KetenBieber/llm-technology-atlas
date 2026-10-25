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

## 第二十八阶段：EAGLE 动态树与多步草稿训练

52. {doc}`EAGLE-2：上下文感知的候选树扩展与重排 <generated/papers/C/02-inference-serving/C023-eagle2>`
53. {doc}`EAGLE-3：多层特征融合、Token 预测与 Training-Time Test <generated/papers/C/02-inference-serving/C024-eagle3>`

在 EAGLE 已经能够快速产生草稿之后，首先解释为什么固定树型会错配不同上下文的难度：EAGLE-2 使用 Draft 的条件置信度构造近似路径价值，分开执行“继续扩展哪些分支”和“Target 最终验证哪些节点”；它并不改动 Target 的接受–残差校正。接着分析 EAGLE-3 的另一重瓶颈：真实 Feature 上训练的 Draft，在多步推测时会读入自己的预测状态；该方法通过多层 Feature 融合、取消 Feature 预测约束、训练时模拟推测展开，改善多步草稿质量。两篇分别处理候选预算与草稿模型，两者的加速数字都不能脱离硬件、任务、采样设置和 Batch 直接搬用。

## 第二十九阶段：候选网络、树形执行与预算优化

54. {doc}`Hydra：顺序依赖的未来预测头 <generated/papers/C/02-inference-serving/C025-hydra>`
55. {doc}`ReDrafter：循环草稿与动态树 <generated/papers/C/02-inference-serving/C026-redrafter>`
56. {doc}`SpecInfer：Token Tree 并行验证 <generated/papers/C/02-inference-serving/C027-specinfer>`
57. {doc}`Sequoia：动态规划与硬件感知的树预算 <generated/papers/C/02-inference-serving/C028-sequoia>`

从 Medusa 的独立预测头改为 Hydra 的顺序依赖，再转到 ReDrafter 的循环草稿状态与 Beam Tree；SpecInfer 单独解释如何在 Target 中共享树前缀、保证祖先条件可见；Sequoia 进一步在预算限制下规划最有价值的树。必须区分候选模型预测质量、树形结构、并行执行正确性与整体系统时延。

## 第三十阶段：候选的非传统来源与短语复用

58. {doc}`Lookahead Decoding：Jacobi Window 与 n-gram <generated/papers/C/02-inference-serving/C029-lookahead-decoding>`
59. {doc}`REST：检索后缀与 Trie 候选 <generated/papers/C/02-inference-serving/C030-rest>`
60. {doc}`Ouroboros：短语辅助草稿与长候选 <generated/papers/C/02-inference-serving/C031-ouroboros>`

Lookahead 的 n-gram 来自同一个模型的 Jacobi 迭代轨迹，REST 的候选来自外部语料数据存储，Ouroboros 则仍用小型 Draft 而在其生成环节利用短语池。三者不能因为都使用短语匹配而归为同一实现。

## 第三十一阶段：自推测与质量—时延折中

61. {doc}`Draft & Verify：不训练的跳层草稿 <generated/papers/C/02-inference-serving/C032-draft-and-verify>`
62. {doc}`LayerSkip：训练时早退与共享预测头 <generated/papers/C/02-inference-serving/C033-layerskip>`
63. {doc}`BiLD：大小模型的 Fallback 与 Rollback <generated/papers/C/02-inference-serving/C034-bild>`

先比较单一模型跳过若干层的 Draft & Verify，与必须在训练时施加 Early Exit Loss / Layer Dropout 的 LayerSkip。最后用 BiLD 的回退和回滚策略区分两种优化命题：“输出分布严格等于 Target”与“允许任务质量差异换取时延改善”；speedup 都应附带真实质量、采样设置与硬件基线。

## 第三十二阶段：KV Cache 的容量与语义保留

64. {doc}`StreamingLLM：为何要保留 Attention Sink <generated/papers/C/02-inference-serving/C035-streamingllm>`
65. {doc}`H2O：在线选择 Heavy Hitter KV <generated/papers/C/02-inference-serving/C036-h2o>`
66. {doc}`SnapKV：用 Prompt 末端观察窗口选关键 KV <generated/papers/C/02-inference-serving/C037-snapkv>`
67. {doc}`KIVI：Key 按通道、Value 按 Token 的低比特 KV <generated/papers/C/02-inference-serving/C038-kivi>`

沿着同一内存约束分清三种方案：保留 Sink 维持长期 Streaming 稳定性；使用在线注意力统计或预生成观察窗口有损驱逐不重要的 KV；保持位置信息而对 KV 做非对称低比特量化。后两类都会引入精度风险，不可把显存缩减和完整长文推理能力画等号。

## 第三十三阶段：激活异常值与权重低比特

68. {doc}`LLM.int8：向量量化 + FP16 异常通道 <generated/papers/C/02-inference-serving/C039-llm-int8>`
69. {doc}`SmoothQuant：等价迁移通道尺度实现 W8A8 <generated/papers/C/02-inference-serving/C040-smoothquant>`
70. {doc}`GPTQ：二阶信息与误差补偿的低比特权重 <generated/papers/C/02-inference-serving/C041-gptq>`
71. {doc}`AWQ：Activation-aware 显著通道缩放 <generated/papers/C/02-inference-serving/C042-awq>`

从 $Y=XW$ 出发：为什么巨大激活会吞噬量化动态范围、LLM.int8 如何把异常特征分流、SmoothQuant 如何用 $XD^{-1}DW$ 保持浮点结果不变，最后理解只压缩权重时 GPTQ 的二阶输出误差与 AWQ 的显著性保护。浮点代数等价、量化后的精度与 GPU 实际吞吐是三个不同层面。

## 第三十四阶段：低显存卸载与低比特服务系统

72. {doc}`FlexGen：GPU/CPU/SSD 分层、LP 调度与 Zig-Zag 执行 <generated/papers/C/02-inference-serving/C043-flexgen>`
73. {doc}`Atom：W4A4、混精度异常通道与 INT4 Kernel <generated/papers/C/02-inference-serving/C044-atom>`

将单张 GPU 放不下模型的内存与传输约束，与在线 Batch 推理中低比特算力利用率的约束分开研究：FlexGen 偏向资源有限、可容忍高延迟的离线吞吐，Atom 面向满足延迟目标的低比特 GPU 服务。不同系统和基线的 headline speedup 不能直接排名。

## 第三十五阶段：量化校准与异常权重的处理策略

74. {doc}`ZeroQuant：硬件友好量化、LKD 与 Kernel 融合 <generated/papers/C/02-inference-serving/C045-zeroquant>`
75. {doc}`RPTQ：按激活动态范围聚类重排 Channel <generated/papers/C/02-inference-serving/C046-rptq>`
76. {doc}`OmniQuant：Learnable Weight Clipping 与等价变换 <generated/papers/C/02-inference-serving/C047-omniquant>`
77. {doc}`SpQR：稀疏例外与低比特密集权重协同存储 <generated/papers/C/02-inference-serving/C048-spqr>`

这一阶段以量化误差来自何处为主线：ZeroQuant 综合校准、逐层蒸馏和整数 Kernel；RPTQ 针对各 Channel 量级错配作无损排列；OmniQuant 对缩放与裁剪参数进行校准优化；SpQR 则对极少数难以低比特表示的权重设置高精度稀疏例外。每篇都要对照浮点等价、量化后误差和硬件实际吞吐这三类不同问题。

## 第三十六阶段：极低比特下的向量与码本表达

78. {doc}`QuIP#：Hadamard Incoherence 与 E8 格码本 <generated/papers/C/02-inference-serving/C049-quip-sharp>`
79. {doc}`AQLM：多码本相加的极限压缩 <generated/papers/C/02-inference-serving/C050-aqlm>`

QuIP# 通过随机正交变换降低单坐标集中性，再采用适合向量分布的格码本；AQLM 则通过多个码本加和与输入感知重建进一步利用超低比特。学习时必须将码本的 nominal bit 与模型总有效 bit、索引/Scale 元信息以及 Kernel 解码开销分开。

## 第三十七阶段：量化微调与从头低比特预训练

80. {doc}`QLoRA：NF4、Double Quantization 和 Paged Optimizer <generated/papers/C/02-inference-serving/C051-qlora>`
81. {doc}`LoftQ：同时量化主干并初始化低秩修正 <generated/papers/C/02-inference-serving/C052-loftq>`
82. {doc}`BitNet：在低比特约束下从头训练语言模型 <generated/papers/C/02-inference-serving/C053-bitnet>`
83. {doc}`BitNet b1.58：三值权重和信息编码极限 <generated/papers/C/02-inference-serving/C054-bitnet-b158>`

这里刻意将「固定低比特主干，只更新 Adapter」与「预训练时就实施量化约束」放在两条相邻但不同的路线：QLoRA 减少微调时的主干存储，LoftQ 尝试用低秩初始化修补量化误差；BitNet 系列在训练阶段直接改变权重所能采取的离散状态，从而不再依赖事后压缩任意 FP16 模型。

读到 V3 / R1 时不需要把所有方法重新推一遍；直接使用文章中的站内链接回到 MLA、MoE 或 GRPO 专题，再返回模型主线即可。

## 第三十八阶段：从随机旋转到可学习仿射的 4-bit 推理

84. {doc}`QuaRot：随机正交重参数化与 W4A4KV4 <generated/papers/C/02-inference-serving/C055-quarot>`
85. {doc}`SpinQuant：用 Cayley 保持正交并直接优化量化损失 <generated/papers/C/02-inference-serving/C056-spinquant>`
86. {doc}`FlatQuant：用 Kronecker 低成本可学习仿射平坦化激活 <generated/papers/C/02-inference-serving/C057-flatquant>`

在 {doc}`SmoothQuant <generated/papers/C/02-inference-serving/C040-smoothquant>` 的缩放与 {doc}`QuIP# <generated/papers/C/02-inference-serving/C049-quip-sharp>` 的 Hadamard 基础上，首先证明正交变换怎样保持 Transformer 浮点输出不变，但其量化误差与选取的坐标系有关；再研究正交流形上的可训练旋转与校准任务损失；最后解释为什么每层独立可学习的一般可逆变换需要 Kronecker 分解和 Kernel Fusion 才能实际部署。比较论文结果时，必须先对齐模型、位宽 W/A/KV、Weight Quantizer、评测数据、Batch 和硬件，不能把 Table 1 的 PPL 和 Figure 的 Kernel Speedup 当同一个指标。

## 第三十九阶段：定向消除极端异常值与低比特云端服务

87. {doc}`DuQuant：块内贪心旋转、Zigzag 重排与二次旋转 <generated/papers/C/02-inference-serving/C058-duquant>`
88. {doc}`QServe：Progressive Group Quantization 与 W4A8KV4 INT8 Tensor Core <generated/papers/C/02-inference-serving/C059-qserve>`

DuQuant 对照 QuaRot 的随机旋转，区分跨 Token 持续存在的 Normal Outliers 与只出现于少数位置的 Massive Outliers，解释为何缩放会将异常值转移到 Weight，以及第二次旋转为何依赖 Zigzag 的跨 Block 均衡。QServe 不再只讨论同位宽 PPL，转向真正的 Tensor Core 主循环：为什么 Reduction Dimension 的 Group Scale 会让 INT4 局部累加频繁反量化，如何采用 W4A8KV4 的 Packed Weight、INT8 Tensor Core、保护性编码、SmoothAttention 与 KV4 内核共同降低成本。前者的 W4A4 质量数据和后者的 W4A8KV4 Token/s 不属于同一项可直接排序的指标。

## 第四十阶段：从线性时不变状态空间到具有内容选择能力的序列模型

89. {doc}`S4：从连续状态方程、HiPPO 记忆到 DPLR/Cauchy 卷积核 <generated/papers/A/09-state-space/A058-s4>`
90. {doc}`Mamba：S6 的 $\Delta,B,C$ 输入选择机制与 Selective Scan <generated/papers/A/09-state-space/A059-mamba>`
91. {doc}`Mamba-2：Semiseparable Matrix 与 Structured State-Space Duality <generated/papers/A/09-state-space/A060-mamba2-ssd>`

本阶段从 {doc}`Transformer <generated/papers/A/03-transformer/A013-transformer-attention-is-all-you-need>` 的显式历史读取出发，研究另一种根本不同的记忆观：将整个历史压缩成固定维度状态。S4 借 LTI 同时获得递推与卷积两种等价计算；Mamba 通过使离散步长和输入/输出投影依赖 Token，学会按内容选择历史，却失去固定卷积核；Mamba-2 再利用状态递推对应的 Semiseparable 结构矩阵，为一个特定子类恢复 Attention-like 矩阵形式。须始终区分 SSD 与标准 Softmax Attention 的数学边界，不能把“某些结构化 Attention 的对偶”称为所有 Transformer 的精确等价。

## 第四十一阶段：并行训练与递推记忆的三种模型架构

92. {doc}`RetNet：Multi-Scale Retention、Parallel/Recurrent/Chunk 三种计算表示 <generated/papers/A/09-state-space/A061-retnet>`
93. {doc}`RWKV-4：Token Shift、带当前 Token Bonus 的归一化 WKV 和 Receptance <generated/papers/A/09-state-space/A062-rwkv>`
94. {doc}`xLSTM：指数门控的 sLSTM 与矩阵记忆的 mLSTM <generated/papers/A/09-state-space/A063-xlstm>`

继 [S4 → Mamba → Mamba-2](state-space/index.md) 后，这三篇以**并行可训练且具有固定状态的序列模型骨干**为共同问题，却沿不同技术思路独立发展，不能被误写成严格的前后继承顺序。RetNet 的核心是具有 Query–Key 内容匹配和多尺度衰减的外积状态；RWKV-4 是逐通道归一化分子/分母与 Token Shift/Receptance 的组合；xLSTM 则比较 sLSTM 的 Hidden-to-Hidden Mixing 与 mLSTM 的输入相关 Gate 和 Matrix Memory。三篇都应明确区分“浮点数学表达等价”“训练的批处理方式”和“有限状态记忆能否精确检索历史”，不能把任意一篇的吞吐数直接推广到其他架构。

## 第四十二阶段：混合模型架构中的显式读取、压缩状态与容量分配

95. {doc}`Griffin：RG-LRU 的实数门控，Hawk 与 Local-Attention Hybrid 的区别 <generated/papers/A/09-state-space/A064-griffin>`
96. {doc}`Jamba-v0.1：Attention/Mamba 层比例与 MoE 专家密度的独立组合 <generated/papers/B/05-moe-complete-llm/B013-jamba>`
97. {doc}`Zamba-7B-v1：Mamba Backbone 与跨深度共享全局 Attention/MLP <generated/papers/B/05-moe-complete-llm/B014-zamba>`

承接 S4、Mamba、Mamba-2、RetNet、RWKV 和 xLSTM，下一层问题不是“再发明一个固定状态”，而是**怎样在真实大模型里分配不同历史读取机制的执行次数与参数容量**。Griffin 设计具有输入相关遗忘门的 RG-LRU 并与 Local Attention 组合；Jamba 将 Attention/Mamba 的层比例与 MoE Expert 的稀疏激活正交拆开；Zamba 则允许重复调用一个共享 Attention/MLP，但各调用位置仍独立保存 KV。这三条架构**并非单线继承**。特别要区分 RG-LRU 的近似状态保护、Jamba 的 1:7 是层级而非 Head 比例、Zamba 的共享 Weight 不等于共享 KV；使用不同训练 Token、模型规模、数据和 Benchmark 的论文结论时不得直接排名。

## 第四十三阶段：从自回归语言模型转向离散扩散生成范式

98. {doc}`D3PM：类别转移矩阵、Mask 吸收态与反向 Markov 后验 <generated/papers/A/11-discrete-diffusion/A065-d3pm>`
99. {doc}`MDLM：SUBS、Rao-Blackwellized NELBO 与可生成文本的 Masked LM <generated/papers/A/11-discrete-diffusion/A066-mdlm>`
100. {doc}`LLaDA：8B 双向 Transformer 的 Mask-Diffusion 预训练和指令微调 <generated/papers/B/05-moe-complete-llm/B015-llada>`

这一阶段有别于之前的 S4/Mamba/RetNet 等状态空间与记忆架构。它首先追问：**为什么语言生成必须按从左到右的 Token 条件概率推进？** D3PM 为离散词表构造随机扰动矩阵并推导真实后验，MDLM 抓住 Mask 吸收态与不可逆拷贝性质消去不必要的 KL 项，LLaDA 则在 8B 规模重新组合双向 Transformer、全句去噪、Prompt-Response SFT 和低置信再遮盖采样。读这三篇要严格区分 D3PM 的 text8 bits/char 上界、MDLM 的 LM1B PPL 上界、LLaDA 的基准任务准确率和推理速度，不应把完全不同的指标跨论文直接计算收益率。
