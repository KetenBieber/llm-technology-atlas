# 技术图谱

## 第一条主干：从 Transformer 到 DeepSeek-R1

```{mermaid}
flowchart TD
  T["Transformer"] --> R["RoPE"]
  T --> N["RMSNorm"]
  R --> YARN["YaRN"]
  YARN --> DCA["Dual Chunk Attention"]
  DCA --> MINF["MInference"]
  FA["FlashAttention"] --> DCA
  FA --> MINF
  FA --> FA2["FlashAttention-2"]
  FA2 --> SYS4D["4D Parallelism"]
  SYS4D --> ZERO["ZeRO"]
  SYS4D --> GPIPE["GPipe"]
  GPIPE --> MEGA["Megatron-LM"]
  MEGA --> RING["Ring Attention"]
  FA2 --> RING
  GPIPE --> PD["PipeDream"]
  PD --> PD2["PipeDream-2BW"]
  ZERO --> ZO["ZeRO-Offload"]
  ZO --> ZI["ZeRO-Infinity"]
  RING --> ULYSSES["DeepSpeed-Ulysses"]
  ULYSSES --> DFA["DistFlashAttn"]
  RING --> DFA
  V2 --> VL["vLLM / PagedAttention"]
  VL --> ORCA["Orca"]
  ORCA --> SARATHI["Sarathi-Serve"]
  SARATHI --> DSERVE["DistServe"]
  DSERVE --> SPLIT["Splitwise"]
  VL --> SGL["SGLang"]
  SGL --> FINFER["FlashInfer"]
  VL --> PUN["Punica"]
  VL --> SPEC["Speculative Decoding"]
  SPEC --> MEDUSA["Medusa"]
  SPEC --> EAGLE["EAGLE"]
  EAGLE --> EAGLE2["EAGLE-2：动态草稿树"]
  EAGLE2 --> EAGLE3["EAGLE-3：训练时多步展开"]
  MEDUSA --> HYDRA["Hydra：顺序依赖预测头"]
  HYDRA --> REDRAFTER["ReDrafter：循环草稿"]
  REDRAFTER --> SPECINFER["SpecInfer：树形验证"]
  SPECINFER --> SEQUOIA["Sequoia：树形预算"]
  SPEC --> LOOKAHEAD["Lookahead：Jacobi 候选"]
  LOOKAHEAD --> REST["REST：语料检索"]
  REST --> OUROBOROS["Ouroboros：短语草稿"]
  SPEC --> DVERIFY["Draft & Verify：跳层草稿"]
  DVERIFY --> LSKIP["LayerSkip：早退训练"]
  SPEC --> BILD["BiLD：回退与回滚，非严格无损"]
  VL --> STREAM["StreamingLLM：Attention Sink"]
  STREAM --> H2O["H2O：Heavy Hitter KV 驱逐"]
  H2O --> SNAPKV["SnapKV：Prompt 尾部预测"]
  SNAPKV --> KIVI["KIVI：不对称 2-bit KV"]
  GQA --> KIVI
  VL --> LLM8["LLM.int8：异常通道混精度"]
  LLM8 --> SMOOTH["SmoothQuant：迁移激活尺度"]
  SMOOTH --> GPTQ["GPTQ：二阶补偿权重量化"]
  GPTQ --> AWQ["AWQ：显著通道保护"]
  AWQ --> ATOM["Atom：W4A4 在线服务"]
  VL --> FLEX["FlexGen：GPU/CPU/SSD 卸载"]
  FLEX --> ATOM
  SMOOTH --> ZEROQUANT["ZeroQuant：LKD 与硬件友好 W8A8"]
  ZEROQUANT --> RPTQ["RPTQ：Channel Range 重排"]
  RPTQ --> OMNIQUANT["OmniQuant：LET 与 LWC"]
  GPTQ --> SPQR["SpQR：少量高精度例外"]
  SPQR --> QUIP["QuIP#：Hadamard 与 E8 Codebook"]
  QUIP --> AQLM["AQLM：可学习加法码本"]
  AWQ --> QLORA["QLoRA：NF4 + Adapter"]
  QLORA --> LOFTQ["LoftQ：量化误差低秩初始化"]
  QLORA --> BITNET["BitNet：二值预训练（方法路径对照）"]
  BITNET --> BIT158["BitNet b1.58：三值权重"]
  SMOOTH --> QUAROT["QuaRot：随机旋转 W4A4KV4"]
  QUIP --> QUAROT
  QUAROT --> SPINQUANT["SpinQuant：正交流形学习"]
  SPINQUANT --> FLATQUANT["FlatQuant：Kronecker 可逆仿射"]
  OMNIQUANT --> FLATQUANT
  T --> MQA["MQA"]
  MQA --> GQA["GQA"]

  SG["Sparsely-Gated MoE"] --> SW["Switch Transformer"]
  SW --> DSM["DeepSeekMoE"]

  T --> V2["DeepSeek-V2"]
  GQA --> V2
  R --> V2
  DSM --> V2
  V2 --> MLA["MLA"]

  V2 --> V3["DeepSeek-V3"]
  DSM --> V3
  V3 --> R1["DeepSeek-R1"]

  L3 --> SYS4D
  GRPO["DeepSeekMath / GRPO"] --> V3
  GRPO --> R1

  T --> L3["Llama 3"]
  R --> L3
  GQA --> L3

  T --> Q25["Qwen2.5"]
  R --> Q25
  GQA --> Q25
  DSM --> Q25
  GRPO --> Q25
  DPO["Direct Preference Optimization"] --> L3
  YARN --> Q25
  DCA --> Q25
  MINF --> Q25
  DPO --> Q25

  KAP["Kaplan Scaling Laws"] --> CH["Chinchilla"]
  CH --> L3
  CH --> Q25
```

## 第二条主干：标准 Dense Transformer 的规模化路线

```{mermaid}
flowchart LR
  T["Transformer"] --> GQA["GQA"]
  T --> R["RoPE"]
  CH["Chinchilla / IsoFLOPs"] --> L3
  R --> L3
  DATA["Data engineering"] --> L3
  SCALE["Scaling laws"] --> L3
  L3 --> LC["128K continued pre-training"]
  L3 --> PT["RM → RS → SFT → DPO"]
  L3 --> SYS["4D parallelism / FP8 inference"]
  GQA --> Q25["Qwen2.5"]
  R --> Q25
  Q25 --> FAMILY["0.5B → 72B dense + MoE services"]
  Q25 --> QPT["SFT → Offline DPO → Online GRPO"]
  Q25 --> QLC["32K / 262K training → 128K / 1M serving"]
  DPO["DPO"] --> PT
  DPO --> QPT
```

## 快速跳转

| 想解决的问题 | 推荐文章 |
|---|---|
| 为什么量化策略、低成本蒸馏和 Integer Kernel 必须一起设计？ | {doc}`ZeroQuant <generated/papers/C/02-inference-serving/C045-zeroquant>` |
| Channel 重排和校准学习分别怎样减少 Activation 与低 bit 误差？ | {doc}`RPTQ <generated/papers/C/02-inference-serving/C046-rptq>` → {doc}`OmniQuant <generated/papers/C/02-inference-serving/C047-omniquant>` |
| 为什么少数权重例外存 FP16 能比全部权重提高位宽更划算？ | {doc}`SpQR <generated/papers/C/02-inference-serving/C048-spqr>` |
| Hadamard 正交混合、E8 码本与加法多码本有什么差别？ | {doc}`QuIP# <generated/papers/C/02-inference-serving/C049-quip-sharp>` → {doc}`AQLM <generated/papers/C/02-inference-serving/C050-aqlm>` |
| 低显存微调冻结量化主干，能否让 Adapter 先补偿量化误差？ | {doc}`QLoRA <generated/papers/C/02-inference-serving/C051-qlora>` → {doc}`LoftQ <generated/papers/C/02-inference-serving/C052-loftq>` |
| 从头训练 1-bit/三值模型与对 FP16 模型做 PTQ 有何根本不同？ | {doc}`BitNet <generated/papers/C/02-inference-serving/C053-bitnet>` → {doc}`BitNet b1.58 <generated/papers/C/02-inference-serving/C054-bitnet-b158>` |
| 为什么随机正交旋转能改善 W4A4KV4，怎样学习正交旋转并用 Kronecker 仿射控制成本？ | {doc}`QuaRot <generated/papers/C/02-inference-serving/C055-quarot>` → {doc}`SpinQuant <generated/papers/C/02-inference-serving/C056-spinquant>` → {doc}`FlatQuant <generated/papers/C/02-inference-serving/C057-flatquant>` |
| 长对话只保留最近 KV 为什么会失稳？ | {doc}`StreamingLLM <generated/papers/C/02-inference-serving/C035-streamingllm>` |
| 保留哪些旧 Token？在线 Heavy Hitter 与 Prompt 预观察有何差别？ | {doc}`H2O <generated/papers/C/02-inference-serving/C036-h2o>` → {doc}`SnapKV <generated/papers/C/02-inference-serving/C037-snapkv>` |
| 不删除位置如何把 KV 降到 2-bit？ | {doc}`KIVI <generated/papers/C/02-inference-serving/C038-kivi>` |
| Activation Outlier 为什么让 INT8 失败，能否用等价变换消解？ | {doc}`LLM.int8 <generated/papers/C/02-inference-serving/C039-llm-int8>` → {doc}`SmoothQuant <generated/papers/C/02-inference-serving/C040-smoothquant>` |
| GPTQ 与 AWQ 都利用 Calibration Activation，有何不同？ | {doc}`GPTQ <generated/papers/C/02-inference-serving/C041-gptq>` → {doc}`AWQ <generated/papers/C/02-inference-serving/C042-awq>` |
| 单 GPU 显存装不下模型与在线 INT4 GEMM 吞吐不足，如何分别解决？ | {doc}`FlexGen <generated/papers/C/02-inference-serving/C043-flexgen>` → {doc}`Atom <generated/papers/C/02-inference-serving/C044-atom>` |
| 多头为何需顺序条件，循环草稿怎样结合 Beam？ | {doc}`Hydra <generated/papers/C/02-inference-serving/C025-hydra>` → {doc}`ReDrafter <generated/papers/C/02-inference-serving/C026-redrafter>` |
| 多条路径怎样共享前缀，固定预算下怎么规划候选树？ | {doc}`SpecInfer <generated/papers/C/02-inference-serving/C027-specinfer>` → {doc}`Sequoia <generated/papers/C/02-inference-serving/C028-sequoia>` |
| 不训练独立草稿网络能否用 Jacobi 或检索找 Token？ | {doc}`Lookahead <generated/papers/C/02-inference-serving/C029-lookahead-decoding>` → {doc}`REST <generated/papers/C/02-inference-serving/C030-rest>` |
| 用短语加快 Draft 自身生成与延长候选的思路是什么？ | {doc}`Ouroboros <generated/papers/C/02-inference-serving/C031-ouroboros>` |
| 不重新训练的跳层草稿，与专门训练的早退模型有何差别？ | {doc}`Draft & Verify <generated/papers/C/02-inference-serving/C032-draft-and-verify>` → {doc}`LayerSkip <generated/papers/C/02-inference-serving/C033-layerskip>` |
| BiLD 何时回退回滚，为什么不自动等于严格分布不变？ | {doc}`BiLD <generated/papers/C/02-inference-serving/C034-bild>` |
| 固定候选树为何浪费 Target 验证预算，置信度如何变成路径排序分数？ | {doc}`EAGLE-2 <generated/papers/C/02-inference-serving/C023-eagle2>` |
| 为什么下一 Feature 的回归约束可能限制草稿效果，怎样训练多步自回灌的 Draft？ | {doc}`EAGLE-3 <generated/papers/C/02-inference-serving/C024-eagle3>` |
| 数据并行下为什么要分片 Adam optimizer、gradient、parameter？ZeRO-1/2/3 与 FSDP 怎么联系？ | {doc}`ZeRO 原始论文 <generated/papers/C/01-distributed-training/C002-zero>` |
| GPipe 为什么必须拆 micro-batch，怎样保留同步梯度与计算 bubble？ | {doc}`GPipe <generated/papers/C/01-distributed-training/C003-gpipe>` |
| 张量并行为什么高频 all-reduce，interleaved 1F1B 怎样改变 bubble 和通信？ | {doc}`Megatron-LM <generated/papers/C/01-distributed-training/C004-megatron-lm>` |
| FlashAttention 后为何还需要跨设备搬 KV，Ring Attention 如何保证精确与通信重叠？ | {doc}`Ring Attention <generated/papers/C/01-distributed-training/C005-ring-attention>` |
| 为什么取消同步 flush 后需要 Weight Stashing？怎样将版本限制为两个？ | {doc}`PipeDream <generated/papers/C/01-distributed-training/C006-pipedream>` → {doc}`PipeDream-2BW <generated/papers/C/01-distributed-training/C007-pipedream-2bw>` |
| CPU optimizer 与 NVMe 异步预取分别解决什么？ | {doc}`ZeRO-Offload <generated/papers/C/01-distributed-training/C008-zero-offload>` → {doc}`ZeRO-Infinity <generated/papers/C/01-distributed-training/C009-zero-infinity>` |
| Ulysses 为什么做两次 All-to-all，DistFlashAttn 为什么要平衡 Causal workload？ | {doc}`DeepSpeed-Ulysses <generated/papers/C/01-distributed-training/C010-deepspeed-ulysses>` → {doc}`DistFlashAttn <generated/papers/C/01-distributed-training/C011-distflashattn>` |
| 推理服务怎样减少动态 KV 碎片并共享 Prefix？ | {doc}`PagedAttention / vLLM <generated/papers/C/01-distributed-training/C012-vllm-pagedattention>` |
| TP / PP / CP / FSDP 为什么要同时存在，4D process groups 怎样映射到网络拓扑 | {doc}`4D Parallelism <generated/papers/C/01-distributed-training/C001-4d-parallelism>` |
| Attention 为什么能替代循环网络 | {doc}`Transformer <generated/papers/A/03-transformer/A013-transformer-attention-is-all-you-need>` |
| 旋转位置编码到底改变了什么 | {doc}`RoPE <generated/papers/A/03-transformer/A014-roformer-rope>` |
| 为什么现代 Decoder 常用 RMSNorm | {doc}`RMSNorm <generated/papers/A/03-transformer/A016-rmsnorm>` |
| RoPE 为什么不能天然无限外推，PI / NTK / YaRN 又分别解决什么 | {doc}`YaRN <generated/papers/A/03-transformer/A018-yarn>` |
| 为什么 YaRN 之后仍要重组同 chunk / 相邻 chunk / 远 chunk 的相对位置 | {doc}`Dual Chunk Attention <generated/papers/A/04-efficient-attention/A053-dual-chunk-attention>` |
| 百万 token prompt 为什么 TTFT 爆炸，动态稀疏 Attention 又怎样真正映射到 GPU kernel | {doc}`MInference 1.0 <generated/papers/A/04-efficient-attention/A054-minference>` |
| 为什么 dense Attention 不少算任何 QK pair，也能靠 SRAM tiling / online softmax / recomputation 显著加速 | {doc}`FlashAttention <generated/papers/A/10-gpu-operators/A055-flashattention>` |
| FlashAttention 已经 IO-aware，为什么还能通过 sequence parallelism 与 split-Q 再快约 2× | {doc}`FlashAttention-2 <generated/papers/A/10-gpu-operators/A056-flashattention2>` |
| KV Cache 为什么成为 Decode 瓶颈 | {doc}`MQA <generated/papers/A/04-efficient-attention/A019-mqa>` / {doc}`GQA <generated/papers/A/04-efficient-attention/A020-gqa>` |
| MoE 为什么能扩大总容量但控制 active compute | {doc}`Sparsely-Gated MoE <generated/papers/A/05-moe/A032-sparsely-gated-moe>` |
| 为什么要细粒度专家 + shared expert | {doc}`DeepSeekMoE <generated/papers/A/05-moe/A034-deepseekmoe>` |
| MLA 为什么能只缓存 latent | {doc}`DeepSeek-V2 <generated/papers/B/05-moe-complete-llm/B008-deepseek-v2>` |
| V3 怎样把模型算法与训练系统拼起来 | {doc}`DeepSeek-V3 <generated/papers/B/05-moe-complete-llm/B009-deepseek-v3>` |
| R1-Zero 与最终 R1 到底是什么关系 | {doc}`DeepSeek-R1 <generated/papers/B/05-moe-complete-llm/B010-deepseek-r1>` |
| GRPO 为什么不训练 Critic | {doc}`DeepSeekMath / GRPO <generated/papers/A/08-post-training/A051-deepseekmath-grpo>` |
| DPO 怎样把 KL-regularized RLHF 消元成 preference classification | {doc}`Direct Preference Optimization <generated/papers/A/08-post-training/A052-dpo>` |
| 为什么小模型实验可以预测大模型、固定 FLOPs 下 N/D 又怎样分配 | {doc}`Kaplan Scaling Laws <generated/papers/A/02-text-representation/A009-kaplan-scaling-laws>` → {doc}`Chinchilla <generated/papers/A/02-text-representation/A010-chinchilla-compute-optimal>` |
| 为什么标准 Dense Transformer 仍能扩到 405B，并把 8K 扩到 128K | {doc}`Llama 3 <generated/papers/B/05-moe-complete-llm/B011-llama3>` |
| 18T 数据、多尺寸模型、Offline DPO / Online GRPO 与 1M context 怎样组成一个模型家族 | {doc}`Qwen2.5 <generated/papers/B/05-moe-complete-llm/B012-qwen2.5>` |
