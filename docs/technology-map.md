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
