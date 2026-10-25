# 现代完整模型主线

把方法重新放回完整模型。第一条主线用 DeepSeek-V2 → V3 → R1 理解 MLA、MoE、训练系统与 reasoning RL；随后用 Llama 3 与 Qwen2.5 建立第二组参照：相对标准的 dense Transformer 如何依赖 data、scale、长上下文训练与 staged post-training 扩展。第三条路线是**混合序列架构**：Jamba 将 Mamba 与 Attention 按层比例组合，并加入稀疏专家；Zamba 则让独立 Mamba Backbone 在多个深度调用共享 Attention/MLP 权重。它们与 {doc}`Griffin 的 RG-LRU <../generated/papers/A/09-state-space/A064-griffin>` 形成三种模型设计取舍。

```{toctree}
:maxdepth: 2

../generated/papers/B/05-moe-complete-llm/B008-deepseek-v2
../generated/papers/B/05-moe-complete-llm/B009-deepseek-v3
../generated/papers/B/05-moe-complete-llm/B010-deepseek-r1
../generated/papers/B/05-moe-complete-llm/B011-llama3
../generated/papers/B/05-moe-complete-llm/B012-qwen2.5
../generated/papers/B/05-moe-complete-llm/B013-jamba
../generated/papers/B/05-moe-complete-llm/B014-zamba
```
