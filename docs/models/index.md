# 现代完整模型主线

把方法重新放回完整模型。第一条主线用 DeepSeek-V2 → V3 → R1 理解 MLA、MoE、训练系统与 reasoning RL；随后用 Llama 3 与 Qwen2.5 建立第二组参照：相对标准的 dense Transformer 如何依赖 data、scale、长上下文训练与 staged post-training 扩展。第三条路线是**混合序列架构**：Jamba 将 Mamba 与 Attention 按层比例组合，并加入稀疏专家；Zamba 则让独立 Mamba Backbone 在多个深度调用共享 Attention/MLP 权重。它们与 {doc}`Griffin 的 RG-LRU <../generated/papers/A/09-state-space/A064-griffin>` 形成三种模型设计取舍。

**第四条主线是生成概率范式的变化**：{doc}`LLaDA 8B <../generated/papers/B/05-moe-complete-llm/B015-llada>` 使用双向 Transformer 训练离散 Mask 扩散模型，先 Mask 全句再迭代修复，预训练与 SFT 仍保持统一的概率思想。它的数学基础可回到 {doc}`离散扩散语言模型专题 <../diffusion-language/index>`；与自回归模型的区别不是“Transformer 已被新的注意力算子取代”，而是训练/生成时的条件概率组织发生了变化。

```{toctree}
:maxdepth: 2

../generated/papers/B/05-moe-complete-llm/B008-deepseek-v2
../generated/papers/B/05-moe-complete-llm/B009-deepseek-v3
../generated/papers/B/05-moe-complete-llm/B010-deepseek-r1
../generated/papers/B/05-moe-complete-llm/B011-llama3
../generated/papers/B/05-moe-complete-llm/B012-qwen2.5
../generated/papers/B/05-moe-complete-llm/B013-jamba
../generated/papers/B/05-moe-complete-llm/B014-zamba
../generated/papers/B/05-moe-complete-llm/B015-llada
```
