# 长上下文：位置外推、Attention 组织与运行时

这一部分不把“支持 128K / 1M”视为单一能力，而是拆成三层：**位置编码如何离开训练长度、Attention relation 如何组织、超长 prompt 如何以可接受的 TTFT 运行**。

当前主线已经形成三层：YaRN 处理 RoPE 位置频率外推；Dual Chunk Attention 处理同 chunk / 相邻 chunk / 远 chunk 的 relative-position relation；MInference 进一步处理超长 prompt 的 dense prefill O(L²) 与 TTFT，用动态结构化稀疏 Attention 将“能处理长上下文”推进到“能更快地实际运行”。

```{toctree}
:maxdepth: 2

../generated/papers/A/03-transformer/A018-yarn
../generated/papers/A/04-efficient-attention/A053-dual-chunk-attention
../generated/papers/A/04-efficient-attention/A054-minference
```
