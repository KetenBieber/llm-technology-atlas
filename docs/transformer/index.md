# Transformer、位置外推与高效注意力

先建立标准 Transformer，再理解位置编码、归一化、KV Cache 压缩；随后进入长上下文的三个不同层次：YaRN 处理位置外推，Dual Chunk Attention 重组跨 chunk 的相对位置关系，MInference 则把问题推进到百万 token prompt 的 prefill / TTFT runtime。

```{toctree}
:maxdepth: 2

../generated/papers/A/03-transformer/A013-transformer-attention-is-all-you-need
../generated/papers/A/03-transformer/A014-roformer-rope
../generated/papers/A/03-transformer/A016-rmsnorm
../generated/papers/A/03-transformer/A018-yarn
../generated/papers/A/04-efficient-attention/A019-mqa
../generated/papers/A/04-efficient-attention/A020-gqa
../generated/papers/A/04-efficient-attention/A053-dual-chunk-attention
../generated/papers/A/04-efficient-attention/A054-minference
```
