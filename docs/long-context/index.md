# 长上下文：位置外推、Attention 组织与运行时

这一部分不把“支持 128K / 1M”视为单一能力，而是拆成三层：**位置编码如何离开训练长度、Attention relation 如何组织、超长 prompt 如何以可接受的 TTFT 运行**。

当前主线已经形成三层：YaRN 处理 RoPE 位置频率外推；Dual Chunk Attention 处理同 chunk / 相邻 chunk / 远 chunk 的 relative-position relation；MInference 进一步处理超长 prompt 的 dense prefill O(L²) 与 TTFT，用动态结构化稀疏 Attention 将“能处理长上下文”推进到“能更快地实际运行”。

系统分支还包括 {doc}`Ring Attention：跨设备精确长上下文 Attention <../generated/papers/C/01-distributed-training/C005-ring-attention>`。它不是位置编码外推，也不是动态稀疏近似，而是把 Query block 的计算所有权固定在本机，让 KV blocks 在设备之间环形移动以隐藏通信。

与 Ring 的 KV blocks 顺序传送不同，{doc}`DeepSpeed-Ulysses <../generated/papers/C/01-distributed-training/C010-deepspeed-ulysses>` 用两次 All-to-all 在 sequence/head 分片间切换，{doc}`DistFlashAttn <../generated/papers/C/01-distributed-training/C011-distflashattn>` 则解决因果负载不均、通信与反向重算共同造成的瓶颈。

```{toctree}
:maxdepth: 2

../generated/papers/A/03-transformer/A018-yarn
../generated/papers/A/04-efficient-attention/A053-dual-chunk-attention
../generated/papers/A/04-efficient-attention/A054-minference
```
