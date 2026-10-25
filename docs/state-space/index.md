# 状态空间与序列建模：S4 → Mamba → Mamba-2

这一专题研究的是**模型怎样表示与选择历史信息**，而非推理服务的调度、Tensor Core 内核优化或训练集群的分布式通信。它从控制理论中的状态演化方程出发，接上 LLM 的长程依赖、内容选择与 Attention-like 表达。

三篇以同一个问题连续推进：固定状态空间模型为什么可以等价为卷积；可变状态参数为什么能对 Token 做内容选择，却破坏了固定卷积核；以及怎样借助半可分结构矩阵，为一部分可变 SSM 恢复类似 Attention 的并行矩阵计算形式。

- **S4**：从连续 $\dot h=Ah+Bu$ 到离散递推、卷积、HiPPO/NPLR/Cauchy；回答“如何在固定状态里保持长程依赖并高效学习”。
- **Mamba**：从一维门控的严格推导进入 Selective SSM，解释为什么必须让 $\Delta,B,C$ 输入相关；回答“如何选择写入、忽略和读出历史”。
- **Mamba-2**：从 Token Mixing Matrix 的代数展开推导 Semiseparable Matrix、SSD；回答“为什么某些 SSM 可以兼具递推和类 Attention 的两种计算形式”。

建议先读 [Transformer 的 Self-Attention](../generated/papers/A/03-transformer/A013-transformer-attention-is-all-you-need.md) 与 [RMSNorm](../generated/papers/A/03-transformer/A016-rmsnorm.md)；数学需要时可回到离散状态空间方程，而不用先学习 GPU Kernel。本专题文章都保留原论文的 Figure/Algorithm/Table 的严格证据边界。

```{toctree}
:maxdepth: 2

../generated/papers/A/09-state-space/A058-s4
../generated/papers/A/09-state-space/A059-mamba
../generated/papers/A/09-state-space/A060-mamba2-ssd
```
