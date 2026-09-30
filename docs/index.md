# 大模型技术图谱

**LLM Technology Atlas** 是一套以论文与完整模型为主线的大模型技术文章站。

这里不按论文编号机械阅读，而是沿着真正的技术依赖组织内容：

- 从反向传播、优化器与 Transformer 建立基础；
- 再进入 RoPE、RMSNorm、MQA / GQA、MoE 等架构组件；
- 以 DeepSeek-V2 → V3 → R1 作为第一条完整现代模型主线；
- 以 Llama 3 → Qwen2.5 建立第二条完整模型参照，重点理解 dense scaling、数据工程、多尺寸模型家族、staged post-training 与长上下文系统；
- 遇到 MLA、DeepSeekMoE、GRPO 等关键节点时，直接跳回原始方法专题；
- 每篇文章内部保留“前置阅读 / 相关技术 / 后续阅读”和就地跳转，形成可连续浏览的技术网络。

## 推荐入口

- {doc}`学习路线 <reading-path>`：按顺序建立完整知识链。
- {doc}`技术图谱 <technology-map>`：按技术依赖快速跳转。
- {doc}`长上下文 <long-context/index>`：从 RoPE/YaRN → DCA → MInference，区分位置外推、Attention relation 与超长上下文 prefill runtime。
- {doc}`GPU / 算子与编译 <gpu-kernels/index>`：从 FlashAttention 开始理解 HBM/SRAM、IO-aware exact attention 与后续 GPU work partition。
- {doc}`分布式训练系统 <distributed-training/index>`：从 TP / PP / CP / FSDP 建立多 GPU 训练的 memory–communication–topology cost model。
- {doc}`Scaling Laws <scaling/index>`：从 Kaplan 到 Chinchilla，理解模型/数据/compute 的训练规划。
- {doc}`完整模型主线 <models/index>`：从 DeepSeek V2/V3/R1 对照到 Llama 3 与 Qwen2.5。
- {doc}`推理后训练 <post-training/index>`：从 PPO/GRPO 进入 reasoning RL。

```{toctree}
:maxdepth: 2
:caption: 开始

reading-path
technology-map
```

```{toctree}
:maxdepth: 3
:caption: 技术专题

foundations/index
transformer/index
long-context/index
gpu-kernels/index
distributed-training/index
scaling/index
moe/index
models/index
post-training/index
```
