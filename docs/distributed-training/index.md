# 分布式训练系统

这一专题研究：当模型、训练状态、长上下文 activation 已经无法由单张 GPU 承担时，怎样把一个训练任务拆到多张 GPU 上，同时控制显存、通信、pipeline bubble 与网络拓扑成本。

入口从 **4D Parallelism** 开始。这里不把 TP / PP / CP / FSDP 当作四个 API 记忆，而是分别追问它们切的是哪一个逻辑维度、产生什么 collective / point-to-point 通信，以及这些通信为什么必须映射到不同层级的物理互联。

```{toctree}
:maxdepth: 2

../generated/papers/C/01-distributed-training/C001-4d-parallelism
```
