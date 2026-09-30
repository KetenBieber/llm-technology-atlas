# Scaling Laws 与训练规划

这一组文章研究的不是某个 Transformer block，而是**怎样用小规模受控实验预测昂贵大模型训练，并在固定预算下分配模型参数、训练 token 与其他训练资源**。

先读 Kaplan，建立 (N/D/C) 的经验 scaling、数据瓶颈、critical batch 与 compute-efficient planning；再读 Chinchilla，看固定 FLOPs 下的 (N)–(D) 前沿为什么被重新估计，以及 IsoFLOP 方法如何成为现代旗舰模型设计工具。

```{toctree}
:maxdepth: 2

../generated/papers/A/02-text-representation/A009-kaplan-scaling-laws
../generated/papers/A/02-text-representation/A010-chinchilla-compute-optimal
```
