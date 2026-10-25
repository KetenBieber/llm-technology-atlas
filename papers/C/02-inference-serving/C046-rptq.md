# RPTQ：为什么只交换通道顺序就会影响量化精度？

> **原始论文**：[RPTQ: Reorder-based Post-training Quantization for Large Language Models](https://arxiv.org/abs/2304.01089)，arXiv 原始论文。


前置：[SmoothQuant](C040-smoothquant.md) / [ZeroQuant](C045-zeroquant.md)，后续：[OmniQuant](C047-omniquant.md)。

## 某一维没有异常值，为什么还是量化不准？

设 $X\in\mathbb R^{T\times d}$ 的三个通道值域分别为 $[-100,100]$、$[-0.1,0.1]$、$[-1,1]$。若整体使用 8-bit 对称 Scale $s=100/127\approx0.787$，第二个通道基本都会被舍入成 0。真正矛盾不仅是个别离群点，而是**不同特征维度使用同一个量化尺度**。

RPTQ 将数值范围相似的 Channel 聚类，调整顺序，使相邻的一组维度共享更匹配的 Scale。这里的 Reorder 不是对 Token 次序重新排序——改变自回归时间次序会直接破坏模型含义——而是**只重排特征通道轴**。

## 浮点输出为何仍然不变？

令 $P$ 是通道的排列矩阵，$PP^\top=I$，则线性层有严格等式

$$
XW=(XP)(P^\top W).
$$

对输入通道采取排列 $P$，对 Weight 沿乘法求和维度同步作逆排列，输出不变。但量化过程不同：令 $Q_G$ 是按连续 Group 共享 Scale 的量化器，则一般

$$
Q_G(XP)\ne Q_G(X)P.
$$

因为排列改变了哪些 Channel 落在同一个 Group 中，这正是分组量化精度变化的数学原因。浮点排列等价**不代表整数化后严格等价**。

~~~{mermaid}
flowchart LR
 X[原激活 X：各通道范围差异大] --> C[估计每个 Channel 的范围]
 C --> G[按范围聚类]
 G --> P[计算排列矩阵 P]
 P --> Q[相近通道共享组 Scale]
 W[原权重 W] --> IW[对应逆排列]
 Q --> Y[低比特 Linear]
 IW --> Y
~~~

*自制解释图，非论文原图。聚类依据、分组规模与 LayerNorm 融合应核对原论文实现。*

## 为什么论文还要讨论 LayerNorm 和算子融合？

如果每次 Linear 之前都新增一次真实 Tensor Shuffle，GPU 需要额外读写激活；量化节省的带宽可能又被 Reorder 吃掉。对于适用结构，可把排列与前置 LayerNorm 的通道仿射参数或后续 Weight Layout 预融合，避免独立执行某些排列 Kernel。可是在残差支路、旋转位置编码或 Attention Head 切分处不能未经检查任意穿越排列：必须保证张量各个消费者使用一致的坐标顺序。

特别要注意 RMSNorm 不含逐通道中心化，与 LayerNorm 的统计不同；可交换性需要分别从算子数学形式验证，不应只因为它们都叫 Normalization 就视为等价。

## 与其他 PTQ 的根本区别

[SmoothQuant](C040-smoothquant.md) 通过连续实数尺度 $D$ 改变两侧张量的范围；RPTQ 通过离散排列 $P$ 改变**共享 Scale 的 Channel 组合**；[GPTQ](C041-gptq.md) 利用二阶信息补偿舍入引起的输出差异。

作者在 [原始论文](https://arxiv.org/abs/2304.01089) 中探索低至 **3-bit Activation** 的情形，并在 OPT 等模型上测试压缩与推理精度。公开摘要提到特定配置下显存缩减最高约 80%，不应理解成全部权重、KV、激活在所有硬件上统一减少 80%。

一个最有价值的复现实验是固定同一 Group Size 和整数 Kernel，比较：不重排、随机重排、按范围聚类重排。这样才能区分效果究竟来自额外 Group Scale 还是“动态范围相近”这条设计假设。下一篇 [OmniQuant](C047-omniquant.md) 直接对量化尺度与剪裁策略做少量校准优化。
