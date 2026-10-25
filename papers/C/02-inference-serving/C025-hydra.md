# Hydra：为什么 Medusa 的未来预测头需要顺序依赖？

> **原始论文**：[Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding](https://openreview.net/forum?id=FbhjirzvJG)，COLM 2024。**类型**：C 类推理机制与系统。 [原文 PDF](https://openreview.net/pdf?id=FbhjirzvJG)。

前置：[Medusa](C021-medusa.md) → [EAGLE](C022-eagle.md)。后续：[ReDrafter](C026-redrafter.md)。

## 一个看似微小、实际致命的条件缺失

[Medusa](C021-medusa.md) 从 Target 模型已知前缀的隐藏状态 $h_t$ 出发，以多个轻量预测头同时猜测未来不同位置。它节约了完整 Draft Transformer 的重复前向，但远处的预测头并不知道前面候选头到底选了什么：

$$
q_i(\hat x_{t+i}\mid x_{\le t})=\operatorname{softmax}(f_i(h_t)).
$$

真正的目标概率则是

$$
p(x_{t+i}\mid x_{\le t},x_{t+1:t+i-1}),
$$

即必须条件化实际经过的路径。例如“我想买一台”后若第一步是“电脑”，后续候选与第一步是“相机”就可能明显不同；独立头却无法直接随已选路径切换预测。这不是把 Head 参数量翻倍就必然能修好的问题，而是**信息条件缺失**。

## Hydra 如何改变张量流

Hydra 将前面已经采样的候选 Token 的 embedding 一起提供给后续预测头：

$$
q_i(\hat x_{t+i}\mid h_t,\hat x_{t+1:t+i-1})
=\operatorname{softmax}
f_{\phi,i}(h_t,E(\hat x_{t+1}),\ldots,E(\hat x_{t+i-1})).
$$

上式是条件依赖关系的教学简写，具体网络结构、Residual 连接及训练目标参见论文第 3 节。

~~~{mermaid}
flowchart LR
 H[Target 已知前缀 h] --> A[Hydra 头1]
 A --> X1[候选1]
 H --> B[Hydra 头2]
 X1 --> B
 B --> X2[候选2]
 H --> C[Hydra 头3]
 X1 --> C
 X2 --> C
 C --> X3[候选3]
~~~

*自制解释图，非原论文 Figure。原论文 [Figure 1](https://openreview.net/pdf?id=FbhjirzvJG) 左右对照了独立头与顺序依赖头的数据流。*

假设第二头在观察到第一个候选“相机”后，能偏向“镜头”；若第一个是“电脑”，则可以改为“配置”。这个小例子只演示条件分布，不是论文的实测 Token。

顺序依赖也有成本：候选 $K$ 步不能全部同时算完，粗略草稿时间为 $\sum_{i=1}^{K}t_i$；但每个 Head 的执行成本远低于完整 Target，若可显著延长连续接受前缀，端到端仍可能获益。

## 顺序条件依赖究竟增加了什么表达能力？

用一个小词表例子就能看出差别。已确认前缀为“我想买一台”，第一头候选可能是“电脑”或“相机”。假设后续词只有“显卡”和“镜头”，在两种历史下理想分布分别是

$$
p(\cdot\mid\text{电脑})=(0.9,0.1),\qquad
p(\cdot\mid\text{相机})=(0.1,0.9).
$$

独立的第二头只观察同一个 $h_t$，因此两条草稿分支使用同一分布 $q_2$。如果两类第一 Token 的出现机会各为 $1/2$，再假设目标分布就是上述两种，最简单的统一提案 $q_2=(0.5,0.5)$ 对每一分支的最大单步重合度为

$$
\alpha=\sum_x\min\{p(x),q_2(x)\}=0.6.
$$

若第二头能够观察第一步的候选，理想情况下分别输出 $(0.9,0.1)$ 与 $(0.1,0.9)$，单步重合度可提高到 $1$。这组数字是**为了证明独立头的信息缺失而构造的教学反例**，并非论文实际接受率；真实模型的顺序 Head 也不可能完全拟合所有 Target 条件分布。

### 代价并非简单增加模型参数

令每个 Hydra Head 的单步耗时为 $t_h$、一次 Target 树形验证为 $t_v$，预计可提交的 Token 数为 $A$，一轮平均成本约

$$
\bar t_{\mathrm{token}}=\frac{K t_h+t_v+t_{\mathrm{tree}}}{A}.
$$

改成依赖 Head 后，$A$ 可以因为更合适的条件分布而增大，但 Head 串行成本与树构建成本也会增大。因此 Hydra 的真正实验判据是：**减少的 Target 验证轮次是否覆盖额外串行计算？** 不能只看提案预测的 Top-1 Accuracy。

## Figure 1 之后还必须读什么？

[Hydra COLM 2024 Figure 1](https://openreview.net/forum?id=FbhjirzvJG) 的左图是多个未来 Head 只看前缀 Hidden State，右图给未来 Head 加上此前已选 Token 的输入。它说明信息路径不同，却不能单独证明 Hydra++ 的加速幅度。论文 §3 中明确要求区分：Heads 的连接方式、后续头的训练选择，以及综合调优的 Hydra++ 配置。看性能表时先固定相同 Base/Decode 任务，再看 Hydra 与 Medusa 的中间版本消融，最后比较端到端 Time per Output Token。

一个系统层面的反例是：即使新增顺序条件使每轮确认由 2 增至 3 个 Token，若串行 Head 使一轮用时从 10 ms 增至 20 ms，那么平均成本从 5 ms/token 变成约 6.67 ms/token，**预测更准却更慢**。因此需要同时报告枝干深度接受率、验证成本与总运行时，而不是只挑作者摘要里的最大 Speedup。

## Hydra++ 与接受验证要分开

论文对头结构和训练目标继续进行设计搜索，组合成名为 **Hydra++** 的更强配置。改进的来源不只“Head 2 读取 Head 1”，不能把 Hydra++ 的全部收益归因于单条输入边。

Hydra 仍需 [Target Tree Verification](C027-specinfer.md)。若要声称严格保持原始随机采样分布，应按 [Speculative Decoding](C020-speculative-decoding.md) 的接受–残差机制及多分支扩展规则校正。更准确的草稿不等于已经验证过的目标结果。尤其 Medusa 中部分 Typical Acceptance 或修改了 Target 的联合微调设置，不应和严格分布不变混称。

## 实验证据怎样解读

[COLM 2024 原论文](https://openreview.net/forum?id=FbhjirzvJG) 报告 Hydra++ 相比 Medusa **最高 1.31×** 的解码吞吐优势，相对普通自回归 **最高 2.70×**。注意两组基线不同，它不是“相比 Medusa 提高 2.70×”。这些值依赖所用模型与评估协议，不能当成所有 GPU 的恒定收益。

复现时应同时记录候选各深度接受率、Head 顺序执行时间、Target 验证位置数和端到端 Token/s。若只观察到接受率提高，却未扣除串行 Head 的新成本，无法证明加速。

下一篇 [ReDrafter](C026-redrafter.md) 用可迭代的循环草稿状态替换多个固定未来位置头；[EAGLE](C022-eagle.md) 则通过 Target 的连续特征空间建立不同形式的草稿生成器。
