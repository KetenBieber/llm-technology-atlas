# LayerSkip：训练时怎样让浅层也能提前给出可用 Token？

> **原始论文**：[LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://aclanthology.org/2024.acl-long.681/)，ACL 2024。**类型**：C 类推理机制与系统。 [原文 PDF](https://aclanthology.org/2024.acl-long.681.pdf)。

前置：[同一模型跳层草稿 Draft & Verify](C032-draft-and-verify.md)。关联：[Speculative Decoding](C020-speculative-decoding.md)。

## 从“运行时硬跳层”走向“训练时鼓励早退”

如果普通 Transformer 的最后一层特征最适合做 Token 预测，直接拿中间层接 LM Head 可能不准。LayerSkip 不只是部署时寻找一个巧妙的 Skip Mask，而是**改变训练机制**，让较早层本身也具备更好的可预测性，并据此开展自推测解码。

原论文有两项核心训练选择：

- **Layer Dropout**：对靠后的层设置更高的随机跳过概率，使模型训练期间经常走较浅或不完整的计算路径，提升面对不同有效深度的适应性。
- **Early Exit Loss**：为多个中间层计算 Token 预测损失，并让不同出口共享同一个输出 Head，避免每个出口都存独立的大型词表映射参数。

设中间层输出为 $h^{(\ell)}$，共有同一 LM Head $W$：

$$
q_\ell(x_{t+1})=\operatorname{softmax}(Wh_t^{(\ell)}).
$$

早退辅助损失可说明性地写成

$$
\mathcal L_{\rm exit}
=\sum_{\ell\in\mathcal I}\lambda_\ell
\,{\rm CE}(q_\ell,y_{t+1}).
$$

权重、位置、训练采样策略都属于论文具体配方；此式只说明其为什么能够对中间层施加直接 Token 监督，非原文精确实现逐字复刻。

~~~{mermaid}
flowchart TD
 T[训练：层 Dropout + 多层共享 Head Loss] --> S[浅层 Token 预测更可用]
 S --> D[推理：浅层先产生候选]
 D --> V[复用激活，剩余层校正并验证]
 V --> O[按原 Target 规则提交结果]
~~~

*自制解释图。原论文 [Figure 3（PDF 第 3 页）](https://aclanthology.org/2024.acl-long.681.pdf) 对照不同训练层 Dropout/Exit Loss 与推理阶段浅层草稿—深层验证的结构。*

## “共享计算”究竟共享了什么？

普通双模型推测要分别运行小模型与大模型，权重和部分计算没有天然重合。LayerSkip 自推测使用同一权重的前 $L_e$ 层作 Draft，Target 验证从中间状态继续执行剩余 $L-L_e$ 层；在合适的实现下，已有的浅层 KV 和激活可以被复用，减少重复计算与额外内存。

但并非所有候选路径都可直接复用一个最终状态：Target 必须在每个候选的**正确因果历史**下生成完整的条件概率。要记录早退深度 $L_e$、草稿 Token 数、拒绝时回滚逻辑及相应 KV 生命周期。

## 为什么共享 LM Head 是一条现实工程约束？

词表大小 $V$、隐藏维度 $d$ 时，一个新的独立输出投影约有 $Vd$ 参数（不计 Bias）。若每层都加一个不同 Head，额外存储可能抵消自推测不用额外模型的意义；LayerSkip 共享 Head，使早退层学会使用已有词表投影。与此同时，早退训练可能影响完整深层最终性能，必须与不加 early exit loss 的训练进行受控比较。

## 浅层共享 Head 并不意味着可直接读取深层输出

一个 $L$ 层 Transformer 的每层隐藏状态 $h_\ell\in\mathbb R^{d}$，语言模型头 $W_{\rm lm}\in\mathbb R^{d\times V}$。训练阶段使用共享 LM Head 对多个深度输出 Logits：

$$
z_\ell=h_\ell W_{\rm lm},\qquad
\mathcal L_{\rm train}=
\mathcal L_{L}+\sum_{\ell\in\mathcal I}\lambda_\ell
\operatorname{CE}(\operatorname{softmax}(z_\ell),y).
$$

如果不额外训练，浅层 $h_\ell$ 可能尚未具备足够的语义区分性，即使 $W_{\rm lm}$ 与最终层相同，$\operatorname{argmax}z_\ell$ 仍会与最终层有很大差异。LayerSkip 将训练期 Layer Dropout 和多出口 Loss 组合起来，**主动让中间层更早具备生成词表概率的能力**；这与 [Draft & Verify](C032-draft-and-verify.md) 的纯推理路径调整不同。

### 为什么 Layer Dropout 要随深度增加？

浅层负责输入特征的基本表达，若频繁跳过会使更深层拿不到稳定的表征；靠后的层更接近输出，可以在训练中对它们使用更高概率 Dropout，迫使模型在缺少部分后层时仍能依赖前层的特征完成预测。此为训练配方的设计动机，**不是严格定理保证后层比前层不重要**。要证明效果必须对照不同 Layer Dropout Schedule、只加 Exit Loss、两者联合的受控消融。

## 共享的部分计算如何真正节省显存？

若前 $L_e$ 层已为多枚 Draft Token 建立中间 Activation/KV，Target Verification 可以复用它们，继续执行 $L_e+1$ 到 $L$ 的层计算，避免同一候选序列的浅层重复前向。设一次浅层计算耗时 $T_e$，剩余深层验证 $T_r(K)$，每轮预测 $K$ 枚 Draft 且确认 $A$ 枚，示意平均成本

$$
\bar t=\frac{K T_e + T_r(K)+T_{\rm cache}}{A}.
$$

实际执行时 $T_{\rm cache}$ 不能省略：多分支候选或被拒绝的后继 KV 都必须回滚/释放，且验证阶段某些浅层结果必须在正确祖先前缀下保持一致。**只有共享的 Activation 真实被重用**，这笔加速才能算在自推测设计上。

[ACL 2024 官方论文](https://aclanthology.org/2024.acl-long.681/) 与 [Meta 作者说明](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/) 的主张分开三段：训练方法提升 Early Exit Accuracy、推理机制复用浅层计算，以及任务相关的端到端速度。不能从最高 2.16× 的 CNN/DM 单次结果推断全部三个组件对所有任务均是正收益。

## 与 Draft & Verify 的实质差异

| 维度 | Draft & Verify | LayerSkip |
|---|---|---|
| 如何获得廉价草稿 | 部分中间层不执行 | 训练中鼓励早退，中间层直接预测 |
| 需要额外训练吗 | 核心方法不需要 | 需要应用专门训练/微调配方 |
| 与 Target 的关系 | 完整路径随后验证 | 浅层草稿与剩余层校正，复用计算 |
| 正确性条件 | 最终仍按完整模型决定 | 最终仍按完整模型决定 |

[ACL 2024 原文](https://aclanthology.org/2024.acl-long.681/) 对不同 Llama 大小和训练阶段做研究，并报告 CNN/DM 摘要最高 **2.16×**、代码任务 **1.82×**、TOPv2 语义解析 **2.0×**。这些是不同实验的峰值，不是同一模型在全部任务上的表现。技术价值不能只看 speedup：应同时检查较浅层的预测质量、深层完整性能、共享缓存开销和批量服务中的真实收益。

**下一节点**：[BiLD](C034-bild.md) 不是同一模型早退，而是两个模型通过回退/回滚协作；并且必须区分它的近似质量目标与严格推测采样的分布恒等保证。
