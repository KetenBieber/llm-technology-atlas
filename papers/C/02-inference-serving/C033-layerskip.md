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

### 原文 Figure 1 与 Figure 3：训练图和解码图不是一个阶段

[LayerSkip ACL 2024 Figure 1（论文首页，印刷 12622）](https://aclanthology.org/2024.acl-long.681.pdf) 的左侧是训练中的 Layer Dropout + Early Exit Loss，中间是直接从浅层出口预测，右侧才是**浅层逐步 Draft 与深层 Verify/Correct**。绿色表示实际执行的层，空心/灰色表示该 Token 在当前阶段无需计算的层。这张图最容易误读的地方是：浅层草稿与深层验证使用**同一套参数**，不是分别训练两个独立 LLM；缓存颜色体现是否复用计算，不是某个 Layer 的参数是否被冻结。

原文 Figure 3（PDF 第 3 页，印刷 12624）则强调一套经 Layer Dropout、Early Exit Loss 训练的模型等效提供**多种可用深度**。它不直接证明某个浅层出口的准确率比标准训练高多少——那需要 Figure 2 中每层能否正确预测 Token 的对照，和不同训练选项的受控消融。

### Early Exit Loss 的梯度与额外算力账本

设一条输入序列形状为 $[B,T,d]$，在层 $\ell$ 的输出为 $H_\ell:[B,T,d]$，共享输出矩阵 $W_{\mathrm{lm}}:[d,V]$。每个出口的预测张量为

$$
Z_\ell=H_\ell W_{\mathrm{lm}}\in\mathbb R^{B\times T\times V}.
$$

为多个出口计算 CE 时，浅层 $H_\ell$ 同时接受来自**本层出口的直接梯度**和来自更深层出口的链式梯度。共享 LM Head 会累积这些出口的梯度，但并不意味着训练 FLOPs 不增加：若真的对所有 $L$ 个层都做全词表投影，额外成本约与 $L B T dV$ 同阶。训练可选择出口位置与权重，或者采用分块投影等策略；不能把“共享参数”混同于“每个出口的 Logits 免费”。

对比增设独立出口投影：若 $d=4096,V=32000$，每新增一个 FP16 输出矩阵就需 $4096\times32000\times2=262{,}144{,}000$ byte，约 **250 MiB**。如果新增 8 个独立出口就是约 2 GiB 的权重，尚未计梯度和优化器状态。共享 LM Head 避免了这笔随出口数增长的参数存储，但不消除计算和激活存储。

### 为什么前层草稿缓存能复用，但仍要验证 KV 的生命期？

假设 $L=32$ 层、早退出口 $L_e=8$ 层、草稿长度 $K=4$。传统独立 Draft 模型需要完整运行自己的网络，并由 Target 再算 32 层对四枚 Token 的验证；LayerSkip 可以先运行前 8 层逐步生成草稿的候选特征，再用剩下的 24 层在合适的 Causal Mask 下并行计算验证。理论上少做重复的前 8 层，而不是完全跳过 Target 验证。

但若第 2 枚候选被拒绝，随后第 3、4 枚候选所对应的浅层 KV 仍然依赖**错误前缀**。它们必须失效或被正确回滚，不能被下一轮当作真实历史状态沿用。这笔 Cache 管理代价很容易在只画 Layer 结构框时消失，实际上决定了多轮解码是否正确。

### 三种实验结论需要三个不同证据

LayerDropout 和 Early Exit Loss 的训练消融应回答“在同一出口深度，Top-1 预测是否更准确”；Early Exit 直接部署应回答“只运行前 $L_e$ 层会损失多少任务质量”；Self-Speculative 验证应回答“同一目标模型、同一输入和采样规则下，最终结果能否恢复，并减少多少墙钟时间”。作者摘要分别报告 CNN/DM **2.16×**、代码 **1.82×**、TOPv2 **2.0×**，但不同任务/模型不能作为共享 LM Head 独立收益的对照。

| 维度 | Draft & Verify | LayerSkip |
|---|---|---|
| 如何获得廉价草稿 | 部分中间层不执行 | 训练中鼓励早退，中间层直接预测 |
| 需要额外训练吗 | 核心方法不需要 | 需要应用专门训练/微调配方 |
| 与 Target 的关系 | 完整路径随后验证 | 浅层草稿与剩余层校正，复用计算 |
| 正确性条件 | 最终仍按完整模型决定 | 最终仍按完整模型决定 |

[ACL 2024 原文](https://aclanthology.org/2024.acl-long.681/) 对不同 Llama 大小和训练阶段做研究，并报告 CNN/DM 摘要最高 **2.16×**、代码任务 **1.82×**、TOPv2 语义解析 **2.0×**。这些是不同实验的峰值，不是同一模型在全部任务上的表现。技术价值不能只看 speedup：应同时检查较浅层的预测质量、深层完整性能、共享缓存开销和批量服务中的真实收益。

**下一节点**：[BiLD](C034-bild.md) 不是同一模型早退，而是两个模型通过回退/回滚协作；并且必须区分它的近似质量目标与严格推测采样的分布恒等保证。
