# Griffin：不是“往 Mamba 里塞几个 Attention”，而是重新设计 RG-LRU 的遗忘门

> **原论文**：Soham De 等，*Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models*（Google DeepMind，2024）。[arXiv:2402.19427v1 官方 PDF](https://arxiv.org/pdf/2402.19427) · [论文摘要](https://arxiv.org/abs/2402.19427)。
>
> **技术关系**：[Transformer](../03-transformer/A013-transformer-attention-is-all-you-need.md) 的全局 Attention → [S4](A058-s4.md) 的线性状态递推 → [Mamba](A059-mamba.md) 的选择性遗忘、[RWKV](A062-rwkv.md) 的门控递推。Griffin 是**并行架构探索**，不应误写成 Mamba 的版本迭代。后续完整混合模型对照：[Jamba](../../B/05-moe-complete-llm/B013-jamba.md)、[Zamba](../../B/05-moe-complete-llm/B014-zamba.md)。

## 一、为什么只把 Self-Attention 换成一条递推公式还不够？

标准解码器的两个职责需要分开讨论。第一个是**不同 Token 之间如何交换信息**（Temporal Mixing）；第二个是**对已经取到的信息做非线性变换**（Channel Mixing / MLP）。Transformer 用 Attention 执行第一项，用 MLP 执行第二项。很多“用 RNN 替代 Transformer”的宣传忽略了第二项仍然存在。

如果强行让所有历史在固定维度 $h_t$ 中压缩，就必须接受一个代价：历史的任意两个细节可能因为映射到相同状态方向而发生冲突；未来查询不能像 Full Attention 一样无条件读取原始历史 KV。另一方面，这条递推如果始终把新输入写入，也可能让“无意义的中间 Token”不断冲淡真正重要的早期信息。

Griffin 的原始出发点是一个较窄但具体的问题：**能否构造一个足够容易训练、仅逐通道更新，却可以按当前输入决定“这一步几乎不碰旧记忆”的门控单元？** 然后在需要显式相邻 Token 关联的位置加入局部 Attention。

作者实际提出两个 Backbone：**Hawk** 用 RG-LRU 递推 Block 替代序列混合层，不用 Attention；**Griffin** 在这些递推 Block 中插入 Local Sliding-Window MQA。只有后者才是混合模型。若一开始连 Hawk 和 Griffin 都没区分，后续所有实验就很容易把 Attention 的收益错误归给门控。

## 二、从 Figure 2 重建完整 Residual Block，不要只盯住一条状态公式

[原始 PDF 第 3 页 Figure 2](https://arxiv.org/pdf/2402.19427) 的图形从左到右分别是 **(a) Residual Block、(b) Gated MLP、(c) Recurrent Block**。三个图不是三种可任选的网络，而是“公共外壳—通道混合—一种可替换的时间混合”：

1. 对状态 $x$ 做 RMSNorm，执行 Temporal Mixing，把结果与输入作 Residual Add；
2. 再做一次 RMSNorm，执行 Gated MLP，与前一步结果作 Residual Add；
3. 将此 Residual Block 堆叠 $N$ 次，最后再 Norm、投影到 Vocabulary，预测下一个 Token。

原文 Gated MLP 由两个并行线性投影构成，其中一支经过 GeLU，再与另一支逐元素相乘，最后通过 Linear 收回 Hidden Dimension。这与 [SwiGLU/MoE](../05-moe/A032-sparsely-gated-moe.md) 可能具有相似门控直觉，但**不是同一公式，更不意味着 Griffin 的创新等于改进了 MLP**。

真正的新组件在 Figure 2(c)：输入先分为两条同维分支。一条经过长度为 4 的 Depthwise Temporal Conv，再进入 RG-LRU；另一条经过 GeLU。两支做逐元素乘积后，再线性投影回 Residual Width。

~~~{mermaid}
flowchart TD
 X["x:[B,T,D]"] --> N["RMSNorm"]
 N --> C["Temporal Mixer: Local MQA 或 Recurrent"]
 C --> R1["Residual Add"]
 X --> R1
 R1 --> N2["RMSNorm"]
 N2 --> M["Gated MLP: GeLU 分支 × 线性分支"]
 M --> R2["Residual Add"]
 R1 --> R2
 R2 --> Y["下一层"]
 C -. "Recurrent Block 展开" .-> B1["Depthwise Conv1D(4) → RG-LRU"]
 C -. "旁路 Gate" .-> B2["Linear → GeLU → 与递推输出相乘"]
~~~

*自制结构解释图，非论文原图。箭头表达模块组合关系，两个 RG-LRU/GeLU 分支都属于 Recurrent Temporal Mixer 内部。*

**图证据边界**：Figure 2 可以确定模块顺序，却不包含独立的性能消融。要证明两种 Temporal Mixer 结合有价值，仍必须查实验中 Hawk、Griffin、MQA Transformer 的相同规模对比。

## 三、逐项推导 RG-LRU：为什么这里有两个完全不同的 Gate？

[原始 PDF 第 4 页 §2.4，Eq.(1)–(4)](https://arxiv.org/pdf/2402.19427) 定义了 RG-LRU：

$$
\begin{aligned}
r_t &= \sigma(W_a x_t+b_a),\\
i_t &= \sigma(W_x x_t+b_x),\\
a_t &= a^{c r_t},\qquad a=\sigma(\Lambda),\\
h_t &= a_t\odot h_{t-1}+\sqrt{1-a_t^2}\odot(i_t\odot x_t).
\end{aligned}
$$

所有 $\odot$、指数、平方根都按 Channel 逐元素操作，$\Lambda$ 训练时可学习，$c=8$ 是作者实验采用的标量超参数。这里 $x_t$ 是传入 RG-LRU 的状态维输入（真实模型里来自 Conv 分支），不是原始词表 Token ID。

分别看两个 Gate：

- **Input Gate $i_t$**：控制新信息 $x_t$ 是否有资格写入。
- **Recurrence Gate $r_t$**：先改变旧状态系数 $a_t=a^{cr_t}$，进而同时改变新信息前的 $\sqrt{1-a_t^2}$。

它并不是传统 GRU 的 $h_t=(1-z_t)h_{t-1}+z_t\tilde h_t$。RG-LRU 把“旧状态保留量”和“新输入写入量”约束在类似单位圆的关系里，且还有额外 $i_t$：

$$
a_t^2+(1-a_t^2)=1.
$$

这个平方关系在简单独立、单位方差的旧状态与创新项假设下，有助于维持方差规模。但它不是“前后两项的绝对幅值必定一样”或“实际 Transformer 层输出方差严格恒定”的证明；数据相关性、Input Gate、Conv、Residual 均会改变统计量。

### 极限分析：为什么 $r_t\to0$ 可以有效保护旧记忆？

因为 $0<a<1$，所以 $a^{cr_t}\in(a^c,1)$：

- 当 $r_t\to0$，$a_t\to1$，$\sqrt{1-a_t^2}\to0$，得到 $h_t\to h_{t-1}$。**这一 Token 既没有强行覆盖记忆，也不会因为随机输入而扰动它**。
- 当 $r_t\to1$，$a_t\to a^c$，衰减更显著，新信息系数更大，是否真正写入还取决于 $i_t$。
- 因为 $a_t>0$，单步并不允许通过负系数翻转旧状态符号；这是采用 Real-Valued Diagonal Recurrence 的建模取舍。

作者把 $a$ 初始化到使 $a^c$ 在约 0.9–0.999 的分布区间内，这是控制初期记忆时间尺度而非硬性给模型规定某个固定最大上下文。

### 一个可完整算出的状态更新

假设只看某一个 Channel，取 $a=0.9,c=8$，当前输入 $x_t=2$、旧状态 $h_{t-1}=1$、两个 Gate 为 $r_t=0.25,i_t=0.5$（只为示例，不是作者实际权重）。那么

$$
a_t=0.9^{8(0.25)}=0.81,\qquad
\sqrt{1-a_t^2}=\sqrt{0.3439}\approx0.58643.
$$

所以

$$
h_t=0.81(1)+0.58643(0.5)(2)
\approx1.39643.
$$

如果当前是被 Recurrent Gate 判断为“应跳过”的 Token，令 $r_t$ 接近 0，则即使 $x_t=2$，更新也接近 $h_{t-1}=1$。这个例子说明 RG-LRU 的 **Gate 是对记忆传播规则的输入依赖控制**，而不只是训练中增加一段 MLP。

## 四、为什么它可以并行扫描，却失去了 S4 的固定卷积核？

这一点与 [Mamba](A059-mamba.md) 的选择性状态十分接近：只要 $a_t$、$b_t=\sqrt{1-a_t^2}\,i_tx_t$ 都由当前输入生成，就可写成

$$
h_t=a_th_{t-1}+b_t.
$$

两个时间片的仿射变换合成满足

$$
(a_2,b_2)\circ(a_1,b_1)
=(a_2a_1,a_2b_1+b_2).
$$

函数复合满足结合律，因此理论上可以使用 Prefix Scan。**但 $a_t$ 随内容变化，输入到输出不再具有一个与输入独立、仅随距离变化的固定卷积核**。这与 [S4](A058-s4.md) 的固定 LTI 核本质不同。

原文第 8 页明确说 RG-LRU 的门控**不兼容静态卷积表示**；虽然数学上仍允许 Associative Scan，作者在 TPU-v3 的具体实现中发现这种并行前缀组合带来的随机访存，比定制线性 Scan 更慢。注意该实验是性能观察，不是“Associative Scan 在所有 GPU 上都比顺序循环慢”的普遍定理。

### 从离散时间记忆核看 RG-LRU 的真实信息传播

单通道门控更新是 $h_t=a_th_{t-1}+b_t$，其中 $b_t=\sqrt{1-a_t^2}\,i_tx_t$。将它展开而不是仅仅描述“门控能保留历史”：

$$
h_t=\Big(\prod_{k=1}^ta_k\Big)h_0
+\sum_{j=1}^t\Big(\prod_{k=j+1}^ta_k\Big)
\sqrt{1-a_j^2}\,i_jx_j.
$$

某个旧输入 $x_j$ 对当前状态的直接影响（暂时固定其它时间的 Gate 值）是

$$
\frac{\partial h_t}{\partial x_j}\Big|_{a,i\ \mathrm{fixed}}
=\Big(\prod_{k=j+1}^ta_k\Big)\sqrt{1-a_j^2}\,i_j.
$$

这里的导数**不包含 Gate 网络自身对输入的导数**，所以只是用于解释固定 Gate 时的状态传播直通路径；完整反向传播还包括 $\partial a_j/\partial x_j$ 和 $\partial i_j/\partial x_j$ 等项。单看直通路径就能发现：

- 如果某些后续位置的 $r_k\to0$，则 $a_k\to1$，它们不会明显压低早期输入的贡献。
- 如果连续很多位置使 $r_k\to1$，直通路径会按 $\prod a_k$ 衰减，远历史逐渐被遗忘。
- Input Gate $i_j$ 只影响**本次新输入**是否进入状态；Recurrence Gate $r_k$ 则改变**已有历史在之后传播**的时间常数。

例如令两个输入的初始状态为 0，取 $a_1=0.8,\;i_1x_1=2$，则 $h_1=0.6\cdot2=1.2$。第二步若“跳过无关词”使 $a_2=0.999$，且 $i_2x_2=10$，得到

$$
h_2=0.999\times1.2+\sqrt{1-0.999^2}\times10
\approx1.6459.
$$

尽管旧状态保留系数接近 1，输入门没有关时，**新输入仍可能很大**；这解释了为什么需要 $i_t$ 与 $r_t$ 两个不同 Gate，而不是只用一个。若同时 $i_2=0.01$、$x_2=10$，新增量则只有约 0.00447，$h_2\approx1.2033$。这种“双重保护”来自具体公式而非任何一阶递推的必然性质。

## 五、Hawk 与 Griffin 的真正差异：局部 Attention 留下了什么能力？

普通 Attention 在第 $t$ 步可对全部历史 $1,\ldots,t$ 动态重新赋权；Hawk 只保留 RG-LRU 状态。Griffin 折中：**少数层保留固定窗口大小 $W$ 的 Attention，其他层由 Recurrent Mixer 承担更长范围的信息传播**。

$$
\operatorname{LocalAttn}(q_t)=
\sum_{j=\max(1,t-W+1)}^t
\operatorname{softmax}_{j\in[t-W+1,t]}
(q_tk_j^\top/\sqrt{d_h})\,v_j.
$$

局部 Attention **不需要完整历史 KV**，但是它不等于有了全局检索：某条信息若超过窗口，只能依靠上游/下游 RG-LRU 递推状态传过来。状态维度有限、噪声过多或 Key 冲突时，无法保证精确恢复。

原文 §3 明确列出 Hawk 为纯 Recurrent Temporal Mixing、Griffin 是 Recurrent/Local Attention 组合，而作者的 Transformer Baseline 用 Global MQA。三者都沿相同的 RMSNorm、Gated MLP 和 Residual Pattern 设计，并尽量匹配参数规模与训练配方。

若全部换成 Local Attention，则模型只能经有限深度的多层局部窗口交互来传播长距离信息；若全部换成纯递推，则局部显式 Token 匹配机制被压缩到固定状态里。Griffin 的设计是在**显式邻近关联与压缩的长程记忆**间作组合，不是让两者在每一层都重复完整计算。

## 六、从 Figure 6 反看一个常被忽略的“检索失败”结果

[原论文 Figure 6（PDF 第 12 页）](https://arxiv.org/pdf/2402.19427) 分三张图：

- (a) Selective Copying：专门训练 5-Block、小约 250K 参数模型，序列长度 1024、其中只有 16 个有效 Token，其他都是 Noise Token；
- (b) Induction Heads：另外训练同量级小模型，长度 256；标记特殊 Token 后取回其后继；
- (c) Phonebook Lookup：**直接用已预训练的 7B Hawk/Griffin 与 6B MQA Transformer，不针对该任务再微调**。

(a)(b) 说明这些架构在收到任务监督后可以学习某些复制/归纳行为；(c) 则指出预训练模型在未经特定训练的精确检索上**弱于 Transformer**。这是关键的能力边界。不能只引用合成题经过监督后学得很好，就得出“递推压缩与 Full Attention 的精确回忆能力完全相同”。

另外，[Figure 1(a)（PDF 第 2 页）](https://arxiv.org/pdf/2402.19427) 的横轴是 Training FLOPs、纵轴 Validation Loss，区分模型 Scaling。图里 Griffin 优于同配方的 MQA Baseline，在论文固定数据/超参数范围内支持混合架构有效；并不能把与外部 Llama-2、Mamba（训练 Token、数据集、调参不同）的成绩直接归因为 RG-LRU 的单一数学设计。

### 原论文 Table 1 对“混合是否有用”的证据比跨模型排行榜更直接

[原论文 Table 1（arXiv v1 PDF 第 6 页）](https://arxiv.org/pdf/2402.19427) 将模型训练预算与评测口径列在同一个表中：

| 模型 | 参数级别 | 训练 Token | MMLU | HellaSwag | Average |
|---|---:|---:|---:|---:|---:|
| MQA Transformer | 3B | 300B | 31.7 | 71.0 | 59.0 |
| Hawk | 3B | 300B | 31.3 | 71.7 | 59.5 |
| **Griffin** | **3B** | **300B** | **32.6** | **73.5** | **60.7** |
| Llama-2 | 7B | 2T | 45.3 | 77.2 | 65.3 |

前三行作者尽量匹配了数据和训练协议，较有说服力地表明：在这组 3B 预算下，纯递推 Hawk 已有竞争力，而加入 Local Attention 的 Griffin 进一步改善部分指标。第四行**参数、数据、训练 Token 均不同**，只能说明外部参照范围，不能解释为单独加入 RG-LRU 的因果效益。

同时注意 Table 1 使用的是 Character-Normalized Accuracy 等特定评估口径，而不是所有任务使用相同的 Raw Accuracy。Figure 6 的 Phonebook Lookup 无微调表现又提醒了相反的限制：**在合成复制任务能从监督学习到正确策略，不等于预训练模型能 Zero-Shot 精确取回任意过去内容**。把正反证据放在一起，才可判断“保留少量 Attention”的真实动机。

## 七、真正需要保留的四个研究边界

**其一，记忆可保持不等于内容可精确检索。** RG-LRU 的 $r_t\to0$ 能令状态忽略新 Token，但它依旧是有限维压缩。完整位置级查询的表达能力与状态容量不能直接混同。

**其二，Local Attention 不等于全局 KV 完全消失。** 只有 Recurrent Layers 不再保存随上下文增长的完整 KV；Local Attention 仍需保留最近 $W$ 枚 Token 的缓存。

**其三，训练规模对比必须对齐协议。** 论文采用 300B Token 等设置训练 Hawk/Griffin，并与使用约 600B Token 的 Mamba 和约 2T Token 的 Llama-2 做外部比较；作者也明确指出数据与 Hyperparameter 不同可能影响结论。相同配方的 MQA Baseline 对结构分析更可靠。

**其四，RG-LRU 的实数门控是设计选择，而非由 SSM 理论唯一推出。** [Mamba](A059-mamba.md) 用输入相关 $\Delta,B,C$，RG-LRU 采用 $a^{cr_t}$ 和 $\sqrt{1-a_t^2}$，它们都能实现内容选择，但记忆谱与归一化行为不同。

**归结到技术演化**：Hawk 验证“有内容敏感遗忘门的实值线性递推能否独立建模”，Griffin 再问“让少量 Local Attention 与之配合，能否补回 Token Pair 关联能力”。后续 [Jamba](../../B/05-moe-complete-llm/B013-jamba.md) 则换了层级：不再设计 RG-LRU 单元，而是将已有的 **Mamba 和全局 Attention** 按层混合，并让 MoE 独立控制模型容量。
