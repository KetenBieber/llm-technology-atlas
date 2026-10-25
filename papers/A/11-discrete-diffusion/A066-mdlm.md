# MDLM：为什么“随机 Mask + 交叉熵”能成为严格的生成模型训练目标？

> **原论文**：Sahoo 等，**Simple and Effective Masked Diffusion Language Models**，NeurIPS 2024。[正式 49 页 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/eb0b13cc515724ab8015bc978fdde0ad-Paper-Conference.pdf) · [可检索的原论文 HTML](https://arxiv.org/html/2406.07524) · [作者说明与可视化](https://s-sahoo.com/mdlm/) · [官方源码](https://github.com/kuleshov-group/mdlm)。
>
> **方法关系**：[D3PM：类别转移矩阵与逆向后验](A065-d3pm.md) → **MDLM：SUBS 参数化 + Rao-Blackwellized NELBO + 生成采样** → [LLaDA：8B 规模与 SFT](../../B/05-moe-complete-llm/B015-llada.md)。联系 [Transformer 自注意力](../03-transformer/A013-transformer-attention-is-all-you-need.md) 与编码器式双向上下文，但 BERT 的原始 15% MLM 损失**不等于**本文的全时间扩散 ELBO。

## BERT 会填空，为什么不直接反复 Mask 就说自己是一种扩散模型？

想象两句话：

- 普通 Masked Language Model：训练时随机取某些位置置为 Mask，让模型输出这些位置的正确词。
- 扩散生成模型：需要给出一条合法的**从已知噪声先验产生完整句子的概率链**，还要有一个与该链的 Likelihood 相关的训练目标。

二者操作很像，但数学承诺不同。BERT 的某个固定 Mask 比例，例如 15%，只是在做 Representation Learning；没有自动指定从全 Mask 到真实文本的逐步逆转移，也不能仅凭这个 CE 宣称是某个有明确 Likelihood 的生成模型。

[D3PM](A065-d3pm.md) 已证明离散扩散可以用类别 Transition $Q_t$、逆向后验和 KL 训练；缺点是一般形式下必须关心巨大词表矩阵、噪声时间概率和许多实际上不可能的状态转移。MDLM 不重新发明所有类别 Corruption，而是问：**如果选最简单的 Absorbing Mask 类别，能否直接从已知过程推导出无需巨大矩阵的合法逆向参数化？**

原论文 Figure 1 左图的句子在不同 Mask 率下被遮住，顶部模型对各个 Mask 位置输出文字；右上特别强调 **MLM 与 MDLM 的区别在于目标来源和是否有 Ancestral Sampling**；右下图才是 LM1B PPL，不是推理时间曲线。见 [正式 Figure 1（PDF 第 2 页）](https://proceedings.neurips.cc/paper_files/paper/2024/file/eb0b13cc515724ab8015bc978fdde0ad-Paper-Conference.pdf)。

## 为什么 Absorbing State 的 Forward Process 简单得多？

设真实 Token 为 $x$，Mask 的 One-Hot 向量为 $m$。令 $\alpha_t\in[0,1]$ 表示**到噪声时间 $t$ 时仍未被 Mask 的概率**，规定 $\alpha_0=1,\alpha_1=0$，随 $t$ 增长递减：

$$
q(z_t\mid x)=\operatorname{Cat}\big(\alpha_t x+(1-\alpha_t)m\big).
$$

该类别只有两种可能：保留原始 $x$ 或变成 $m$，**不会跳成词表的第三个普通词**。一旦 Forward 过程到达 $m$，后续仍是 $m$；这就是“吸收态”的严格定义。

给定 $0\le s<t\le1$，若到较晚时刻 $z_t=m$，利用 Bayes 法则：

$$
q(z_s=x\mid z_t=m,x)=\frac{\alpha_s-\alpha_t}{1-\alpha_t},
\qquad
q(z_s=m\mid z_t=m,x)=\frac{1-\alpha_s}{1-\alpha_t}.
$$

这两项之和为 1，且均非负，因为 $\alpha_s\ge\alpha_t$。反之若 $z_t\ne m$，由于 Mask 在 Forward 中不可逆，它必然等于 $x$，所以更早状态 $z_s$ 也等于 $z_t$。

这一**确定的 Carry-Over** 是 MDLM 的第一块“免费数学约束”。没有必要让神经网络在训练中额外猜测，未被 Mask 的 Token 是不是应该变成另一个词。

### 一个 0.75 → 0.5 的反向时间小例子

令 $\alpha_t=1-t$。考虑已到 $t=0.75$、当前 Token 是 Mask，想沿反向到 $s=0.5$：

$$
P(\text{仍 Mask})=\frac{s}{t}=\frac{0.5}{0.75}=\frac23,\qquad
P(\text{解除 Mask})=\frac{t-s}{t}=\frac13.
$$

所以采样时，应有约三分之一概率提出一个真实类别并保持，其余三分之二维持 Mask，供后续修复。**所有位置在同一步都向网络提出预测，但并非所有预测都会在该步确认**。这解释了为何扩散采样既可以并行考虑多个 Token，又需要多轮迭代。

## SUBS 参数化为什么能让网络永远不输出违反 Forward Chain 的东西？

[原论文 §3.2.3，Eq.(10)](https://arxiv.org/html/2406.07524) 的 SUBS（SUBStitution）包含两个强制规则：

**Zero Masking Probabilities**：网络 $\mathbf x_\theta(z_t)$ 的清洁 Token 预测分布应满足

$$
p_\theta(x_0=m\mid z_t)=0.
$$

原因是 Mask 只是噪声过程里引入的特殊符号，**不属于原始训练数据的合法 Token 预测**。工程上把 Mask 类别的 Logit 设为 $-\infty$，再做 Softmax；不能仅盼望 SGD 自己学会“不要输出 Mask”。

**Carry-Over Unmasking**：如果 $z_t\ne m$，反向必须严格拷贝 $z_t$，不能让网络对它修改。因为对于吸收态 Forward，干净可见的位置已经被确定；若把它改掉，就违反了前节的 Bayesian Posterior 结构。

将两条直接嵌入 $p_\theta(z_s\mid z_t)$，当 $z_t=m$ 时：

$$
p_\theta(z_s\mid z_t=m)
=\frac{1-\alpha_s}{1-\alpha_t}\delta_m(z_s)
+\frac{\alpha_s-\alpha_t}{1-\alpha_t}
p_\theta(x_0=z_s\mid z_t).
$$

当 $z_t\ne m$ 时，则 $p_\theta(z_s=z_t\mid z_t)=1$。这套概率归一化，且永不让已确认 Token 倒退为 Mask。**这里描述的是论文 SUBS 和 Ancestral Sampler**；LLaDA 后续采用的 Low-Confidence Remasking 是一种不同的近似启发式，不能反过来声称 SUBS 允许任意改写已经确认的非 Mask Token。

## Rao-Blackwellized 目标为什么只对 Mask 位置有贡献？

对普通 D3PM，需要比较两个离散分布的 KL。在 SUBS 条件下：

- 如果 $z_t\ne m$，真实后验和模型反向规则都是确定的拷贝，KL 恰好为 0；
- 如果 $z_t=m$，需要预测的未知对象只有**最初干净词 $x$ 是什么**。模型把剩余概率质量放到普通词表上，其他 Mask 保留概率由已知 $\alpha$ 精确给出。

因此原论文 §3.3 Eq.(11) 中逐时步 KL 直接化简为一个乘常数的 Masked CE；连续时间版本是 [§3.4 Eq.(13)](https://arxiv.org/html/2406.07524)：

$$
\mathcal L_{\rm NELBO}^{\infty}
=\mathbb E_{x\sim p_{\rm data}}\int_{0}^{1}
\mathbb E_{z_t\sim q(\cdot\mid x)}
\frac{-\alpha_t'}{1-\alpha_t}\;
\mathbf 1[z_t=m]\;
\big[-\log p_\theta(x\mid z_t)\big]\;dt.
$$

这里为便于解释，写出了对 Mask 事件的 Indicator 和正的 CE 形式；积分的 $t$ 是虚变量，**不能又在积分号外对同一个 $t$ 做第二次独立期望**。原文 Eq.(13) 则把被 Carry-Over 拷贝的非 Mask 位视为 $\log1=0$，因此无需显式 Indicator。二者的**非负目标**是一致的。

如果选 $\alpha_t=1-t$，权重就是

$$
\frac{-\alpha_t'}{1-\alpha_t}=\frac1t.
$$

在长度为 $L$ 的整个序列中，每个位置都独立以概率 $t$ Mask，得到实际可训练的 Objective：

$$
\boxed{
\mathcal L_\theta
=\mathbb E_{t\sim U(0,1),\,x,\,z_t}
\left[\frac1t\sum_{i=1}^{L}
\mathbf1[z_t^i=m]\left(-\log p_\theta(x^i\mid z_t)\right)\right].
}
$$

**这就是加权的 Masked LM Cross-Entropy，但权重 $1/t$ 不是随便加的 Training Trick。** 它来自当前噪声残留概率 $1-\alpha_t$ 的导数；不加入该权重的“随机 Mask CE”通常不能宣称就是相同 Diffusion Model 的同一 ELBO。

### 从离散 KL 一步步消出“只在 Mask 位置交叉熵”

既然这个目标来自扩散模型，最应该追问的是**为何复杂的 KL 可以少掉大部分分布项**。令 $0\le s<t\le1$，当前 $z_t=M$，正确的原始词是 $x$，定义

$$
a=\frac{\alpha_s-\alpha_t}{1-\alpha_t},\qquad
b=\frac{1-\alpha_s}{1-\alpha_t},\qquad a+b=1.
$$

真实反向后验只有两个非零项：

$$
q(z_s=x\mid z_t=M,x)=a,\qquad q(z_s=M\mid z_t=M,x)=b.
$$

SUBS 模型将 $b$ 的概率**原样固定**在 Mask；另外把 $a$ 分配给网络预测的各个非 Mask 词：

$$
p_\theta(z_s=M\mid z_t=M)=b,\qquad
p_\theta(z_s=y\mid z_t=M)=a\,\pi_\theta(y\mid z_t),\quad y\ne M,
$$

其中 $\pi_\theta(M\mid z_t)=0$ 且 $\sum_{y\ne M}\pi_\theta(y\mid z_t)=1$。于是 KL 直接展开为

$$
\begin{aligned}
D_{\rm KL}(q\Vert p_\theta)
&=b\log\frac{b}{b}
+a\log\frac{a}{a\,\pi_\theta(x\mid z_t)}\\
&=\boxed{a\big[-\log\pi_\theta(x\mid z_t)\big]}.
\end{aligned}
$$

**第一项归零不是模型训练学来的；它已被 SUBS 的硬约束规定。** 第二项前的 $a$ 也不是人为定的超参数，而是由真实 Markov 后验决定的“本轮解除 Mask 的概率”。

以 $\alpha_t=1-t,\ t=0.75,\ s=0.5$ 为例，$a=1/3,\ b=2/3$。若网络对正确词的预测概率为 $0.4$，两分布的 KL 为

$$
D_{\rm KL}=\frac13(-\log0.4)\approx0.30543.
$$

用完整三状态表直接算也一样：真实 $q$ 在 $M,x$ 上分别为 $2/3,1/3$；预测 $p_\theta$ 在 $M,x,\text{其他词}$ 上分别为 $2/3,2/15,1/5$，总和仍为 1。**其它词的概率虽为非零，却不出现在 KL 的 $q$ 权重项中**，这正是交叉熵还能反传给所有词表 Logit 的原因。

如果 $z_t\ne M$，真实和模型后验都只拷贝当前词，故 KL 恒为 0。把一个时间步的两个情况加在一起，就得到

$$
\mathbb E_{z_t\mid x}\Big[
\mathbf1[z_t=M]\,a\,(-\log\pi_\theta(x\mid z_t))
\Big].
$$

时间划分足够细时，$a\approx[-\alpha_t'/(1-\alpha_t)](t-s)$，求和的极限才是原文 §3.4 Eq.(13) 的连续时间积分。于是 $\alpha_t=1-t$ 导出 $1/t$ 权重。**没有这个离散 KL 消项与极限，就不能仅凭“随机 Mask”倒推出 LLaDA/MDLM 的损失形式。**

### 1/t 为什么并不是“Mask 越少，Loss 就随便乘越大”？

假设只看一个 Token，$m\sim\mathrm{Bernoulli}(t)$，忽略“Mask 上下文改变预测”的因素，给定 Mask 事件的损失为 $\ell$：

$$
\mathbb E\left[\frac mt\,\ell\right]
=t\cdot\frac1t\ell+(1-t)\cdot0=\ell.
$$

所以 $1/t$ 恰好补偿低 Mask 率下该 Token 较少参与训练的概率；在实际整句中，CE 还依赖其它位置哪些被 Mask，因此仍需对整个 Mask Pattern 取期望，不能将其误解为“Loss 与时间 $t$ 完全无关”。

例如长度为 4 的句子、$t=1/4$，观察到只遮住第 3 个位置且它的 CE 为 $\ell$，未做 Sequence Mean 时本样本贡献是 $4\ell$；若训练采用除以 $L=4$ 的 Token 平均，则为 $\ell$。MDLM 原文 Eq.(14) 通常报告未除以 $L$ 的逐序列求和形式；工程代码的 Loss Reduction 还要与 Batch/Token 约定对齐。

数学上的奇异点：$t=0$ 时 $1/t$ 发散，但此时 Mask 概率也趋于 0，理论期望可保持有限。实现时不能手工指定 $t=0$ 然后做 $0\times\infty$；需要从开区间采样/作数值保护。

## “Rao-Blackwellization”与传统统计教科书的用法完全相同吗？

作者 §3.3 说得很谨慎：他们通过硬编码 Zero Mask Probability、Carry-Over，将若干已知的随机事件**解析消去**，获得更简单、更低方差的损失估计；这与 Rao–Blackwell 利用条件期望降低估计方差有相同精神，但严格说作者使用这个名称**略有术语扩展**。

原论文 §3.3 和 Supplement B.1 的推导重点不是“把网络换成某种 Attention”，而是**模型分布家族的硬约束带来代数消项**。如果一个模型仍允许给 Mask 概率非零，或允许改写非 Mask Token，原来的消项就不成立，不能照抄简化权重自称实现了同样的 ELBO。

### 原论文最能说明因果来源的是 Table 8，而不只是主榜 Table 1

[原文 §5.3、Table 8（正式版 Supplement 的对应结果）](https://arxiv.org/html/2406.07524#S5.SS3) 在 LM1B、110M 级别模型上逐步消去设计条件；作者在**离散时间模型使用 $T=1000$**，PPL 越低越好：

| 逐步变更 | PPL 上界 ↓ |
|---|---:|
| 完整 MDLM（连续时间） | **≤27.04 ± 0.01** |
| 去掉连续时间（离散 $T=1000$） | ≤27.19 ± 0.07 |
| 再去掉 Carry-Over | ≤28.56 ± 0.15 |
| 再去掉 Zero Mask Probability | ≤28.51 ± 0.15 |

这张表只能支持**按作者这一消融序列逐次变化的效果**；例如 28.56→28.51 反而略有改善，所以不能写成“Zero Mask 一定独立提升 1 点以上 PPL”。作者自己解释，虽然 SUBS 与连续时间使数学形式漂亮、训练方差更小，**最重要的综合收益之一其实来自训练实现方式的现代化**。

对照 [§3.5.1 Training Considerations](https://arxiv.org/html/2406.07524#S3.SS5.SSS1)：

- **Tokenizer**：D3PM LM1B 使用 8192 类 SentencePiece；MDLM 在 LM1B 采用 bert-base-uncased 分词器。更大的词表可能减少 Token 序列长度，进而改变表面上的远距离依赖难度。不同 Tokenizer 的 PPL 不可机械视为完全等条件指标。
- **模型与位置处理**：MDLM 采用带 RoPE 的 DiT / Encoder Transformer 路线，而不是原始 D3PM 的 T5 架构。此改动可能影响表达力，不能全部归功于损失简化。
- **数值实现**：只在实际 Mask Token 索引上计算目标，不再为通用 $K\times K$ 类别转移显式物化完整后验矩阵。
- **噪声时间抽样**：使用 Low-Discrepancy Sampler 降低估计方差，不再让一个 Batch 的所有 $t$ 完全独立、无结构抽样。
- **训练预算**：LM1B 33B 与 327B Token 的比较对应 1M / 10M Step；作者明确对 AR Baseline 控制**实际处理过的 Token 数**，不能只看迭代步数一致与否。

因此研究结论应从“MDLM 的一条公式直接战胜所有旧工作”修正为：**吸收态的概率约束简化了目标，同时 Tokenizer、结构、训练预算与估计方差管理共同使性能提高。** 想证明某一因素独立有效，必须控制其它条件，不能拿不同论文原始主榜差值充当单因素消融。

## 为什么无需 Time Embedding，也可以选择按时间采样？

对线性遮盖规则，当前 $z_t$ 显式告诉模型哪些 Token 完好、哪些变成 Mask。作者在 §3.2 和 §4.1 指出在选定 Reverse Parameterization 下，可以令干净词预测器不显式接受 $t$；同一 Mask Pattern 的合理条件预测无需在网络中额外读入一个浮点 Noise Level。

这**不等于扩散时间从数学上消失**：反向采样时仍需使用 $s/t$ 计算本步残留 Mask 概率，训练时仍需要 $1/t$ 目标权重。消失的是网络输入中的时间特征，而不是概率模型对时间的依赖。

这一点还有实际好处：如果上一步没有任何新位置解除 Mask，输入未变化，模型预测向量也可以复用，避免不必要的另一次 Denoser Forward。作者对这种 Efficient Ancestral Sampling 的分析在 §4.1；这不是保证所有采样步骤都被跳过。

## Figure 1 与 Table 1 的 PPL 只能怎么比较？

[NeurIPS 2024 Table 1（正式 PDF 第 7 页）](https://proceedings.neurips.cc/paper_files/paper/2024/file/eb0b13cc515724ab8015bc978fdde0ad-Paper-Conference.pdf) 在 LM1B 给出：

| 模型 | 参数 / 训练 Token | PPL（↓） |
|---|---|---:|
| Transformer 自回归（作者重训） | 110M / 33B | **22.32** |
| SEDD（报告的基线） | 110M / 33B | ≤32.79 |
| MDLM | 110M / 33B | **≤27.04** |
| Transformer 自回归（作者重训） | 110M / 327B | 20.86 |
| MDLM | 110M / 327B | **≤23.00** |

这张表显示在**固定 Token 预算 33B** 下，MDLM 相对 SEDD 有较明显改进，而仍落后同预算 AR；增加预算到 327B 又进一步缩小差距。论文的 PPL 带不等号，因为扩散模型使用 **Variational NLL 上界**，对其指数化形成 PPL 上界；通常不能与 AR 直接算“严格的 Likelihood 差”。

**Figure 1 右下**还画了早期 Diffusion-LM、D3PM、Diffusion-BERT、SEDD、MDLM 的多个数字。它用于展示研究历史而非严格相同数据/分词器/计算预算的统一消融，因此**不要把图中每一个数字当成只改变 MDLM 一项技巧的因果提升**。

[Table 5（PDF 第 8 页）](https://proceedings.neurips.cc/paper_files/paper/2024/file/eb0b13cc515724ab8015bc978fdde0ad-Paper-Conference.pdf) 则验证 Semi-Autoregressive Sampler：固定生成 2048 Token、RTX 3090、200 条序列，报告 SSD-LM 的 Generative PPL 35.43、2473.9 s/seq；MDLM 为 **27.18、89.3 s/seq**。这里的 Gen. PPL 是用另一个预训练 GPT-2 给生成样本评分，与 Table 1 的**真实数据 Test PPL** 不是同一种测量对象；更不能把 27.18 与 23.00 当成一步一步提高的结果。

### 一次可执行的 Reverse Sampling：哪些变量由模型产生，哪些根本不应预测？

在长度 $L=4$、$T=4$ 的反向过程中，$t$ 从 1 下降到 0，每步 $s=t-1/4$。当 $\alpha_t=1-t$，若当前位置尚是 Mask，当前网络产生 Clean Token 分布 $\pi_\theta(\cdot\mid z_t)$，则：

$$
\begin{aligned}
P(z_s=M\mid z_t=M)&=\frac{s}{t},\\
P(z_s=y\ne M\mid z_t=M)&=\frac{t-s}{t}\pi_\theta(y\mid z_t).
\end{aligned}
$$

已确认 Token 不参与采样；从全 Mask 起始时，各位置第一轮有 $1/4$ 概率解除 Mask，第二轮未确认位置有 $1/3$ 概率解除，第三轮 $1/2$，最后一轮 1。**这个条件概率递增并不是网络变得更有自信了，而是整个 Markov 链必须最终回到真实数据状态的数学要求。**

以下代码仅演示**在已知 Denoiser 概率时的祖先采样**，使用固定的演示分布（不是经过训练的 MDLM 网络），并且显式保持已确认位置：

~~~python
import numpy as np

rng = np.random.default_rng(7)
MASK, T, L = 3, 4, 4
z = np.full(L, MASK, dtype=np.int64)
clean_probs = np.array([.55, .30, .15])  # 教学替身: p(x0|z_t)

for step in range(T, 0, -1):
    t, s = step / T, (step - 1) / T
    unknown = np.flatnonzero(z == MASK)
    # 真正的模型应在此根据整条 z 同时产生 [L,K] 条件概率
    proposed = rng.choice(3, size=len(unknown), p=clean_probs)
    reveal = rng.random(len(unknown)) < (t - s) / t
    z[unknown[reveal]] = proposed[reveal]
    print(f"t={t:.2f} -> {s:.2f}: {z}")

assert MASK not in z
~~~

只在 Mask 位置运行最后两行并不意味着真实模型只处理 Mask Token：**双向 Transformer 仍读取整条 $z_t$ 作为上下文**。而这个小例子因为网络分布固定，故不会表现出上下文更新造成的词义变化；真实 MDLM/ LLaDA 的主要学习能力正来自 $\pi_\theta(\cdot\mid z_t)$ 随已确认 Token 改变。

## “可以生成任意长文本”为什么仍需分块？

模型一次全序列采样要先指定长度 $L$ 的全 Mask 序列。为了超过单轮上下文窗口，原文 §4.2 的 Semi-AR 方法是在当前块生成完成后，将一部分已确认 Token 带入下一块作 Context，剩余新位置全 Mask，再对下一块进行 Diffusion Sampling。

~~~{mermaid}
flowchart TD
 X["干净数据 x0"] --> MASK["随机 t ~ U(0,1)；每 Token 以概率 t 变 Mask"]
 MASK --> MODEL["双向 Transformer 预测所有 Mask 位置"]
 MODEL --> LOSS["仅 Mask CE，权重 1/t"]
 LOSS --> MODEL
 SEED["推断：全 Mask 块"] --> PRED["预测 Clean Token 概率"]
 PRED --> REV["按 s/t 保留 Mask，按 (t-s)/t 确认"]
 REV --> PRED
 REV --> DONE["块完成：保留 Prefix，延伸下个 Mask 块"]
~~~

*自制训练—采样两路径图，非原论文 Figure；图上循环仅表示一次反向去噪内的多步预测，正式实现需遵守 SUBS 的已确认 Token 不改写约束。*

这种方式能无限**增加生成总长度**，但不意味着模型每一步能无损读到无限过去：窗口内留存的 Context 依然有限，更早历史可能需要截断或摘要。每个 Block 内仍有多个去噪 Forward，延迟不自动比一个 Token 一步的 AR 更低。

## 这篇论文真正推进了哪个技术问题？

D3PM 已证明吸收态离散扩散是合法的生成模型。MDLM 的本质贡献是**利用吸收态不可逆性，直接删除网络无需学习的概率分支**，从复杂的 Posterior KL 推到加权 Masked CE，并用简洁的采样机制将 BERT-like 双向网络变成真正能生成文本的概率模型。

局限仍在：训练时的双向 Denoiser 会访问整段 Masked Context；采样需要多轮 Network Forward；其准确率、模型大小和 Token Budget 与主流大规模 LLM 仍有距离。下一篇 [LLaDA](../../B/05-moe-complete-llm/B015-llada.md) 将把这一问题推进到完整大模型：**如何沿用遮盖扩散训练逻辑，在 8B 参数规模上做预训练、Prompt-Response SFT，并检验其 In-Context Learning、数学推理和反向事实检索能力**。
