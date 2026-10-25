# RWKV：一种“有归一化的递推注意力”，究竟怎样兼得 Token Shift、WKV 与 RNN 状态？

> **原始论文**：Bo Peng 等，*RWKV: Reinventing RNNs for the Transformer Era*，Findings of EMNLP 2023，pp. 14048–14077。[ACL Anthology 正式页面](https://aclanthology.org/2023.findings-emnlp.936/) · [正式 PDF（30 页）](https://aclanthology.org/2023.findings-emnlp.936.pdf) · [官方项目](https://github.com/BlinkDL/RWKV-LM)。
>
> **版本界限**：本文研究原论文对应的 **RWKV-4**，而不是把 RWKV-5/6 的矩阵状态或 RWKV-7 的 Dynamic State Evolution 倒灌进 2023 年原方法。前置：[Transformer](../03-transformer/A013-transformer-attention-is-all-you-need.md)、[S4](A058-s4.md)、[RetNet](A061-retnet.md)；可沿 [xLSTM](A063-xlstm.md) 继续比较归一化与门控记忆。

## 旧式 RNN 已经有常量状态，RWKV 为什么还要提出一套复杂的新网络？

传统 RNN 的一步递推 $h_t=f(h_{t-1},x_t)$ 确实可以只保存一个状态。但“状态长度固定”解决的仅仅是**资源上如何保存历史**，不能证明模型具备 Transformer 的内容相关读写，也不能证明长模型可稳定预训练。

标准 Attention 在最后一个 Token 到来时用 $q_t$ 重新查询全部 $k_i,v_i$：

$$
\operatorname{Attn}(q_t)=\sum_{i\le t}
\frac{e^{q_t^\top k_i}}{\sum_{j\le t}e^{q_t^\top k_j}}v_i.
$$

RWKV 试图保留这种“权重归一化后对 Value 加权求和”的结构，却移除随当前 Query 和历史位置两两成对计算的 $q_t^\top k_i$。作者把 Key 相关的指数权重和每通道时间衰减直接做成**可以递推更新的分子/分母**，然后用当前 Token 的 Receptance Gate 决定有多少信息被读出。

所以 RWKV 的名称不是四个相互独立、任选其一的算法：**Receptance（读出门控）+ Weighted Key Value（按时间衰减归一化的记忆）** 合起来才构成架构。

## 关键原始公式：为什么 RWKV 要为“当前 Token”额外设置 Bonus？

[论文 §3.1.2 Eq.(16)，PDF 第 4 页、印刷第 14051 页](https://aclanthology.org/2023.findings-emnlp.936.pdf) 将每个通道的 WKV 写成

$$
\operatorname{wkv}_t=
\frac{
\sum_{i=1}^{t-1}e^{-(t-1-i)w+k_i}\odot v_i
+e^{u+k_t}\odot v_t
}{
\sum_{i=1}^{t-1}e^{-(t-1-i)w+k_i}
+e^{u+k_t}
}.
$$

这里 $w,u,k_i,v_i\in\mathbb R^d$，加法、指数和除法都**逐通道执行**。$w$ 控制时间衰减（按正的衰减率写法），$u$ 是只针对当前 Token 的 Time-First Bonus。历史求和最大索引为 $t-1$，**当前 Token 不进入这条历史衰减和式，而走单独的 $e^{u+k_t}$ 支路**。

为什么必须特别写这一项？假设当前 Token 为一条边界指令，模型可能需要它立即影响输出，而不希望等下一步成为历史之后才参与衰减权重。$u$ 允许当前 Token 的贡献独立于长期记忆时间常数调整。这也是 RWKV 和只写 $S_t=\gamma S_{t-1}+k_t^\top v_t$ 的 [RetNet](A061-retnet.md) 的关键不同。

**数学提醒**：本段按论文式写 $e^{-jw}$；当模型内部使用 Log-Decay $w_{\rm raw}$ 时，可能参数化成 $e^{w_{\rm raw}}\in(0,1)$。不能把论文“时间衰减指数”与某个源码的预激活 Log-Decay 参数当作同一正负号来抄。

## 逐项推导递推：WKV 的历史分子、分母究竟怎样维护？

定义用于**即将计算第 $t$ 步输出**的历史状态

$$
A_t=\sum_{i=1}^{t-1}e^{-(t-1-i)w+k_i}\odot v_i,\qquad
B_t=\sum_{i=1}^{t-1}e^{-(t-1-i)w+k_i}.
$$

因此当前输出先读取

$$
\operatorname{wkv}_t=
\frac{A_t+e^{u+k_t}\odot v_t}
{B_t+e^{u+k_t}}.
$$

再把当前 Token 写入历史，给下一步使用：

$$
\boxed{
A_{t+1}=e^{-w}\odot A_t+e^{k_t}\odot v_t,\qquad
B_{t+1}=e^{-w}\odot B_t+e^{k_t}.
}
$$

这就是从原论文 Eq.(16) 得到的精确逐 Token 递推，前提是采用相同 $w,k,u$ 的参数化与同一初始状态。**每个通道只需分子、分母各一份缓存**，而不必存下所有过去的 $k_i,v_i$。

### 三步计算：不要把它误写成“只对最后一个 Value 做指数平滑”

用一维极简例子，取 $e^{-w}=1/2$、$u=0$、$k_1=k_2=k_3=0$、Value 依次为 $[2,10,4]$。令历史初始 $A_1=B_1=0$。

- 第 1 步：$\mathrm{wkv}_1=(0+1\cdot2)/(0+1)=2$；写历史后 $A_2=2,B_2=1$。
- 第 2 步：$\mathrm{wkv}_2=(2+10)/(1+1)=6$；写历史后 $A_3=0.5(2)+10=11,B_3=0.5(1)+1=1.5$。
- 第 3 步：$\mathrm{wkv}_3=(11+4)/(1.5+1)=15/2.5=6$；写历史后 $A_4=9.5,B_4=1.75$。

直接用第三步的历史公式：先前第 1 项的系数 $e^{-w}=0.5$，贡献 $0.5\cdot2=1$；第 2 项贡献 $10$；当前第 3 项贡献 $4$，分母 $0.5+1+1=2.5$，输出还是 6。

这里与普通 EMA 的区别十分具体：**归一化分母也在积累历史权重**，而且当前 Token 的 Bonus 权重可以与其下一步进入历史的权重不同。此数值例子是教学构造，不是 RWKV-4 的训练预测。

## 为什么需要数值稳定的重参数化，而不能直接在实际模型里维护 $e^{k_t}$？

如果 $k_t$ 的某个通道值达到 120，直接使用 FP32 的 $\exp(120)$ 已超出普通浮点可表示范围，分子和分母都会溢出。这不是新算法的语义问题，而是指数权重实现中的数值稳定性问题。

可以改成一个类似 Log-Sum-Exp 的有界表示：记录历史共同的指数最大值 $p_t$，并保存以 $e^{-p_t}$ 缩放过的 $\tilde A_t,\tilde B_t$。每次更新先求

$$
p_{t+1}=\max(p_t-w,k_t)
$$

再让旧历史乘 $e^{p_t-w-p_{t+1}}$、当前贡献乘 $e^{k_t-p_{t+1}}$。由于两个指数都不大于 0，指数值均在 $[0,1]$，避免直接巨大 $\exp(k_t)$。**$u$ 的当前 Bonus 只进入本步的读出归一化，不应错误地写入未来历史状态**。

这类稳定更新要求每个通道维护额外 Log-Max 状态 $p_t$；因此说“RWKV 只有一个隐藏向量”并不准确。原始论文 Appendix D/I 分别给出时间顺序推理与当前 Token 优待机制，对应的状态需要完整复现，不能只保留 $A_t$ 忘掉 $B_t$ 和 $p_t$。

## Token Shift 为什么不是 Attention，也不是把整个历史一股脑平均？

上面讨论的 $\mathrm{wkv}_t$ 已可递推，但 $k_t,v_t$ 的生成还需要让网络看到当前 Token 与前一步的局部差异。原文 §3.1.1 Eq.(11)–(15) 在 Time-Mixing 与 Channel-Mixing 中使用

$$
\begin{aligned}
\tilde x_t^{(r)}&=\mu_r\odot x_t+(1-\mu_r)\odot x_{t-1},\\
r_t&=W_r\tilde x_t^{(r)},\\
\tilde x_t^{(k)}&=\mu_k\odot x_t+(1-\mu_k)\odot x_{t-1},\\
k_t&=W_k\tilde x_t^{(k)}
\end{aligned}
$$

$V$ 使用它自己的 $\mu_v$，Channel-Mixing 的 $r',k'$ 也各有混合参数。这些 $\mu$ 是可学习的**逐通道 Mixing 系数**，不是所有 $r,k,v$ 都简单共享同一数值的 $0.5$。

若 $\mu_k=1$，Key 完全取当前；若 $\mu_k=0$，Key 完全取前一步。**当前/上一步的混合只处理局部时间连接；真正远历史来自 WKV 的递推统计**。原文 Figure 3 中从相邻 Token 画向 Time Mix 与 Channel Mix 的箭头应按这个职责解释，不可把 Token Shift 自己描述成全局历史注意力。

## Receptance 为什么是“读出门”，而不是 WKV 的 Query？

原论文 §3.1.3 Eq.(17) 将 Time-Mixing 输出写为

$$
o_t=W_o\big(\sigma(r_t)\odot\operatorname{wkv}_t\big).
$$

$\sigma(r_t)$ 在 $[0,1]$ 之间逐通道缩放当前的 WKV 读出，具有**门控**含义；它不是对历史每个 $k_i$ 分别计算 $q_t^\top k_i$ 的标准 Query。两者控制信息的粒度不同：Softmax Query 可为每个历史位置指定不同权重，RWKV Receptance 在历史已经被汇总为 $d$ 维 WKV 后，只控制这 $d$ 个通道的读出强度。

Channel-Mixing 则见原文 Eq.(18)：

$$
o'_t=\sigma(r'_t)\odot
\big[W'_v\,(\max(k'_t,0))^2\big].
$$

即 Squared-ReLU Feedforward 再乘 Receptance Gate。它不是第二个 WKV 历史池，也不是传统 Transformer FFN 可以无条件逐层替换后的严格等价操作。

~~~{mermaid}
flowchart TD
 X["每层输入 x_t 与 x_{t-1}"] --> TM["Token Shift / Time-Mix：产生 r,k,v"]
 TM --> WKV["WKV：分子 A、分母 B、Log-Max"]
 TM --> REC["Receptance Gate：sigmoid(r)"]
 WKV --> OUT["门控读出与投影"]
 REC --> OUT
 OUT --> RES["残差与 LayerNorm"]
 RES --> CH["Channel-Mix：Squared ReLU + Receptance"]
 CH --> Y["下一层 / LM Head"]
~~~

*自制解释图，非论文 Figure 3。两条 Time/Channel Mix 都带 Token Shift，但历史聚合递推由 WKV 的额外状态完成。*

## 为什么原论文 Figure 3 的网络图具有“两个状态时间尺度”？

[RWKV 正式版 Figure 3（PDF 第 4 页，印刷 14051）](https://aclanthology.org/2023.findings-emnlp.936.pdf) 画了连续的 “My → name → is → Bob” 中间的三步，每步包括 **Time Mix** 与 **Channel Mix** 两个带残差的模块；Token Shift 箭头连接前后层输入，States 箭头连接 WKV 的长期历史。这张图要读出：

1. LayerNorm → Time Mix → Residual，负责长历史加权与读出；
2. LayerNorm → Channel Mix → Residual，负责 Token 内的非线性特征变换；
3. Token Shift 只需要缓存上一步输入，各层的 WKV 则维护跨更多 Token 的加权历史。

Figure 2 与 Figure 3 解决的问题不同：前者展示 Time/Channel Mix Block 的网络连接，后者把三枚输入按时间展开。它们都**不**是在测模型的 Benchmark Speedup。

### Tensor Shape 的精确账本

在第 $\ell$ 层，令 Hidden Dimension 为 $d$、Batch 为 $B$。Time-Mixing 当前 $r,k,v$ 为 $[B,d]$，WKV 历史状态 $A,B,p$ 各为 $[B,d]$（注意这里分母状态 $B$ 与 Batch 变量同名，仅在叙述中复用，工程代码应区别命名）。Token Shift 额外需要 $x_{t-1}:[B,d]$。

所以 **RWKV-4 的上述时间记忆部分每层为若干 $O(Bd)$ 状态**，不随历史长度增加；但 LayerNorm、Channel-Mixing、多个 Transformer-like Projection 的参数量仍与 $d^2$ 成正比。与 [RetNet](A061-retnet.md) 保存 $[B,d_k,d_v]$ 的外积矩阵并不完全相同。

## 平行训练的真正含义：有没有把时间上的递推依赖全部消除？

[论文 §3.2（PDF 第 4 页）](https://aclanthology.org/2023.findings-emnlp.936.pdf) 说明 $r,k,v$ 的投影与一般 Transformer 一样能同时对所有 Token 计算，主要 FLOPs 来自 $O(BTd^2)$ 的线性矩阵乘；WKV 的时间传播虽然需要 Scan，但其更新是每通道元素级运算，复杂度为 $O(BTd)$。

不能把原文“time-parallel training”误写成“任何实现都使用全局 $T\times T$ Attention 矩阵”或“时间轴依赖在代数上不存在”。论文正式版写明 WKV 会执行扫描，作者还为之实现了计算核；相对于 Transformer 的优势是**大多数计算可并行投影、而历史状态不增长**，不意味着无论 Hardware/Batch 都能得到同样训练吞吐。

在给定初始状态和相同舍入顺序的条件下，训练时用整段 Scan、推理时逐 Token 更新，计算同一时间加权表达式。若采用不同 Kernel、低精度重排，存在浮点舍入偏差；不能因此宣称训练过程的所有层和每个 Token 都逐位一致。

## 实验该从 Figure 6、Table 2 看什么？

[原文 Table 2（正式 PDF 第 5 页，印刷 14052）](https://aclanthology.org/2023.findings-emnlp.936.pdf) 列出了 **169M、430M、1.5B、3B、7B、14B** 的六种网络深度、宽度、参数量和每 Token FLOPs。7B 的模型宽 4096、深 32；14B 宽 5120、深 40。训练总计采用 **330B Pile Tokens**。这支撑“RWKV-4 的架构能够规模化预训练”，但单凭参数表不能证明任何模型质量提升。

[Figure 6（正式 PDF 第 7 页，印刷 14054）](https://aclanthology.org/2023.findings-emnlp.936.pdf) 的横轴是 Context Length（1K→8K），纵轴为 Pile Test Loss，展示同模型的长上下文扩展后测试 Loss 下降；作者将 7B/14B 的 Context 从 1024 分阶段延长至 8192。这个图只能说明在其 Continued Training/评估中更长上下文有收益，不能证明 RWKV 有**无限精确记忆能力**。因为模型状态仍是有限通道向量。

其余模型比较需要留意 Benchmark 与参数规模：作者报告最大 14B Dense RNN 与类似规模 Transformer 的性能具有竞争力，但这些对照通常涉及不同数据和训练过程；可比较的是具体表中的任务、Token 预算和评估协议，不能套用“RWKV 必然等于 GPT 的全部能力”。

## 失效边界：当查询需要“取回具体某一条旧记录”会怎样？

设很早的上下文里出现了很多独立键值映射，而最后一个问题要取出其中任意一条。标准 Attention 至少理论上可以针对当前 Query 重新按 Key 打分；RWKV-4 已经把所有旧 Value 压成每通道一个加权分子与分母，如果两个键值对在这些统计上互相覆盖，**没有额外的可逆数据存储就不可能精确还原每一条**。

这不是说 RWKV 不能处理实际检索任务，而是说“固定有限状态 + 加权汇总”在数学上不等于可无损存储无限随机键值对。门控可以改善特征组合，但无法违反有限容量约束。

把本篇放进更广的架构演化：

- [RetNet](A061-retnet.md) 通过多尺度 $\gamma_h$、矩阵状态和 GroupNorm 建立 Parallel/Recurrent/Chunk 三重视角；
- **RWKV-4** 用逐通道 WKV 的指数归一化及 Time-Mix/Channel-Mix/Token Shift 构成另一条 Dense RNN Backbone；
- [xLSTM](A063-xlstm.md) 把归一化的门控更新形式化，并让 mLSTM 引入真正的 Key–Value **矩阵记忆**，与 RWKV-4 在状态容量、读出形式上产生实质差异；
- [Mamba](A059-mamba.md) 则从 Selective State Space 出发，用 Token-dependent $\Delta,B,C$ 改变状态衰减和投影，而非使用 WKV 的固定时间衰减加权分母。

所以这篇论文的核心不是“RNN 比 Attention 快”，而是**如何把时间加权、内容值、归一化和读出门控组织成可稳定预训练、又能固定状态解码的完整语言模型 Block**。
