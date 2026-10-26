# AWQ：为什么用激活衡量权重重要性，而不是直接看权重绝对值？

> **原始论文**：[AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html)，MLSys 2024。 [官方/作者 PDF](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf)。


前置：[GPTQ 近似二阶补偿](C041-gptq.md)；对照：[SmoothQuant 的激活/权重难度迁移](C040-smoothquant.md)。

## 一个小权重也可能极重要：为什么不能看 $|W|$ 决定保留精度？

模型输出 $Y=XW$ 中，同样大小的权重误差 $\Delta W$，会被其对应的激活 $X$ 放大。举例一个权重 $w_1=0.1$、对应激活 $x_1=100$，另一个权重 $w_2=10$、对应激活 $x_2=0.01$。若它们都产生绝对量化误差 $0.02$，在这个 Token 上的输出扰动分别约 $2$ 与 $0.0002$。因此“权重越大越重要”不是可靠准则。

AWQ 的关键实证发现是：有一小部分 **Salient Weight 通道** 对量化精度极其敏感；观察 Activation 幅度有助于识别并保护它们。论文强调与仅凭 Weight 自身幅值选通道的方案不同，Activation 更能反映部署输入时误差的实际放大程度。

## 保护 1% 是否意味着把那 1% 单独做 FP16？

常见误解是“AWQ = 99% 权重 4-bit + 1% 权重 FP16”。作者确实观察到保护少量重要权重能显著减小误差，但实际的高效算法**不是一定用显式的高精度稀疏支路**；它通过 Per-Channel Scaling 放大显著通道，让它们在量化网格上拥有更好的表示，然后把相反尺度合并到激活路径。

对输入通道选择 $s_j>0$，数学重参数化与 [SmoothQuant](C040-smoothquant.md) 同样满足

$$
XW=(XD^{-1})(DW),\quad D=\operatorname{diag}(s_1,\ldots,s_d).
$$

但**优化目标不同**：
- SmoothQuant 关注迁移激活离群范围，让 Activation + Weight 都适合 W8A8；
- AWQ 关注 Weight-Only 低比特量化的敏感通道，用激活统计指导 Scale 搜索，最小化输出/重建误差，而不量化激活为 W8A8 主目标。

AWQ 的条件优化可抽象为

$$
s^\star=\arg\min_s
\mathbb E_{X_{\rm calib}}
\left\|XW-(XD^{-1})\,Q(DW)\right\|_F^2.
$$

这个式子说明目标是**量化后**重建误差，不是未量化前的等价恒等式；原文对 Scale 的参数化/搜索方式还有更具体的近似策略。

~~~{mermaid}
flowchart LR
 X[校准激活] --> S[识别敏感通道/缩放搜索]
 W[原始权重] --> S
 S --> D[按输入通道放大权重]
 D --> Q[Weight-Only 4-bit 量化]
 Q --> K[高效 INT4 推理 Kernel]
~~~

*自制解释图，非原论文 Figure。原论文 Figure 2 对比 RTN、1% FP16 Mixed Precision 与 AWQ 的机制；Figure 3 才是 TinyChat 的 Prefill/Generation 性能瓶颈分析。低比特量化收益不应全部归因于推理框架。*

## 一个真的能看见收益、又能看见边界的量化手算

为避免一上来使用真实模型几千维的 Group Quantizer，构造只有**两个输入通道、一个输出通道**的教学例子。设 $X=[10,1]$、$W=[0.4,1]^\top$，原始输出为 $XW=5$。假设两枚 Weight 属于同一个量化组，采用一个特别简单的**对称 2-bit 教学量化器**：

$$
\Delta=\max_j|w_j|,\qquad
Q(w_j)=\Delta\,\operatorname{clip}\left(
\operatorname{round}(w_j/\Delta),-1,1\right).
$$

合法反量化级只有 $\{-\Delta,0,\Delta\}$；这**不是作者 INT3/INT4 的实际量化格式**，只是能手算量化步长和通道缩放的最小实验。先令 $D=I$，则 $\Delta=1$、$Q(W)=[0,1]^\top$，量化输出为 $1$，与真值相差 $4$。

现在只对激活较大的第一通道乘以权重尺度 $s_1=3$，同时把输入除以 3，得到 $D=\operatorname{diag}(3,1)$、$DW=[1.2,1]^\top$、$XD^{-1}=[10/3,1]$。此时组步长为 $\Delta'=1.2$，两枚权重量化后都变成 $1.2$：

$$
\widehat y=(XD^{-1})Q(DW)
=\frac{10}{3}\times1.2+1\times1.2
=5.2.
$$

浮点重参数化仍严格输出 $5$，而量化误差从 $4$ 降为 $0.2$。这里的收益不是“Weight 数值变大了就天然更准确”，而是**本组量化格点、激活大小和逆向缩放一起改变**了误差。

把 $s_1$ 再增大到 10 反而使 $DW=[4,1]^\top$、组步长变成 4，于是 $Q(DW)=[4,0]^\top$、$XD^{-1}=[1,1]$，输出退回 $4$，误差回升为 1：

| 第一通道权重尺度 $s_1$ | 组步长 $\Delta$ | 量化输出 | 绝对误差 |
|---|---:|---:|---:|
| 1（不缩放） | 1 | 1 | 4 |
| 3（适度保护） | 1.2 | 5.2 | 0.2 |
| 10（影响其它通道） | 4 | 4 | 1 |

这组数值为教学构造，未运行模型或 Kernel；它证明**Scale 必须搜索而不是尽可能大**，也解释了原论文为什么不仅关心显著通道，还要保护同组的其它 Weight。

若按某一通道 $j$ 的量化权重误差 $\epsilon_j=Q(s_jw_j)-s_jw_j$，经过对应逆尺度的激活后，输出误差项可表达为

$$
\Delta y_j=x_j\,\frac{\epsilon_j}{s_j}.
$$

增加 $s_j$ 在有利的网格配置下可能减小有效误差 $\epsilon_j/s_j$，但 $\epsilon_j$ 本身也会因 Round、Group 范围改变，不能当作与 $s_j$ 无关的常数。AWQ 的 Scaling Search 正是平衡二者。

原文 [§3.2、Eq.(4)(5)](https://arxiv.org/html/2306.00978) 并不是分别对每个 $s_j$ 做无限制的梯度优化，而是先计算校准激活每个通道的**平均绝对幅度** $s_{X,j}$，取一个可搜索的参数族

$$
s_j=s_{X,j}^{\alpha},\qquad
\alpha^*=\arg\min_{\alpha\in[0,1]}
\left\|XW-(XD(\alpha)^{-1})Q(D(\alpha)W)\right\|_F.
$$

然后在 $\alpha\in[0,1]$ 上 Grid Search，并额外做 Weight Clipping 以减少量化 MSE。我们采用输入通道是 $X$ 的**列**的约定，故右乘 $D^{-1}$、左乘 $DW$；原论文 Eq.(4) 采用输入特征排在**行**的转置布局。$\alpha=0$ 对应不缩放，$\alpha=1$ 是所定义候选族中最激进的指数，不等于“所有 Weight 变 FP16”。

## 1% 保护实验究竟支持哪个结论？

[MLSys 2024 AWQ 正式论文](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) 通过将不同百分比的显著权重保留为高精度，显示保护少量高敏感项即可显著改善量化误差，这个**观察结果**是引入 Activation-aware Search 的动机；最终的 AWQ 实现不要求真的把整段稀疏权重都交由 FP16 Sparse GEMM 运算。读原论文 Figure 1 的显著权重保护现象时应关注保护比例和 Loss 的对应关系，后续 Figure/表的 TinyChat 速度又额外依赖 Kernel。

具体看 [MLSys 2024 / arXiv HTML 版的 Table 1 与 Figure 2](https://arxiv.org/html/2306.00978)：在 **INT3、Group Size 128、WikiText PPL** 下，OPT-6.7B 的普通 RTN 为 **23.54**，按激活幅度选 1% 通道保留 FP16 后为 **11.39**，按权重幅值选 1% 后仍为 **22.37**，随机选 1% 后为 **24.23**。PPL 越低越好。这个同模型、同位宽对照说明**怎样选通道**有决定性影响，而不是“任意保留 1% 权重就有效”。

但“保留 1% FP16”是揭示显著性的研究实验，不是最终 AWQ 的异构计算路径。原论文 Table 3 在同样 INT3-g128 的 OPT-6.7B 上报告基于 Scale Search 的 AWQ PPL **11.39**、固定 $s=2$ 时 **11.92**；搜索能适配不同层与模型，不能只用一个固定乘数代替。Table 2 更展示了一个反例：显著通道统一乘 $s=4$ 虽使其单通道误差尺度更小，但整体 PPL 从 $s=2$ 时的 **11.92** 退化为 **12.36**，对应非显著通道的量化步长也被改变。这是上面的教学例子在论文真实模型上对应的实验逻辑，不是两组手算数值本身来自论文。

要做公平消融，至少固定相同 Group Size 与 Bit Width 对比：无缩放 RTN、按 Weight AbsMax 缩放、按 Activation 搜索最佳 Scale。前两者是有价值的简单对照，第三者如果准确率更好，才支持“Activation 不只是口头设计动机”。

## 为什么强调“不依赖 Backpropagation/重建优化”？

GPTQ 的二阶补偿对 Calibration 数据构造误差模型并依次更新未量化权重；AWQ 则使用 Activation 驱动的通道重缩放搜索，不需要为量化做反向传播训练。对于大小不同、分布不稳定的指令模型或多模态模型，这可能减轻训练数据过拟合风险，但“无需反向传播”**不表示完全不使用 Calibration 数据**。它仍要通过一小批 Activation 统计选取显著性与 Scale。

同样，W4A16（Weight-Only）与 W8A8 使用的是不同 Kernel 路线；某模型用 AWQ 后是否比 SmoothQuant 快，取决于设备 INT4 解包效率、Batch、Prompt/Decode 阶段的算力/带宽属性，不能只比较 bit 数。

## 哪些结果可以复现？

[MLSys 2024 正式论文](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) 报告将 4-bit AWQ 用于 Llama、MPT、Falcon、Mistral 和多模态模型等；结合作者 TinyChat 推理框架，边缘 GPU / 手机等实测相对 Hugging Face FP16 有超过 **3×** 的速度提升。该结果同时来自量化算法和特定低比特 Kernel 的实现，不能全归因于“找到那 1%”。

工程验证应该给出同一校准集、同样 4-bit Group Size 下 AWQ 与 RTN/GPTQ 的 Perplexity、任务指标与推理延迟；至少测 Activation 分布变化、异常通道比例、Kernel 是否可用。对于 [FlexGen](C043-flexgen.md) 的 CPU/disk offloading，Weight-Only 压缩还可能改变加载与带宽，而非仅影响算术。
