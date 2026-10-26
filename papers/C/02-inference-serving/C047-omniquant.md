# OmniQuant：极低比特量化为何需要学习裁剪边界与等价变换？

> **原始论文**：[OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c6483c8a68083af3383f91ee0dc6db95-Abstract-Conference.html)，ICLR 2024。本文技术细节依据 [arXiv v3（2024-03-18）](https://arxiv.org/html/2308.13137) 的 §3、Appendix A1 与正式版 Figure 3；代码参见 [作者仓库](https://github.com/OpenGVLab/OmniQuant)。


前置：[SmoothQuant](C040-smoothquant.md)、[RPTQ](C046-rptq.md)、[GPTQ](C041-gptq.md)；下一篇：[SpQR](C048-spqr.md)。

## 最大值并不总是量化范围的最佳上界

设一组数据大部分落在 $[-1,1]$，只有少数值达到 10。如果在 2-bit 对称或非对称量化中执意完整表示 10，会让大量正常数值只能投影到非常稀疏的网格。适度裁剪 10 反而可能降低**平均输出重建误差**，尽管必然提高异常点自身的截断误差。

这个权衡是为什么从简单的最大值统计走向校准学习是合理的：**要优化最终层输出误差，而不是强求每个浮点极值都能被无损表示**。

## 用四个样本证明“裁剪 Outlier”并不总是正确

先定义 2-bit 非对称均匀量化，共有 4 个码点。若样本是 $[0,0.1,0.2,4]$，以区间 $[0,4]$ 编码，步长 $\Delta=4/3$，前三个小值都会被映到 0，平方误差总和为

$$
E_{\mathrm{noClip}}=0^2+0.1^2+0.2^2+0^2=0.05.
$$

若改为裁剪到 $[0,0.3]$，前三个值精确对应码点 0、0.1、0.2，但最后的 4 只能表示为 0.3，平方误差变成

$$
E_{\mathrm{Clip}}=(4-0.3)^2=13.69.
$$

这种数据下裁剪显然不合理。**现在让前三个数各出现 1000 次，Outlier 4 只出现一次**：不裁剪的误差总和变为 $1000(0.1^2+0.2^2)=50$，裁剪仍只有 13.69。于是同样的 Clip 策略从劣解变成优势解。LWC 的核心不是“总是裁掉离群值”，而是依据校准输出目标权衡尾部截断和普通值 Round 误差。上面数字仅说明选择逻辑，不是论文 Figure 的实验数据。

## 两个关键可学习部件

OmniQuant 提出 **Learnable Weight Clipping (LWC)** 与 **Learnable Equivalent Transformation (LET)**。它并不重训整个大型模型，而使用少量校准样本优化用于量化的参数。

对于 $b$-bit 非对称量化，示意性的裁剪量化形式为

$$
\Delta=\frac{u-l}{2^b-1},\qquad
\widehat w=l+\Delta\cdot
\operatorname{round}\frac{\operatorname{clip}(w,l,u)-l}{\Delta}.
$$

LWC 学习左右裁剪界 $l,u$ 相关的量化参数。范围缩小意味着更多大值被 Clip，但内点分辨率提高；这两类误差必须合起来考虑。作者的设计不是直接为每层学习任意无约束的 $l,u$，而是围绕预量化 Weight 的 Min/Max **学习两个相对裁剪强度**。原论文 §3.2 Eq.(2) 更具体地给出

$$
\begin{aligned}
h&=\frac{\gamma\max(W)-\beta\min(W)}{2^N-1},\\
z&=-\operatorname{round}\left(\frac{\beta\min(W)}h\right),\\
W_q&=\operatorname{clamp}\left(\operatorname{round}(W/h)+z,\ 0,\ 2^N-1\right),
\end{aligned}
$$

其中 $\gamma,\beta\in[0,1]$，通过 Sigmoid 对潜在参数做映射，$N$ 为位宽；部署时仍要以 $(W_q-z)h$ **反量化**才回到近似的浮点权重。若 $\gamma=\beta=1$，退化为常规 Min-Max；缩小二者不保证逐点误差都降低，损失函数会决定如何牺牲尾部以保留高密度区间。$h=0$、全正或全负区间的 Zero Point、有限精度饱和，都需要在 Kernel 实现中做边界处理。**这是作者的 LWC 参数化，不是文章前面用任意区间 $[l,u]$ 举例的同一公式。**

LET 利用浮点等价变换，例如对角缩放

$$
XW=(XD^{-1})(DW),\qquad D_{ii}>0.
$$

未经量化时，这个输出完全不变；一旦量化后，因为 Weight/Activation 分布发生变化，整体舍入误差也会变化。LET 校准决定怎样把数值难度在不同操作之间迁移，而不必只依赖手工固定的 Max-Abs 规则。

~~~{mermaid}
flowchart LR
 CAL[Calibration Input] --> FP[原始 FP16 层输出]
 CAL --> LET[可学习等价变换 LET]
 LET --> Q[Weight / Activation 量化]
 LWC[可学习剪裁范围 LWC] --> Q
 Q --> LOSS[输出重建损失]
 FP --> LOSS
 LOSS --> PARAM[更新 LET/LWC]
 PARAM --> LET
 PARAM --> LWC
~~~

*自制校准闭环图，非原论文 Figure。校准时调的是 LET/LWC 等辅助参数而非全量原始 Weight；作者原文 Figure 3 标出了训练时可学习（火焰）、固定（雪花）与可在量化后合并的变换。*

## LET 不止是在参数表里写一个 Scale

对于浮点 Linear：

$$
XW+b=(XD^{-1})(DW)+b.
$$

如果还考虑可吸收到 Bias 的通道平移向量 $c$，则

$$
XW+b=(X-c)W+(cW+b).
$$

两类变换都在浮点意义下保持输出不变，但有不同的工程融合要求：$D$ 是逐通道乘法，$c$ 是逐通道平移；若相邻算子没有可以合并的 Bias，或中间激活被两个分支复用，就不能想当然地声称所有移动都是零 Runtime Cost。OmniQuant 的 Learnable Equivalent Transformation 是在合法的模型连接约束下优化这些重参数化自由度，不是添加一个必须永久运行的新网络 Head。

原论文 [§3.3 Eq.(3)](https://arxiv.org/html/2308.13137) 实际把两者**同时**写进线性层。令 $X:[T,C_{\rm in}]$、$W:[C_{\rm in},C_{\rm out}]$，$s,\delta:[1,C_{\rm in}]$，$\tilde X=(X-\delta)\oslash s$，$\tilde W=s\odot W$，$\tilde b=b+\delta W$，则

$$
\begin{aligned}
\tilde X\tilde W+\tilde b
&=((X-\delta)D^{-1})(DW)+(b+\delta W)\\
&=(X-\delta)W+b+\delta W=XW+b .
\end{aligned}
$$

这里把 $\delta$ 视为按 Token 维广播的行向量；增补的 $\delta W$ 是**输出通道 Bias**，其 Shape 为 $[1,C_{\rm out}]$，不能误写成 $W\delta$。用一个自编数值例子核对方向：取 $X=[5,2]$、$W=[2,3]^\top$、$b=1$、$\delta=[1,0]$、$s=[2,1]$。原始输出为 $5\cdot2+2\cdot3+1=17$；变换后输入 $[2,2]$、权重 $[4,3]^\top$、偏置 $3$，输出 $8+6+3=17$。这只是**浮点恒等式核查**，不是声称量化后也一定得到 17。

LET 还处理 Attention 的 $QK^\top$。论文 Eq.(5) 使用同一可学习向量 $s_a$，令 $\tilde Q=Q\oslash s_a$、$\tilde K=K\odot s_a$，于是

$$
(\tilde Q)(\tilde K)^\top
=QD_a^{-1}D_aK^\top=QK^\top.
$$

按特征维同步做反向缩放才能保持 Logit 不变；只缩放 Q 而不缩放 K 会改变 Softmax 前的点积。为了执行效率，论文将该缩放吸收到 Q/K 的上游 Linear Projection 参数。更重要的是，§3.3 明确**没有**把同样的线性 LET 无差别应用到 FFN 第二层（非线性后的稀疏特征会造成不稳定），因此本文的泛化示意不能替代逐算子实现约束。

### Round 不可微，优化器为什么还能更新 LWC？

硬量化函数的 Round 在区间内部几乎处处导数为零。训练量化参数时，优化通常采用 STE 或其他替代梯度。必须区分**前向使用离散量化值**与**反向用近似梯度调整 Scale/Clip**。后者是优化手段，并不意味最终部署输出在数学上和原 FP16 完全一致。

相对于只用 $|W-Q(W)|$ 的逐权重误差，真正需要优化的是包含 Attention、Norm 和 FFN 相互作用的 Block Output 重建；但重建误差减小也不意味着自动提升全部语言任务准确率，因此仍要独立评测 PPL 和下游任务。

## 为什么称为 PTQ，却还有优化？

Post-Training Quantization 的定义是对已有预训练模型做量化处理，而不是“不准出现任何梯度计算”。OmniQuant 优化少量量化相关参数，不等于将所有 FP16 权重重新进行大规模任务微调。对大模型而言，这可以比 Full Fine-Tuning 廉价，但相对 RTN/固定 Scale 方案有额外校准时间与内存消耗。

原文 §3.1 Eq.(1) 指定的是 **Transformer Block 级输出重建**，而非每个 Weight 单独最小化 $|w-\hat w|$。把已冻结的原始权重记作 $W$，可学习参数分别记作 $\Theta_{\rm LWC},\Theta_{\rm LET}$，形式为

$$
\min_{\Theta_{\rm LWC},\Theta_{\rm LET}}
\left\|F_{\rm FP}(X;W)-
F_{\rm quant}(X;Q_w(W;\Theta_{\rm LWC},\Theta_{\rm LET}),Q_a(X;\Theta_{\rm LET}))\right\| .
$$

这是根据论文 Eq.(1) 明确区分教师输出与量化输出的符号写法，实际源码还决定 Batch Reduction 和辅助损失细节。原论文 **Appendix A1 Algorithm 1** 的顺序是：对**当前一个 Block** 初始化 LWC/LET 参数 → 从校准样本得到浮点目标输出 → 用 STE/近似梯度仅更新这些辅助参数 → 将学得的变换折叠回真实 Weight/Bias/量化参数 → 冻结该块并转向下一个 Block。不是把整个模型的 FP16 Weight 当作训练变量，也不是“每层独立优化完成后还需要永久运行平移/缩放网络”。

~~~{mermaid}
flowchart LR
 X["校准输入 X"] --> FP["固定 FP Block：目标输出"]
 X --> Q["当前 Block：LWC Clip + LET 缩放/平移"]
 Q --> R["低比特 Qw/Qa → Block 输出"]
 FP --> E["重建误差"]
 R --> E
 E --> U["反向更新 LWC/LET 参数"]
 U --> Q
 Q --> F["校准结束：变换折叠并量化"]
 F --> N["冻结并进入下一 Block"]
~~~

*自制训练/部署区分图，非原论文 Algorithm 1。循环只在离线校准阶段发生，部署的主网络保留正常的低比特算子执行路径。*

## 最容易误判的三个结论

首先，等价变换保证的是量化前的数学恒等，不是量化后输出毫无误差。其次，层重建指标改善不必然在每项语言任务上都改善。第三，“量化参数经过学习”不意味着部署时任意 Kernel 都能直接支持获得的 Scale、Zero Point 或 Clip 格式。

[ICLR 2024 原论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c6483c8a68083af3383f91ee0dc6db95-Abstract-Conference.html) 系统研究了极低比特和不同 Weight/Activation 组合。做横向比较时应固定 W/A 位宽、校准样本、量化 Group Size 与硬件，再分别消融 LET、LWC。下一篇 [SpQR](C048-spqr.md) 选择另一种出路：不尝试把所有稀有异常值挤进同一个低 bit 格点，而让少数它们单独存为高精度。

## 为什么正式论文的耗时数字很重要？

[ICLR 2024 OmniQuant](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c6483c8a68083af3383f91ee0dc6db95-Abstract-Conference.html) 明确给出一种条件：使用 128 个校准样本，在单张 A100 40GB 上处理 LLaMA-2 7B–70B，量化可在大约 1–16 小时内完成。它同时研究 W4A4、W6A6、W4A16、W3A16、W2A16 等不同配置。不能把“1–16 小时”说成一次 70B 量化固定只需一小时，也不能将不同位宽的 PPL 归入一列比较。

更具体地，[ICLR 2024 Figure 1（PDF 第 2 页）](https://arxiv.org/pdf/2308.13137) 的三个 Panel 对应不同证据：(a) 在 LLaMA-7B W4A4 设置下画出“精度—时间/数据”的权衡；(b)(c) 比较 LLaMA-13B 不同 W/A 位宽的 WikiText2 PPL。按原文 **Table 1 的 LLaMA-1 13B / W2A16 / Per-Channel Weight** 条件，OmniQuant PPL 为 **13.21**，GPTQ 为约 **5.5×10³**。这是一项特定模型、校准与评估配置的极端差异，而不是声称 OmniQuant 在所有位宽都会有几百倍效果。**Figure 2（PDF 第 3 页）** 则是固定/可学习变量和量化流水线结构示意；它本身不构成性能实验。

如果希望判断为什么“额外优化参数”值得做，须进一步读 [Table 2（§4.3）](https://arxiv.org/html/2308.13137)：在 **LLaMA-1 7B、W4A4、6 项 Zero-Shot 准确率的平均值**这一共同口径下，SmoothQuant 是 **38.41%**，OmniQuant 是 **52.65%**，而 FP16 为 **64.09%**。它支持作者在**这一特定位宽、模型、测试集**的精度收益，并不等价于百分比相同的 Token/s 增加。再看正式 Figure 3（PDF 第 4 页），其左、中、右分别画出 Weight 的可学习裁剪和变换、Q/K/V Attention 及 FFN 的量化位置；那些“可在校准后合并”的节点才说明部署为什么不需要额外独立校准网络。

验证作者技术选择的三个关键消融是：只开 LWC、只开 LET、两个同时开；固定 $b_W,b_A$、相同 Calibration 与 Group Size；分别测层重建误差、PPL 与实际 INT Kernel 时间。由此才能判断校准变量自由度的收益是否抵得上额外离线优化时间。
