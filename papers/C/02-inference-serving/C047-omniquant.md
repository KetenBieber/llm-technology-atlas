# OmniQuant：极低比特量化为何需要学习裁剪边界与等价变换？

> **原始论文**：[OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c6483c8a68083af3383f91ee0dc6db95-Abstract-Conference.html)，ICLR 2024。


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

LWC 学习左右裁剪界 $l,u$ 相关的量化参数。范围缩小意味着更多大值被 Clip，但内点分辨率提高；这两类误差必须合起来考虑。

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

*自制校准闭环图，非原论文 Figure。请特别检查真正代码对相邻线性层、Attention、Norm 的 LET 参数合并方式。*

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

### Round 不可微，优化器为什么还能更新 LWC？

硬量化函数的 Round 在区间内部几乎处处导数为零。训练量化参数时，优化通常采用 STE 或其他替代梯度。必须区分**前向使用离散量化值**与**反向用近似梯度调整 Scale/Clip**。后者是优化手段，并不意味最终部署输出在数学上和原 FP16 完全一致。

相对于只用 $|W-Q(W)|$ 的逐权重误差，更合理的校准目标是 Block Output 重建：

$$
\min_{\theta_{\mathrm{LET}},\theta_{\mathrm{LWC}}}
\mathbb E_{x\sim\mathcal D_{\rm cal}}
\|F_{\rm FP}(x)-F_{\rm Quant}(x;\theta)\|_2^2.
$$

这里 $F$ 是一个包含注意力/Norm/FFN 关联路径的 Block 近似，校准变量 $\theta$ 则是两个部件所控制的参数。重建误差减小不意味着自动提升全部语言任务准确率，因此 PPL 与下游评测还要独立执行。

## 为什么称为 PTQ，却还有优化？

Post-Training Quantization 的定义是对已有预训练模型做量化处理，而不是“不准出现任何梯度计算”。OmniQuant 优化少量量化相关参数，不等于将所有 FP16 权重重新进行大规模任务微调。对大模型而言，这可以比 Full Fine-Tuning 廉价，但相对 RTN/固定 Scale 方案有额外校准时间与内存消耗。

可将一个校准目标写为

$$
\min_{\theta_{\rm LET},\theta_{\rm LWC}}
\mathbb E_{x\sim\mathcal D_{\rm cal}}
\|F_{\rm FP}(x)-F_{\rm quant}(x;\theta)\|_F^2.
$$

注意这是**教学简化的层重建目标**，实际优化调度、STE 和块划分请参照论文方法章节。

## 最容易误判的三个结论

首先，等价变换保证的是量化前的数学恒等，不是量化后输出毫无误差。其次，层重建指标改善不必然在每项语言任务上都改善。第三，“量化参数经过学习”不意味着部署时任意 Kernel 都能直接支持获得的 Scale、Zero Point 或 Clip 格式。

[ICLR 2024 原论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c6483c8a68083af3383f91ee0dc6db95-Abstract-Conference.html) 系统研究了极低比特和不同 Weight/Activation 组合。做横向比较时应固定 W/A 位宽、校准样本、量化 Group Size 与硬件，再分别消融 LET、LWC。下一篇 [SpQR](C048-spqr.md) 选择另一种出路：不尝试把所有稀有异常值挤进同一个低 bit 格点，而让少数它们单独存为高精度。

## 为什么正式论文的耗时数字很重要？

[ICLR 2024 OmniQuant](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c6483c8a68083af3383f91ee0dc6db95-Abstract-Conference.html) 明确给出一种条件：使用 128 个校准样本，在单张 A100 40GB 上处理 LLaMA-2 7B–70B，量化可在大约 1–16 小时内完成。它同时研究 W4A4、W6A6、W4A16、W3A16、W2A16 等不同配置。不能把“1–16 小时”说成一次 70B 量化固定只需一小时，也不能将不同位宽的 PPL 归入一列比较。

更具体地，[ICLR 2024 Figure 1（PDF 第 2 页）](https://arxiv.org/pdf/2308.13137) 的三个 Panel 对应不同证据：(a) 在 LLaMA-7B W4A4 设置下画出“精度—时间/数据”的权衡；(b)(c) 比较 LLaMA-13B 不同 W/A 位宽的 WikiText2 PPL。作者给的 W2A16 示例中 OmniQuant PPL 约 **13.21**，对照 GPTQ 在该极低 bit 设置下为 **3832**。这是一项特定模型、校准与评估配置的极端差异，而不是声称 OmniQuant 在所有位宽都会有几百倍效果。**Figure 2（PDF 第 3 页）** 则是固定/可学习变量和量化流水线结构示意；它本身不构成性能实验。

验证作者技术选择的三个关键消融是：只开 LWC、只开 LET、两个同时开；固定 $b_W,b_A$、相同 Calibration 与 Group Size；分别测层重建误差、PPL 与实际 INT Kernel 时间。由此才能判断校准变量自由度的收益是否抵得上额外离线优化时间。
