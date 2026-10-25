# OmniQuant：极低比特量化为何需要学习裁剪边界与等价变换？

> **原始论文**：[OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c6483c8a68083af3383f91ee0dc6db95-Abstract-Conference.html)，ICLR 2024。


前置：[SmoothQuant](C040-smoothquant.md)、[RPTQ](C046-rptq.md)、[GPTQ](C041-gptq.md)；下一篇：[SpQR](C048-spqr.md)。

## 最大值并不总是量化范围的最佳上界

设一组数据大部分落在 $[-1,1]$，只有少数值达到 10。如果在 2-bit 对称或非对称量化中执意完整表示 10，会让大量正常数值只能投影到非常稀疏的网格。适度裁剪 10 反而可能降低**平均输出重建误差**，尽管必然提高异常点自身的截断误差。

这个权衡是为什么从简单的最大值统计走向校准学习是合理的：**要优化最终层输出误差，而不是强求每个浮点极值都能被无损表示**。

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
