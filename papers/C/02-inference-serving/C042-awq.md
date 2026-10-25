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

*自制解释图，非原论文 Figure。原论文 Figures 2/3 更集中展示显著通道及 Scaling 对量化误差的影响；TinyChat 属于其配套推理系统，不能把它与量化算法本体视为同一个贡献。*

## 一个量化误差会随着通道 Scaling 如何变化的手算例子

设两个输入通道均为 1，一个 Weight 为 $w_1=0.1$，另一个为 $w_2=0.9$。对称整数 2-bit 为了便于演示，令统一 Scale $s_q=1$，权重只能映到 $\{-1,0,1\}$。简单 Round 得 $\hat w_1=0$、$\hat w_2=1$，输出残差可以互相抵消，但逐通道误差分别为 $0.1$ 和 $-0.1$。

现在将第一通道按 $D=\operatorname{diag}(5,1)$ 放大权重，获得 $Dw=[0.5,0.9]^\top$；若仍使用同样的离散网格并规定 `round(0.5)=1`，反量化前第一通道误差为 0.5，经输入侧除 5 后折算为 0.1。它没有改善原值。此反例说明：**对通道施加任何大于 1 的 Scale 不会自动提高量化精度**；量化 Group Scale 与 Round/Clip 的边界必须一起考虑。AWQ 的设计是根据 Calibration Activation 统计搜索有用的尺度，而不是“重要通道一律乘 5”。

若按某一通道 $j$ 的量化权重误差 $\epsilon_j=Q(s_jw_j)-s_jw_j$，经过对应逆尺度的激活后，输出误差项可表达为

$$
\Delta y_j=x_j\,\frac{\epsilon_j}{s_j}.
$$

增加 $s_j$ 在有利的网格配置下可能减小有效误差 $\epsilon_j/s_j$，但 $\epsilon_j$ 本身也会因 Round、Group 范围改变，不能当作与 $s_j$ 无关的常数。AWQ 的 Scaling Search 正是平衡二者。

## 1% 保护实验究竟支持哪个结论？

[MLSys 2024 AWQ 正式论文](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) 通过将不同百分比的显著权重保留为高精度，显示保护少量高敏感项即可显著改善量化误差，这个**观察结果**是引入 Activation-aware Search 的动机；最终的 AWQ 实现不要求真的把整段稀疏权重都交由 FP16 Sparse GEMM 运算。读原论文 Figure 1 的显著权重保护现象时应关注保护比例和 Loss 的对应关系，后续 Figure/表的 TinyChat 速度又额外依赖 Kernel。

要做公平消融，至少固定相同 Group Size 与 Bit Width 对比：无缩放 RTN、按 Weight AbsMax 缩放、按 Activation 搜索最佳 Scale。前两者是有价值的简单对照，第三者如果准确率更好，才支持“Activation 不只是口头设计动机”。

## 为什么强调“不依赖 Backpropagation/重建优化”？

GPTQ 的二阶补偿对 Calibration 数据构造误差模型并依次更新未量化权重；AWQ 则使用 Activation 驱动的通道重缩放搜索，不需要为量化做反向传播训练。对于大小不同、分布不稳定的指令模型或多模态模型，这可能减轻训练数据过拟合风险，但“无需反向传播”**不表示完全不使用 Calibration 数据**。它仍要通过一小批 Activation 统计选取显著性与 Scale。

同样，W4A16（Weight-Only）与 W8A8 使用的是不同 Kernel 路线；某模型用 AWQ 后是否比 SmoothQuant 快，取决于设备 INT4 解包效率、Batch、Prompt/Decode 阶段的算力/带宽属性，不能只比较 bit 数。

## 哪些结果可以复现？

[MLSys 2024 正式论文](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) 报告将 4-bit AWQ 用于 Llama、MPT、Falcon、Mistral 和多模态模型等；结合作者 TinyChat 推理框架，边缘 GPU / 手机等实测相对 Hugging Face FP16 有超过 **3×** 的速度提升。该结果同时来自量化算法和特定低比特 Kernel 的实现，不能全归因于“找到那 1%”。

工程验证应该给出同一校准集、同样 4-bit Group Size 下 AWQ 与 RTN/GPTQ 的 Perplexity、任务指标与推理延迟；至少测 Activation 分布变化、异常通道比例、Kernel 是否可用。对于 [FlexGen](C043-flexgen.md) 的 CPU/disk offloading，Weight-Only 压缩还可能改变加载与带宽，而非仅影响算术。
