# RPTQ：为什么只交换通道顺序就会影响量化精度？

> **原始论文**：Yuan 等，[RPTQ: Reorder-based Post-training Quantization for Large Language Models](https://arxiv.org/abs/2304.01089)，2023。[原文 PDF（重点见 §4、Figure 3、Table 3、Figure 4）](https://arxiv.org/pdf/2304.01089)。


前置：[SmoothQuant](C040-smoothquant.md) / [ZeroQuant](C045-zeroquant.md)，后续：[OmniQuant](C047-omniquant.md)。

## 某一维没有异常值，为什么还是量化不准？

设 $X\in\mathbb R^{T\times d}$ 的三个通道值域分别为 $[-100,100]$、$[-0.1,0.1]$、$[-1,1]$。若整体使用 8-bit 对称 Scale $s=100/127\approx0.787$，第二个通道基本都会被舍入成 0。真正矛盾不仅是个别离群点，而是**不同特征维度使用同一个量化尺度**。

RPTQ 将数值范围相似的 Channel 聚类，调整顺序，使相邻的一组维度共享更匹配的 Scale。这里的 Reorder 不是对 Token 次序重新排序——改变自回归时间次序会直接破坏模型含义——而是**只重排特征通道轴**。

## 复算一次 Channel Reorder：同样的 4-bit，误差为何不同？

假设四个通道的绝对最大激活值分别为 $[1,16,1,16]$，两个相邻通道共用一组对称 4-bit Scale，允许整数码为 $[-7,7]$。原排列为 `[1,16] | [1,16]`，两组的步长均为 $16/7\approx2.286$。对第一个通道的激活 $x=1$，舍入到 0 后误差为 1。

如果只把特征轴排列成 `[1,1] | [16,16]`，第一组步长为 $1/7$，该组里的 1 可精确落在整数 7，零量化误差；第二组仍使用 $16/7$。**同位宽、同 Group 数、同权重信息量，精度却可能显著改变**。这不是因为通道排序本身有语义作用，而是不同分布的数据重新归入了量化组。

但若 Weight 不相应调整，模型会被改坏：取 $x=[1,2]$、$w=[3,4]^\top$，原 $xw=11$。把输入换成 $[2,1]$ 却保留权重，会得到 10；同步把权重改成 $[4,3]^\top$ 才恢复 11。因此 $XP$、$P^\top W$ 的同步变换是该方法的正确性条件。

## 浮点输出为何仍然不变？

令 $P$ 是通道的排列矩阵，$PP^\top=I$，则线性层有严格等式

$$
XW=(XP)(P^\top W).
$$

对输入通道采取排列 $P$，对 Weight 沿乘法求和维度同步作逆排列，输出不变。但量化过程不同：令 $Q_G$ 是按连续 Group 共享 Scale 的量化器，则一般

$$
Q_G(XP)\ne Q_G(X)P.
$$

因为排列改变了哪些 Channel 落在同一个 Group 中，这正是分组量化精度变化的数学原因。浮点排列等价**不代表整数化后严格等价**。

~~~{mermaid}
flowchart LR
 X[原激活 X：各通道范围差异大] --> C[估计每个 Channel 的范围]
 C --> G[按范围聚类]
 G --> P[计算排列矩阵 P]
 P --> Q[相近通道共享组 Scale]
 W[原权重 W] --> IW[对应逆排列]
 Q --> Y[低比特 Linear]
 IW --> Y
~~~

*自制解释图，非原论文 Figure。它只表示数据流的因果顺序；真实论文的 Figure 3 还必须分别处理 Attention 的 Q/K 对齐、残差相加的通道对齐和 LayerNorm 输出写入布局。*

## 聚类目标不等于单纯“按最大值排序”

对各 Channel $j$ 的校准输入，先统计激活的上下界 $(l_j,u_j)$；若同一个 Group 内混合了强正、强负、窄范围与宽范围的通道，Group 的 Scale/Zero Point 都会受影响。RPTQ 的聚类围绕**每通道范围相似度**而非单独的绝对最大值。教学上排序已经能展示机制，但不应把一个 `sort(max_abs)` 伪代码冒充作者真实 Cluster Algorithm。

一个合理的执行顺序是：收集校准激活 → 统计每通道范围 → 按范围划分 Cluster → 计算通道排列和 Group Scale → 将重排吸收到相邻权重与 Norm 中 → 测量新的量化输出与原输出的差异。离线构建后的部署不应继续为每个 Token 重新做完整的通道统计和聚类。

这里要把“**按固定位置成组**”与论文真正的“**聚类后成组**”分开。原文 [§4.1 Eq.(3)](https://arxiv.org/pdf/2304.01089) 对校准张量 $X\in\mathbb R^{B\times N\times C}$ 的每个通道记录

$$
\ell_c=\min_{b,n}X_{b,n,c},\qquad
u_c=\max_{b,n}X_{b,n,c},\qquad
f_c=(\ell_c,u_c)\in\mathbb R^2.
$$

作者对 $f_c$ 做 **K-Means**，得到 $g$ 个通道集合 $S_1,\ldots,S_g$，再把同一簇的通道排列到一起，各簇分别用 Min-Max 计算非对称量化的 Scale 与 Zero Point。这里的 $g$ 是**簇数**，不等于“每簇固定有 $g$ 个通道”；K-Means 也不保证簇大小相同。正文前面两通道固定分组只是教学反例，不是作者的 K-Means 实现。

假设 INT4 非对称码为 $\{0,\ldots,15\}$，簇中所有校准值的最小/最大值为 $L_g,U_g$，可以写出普通的 Min-Max 量化步骤：

$$
\Delta_g=\frac{U_g-L_g}{15},\qquad
z_g=\operatorname{round}(-L_g/\Delta_g),\qquad
Q_g(x)=\operatorname{clip}\big(\operatorname{round}(x/\Delta_g)+z_g,0,15\big).
$$

这是为解释 Scale/Zero Point 的**教学表达**；当零不在原区间、范围退化为常数、或采用其它 Zero-Point 约定时，还需先修正边界。它说明为什么不仅要比较各通道 $\max|X|$：两个通道即便绝对最大值相同，$(\ell,u)=(-8,1)$ 和 $(-1,8)$ 也可能需要不同的非对称 Zero Point。

### 为什么 LayerNorm 允许在一些位置吸收排列？

在没有逐通道仿射参数时，LayerNorm 的均值和方差对通道置换不变。对排列矩阵 $P$：

$$
\mu(XP)=\mu(X),\qquad
\sigma^2(XP)=\sigma^2(X),
$$

$$
\mathrm{LN}_0(XP)=\mathrm{LN}_0(X)P.
$$

若 LayerNorm 含 $\gamma,\beta$，它们必须同步置换；在残差连接、多个注意力头或其他 Consumer 分叉处，不一定能免费把通道重排穿过所有算子。因此“将重排融合到 Norm/Weight”不是实现附注，而是方法能够避免额外 HBM Copy 的系统关键。

### Figure 3：为什么一层 Transformer 不能只有一个全局排列？

请直接对照 [RPTQ Figure 3（PDF 第 6 页）](https://arxiv.org/pdf/2304.01089)。原图绿色模块标出 FP16 操作，橙色模块表示量化计算；R1 至 R5 是**不同位置使用的通道排列索引**，不是把整段 Token 时序重排五遍。两次 LayerNorm 的结果可按排列索引写入目标地址；线性投影使用同步重排的 Weight，以免多出独立的运行时 Tensor Shuffle。

其中最容易出错的是 $QK^\top$：设 $Q,K\in\mathbb R^{T\times d}$，若两者在 Head 特征轴采用**同一个**置换 $P$，有

$$
(QP)(KP)^\top=QPP^\top K^\top=QK^\top.
$$

反之，若 Q 与 K 采用不同排列 $P_Q,P_K$，中间留下 $P_QP_K^\top$，通常不再等于单位矩阵。取单 Token 的 $q=[1,2]$、$k=[3,4]$，原内积为 $11$；仅交换 Q 的两维得到 $[2,1]\cdot[3,4]=10$，同时交换 Q/K 才恢复 $[2,1]\cdot[4,3]=11$。**这比“Weight 也跟着重排”多了一个跨张量的正确性约束。**

原文 §4.2 不是简单把 Q、K 各自按范围分成簇，而是把它们对应通道的 $Q_{\max},Q_{\min},K_{\max},K_{\min}$ 合成四维聚类特征，以便学得**相同排列**；Q/K 还按 Head 分开处理。Residual Add 两端也必须处在一致的坐标顺序，论文因此不会随便重排 Out Projection 和 FFN 最终输出的通道。Figure 3 说明“在哪些算子位置能重排”，**不证明所有重排完全无执行代价**。

## 为什么论文还要讨论 LayerNorm 和算子融合？

如果每次 Linear 之前都新增一次真实 Tensor Shuffle，GPU 需要额外读写激活；量化节省的带宽可能又被 Reorder 吃掉。对于适用结构，可把排列与前置 LayerNorm 的通道仿射参数或后续 Weight Layout 预融合，避免独立执行某些排列 Kernel。可是在残差支路、旋转位置编码或 Attention Head 切分处不能未经检查任意穿越排列：必须保证张量各个消费者使用一致的坐标顺序。

特别要注意 RMSNorm 不含逐通道中心化，与 LayerNorm 的统计不同；可交换性需要分别从算子数学形式验证，不应只因为它们都叫 Normalization 就视为等价。

## 与其他 PTQ 的根本区别

[SmoothQuant](C040-smoothquant.md) 通过连续实数尺度 $D$ 改变两侧张量的范围；RPTQ 通过离散排列 $P$ 改变**共享 Scale 的 Channel 组合**；[GPTQ](C041-gptq.md) 利用二阶信息补偿舍入引起的输出差异。

作者在 [原始论文](https://arxiv.org/abs/2304.01089) 中探索低至 **3-bit Activation** 的情形，并在 OPT 等模型上测试压缩与推理精度。公开摘要提到特定配置下显存缩减最高约 80%，不应理解成全部权重、KV、激活在所有硬件上统一减少 80%。

一个最有价值的复现实验是固定同一 Group Size 和整数 Kernel，比较：不重排、随机重排、按范围聚类重排。这样才能区分效果究竟来自额外 Group Scale 还是“动态范围相近”这条设计假设。下一篇 [OmniQuant](C047-omniquant.md) 直接对量化尺度与剪裁策略做少量校准优化。

## 用原论文证据排除两类错误结论

[原始论文 arXiv 2304.01089 §3.2（PDF 印刷第 4 页）](https://arxiv.org/pdf/2304.01089) 明确指出：激活难量化不仅因为存在少量 Outlier，还因为**不同通道的数值范围不一致**。该位置紧接着对比 LLM.int8、SmoothQuant 的处理逻辑。阅读前面的 Figure 1 时，要找的是不同通道的量级差异如何让统一量化参数失效。

作者摘要所提 OPT-175B 特定配置下最高约 80% 内存降低来自量化配置的整体效果，**不应解释为仅仅排列了通道就让模型少占 80%**。更具因果力的实验是固定 W/A 位宽与每 Group 的 Scale 数量，只改变 No-Reorder、Random-Reorder、Range-Clustering Reorder。若随机排列几乎无增益而按范围聚类有收益，才能支持作者的分类依据。

### Table 3 与 Figure 4：怎样看有约束的真实证据？

[Table 3（PDF 第 9 页）](https://arxiv.org/pdf/2304.01089) 中，**OPT-175B、Batch=1、Sequence Length=2048** 的 W16A16 为 **335.4 GB**，W3A3KV 为 **63.2 GB**。在这一**同模型、同 Batch/Length 的论文内存估算**下，相对降低 $(335.4-63.2)/335.4\approx81.2\%$，正是“最高约 80%”量级的依据；并不是所有设备上的实测 CUDA 峰值显存，更不是单独 RPTQ 排列步骤的节省。

紧接的 **Figure 4**（同页）是更接近聚类方法自身的消融：在 W16A4 配置中，分别扫描 R1–R5 的簇数 $1,2,4,8,32$，纵轴为 PPL（越低越好）。**大多数分支增加簇数后 PPL 改善，但 R2 波动更明显**；作者将其与每 Head 独立重排、校准数据规模相关联，并在实验中为 R1/R4/R5 使用 32 簇、R2/R3 使用 4 簇。由此可以得到“实际簇数必须在校准、量化精度和开销之间选择”，却不能推出“越多簇必然越好”或“各通道独立 Scale 的代价为零”。
