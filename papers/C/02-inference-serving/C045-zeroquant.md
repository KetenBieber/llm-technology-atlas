# ZeroQuant：量化策略、逐层蒸馏与 GPU Kernel 为什么要一起设计？

> **原始论文**：[ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html)，NeurIPS 2022。


前置：[LLM.int8 的异常值双精度路径](C039-llm-int8.md)，以及 [SmoothQuant 的 W8A8](C040-smoothquant.md)。后续：[RPTQ 的通道重排](C046-rptq.md)。

## 压缩参数为什么不等于加快 Transformer？

一次线性层原本只需要访问 FP16 权重、激活并执行 GEMM。改为 INT8 后虽然输入体积减半，却可能新增量化、解码、格式转换和临时张量分配：

$$
T_{\mathrm{new}}=T_{\mathrm{Quant}}+T_{\mathrm{intGEMM}}+
T_{\mathrm{Dequant}}+T_{\mathrm{repack}}.
$$

这说明理论存储比为 $16/8=2$，不能推导端到端加速一定是 $2\times$。ZeroQuant 是完整部署方案：**硬件友好的细粒度权重/激活量化、低成本逐层知识蒸馏、量化内核运行时优化**。单独讨论整模型 Scale 无法解释其工程贡献。

### 为什么不能每个元素都单独设置 Scale？

对一个张量 Group 的均匀量化可写成

$$
q=\operatorname{clip}\left(\operatorname{round}(x/s)+z,q_{\min},q_{\max}\right).
$$

Scale 越细，越能避免一个异常值挤压其他数值的整数格点；但是硬件的 INT8 GEMM 按矩阵块操作，若给求和维度里的每个元素任意不同的 Scale，整数累加后的重缩放难以直接通过一次简洁的 Epilogue 完成。ZeroQuant 兼顾分组的精度与向量/矩阵内核可执行性：Activation 适配 Token Wise 的动态范围，权重的精度与分组策略考虑硬件实现。

## 再深入一层：什么样的 Scale 才能进入标准整数 GEMM？

设 `X:[M,K]`、`W:[K,N]`，忽略零点后的一种粒度是：每个输入 Token 行使用 $s^X_m$，每个输出通道使用 $s^W_n$。量化值分别满足 $x_{mk}\approx s^X_m q^X_{mk}$ 与 $w_{kn}\approx s^W_n q^W_{kn}$，所以

$$
\widehat y_{mn}=s^X_m s^W_n\sum_{k=1}^{K}q^X_{mk}q^W_{kn}.
$$

这里最重要的不是公式形式，而是**两项 Scale 均不依赖求和变量 $k$**。因此可以先执行高吞吐 INT GEMM，把整数积累加到 INT32，再按外部行/列 Scale 恢复浮点输出。若粗暴地沿 $K$ 方向给每个元素单独设计 Scale，Scale 不能直接提出求和号外，标准 GEMM 的一次性 Epilogue 就不够用了。这是“细粒度改善误差”和“硬件友好”之间必须满足的共同约束。

更一般地，带零点的非对称量化 $x_i=s_x(q_{x,i}-z_x)$、$w_i=s_w(q_{w,i}-z_w)$ 会引入

$$
\sum_i x_iw_i\simeq s_xs_w\bigg[
\sum_i q_{x,i}q_{w,i}
-z_w\sum_iq_{x,i}
-z_x\sum_iq_{w,i}
+Kz_xz_w\bigg].
$$

于是 Kernel 除整数点积外，还可能需要行和、列和的修正。使用对称量化、预计算和 Kernel Fusion 能减轻成本，但不能把非对称 Zero Point 的代价直接删掉。

### 一个真实可以算出量化误差的玩具例子

令激活 $x=[1,0.1]$，另一个 Token 为 $x'=[15,1]$。如果两行共享对称 INT8 Scale $s=15/127$，第一行的第二个值编码约为 $\operatorname{round}(0.1/s)=1$，反量化为 $0.1181$，绝对误差约 $0.0181$。若每行独立缩放，第一行 Scale 是 $1/127$，$0.1$ 约映到整数 13，反量化为 $0.1024$，误差约 $0.0024$。**这些数字展示了动态粒度为何影响量化误差；它们并不代表论文任何特定层的统计。**

## Layer-by-Layer Knowledge Distillation 蒸馏的是什么？

低比特导致局部层输出出现误差。对于第 $\ell$ 层教师 $F_\ell$ 和量化学生 $\widehat F_\ell$，可使用解释性的重建目标：

$$
\mathcal L_\ell=\mathbb E_{h_\ell\sim\mathcal D_{\rm cal}}
\left\|F_\ell(h_\ell)-\widehat F_\ell(h_\ell)\right\|_2^2.
$$

注意训练输入 $h_\ell$ 不必来自当初的私有 Pretraining Corpus。作者提出的是**无原始训练数据访问条件下**、使用可获得样本进行逐层重建的廉价方案，而不是无任何样本、无需模型前向就完成蒸馏。并且层 MSE 较低也不能严格推出下游所有任务不降质量。

~~~{mermaid}
flowchart LR
 FP[FP16 Teacher 层输出] --> L[逐层匹配损失]
 Q[细粒度量化的层] --> L
 L --> C[少量校准更新]
 C --> Q
 Q --> K[Quant/Dequant Kernel 融合]
 K --> T[真实 Token 吞吐]
~~~

*自制三环节解释图，非论文原图。应分别核查原文 Quantization、Layer-wise KD 和推理后端章节。*

## Layer-wise Knowledge Distillation 到底优化什么？

给定一层的校准输入 $h_\ell$，先通过当前层未量化的 $F_\ell$ 求输出，再经过当前量化学生层 $\widehat F_\ell$，以两者误差作为局部损失。**原论文 §4.2 Eq. (1) 的实现重点是两条路径共用前 $1,\dots,\ell-1$ 层的输入表示，只为当前层保留原精度版本作为教师。**它不需要在 GPU 上常驻另一个完整的教师 LLM。这一设计区别于把完整教师与完整学生同时载入的 Hidden-State Distillation，直接降低了显存开销。

训练只针对可调整的量化变量，不要求再执行整个模型的全参数语言预训练。以一个二维输出示意：

- 教师输出 $y=[1,2]$，未经补偿的量化学生输出 $\hat y=[0.6,2.2]$，平方差之和 $0.16+0.04=0.20$。
- 校准后若变为 $\hat y'=[0.9,2.1]$，平方差之和 $0.01+0.01=0.02$。

这里仅演示**层局部重建**的目标。下游语言建模的 Token NLL、PPL 和真实问答准确率并非层 MSE 的同义词；如果要证明任务效果，仍要运行整模型评估。尤其低 bit 模型的长层链可能放大细小误差，“每个单层都还不错”不推出每个生成任务保持原样。

## 为什么 Kernel 融合不可缺少？

即使 INT8 GEMM 计算快，若分别调用量化 Kernel、反量化 Kernel，反复将中间张量存回 HBM，时间可能被 Memory Traffic 与 Kernel Launch 支配。因此应检查 Quantization 操作能否并入前后矩阵乘、归一化和 Epilogue，尤其区分 Prefill 大矩阵与 Decode 小 Batch 的真实 Shape。

[NeurIPS 正式论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html) 在所测 BERT / GPT 风格模型的 W8A8 下报告最高约 **5.19× / 4.16×**，混合低比特加 LKD 可减小权重存储；这些是**不同模型及对照设置**的成绩，不可一概写成“所有 LLM 快 5 倍”。

验证时应同时测 PPL、层输出差、Quant/Dequant 分别耗时、内存峰值和端到端吞吐。下一篇 [RPTQ](C046-rptq.md) 不主要引入新的整数 Kernel，而是问：通道的动态范围能否通过重排重新组织？

## 把原论文的三项主张拆开看证据

[NeurIPS 2022 的作者摘要与论文入口](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html) 明确将贡献拆为细粒度量化、可负担逐层蒸馏、系统后端优化。读论文时，应把每个主张与可观察量一一对应：

| 干预 | 正确的观察量 | 不能直接证明 |
|---|---|---|
| 改 Quantizer 粒度 | 量化重建误差、PPL | Kernel 一定加速 |
| 加入逐层蒸馏 | 同 bit 下学生与教师差距 | 全模型和教师在所有任务等价 |
| 融合量化 Kernel | Prefill/Decode 的 Wall Time、访存 | 蒸馏提高了任务能力 |
| 混合 INT4 FC + INT8 Attention | 指定权重/激活配置的峰值显存 | 所有层都是 W4A4 |

原文提到 BERT/GPT 风格模型相对 FP16 的最高 **5.19× / 4.16×**，那是不同模型基线的系统实验；混合 INT4/INT8 配置报告的约 **3×** 内存压缩是另一组权重量化条件。要确认具体哪项组件贡献吞吐，必须查看系统消融，不能拿摘要最高倍数直接推出单组件因果。

### 原论文 Figure 1 究竟是在为哪个设计选择提供证据？

阅读 [ZeroQuant 原始 PDF 第 3 页 Figure 1](https://arxiv.org/pdf/2206.01861)：**左侧**显示 GPT-3 350M 不同层的 Token 激活绝对幅度分布，层之间以及 Token 之间的动态范围都有差异；**右侧**显示 Attention Output Weight 的不同 Row 取值范围存在离散的异常点。Figure 1 支持“激活用 Token-Wise、权重用合适 Group/Row 粒度”的设计动机，但它本身**没有测量 INT8 GEMM 的执行速度**；为什么还要融合 Quant/Dequant，必须回到原文 §4.1 与 §4.3 的内核描述及吞吐实验。读图时把这两条证据链明确分开。

最小可复现实验可固定相同 GPU，比较 `X:[1,4096]` 与 `X:[128,4096]` 乘 `W:[4096,4096]`。分别测 Quant/INT GEMM/Dequant 各阶段，便能发现小 Batch Decode 与大 Batch Prefill 对整数核的利用率可能截然不同。上述形状只是教学工况，不是论文指定的 Benchmark。
