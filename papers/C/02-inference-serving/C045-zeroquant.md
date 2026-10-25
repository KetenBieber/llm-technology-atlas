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

## 为什么 Kernel 融合不可缺少？

即使 INT8 GEMM 计算快，若分别调用量化 Kernel、反量化 Kernel，反复将中间张量存回 HBM，时间可能被 Memory Traffic 与 Kernel Launch 支配。因此应检查 Quantization 操作能否并入前后矩阵乘、归一化和 Epilogue，尤其区分 Prefill 大矩阵与 Decode 小 Batch 的真实 Shape。

[NeurIPS 正式论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html) 在所测 BERT / GPT 风格模型的 W8A8 下报告最高约 **5.19× / 4.16×**，混合低比特加 LKD 可减小权重存储；这些是**不同模型及对照设置**的成绩，不可一概写成“所有 LLM 快 5 倍”。

验证时应同时测 PPL、层输出差、Quant/Dequant 分别耗时、内存峰值和端到端吞吐。下一篇 [RPTQ](C046-rptq.md) 不主要引入新的整数 Kernel，而是问：通道的动态范围能否通过重排重新组织？
