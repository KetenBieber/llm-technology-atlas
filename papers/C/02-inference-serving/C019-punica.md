# Punica：不同请求使用不同 LoRA，为什么还能共享一次大模型矩阵计算？

> **原论文**：Chen 等，[Punica: Multi-Tenant LoRA Serving](https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html)，MLSys 2024。**类型**：C · 多租户模型服务。

## 一、不能把“不同 Adapter”当成“不同完整模型”

假设模型有一份巨大公共权重 $W\in\mathbb R^{d_o\times d_i}$，每个租户 $u$ 有自己训练的 LoRA 增量：

$$
\Delta W_u=B_uA_u,
\qquad
A_u\in\mathbb R^{r\times d_i},
\quad
B_u\in\mathbb R^{d_o\times r},
\quad r\ll \min(d_i,d_o).
$$

对第 u 个请求，列向量输入 $x_u$ 的输出：

$$
\boxed{y_u=Wx_u+B_u(A_ux_u)}.
$$

可以看出公共基座 $Wx_u$ 可以 batch，而第二项按请求选 Adapter。若每个租户独占一份完整 $W+\Delta W_u$，显存放大主要来自重复主模型。Punica 正是把 **共享 Base** 和 **不同 Adapter 增量** 分开执行。

![Punica 的共享基座与多 Adapter 请求](../../../figures/explainers/C019/01-overview.svg)

*自制解释图，原始 SGMV Kernel 与 Serving Architecture 见 [MLSys 正式论文](https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html)。*

## 二、不同 LoRA 为什么不直接凑成普通 GEMM？

### 正式论文 Figure 2 与 Figure 3：系统结构与 Kernel 语义不是同一层

先查看 [MLSys 2024 论文第 3 页 Figure 2–3](https://proceedings.mlsys.org/paper_files/paper/2024/file/054de805fcceb78a201f5e9d53c85908-Paper-Conference.pdf)。**Figure 2** 左侧描绘用户、前端、Scheduler 与 Runner/GPU 的关系：同一 GPU 包含共享的 LLM Base、KV Cache 与多份小型 LoRA，说明的是租户合并和模型驻留，并不是矩阵计算的精确实现。**Figure 3** 右侧才给出 SGMV 的分段数学含义：

$$
Y[s_i:s_{i+1}]\mathrel{+}=
X[s_i:s_{i+1}]W_i.
$$

这里的 $s_i$ 是按照相同 Adapter 聚集后，分段边界的索引。各段输入与不同的 $W_i$ 相乘，但由同一 Kernel 处理；这是避免为每条小请求单独发射 Kernel 的核心。Figure 3 不意味着将不同 Adapter 拼成一个共享的 Dense Matrix：乘法使用的 Weight 依然随 Segment 变化，且 LoRA 的两个低秩矩阵需要分别执行或融合计算。原文 §4 对 Segment 的连续分组与 Tensor Core 优化给出更多细节。

若只拿 Figure 2 的 GPU 方框解释“为什么快”，就遗漏了实际的分段更新；若只看 Figure 3，又遗漏了多租户调度、KV 占用和 GPU Consolidation 成本。两类证据需要同时成立，才能理解论文的最高吞吐成绩。

同 Batch 三个请求的 Adapter IDs 为 $A,B,A$，则：

$$
Y_{\mathrm{base}}=XW^\top
$$

可以一次计算，但增量分别是：

$$
\Delta y_1=B_AA_Ax_1,\quad
\Delta y_2=B_BA_Bx_2,\quad
\Delta y_3=B_AA_Ax_3.
$$

如果按租户拆成多个小 batch 处理，各 Adapter 子批很小，Decode 时容易产生大量小规模 GEMV，GPU Launch 和 occupancy 都不理想。

Punica 的 **SGMV（Segmented Gather Matrix-Vector Multiplication）** 使同一个 Kernel 能让不同 Request Rows 从不同 Adapter 索引中选择需要的权重，统一处理批次里的 LoRA 增量。它并不是说所有 Adapter 参数拼成同一个普通 Dense 矩阵然后毫无索引差异。

在运行时，调度器还要决定 Adapter 是否已驻留、是否需要从主机加载、哪些请求共享同一 GPU 以及 KV Cache 是否仍有足够容量。因此其完整问题是：

$$
\text{Shared Base Weight}
+\text{Per-request Adapter Placement}
+\text{Dynamic Batch}
+\text{KV Capacity}.
$$

## 三、显存账本为什么很诱人？

若基础模型需 $M_W$ 字节，单 Adapter 为 $M_A$，共有 k 个租户：

$$
M_{\rm naive}\approx kM_W,
$$

$$
M_{\rm shared}\approx M_W+kM_A+M_{\rm KV}+M_{\rm workspace}.
$$

上式只是概念比较（可能存在 base replicas 和多精度缓存），但当 $M_A\ll M_W$ 时，差距是结构性的。服务仍需注意 Adapter 切换的 IO，特别是冷 Adapter 多、GPU KV Cache 很满时，装载一个新的 Adapter 可能造成尾延迟和驱逐压力。

## 四、不同 Adapter 的 Prefix KV 可以共享吗？

**不能仅因为 Prompt 文本相同就共享。** Adapter 如果修改了 Attention 或上游 MLP，中间 Hidden/Key/Value 可能与另一个 Adapter 不同。即使两个请求的 Token IDs 完全一样，其 KV Cache 也未必等价。

这与 [SGLang](C017-sglang.md) 的 Prefix caching 是重要交界：多租户 Prefix Cache 的正确命中键需包含与实际前向计算相容的**模型/Adapter 身份**与位置信息。

## Segmented Gather 的实际数据流：一次 Kernel 为什么能处理多种 Adapter？

设三条请求的 LoRA ID 为 `[A,B,A]`，隐藏向量长度为 $d=4096$、适配器 Rank 为 $r=8$。若公共基础层只执行一份 GPU 权重 $W\in\mathbb R^{d\times d}$，则一次 Base GEMM 负责所有请求。但适配器项可以拆成两个算子：

$$
Z_u=A_{a(u)}x_u\in\mathbb R^{r},\qquad
\Delta y_u=B_{a(u)} Z_u\in\mathbb R^{d},
$$

其中 $a(u)$ 给出请求 $u$ 选择哪个 Adapter。关键在于 **$A_A$ 与 $A_B$ 无法像普通 Dense GEMM 那样把所有 Row 都统一乘同一个 Weight Matrix**。若把每条 Request 拆成单独 GEMV，Batch 很小，Kernel Launch 和 SM 利用率可能主导延迟；若反复按 Adapter 分组，调度和 Batch 重排又引入额外开销。

SGMV（Segmented Gather Matrix-Vector Multiplication）把请求序列组织为若干带 Adapter ID 的 Segment，在一个 Kernel 里按段读取对应权重，避免为每段单独 Launch。适配器权重的 Gather 并不免费：如果同 Batch 里有上百个不同的冷 Adapter，布局随机、无法被 Cache 复用，仍可能很慢。

### 内存与计算账本不能只写“共享基础模型”

对每个 LoRA Linear，以 Rank $r$ 的两个矩阵表示增量时，Adapter 参数规模约为 $r(d_i+d_o)$；完整 Dense 权重则是 $d_i d_o$。若 $d_i=d_o=4096,r=8$：

$$
\frac{r(d_i+d_o)}{d_id_o}
=\frac{8(4096+4096)}{4096^2}
=\frac1{256}.
$$

同精度下，这层 Adapter 参数量是完整 Weight 的 $1/256$，这解释为什么多租户共享 Base 的潜在显存节省很大。但实际模型里 LoRA 可能只插入 Q/V 或部分 FFN，或者 Adapter Rank 更大；上述比例是**单层玩具规模**，并不是整模型的固定倍数。

## 为什么请求组的公平性会影响收益？

若 Scheduler 总优先让相同 Adapter 的请求聚集在一起，有利于 Adapter KV 和权重驻留，但容易使低频租户的请求排队；若每个租户请求均即时执行，Batch 分组能力又会下降。因此**多租户服务性能**必须同时报告 TTFT、TPOT、吞吐、尾延迟与 Adapter 切换命中率。

[Punica MLSys 2024 正式论文](https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html) 把内核级 SGMV 和多租户完整 Serving 系统结合，论文里的最高吞吐比较不应被写成 SGMV 单 Kernel 的纯浮点算力增加 12 倍；一定要分清减少模型复制、提高 Batch Consolidation 与 Kernel 优化各自的来源。

## 五、实测收益来自什么工况？

原论文报告在多租户 LoRA workload 和其比较系统中最高约 **12× 吞吐提升**，并报告每 Token 约 **2ms 的附加时延**这一测试条件下的系统代价。不能推导所有 Adapter 都只增加 2ms，也不能把此倍率推广到单租户普通 Decode。

最大的相对收益通常来自：租户很多、单个租户请求稀疏、公共 Base 大、重复基座副本很浪费。论文的系统证明不仅有 SGMV 内核，还包括跨租户整合批次的 Scheduler。

从这里继续 [Speculative Decoding](C020-speculative-decoding.md)：接下来的加速不再利用不同请求之间的公共计算，而是在**同一个请求里预测多个未来 Token，目标模型一次并行验证**。

**原始来源**：[MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html)。
