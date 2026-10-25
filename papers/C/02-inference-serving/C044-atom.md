# Atom：既要 W4A4，又要让 GPU 真正跑上低比特算子，如何设计服务系统？

> **原始论文**：[Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving](https://proceedings.mlsys.org/paper_files/paper/2024/hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html)，MLSys 2024。 [官方/作者 PDF](https://arxiv.org/pdf/2310.19102)。


依赖：[LLM.int8 的异常通道](C039-llm-int8.md)、[SmoothQuant 的激活缩放](C040-smoothquant.md) 和 [AWQ 的重要通道](C042-awq.md)；系统对照：[FlexGen](C043-flexgen.md)。

## 低比特已经能减少显存，为什么服务器吞吐还不够高？

Weight-Only W4A16 的一个优势是权重从 FP16 变 4-bit，可以减少带宽读取，尤其对低 Batch 的 Token-by-Token 解码有效。但如果激活仍为 FP16，实际矩阵乘法不一定能直接使用 GPU 原生高吞吐 **INT4×INT4** 算子。另一个方案 W8A8 兼容量化 Tensor Core，却在某些硬件上放弃了更低 bit 算子和更高压缩率。

Atom 的问题非常明确：**为在线 Batch Serving 做到 W4A4 的大规模矩阵乘法，同时保住模型精度。** 它不只是一个静态模型量化脚本，而是“量化算法 + 混精度安排 + Kernel + 服务器测量”的联合系统设计。

## 为什么单纯 W4A4 容易误差暴增？

设某输入通道上真实的乘法为 $x_jw_{j,k}$，若对激活和权重都量化：

$$
\hat x_j=x_j+\epsilon^X_j,\qquad
\hat w_{j,k}=w_{j,k}+\epsilon^W_{j,k}.
$$

乘积误差是

$$
\hat x_j\hat w_{j,k}-x_jw_{j,k}
=x_j\epsilon^W_{j,k}+w_{j,k}\epsilon^X_j+
\epsilon^X_j\epsilon^W_{j,k}.
$$

如果激活/权重的离群通道 $|x_j|,|w_{j,k}|$ 很大，即使 $\epsilon$ 绝对值不夸张，误差也会被放大。更关键的是，4-bit 只有 16 个量化等级，简单全 Tensor Scale 很容易丢失普通通道的分辨率；这不是把 W8A8 的 bit 数减半就能自动成功。

## Atom 用不同精度对待不同 Channel

Atom 利用 Outlier/Salient Channel 对输出贡献不均匀的现象，对重要通道安排更高精度处理，同时让其余绝大部分计算使用低比特算子。与 [AWQ](C042-awq.md) 的 Weight-Only 4-bit 不同，Atom 希望把**激活也降到 4-bit**，让计算密集的 GEMM 受益于原生整数硬件。

为了避免稀疏高精度通道分离导致内存访问碎片化，系统还会组织通道重排、Group-Wise Quantization 和 Mixed Precision 的执行路径，把权重/Activation 的数据布局配合 Kernel 使用。抽象地可写成

$$
XW=X_{:,O}W_{O,:}+X_{:,N}W_{N,:},
$$

其中 $O$ 为受保护通道，$N$ 为低比特通道；这是未量化的代数分解。实际系统需处理两个子 GEMM 的 Scale 和累积精度，在正确布局下实现高吞吐，不能把分解式本身当作精度保证。

~~~{mermaid}
flowchart TD
 X[校准的权重和激活] --> H[识别离群/重要 Channel]
 H --> C[调整通道布局与分组]
 C --> I[常规路径：INT4 GEMM]
 C --> O[受保护路径：混合精度]
 I --> A[合并结果]
 O --> A
 A --> S[在线 Batch Serving]
~~~

*自制解释图，非原论文 Figure。原文方法图中的通道重排与 Mixed Precision 和 Kernel 绑定，是理解为什么理论 bit 数真的能成为服务吞吐的关键。*

## 为什么 Batch 和目标延迟是评估的必要条件？

在 Decode 阶段，低 Batch 往往首先受限于权重带宽；在较大 Batch 下，矩阵乘法可能更受 GPU 计算吞吐影响。W4A4 可提高每单位显存存储的权重密度，并尝试直接利用更高的低精度算力，但如果新增的高精度支路、重新布局和反量化引入高固定开销，小 Batch 也可能不划算。

把某个方案的服务吞吐写为

$$
R=\frac{\text{完成的总输出 Token 数}}{\text{墙钟时间}},
$$

**仍必须同时满足相同的延迟服务目标**，否则提升 Batch 就可以制造“吞吐更高”的假象。[Atom MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html) 在所测在线 Serving 情境报告 W4A4 相对 FP16 最多 **7.73×** 吞吐、相对 INT8 最多 **2.53×**，并要求维持相同延迟目标。不能把这两个数字用于衡量模型本身推理逻辑质量，也不能推断任何手机端硬件具有对应 INT4 指令就能复现。

## 技术路线闭环：该怎样选择量化？

如果优先降低权重存储和边缘部署复杂度，研究 [GPTQ](C041-gptq.md)/[AWQ](C042-awq.md) 的 Weight-Only；如果要降低 W8A8 激活异常问题，研究 [SmoothQuant](C040-smoothquant.md)；如果重点是**在线 Batch 下借助 INT4 GEMM 获取吞吐优势**，再研究 Atom 的细粒度混精度 Kernel。如果显存不足导致模型根本无法装载，则还要同时研究 [FlexGen](C043-flexgen.md) 的 CPU/SSD Offloading。

这套选择不是单靠哪篇论文的最高倍数判断，必须先辨认：当前系统的瓶颈是权重访存、KV 占用、GEMM 算力、传输，还是 SLO 约束。
