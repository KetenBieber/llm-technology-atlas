# KIVI：为什么 Key 按通道量化，而 Value 按 Token 量化？

> **原始论文**：[KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://proceedings.mlr.press/v235/liu24bz.html)，ICML 2024。 [官方/作者 PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/liu24bz/liu24bz.pdf)。


技术前置：[MQA/GQA 降低 KV Head 数量](../../A/04-efficient-attention/A020-gqa.md)、[SnapKV 按重要位置驱逐](C037-snapkv.md)。后续：[SmoothQuant 的激活缩放](C040-smoothquant.md)。

## 把 FP16 KV 压到 2-bit，为什么不能直接整个张量统一缩放？

每个 Decoder 层、每个 KV Head 为历史长度 $L$ 保存 Key 和 Value。如果元素是 FP16，缓存近似为

$$
M_{\rm KV}=2\cdot N_{\rm layer}\cdot N_{\rm kvhead}
\cdot L\cdot d_{\rm head}\cdot 2\ \text{bytes}.
$$

左侧第一个 2 是 K/V 两套矩阵，最后的 2 是每个 FP16 元素的字节数。若仅考虑纯数据位数，把它们缩至 2-bit，理论上每元素从 16 bit 变 2 bit，相差 **8 倍**；但现实系统还要存每 Group Scale/Zero Point、未量化的近期缓冲区、模型权重、临时激活，所以整体峰值显存不可能自动缩成八分之一。

难点是 KV 的异常值方向并不一样。若所有数据共享一个 $\Delta$，极端值会让普通值的量化台阶太粗。量化误差进一步影响 Attention 的两处不同运算：

$$
\operatorname{Attention}(q,K,V)=
\operatorname{softmax}(qK^\top/\sqrt d)V.
$$

- Key 的误差先扰动 logits，然后经 Softmax 改变**关注哪个位置**。
- Value 的误差影响加权求和的**取回内容数值**。

两者误差传播机制不同，分布规律也不同。

## KIVI 的“非对称”到底指什么？

原文研究发现：**Key 中的离群特征更明显沿通道分布，Value 的离群行为则更适合按 Token 处理**。据此提出不对称 Granularity：K 按 Channel 聚合组统计，V 按 Token 聚合组统计。这不是“Key 使用 2-bit、Value 使用 4-bit”的意思，论文提出的核心是二者**不同的组方向**。

从一个 $K\in\mathbb R^{L\times d}$ 的教学例子看：
- 对 K 按 Channel 处理时，可在相同 Feature 维度上跨若干 Token 统计 scale。
- 对 V 按 Token 处理时，可在每个 Token 的若干 Feature 分量上估计 scale。

对 Group $G$，常见 2-bit 非对称均匀量化可写成

$$
\Delta_G=\frac{\max_{x\in G}x-\min_{x\in G}x}{3},\quad
Q_G(x)=\operatorname{clip}_{[0,3]}
\left(\operatorname{round}\frac{x-\min_G}{\Delta_G}\right),
$$

$$
\widehat x=\min_G+\Delta_G Q_G(x).
$$

这是用于理解 **4 个量化等级与 Group 参数** 的等价教学表达；实际实现的 Scale、Zero Point、Packing、残差缓冲等以作者代码为准。若 $\max_G=\min_G$，工程实现必须对零步长处理；这也是公式与 Kernel 的区别。

~~~{mermaid}
flowchart TD
 KV[原始 FP16 KV] --> K[Key：按 Channel 分组]
 KV --> V[Value：按 Token 分组]
 K --> B[2-bit Packed K + Scales]
 V --> C[2-bit Packed V + Scales]
 B --> A[在 Attention 中解量化/计算]
 C --> A
 R[最近未量化 Residual Buffer] --> A
~~~

*自制解释图，非原论文 Figure。原论文 Figure 2/3 比较 Key、Value 的分布及量化方向；教学图额外强调 Residual FP16 缓冲区的实现意义。*

## 为什么还要留一部分近期 KV 不量化？

直接每来一个 Token 都重新计算历史 Group Scale，并打包量化缓存，会使每次 Decode 发生频繁的内存变换与 Copy。KIVI 的实现采用**未量化 Residual Cache** 收集近期 KV，积累到条件允许时按既定 Group 进行量化与打包。它降低重复量化开销，也确保刚生成的位置能以高精度参与近期 Attention。

于是系统必须分别维护：缓存中已量化的旧 Token、近期未量化 Token 的地址，以及按 Head/Layer 的组尺度。低比特的理论容量优势并不等于 Kernel 一定更快——如果反量化、访存合并和 Scale 加载很昂贵，延迟仍可能变差。

## 两种 Group Orientation 为什么产生不同的量化误差？

令 Key、Value 的形状均为 $[T,d_h]$。假设有 4 个 Token、2 个特征通道：

$$
K=\begin{bmatrix}
100&1\\100&-1\\100&0.5\\100&-0.5
\end{bmatrix},\qquad
V=\begin{bmatrix}
1&100\\-1&2\\0.5&3\\-0.5&4
\end{bmatrix}.
$$

对 $K$ 若每 Token 行共享一个量化 Scale，第一列的 100 会让同一行第二列的 0.5 或 1 几乎全部损失精度；若沿 Channel 跨 Token 分别求 Scale，就能让第二通道的 $[-1,1]$ 保留更密的量化格点。

对 $V$，这里的最大值 100 集中在第一 Token 的第二通道。按 Token 选择 Scale 可以让其他三个 Token 各自处于较窄范围；若为每个 Channel 跨全历史共享 Scale，第二列的那个 100 会连带污染后面所有 Token 的小值。**Key 与 Value 有不同异常值结构**才是非对称方向的原因，而不是认为两套矩阵的“语义”天然必须对应两种方向。

这些值是为了说明 Scale 污染而构造；[KIVI ICML 2024 论文](https://proceedings.mlr.press/v235/liu24bz.html) 的 Key/Value 异常方向来自实际 LLaMA 层的可视化和统计，工程中不同模型/Head 需要复核其是否成立。

### 计算真实显存时不能只看 2-bit 对 16-bit

假设 $L=32$ 层，$H_{kv}=8$ 个 KV Heads，$d_h=128$，缓存 $T=16{,}384$ Token。FP16 纯 KV 数据量为

$$
2\times32\times8\times128\times16{,}384\times2
=2{,}147{,}483{,}648\ {\rm bytes}
\approx2\ {\rm GiB}.
$$

理想 2-bit 纯值缓存是 $1/4$ GiB，但必须增加 Group Scale/Zero Point、Index/Packing、近期 FP16 Residual Buffer 和其他模型内存。若尾部 $R=128$ Token 未量化，其 FP16 数据单独还占

$$
2\times32\times8\times128\times128\times2
=16{,}777{,}216\ {\rm bytes}\approx16\ {\rm MiB}.
$$

此例把 KV 存储量的数量级算清楚，不能把 $1/4$ GiB 说成完整服务端进程的峰值显存。

## 残差缓冲区的更新顺序

当前 Decode 产生新 Token 的 KV：先插入未量化的 Tail Buffer；随后 Attention 读取历史量化 KV 与新的 FP16 KV（需要保证逻辑时间索引与 Mask 一致）；一旦 Tail Buffer 累积到适合 Quantize 的 Group 大小，再把旧段按 Key Channel、Value Token 方向打包成 2-bit，并释放对应旧 FP16 Tail 项。若每生成一枚 Token 就回头重算全部 Group Scale，便破坏了 KIVI 引入 Residual 以减少量化频率的关键目的。

因此实验除 PPL 外还应记录量化打包耗时、按 Group 读取 Scale 成本、GPU HBM 字节、模型权重与 KV 占比。否则“KV 理论节省 8 倍”无法推出实际 Token/s。

## 对照结果与其它 KV 方案

[ICML 2024 正式论文](https://proceedings.mlr.press/v235/liu24bz.html) 报告 Llama-2、Falcon、Mistral 等设置下，模型权重也计入的总体峰值显存可减少至原来的约 $1/2.6$，由此 Batch 最高可扩大 **4 倍**，报告真实生成任务吞吐 **2.35–3.47×**。这些数字不是“每张 GPU / 每个模型统一 8 倍”。

与 [H2O](C036-h2o.md)、[SnapKV](C037-snapkv.md) 对照：两者删除部分历史位置，降低“还有多少 Token”；KIVI 量化已保留 KV，降低“每个值占多少字节”。理论上可以组合，但兼容性和误差叠加需实验确认，不能默认组合两种方法必然带来乘法加速。

研究检查点包括：按 Token、按 Channel 交换量化方向的消融；不同 Group Size；Residual Buffer 大小；不同 Prompt 长度下的解码延迟和长文检索准确率；GQA 等 KV Head 不同架构的兼容性。若文章里只写“2-bit 省 8 倍”而没有这四组实际因素，就是把编码容量当成了系统性能。
