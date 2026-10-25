# Draft & Verify：无需训练，如何让同一个模型同时充当草稿与验证器？

> **原始论文**：[Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://aclanthology.org/2024.acl-long.607/)，ACL 2024。**类型**：C 类推理机制与系统。 [原文 PDF](https://aclanthology.org/2024.acl-long.607.pdf)。

前置：[Speculative Decoding 的严格校正](C020-speculative-decoding.md)；对照：[LayerSkip](C033-layerskip.md)。

## 一定要把 Draft 单独训练成小模型吗？

普通推测解码要额外准备权重、Tokenizer 对齐、模型加载与缓存。如果 Target 已是 $L$ 层 Transformer，能否暂时跳过若干中间层，让**同一个模型走一条更短的路径**快速猜候选，再用完整路径验证？

Draft & Verify 的回答是：可以在不额外训练模型参数的情况下，选择性跳过中间层形成轻量计算路径。设完整 Transformer 为

$$
h^{(0)}=E(x),\quad
h^{(\ell+1)}=F_\ell(h^{(\ell)}),\quad
p=\operatorname{softmax}(Wh^{(L)}).
$$

对若干中间层设置二值跳过掩码 $z_\ell$，说明性地写作

$$
\tilde h^{(\ell+1)}=
\begin{cases}
\tilde h^{(\ell)},&z_\ell=1,\\
F_\ell(\tilde h^{(\ell)}),&z_\ell=0.
\end{cases}
$$

然后用较短路径给出草稿 $q$。这里的形状与 Residual 兼容是关键工程前提：中间层并非随便删除后都能保持足够好的语言建模能力。

~~~{mermaid}
flowchart LR
 P[已确认前缀] --> D[部分层跳过: 快速 Draft]
 D --> C[多 Token 草稿]
 C --> V[同一模型完整层验证]
 V --> R[保留连续一致的 Token]
 R --> P
~~~

*自制流程图。原论文 [Algorithm 2（PDF 第 4 页）](https://aclanthology.org/2024.acl-long.607.pdf) 明确描述贪心模式下如何逐个比对 Target argmax 并在首次不同处回滚。*

## 为什么跳层配置也是优化问题？

跳过层数多，Draft 计算很省，却容易产生与 Target 不同的 Token；跳层少，Draft 质量好，但加速空间小。可设 $t_d(z)$ 为给定跳层模式的草稿 Forward 时间，$\ell(z)$ 为平均被 Target 接受的连续 Token 长度：

$$
T_{\rm per\ token}(z)\approx
\frac{K\,t_d(z)+t_{\rm verify}(K)}
{\ell(z)+1}.
$$

这是解释性成本模型，不是原论文逐字的优化式。配置 $z$ 要同时影响计算时延和接受长度，不能只用困惑度/分类精度选择。论文研究哪些中间层值得跳过，以及何时结束当前 Draft 阶段，以减少无效候选。

## 为什么跳过若干层不会让最终答案“变差”？

对**贪心**版本，若 Target 已知真实前缀、完整模型预测序列为 $y_1,y_2,\ldots$，验证器只有在候选 $\hat y_i=\arg\max_x p(x\mid \text{真实前缀与已接受 Token})$ 时才保留它。首次不同就停止，将该位置替换为 Target 的选择。因此可以对已提交位置归纳：每一步都与完整模型逐 Token 贪心生成一致。

随机采样下不能把“碰巧不一致就替换成 argmax”当成无损证明，需要另用正确的接受–残差机制。原论文同时讨论 sampling 版本，实际实现应按各自算法分开验证。**可复现的精确条件是完整 Target 仍决定提交结果**；部分层输出只是提议。

## 论文证据及限制

[ACL 2024 正式论文](https://aclanthology.org/2024.acl-long.607/) 报告 LLaMA-2 等设置下最高 **1.99×**，并强调不需要附加网络训练和额外权重。这里“不需要额外模型权重”不等于“运行时没有新增 Mask/KV 缓存或临时激活开销”；更不能等同于论文已保证所有 Transformer 都能以同一跳层配置维持质量。

这项方法最有价值的实验包括：不同 skip sets 的加速和接受率、同样 K 下 Target 完整验证开销、草稿长度自适应策略。与 [LayerSkip](C033-layerskip.md) 比，它的卖点是**不做专门的早退训练**；后者通过训练让较浅层更早具备输出质量。
