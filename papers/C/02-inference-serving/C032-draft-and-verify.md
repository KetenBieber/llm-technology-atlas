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

### 原论文 Figure 2 显示为什么 K 与接受率不能分开优化

[Draft & Verify ACL 2024 Figure 2（正式 PDF 第 2 页）](https://aclanthology.org/2024.acl-long.607.pdf) 的横轴是**草稿 Token 被接受的概率（70%、80%、90%、95%）**，纵轴是端到端 Speedup，三条曲线分别表示 $K=1,2,4$ 的候选长度，作者在这一图中假设 Draft 的单步计算为完整 Verify 的 **一半**。读图时会发现，接受率只有 70% 时，较大的 $K$ 反而可能导致减速；到 95% 时，$K=4$ 才更有价值。这些曲线是**在指定成本假设下的分析结果**，不能拿来充当每张 GPU 的实际测量。

例如若 $p=0.7$，连续 $K$ 枚草稿全部被接受的概率可在独立同分布的简化假设下写为 $p^K$：$K=1$ 时为 0.7，$K=4$ 时仅约 0.2401。Draft 多执行三步付出了确定成本，能够多提交四枚的事件却变得较少，因此存在最优 K。实际接受事件跨 Token 并不独立，所以这个 $p^K$ 只是解释性示例，作者的自适应提前结束机制还需要用实际模型对候选置信度与接纳长度的关系进行评估。

**原文 Figure 1（同 PDF 第 2 页）** 另用颜色表示已接受（绿）、拒绝（红）与完整 Target 补充预测（蓝），验证单轮进度的正确性语义。Figure 2 则是**性能设计空间**，不应把两张图混成“所有 $K=4$ 都更快”的结论。

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

## 贪心一致性证明：为什么不能只“跳层生成更快”就宣称无损？

设完整模型在前缀 $s$ 下的贪心输出为 $g(s)=\arg\max_xp_T(x\mid s)$。草稿使用跳层模型提出 $\hat x_1,\ldots,\hat x_K$。Target 并行计算这些候选在各自真实条件历史上的 $g(s),g(s\hat x_1),\ldots$。第一个不匹配位置记为 $j$，本轮仅提交

$$
(\hat x_1,\ldots,\hat x_{j-1},\,g(s\hat x_{<j})).
$$

如果所有草稿均一致，还可以利用 Target 的附加下一位置预测提交另一 Token。为什么这样生成的最终序列与普通 Target 贪心完全一致？对本轮第一个 Token，提交的必定是 $g(s)$；如果已提交前 $i-1$ 枚 Token 与普通 Target 相同，那么对第 $i$ 个候选的验证使用的**条件前缀也相同**，故相同时可提交它；首次不同就换成 Target 自己的 $g$ 并停止本轮。按归纳法每个提交位置都相同。

注意这是在同一 Tokenizer、相同 Mask、相同数值 Kernel 判定和固定 Tie-breaking 下的贪心一致性分析；如果把随机采样的“没有命中 argmax”也按这个逻辑回滚，将改变其概率分布。

### 跳层为什么可以通过减少 FLOPs，却未必节省按比例的时间？

一个 $L$ 层 Decoder，每个 Token 的层计算大致按 $L$ 增长。若跳过 $k$ 层并假设各层成本相同，草稿前向的理想层计算比例是 $(L-k)/L$。例如 $L=32,k=16$，草稿层计算约减半；但整个系统还需要一次完整 Target 验证、已经生成的草稿 KV/激活维护以及首次拒绝后的清理。与普通全模型相比，内存读入的模型权重也不一定能按一半完全省掉，因为 Target 随后依旧需要全部层。

原论文 [Algorithm 2](https://aclanthology.org/2024.acl-long.607/) 给出 Greedy Verification 的循环；Figure 与层选择实验说明为什么固定跳过某一段层不是对所有模型都最优。报告 1.99× 时应同时给 Draft Layer Pattern、$K$、接受率及基线，而不能把“理论省 50% 层”当成整体 2× 保证。

## 论文证据及限制

[ACL 2024 正式论文](https://aclanthology.org/2024.acl-long.607/) 报告 LLaMA-2 等设置下最高 **1.99×**，并强调不需要附加网络训练和额外权重。这里“不需要额外模型权重”不等于“运行时没有新增 Mask/KV 缓存或临时激活开销”；更不能等同于论文已保证所有 Transformer 都能以同一跳层配置维持质量。

这项方法最有价值的实验包括：不同 skip sets 的加速和接受率、同样 K 下 Target 完整验证开销、草稿长度自适应策略。与 [LayerSkip](C033-layerskip.md) 比，它的卖点是**不做专门的早退训练**；后者通过训练让较浅层更早具备输出质量。
