# Sequoia：将树形候选预算变成动态规划与硬件优化

> **原始论文**：[Sequoia: Scalable and Robust Speculative Decoding](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract.html)，NeurIPS 2024。**类型**：C 类推理机制与系统。 [原文 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea1f5f0878d43ff4fb8bf64ef4a2326c-Paper-Conference.pdf)。

前置：[树形验证](C027-specinfer.md)；对照：[EAGLE-2 的在线动态树](C023-eagle2.md)。

## 预算相同，树为什么仍能相差很远？

用 $n$ 个 Token 节点生成候选，可以组织成 $n$ 条深度为 1 的宽树，也可以生成一条深度为 $n$ 的链，还可以是一棵非均匀树。宽树覆盖首步不确定性，但难以一次确认多个连续 Token；深链若第一步拒绝，其余节点无用。

Sequoia 从**在既定验证预算 $n$ 下最大化预期提交 Token 数**出发，而不是默认固定分支数。论文 §3.1 采用 positional acceptance 假设：对于已接受父节点，其第 $k$ 个孩子的被接受概率仅由其孩子次序 $k$ 决定，构造接受向量 $p=(p_1,p_2,\ldots)$。这是一个建模假设，非对任意 Target、输入都成立的定理。

对于树 $T$，可把目标写成

$$
T_n^\star=\arg\max_{|T|\le n}F(T),\qquad
F(T)=\mathbb E[\text{一轮验证可提交的 Token 数}].
$$

在位置接受模型中，某个节点能贡献一个额外的已提交 Token，需要其整条祖先路径都成功；因此 $F(T)$ 可写成节点路径生存概率的求和，再用动态规划求满足预算的最优组合。

### 为什么能用动态规划？

朴素枚举所有 $n$ 节点树成本极高。若第 $k$ 个子树的预算为 $n_k$，一旦给定父节点已接受，该子树带来的期望贡献只依赖自己的预算和第 $k$ 位置的接受参数，于是可把复杂树拆成独立的子问题。概念上递推为：

$$
V(n)=\max_{\sum_k n_k\le n-1}
\sum_k \mathcal G(k,n_k;p),
$$

其中 $\mathcal G$ 是已考虑第 $k$ 分支条件与子树贡献的**抽象子问题价值**；该式只说明 DP 分解，不是冒称原文逐字给出的递推。精准公式应查阅 [原文 Proposition 3.4 和 Algorithm](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea1f5f0878d43ff4fb8bf64ef4a2326c-Paper-Conference.pdf)。

~~~{mermaid}
flowchart TD
 B[节点预算 n] --> O[基于接受向量规划树结构]
 O --> T[生成有限节点候选树]
 T --> V[树形 Target 验证]
 V --> H[结合硬件成本模型调 n 与深度]
 H --> B
~~~

*自制解释图。请比较原论文 §2 的 SpecInfer / SpecTr / top-k 等不同树形采样与 §3 Sequoia 的构造过程。*

## 为什么高温采样需要换验证策略？

贪心或接近确定性的分布容易被少数高概率子节点覆盖。温度升高后分布更平坦、多个候选可能被选中，重复抽样的碰撞更频繁；有限候选预算的覆盖行为也改变。

论文不仅设计树，还讨论**不放回采样（sampling without replacement）**相关策略以提升不同温度下的有效覆盖。它是“如何利用给定节点集”的另一设计层次，不等同于固定 top-k 或不断重复抽样。若想精确维持 Target 分布，就必须将候选去重与剩余分布校正写入验证逻辑，不能把“避免重复”直接当作已经证明 lossless。

## 硬件感知为什么不是锦上添花？

节点越多，一轮 Target Forward 的算力/显存/通信开销可能上升；节点太少，一次 Target Forward 仅提交很少 Token，无法摊薄模型参数读取。以一轮提交期望 $L(n)$、草稿时间 $t_d(n)$、验证时间 $t_v(n)$ 示意：

$$
\text{平均每 Token 耗时}\approx
\frac{t_d(n)+t_v(n)}{L(n)}.
$$

即便树 $T$ 在固定 $n$ 下有更高的 $F(T)$，**最优 $n$ 仍依赖硬件、显存预算和 offloading 代价**。这解释了作者额外加入 hardware-aware tree optimizer。

## 精确报告作者成绩

[NeurIPS 2024 正式论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract.html) 在 A100 上报告 Llama2-7B、Llama2-13B、Vicuna-33B 最高分别 **4.04×、3.73×、2.27×**。对 Llama3-70B-Instruct 在单张 L40 上的 offloading 设置，报告 **0.60 s/token**，相对 DeepSpeed-Zero-Inference 约 **9.5×**。别与旧版 arXiv 对 Llama2-70B 的指标混为一谈。论文 Figure/表格分别支撑树结构算法、采样策略和系统 benchmark，而不是单张表能证明所有设计的因果效果。

**研究边界**：位置接受假设可能不能捕捉 Query 难度或词位置相关性；[EAGLE-2](C023-eagle2.md) 正是从在线草稿置信度进行树扩展。对比二者时要固定同样的 Target/采样温度/树节点上限/硬件，避免以不同论文的 headline speedup 做直接胜负判定。
