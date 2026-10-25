# Sequoia：将树形候选预算变成动态规划与硬件优化

> **原始论文**：[Sequoia: Scalable and Robust Speculative Decoding](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract.html)，NeurIPS 2024。**类型**：C 类推理机制与系统。 [原文 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea1f5f0878d43ff4fb8bf64ef4a2326c-Paper-Conference.pdf)。

前置：[树形验证](C027-specinfer.md)；对照：[EAGLE-2 的在线动态树](C023-eagle2.md)。

## 预算相同，树为什么仍能相差很远？

### Figure 1 说明“树越大就越好”为什么在旧构型里不成立

[Sequoia NeurIPS 2024 正式论文 Figure 1（PDF 第 2 页）](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea1f5f0878d43ff4fb8bf64ef4a2326c-Paper-Conference.pdf) 左图横轴为对数尺度的候选**树节点数量**（$10^0$ 到 $10^4$），纵轴为一次 Target 验证后**平均生成/接受的 Token 数**。右图分别画出单链（Single Sequence）、满 $k$ 叉树、$k$ 条独立链和 Sequoia 优化树的拓扑结构。

从左图可读出：单链与固定 $k$ 独立序列的收益较快趋于饱和，优化构型的绿色曲线随节点预算继续增长。顶部 V100/A100/H100/Offloading 标志不是同一条曲线的四组硬件吞吐测量，而是作者用来说明**不同内存带宽条件可能容忍的候选树预算范围**。不能据此说绿色曲线显示 H100 实测 Token/s 永远最高；该图横轴是 Tree Size，不是执行时间。

为什么固定 $k$ 条链会饱和？如果第一个候选前缀选择错误，那么在同一无效前缀下再延长该链，只会产生更多不能沿真实生成路径接受的节点。动态规划允许将预算在**扩展当前高概率前缀**与**为第一步高不确定性增加备选分支**之间重新分配；目标是提高单位验证预算下的**路径命中深度**。作者的定位是建模最优树结构，不是随机加几条分支就天然变快。

原论文 §3.1 的 DP 优化目标在给定 Positional Acceptance 模型下成立；若数据分布、Temperature 或 Draft/Target 匹配情况变化，该模型参数也会改变。端到端 Speedup 还必须结合 Tree Verification 时间与 Offloading 约束，不能直接用 Figure 1 左图的平均生成数当成速度倍数。

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

## 在理想位置接受模型下，如何计算一棵树的“价值”？

设树中每个深度 1 的孩子按顺序 $k=1,2,\ldots$，论文假设被接受的概率主要取决于兄弟排序位置，给出 $p_k$。对于一条路径 $u$，若从根到它依次选择排名 $k_1,\ldots,k_d$ 的孩子，概念上的路径生存概率为

$$
w(u)=\prod_{j=1}^{d}p_{k_j}.
$$

在节点分布和接受事件满足模型条件时，预期可以提交的节点数可表达为节点路径概率之和

$$
F(T)=\sum_{u\in T}w(u).
$$

这一式突出**Deep 分支需要祖先全部成功**这一点；论文的精确问题还包含采样与不放回验证的条件，不能无条件把 $p_k$ 当成真实逐 Token 独立概率。

### 三节点玩具树：宽与深哪个更值得？

为简单起见预算只允许三个候选节点，假设 $p_1=0.7,p_2=0.2,p_3=0.05$。三节点深链的路径权重为 $0.7,0.7^2,0.7^3$，合计

$$
F_{\rm chain}=0.7+0.49+0.343=1.533.
$$

宽度三的根层候选权重为 $0.7,0.2,0.05$，合计 $0.95$。而把前两个根层孩子与第一个孩子的后代结合，权重 $0.7,0.2,0.49$，总计 $1.39$。这些计算不等于 Sequoia 的全部 DP，因为真实树采样的 sibling acceptance 有相关性；它说明一种重要的结构性趋势：**当第一提案被接受的概率高时，更多深度可提高一轮连续提交数；当第一提案低而其他候选可能覆盖目标时，预算需向宽度转移。**

严格优化里可定义 $V(n)$ 为预算 $n$ 的最优 Value。若根层可放 $k$ 个孩子，给它们分配预算 $n_1,\ldots,n_k$，递推会合并各个子树的 Conditional Expected Gain。最终的约束是 $\sum_j(n_j+1)\le n$，其中 $+1$ 为每棵子树的根层孩子本身；漏掉孩子本身容易出现 “把预算数多算一层” 的 off-by-one 错误。

## 一个预算最优的树可能不是硬件速度最优

目标机假设草稿开销 $D(n)$、Target 验证开销 $V(n)$、期望提交数 $A(n)$，那么平均每 Token 耗时为

$$
\tau(n)=\frac{D(n)+V(n)}{A(n)}.
$$

如果节点预算从 10 扩到 20，平均提交数从 2 提升至 2.5，但 Target Tree Forward 从 8 ms 增至 16 ms、Draft 与建树分别为 2 ms 和 4 ms，则

$$
\tau(10)=\frac{2+8}{2}=5\text{ ms/token},\quad
\tau(20)=\frac{4+16}{2.5}=8\text{ ms/token}.
$$

**更多候选价值带来了更差的实际性能**。Sequoia 的 Hardware-aware Tree Optimizer 并非补充装饰，而是要在真实测得的 Target cost curve 上选预算与深度。

[NeurIPS 2024 正式版](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract.html) 的系统结果包含单 A100 的多种模型与 L40 上 70B Offloading。看图时要区分：验证同样树预算下的算法收益、温度变化时 Sampling without Replacement 的覆盖能力，以及 Offloading 时 Target Forward 主导的成本。若只摘录“最高 9.5×”，便遗漏了其成立的硬件条件。

## 精确报告作者成绩

[NeurIPS 2024 正式论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract.html) 在 A100 上报告 Llama2-7B、Llama2-13B、Vicuna-33B 最高分别 **4.04×、3.73×、2.27×**。对 Llama3-70B-Instruct 在单张 L40 上的 offloading 设置，报告 **0.60 s/token**，相对 DeepSpeed-Zero-Inference 约 **9.5×**。别与旧版 arXiv 对 Llama2-70B 的指标混为一谈。论文 Figure/表格分别支撑树结构算法、采样策略和系统 benchmark，而不是单张表能证明所有设计的因果效果。

**研究边界**：位置接受假设可能不能捕捉 Query 难度或词位置相关性；[EAGLE-2](C023-eagle2.md) 正是从在线草稿置信度进行树扩展。对比二者时要固定同样的 Target/采样温度/树节点上限/硬件，避免以不同论文的 headline speedup 做直接胜负判定。
