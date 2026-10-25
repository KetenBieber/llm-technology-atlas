# 离散扩散语言模型：D3PM → MDLM → LLaDA

自回归语言模型把一条文本按从左到右的条件分布分解，并顺序生成下一个 Token；**离散扩散语言模型**则定义一条已知的 Token 扰动过程，通过学习它的反向恢复过程生成整个序列。两者都是合法的概率生成模型，但训练目标、模型可见上下文、生成时的信息确认顺序和 Likelihood 评估方式完全不同。

本专题按数学上的问题依赖阅读，而非按模型参数量或框架速度排名。

- **D3PM（NeurIPS 2021）**：怎样把 Token 的 Forward Noise 写成行随机矩阵 $Q_t$？怎样从 $\bar Q_t$ 用 Bayes 推出合法的反向后验？Uniform、Nearest-Neighbor、Absorbing State 的噪声几何为何不同？
- **MDLM（NeurIPS 2024）**：Absorbing Mask 具有哪些确定转移？为什么 SUBS 能把复杂 KL 化简为正权的 Masked Cross-Entropy？$1/t$ 权重从哪里来？为何能用双向网络训练真正的生成模型？
- **LLaDA（NeurIPS 2025）**：怎样将 Mask Predictor 扩展为 8B Transformer、2.3T Token 从头预训练，再对 Prompt/Response 仅遮盖 Response 做 SFT？如何解释 Low-Confidence Remasking、反向诗句测试与 AR 基线？

## 一条线连接三篇论文

```{mermaid}
flowchart LR
 A["AR：p(x)=∏p(x_i|x_<i)"] --> CH{"序列一定要左到右生成吗？"}
 CH --> D["D3PM：分类 Markov 前向 + 后验"]
 D --> M["MDLM：吸收态 SUBS + Masked CE / ELBO"]
 M --> L["LLaDA：8B 全双向 Transformer + SFT"]
 L --> X["多位置预测 + 迭代确认"]
```

*自制教学导航图，非原论文 Figure；图表示技术论证路线，不表示后续作者必然沿同一代码库复用实现。*

**特别区分**：Mask GIT 类启发式填空可生成文本，却不能因为用了随机 Mask 就自动拥有与 MDLM 相同的连续时间 ELBO；而 LLaDA 的训练目标以 MDLM 等理论工作为基础，主要新证据在大模型训练、SFT 与能力评估。D3PM 的 text8 NLL（bits/char）、MDLM 的 LM1B PPL 上界、LLaDA 的 MMLU/GSM8K Accuracy 是不同指标，不能直接计算“提升百分比”。

对照基础：[Transformer](../generated/papers/A/03-transformer/A013-transformer-attention-is-all-you-need.md) 的 Causal Attention 与自回归概率分解；扩散技术与其他状态空间序列模型是**并列研究轴**，不应与 S4/Mamba 的固定状态记忆链混为一谈。

```{toctree}
:maxdepth: 2

../generated/papers/A/11-discrete-diffusion/A065-d3pm
../generated/papers/A/11-discrete-diffusion/A066-mdlm
../generated/papers/B/05-moe-complete-llm/B015-llada
```
