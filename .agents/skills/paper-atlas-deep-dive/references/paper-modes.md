# 三种论文的文章结构与工作流

目的：**逻辑顺序因论文而异**。下面是完整性问题清单，不是要求所有文章都有相同的 H2。

## B：模型技术报告（Llama、Qwen、DeepSeek、VLM/VLA、reasoning model）

**开头**以“这个模型为什么在此时出现；对前代改变了什么”拉出主线；先写完整模型的 1 张总图，再按真实执行路径展开。

1. **谱系与贡献去重**：前代版本、提出/采用/改造/工程实现关系；模型的 total vs active params、dense/MoE、多模态形态。
2. **整个 forward**：tokenizer → embeddings → block × L → attention → norm/residual → FFN/MoE → logits；明确 `[B,T,d]` 等 shape，视觉与 action 路径额外画接入点。区别 training-only head 和 inference path。
3. **关键模块为什么替换旧模块**：MHA→GQA/MLA 要讲 decode KV，dense FFN→MoE 要讲容量/激活/通信；不凭术语堆创新点。
4. **预训练**：训练数据、tokenizer、objective、token budget、context schedule、optimizer、precision、并行、稳定性；原文未披露则明确。
5. **后训练**：SFT/RL/rejection sampling/RM/DPO/GRPO 等按论文**真实**阶段顺序，记录输入 checkpoint、数据、loss、reward、参数、rollout、继承关系。不预设 SFT→RLHF。
6. **推理/部署**：prefill/decode、KV、batch、专家 placement、量化、延迟/吞吐、时延与显存权衡。FLOPs≠HBM bytes≠时间。
7. **实验证据**：结构消融 vs 全模型跨系统榜单分开；规模、训练数据、提示协议、采样设置等混杂必须指出。
8. **技术树**：指向已写的原始方法或系统论文；阻塞节点按需下钻。

**适合的图**：整体结构图、单 block 张量流、两个版本的变化对照、training stages、MHA/GQA/MLA 缓存图、成本构成。

**示例阅读主干**：DeepSeek-V2 → DeepSeek-V3 → DeepSeek-R1；Llama 3 ↔ Qwen2.5 为横向对照，不表示某模型一定继承另一模型。

## A：原始方法论文（RoPE、GRPO、DPO、RMSNorm、LoRA、MoE 等）

**核心是推导出发明，而非把技术塞到“完整模型报告”框架。**

1. **现实瓶颈**：原来怎么做、为什么具体条件下失效；
2. **想要的性质**：数学约束或工程约束是什么；例如 RoPE 希望 q_m·k_n 的位置贡献只依赖相对位置 m−n；
3. **朴素方案**：至少举一条可信的直观修法，以及它哪里不够；
4. **作者方法**：逐步推出符号、算法与步骤，给可复算的小例子，能写则写最小代码；
5. **超参或近似**：它不是数学必然的地方，作者用了什么替代选择；
6. **原始证据**：方法 vs baseline、受控消融、理论条件、失效与后来修正；
7. **回链现代模型**：哪些模型实际采用/改造它，原文缺乏信息则不猜。

**例一，GRPO**：PPO 的 learned value baseline 为什么重；同 prompt 一组 response 的 reward mean/std 怎样得到 relative advantage；为何 clipping 没被舍弃；group reward 全同/variance 为 0 会怎样；K L estimator 与长度归一化选法带来什么偏差/方差问题；哪些点出自 DeepSeekMath，哪些是后续论文争论。

**例二，RoPE**：绝对位置编码→需要相对位置信息→二维旋转如何满足相对位移不变性→推广至 d 维→实际注意力内积→长上下文外推问题；不要虚构一套独立 tokenizer、预训练 pipeline。

**适合的图**：旧法新法并排、几何直觉图、算法框、控制变量实验、理论条件的边界图。

## C：系统与算法—硬件协同论文（FlashAttention、4D Parallelism、ZeRO 等）

1. **瓶颈工况先行**：硬件型号、batch/seq/shape、baseline kernel或 distributed execution，具体慢在哪；
2. **度量**：算术强度、FLOPs、HBM traffic、SRAM/shared memory、网络通信量、launch overhead、利用率、显存容量；
3. **设计空间**：除了作者解法，另一种布局/调度/并行选择的代价；
4. **核心实现**：tiling、online softmax、recomputation、sharding、all-gather/reduce-scatter、pipeline overlap、kernel 与线程映射；
5. **正确性**：等价/近似、数值精度、边界条件；
6. **真实 benchmark**：硬件、软件版本、baseline、序列长度、batch、精度、训练/推理、端到端与算子级不可互换；
7. **部署边界**：其它 GPU/小 batch/较短序列可能收益不同。

**适合的图**：数据在 HBM↔SRAM 之间的路线、kernel 线程分配、compute/communication 时间线、吞吐 vs shape 的多场景曲线。

## 多篇对照与批量工作

- **依赖优先**：一个完整模型主干 + 1–2 个阻塞技术节点，不按编号 A001→A166。
- **横向对照**：对齐问题定义、模型规模、训练预算、实现与评估协议；没有可比性则列出原因。
- **旧稿改造**：保留有价值的推导与证据；重写导航、论述顺序、原图说明和技术关系，不搞无差别批量插入空标题。
- **工程范围**：初始批次 2–4 篇；把“论文已写”“原图审核”“站点渲染”“科学终审”分别记录，不混为一次 PASS。
