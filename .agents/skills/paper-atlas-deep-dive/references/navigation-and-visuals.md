# 文章导航与可视化设计

## 站内知识网络

给文章建立两层连接：

1. **顶部导航**：论文原文链接、前置主线、阻塞技术下钻、后续论文；只给最相关的 3–7 个项目，不做 20 条列表。
2. **正文首次就地跳转**：术语出现并成为理解负担时，就地链接它的源稿。首次出现 MLA→DeepSeek-V2，GRPO→DeepSeekMath，MoE specialization→DeepSeekMoE。后面不必每次重复链接。
3. **结尾定向分流**：卡在“公式/数据流/系统代价/实验”时分别去哪里，展示真正依赖关系。

相邻正文源稿的相对路径从**当前稿所在目录**计算。例如：
- `papers/B/05-moe-complete-llm/B009-deepseek-v3.md` → `[DeepSeek-V2](B008-deepseek-v2.md)`
- 同篇 → `[DeepSeekMoE](../../A/05-moe/A034-deepseekmoe.md)`
- 同篇 → `[GRPO](../../A/08-post-training/A051-deepseekmath-grpo.md)`

链接前验证目标真实存在。不存在：只写“计划精读”或使用已核实的公开原文链接。不要生成死链接、空锚点、虚假的论文 ID。

站点入口由 `docs/reading-path.md` 与 `docs/technology-map.md` 维护，最终站点通过 `tools/build_site_sources.py` 生成；**正文源稿不能直接写在 `docs/generated`**。Sphinx MyST 的内部 `{doc}` 导航使用已存在 generated 路径，但不要把该语法无差别塞回普通 Markdown 源稿。

## 选择图，而不是拼图

### 原论文图（Evidence Figure）
- 从一手 PDF/论文源包提取；留存来源版本、原 Fig./Table 号、页码或文件成员、图注和 manifest。
- 只保留解释关键命题所需图，裁图必须完整可阅读，坐标轴/图例不能剪掉。
- 正文标“原论文 Figure n”；原始图不能冒称“自制”。
- 不应把带版权问题的原始图在未经确认下公开发布。

### 自制教学图（Explanatory Figure）
- 标“**自制解释图，非原论文 Figure**”；
- 可合理抽象旧法新法，不要让抽象曲线/数值看起来像论文实验原始数据；
- 优先 SVG/Mermaid 与清晰中文标签。图片可存 `figures/explainers/<topic>.svg`；
- 若确实需要曲线，坐标和数值要从公式/公开可追溯数据计算，标注“示意/复算”；不杜撰实验点。
- 每图必须说明：**它降低了哪个认知门槛？下一段如何回到推导或实验？**
- 宽图下可以排**两阶段**：概念示意→关键方程/Shape；复杂图如果首屏看不清就拆图。

## 图型决策表

| 论文难点 | 更合适的图 | 典型场景 |
|---|---|---|
| 一堆模块但不懂谁为谁服务 | 三列因果流“瓶颈→机制→系统量” | DeepSeek-V3 |
| 缓存维度不直观 | MHA/GQA/MLA 并排缓存数据结构 | DeepSeek-V2 |
| 算法前提与替代不清 | 旧法→失败→新法决策树 | PPO→GRPO |
| 多阶段 SFT/RL 混淆 | 带数据回流的 checkpoint/数据管线 | DeepSeek-R1 |
| 模型整体不完整 | 带层数、张量 shape 的 token forward 图 | Llama/Qwen/VLA |
| GPU 速度原因不明 | HBM/SRAM 时间线与读写量 | FlashAttention |
| 分布式吞吐被卡住 | TP/PP/EP 通信重叠与瓶颈时间线 | MoE |
| 实验误读 | baseline vs variant 的受控变量图 | 消融分析 |

## 视觉质量

- 优先一个问题一张图，限制一张图中单独颜色含义数量。
- 字号、线条、单位要可读；结构模块不要遮挡公式或箭头。
- 与已有的 `tools/redraw_figures.py`、`tools/generate_overviews.py` 兼容前先阅读脚本输入和输出规则；不调用会覆盖现有图片的全量脚本做盲目批处理。
- 需要用图像文件时，以当前工具能力为准；如无图像生成器，用有文本可读退化的 Mermaid/SVG，不要假装已经生成某个未创建的 PNG。
- 图片生成后用图像查看器或 SVG/XML 解析检查；**XML 解析通过不等于审美/可读性通过**。

## 完整论述示例（导航的价值）

> 普通 MHA 的 decode 并非每次重新算全部历史 token 的网络前向，但仍需要读取历史 K/V。随着 context length 增大，KV 容量与 HBM 读流量成为瓶颈。MLA 并不是只把 K/V 投影成一个更小向量就结束了；关键是让压缩后的状态能直接用于 decode 的注意力计算。这里若不理解 projection absorption 如何成立，就去 [DeepSeek-V2 / MLA](相对路径指向实际文件)；读懂后再返回 V3，看这笔缓存收益怎样与 MoE 训练/推理系统配合。

这类引导只在自然承接处出现，不把所有段落都变成“摘要卡片”。
