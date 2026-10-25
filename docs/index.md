# 大模型技术图谱

**LLM Technology Atlas** 是一套以论文与完整模型为主线的大模型技术文章站。

这里不按论文编号机械阅读，而是沿着真正的技术依赖组织内容：

- 从反向传播、优化器与 Transformer 建立基础；
- 再进入 RoPE、RMSNorm、MQA / GQA、MoE 等架构组件；
- 以 DeepSeek-V2 → V3 → R1 作为第一条完整现代模型主线；
- 以 Llama 3 → Qwen2.5 建立第二条完整模型参照，重点理解 dense scaling、数据工程、多尺寸模型家族、staged post-training 与长上下文系统；
- 遇到 MLA、DeepSeekMoE、GRPO 等关键节点时，直接跳回原始方法专题；
- 每篇文章内部保留“前置阅读 / 相关技术 / 后续阅读”和就地跳转，形成可连续浏览的技术网络。

## 大模型技术全景导图

技术发展图梳理“方法从哪里来、后来发展出什么”；难点—方案图从“工程瓶颈是什么”反向定位研究路线。两张图均由本站已发表的论文注册表生成，随新论文自动补充，分类关系由作者维护。

```{raw} html
<section id="atlas-map" class="atlas-explorer" aria-label="大模型技术论文交互导图">
  <div class="atlas-toolbar">
    <label for="atlas-map-view">知识视角</label>
    <select id="atlas-map-view" data-map-view aria-label="导图视角">
      <option value="evolution">技术发展脉络</option>
      <option value="challenges">技术难点 → 解决方案</option>
    </select>
    <input data-map-search type="search" aria-label="搜索论文" placeholder="搜索论文名称 / 技术编号..." />
    <button data-map-shrink type="button" aria-label="缩小导图" title="缩小">−</button>
    <button data-map-expand type="button" aria-label="放大导图" title="放大">+</button>
    <button data-map-fit type="button" title="适配可视区域">适配</button>
  </div>
  <div class="atlas-map-surface"><svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="可展开的技术关系图。点击分类展开，点击论文前往正文；空白处拖动画布，Ctrl 加滚轮缩放。"></svg></div>
  <div class="atlas-meta">
    <div class="atlas-status" data-map-status>正在读取已发表的论文分类…</div>
    <div class="atlas-detail" data-map-detail aria-live="polite">点击分类节点展开；点击论文节点打开原文解读。</div>
  </div>
  <div class="atlas-fallback">
    <strong>技术分类快速入口（不依赖脚本）</strong>
    <a href="transformer/index.html">模型与注意力</a>
    <a href="distributed-training/index.html">分布式训练</a>
    <a href="long-context/index.html">长上下文</a>
    <a href="inference-serving/index.html">推理、量化与部署</a>
    <a href="models/index.html">完整模型</a>
    <a href="post-training/index.html">对齐与后训练</a>
  </div>
</section>
```

浏览方式：点击技术分类展开/折叠，点击论文节点打开拆解文章；可以搜索文章、拖动视野、缩放。导图不接受访客编辑，避免产生与正式论文目录不一致的节点。完整线性入口仍见下方学习路线。

## 推荐入口

- {doc}`学习路线 <reading-path>`：按顺序建立完整知识链。
- {doc}`技术图谱 <technology-map>`：按技术依赖快速跳转。
- {doc}`长上下文 <long-context/index>`：从 RoPE/YaRN → DCA → MInference，区分位置外推、Attention relation 与超长上下文 prefill runtime。
- {doc}`GPU / 算子与编译 <gpu-kernels/index>`：从 FlashAttention 开始理解 HBM/SRAM、IO-aware exact attention 与后续 GPU work partition。
- {doc}`分布式训练系统 <distributed-training/index>`：从 TP / PP / CP / FSDP 建立多 GPU 训练的 memory–communication–topology cost model。
- {doc}`推理服务系统 <inference-serving/index>`：从 Orca/vLLM 进入 Prefill/Decode 分离、Prefix KV 共享、GPU kernel 与精确投机采样。
- {doc}`EAGLE 后续演化 <generated/papers/C/02-inference-serving/C023-eagle2>`：从动态树预算继续到 {doc}`EAGLE-3 的多步草稿训练 <generated/papers/C/02-inference-serving/C024-eagle3>`。
- {doc}`推测解码技术分支 <generated/papers/C/02-inference-serving/C025-hydra>`：顺序头与循环草稿 → {doc}`SpecInfer <generated/papers/C/02-inference-serving/C027-specinfer>` → {doc}`Sequoia <generated/papers/C/02-inference-serving/C028-sequoia>`；另有 {doc}`Lookahead <generated/papers/C/02-inference-serving/C029-lookahead-decoding>` 的无独立草稿路径。
- {doc}`长上下文 KV 优化 <generated/papers/C/02-inference-serving/C035-streamingllm>`：从 Attention Sink、H2O、SnapKV 进入 {doc}`KIVI 2-bit KV <generated/papers/C/02-inference-serving/C038-kivi>`。
- {doc}`模型量化与低显存部署 <generated/papers/C/02-inference-serving/C039-llm-int8>`：从异常值分流到 {doc}`SmoothQuant <generated/papers/C/02-inference-serving/C040-smoothquant>`、GPTQ、AWQ、FlexGen 和 {doc}`Atom <generated/papers/C/02-inference-serving/C044-atom>`。
- {doc}`低比特后训练与码本 <generated/papers/C/02-inference-serving/C045-zeroquant>`：ZeroQuant → RPTQ → OmniQuant → SpQR → QuIP# → AQLM；{doc}`QLoRA <generated/papers/C/02-inference-serving/C051-qlora>` 与 {doc}`BitNet b1.58 <generated/papers/C/02-inference-serving/C054-bitnet-b158>` 则分别连接低显存微调、从头低比特训练。
- {doc}`低比特旋转与可逆仿射 <generated/papers/C/02-inference-serving/C055-quarot>`：QuaRot 正交变换 → {doc}`SpinQuant 学习旋转 <generated/papers/C/02-inference-serving/C056-spinquant>` → {doc}`FlatQuant Kronecker 分解 <generated/papers/C/02-inference-serving/C057-flatquant>`，研究 W4A4 与可选 KV4 的精度、校准成本与 GPU 执行；三篇的具体位宽/基线需要逐表对齐，不能直接排名。
- {doc}`极端异常值的定向消解 <generated/papers/C/02-inference-serving/C058-duquant>`：DuQuant 的两次局部旋转与 Zigzag 重排；再到 {doc}`QServe 的 W4A8KV4 服务实现 <generated/papers/C/02-inference-serving/C059-qserve>`，分析 INT4 存储、INT8 Tensor Core 以及 KV4 Attention 的真正性能瓶颈。
- {doc}`Scaling Laws <scaling/index>`：从 Kaplan 到 Chinchilla，理解模型/数据/compute 的训练规划。
- {doc}`完整模型主线 <models/index>`：从 DeepSeek V2/V3/R1 对照到 Llama 3 与 Qwen2.5。
- {doc}`推理后训练 <post-training/index>`：从 PPO/GRPO 进入 reasoning RL。

```{toctree}
:maxdepth: 2
:caption: 开始

reading-path
technology-map
```

```{toctree}
:maxdepth: 3
:caption: 技术专题

foundations/index
transformer/index
long-context/index
gpu-kernels/index
distributed-training/index
inference-serving/index
scaling/index
moe/index
models/index
post-training/index
```
