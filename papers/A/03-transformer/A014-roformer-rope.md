> **原文**：Jianlin Su 等，[*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864)，2021，[可检索原文](https://ar5iv.labs.arxiv.org/html/2104.09864)，重点 `3.1–3.4、`4.1–4.5、Table 1–5。此文是 RoPE 的原始方法论文，原文主要展示**Encoder 式长文本分类、BERT 式 MLM 和英德翻译**；现代 Decoder-only LLM 的 KV Cache 与长上下文扩展属于后续采用与工程分析，不能当成原论文已验证的应用。对应 Paperlist A/03 第 4 篇。

# 模型定位与谱系

## 不是新的 Attention 算法：论文只改动 Q/K 的位置注入方式

**主要类型 A：原始方法论文；次要属性 B：RoFormer 模型及其预训练/下游实验。** Transformer 自注意力在没有位置信息时对序列置换等变，原版在 Token Embedding 上直接相加正弦绝对位置编码。若我们关心的是「query 与 key 相距多少个 token」，直接相加绝对位置未必以最简洁方式向 Attention score 暴露相对位移；一些后续相对位置方案在打分矩阵中引入额外偏置或相对位置表，需要考虑长度截断及其与其他注意力 kernel 的兼容性。

RoPE 想同时满足两个条件：单个 token 在位置 `m` 的 Q/K 含有**绝对位置相位**；两个 token 的 Attention 内积里，这两个绝对相位却能相消，只剩 `n-m` 这种**相对位移**。最自然的数学对象是二维旋转：两次旋转的相对角度只取决于角度差，旋转还保留向量范数。

~~~text
原始 Transformer：Token Embedding + Sin/Cos 绝对位置
                             ↓
           Attention(score) 里位置/内容项混合
                             ↓
        相对位置 bias / relative key embedding 等改进
                             ↓
           RoPE：每两维构造一个二维旋转子空间
              Q(m)=R(mθ)·Wq x_m
              K(n)=R(nθ)·Wk x_n
                             ↓
          Q(m)ᵀK(n)=qᵀR((n−m)θ)k
                             ↓
        多频率、低额外参数、Q/K 定位、可用于多头
                             ↓
        后续 RoPE 长上下文扩展：NTK scaling / YaRN
        Llama 系列等 Encoder/Decoder Transformer 采用
~~~

主技术树：`Transformer → Attention → Position Information → Relative Position Mechanism → Rotary Position Embedding (RoPE)`。RoPE 不是 Tokenizer，也不是对原始输入 `X` 应用 2D 图像几何旋转；它发生在 Attention 头内部**Q/K 的特征维**。后续 Llama、其他模型具体是否使用全部或部分维度旋转、不同 base 和 scaling 要逐个报告核验。


## 总结架构图

![教学总结图：A014-roformer-rope](../../../figures/explainers/A014-roformer-rope-summary.svg)

> **教学总结图**：RoPE 对 Q/K 做位置相关旋转，使点积中的位置依赖自然化为相对位移。

# 输入、输出与任务

## 三种空间必须分开：token、Attention head、二维旋转对子空间

设 `X\in\mathbb R^{B\times T\times d_{\rm model}}`，通常先通过独立线性映射和多头 reshape 得

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
$$

每个头 `Q,K,V\in\mathbb R^{B\times h\times T\times d_h}`，其中 `d_h` 是偶数。对每个头的第 `i` 个**相邻两维** `(2i,2i+1)`，在 token 的绝对位置 `m` 使用一个角度 `m\theta_i` 旋转：

$$
q'_{m,i}=R(m\theta_i)q_{m,i},\qquad
k'_{m,i}=R(m\theta_i)k_{m,i},
$$

`V` 在论文的 RoPE Attention 实现中**不旋转**；最后仍算 `\operatorname{softmax}(Q'K'^\top/\sqrt{d_h}+M)V`。这里 `R` 是特征空间 `2\times2` 的旋转矩阵，与真实机器人机体在二维物理空间内的旋转矩阵形式相同，却具有不同的语义：它表示**序列位置相位**，不是坐标姿态估计。

| 对象 | Shape | 产生/读取方 |
|---|---|---|
| 未旋转 Q/K | `[B,h,T,d_h]` | 先做线性投影 |
| 频率 `\theta_i` | `[d_h/2]` | 通常固定，`\theta_i=10000^{-2i/d_h}` |
| 当前位置 `m` | `[T]` 或 Decode 单一绝对 position ID | Prefill 与单步 Decode 必须同一坐标体系 |
| `\cos(m\theta_i),\sin(m\theta_i)` | `[T,d_h/2]` | 可预生成并广播到 batch/head |
| 旋转 Q'/K' | `[B,h,T,d_h]` | 形状、参数总维度不变 |
| 未改动 V | `[B,h,T,d_v]` | 被 Attention 权重加权 |
| 打分/权重 | `[B,h,T,T]`（self-attention） | 编码相对位置的点积 |
| 输出 | `[B,h,T,d_v]` → 拼头 `[B,T,hd_v]` | 后接原有 `W_O` |

如果只在 Q 上施加旋转、K 不旋转，`q_m^\top k_n` 只含 `m` 而不会自然产生 `n-m`，破坏主要构造。现代某些 Partial RoPE / 交错或前后半维 pairing 的实现布局不同，**必须保证 Q、K 采用完全相同的维度配对与频率约定**；仅改变实现布局而不相应调整预训练权重通常不等价。

# 骨干架构与信息交互

~~~mermaid
flowchart TD
  X["输入隐藏状态 X [B,T,d]"] --> Proj["线性映射 Wq/Wk/Wv"]
  Proj --> Q["Q [B,h,T,dh]"]
  Proj --> K["K [B,h,T,dh]"]
  Proj --> V["V [B,h,T,dv]"]
  Pos["绝对位置 m → 各对维度相位 mθᵢ"] --> Trig["预计算 cos/sin [T,dh/2]"]
  Q --> QR["按两维旋转 Q'=R(m)Q"]
  K --> KR["按两维旋转 K'=R(n)K"]
  Trig --> QR
  Trig --> KR
  QR --> Score["Q'K'ᵀ / sqrt(dh) + Attention mask"]
  KR --> Score
  Score --> Soft["沿 Key 轴做 softmax"]
  Soft --> Out["权重 × 未旋转 V"]
  V --> Out
  Out --> Head["拼头 + 输出投影"]
~~~

**模块删减分析：** 去掉两条旋转支路只剩普通无位置点积；把绝对位置向量简单叠加到词向量则回到另一种位置注入方式，其分数会出现内容—位置交叉项而不天然只依赖相对位移；去掉 `V` 分支仍能算 score，却失去被汇总的内容；把 RoPE 同时加到 `V` 会改变 Value 加权和的语义，不属于原论文的常规实现。RoPE 对一个 self-attention 块的**输出 Shape 完全不变**，因此可以把它作为与 FFN、Norm、MoE 无关的可插拔位置机制。

# 关键技术

## 1. 先把需求写成数学约束，而不是直接背公式

考虑位置 `m` 的内容向量 `q=W_Qx_m` 与位置 `n` 的 `k=W_Kx_n`。我们希望编码函数 `f_q(q,m),f_k(k,n)` 满足

$$
\left\langle f_q(q,m),f_k(k,n)\right\rangle
=g(q,k,n-m),
$$

也就是 score 允许依赖词内容 `q,k`，但位置只以差 `n-m` 出现。朴素绝对位置相加得到

$$
(q+p_m)^\top(k+p_n)
=q^\top k+q^\top p_n+p_m^\top k+p_m^\top p_n.
$$

其中 `q^\top p_n` 独立依赖 `n`，`p_m^\top k` 独立依赖 `m`，要靠后续参数自行组合才能得到只依赖相对位移的规律。RoPE 则用满足群运算的正交旋转，使位置差**成为内积的代数恒等式**。[原文 `3.1](https://ar5iv.labs.arxiv.org/html/2104.09864)

## 2. 从一个二维向量推到相对位移：不跳过转置和三角恒等式

定义标准二维旋转矩阵

$$
R(\phi)=
\begin{bmatrix}
\cos\phi&-\sin\phi\\
\sin\phi&\cos\phi
\end{bmatrix}.
$$

转置为

$$
R(\phi)^\top=
\begin{bmatrix}
\cos\phi&\sin\phi\\
-\sin\phi&\cos\phi
\end{bmatrix}
=R(-\phi).
$$

根据 `\cos(a+b)=\cos a\cos b-\sin a\sin b`、`\sin(a+b)=\sin a\cos b+\cos a\sin b`，逐项矩阵乘法可验证

$$
R(a)R(b)=R(a+b),\qquad R(a)^\top R(a)=I.
$$

给 `q_m'=R(m\theta)q`、`k_n'=R(n\theta)k`，则

$$
\begin{aligned}
(q_m')^\top k_n'
&=(R(m\theta)q)^\top(R(n\theta)k)\\
&=q^\top R(m\theta)^\top R(n\theta)k\\
&=q^\top R(-m\theta)R(n\theta)k\\
&=\boxed{q^\top R((n-m)\theta)k}.
\end{aligned}
$$

**两个绝对位置相减的来源不是模型自发学到的，而是旋转矩阵的群结构。** 对任意共同平移 `\Delta`，

$$
R((m+\Delta)\theta)^\top R((n+\Delta)\theta)
=R((n-m)\theta),
$$

因此如果内容 `q,k` 不变，两者同时整体平移相同位置数，未加 mask 的 Attention score 严格相同。它不代表一个实际 LLM 在把文本整体移动后**所有输出必然严格相同**：例如 attention causal mask、特殊 tokens、上下文裁剪、其他绝对位置组件及数值精度也影响前向。

## 3. 为什么可以用复数写，且不是必须使用复数 tensor

把实数二维向量 `(a,b)` 看作复数 `z=a+ib`，乘单位复数 `e^{i\phi}=\cos\phi+i\sin\phi`：

$$
(a+ib)(\cos\phi+i\sin\phi)
=(a\cos\phi-b\sin\phi)+i(a\sin\phi+b\cos\phi).
$$

实部/虚部正好是矩阵 `R(\phi)(a,b)^\top`。另外复数内积 `\operatorname{Re}[\overline{q'}\,k']` 对应二维实点积，因而

$$
\operatorname{Re}\left[
\overline{qe^{im\theta}}\,(ke^{in\theta})
\right]
=\operatorname{Re}[\bar q k e^{i(n-m)\theta}].
$$

这与上一节旋转矩阵推导完全一致。**实现中无需使用 complex dtype**：将相邻两个实数通道作交错偶奇分组，乘 `\cos` 和 `\sin`，再用加减完成旋转，更便于普通浮点 GPU kernel。

举例取 `q=(1,0),k=(1,0)`、`\theta=\pi/2`。两者同位置时点积 1；`m=0,n=1` 时 `k'=(0,1)`，点积 0；`m=0,n=2` 时 `k'=(-1,0)`，点积 -1；`m=0,n=4` 时 `k'=(1,0)`，又回到 1。**最后一项警告了单频率位置表示的周期性混叠：距离越大不意味着单对 Q/K 的点积必然越来越小。**

## 4. 从二维推广到 `d_h` 维：Block Diagonal Rotation 的代价

若 `d_h` 为偶数，把每个 head 的向量拆为 `d_h/2` 个二维对：

$$
q=[q^{(0)},q^{(1)},\ldots,q^{(d_h/2-1)}],
\qquad q^{(i)}\in\mathbb R^2.
$$

每对赋不同频率

$$
\theta_i=10000^{-2i/d_h},\quad i=0,\ldots,d_h/2-1.
$$

构成块对角矩阵

$$
R_{\Theta,m}^{d_h}
=\operatorname{diag}\bigl(R(m\theta_0),R(m\theta_1),
\ldots,R(m\theta_{d_h/2-1})\bigr).
$$

应用到 `q,k` 后：

$$
\boxed{
(q'_m)^\top k'_n
=\sum_{i=0}^{d_h/2-1}
 (q^{(i)})^\top R((n-m)\theta_i)k^{(i)}.
}
$$

每个频率贡献不同的振荡尺度，增加多尺度相对位移线索。由于各小块正交，

$$
\|R_{\Theta,m}q\|_2^2
=q^\top R_{\Theta,m}^\top R_{\Theta,m}q
=\|q\|_2^2.
$$

所以 RoPE **不会改变单个 Q/K 向量的二范数**，却会改变它们相对方向及 score；这不是归一化 Q/K 的替代品，模型训练后的 Q/K 范数仍可发生变化。

实际矩阵 `R_{\Theta,m}` 看起来是 `d_h\times d_h`，但每行只有不超过两个非零系数。直接以稠密矩阵乘法 `O(d_h^2)` 实现完全浪费；对每个二维对进行四次逐元素乘法、两次加减即可，时间 `O(d_h)`、存储 cos/sin 表 `O(Td_h)`（可以现场生成或缓存），没有额外 trainable 参数。[原文 `3.2、`3.4.2](https://ar5iv.labs.arxiv.org/html/2104.09864)

## 5. 长距离衰减：论文说的是多频率振荡和一个上界，不是严格单调

原论文 `3.4.3 将复数点积写成不同频率的和：

$$
F(r)=\sum_{i=0}^{d_h/2-1}h_i e^{ir\theta_i},
\qquad r=m-n,
$$

其中 `h_i` 来自当前 Q/K 两维的复乘。令 `S_j(r)=\sum_{i=0}^{j-1}e^{ir\theta_i}`，利用离散 Abel 分部求和，可得在取 `h_{d_h/2}=0` 时

$$
F(r)=-\sum_{i=0}^{d_h/2-1}
S_{i+1}(r)(h_{i+1}-h_i).
$$

取绝对值应用三角不等式

$$
|F(r)|\le
\max_i|h_{i+1}-h_i|
\sum_{i=0}^{d_h/2-1}|S_{i+1}(r)|.
$$

论文借此研究一定频率配置下 `S_j(r)` 的整体振荡相消趋势，并给出图示；**该上界本身仍依赖内容 `h_i` 和具体距离**，并没有证明每一对词、任意 `r_1<r_2` 都满足 `|F(r_2)|<|F(r_1)|`，也不能据此保证训练长度之外的推理精度。此前二维 `\pi/2` 例子已构成单频率严格单调说法的反例。[原文 `3.4.3、Fig.2](https://ar5iv.labs.arxiv.org/html/2104.09864)

长期外推还会遇到高频相位快速旋转、训练未覆盖的相位组合和有限精度误差，后来的 RoPE scaling / YaRN 才需要重新调整频率、位置或做继续训练；那是**后续问题**，不是 RoPE 原文保证无界外推。

## 6. 可运行 PyTorch：interleaved RoPE、位置平移不变与正确的 Decode offset

~~~python
import torch
import math

def rope_even_odd(x, positions, base=10000.0):
    """x [B,H,T,D]；D 偶数；positions [T]，绝对位置索引。"""
    b, heads, t, dim = x.shape
    assert dim % 2 == 0 and positions.numel() == t
    half = dim // 2
    freqs = base ** (-2 * torch.arange(half, device=x.device,
                                      dtype=x.dtype) / dim)  # [D/2]
    angles = positions.to(dtype=x.dtype)[:, None] * freqs[None, :]
    cos = angles.cos()[None, None, :, :]           # [1,1,T,D/2]
    sin = angles.sin()[None, None, :, :]           # [1,1,T,D/2]
    pairs = x.reshape(b, heads, t, half, 2)
    even, odd = pairs[..., 0], pairs[..., 1]      # 两者 [B,H,T,D/2]
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    return torch.stack((rot_even, rot_odd), -1).reshape(b, heads, t, dim)

torch.manual_seed(8)
q = torch.randn(2, 4, 5, 8, dtype=torch.float64, requires_grad=True)
k = torch.randn(2, 4, 5, 8, dtype=torch.float64, requires_grad=True)
positions = torch.arange(5)
rot_q = rope_even_odd(q, positions)
rot_k = rope_even_odd(k, positions)
logits = rot_q @ rot_k.transpose(-1, -2) / math.sqrt(8)  # [2,4,5,5]

# 两边绝对位置共同增加 100，所有相对位置 n-m 未变。
shifted_q = rope_even_odd(q, positions + 100)
shifted_k = rope_even_odd(k, positions + 100)
shifted_logits = shifted_q @ shifted_k.transpose(-1, -2) / math.sqrt(8)
torch.testing.assert_close(logits, shifted_logits, rtol=1e-11, atol=1e-11)
torch.testing.assert_close(rot_q.norm(dim=-1), q.norm(dim=-1))
assert logits.shape == (2, 4, 5, 5)

# Decode：历史 K 必须已用创建时的原始 position 旋转；
# 新生成位置 m=5，只需将当前一个 Q/K 按绝对 m 旋转后追加 K cache。
new_q = torch.randn(2, 4, 1, 8, dtype=torch.float64)
new_k = torch.randn(2, 4, 1, 8, dtype=torch.float64)
q_now = rope_even_odd(new_q, torch.tensor([5]))
k_now = rope_even_odd(new_k, torch.tensor([5]))
k_cache = torch.cat([rot_k.detach(), k_now], dim=2)      # [2,4,6,8]
score_now = q_now @ k_cache.transpose(-1, -2)           # [2,4,1,6]
assert score_now.shape == (2, 4, 1, 6)

logits.square().mean().backward()
assert q.grad is not None and k.grad is not None
print("rotation norm, relative shift, cache offset and gradients OK")
~~~

逐行对应：`freqs` 就是论文 `\theta_i`，`angles` 由不同 token 绝对 `positions` 与不同子空间频率外积得到；`cos,sin` 在 B、head 轴广播；`reshape(...,half,2)` 指明相邻偶奇配对，不能随意把顺序换成前后半通道；`rot_even,rot_odd` 是二维 `R(\phi)` 乘法的两个输出分量；`stack` 按原次序复原 `D` 维；`logits` 仍用标准 Attention 缩放；同时平移测试对同一 `q,k` 的所有 token 改位置但保持相对差，验证推导的代数恒等式；norm 测试验证正交性；后半段展示增量推理应给第 6 个 token 设置绝对 `position=5`，历史 `K` 已经在写入缓存时旋转一次，**不能每步对旧 K 再乘一次当前时刻的旋转矩阵**；`backward` 测试旋转加点积的梯度正常传播。

此代码需要实际安装 PyTorch 才可运行；它验证数学与 Shape，但不等于实现了整篇 RoFormer MLM/翻译训练或生产级融合 kernel。

## 7. 原文实验：哪些支持 RoPE，哪些不能被概括成「全面超过 BERT」

| 实验位置 | 控制关系 | 原文数字/结论 | 证据边界 |
|---|---|---|---|
| `4.1 Table 1，WMT14 英→德 | 在同类 Transformer 翻译配置中更换位置机制 | 原文 Transformer-base 27.3 BLEU，RoFormer 27.5 BLEU | 约 **+0.2 BLEU**；不能夸张成跨数据集的巨大统一提升。 |
| `4.2 Fig.3，BERT vs RoFormer MLM | 对照预训练损失曲线，原文实验 100k 步、最大长 512、AdamW | 作者报告 RoFormer 在其设置中收敛更快 | 原文用语把 BERT 位置编码写成「sinusoidal」，但**标准 BERT 使用可学习绝对位置 Embedding**；解读代码和原始 BERT 时应以真实实现为准。 |
| `4.3 Table 2，六个 GLUE 任务 | 预训练后下游微调表现是否一致 | RoFormer MRPC 89.5 vs BERT 88.9，QQP 86.4 vs 71.2，STS-B 87.0 vs 85.8；**SST-2 90.7 vs 93.5、QNLI 88.0 vs 90.5、MNLI 80.2 vs 84.6 更低** | 作者称六个任务中三项改善；表中存在明显任务间差异，不能写成全面领先。 |
| `4.4 PerFormer（线性 Attention） | 用 RoPE 旋转线性 Attention 分子，收敛如何 | Enwik8 的作者实验在同等训练步数下显示更低训练损失 | 原文公式保留未旋转的**分母**以减小零除风险，可能得到负分子/非概率权重；不是与标准 softmax RoPE 数学完全相同。 |
| `4.5 Table 5，中国 CAIL2019-SCM | 对比 512 vs 1024 截断长度的长文本匹配 | 测试 RoFormer-512 68.29%，RoFormer-1024 69.79%；WoBERT-512 68.10% | `68.29\to69.79` 的 **+1.50 个百分点**同时改了输入长度，不可全归因于「RoPE 单独效果」；与 WoBERT-512 比较也混合 tokenizer/位置和长度等因素。 |

[全部数字与实验限制：原文 `4、Table 1–5、`4.5.5](https://ar5iv.labs.arxiv.org/html/2104.09864)。这也解释了不能仅依据原文对长距离衰减的理论分析就保证无限窗口性能：作者自己在局限性中承认尚未充分解释其收敛优势和长文本性能机制。

# 预训练与后训练

**RoPE 是位置注入方式，不是新损失。** 原文分别把它接入原有翻译 Transformer（监督翻译 CE）、BERT 式 Encoder（MLM 预训练）、Performer 变体（字符语言建模），以及中文 WoBERT 衍生的长文本模型（约 34GB 中文 Wikipedia、新闻及论坛训练语料，分不同上下文长度阶段训练）。若仅交换 Q/K 位置机制，模型的 `p_\theta(y\mid x)` 或 MLM 中交叉熵定义不必改变：

$$
L_{\rm MLM}=-\frac{1}{|\mathcal M|}
\sum_{i\in\mathcal M}\log p_\theta(x_i\mid X_{\rm masked}),
$$

其中 `\mathcal M` 是被遮挡位置集合；改变的是最终预测 `p_\theta` 内部 Attention 的 score 函数。`\theta` 包含 Transformer 的 QKV/FFN/词表参数，固定频率 `\theta_i` 不是由这个任务损失更新的参数（为避免符号混淆，这里频率仍记作 `\theta_i`，模型整体参数可记 `\Theta`）。

论文原始训练设置：英德 4.5M 平行句，沿用原 Transformer 类似 BPE/Adam/label smoothing 的翻译训练；英语 BookCorpus+Wikipedia MLM，对照模型训练 100k steps、batch64、最大序列512、AdamW `1e-5`；中文按多个上下文长度与 batch 阶段进行预训练并在文本匹配任务微调。**没有提出 SFT、Reward Model、PPO、DPO、GRPO、机器人动作 chunk 或世界模型训练目标**。后续 Llama 等对 Decoder-only next-token CE 的采用应在相应模型报告中单独描述。

# 推理与部署

## 为什么 RoPE 能保持 Attention 的主复杂度，又增加少量运算

RoPE 对 `Q,K` 的每个二维对做常数次乘加，单层在 `B,h,T,d_h` 上的总额外运算 `O(BhTd_h)`；普通 full self-attention 的 `QK^\top` 和 `AV` 仍是 `O(BhT^2d_h)`。因此**它不会把 dense Attention 的 `O(T^2)` 变成 `O(T)`**，也不会单独消除朴素显式 `[B,h,T,T]` Attention Matrix 的显存。

RoPE 的 `\cos,\sin` 表可按位置预计算、在整个 batch/head 间广播；或者在融合 Attention kernel 中按块生成，以权衡表访存与三角函数/近似计算。由于旋转不改变 Q/K Shape，KV Cache 的标量容量也不因 RoPE 单独下降。原文没有提出 FlashAttention、GQA 或 PagedAttention；它们可以和 RoPE 组合，但各自优化位置不同。

### Prefill 与 Decode：绝对 position_id 是真正的工程陷阱

Prefill 已有 `T` token 时，为每个位置 `0,\ldots,T-1` 计算旋转 Q/K，生成 causal Attention，并将**已按各自 position 旋转的 K** 和未旋转的 V 写入缓存。Decode 新 token 的绝对位置 `m=T` 时，只给它的当前 Q/K 用 `R_{\Theta,m}` 旋转，再与缓存 K 做点积，不应重复旋转旧 K。

如果滑动窗口保留最后 `W` 个 token 却把所有缓存 Key 每次都擅自重编号为 `0,\ldots,W-1`，而当前 Q 继续用全局 `m`，就改变了相对位置关系；若确需重新定位，必须保证窗口内所有 Q/K 的**共同坐标偏移一致**（或者做具有数学证明的重新旋转）。这正是 Position ID 与 KV Cache 的耦合点，和「删除旧 Key 后容量变小」不是同一层问题。

常用的现代 KV 容量公式仍为

$$
M_{\rm KV}\approx2\,L\,B\,T\,n_{\rm kv}\,d_h\,s_{\rm kv},
$$

`n_{\rm kv}` 是否小于 Q heads 取决于 MHA/MQA/GQA 等**另一个技术**；RoPE 自身不修改 `n_{\rm kv}`。它在计算复杂度与内存大小上几乎保持原主项，但很长上下文的三角函数相位、浮点精度和训练外推分布需要额外验证。

## 技术 → 模型

- `RoPE → RoFormer`：`PROPOSES` Q/K 绝对位置旋转与 Attention score 显式相对位移的实现。
- `Sinusoidal PE → RoPE`：`DERIVED_FROM` 多频率三角函数启发，但把绝对位置从输入加法改为 Q/K 的乘法旋转。
- `RoPE → 英德 NMT / BERT 式 MLM / PerFormer`：`IMPLEMENTS` 原文多任务中不同主架构的具体实验。
- `RoPE → Llama 系列`：`ADOPTS` 后续自回归 Decoder 的位置机制，具体 head 维和频率参数按每代报告核对。
- `RoPE 长上下文扩展 → YaRN`：`EXTENDS` 后续对 RoPE 频率/位置范围做重新缩放与训练适配；不是本文发明 YaRN。

## 模型 → 技术

- `RoFormer`：`DERIVED_FROM` 已有 Transformer 自注意力和 BERT/WoBERT 类 Encoder；`PROPOSES` 特征对子空间旋转；`ADOPTS` 原有 MLM/翻译/分类目标及优化器。
- `PerFormer（文中变体）`：`DERIVED_FROM` Performer 线性 Attention；`EXTENDS` 使用分子旋转而分母保持未旋转的相对位置实现，不能视为 vanilla softmax-RoPE 的严格等价替换。
- `后续带 RoPE 的 Decoder-only LM`：`ADOPTS` Q/K 位置旋转；`OPTIONAL` 可叠加 GQA、KV Cache、FlashAttention、NTK/YaRN 等，需按具体模型技术报告标明是否实际采用。

**验收**：写出 `R(m\theta)^\top R(n\theta)=R((n-m)\theta)` 的全部步骤、证明二维旋转保范数、给出 `[B,h,T,d_h]` 的偶奇维广播代码、检查缓存 Key 被旋转的唯一时机，并解释为什么「天然相对位置」既不等于「严格单调长程衰减」也不等于「无限长度外推保证」。


## 从这里继续：RoPE 为什么还不能天然支持无限上下文？

RoPE 的点积虽然只显式依赖相对位移，但训练只覆盖有限的相对距离与 phase range，因此“relative position”并不自动推出“无限长度外推”。下一步直接进入 [YaRN：RoPE 长上下文外推](A018-yarn.md)：从 direct extrapolation failure、Position Interpolation 一直推到 NTK-by-parts、attention magnitude scaling 与 Dynamic YaRN。