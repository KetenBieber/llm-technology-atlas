> **原文**：Vaswani 等，[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)，NeurIPS 2017，[全文（带公式/表格）](https://ar5iv.labs.arxiv.org/html/1706.03762)。本文以**原始 Encoder–Decoder、Post-LN Transformer** 为对象；后来的 Decoder-only GPT、Pre-LN、RoPE、GQA、FlashAttention 是相关延伸，不能倒灌成 2017 年原文提出的模块。对应 Paperlist A/03 第 3 篇。

# 模型定位与谱系

## 论文属于哪一类，究竟替换了什么

**主要类型：A 原始方法论文；次要属性：可训练的完整英德/英法翻译模型报告。** 本文的核心贡献不只是「使用 Attention」——上一篇 Bahdanau 已有可微加权读取。作者真正改动的是：**让 Encoder 与 Decoder 的序列内部表示交互不再依赖逐时间步 RNN/CNN，而由 self-attention 与逐位置 FFN 构成，同时保留 Encoder–Decoder 的条件翻译框架**。

~~~text
RNN Seq2Seq：Encoder 随时间递推 → 固定语句状态 → Decoder 递推
                         ↓ 长程梯度路径及前向串行
Bahdanau：Encoder 保存全源状态，Decoder 对源做交叉 Attention
                         ↓ 仍需按时间递推 RNN 才能更新源/目标状态
替代思路：所有位置并行构造 Q/K/V，让每个位置直接访问其他位置
                         ↓
            Scaled Dot-Product Attention
                         ↓
             Multi-Head + Positional Encoding
                         ↓
 Encoder：全可见 self-attention + FFN
 Decoder：Masked self-attention + Cross-attention + FFN
                         ↓
             Transformer Encoder–Decoder
                         ↓
BERT Encoder / GPT Decoder-only / ViT / VLM / VLA 的后续技术谱系
~~~

技术树主路径：`序列架构 → Attention-based Encoder–Decoder → 全 Attention 主干 → Transformer`；其内部支路应是 `Attention → Scaled Dot-Product → Multi-Head`，不能把「Position Embedding」或「LayerNorm」与整个 Transformer 同层标成论文主架构。

**为什么需要出现：** RNN 的隐藏状态 `h_t=f(h_{t-1},x_t)` 导致同一层的第 `t` 步必须等第 `t-1` 步结束，训练一个长句存在跨时间递推串行依赖。CNN 可并行但若局部卷积核宽有限，远距离信息要穿过多层或使用 dilation。Self-attention 在单层中允许任意两个位置建立直接数据依赖，使整条序列在一次批量矩阵计算中共同更新。代价是所有位置两两相关时计算与中间注意力矩阵随 `T^2` 增长，而不是「注意力完全消除了长序列成本」。[原文 `1–2 Table 1](https://ar5iv.labs.arxiv.org/html/1706.03762)

# 输入、输出与任务

## 首先区分翻译模型与今天的纯 Decoder 语言模型

本文目标是给源序列 `x_{1:S}` 生成目标序列 `y_{1:T}`，训练仍使用教师强制的目标输入右移。定义批量大小 `B`，源长度 `S`，目标长度 `T`，模型宽 `d_{\rm model}=512`，Attention 头数 `h=8`，每头 `d_k=d_v=64`，前馈内层宽 `d_{\rm ff}=2048`，Encoder/Decoder 各 `L=6` 层。

| 对象 | Shape | 所在位置 |
|---|---|---|
| 源 token ID | `[B,S]` | Encoder 输入 |
| 目标前缀 token ID | `[B,T]` | Decoder 输入，训练时右移 |
| Token Embedding + Positional Encoding | `[B,S,512]` / `[B,T,512]` | 两侧输入 |
| 每个头的 Q/K/V（Encoder） | `[B,8,S,64]` | Encoder self-attention |
| 每头 Encoder 注意力分数/概率 | `[B,8,S,S]` | 源位置两两相连 |
| Encoder 输出 Memory | `[B,S,512]` | 一次 Encoder 完成，供每层 Decoder Cross-Attention |
| Decoder self Q/K/V | `[B,8,T,64]` | 目标位置因果注意力 |
| Decoder Cross Q | `[B,8,T,64]` | 当前目标状态提取 Query |
| Decoder Cross K/V | `[B,8,S,64]` | Encoder Memory 提取 Key/Value |
| Decoder Cross Attention | `[B,8,T,S]` | 每个目标位置读取源位置 |
| Token logits | `[B,T,V_t]` | 预测目标词表 |
| 监督目标 | `[B,T]` | 真实目标 token，忽略 PAD |

**Self-attention**：Query/Key/Value 来自同一序列的不同线性映射。**Cross-attention**：Query 来自 Decoder，Key/Value 来自 Encoder。**Masked self-attention**：计算当前目标 token 时不得读取后续真实目标 token；这跟 PAD mask 是两个逻辑不同的 mask。

原文 Decoder 并非生成 B×T×512 的「动作表示」，输出是源条件的目标语言词表分布。VLM/VLA 后续可以把视觉或动作也编码为 token，但这种模态扩展不属于本文原始任务。

# 骨干架构与信息交互

## 标准原始六层 Encoder + 六层 Decoder

~~~mermaid
flowchart TD
  Src["源 token Embedding × sqrt(d) + Sinusoidal PE"] --> E1["Encoder Block 1"]
  E1 --> EN["重复 6 层：Self-MHA → Add&Norm → FFN → Add&Norm"]
  EN --> Mem["Encoder Memory [B,S,512]"]
  Tgt["右移目标 Embedding × sqrt(d) + Sinusoidal PE"] --> D1["Decoder Masked Self-MHA"]
  D1 --> DN1["Residual + LayerNorm"]
  DN1 --> Cross["Cross-MHA：Decoder Q 读取 Encoder K/V"]
  Mem --> Cross
  Cross --> DN2["Residual + LayerNorm"]
  DN2 --> FFN["逐 token FFN + Residual + LayerNorm"]
  FFN --> More["重复 Decoder Block，共 6 层"]
  Mem --> More
  More --> Head["线性投影 + Softmax → 目标词"]
~~~

**位置编码**先与 Token Embedding 相加，不是被当成另一条需要 softmax 归一化的词序列。Encoder 每层先 self-attention 再逐 token FFN；Decoder 每层多出一个 Encoder–Decoder Cross-Attention 子层。每个子层用 `\operatorname{LayerNorm}(x+\operatorname{Dropout}(\operatorname{Sublayer}(x)))`（原文 Post-LN）连接；本论文并非后来的 `x+\operatorname{Sublayer}(\operatorname{LayerNorm}(x))` Pre-LN 布局。

### 如果删去某个模块会怎样

删除 Encoder self-attention，源词无法融合其他源位置，Encoder Memory 将接近独立词表查值；删除 Decoder causal mask，训练时当前位置可读取未来真实词，导致数据泄漏；删除 Cross-Attention，目标生成退化为不带源条件的语言模型；删除 Positional Encoding，若其余层仅有全局 self-attention 和共享逐位置 FFN，重排输入词会造成相应的置换等变，无法表达一般的词序依赖；删除逐位置 FFN，虽然仍有 token 间消息，但每个位置的特征变换能力与非线性组合变弱。

原始位置编码是固定三角函数，而词向量、QKV 和 FFN 都参与训练。论文也对可学习位置编码做过消融，不能把固定三角函数解释为**唯一**可行的序列位置信息。

# 关键技术

## 1. 从注意力的一般概率读取需求推导 Q/K/V

给定某目标位置的查询向量 `q`，现有源或历史序列的 `S` 个被查询 Key `k_j` 及对应 Value `v_j`，最自然的加权读取是

$$
o=\sum_{j=1}^{S}\alpha_j v_j,\qquad
\alpha_j\ge0,\quad\sum_j\alpha_j=1.
$$

问题在于如何学习 `\alpha`。Bahdanau 采用带隐藏层的加性评分；本论文使用线性投影把原始位置表示 `x_i` 转换成

$$
q_i=x_iW_Q,\quad k_i=x_iW_K,\quad v_i=x_iW_V.
$$

对于整个 batch，`X\in\mathbb R^{B\times T\times d_{\rm model}}`，用 `W_Q\in\mathbb R^{d_{\rm model}\times hd_k}` 得到 `Q\in\mathbb R^{B\times T\times hd_k}`；`K,V` 同理。reshape 后是 `[B,h,T,d_k]` 与 `[B,h,T,d_v]`。Key 的职责是决定相关性，Value 的职责是贡献内容；它们来自同一位置但**并非必须相同**。

评分写 `e_{ij}=q_i^\top k_j/\sqrt{d_k}`，在所有合法 Key 上 softmax：

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}
\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
}
$$

`M` 在合法位置为 0、非法位置为 `-\infty`；原文核心公式省略可选 Mask，而实际 Decoder 必须添加 causal mask。**softmax 沿 Key 位置轴 `S` 归一化**，因此矩阵顺序是 `[T,S]@[S,d_v]\to[T,d_v]`，不能颠倒成 `KV^\top`。

## 2. 为什么除以 `\sqrt{d_k}`：方差必须具体算出来

假设用于分析的每个坐标满足 `\mathbb E[q_\ell]=\mathbb E[k_\ell]=0`、`\operatorname{Var}(q_\ell)=\operatorname{Var}(k_\ell)=1`，不同坐标及 `q,k` 彼此独立。这些是假定条件，不代表训练后的 Q/K 恰好独立同分布。未经缩放的内积

$$
z=q^\top k=\sum_{\ell=1}^{d_k}q_\ell k_\ell.
$$

因为各项均值 0，而 `\operatorname{Var}(q_\ell k_\ell)=\mathbb E[q_\ell^2]\mathbb E[k_\ell^2]=1`，独立求和得到

$$
\operatorname{Var}(z)=\sum_{\ell=1}^{d_k}1=d_k,
\qquad
\operatorname{Std}(z)=\sqrt{d_k}.
$$

因此 `d_k=64` 时标准差为 8；除以 `\sqrt{64}=8`，理论标准差回到 1 的数量级。softmax 在 logits 绝对值差距过大时接近 one-hot，其他位置梯度很小，缩放可缓解此效应。若 Q/K 坐标相关或做了其他归一化，这个方差论证不再精确；**`\sqrt{d_k}` 是这组常见假设下的尺度控制，不是普适概率等式**。[原文 `3.2.1](https://ar5iv.labs.arxiv.org/html/1706.03762)

### 两个 Key 的小数字例子

取 `d_k=2`、`q=(1,1)`、`k_1=(1,1)`、`k_2=(0,1)`、Value 分别 `v_1=(2,0)`、`v_2=(0,4)`。未归一化分数是 `[2,1]`，缩放后 `[1.4142,0.7071]`；softmax 近似 `[0.6698,0.3302]`，输出

$$
o\approx0.6698(2,0)+0.3302(0,4)
=(1.3396,1.3208).
$$

这个向量不是简单复制任何一个 Value，而是从候选位置按查询相关性汇总信息；当位置被 causal mask 时，该位置概率**严格变为 0**（有限精度实现下需使用足够小的数、mask 语义正确）。

## 3. 多头为何不等于「把相同 Attention 重复 8 次」

单头如果只有一组 `W_Q,W_K,W_V`，网络必须用同一相关性表示来兼顾句法、指代、词组搭配等不同关系。多头对相同输入采用**不同可学习投影**：

$$
\operatorname{head}_r
=\operatorname{Attention}(QW_Q^{(r)},KW_K^{(r)},VW_V^{(r)}),
\quad r=1,\ldots,h,
$$
$$
\operatorname{MHA}(Q,K,V)
=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W_O.
$$

原文基本配置 `h=8,d_k=d_v=64`，concat 回 `8\times64=512`，再经 `W_O\in\mathbb R^{512\times512}` 得到模型宽度。每头独立参数能学习不同的相似性空间；没有任何数学定理保证头 `r` 一定对应语法关系 `r`，这些解释需要单独可解释性实验。多头不会凭空把最终隐藏宽 `512` 增成 `8\times512`；它是在相近的总投影宽度内分割多种关系。

单层 Decoder self-attention 的 `Q,K,V:[B,8,T,64]`，每头输出 `[B,T,64]`，拼接后 `[B,T,512]`；Cross-Attention 改变的是 `K,V` 的位置长度从 `T` 变成 `S`，并不要求 `S=T`。

## 4. 自注意力没有时间递推，为什么位置编码不能省

设输入序列 `X=[x_1,\ldots,x_T]`，对任何位置置换矩阵 `P`，若只保留无 mask 的 self-attention 和逐位置 FFN，则 `Q'=PQ,K'=PK,V'=PV`。其相关矩阵

$$
Q'K'^\top=(PQ)(PK)^\top=P(QK^\top)P^\top.
$$

softmax 沿行计算时满足 `\operatorname{softmax}(PAP^\top)=P\operatorname{softmax}(A)P^\top`（对同一置换后的轴）；因此

$$
\operatorname{Attention}(PX)
=P\operatorname{Attention}(X).
$$

这意味着不加入位置表示时，模型对源词顺序的重新排列只能相应重排输出，而很难区分「狗咬人」与「人咬狗」的次序信息。原文给位置 `pos`、偶/奇通道 `2i,2i+1`：

$$
PE_{pos,2i}=\sin\left(pos/10000^{2i/d_{\rm model}}\right),\qquad
PE_{pos,2i+1}=\cos\left(pos/10000^{2i/d_{\rm model}}\right).
$$

不同通道具有不同频率，覆盖近到远的序列位置。三角恒等式 `\sin(a+b)=\sin a\cos b+\cos a\sin b` 说明位置偏移可以在线性变换意义下从这对通道表示中组合出来，给网络学习相对位置信号的机会，但**原文并没有保证模型能精确外推到任何未训练长度**。

## 5. 逐位置 FFN、Residual 和 Post-LN：注意力之外还做了什么

Encoder/Decoder 各层的 FFN 对任意位置表示 `x\in\mathbb R^{512}` 使用共享两层 MLP：

$$
\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2,
$$

`W_1:[512,2048]`、`W_2:[2048,512]`；它对每个位置**独立**执行，不在序列轴聚合信息。Attention 做跨 token 信息交互，FFN 负责在每个 token 表示维度上进行非线性特征重组，两者不可互相替代。

原文每个子层的输出

$$
z=\operatorname{LayerNorm}(x+\operatorname{Dropout}(F(x))).
$$

梯度仍有残差的 `I` 支路，但经过后置 LayerNorm 的 Jacobian，不能直接照搬 Pre-LN 长网络中「未经 Norm 的恒等梯度可贯穿所有层」的结论。以后阅读 RMSNorm、Pre-LN、HyperConnections 时，应记录它们改变的是**归一化/残差流结构**，不是本篇已拥有这些设计。

## 6. PyTorch 最小程序：完整显式 QKV、自/交叉 Attention 与 Post-LN 的张量流

~~~python
import math
import torch
from torch import nn
import torch.nn.functional as F

class ExplicitMHA(nn.Module):
    def __init__(self, width=32, heads=4):
        super().__init__()
        assert width % heads == 0
        self.h, self.d = heads, width // heads
        self.wq = nn.Linear(width, width, bias=False)
        self.wk = nn.Linear(width, width, bias=False)
        self.wv = nn.Linear(width, width, bias=False)
        self.wo = nn.Linear(width, width, bias=False)

    def split(self, x):                                # [B,T,width]
        b, t, _ = x.shape
        return x.reshape(b, t, self.h, self.d).transpose(1, 2)

    def forward(self, query, memory, causal=False):   # [B,T,w],[B,S,w]
        q = self.split(self.wq(query))                 # [B,h,T,d]
        k = self.split(self.wk(memory))                # [B,h,S,d]
        v = self.split(self.wv(memory))                # [B,h,S,d]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d)
        # scores [B,h,T,S]；严格因果掩码只用于 T==S 的目标自注意力。
        if causal:
            t, s = query.shape[1], memory.shape[1]
            assert t == s
            forbidden = torch.ones(t, s, dtype=torch.bool,
                                   device=query.device).triu(diagonal=1)
            scores = scores.masked_fill(forbidden, float("-inf"))
        alpha = scores.softmax(dim=-1)                 # [B,h,T,S]
        weighted = alpha @ v                           # [B,h,T,d]
        joined = weighted.transpose(1, 2).contiguous().reshape(
            query.shape[0], query.shape[1], -1)        # [B,T,width]
        return self.wo(joined), alpha

class EncoderBlockV1(nn.Module):
    def __init__(self, width=32, heads=4, ffn=64):
        super().__init__()
        self.attn = ExplicitMHA(width, heads)
        self.ffn = nn.Sequential(nn.Linear(width, ffn),
                                 nn.ReLU(), nn.Linear(ffn, width))
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)

    def forward(self, x):
        a, _ = self.attn(x, x)                          # 全源 self-attn
        x = self.norm1(x + a)                         # 原始 Post-LN
        return self.norm2(x + self.ffn(x))

class DecoderBlockV1(nn.Module):
    def __init__(self, width=32, heads=4, ffn=64):
        super().__init__()
        self.self_attn = ExplicitMHA(width, heads)
        self.cross_attn = ExplicitMHA(width, heads)
        self.ffn = nn.Sequential(nn.Linear(width, ffn),
                                 nn.ReLU(), nn.Linear(ffn, width))
        self.n1, self.n2, self.n3 = (nn.LayerNorm(width)
                                    for _ in range(3))

    def forward(self, prefix, encoded):
        a, causal_weights = self.self_attn(prefix, prefix, causal=True)
        x = self.n1(prefix + a)
        c, cross_weights = self.cross_attn(x, encoded, causal=False)
        x = self.n2(x + c)
        return self.n3(x + self.ffn(x)), causal_weights, cross_weights

torch.manual_seed(4)
src = torch.randn(2, 5, 32, requires_grad=True)
tgt = torch.randn(2, 4, 32, requires_grad=True)
encoder, decoder = EncoderBlockV1(), DecoderBlockV1()
memory = encoder(src)                                   # [2,5,32]
out, causal_weights, cross_weights = decoder(tgt, memory)
assert out.shape == (2, 4, 32)
assert causal_weights.shape == (2, 4, 4, 4)
assert cross_weights.shape == (2, 4, 4, 5)
assert causal_weights[..., 0, 1:].abs().sum() == 0     # 不能偷看未来
out.square().mean().backward()
assert src.grad is not None and tgt.grad is not None
print("encoder memory / causal self / cross / backward shapes OK")
~~~

逐行对应：`wq,wk,wv` 是三组**独立学习**矩阵，`split` 把宽度 `32` 分成四个 `d=8` 头；`scores` 对 Query 和 Key 的最后维度做点积并除 `sqrt(d)`；`causal` 用严格上三角 mask 抹掉未来 Key 的概率，不能用于 Encoder 或普通 Cross-Attention；`alpha@v` 完成按 Key 长度加权；`transpose+reshape` 拼接头，`wo` 回到原始 `width`；两个 Block 均按原论文 Post-LN 组织残差，Decoder 多一个 Cross-Attention；末尾两个 alpha 张量分别体现 `[T,T]` 与 `[T,S]`，其不必是相同正方矩阵；第一个目标位置对未来 token 的注意力和为 0，最后 `backward` 证明目标梯度能穿过 Cross-Attention 回到 Encoder。为突出张量依赖，上述示例省略 PAD masks、位置编码、dropout、目标词 Embedding 和实际翻译训练；完整原文实验须添加它们。

## 7. 实验解读：作者真正控制了哪些因素

| 原文实验/表 | 试图检验的关系 | 作者报告 | 不能因此声称 |
|---|---|---|---|
| Table 2：WMT’14 EN→DE | 原始全 Attention 架构在相近公开机器翻译指标上的效果 | Transformer Base 测试约 27.3 BLEU；Big 约 28.4 BLEU | Big 与 Base 同时改变宽度、dropout 与训练预算，不能把差异解释为头数单独收益。 |
| Table 2：WMT’14 EN→FR | 扩展到更大平行语料是否可行 | Big 达到原文表格的高质量 BLEU 水平（不同版本正文/表格存在细节） | 不应把它当成 Seq2Seq 在相同参数/硬件条件下的严格速度比较。 |
| `5.3 Table 3：head count / `d_k` | 多头与单头的训练效果关系 | 单头损失/翻译指标不及若干多头配置；头数太多也不总是更好 | 不是「头数越多必然越强」的单调定理。 |
| `5.3 Table 3：位置编码 | 固定正弦 vs learned position embeddings | 论文报告两种选择的性能接近 | 不能据此认定正弦编码支持任意长度的无损外推。 |
| `5.4 English constituency parsing | 架构能否迁移至非翻译序列任务 | 在特定解析设置上得到有竞争力的分数 | 与所有下游自然语言理解或现代具身决策表现无直接等价关系。 |

[原文 `5、Table 2–4](https://ar5iv.labs.arxiv.org/html/1706.03762)。**模型最终 BLEU、单步训练 FLOPs、集群 wall-clock、参数量和单 token 推理延迟是不同度量**；不能只因为 GPU 训练更能并行，就推断 autoregressive decoder 一次只生成一个 token 的串行依赖被消除了。

# 预训练与后训练

**论文主体是有监督机器翻译训练，不提出现代 LLM CPT/SFT/RLHF/DPO 流程。** WMT’14 英德训练集约 450 万句对，英法约 3600 万句对；原文将文本分割为 subword 单元，英德共享 BPE 词表约 37K。对于目标输入前缀 `y_{<t}` 的因果遮罩 Decoder，优化

$$
L(\theta)=-\frac{1}{\sum_bT_b}\sum_{b,t}
\log p_\theta(y_{b,t}\mid y_{b,<t},x_{b,1:S_b}).
$$

原论文的实现以 label smoothing `\epsilon_{\rm ls}=0.1` 修正训练目标分布，例如一种常见等价记号为

$$
q_k=(1-\epsilon_{\rm ls})\mathbf1_{k=y}
+\epsilon_{\rm ls}/V_t,
\qquad
L_{\rm smooth}=-\sum_{k=1}^{V_t}q_k\log p_\theta(k).
$$

若框架把平滑质量分配到「非目标 `V_t-1` 个类别」而不是全部 `V_t` 类别，具体数值略不同，须核对 API；论文还在评测时使用未平滑的 token NLL/perplexity 与 BLEU 等指标，不能把提高训练平滑 loss 直接说成概率校准必然改善。

原文采用 **Adam**，`\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9}`，以及现称 Noam 的学习率计划：

$$
\operatorname{lr}(step)
=d_{\rm model}^{-1/2}
\min\left(step^{-1/2},
step\cdot warmup^{-3/2}\right),
\qquad warmup=4000.
$$

前 4000 步近似线性 warmup，之后按 `step^{-1/2}` 下降。这是**优化器调度**，不是 Attention 的一部分；原文 Post-LN、残差 dropout 等也属于训练/架构具体实现。

# 推理与部署

## 不许混淆数学计算、注意力矩阵存储和 HBM 访存

对于单层 Decoder self-attention，长度为 `T`、宽度为 `d_{\rm model}`、`h` 头：线性投影近似 `O(BTd_{\rm model}^2)`；`QK^\top` 和 `AV` 的主项约 `O(BhT^2d_k)`；若朴素显式存储全部注意力矩阵，`A:[B,h,T,T]` 占 `BhT^2` 元素。**原始 Transformer 没有把标准 Attention 数学 FLOPs 从 `O(T^2)` 降到线性**；后来的 FlashAttention 主要优化中间张量 materialization/HBM IO，而不取消精确 dense Attention 的 `T^2` 点积数量级。

### 一次实际的机器翻译请求

~~~text
输入源文本 → BPE/Embedding + PE
          → Encoder 六层：对全部源位置计算全向 self-attention
          → 保存最终 Encoder Memory
          → Decoder 初始 <bos>
          → 当前目标前缀 masked self-attention
          → cross-attention 读取同一份 Encoder Memory
          → FFN → 词表 logits → beam/greedy 选择目标 token
          → 将新 token 加入前缀，重复到 <eos>
~~~

在现代优化实现里，Encoder 可以只算一次；Decoder 各层的**源 K/V 投影**可由固定 Encoder Memory 预计算并复用；Decoder 自注意力的历史 K/V 则随生成 token 数增长。假设 Decoder `L` 层、self-attention `h` 个 KV heads、单头 `d_k`、历史目标长 `T`、批大小 `B`、每元素 `s` 字节，self-attention KV Cache 大小近似

$$
M_{\rm selfKV}=2BLThd_ks.
$$

Decoder 的 Cross-Attention K/V 另约 `2BLS\,h\,d_ks`（各层 Cross-Attention 可具有不同投影，不能只算一次所有层共用），Decoder 工作空间与 Encoder Memory 再另外统计。原文并未提出 KV Cache 分页、连续批处理或 Speculative Decoding；上述是依据其架构作的现代部署预算。

例如 `B=1,L=6,T=512,h=8,d_k=64,s=2` bytes 时，Decoder self-KV 约 `2\times1\times6\times512\times8\times64\times2\approx6.29` MB（十进制），**不含 Encoder Memory 与 Cross-KV**。Prefill 可对已有目标前缀并行注意力；Decode 每次新增一个 token 的 self-attention 单步工作量约 `O(BhTd_k)`，但仍须读取模型权重、历史 KV，并按自回归顺序依次生成 `T_{\rm gen}` 个输出；这些硬件瓶颈不能从训练表格的 8 GPU 用时直接推断。

## 技术 → 模型

- `Scaled Dot-Product Multi-Head Self-Attention → Transformer`：`PROPOSES` 将缩放点积与多头设计用于完全基于 Attention 的主架构。
- `Additive Attention → Transformer Cross-Attention`：`DERIVED_FROM` 动态读取源序列的任务结构；`EXTENDS` 为点积式多头而不是照抄加性 scorer。
- `Sinusoidal Positional Encoding → 原始 Transformer`：`IMPLEMENTS` 无时间递推下的位置注入方案。
- `Transformer Encoder → BERT、ViT`：`ADOPTS/EXTENDS` 可并行处理可见上下文的骨干思想，具体预训练目标与视觉 Patch 表示需逐篇核实。
- `Transformer Decoder → GPT 家族`：`ADOPTS/EXTENDS` Masked self-attention、自回归生成，通常不再使用本文独立 Encoder→Cross-Attention 的完整翻译框架。

## 模型 → 技术

- `Transformer Base`：`PROPOSES` 完整六层 Encoder/Decoder 全 Attention 架构；`IMPLEMENTS` 8 heads、512 宽、2048 FFN、固定正弦位置；`ADOPTS` Adam、监督翻译 CE、残差和 LayerNorm。
- `Bahdanau RNNsearch`：`ADOPTS` RNN Encoder–Decoder 和动态源信息检索；`PROPOSES` 加性打分与 soft alignment，仍使用 RNN 递推。
- `后续 Decoder-only GPT`：`DERIVED_FROM` Transformer Decoder 的因果自注意力构造；`IMPLEMENTS` 不同模型的 pretraining tokenizer、Norm、位置编码和 Attention 变体，而不是原文自动包含所有后续发明。

**验收**：给 `B=2,S=5,T=4,d=512,h=8`，推导 Encoder `[2,8,5,5]`、Decoder 自注意力 `[2,8,4,4]`、Cross `[2,8,4,5]` 三张权重矩阵；证明 `\sqrt{d_k}` 的独立分量方差来源，解释为什么 original Post-LN/正弦位置与今天的 Pre-LN/RoPE 不能混写，并分别估算训练显式注意力存储、推理 self-KV 与 Cross-KV。