# 偏好学习与推理后训练

这一部分把两条现代后训练路线放到一起：DPO 从 pairwise preference 直接得到 policy objective；GRPO 则保留在线 rollout 与 policy gradient，但用同题多样本构造 group-relative advantage。

```{toctree}
:maxdepth: 2

../generated/papers/A/08-post-training/A051-deepseekmath-grpo
../generated/papers/A/08-post-training/A052-dpo
```
