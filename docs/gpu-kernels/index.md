# GPU、算子与编译

这一专题不再只看模型公式，而是研究公式怎样映射到真实 accelerator：哪些算子是 compute-bound，哪些是 memory-bound，HBM / SRAM 的数据搬运如何决定 wall-clock，以及怎样让算法结构与 GPU execution model 一起设计。

FlashAttention 是入口：它保持 dense Attention 数学定义不变，却通过 IO-aware tiling、online softmax 与 backward recomputation 避免 materialize (N\times N) 中间矩阵。

FlashAttention-2 在这个基础上继续向 GPU execution model 下钻：减少昂贵的非 matmul FLOPs，把 sequence length 也变成 thread-block parallel dimension，并把 warp 内 split-K 重构为 split-Q，减少 shared-memory communication。

FlashAttention-3 再把问题推进到 Hopper 的异步执行模型：TMA 负责数据搬运，WGMMA 负责 Tensor Core GEMM，consumer warpgroups 与 CUDA/SFU 路径执行 softmax；通过 warp specialization、circular SMEM buffer 和 GEMM-softmax pipeline，让不同硬件引擎尽量同时工作。FP8 部分则进一步把 memory layout 与 quantization geometry 纳入 kernel 设计。

```{toctree}
:maxdepth: 2

../generated/papers/A/10-gpu-operators/A055-flashattention
../generated/papers/A/10-gpu-operators/A056-flashattention2
../generated/papers/A/10-gpu-operators/A057-flashattention3
```
