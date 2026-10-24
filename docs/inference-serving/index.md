# LLM 推理系统：从动态请求到精确投机采样

从 [vLLM / PagedAttention](../generated/papers/C/01-distributed-training/C012-vllm-pagedattention.md) 进入该方向时，先明确要优化的对象：

- **动态调度**：Orca、Sarathi-Serve —— 请求会持续到达、结束，Prefill 与 Decode 的工作量完全不同。
- **分阶段 GPU 池**：DistServe、Splitwise —— TTFT/TPOT 的 SLO、KV 跨卡转移、GPU 成本与功率。
- **Serving Runtime / Kernel**：SGLang、FlashInfer、Punica —— 公共前缀、分页 KV、LoRA Adapter 多租户。
- **单请求生成算法**：Speculative Decoding、Medusa、EAGLE —— 提前预测多个 Token，Target 验证，分布精确性与接受长度。

推理系统不要用同一个“更快”概括四类指标：**TTFT、TPOT/TBT、带 SLO 的 Goodput、Token/Sec** 分别测量不同对象。系统能支持更长 Context 也不表示模型一定能理解该 Context。

~~~{toctree}
:maxdepth: 2

../generated/papers/C/01-distributed-training/C012-vllm-pagedattention
../generated/papers/C/02-inference-serving/C013-orca
../generated/papers/C/02-inference-serving/C014-sarathi-serve
../generated/papers/C/02-inference-serving/C015-distserve
../generated/papers/C/02-inference-serving/C016-splitwise
../generated/papers/C/02-inference-serving/C017-sglang
../generated/papers/C/02-inference-serving/C018-flashinfer
../generated/papers/C/02-inference-serving/C019-punica
../generated/papers/C/02-inference-serving/C020-speculative-decoding
../generated/papers/C/02-inference-serving/C021-medusa
../generated/papers/C/02-inference-serving/C022-eagle
~~~
