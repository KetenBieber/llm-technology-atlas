# LLM 推理系统：从动态请求到精确投机采样

从 [vLLM / PagedAttention](../generated/papers/C/01-distributed-training/C012-vllm-pagedattention.md) 进入该方向时，先明确要优化的对象：

- **动态调度**：Orca、Sarathi-Serve —— 请求会持续到达、结束，Prefill 与 Decode 的工作量完全不同。
- **分阶段 GPU 池**：DistServe、Splitwise —— TTFT/TPOT 的 SLO、KV 跨卡转移、GPU 成本与功率。
- **Serving Runtime / Kernel**：SGLang、FlashInfer、Punica —— 公共前缀、分页 KV、LoRA Adapter 多租户。
- **候选与验证**：Speculative Decoding、Medusa、EAGLE 系列、Hydra、ReDrafter、SpecInfer、Sequoia —— 从分布校正进入候选网络、树形并行验证和动态预算。
- **不依赖独立神经草稿的技术**：Lookahead、REST、Draft & Verify、LayerSkip；Ouroboros 则在保留 Draft 的基础上复用短语。
- **近似大小模型协作**：BiLD 以回退和回滚权衡质量与延迟，不能与严格无损采样混称。

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
../generated/papers/C/02-inference-serving/C023-eagle2
../generated/papers/C/02-inference-serving/C024-eagle3
../generated/papers/C/02-inference-serving/C025-hydra
../generated/papers/C/02-inference-serving/C026-redrafter
../generated/papers/C/02-inference-serving/C027-specinfer
../generated/papers/C/02-inference-serving/C028-sequoia
../generated/papers/C/02-inference-serving/C029-lookahead-decoding
../generated/papers/C/02-inference-serving/C030-rest
../generated/papers/C/02-inference-serving/C031-ouroboros
../generated/papers/C/02-inference-serving/C032-draft-and-verify
../generated/papers/C/02-inference-serving/C033-layerskip
../generated/papers/C/02-inference-serving/C034-bild
~~~
