# What N=4 results can and cannot imply for full MiniMax-H3

MiniMax-H3's real transformer = 50 main blocks + 2 token-refiner blocks, all
structurally identical (same hidden 5376 / 56 heads / RoPE partial-96-path /
flash pattern). We measured reduced N=4 real SFT on 2×A100 40GB only.

## 每一个 KEEP/agent 候选按三个角度回答

### 1. 是否作用于重复 transformer block?

- regional compile: yes — 每个 main/refiner block 上各自编译,dtype/shape/stride 的 stable 签名逐 block 重复出现
- final AdaLNOut 2-row: partially — 只有 final 的 norm_out.linear 一处;与模型深度无关,不随 N 扩展 (但它的 isolated 值 509ms 在 50 层全步时长同样恒定)
- 其余被 DROP 的候选(skip 列表):无论是否作用于重复 block,都因为收益/天花板不足或方向禁止而保留为研究结论。

### 2. 不线性扩展的因素

- optimizer step 时间含冷启 compile 雅 cache 条件、全局同步点(维度小的 all gather barrier、profiler/日志等),不随 N 线性涨。
- FSDP ReduceScatter/AllGather 的通信时间 与 A2A skew 层数无关,在 50 层会占比相对更大(总步时间变长,通信暴露会被分摊)。
- 显存: 全 46 层瞬时栈/帧分配依然是在最大 layer_cycle 重复,并不随 N 一线性扩。
- Kernel cache/Inductor cache: 首次编译摊一次。

### 3. 允许的表述(只有这种可以)

> 在真实 N=4 MiniMax H3 SFT(2×A100,SP=2,bf16)上测得 step-time -21.72%(regional compile + final AdaLN 组合)。这些机制作用于同构的 transformer block 或全局算子(AbstractLevel),因此预期在全模型上仍能提供收益;但当前 2×A100 40GB 全模型 SFT 显存不可行,未能实测 50 层,故不把 N=4 百分比视为 full-H3 已验证结果。

### 4. 禁止的表述

- “N=4 -21.72%,所以 50 层也是 -21.72%。”
- “full MiniMax-H3 已验证提速 X%。”

## 结构性外推说明(合规的)

- 主 transformer block ×46 + refiner ×2 见到更多 as(block×同样的调度要素): all-slice RMSNorm、attention、FFN、AdaLN tables,因此 regional compile 的每 block 收益是单 block 内的,而 repeat count 在 50 层是本来设计——结构上更大的分母让恒右的 optimizer/FSDP 占比变小,因此可以**合理预期 N=4 到 50 层的相对收益不会缩水反而温和放大**(flash 占比上升);但该推断并未被 50 层 direct measure 证明,报告不用于承诺。
