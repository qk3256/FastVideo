# AMX-GEMM methodology → H3 Attention 映射

| AMX-GEMM technique | H3 Attention analogue | 适用? | 证据 |
|---|---|---|---|
| ISA/硬件利用率统计(perf计数器) | Nsight Compute / torch profiler stall rows | 部分 | `B_fa_ncu_diagnosis.md`；fwd 57.8% TC, bwd 53.5% TC(受限 ncu 权限，其余直测) |
| L1/L2/L3 bottleneck 分解 | register-file + SMEM + L2 + HBM 四层 | 是 | 报告 fwd: TC 受限,L2/DRAM 远未饱和;bwd: 1 CTA/SM SMEM 上线 |
| TM/TN/TK blocking | FA block-tile 配置 | 部分 | thread config 是来自 库选择;tile 明面参数不动 modify — PA exec 提示 |
| weight packing(consumer-native layout) | QKV packed producer layout | 否 | flash-attn 没有训练可用 packed API(Drop) |
| strided→dense layout | producer 写 consumer 布局 | 部分 | D1 冻结 dtype 路径；D2 手写代价不合算 |
| software prefetch | compute/comm overlap | 部分 | vB QKV-A2A overlap 有 ~0.5%/step;INVESTIGATE |
| autotune per shape | 每 shape 固定 config | 否 | SDPA flash 已是 locked choice,sm80 无更优 kernel |
| hot/cold cache test | compile cache 膨胀检测 | 是 | compile integration 已测 warm 2.5s/冷 23.9s |

**核心结论**:

可迁移的是"方法"：分灶拆解 compute/occupancy/memory/stall → 结论一致(FA fwd/bwd 接近几何极限)。

不可迁移的是"实现"：你做不了换 tile、换 kernel、换 layout 的固定动作，因为 FA 那 39.6% 是 dense attention 的不可避计算；真正能"多拿走"的那点尾巴已经被 regional compile -11.65% + Final AdaLN -8.41% 拿走，组合 -21.7% 为可交付成果。
