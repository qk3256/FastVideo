# Attention & 全链路优化最终决策（N=4 H3 SFT, 2xA100, SP=2, baseline 653040aa family)

全部为实测 GPU 结果（isolated 与 full-step 分列）；不允许用 FLOPs 反推 full-step。

| candidate | baseline cost | 碰 FA 核心? | exact? | isolated speedup | full-step speedup | memory | 实现成本 | 数值风险 | 证据强度 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| regional torch.compile (blocks) | cast/pointwise soup ~460ms+ bubbles | 否 | 是(loss 步内扰动<噪声+step1 loss 逐位) | — | **-11.65%(5.484→4.845s)** | peak_alloc -4.45GiB / rsv 不变 | 低(opt-in flag, merged 40627682) | 低 | 强(A/B+trace 对照) | **KEEP** |
| Final AdaLNOut 2-row bwd reduce | index_add 623.4ms | 否 | 是(step1 loss 逐位一致 1997.5276; fp32 accum 更接近 fp64 金标) | 286.9→0.97ms(296x isolated) | **-8.41%**(与 compile 组合 -21.72%) | 不变 | 低(merged 563440cf) | 低-中(bf16 原子序噪声放大) | 强 | **KEEP** |
| QKV packed FA API | cat+chunk 链 ~124ms | — | — | 无成熟训练 API(FlashFattn/FA3/FA4/TE 均无可用 packed 训练路径; SDPA 只收分离 q/k/v) | — | — | 高(需自写) | 高 | 强(代码+trace+包审计) | **DROP**(无可接入对象且 ceiling 0.66–0.74% < 3% 门) |
| QK Norm+RoPE training fusion | compile 后残 19.3ms | 否 | 是 | — | abs ceiling 0.40% < 0.5% 门 | — | 2-3 人日自写 Triton bwd | 中 | 强(retro trace 实测) | **DROP**(推理已有 fwd-only kernel; training 价值不足) |
| post-A2A reorder + consumer fusion | preprocess_qkv 恒等恒 0-cost | 否 | 是(两等价性命题成立需满足 dtype/位置条件) | 零收益(重排不动 FLOPs/bytes, 且 preprocess identity) | — | — | 中 | 中 | 强(E 拆账 + J 调研) | **DROP** |
| FA 直接写 consumer-ready layout (D1/D2 cat/clone 精减) | fp32 布局区 135.1ms(+cast 23.9) 但 dtype 路径此轮冻结 | 否 | 是 | ~30-40ms/step 可扣除区间(D2) | 未实测(该测量等待冻结解除) | — | 中 | 低-中 | 中 | P2 / 待 dtype 实验解冻后做 parity A/B |
| FA tile/tile tuning | flash fwd 113.2ms(57.8% TC)/bwd 305.4ms(53.5% TC) | 是 | 是 | 后端对照 cuDNN bwd -32%, mem_effeff bwd +159% → flash 已最优 | — | — | 高(新 kernel) | 中 | 中等(ncu 权限被拒 → stall 数据为推断标注;带宽/占用/资源均为直测) | **DROP**(结构已贴近几何极限; 换 tile 不改 Bwd 1-CTA/SM 上限) |
| QKV-A2A compute/comm overlap | region 46.25ms,通信暴露 5.4ms | 否 | 是(bit-exact) | region -5.4%(vB 并行) | 全step仅 ~0.5-0.6% < 阈值 | 微增 | 高(双 stream+event) | 低 | 中(微测实跑, 数字 contentious 详 H 报告) | **INVESTIGATE**(下一步:NCCL_MAX_CTAS=8 降低竞争税再估) |
| fused AdamW | optimizer 区 70.3ms(G 审计) | 否 | 是 | optimizer 区 14.28→1.67ms CUDA busy(-88%) | **full-step ≤0.58%(噪声内)** | 不变 | 极低(toggle, merged 31a09b5f) | 极低 | 强 | **DROP as perf**(toggle 保留, 不默认开) |
| small-kernel fragmentation (G 桶) | <500us 桶仅 2.8% | — | — | — | — | — | — | — | 强(分桶+关键出处) | **DROP**(非 launch-bound; cast/pointwise 真正体质由 compile 吸收) |
| stream bubble 修复(allocator/FSDP config) | **all-stream idle 14.56%/step(807.8ms)**;nsys 交叉 15.98% | 否 | 是 | — | 未实测;佳上界 ~235-542ms(realistic, 不动 NCCL 拓扑或 sparse) | 不变 | 低(env 级) | 需先验证 expandable_segments=False 的显存截(本档 34.96GiB/40GiB 缺余裕) | 中-强(I 报告,交叉两源) | **INVESTIGATE next**(需在显存余量足够后先 A/B `expandable_segments=False`,必须先看是否仍 OOM) |
| Agent J 检查: Apex/TE native RMSNorm/RoPE | RMSNorm 已是原生单 kernel 融合(37.8ms/step=0.7%) | 否 | 是 | 无成熟替代(Apex/TE 未装;仓内 fused RMSNorm bwd 解体为 10 个 eager kernel 不可行) | — | — | 低 | — | 强 | **DROP** |

## 关键归并说明

1. **KEEP** = regional compile + finalAdaLN two-row。组合 A/B 实测: base 5.484s → finalonly 5.022s → compileonly 4.845s → combo 4.293s(**-21.72%,即 1.277x**),非简单相加(20.07%),交互为正(compile 消除 fp32 边带之后 final-adaln 修复仍然独立奏效)。
2. 本轮对 QK-Norm/RoPE dtype 归一没再做实验,按“人工分析占线”指令冻结。
3. Part F 的 AMX-GEMM 方法映射见 AMX_METHOD_MAPPING.md。
