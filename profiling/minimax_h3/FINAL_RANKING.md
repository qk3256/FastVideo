# MiniMax H3 优化最终排名(N=4, 2×A100, SP=2, baseline a06167d3 / 038902a3)

口径: 每条 status 都以真实 optimizer-step 结果为剪裁; 只写孤立 benchmark 或静态分析的一律标 INVESTIGATE 或 DROP.

## 主表

| candidate | N=4 baseline 成本 | 机制 | 碰核心 compute? | is isolated | 孤 speedup | N=4 全 step speedup | 峰值显存 | 数值风险 | 实现成本 | 证据强度 | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| regional torch.compile(Transformer block 区域编译) | cast/pointwise/bubble soup ~460ms + torch dispatch + partial bubble | Inductor 自动融合 norm/pointwise/rope/cat/cast | 否 | 训练级 A/B | — | **-11.65%(5.484→4.845s)** | peak_alloc -4.45GiB, rsv 不变 35.00 | 低 | 低(opt-in `compile_blocks: true`) | 强(multi-A/B+trace) | **KEEP** |
| Final AdaLNOut 2-row bwd reduce(one-hot GEMM, fp32 accum) | bf16 ReduceAdd atomics 508.8ms | fp32 one-hot GEMM @ cuBLAS | 否 | 是 | 286.9→0.97ms(296x) | **-8.41%(5.484→5.022s)** | 不变 34.96 | 中低(原子序噪声, step1 逐位一致) | 低(opt-in `final_adaln_two_row_backward: true`) | 强(fp64 金标+A/B) | **KEEP** |
| QKV packed FA API | 链 ~124ms | 无成熟训练 API | 是 | — | — | — | — | — | 高(自写) | 强(包+源码+trace 穷举) | **DROP**(天花板 0.66–0.74%<3% 门) |
| QKNorm+partial RoPE 训练融合 | compile 后残 19.3ms | TE 部类 RoPE 不含 H3 partial 3-axis;Apex 未装;推理专用 Triton 禁 training | 否 | — | abs 上限 0.40%<0.5% 门 | — | — | 中 | 2–3人日自写 bwd | 强(trace 取证) | **DROP** |
| FA H3-specific tile/config tuning | flash fwd 113.2ms(57.8% TC)/bwd 305.4ms(53.5% TC) | flash 已在 FA2 A100 文献带内 | 是 | cuDNN bwd -32%、mem_eff +159% 都更差 | — | — | — | ncu 权限被群; 其余直测 | 高(写 kernel 违规) | 中强 | **DROP**(已贴近几何极限) |
| fused AdamW | optimizer 区 70.3ms | torch.optim.AdamW(fused=True) | 否 | optimizer 区 14.28→1.67ms(-88% launch) | ≤0.58%(噪声内) | 不变 34.96 | 极低 | 极低(opt-in `optimizer.fused: true`,已提交 31a09b5f) | 强 | **DROP as perf**;开关本身保留 |
| bubble/sync cleanup | all-stream idle 807.8ms=14.6%/step | FSDP/allocator/通信暴露为主 | 否 | — | 未实测 | 现实上限 ~235-542ms(FSDP+alloc 不动 NCCL) | — | 需先验 expandable_segments=False 的 OOM 余裕 | 中(env 级) | 中-强(I 报告) | **INVESTIGATE next**(想再挖的必走 expandable_segments=False A/B) |
| QKV-A2A compute/comm overlap | QKV region 46.25ms | NCCL×GEMM 竞争税 | 否 | vB region -5.4% | 仅 ~0.5-0.6% step | — | 高(双 stream) | 低 | 高 | 中(H 微测) | **INVESTIGATE**(NCCL_MAX_CTAS=8 再验) |
| producer→consumer 直接写 layout(D1/D2 cat/clone) | fp32 布局区 ~135ms | 归一化 dtype 路径已被冻结(本轮不动) | 否 | — | 未实测 | — | 中 | 低 | 中 | 中 | **INVESTIGATE**(等 dtype 冻结解冻) |
| QKV wide GEMM | 3×独立 GEMM | cuBLAS 已最优, kernel 名同族 | 是 | fwd+bwd 1.025x | ~0.11% | — | — | 低 | 低 | 强(bench) | **DROP** |
| SwiGLU 手写 training fusion | eager 12.0ms/层·step | compile 已自融 ~93% | 否 | eager→fused 2.7x | <0.5% step | — | 中 | 中 | 中 | 强 | **DROP(compile 吸收)** |
| backend 换 FA | flash 已最接近 | — | — | — | — | — | — | — | — | 强 | **DROP**(无可用替代) |
| selective activation ckpt / sparse / FA 重写 / 50 层 | — | — | — | — | — | — | — | — | — | — | **DROP**(本轮明确禁止) |

## Top 3 deliverables(KEEP)

1. **Regional torch.compile**——`branch minimax-h3-opt-regional-compile @ 40627682`, flag 默认关; 数证: 0 graph break, 1 stride-driven recompile/rank, 编译 +5.99s 热 cache, 兼容 FSDP2/full AC/SDPA。
2. **Final AdaLNOut 2-row backward reduction**——`branch minimax-h3-opt-final-adaln-reduce @ 563440cf`, 默认关; fp64 金标优著更稳; 不增量 key count; 与 compile 组合 -21.72%。
3. **四-combo A/B 基线(off/final-adaln/compile/combo = 5.484/5.022/4.845/4.293 s)** — 最终合并产出位置。

## Top 3 follow-up research(下一代)

1. 分配器/CPU 分配同步(bubble 台账):`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` 或手动活动空间隔离,先在确认 34.96GiB<40GiB 的余量下验证不 OOM。
2. `QK-Norm 端 bf16 归一`(本轮冻结,需人工同业结算后 parity 验证)。
3. QKV-A2A overlap v2(NCCL_MAX_CTAS=8 竞争税压制)。

## DROP 列表

QKV packed FA(无可调用训练 API)、QKNorm+partial RoPE 融合(下限远低于门)、FA tile tuning(已贴极限)、fused AdamW(数值/性能收益不到 1%)、QKV wide GEMM(0.11%)、SwiGLU 手写(compile 自融)、backend switch(无替)、选择性 AC(本轮禁)、稀疏 KV(本轮禁)、自写 FA kernel(禁)、Main GEMM 广义调优(已 falsified)。
## 本轮(最终收尾)新增

| candidate | N=4 结果 | 原因 | status |
|---|---|---|---|
| expandable_segments=False(allocator bubble 进攻) | step −16.8%(变差), bubble 15.05%→19.07% | VMM churn 是 heparable host 时间,cudaFree/cudaMalloc 锁的是设备本体 | **DROP** |
| QKV-A2A overlap v2 + NCCL_MAX_CTAS=8 | region 反 +1.76ms,CTA 节流付带宽从 176.8→71.3GB/s | 竞争税主项是共享带宽,不是 SM 抢占 | **DROP** |
| Model-Specific Specialization 挖掘 (Part A–D) | 未发现新 ≥0.7% 项 | 唯一 material 低基数 pathology 就是 Final AdaLNOut 已解决；近项 A2A 直写只 0.68% 未达门,且带 dtype 冻结 | **DROP** |

结论强化:Final AdaLNOut 的 low-cardinality specialization 是整张图里**第一个、也是唯一一个**真实 material 的强项。COVERAGE_MATRIX 的「无未处理 exact candidate」在本轮三处全部复读确凿。
