# Optimization Coverage Matrix — MiniMax H3 N=4 SFT

Audit standard: 只有被“profile 证明存在”且经过成熟库/compile/显式 DROP 三条至少一条处理的 candidate 才算闭环。完成度按 full-step cost 排序。

| idea | methodology category | profiled? | isolated tested? | E2E tested? | subsumed by compile? | blocked by mature backend? | remaining gap | status |
|---|---|---|---|---|---|---|---|---|
| regional torch.compile (blocks) | Inductor fusion | 是 | 是 | 是 (N=4 3-step,-11.65%) | — | — | 50 层外推需硬件 | **KEEP** |
| Final AdaLNOut 2-row bwd reduce | low-cardinality specialization (18866→2) | 是 | 是 (296×) | 是 (N=4 3-step,-8.41%) | 部分（compile 吃 fp32 侧） | — | 冻结 dtype 路径后再 parity | **KEEP** |
| QK Norm+partial RoPE training fusion | inductor pointwise fusion + induction GEMM overlap | 是 | — | 部分（compile 吸收） | 是（残 19.3ms/4.83s=0.4%) | 是 | Apex/TE 无 H3 partial-3-axis | **DROP**(低于 0.5% 门） |
| SwiGLU fusion | inductor pointwise | 是 | 是 | 部分（compile 吸收 ~93%) | 是 | 是 | — | **DROP** |
| QKV packed FA API | 无成熟 packed training 接口 | 是 | 候选不可执行 | — | — | 是 | flash-attn 未装，SDPA 无 packed | **DROP** |
| FA backend switch | FA2 已最佳 backend | 是 | 是 (cuDNN bwd -32%, mem_eff +159%) | — | — | 是 | FA3/FA4 sm90 gated | **DROP** |
| FA tile/config tuning | 资源占用已近几何极限 (fwd 57.8% TC, bwd 53.5% TC) | 是 | 是 | — | — | 是 | — | **DROP** |
| QKV wide GEMM | cuBLAS 已到峰值 | 是 | 是 (1.025×) | — | — | 是 | — | **DROP** |
| QKV-A2A overlap | compute/comm overlap,资源节流 | 是 | 是 (vB region -5.4%,step ~0.5-0.6%) | 是 (NCCL_MAX_CTAS=8 v2) | 部分（compile 消 fp32 链） | 是 | 资源竞争税未压下，低于门槛 | **DROP** |
| producer→consumer direct layout (fp32 attn 区归一 bf16) | dtype 归一 | 是 | 是（model-specific specialization audit） | — | 部分（compile 消 cast soup) | 是 | 无 material exact candidate | **DROP** |
| all-stream bubble / allocator churn | expandable_segments / FSDP 主机暴露 | 是 (14.56%/step,NSys 交叉) | 是 (`expandable_segments:False`) | 是 (-16.8% regression) | 部分（compile 消 183ms FSDP/183ms) | 是 | 默认 expandable segments 保留 | **DROP** |
| fused AdamW | PyTorch fused Adam | 是 | 是 (-88% optimizer CUDA) | 是 (4-step×2,+≤0.58% step) | — | — | 数值=foreach | **DROP as perf / toggle 保留** |
| Apex/TE RMSNorm 替换 | 已原生 fused | 是 | — | — | — | 是（未装，bwd 反向组装更差） | — | **DROP** |
| Final-AdaLN gather fusion（被 rejected 的旧手路） | 6 gather→1 gather | 是 | 是 | 是 (-34.9%,rejected) | — | — | 报告留档 | **DROP**(A/B 已证负) |
| bf16 autocast 边界 in-place（第一个 rejected fp32-SGEMM 路径） | autocast | 是 | — | — | 是 | — | 报告留档 | **DROP as artifact** |
| selective activation ckpt | — | — | — | — | — | — | 本轮禁 | **DROP（按用户本轮规则排除）** |
| sparse/VSA/approx | — | — | — | — | — | — | 本轮禁 | **DROP（本轮禁）** |
| regional compile + Final AdaLN combination | 两个 KEEP 叠加 | 是 | 是 | 是 (N=4 fresh A/B,-22.27%) | — | — | 仅 N=4 实测 | **KEEP (integrated)** |

## 结论

```
NO MATERIAL UNPROFILED EXACT CANDIDATE FOUND
```

所有 baseline cost ≥0.5% 的项都已：(a) 被 compile subsume，(b) 被 mature backend 证明不可替换，(c) 被 E2E 证实负收益，或 (d) 被本轮显式排除。

值得保留且已 KEEP 的只有两条；剩余全部归类为「结构成本」或「已被消化」。
