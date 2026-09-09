# SPDX-License-Identifier: Apache-2.0
"""Specialized two-row backward reduction for the H3 final AdaLayerNormOut.

Profiling evidence (see profiling/minimax_h3/INDEX_ADD_ATTRIBUTION.md):
the two backward ``scale.index_select`` / ``shift.index_select`` of
``MiniMaxH3AdaLayerNormOut`` scatter 18,866 bf16 gradient rows into a 2-row
destination through the generic index_add_ ReduceAdd path and dominate
backward index time (~509 ms of a 5.4 s step on 2xA100).

This module provides ``two_row_index_select`` — forward keeps the exact eager
``index_select`` values; backward replaces the generic scatter with either a
two-stage Triton segmented reduction (per-token-block fp32 partials, then a
tiny second-stage reduction) or a one-hot GEMM fallback. Both accumulate in
fp32 and cast once to the table dtype.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _two_row_stage1(
        dy_ptr, idx_ptr, partial_ptr,
        S, H: tl.constexpr, TS: tl.constexpr, TH: tl.constexpr, B: tl.constexpr,
    ):
        pid = tl.program_id(0)
        hb = tl.program_id(1)
        offs_t = pid * TS + tl.arange(0, TS)
        mask_t = offs_t < S
        offs_h = hb * TH + tl.arange(0, TH)
        mask_h = offs_h < H
        i = tl.load(idx_ptr + offs_t, mask=mask_t, other=-1)
        y = tl.load(dy_ptr + offs_t[:, None] * H + offs_h[None, :],
                    mask=mask_t[:, None] & mask_h[None, :], other=0.0)
        yf = y.to(tl.float32)
        acc0 = tl.sum(tl.where(i[:, None] == 0, yf, 0.0), axis=0)
        acc1 = tl.sum(tl.where(i[:, None] == 1, yf, 0.0), axis=0)
        pbase = partial_ptr + (pid * 2) * (B * TH)
        tl.store(pbase + 0 * (B * TH) + hb * TH + tl.arange(0, TH), acc0, mask=mask_h)
        tl.store(pbase + 1 * (B * TH) + hb * TH + tl.arange(0, TH), acc1, mask=mask_h)

    @triton.jit
    def _two_row_stage2(
        partial_ptr, out_ptr,
        P: tl.constexpr, H: tl.constexpr, TH: tl.constexpr, B: tl.constexpr,
    ):
        hb = tl.program_id(0)
        offs_h = hb * TH + tl.arange(0, TH)
        mask_h = offs_h < H
        acc0 = tl.zeros((TH,), dtype=tl.float32)
        acc1 = tl.zeros((TH,), dtype=tl.float32)
        for p in range(0, P):
            base = partial_ptr + (p * 2) * (B * TH) + hb * TH
            acc0 += tl.load(base + tl.arange(0, TH), mask=mask_h, other=0.0)
            acc1 += tl.load(base + (B * TH) + tl.arange(0, TH), mask=mask_h, other=0.0)
        out_dtype = out_ptr.dtype.element_ty
        tl.store(out_ptr + offs_h, acc0.to(out_dtype), mask=mask_h)
        tl.store(out_ptr + H + offs_h, acc1.to(out_dtype), mask=mask_h)


def two_row_scatter_add_triton(grad_output: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Two-stage Triton segmented reduction: [S,H] + idx in {0,1} -> [2,H]."""
    S, H = grad_output.shape
    device = grad_output.device
    TS, TH = 512, 1024
    P = (S + TS - 1) // TS
    B = (H + TH - 1) // TH
    partials = torch.zeros(P, 2, B * TH, device=device, dtype=torch.float32)
    _two_row_stage1[(P, B)](grad_output.contiguous(), index, partials, S, H, TS, TH, B)
    out = torch.empty(2, H, device=device, dtype=torch.bfloat16)
    _two_row_stage2[(B,)](partials, out, P, H, TH, B)
    return out


def two_row_scatter_add_gemm(grad_output: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """One-hot GEMM reduction: onehot(idx)[S,2]^T fp32 @ dY[S,H] fp32 -> [2,H]."""
    onehot = torch.nn.functional.one_hot(index.to(torch.int64), num_classes=2).to(torch.float32)
    return (onehot.t() @ grad_output.to(torch.float32)).to(torch.bfloat16)


def two_row_scatter_add_backward(
    grad_output: torch.Tensor,
    index: torch.Tensor,
    *,
    table_dtype: torch.dtype,
    prefer_triton: bool = False,
    validate: bool = False,
) -> torch.Tensor:
    if index.numel() == 0:
        return grad_output.new_zeros(2, grad_output.shape[1], dtype=table_dtype)
    if validate:  # per-call min/max forces a host sync; enable only in tests
        if int(index.min().item()) < 0 or int(index.max().item()) > 1:
            raise ValueError("two_row backward requires destination indices in {0, 1}")
    if HAVE_TRITON and grad_output.is_cuda and prefer_triton:
        out = two_row_scatter_add_triton(grad_output, index)
    else:
        out = two_row_scatter_add_gemm(grad_output, index)
    return out.to(table_dtype)


class TwoRowIndexSelect(torch.autograd.Function):
    """Forward exactly like index_select(0); backward uses the 2-row reducer."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(index)
        ctx.table_dtype = x.dtype
        return x.index_select(0, index)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (index,) = ctx.saved_tensors
        grad = two_row_scatter_add_backward(grad_output.contiguous(), index,
                                            table_dtype=ctx.table_dtype)
        return grad, None


def two_row_index_select(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """API wrapper keeping the call shape close to Tensor.index_select."""
    return TwoRowIndexSelect.apply(x, index)
