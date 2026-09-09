# SPDX-License-Identifier: Apache-2.0
"""Rigorous numerics for the two-row backward reduction.

The honest baseline comparison: eager bf16 index_select-backward scatters and
accumulates in bf16 (lossy over ~9k rows); our reducer accumulates in fp32 and
casts once. Assert the custom path is strictly closer to an fp64 golden answer
than eager and no NaN/Inf appear.
"""

import torch
import pytest

from fastvideo.layers.two_row_index_reduce import (
    two_row_index_select,
    two_row_scatter_add_backward,
    two_row_scatter_add_gemm,
    HAVE_TRITON,
)

S, H = 18866, 5376


def _make_patterned_index(s: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    idx = torch.empty(s, dtype=torch.int64)
    c1, c2 = s // 3, 2 * s // 3
    idx[:c1] = 0
    idx[c1:c2] = 1
    idx[c2:] = torch.randint(0, 2, (s - c2,), generator=gen)
    return idx[torch.randperm(s, generator=gen)]


def _stats(tag: str, got: torch.Tensor, want: torch.Tensor) -> dict:
    diff = (got.to(torch.float64) - want.to(torch.float64)).abs()
    denom = want.to(torch.float64).abs().clamp_min(1e-12)
    return {
        "tag": tag,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float((diff / denom).max()),
        "nan_count": int(torch.isnan(got.to(torch.float64)).sum()),
    }


def _bi16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def test_forward_bitwise_identical():
    torch.manual_seed(0)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    idx = _make_patterned_index(1000)
    assert torch.equal(two_row_index_select(x, idx), x.index_select(0, idx))


def test_gemm_path_beats_eager_bf16_scatter_against_fp64():
    torch.manual_seed(1)
    h, s = 256, 4096
    idx = _make_patterned_index(s)
    dy = _bi16(torch.randn(s, h))
    gold = torch.nn.functional.one_hot(idx, 2).to(torch.float64).t() @ dy.to(torch.float64)

    eager_tab = _bi16(torch.randn(2, h)).requires_grad_(True)
    eager_tab.index_select(0, idx).backward(dy)
    eager_stats = _stats("eager", eager_tab.grad, gold)

    custom = two_row_scatter_add_gemm(dy, idx)
    custom_stats = _stats("custom", custom, gold)
    assert custom_stats["nan_count"] == 0
    assert custom_stats["max_abs_diff"] <= 2.0, custom_stats  # within ~1 bf16 ulp at |sum|~70
    assert custom_stats["mean_abs_diff"] <= eager_stats["mean_abs_diff"]


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA+Triton required")
def test_triton_equals_gemm_bitwise_and_beats_eager():
    torch.manual_seed(2)
    idx = _make_patterned_index(S).cuda()
    dy = _bi16(torch.randn(S, H)).cuda()
    gold = torch.nn.functional.one_hot(idx, 2).to(torch.float64).t() @ dy.to(torch.float64)

    gemm = two_row_scatter_add_backward(dy, idx, table_dtype=torch.bfloat16, prefer_triton=False)
    tri = two_row_scatter_add_backward(dy, idx, table_dtype=torch.bfloat16, prefer_triton=True)
    # fp32 partials in different block orders round a few elements by 1 bf16 ulp.
    diff = (tri.float() - gemm.float()).abs()
    assert float(diff.max()) <= 2.0 and float(diff.mean()) < 0.5

    eager_tab = _bi16(torch.randn(2, H)).cuda().requires_grad_(True)
    eager_tab.index_select(0, idx).backward(dy)
    eager_stats = _stats("eager", eager_tab.grad, gold)
    tri_stats = _stats("triton", tri, gold)
    assert tri_stats["nan_count"] == 0
    assert tri_stats["mean_abs_diff"] <= eager_stats["mean_abs_diff"]
    assert tri_stats["max_abs_diff"] <= 2.0, tri_stats


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_autograd_forward_identical_and_backward_closer_to_gold():
    torch.manual_seed(3)
    idx = _make_patterned_index(S).cuda()
    x0 = _bi16(torch.randn(2, H)).cuda()
    dy = _bi16(torch.randn(S, H)).cuda()
    gold = torch.nn.functional.one_hot(idx, 2).to(torch.float64).t() @ dy.to(torch.float64)

    xe = x0.clone().requires_grad_(True)
    out_e = xe.index_select(0, idx)
    out_e.backward(dy)
    eager_stats = _stats("eager", xe.grad, gold)

    xc = x0.clone().requires_grad_(True)
    out_c = two_row_index_select(xc, idx)
    assert torch.equal(out_c, out_e)
    out_c.backward(dy)
    custom_stats = _stats("custom", xc.grad, gold)
    assert custom_stats["nan_count"] == 0
    assert custom_stats["max_abs_diff"] <= 2.0
    assert custom_stats["mean_abs_diff"] <= eager_stats["mean_abs_diff"]
