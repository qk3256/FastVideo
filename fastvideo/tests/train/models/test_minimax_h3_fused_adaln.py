# SPDX-License-Identifier: Apache-2.0
"""Numerical equivalence test for the fused AdaLN gather training path.

The fused path gathers the joint modulation table once and then chunks;
the reference path gathers each of the six tables individually. Values must
be bitwise identical: same rows, same data order, same casts.
"""

import torch
import torch.nn.functional as F

from fastvideo.models.dits.minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3AdaLayerNormModulation,
)


def _init_modulation(hidden: int, time_embed_dim: int) -> MiniMaxH3AdaLayerNormModulation:
    torch.manual_seed(0)
    # FastVideo linears leave weights uninitialized for checkpoint loading;
    # tests must seed explicit values so NaN-free comparisons are possible.
    mod = MiniMaxH3AdaLayerNormModulation(
        time_embed_dim=time_embed_dim, hidden_size=hidden, prefix="t")
    with torch.no_grad():
        for p in mod.parameters():
            p.normal_(0.0, 0.02)
    return mod


def test_fused_gather_matches_per_table_gather():
    hidden = 8
    mod = _init_modulation(hidden, 16)
    temb = torch.randn(2, 16, dtype=torch.float32)  # 2 batch rows x modalities valid rows
    valid_rows = 2 * MINIMAX_H3_MODALITY_NUM
    adaln_indices = torch.randint(0, valid_rows, (10,))

    with torch.no_grad():
        reference = [t.to(torch.float32).index_select(0, adaln_indices) for t in mod(temb)]
        raw, _ = mod.linear(F.silu(temb).to(mod.linear.weight.dtype))
        gathered = raw.view(-1, 6 * hidden).index_select(0, adaln_indices).to(torch.float32)
        fused = list(gathered.chunk(6, dim=-1))

    assert len(reference) == 6 and len(fused) == 6
    for ref, new in zip(reference, fused):
        torch.testing.assert_close(ref, new)


def test_fused_gather_chunk_views_match_chunk_layout():
    """Chunk order after gather must reproduce shift/scale/gate semantics."""
    hidden = 4
    mod = _init_modulation(hidden, 8)
    temb = torch.randn(MINIMAX_H3_MODALITY_NUM, 8, dtype=torch.float32)
    valid_rows = MINIMAX_H3_MODALITY_NUM * MINIMAX_H3_MODALITY_NUM
    idx = torch.arange(valid_rows)
    with torch.no_grad():
        ref = [t.index_select(0, idx) for t in mod(temb)]
        raw, _ = mod.linear(F.silu(temb).to(mod.linear.weight.dtype))
        new = list(raw.view(-1, 6 * hidden).index_select(0, idx).chunk(6, dim=-1))
    assert all(torch.equal(a, b) for a, b in zip(ref, new))
