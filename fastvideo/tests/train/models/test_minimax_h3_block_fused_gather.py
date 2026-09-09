# SPDX-License-Identifier: Apache-2.0
"""End-to-end equivalence: full transformer block with fused AdaLN gather.

The fused path gathers the joint modulation table once (tables are already
row-expanded to the sequence axis) and turns use-site gathers into no-ops.
This test compares complete block outputs, not just the projection helper —
a regression here changed real training losses, so the whole block path is
the contract.
"""

import os

# Required by the ``distributed_setup`` fixture pulled from fastvideo/tests/
# conftest.py. Set before any fastvideo import.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29771")

import torch
import pytest

from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3TransformerBlock
from fastvideo.pipelines import ForwardBatch
from fastvideo.platforms import AttentionBackendEnum


def _build_block(seed: int = 0) -> MiniMaxH3TransformerBlock:
    torch.manual_seed(seed)
    block = MiniMaxH3TransformerBlock(
        hidden_size=32,
        num_attention_heads=2,
        attention_head_dim=16,
        ffn_dim=64,
        time_embed_dim=16,
        norm_eps=1e-6,
        qk_norm_eps=1e-6,
        supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA,),
        quant_config=None,
        prefix="t",
    )
    with torch.no_grad():
        # FastVideo linears leave weights uninitialized; seed explicit values.
        for p in block.parameters():
            p.normal_(0.0, 0.02)
    return block


@pytest.mark.usefixtures("distributed_setup")
def test_block_fused_gather_matches_reference_forward_and_backward():
    block = _build_block()
    seq, temb_rows = 24, 3
    valid_rows = temb_rows * MINIMAX_H3_MODALITY_NUM
    adaln_indices = torch.randint(0, valid_rows, (seq,))

    x0 = torch.randn(1, seq, 32, dtype=torch.float32)
    temb0 = torch.randn(temb_rows, 16, dtype=torch.float32)

    def run_once(fuse: bool) -> tuple[torch.Tensor, float]:
        block.fuse_adaln_gather = fuse
        x = x0.clone().requires_grad_(True)
        with set_forward_context(
                current_timestep=torch.tensor([0.5]),
                attn_metadata=None,
                forward_batch=ForwardBatch(data_type="video", fps=24.0),
        ):
            out = block(x, temb0.clone(), adaln_indices, None, seq)
            (out.square().sum()).backward()
        return out.detach(), float(x.grad.abs().sum())

    ref_out, ref_grad = run_once(False)
    fused_out, fused_grad = run_once(True)
    torch.testing.assert_close(fused_out, ref_out)
    assert abs(fused_grad - ref_grad) <= 1e-4 * abs(ref_grad)
