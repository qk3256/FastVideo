# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the explicit tracker opt-out token."""

from fastvideo.train.utils.tracking import _coerce_trackers


def test_none_token_disables_trackers():
    assert _coerce_trackers(["none"]) == []
    assert _coerce_trackers(["NONE", "off", "disabled"]) == []


def test_real_trackers_pass_through():
    assert _coerce_trackers([]) == []
    assert _coerce_trackers(["wandb"]) == ["wandb"]
    assert _coerce_trackers(["none", "wandb"]) == ["wandb"]
