# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the explicit tracker opt-out token and W&B auto-enable."""

from fastvideo.train.utils.tracking import _resolve_trackers
from fastvideo.training.trackers import Trackers


def test_none_token_disables_trackers_and_auto_enable():
    assert _resolve_trackers(["none"], "fastvideo_minimax_h3") == []
    assert _resolve_trackers(["NONE", "off", "disabled"], "proj") == []


def test_empty_list_with_project_auto_enables_wandb():
    assert _resolve_trackers([], "fastvideo") == [Trackers.WANDB.value]


def test_empty_list_without_project_stays_disabled():
    assert _resolve_trackers([], "") == []


def test_real_trackers_pass_through():
    assert _resolve_trackers(["wandb"], "proj") == ["wandb"]
    assert _resolve_trackers(["none", "wandb"], "proj") == ["wandb"]
