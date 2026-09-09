# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

from fastvideo.distributed import get_world_group
from fastvideo.training.trackers import (
    initialize_trackers,
    Trackers,
)

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        CheckpointConfig,
        TrackerConfig,
    )

# Explicit opt-out tokens for ``trackers:`` in run YAML.  ``build_tracker``
# auto-enables W&B when the tracker list is empty and a project name is set,
# so fully-offline runs need an explicit, non-empty way to say "no tracker".
_DISABLED_TRACKER_TOKENS = {"none", "off", "disabled"}


def _coerce_trackers(trackers: list[str]) -> list[str]:
    """Drop explicit disable tokens; leave real tracker names untouched."""
    return [t for t in trackers if str(t).lower() not in _DISABLED_TRACKER_TOKENS]


def build_tracker(
    tracker_config: TrackerConfig,
    checkpoint_config: CheckpointConfig,
    *,
    config: dict[str, Any] | None,
) -> Any:
    """Build a tracker instance for a distillation run."""

    world_group = get_world_group()

    trackers = _coerce_trackers(list(tracker_config.trackers))
    if not trackers and str(tracker_config.project_name):
        trackers.append(Trackers.WANDB.value)
    if world_group.rank != 0:
        trackers = []

    tracker_log_dir = (checkpoint_config.output_dir or os.getcwd())
    if trackers:
        tracker_log_dir = os.path.join(tracker_log_dir, "tracker")

    tracker_config_dict = config if trackers else None
    tracker_entity = tracker_config.entity or None
    tracker_run_name = tracker_config.run_name or None
    project = (tracker_config.project_name or "fastvideo")

    return initialize_trackers(
        trackers,
        experiment_name=project,
        config=tracker_config_dict,
        log_dir=tracker_log_dir,
        entity=tracker_entity,
        run_name=tracker_run_name,
    )
