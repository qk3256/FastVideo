# SPDX-License-Identifier: Apache-2.0
"""cudaProfilerApi step bracket for Nsight Systems captures (P4).

With ``nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop``
the profiler starts paused; this callback calls ``cudaProfilerStart`` at the
end of ``start_after_step`` and ``cudaProfilerStop`` at the end of
``stop_step`` so exactly one stable step is captured. Disabled by default and
harmless without CUDA or without nsys attached.
"""

from __future__ import annotations

import torch

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback

logger = init_logger(__name__)


class NsysCaptureCallback(Callback):
    """Bracket one training step between cudaProfilerStart and Stop."""

    def __init__(self, *, enabled: bool = False, start_after_step: int = 2, stop_step: int = 3) -> None:
        self.enabled = bool(enabled)
        self.start_after_step = int(start_after_step)
        self.stop_step = int(stop_step)
        self.capturing = False

    def _start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.profiler.start()
            self.capturing = True
            logger.info("nsys capture started (cudaProfilerApi)")

    def _stop(self) -> None:
        if self.capturing and torch.cuda.is_available():
            torch.cuda.profiler.stop()
            self.capturing = False
            logger.info("nsys capture stopped (cudaProfilerApi)")

    def on_training_step_end(self, method, loss_dict, iteration: int = 0) -> None:
        if not self.enabled:
            return
        if iteration == self.start_after_step and not self.capturing:
            self._start()
        elif iteration == self.stop_step and self.capturing:
            self._stop()

    def on_train_end(self, method, iteration: int = 0) -> None:
        self._stop()
