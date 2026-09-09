# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the nsys cudaProfilerApi step bracket."""

from types import SimpleNamespace

from fastvideo.train.callbacks.nsys_capture import NsysCaptureCallback


def test_bracket_starts_and_stops_once(monkeypatch):
    import fastvideo.train.callbacks.nsys_capture as mod
    calls: list[str] = []

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        class profiler:  # noqa: D106 - test stub
            @staticmethod
            def start() -> None:
                calls.append("start")

            @staticmethod
            def stop() -> None:
                calls.append("stop")

    monkeypatch.setattr(mod, "torch", SimpleNamespace(cuda=_Cuda))
    cb = NsysCaptureCallback(enabled=True, start_after_step=2, stop_step=3)
    method = SimpleNamespace()
    for step in (1, 2, 3):
        cb.on_training_step_end(method, {}, iteration=step)
    assert calls == ["start", "stop"]
    cb.on_train_end(method, iteration=3)
    assert calls == ["start", "stop"]  # idempotent


def test_disabled_never_starts(monkeypatch):
    import fastvideo.train.callbacks.nsys_capture as mod
    calls: list[str] = []

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        class profiler:  # noqa: D106 - test stub
            @staticmethod
            def start() -> None:
                calls.append("start")

            @staticmethod
            def stop() -> None:
                calls.append("stop")

    monkeypatch.setattr(mod, "torch", SimpleNamespace(cuda=_Cuda))
    cb = NsysCaptureCallback(enabled=False)
    for step in (1, 2, 3):
        cb.on_training_step_end(SimpleNamespace(), {}, iteration=step)
    assert calls == []
