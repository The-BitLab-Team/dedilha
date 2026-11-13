"""Utilidades pequenas: progresso e cronometragem."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable

# stage (id curto), fração 0..1 do processo inteiro, mensagem para o usuário
ProgressFn = Callable[[str, float, str], None]


def noop_progress(stage: str, fraction: float, message: str) -> None:  # pragma: no cover
    pass


class Window:
    """Mapeia progresso local (0..1) para uma janela [lo, hi] do progresso global."""

    def __init__(self, progress: ProgressFn, stage: str, lo: float, hi: float):
        self._p, self._stage, self._lo, self._hi = progress, stage, lo, hi

    def __call__(self, local: float, message: str) -> None:
        local = min(max(local, 0.0), 1.0)
        self._p(self._stage, self._lo + (self._hi - self._lo) * local, message)


@contextmanager
def timed(store: dict[str, float], name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[name] = store.get(name, 0.0) + (time.perf_counter() - t0)
