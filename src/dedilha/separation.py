"""Separação de instrumentos (stems) com o Demucs em ONNX — sem PyTorch.

Modelo `htdemucs_6s`: bateria, baixo, voz, outros, **violão/guitarra** e **piano**.
Aviso do próprio projeto Demucs: os stems de guitarra e piano são "experimentais"
(o de piano tem mais vazamento/artefatos). Por isso o app sempre analisa também a mix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np

from .audio import ANALYSIS_SR, STEM_SR, to_mono_sr

ALL_STEMS = ("vocals", "drums", "bass", "other", "guitar", "piano")


class SeparationError(RuntimeError):
    pass


class Separator(Protocol):
    name: str

    def separate(self, wav_path: Path, stems: tuple[str, ...],
                 progress: Callable[[float, str], None]) -> dict[str, np.ndarray]:
        """Devolve {nome_do_stem: áudio mono float32 em ANALYSIS_SR}."""
        ...


class DemucsOnnxSeparator:
    """Wrapper do pacote `demucs-onnx` (modelo htdemucs_6s)."""

    name = "demucs-onnx/htdemucs_6s"
    model = "htdemucs_6s"

    def __init__(self, save_dir: Optional[Path] = None, providers: str = "auto"):
        self.save_dir = save_dir
        self.providers = providers

    @staticmethod
    def is_available() -> tuple[bool, str]:
        try:
            import demucs_onnx  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError as exc:
            return False, f"pacote ausente ({exc.name}); rode: python install.py"
        return True, ""

    @staticmethod
    def prefetch() -> None:
        """Baixa o modelo (~100 MB) antecipadamente."""
        import demucs_onnx

        demucs_onnx.prewarm([DemucsOnnxSeparator.model])

    def separate(self, wav_path: Path, stems: tuple[str, ...],
                 progress: Callable[[float, str], None] = lambda f, m: None
                 ) -> dict[str, np.ndarray]:
        ok, why = self.is_available()
        if not ok:
            raise SeparationError(why)
        import demucs_onnx

        wanted = tuple(s for s in stems if s in ALL_STEMS)
        progress(0.02, "Carregando o modelo de separação (na 1ª vez baixa ~100 MB)…")
        try:
            out = demucs_onnx.separate(
                wav_path, output_dir=None, model=self.model, stems=wanted,
                providers=self.providers, progress=False, verbose=False)
        except Exception as exc:
            msg = str(exc)
            if any(k in msg.lower() for k in ("connection", "resolve", "timed out", "huggingface",
                                               "offline", "network", "ssl")):
                raise SeparationError(
                    "Não consegui baixar o modelo de separação (precisa de internet na primeira "
                    f"vez; acesso ao huggingface.co). Detalhe: {msg[:160]}") from exc
            raise SeparationError(f"Falha na separação de instrumentos: {msg[:200]}") from exc

        progress(0.9, "Preparando os stems…")
        result: dict[str, np.ndarray] = {}
        for name, arr in out.items():
            if name not in wanted:
                continue
            arr = np.asarray(arr, dtype=np.float32)
            if self.save_dir is not None:
                self._save(name, arr)
            result[name] = to_mono_sr(arr, STEM_SR, ANALYSIS_SR)
        if not result:
            raise SeparationError("O modelo não devolveu nenhum stem.")
        return result

    def _save(self, name: str, arr: np.ndarray) -> None:
        import soundfile as sf

        self.save_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        data = arr.T if arr.ndim == 2 and arr.shape[0] <= 8 else arr
        sf.write(str(self.save_dir / f"{name}.wav"), data, STEM_SR, subtype="PCM_16")
