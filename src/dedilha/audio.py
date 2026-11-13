"""Entrada/saída de áudio: ffmpeg, conversão para WAV, carregamento e afinação."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

# Formatos aceitos como arquivo de entrada (o ffmpeg decodifica todos; vídeos têm o áudio extraído).
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".wma", ".aif", ".aiff",
    ".webm", ".weba", ".mka", ".amr", ".ac3", ".mp4", ".mkv", ".mov", ".avi",
}

ANALYSIS_SR = 22050     # taxa usada pelo basic-pitch
STEM_SR = 44100         # taxa nativa do Demucs


class AudioError(RuntimeError):
    """Erro amigável ao usuário ao ler/converter áudio."""


def find_ffmpeg() -> str:
    """ffmpeg do PATH; se não houver, usa o binário embutido do imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depende do ambiente
        raise AudioError(
            "ffmpeg não encontrado. Instale o ffmpeg (https://ffmpeg.org) ou rode "
            "'pip install imageio-ffmpeg'."
        ) from exc


def probe_duration(path: Path) -> float:
    """Duração em segundos, lida da saída do ffmpeg (não precisa de ffprobe)."""
    proc = subprocess.run(
        [find_ffmpeg(), "-nostdin", "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, errors="replace",
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not m:
        raise AudioError(f"Não consegui ler o arquivo de áudio: {path.name}")
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + s


def to_wav(src: Path, dst: Path, sr: int = STEM_SR, channels: int = 2) -> Path:
    """Converte qualquer áudio/vídeo para WAV PCM 16 bits (estéreo, 44,1 kHz por padrão)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_ffmpeg(), "-nostdin", "-y", "-v", "error", "-i", str(src),
        "-vn", "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_s16le", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size < 1000:
        msg = (proc.stderr or "").strip().splitlines()[-1:] or ["erro desconhecido"]
        raise AudioError(f"O ffmpeg não conseguiu converter '{src.name}': {msg[0]}")
    return dst


def load_mono(path: Path, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Carrega WAV, mistura para mono e reamostra para `sr`. Retorna float32 em [-1, 1]."""
    import soundfile as sf

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    return to_mono_sr(data.T, file_sr, sr)


def to_mono_sr(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Aceita (canais, amostras) ou (amostras,), devolve mono float32 reamostrado."""
    import librosa

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        # eixo dos canais é o menor (<= 8); aceita (ch, n) ou (n, ch)
        ch_axis = 0 if arr.shape[0] <= 8 else 1
        arr = arr.mean(axis=ch_axis)
    if src_sr != dst_sr:
        arr = librosa.resample(arr, orig_sr=src_sr, target_sr=dst_sr)
    return np.ascontiguousarray(arr, dtype=np.float32)


# --------------------------------------------------------------------------- afinação


def estimate_tuning(y: np.ndarray, sr: int) -> float:
    """Desvio de afinação em semitons (-0.5..0.5) em relação a A4 = 440 Hz."""
    import librosa

    try:
        if np.max(np.abs(y)) < 1e-4:
            return 0.0
        return float(librosa.estimate_tuning(y=y, sr=sr, resolution=0.01))
    except Exception:
        return 0.0


def retune(y: np.ndarray, sr: int, tuning: float) -> tuple[np.ndarray, float]:
    """Corrige o tom em `tuning` semitons para cair na afinação padrão (A=440).

    Reinterpreta o sinal numa taxa de amostragem ligeiramente diferente. Isso também muda a
    duração; devolve (sinal, fator_de_tempo) onde  tempo_novo = tempo_original * fator.
    Para voltar aos tempos originais: tempo_original = tempo_novo / fator.
    """
    import librosa

    factor = 2.0 ** (tuning / 12.0)
    out = librosa.resample(y, orig_sr=sr, target_sr=sr * factor)
    return np.ascontiguousarray(out, dtype=np.float32), factor
