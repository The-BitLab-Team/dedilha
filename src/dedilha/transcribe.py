"""Transcrição polifônica de notas com o basic-pitch (Spotify), rodando em ONNX."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .audio import ANALYSIS_SR
from .models import NoteEvent
from .music import midi_to_name

MIDI_LOWEST = 21   # A0: primeira linha do posteriorgrama do basic-pitch (88 teclas de piano)


class DependencyError(RuntimeError):
    pass


@dataclass
class Transcription:
    notes: list[NoteEvent]
    posteriorgram: np.ndarray    # (frames, 88): probabilidade de cada nota estar soando
    onsets: np.ndarray           # (frames, 88)
    frame_times: np.ndarray      # (frames,) em segundos


class Transcriber:
    """Carrega o modelo uma vez e transcreve vários sinais."""

    def __init__(self, min_note_seconds: float = 0.10, onset_threshold: float = 0.5,
                 frame_threshold: float = 0.3):
        self.min_note_seconds = min_note_seconds
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        logging.disable(logging.WARNING)  # o basic-pitch reclama de runtimes que não usamos
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch import constants as bpc
            from basic_pitch.inference import Model, run_inference
            from basic_pitch.note_creation import model_frames_to_time, model_output_to_notes
        except ImportError as exc:
            raise DependencyError(
                "O pacote 'basic-pitch' não está instalado. Rode: python install.py "
                "(ele instala o basic-pitch sem o TensorFlow)."
            ) from exc
        finally:
            logging.disable(logging.NOTSET)
        self._model = Model(ICASSP_2022_MODEL_PATH)
        self._run_inference = run_inference
        self._to_notes = model_output_to_notes
        self._frame_times = model_frames_to_time
        self._fps = bpc.AUDIO_SAMPLE_RATE / bpc.FFT_HOP
        self._loaded = True

    def transcribe(self, y: np.ndarray, sr: int = ANALYSIS_SR, flats: bool = False
                   ) -> Transcription:
        if sr != ANALYSIS_SR:
            raise ValueError(f"o sinal precisa estar em {ANALYSIS_SR} Hz")
        self._load()
        import soundfile as sf

        if y.size < ANALYSIS_SR // 2:   # menos de meio segundo: nada a transcrever
            empty = np.zeros((0, 88), dtype=np.float32)
            return Transcription([], empty, empty, np.zeros(0))

        with tempfile.TemporaryDirectory(prefix="dedilha_bp_") as tmp:
            wav = Path(tmp) / "sig.wav"
            sf.write(str(wav), np.clip(y, -1.0, 1.0), sr, subtype="FLOAT")
            out = self._run_inference(wav, self._model)

        post = np.asarray(out["note"], dtype=np.float32)
        onsets = np.asarray(out["onset"], dtype=np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):    # áudio silencioso gera 0/0 no basic-pitch
            _, events = self._to_notes(
                out, onset_thresh=self.onset_threshold, frame_thresh=self.frame_threshold,
                min_note_len=max(1, int(round(self.min_note_seconds * self._fps))),
                include_pitch_bends=False, melodia_trick=True)
        notes = [NoteEvent(float(s), float(e), int(p), midi_to_name(p, flats), float(a))
                 for s, e, p, a, _ in sorted(events, key=lambda x: (x[0], x[2]))]
        return Transcription(notes, post, onsets, np.asarray(self._frame_times(post.shape[0])))
