"""Notas cantadas: rastreia a altura (pYIN) e converte em notas discretas (G5, F3, A4...).

pYIN é monofônico: precisa de uma voz isolada. Por isso a voz vem do stem `vocals` do Demucs.
Notação científica: C4 = dó central (MIDI 60), A4 = 440 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .audio import ANALYSIS_SR
from .models import NoteEvent
from .music import hz_to_midi, midi_to_name

FMIN_HZ = 65.41      # C2
FMAX_HZ = 1046.5     # C6
HOP = 441            # 20 ms a 22,05 kHz
FRAME = 2048


@dataclass
class VocalAnalysis:
    notes: list[NoteEvent]
    range_low: Optional[str]
    range_high: Optional[str]


def track_pitch(y: np.ndarray, sr: int = ANALYSIS_SR
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna (tempos, midi_float, energia_rms) por quadro; midi é NaN onde não há voz."""
    import librosa

    with np.errstate(invalid="ignore", divide="ignore"):    # NaN esperado onde não há voz
        f0, voiced, _ = librosa.pyin(y, fmin=FMIN_HZ, fmax=FMAX_HZ, sr=sr, frame_length=FRAME,
                                     hop_length=HOP, fill_na=np.nan)
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0, voiced, times, rms = f0[:n], voiced[:n], times[:n], rms[:n]
    # descarta quadros "vozeados" muito fracos (vazamento de outros instrumentos)
    floor = 0.06 * np.percentile(rms, 95) if n else 0.0
    midi = np.full(n, np.nan)
    ok = voiced & (rms >= floor) & np.isfinite(f0)
    midi[ok] = hz_to_midi(f0[ok])
    return times, midi, rms


def _fill_short_gaps(midi: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Preenche lacunas curtas (<= max_gap quadros) se a altura dos dois lados for parecida."""
    out = midi.copy()
    n = len(out)
    i = 0
    while i < n:
        if np.isnan(out[i]):
            j = i
            while j < n and np.isnan(out[j]):
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap and abs(out[i - 1] - out[j]) < 1.0:
                out[i:j] = np.linspace(out[i - 1], out[j], j - i + 2)[1:-1]
            i = j
        else:
            i += 1
    return out


def _split_on_energy_dips(energy: np.ndarray, start: int, last: int,
                          depth: float = 0.45, min_dip: int = 2) -> list[tuple[int, int]]:
    """Divide [start,last] onde a energia cai a < depth × mediana por >= min_dip quadros.

    Pega a mesma nota repetida ("la-la-la"), que o rastreador de altura une numa nota só."""
    e = energy[start: last + 1]
    if len(e) < 2 * min_dip + 2:
        return [(start, last)]
    low = e < depth * np.median(e)
    pieces, piece_start, i = [], 0, 0
    while i < len(e):
        if low[i]:
            j = i
            while j < len(e) and low[j]:
                j += 1
            if j - i >= min_dip and i > piece_start:
                pieces.append((piece_start, i - 1))
                piece_start = j
            i = j
        else:
            i += 1
    if piece_start < len(e):
        pieces.append((piece_start, len(e) - 1))
    return [(start + a, start + b) for a, b in pieces] or [(start, last)]


def segment_notes(times: np.ndarray, midi: np.ndarray, energy: Optional[np.ndarray] = None, *,
                  split_semitones: float = 0.75, confirm_frames: int = 3, min_note: float = 0.09,
                  flats: bool = False) -> list[NoteEvent]:
    """Agrupa a curva de altura em notas. Vibrato (±~0,5 semitom) não divide a nota."""
    midi = _fill_short_gaps(midi)
    n = len(midi)
    hop = float(np.median(np.diff(times))) if n > 1 else HOP / ANALYSIS_SR
    notes: list[NoteEvent] = []

    def close(start_i: int, end_i: int) -> None:
        ranges = (_split_on_energy_dips(energy, start_i, end_i) if energy is not None
                  else [(start_i, end_i)])
        for a, b in ranges:
            if (b - a + 1) * hop < min_note:
                continue
            m = int(round(float(np.nanmedian(midi[a: b + 1]))))
            notes.append(NoteEvent(float(times[a] - hop / 2), float(times[b] + hop / 2), m,
                                   midi_to_name(m, flats), 1.0))

    i = 0
    while i < n:
        if np.isnan(midi[i]):
            i += 1
            continue
        start, vals = i, [midi[i]]
        last = i
        i += 1
        while i < n and not np.isnan(midi[i]):
            center = float(np.median(vals[-12:]))
            m = midi[i]
            if abs(m - center) <= split_semitones:
                vals.append(m)
                last = i
                i += 1
                continue
            # candidato a nova nota: precisa se manter por `confirm_frames` quadros
            ahead = midi[i: i + confirm_frames]
            sustained = (len(ahead) == confirm_frames and not np.any(np.isnan(ahead))
                         and np.all(np.abs(ahead - m) <= split_semitones))
            if sustained:
                break
            i += 1          # glitch isolado (ex.: salto de oitava de 1-2 quadros): ignora
        close(start, last)
    return notes


def extract_vocal_notes(y: np.ndarray, sr: int = ANALYSIS_SR, flats: bool = False) -> VocalAnalysis:
    if y.size < sr // 2 or float(np.max(np.abs(y))) < 1e-4:
        return VocalAnalysis([], None, None)
    times, midi, energy = track_pitch(y, sr)
    notes = segment_notes(times, midi, energy, flats=flats)
    low, high = vocal_range(notes)
    return VocalAnalysis(notes, low, high)


def vocal_range(notes: list[NoteEvent], trim: float = 0.01) -> tuple[Optional[str], Optional[str]]:
    """Nota mais grave e mais aguda, ignorando os 1% de tempo mais extremos (erros do rastreador)."""
    if not notes:
        return None, None
    ordered = sorted(notes, key=lambda n: n.midi)
    durs = np.array([n.end - n.start for n in ordered])
    cum = np.cumsum(durs) / durs.sum()
    lo = next(n for n, c in zip(ordered, cum) if c >= trim)
    hi = next(n for n, c in zip(reversed(ordered), (1 - cum + durs / durs.sum())[::-1]) if c >= trim)
    return lo.name, hi.name
