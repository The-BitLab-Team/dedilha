"""Gerador de música sintética com gabarito conhecido (acordes e melodia vocal).

Sem acesso a gravações reais, é assim que medimos a precisão: tocamos progressões com timbres
parecidos com violão, piano, baixo, voz e bateria, e comparamos com o que realmente foi tocado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, lfilter

from dedilha.music import CHORD_TYPES, chord_label

SR = 22050


def _env(n: int, sr: int, attack: float = 0.004, release: float = 0.03) -> np.ndarray:
    e = np.ones(n)
    a, r = max(1, int(attack * sr)), max(1, int(release * sr))
    e[:a] *= np.linspace(0, 1, a)
    e[-r:] *= np.linspace(1, 0, min(r, n))
    return e


def _mtof(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def guitar_note(midi: float, dur: float, sr: int = SR, rng=None) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = _mtof(midi)
    y = np.zeros(n)
    for k in range(1, 9):
        if f * k > sr / 2 - 500:
            break
        y += (1.0 / k) * np.exp(-t * (1.5 + 1.3 * k)) * np.sin(2 * np.pi * f * k * t)
    if rng is not None:                                   # transiente de palheta
        burst = rng.standard_normal(int(0.006 * sr)) * 0.15
        y[: len(burst)] += burst
    return y * _env(n, sr, 0.002, 0.04)


def piano_note(midi: float, dur: float, sr: int = SR) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = _mtof(midi)
    y = np.zeros(n)
    for k in range(1, 11):
        fk = f * k * np.sqrt(1 + 2e-4 * k * k)
        if fk > sr / 2 - 500:
            break
        y += (1.0 / k ** 1.1) * np.exp(-t * (1.0 + 0.45 * k)) * np.sin(2 * np.pi * fk * t)
    return y * _env(n, sr, 0.003, 0.06)


def bass_note(midi: float, dur: float, sr: int = SR) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = _mtof(midi)
    y = sum(a * np.exp(-t * d) * np.sin(2 * np.pi * f * k * t)
            for k, a, d in ((1, 1.0, 1.6), (2, 0.5, 2.5), (3, 0.25, 3.5), (4, 0.12, 4.5)))
    return y * _env(n, sr, 0.005, 0.05)


def voice_note(midi: float, dur: float, sr: int = SR, vibrato_cents: float = 25.0) -> np.ndarray:
    """Voz sintética: série harmônica com formantes de 'a' e vibrato de ~5,5 Hz."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    vib = np.where(t > 0.15, np.sin(2 * np.pi * 5.5 * t), 0.0) * vibrato_cents / 100.0
    f0 = _mtof(midi + vib)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = np.zeros(n)
    for k in range(1, 25):
        fk = f0 * k
        if np.max(fk) > sr / 2 - 500:
            break
        formant = (np.exp(-((fk - 700) / 220) ** 2) + 0.6 * np.exp(-((fk - 1150) / 250) ** 2)
                   + 0.2 * np.exp(-((fk - 2600) / 400) ** 2))
        y += (0.2 + formant) / k ** 0.7 * np.sin(k * phase)
    return y * _env(n, sr, 0.03, 0.06)


def drums(total: float, bpm: float, sr: int = SR, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(int(total * sr) + sr)
    beat = 60.0 / bpm
    hp = butter(2, 6000 / (sr / 2), btype="high")
    bp = butter(2, [1500 / (sr / 2), 5000 / (sr / 2)], btype="band")
    for i in range(int(total / beat)):
        pos = int(i * beat * sr)
        if i % 4 in (0, 2):                                # bumbo
            t = np.arange(int(0.15 * sr)) / sr
            k = np.sin(2 * np.pi * (50 * t + 70 * (1 - np.exp(-t * 30)) / 30)) * np.exp(-t * 25)
            out[pos: pos + len(k)] += 0.9 * k
        if i % 4 in (1, 3):                                # caixa
            s = lfilter(*bp, rng.standard_normal(int(0.12 * sr))) * np.exp(-np.arange(int(0.12 * sr)) / sr * 30)
            out[pos: pos + len(s)] += 0.5 * s
        h = lfilter(*hp, rng.standard_normal(int(0.04 * sr))) * np.exp(-np.arange(int(0.04 * sr)) / sr * 90)
        out[pos: pos + len(h)] += 0.2 * h
    return out[: int(total * sr)]


@dataclass
class Song:
    sr: int
    bpm: float
    duration: float
    stems: dict[str, np.ndarray]
    chords: list[tuple[float, float, int, str]]          # (início, fim, fundamental, tipo)
    melody: list[tuple[float, float, int]] = field(default_factory=list)   # (início, fim, MIDI)

    def mix(self, keys=("guitar", "piano", "bass", "vocals", "drums")) -> np.ndarray:
        n = max(len(self.stems[k]) for k in keys)
        out = np.zeros(n)
        for k in keys:
            out[: len(self.stems[k])] += self.stems[k]
        return (out / max(1e-9, np.max(np.abs(out))) * 0.9).astype(np.float32)

    def truth_label(self, t: float, flats: bool = False) -> str:
        for s, e, r, q in self.chords:
            if s <= t < e:
                return chord_label(r, q, flats)
        return "N"


def _guitar_voicing(root: int, quality: str) -> list[int]:
    base = 40 + (root - 4) % 12                            # E2..D#3
    iv = CHORD_TYPES[quality].intervals
    third = iv[1] if len(iv) > 2 else 7
    seventh_or_octave = base + (iv[3] if len(iv) == 4 else 12)
    return [base, base + 7, seventh_or_octave, base + 12 + third, base + 19]


def _piano_voicing(root: int, quality: str) -> tuple[int, list[int]]:
    iv = CHORD_TYPES[quality].intervals
    r = 60 + root % 12
    if r > 65:
        r -= 12
    return 36 + root % 12 + 12, [r + i for i in iv]


def make_song(progression: list[tuple[int, str]], *, bpm: float = 100.0, repeats: int = 4,
              bars_each: int = 1, sr: int = SR, seed: int = 0, detune_cents: float = 0.0,
              with_voice: bool = True, with_drums: bool = True) -> Song:
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = 4 * beat
    seq = progression * repeats
    total = len(seq) * bars_each * bar
    n = int(total * sr) + sr
    stems = {k: np.zeros(n) for k in ("guitar", "piano", "bass", "vocals")}
    chords, melody = [], []
    shift = detune_cents / 100.0

    def put(stem: str, y: np.ndarray, t0: float) -> None:
        i = int(t0 * sr)
        stems[stem][i: i + len(y)] += y[: max(0, n - i)]

    for ci, (root, q) in enumerate(seq):
        t0 = ci * bars_each * bar
        t1 = t0 + bars_each * bar
        chords.append((t0, t1, root, q))
        for b in range(bars_each):
            tb = t0 + b * bar
            # violão: batida de 4 tempos (2 strums de 2 tempos), notas levemente escalonadas
            for half in range(2):
                ts = tb + half * 2 * beat
                for j, m in enumerate(_guitar_voicing(root, q)):
                    put("guitar", 0.55 * guitar_note(m + shift, 2 * beat, sr, rng), ts + j * 0.012)
            # piano: acorde sustentado + nota grave
            bassm, tri = _piano_voicing(root, q)
            put("piano", 0.6 * piano_note(bassm + shift, bar, sr), tb)
            for m in tri:
                put("piano", 0.5 * piano_note(m + shift, bar, sr), tb)
            # baixo: fundamental nos tempos 1 e 3
            for half in range(2):
                put("bass", 0.9 * bass_note(36 + root % 12 + shift, 2 * beat * 0.9, sr), tb + half * 2 * beat)
            # voz: 4 notas por compasso, tons do acorde entre G3 e G4/A4
            if with_voice:
                tones = sorted({(root + i) % 12 for i in CHORD_TYPES[q].intervals})
                for k in range(4):
                    pc = tones[int(rng.integers(len(tones)))]
                    m = 57 + (pc - 57) % 12                # A3..G#4
                    if k == 3 and rng.random() < 0.3:
                        continue                           # pausa
                    s, e = tb + k * beat, tb + (k + 1) * beat * 0.95
                    melody.append((s, e, m))
                    put("vocals", 0.5 * voice_note(m + shift, e - s, sr), s)
    stems = {k: v[: int(total * sr)] for k, v in stems.items()}
    stems["drums"] = drums(total, bpm, sr, seed) if with_drums else np.zeros(int(total * sr))
    return Song(sr, bpm, total, stems, chords, melody)


def chord_accuracy(pred_segments, song: Song, *, hop: float = 0.1, guard: float = 0.3,
                   flats: bool = False) -> float:
    """Fração dos instantes (fora de `guard` s após cada troca) em que o acorde previsto = verdadeiro."""
    def pred_at(t: float) -> str:
        for s in pred_segments:
            if s.start <= t < s.end:
                return s.label
        return "N"

    ok = tot = 0
    for s, e, r, q in song.chords:
        t = s + guard
        while t < e - 0.1:
            tot += 1
            ok += pred_at(t) == song.truth_label(t, flats)
            t += hop
    return ok / max(tot, 1)


# ---------------------------------------------------------------------------------------------
# Música com riff conhecido

# (início em segundos a 120 BPM, duração, nota MIDI): E2 G2 A2 Bb2 A2 G2 — um compasso de 2 s
RIFF_120 = [(0.0, 0.25, 40), (0.25, 0.25, 43), (0.5, 0.25, 45), (0.75, 0.25, 46), (1.0, 0.5, 45),
            (1.5, 0.5, 43)]
BAR_120 = 2.0


@dataclass
class RiffSong:
    sr: int
    bpm: float
    duration: float
    stems: dict[str, np.ndarray]
    riff: list[tuple[float, float, int]]
    starts: list[float]                       # onde o riff começa (s)
    transposed: dict[float, int]              # início -> semitons (ocorrências em outra altura)

    def mix(self) -> np.ndarray:
        n = max(len(v) for v in self.stems.values())
        out = np.zeros(n)
        for v in self.stems.values():
            out[: len(v)] += v
        return (out / max(1e-9, np.max(np.abs(out))) * 0.9).astype(np.float32)


def make_riff_song(*, riff_bars_a: int = 4, verse_bars: int = 4, riff_bars_b: int = 4,
                   transpose_b: int = 0, sr: int = SR, seed: int = 5, with_drums: bool = True
                   ) -> RiffSong:
    """riff x N compassos, verso com acordes batidos (sem riff), riff de novo (talvez transposto).

    O baixo toca só a fundamental de cada compasso; a bateria marca 120 BPM.
    """
    rng = np.random.default_rng(seed)
    bar = BAR_120
    total = (riff_bars_a + verse_bars + riff_bars_b) * bar
    n = int(total * sr) + sr
    stems = {k: np.zeros(n) for k in ("guitar", "piano", "bass", "vocals")}

    def put(stem: str, y: np.ndarray, t0: float) -> None:
        i = int(t0 * sr)
        stems[stem][i: i + len(y)] += y[: max(0, n - i)]

    starts, transposed = [], {}
    for b in range(riff_bars_a + verse_bars + riff_bars_b):
        t0 = b * bar
        if b < riff_bars_a or b >= riff_bars_a + verse_bars:
            shift = transpose_b if b >= riff_bars_a + verse_bars else 0
            starts.append(t0)
            if shift:
                transposed[t0] = shift
            for off, dur, m in RIFF_120:
                put("guitar", 0.7 * guitar_note(m + shift, dur * 0.95, sr, rng), t0 + off)
            put("bass", 0.8 * bass_note(28 + shift, 1.8, sr), t0)         # E1 sustentado
        else:                                                              # verso: C G Am F batidos
            root, q = [(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")][(b - riff_bars_a) % 4]
            for half in range(2):
                for j, m in enumerate(_guitar_voicing(root, q)):
                    put("guitar", 0.55 * guitar_note(m, bar / 2, sr, rng), t0 + half * bar / 2 + j * 0.012)
            put("bass", 0.8 * bass_note(36 + root % 12, 1.8, sr), t0)
    stems = {k: v[: int(total * sr)] for k, v in stems.items()}
    stems["drums"] = drums(total, 120.0, sr, seed) if with_drums else np.zeros(int(total * sr))
    return RiffSong(sr, 120.0, total, stems, RIFF_120, starts, transposed)
