"""Teoria musical básica: nomes de notas, vocabulário de acordes, tom (key)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def pc_name(pc: int, flats: bool = False) -> str:
    """Nome da classe de altura (0=C ... 11=B)."""
    return (NOTE_NAMES_FLAT if flats else NOTE_NAMES_SHARP)[int(pc) % 12]


def midi_to_name(midi: float, flats: bool = False) -> str:
    """Nome científico da nota, com C4 = MIDI 60 e A4 = 440 Hz. Ex.: 79 -> 'G5'."""
    n = int(round(midi))
    return f"{pc_name(n % 12, flats)}{n // 12 - 1}"


def name_to_midi(name: str) -> int:
    """Inverso de midi_to_name. Aceita 'G5', 'F#3', 'Bb2'."""
    name = name.strip()
    letter = name[0].upper()
    rest = name[1:]
    acc = 0
    while rest and rest[0] in "#b":
        acc += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    return (int(rest) + 1) * 12 + base + acc


def hz_to_midi(hz: float | np.ndarray) -> float | np.ndarray:
    return 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=float) / 440.0)


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


# --------------------------------------------------------------------------- acordes


@dataclass(frozen=True)
class ChordType:
    name: str                 # identificador interno
    suffix: str               # sufixo na cifra: "", "m", "7", "maj7"...
    intervals: tuple[int, ...]
    weights: tuple[float, ...]
    bias: float               # penalidade a priori (tipos raros/complexos sofrem mais)


# Pesos: fundamental > terça > quinta > sétima. A quinta pesa menos porque costuma
# ser omitida ou fraca; a sétima pesa pouco para não "inventar" sétimas em tríades.
CHORD_TYPES: dict[str, ChordType] = {
    "maj":  ChordType("maj",  "",     (0, 4, 7),     (1.0, 0.9, 0.8),       0.00),
    "min":  ChordType("min",  "m",    (0, 3, 7),     (1.0, 0.9, 0.8),       0.00),
    "7":    ChordType("7",    "7",    (0, 4, 7, 10), (1.0, 0.9, 0.8, 0.7),  0.05),
    "maj7": ChordType("maj7", "maj7", (0, 4, 7, 11), (1.0, 0.9, 0.8, 0.7),  0.06),
    "min7": ChordType("min7", "m7",   (0, 3, 7, 10), (1.0, 0.9, 0.8, 0.7),  0.04),
    "sus4": ChordType("sus4", "sus4", (0, 5, 7),     (1.0, 0.9, 0.8),       0.07),
    "sus2": ChordType("sus2", "sus2", (0, 2, 7),     (1.0, 0.9, 0.8),       0.07),
    "dim":  ChordType("dim",  "dim",  (0, 3, 6),     (1.0, 0.9, 0.8),       0.08),
    "aug":  ChordType("aug",  "aug",  (0, 4, 8),     (1.0, 0.9, 0.8),       0.10),
    "m7b5": ChordType("m7b5", "m7b5", (0, 3, 6, 10), (1.0, 0.9, 0.8, 0.7),  0.08),
    "5":    ChordType("5",    "5",    (0, 7),        (1.0, 0.9),            0.03),
}

# Vocabulários disponíveis. "standard" é o padrão: tríades + sétimas mais comuns.
VOCABULARIES: dict[str, tuple[str, ...]] = {
    "simple": ("maj", "min"),
    "standard": ("maj", "min", "7", "maj7", "min7"),
    "extended": ("maj", "min", "7", "maj7", "min7", "sus4", "sus2", "dim", "aug", "m7b5", "5"),
}


def chord_label(root: int, quality: str, flats: bool = False, bass: int | None = None) -> str:
    label = pc_name(root, flats) + CHORD_TYPES[quality].suffix
    if bass is not None and bass % 12 != root % 12:
        label += "/" + pc_name(bass, flats)
    return label


def chord_tones(root: int, quality: str) -> set[int]:
    return {(root + i) % 12 for i in CHORD_TYPES[quality].intervals}


# --------------------------------------------------------------------------- tom

# Perfis de Krumhansl-Schmuckler (percepção de tonalidade).
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_FLAT_MAJOR_TONICS = {5, 10, 3, 8, 1}      # F, Bb, Eb, Ab, Db
_FLAT_MINOR_TONICS = {2, 7, 0, 5, 10, 3}   # Dm, Gm, Cm, Fm, Bbm, Ebm


@dataclass(frozen=True)
class KeyEstimate:
    tonic: int
    mode: str          # "major" | "minor"
    correlation: float

    @property
    def uses_flats(self) -> bool:
        return self.tonic in (_FLAT_MAJOR_TONICS if self.mode == "major" else _FLAT_MINOR_TONICS)

    def name(self) -> str:
        """Ex.: 'C maior', 'A menor' (em português, como o app é voltado ao Brasil)."""
        return f"{pc_name(self.tonic, self.uses_flats)} {'maior' if self.mode == 'major' else 'menor'}"

    def short(self) -> str:
        return pc_name(self.tonic, self.uses_flats) + ("" if self.mode == "major" else "m")


def estimate_key(chroma_profile: np.ndarray) -> KeyEstimate | None:
    """Estima o tom a partir de um vetor de 12 posições (energia por classe de altura)."""
    v = np.asarray(chroma_profile, dtype=float)
    if v.shape != (12,) or v.sum() <= 1e-9 or np.std(v) < 1e-12:
        return None
    best: KeyEstimate | None = None
    for mode, prof in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(v, np.roll(prof, tonic))[0, 1])
            if best is None or r > best.correlation:
                best = KeyEstimate(tonic, mode, r)
    return best


def fmt_time(seconds: float) -> str:
    """mm:ss (ou h:mm:ss)."""
    s = max(0, int(math.floor(seconds + 1e-6)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
