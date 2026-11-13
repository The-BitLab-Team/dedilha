"""Estruturas de dados do resultado da análise (serializáveis em JSON)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def _r(x: float, nd: int = 2) -> float:
    return round(float(x), nd)


@dataclass
class NoteEvent:
    start: float
    end: float
    midi: int
    name: str                 # "G5", "F#3"...
    strength: float = 1.0     # 0..1 (amplitude/confiança)

    def to_dict(self) -> dict[str, Any]:
        return {"start": _r(self.start), "end": _r(self.end), "midi": int(self.midi),
                "name": self.name, "strength": _r(self.strength)}


@dataclass
class ChordSegment:
    start: float
    end: float
    label: str                # "Am7", "C/E", "N" (sem acorde)
    root: Optional[int] = None
    quality: str = ""
    confidence: float = 0.0   # 0..1

    def to_dict(self) -> dict[str, Any]:
        return {"start": _r(self.start), "end": _r(self.end), "label": self.label,
                "root": self.root, "quality": self.quality, "confidence": _r(self.confidence)}


@dataclass
class InstrumentResult:
    key: str                  # "mix" | "guitar" | "piano" | "bass" | "voice"
    label: str                # nome de exibição em português
    available: bool = True
    reason: Optional[str] = None            # por que não está disponível
    chords: list[ChordSegment] = field(default_factory=list)
    notes: list[NoteEvent] = field(default_factory=list)
    range_low: Optional[str] = None         # voz e baixo: nota mais grave
    range_high: Optional[str] = None        # voz e baixo: nota mais aguda

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "label": self.label, "available": self.available,
            "reason": self.reason,
            "chords": [c.to_dict() for c in self.chords],
            "notes": [n.to_dict() for n in self.notes],
            "range_low": self.range_low, "range_high": self.range_high,
        }


@dataclass
class RiffNote:
    offset: float             # segundos desde o início do riff
    duration: float
    midi: int
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {"offset": _r(self.offset, 3), "duration": _r(self.duration, 3),
                "midi": int(self.midi), "name": self.name}


@dataclass
class RiffOccurrence:
    start: float
    end: float
    transpose: int = 0        # semitons em relação à versão mostrada (riff repetido em outra altura)

    def to_dict(self) -> dict[str, Any]:
        return {"start": _r(self.start), "end": _r(self.end), "transpose": int(self.transpose)}


@dataclass
class Riff:
    id: str                   # "riff-1"
    label: str                # "Riff 1"
    instrument: str           # "guitar" | "piano" | "bass"
    instrument_label: str     # nome de exibição em português
    notes: list[RiffNote] = field(default_factory=list)
    occurrences: list[RiffOccurrence] = field(default_factory=list)
    duration: float = 0.0     # segundos
    beats: Optional[float] = None       # duração em batidas (aproximada; precisa do andamento)
    summary: str = ""         # "E2 G2 A2 Bb2 A2 G2"
    coverage: float = 0.0     # fração das notas do instrumento ocupada por este riff (0..1)
    tab: Optional[list[str]] = None     # tablatura (violão/guitarra e baixo, afinação padrão)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "instrument": self.instrument,
            "instrument_label": self.instrument_label,
            "notes": [n.to_dict() for n in self.notes],
            "occurrences": [o.to_dict() for o in self.occurrences],
            "duration": _r(self.duration, 3),
            "beats": None if self.beats is None else _r(self.beats),
            "summary": self.summary, "coverage": _r(self.coverage, 3), "tab": self.tab,
        }


@dataclass
class TrackInfo:
    source: str
    source_kind: str                        # "file" | "youtube" | "spotify" | "url"
    title: Optional[str] = None
    artist: Optional[str] = None
    resolved_url: Optional[str] = None      # no Spotify: o vídeo do YouTube que foi usado
    duration: float = 0.0

    def display_name(self) -> str:
        if self.title and self.artist:
            return f"{self.title} — {self.artist}"
        return self.title or self.source

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "source_kind": self.source_kind, "title": self.title,
                "artist": self.artist, "resolved_url": self.resolved_url,
                "duration": _r(self.duration), "display_name": self.display_name()}


@dataclass
class AnalysisResult:
    track: TrackInfo
    key: Optional[str] = None               # "C maior"
    key_short: Optional[str] = None         # "C", "Am"
    tempo: Optional[float] = None           # BPM
    tuning_cents: float = 0.0               # afinação estimada (cents em relação a A=440)
    separated: bool = False                 # houve separação de instrumentos?
    instruments: dict[str, InstrumentResult] = field(default_factory=dict)
    riffs: list[Riff] = field(default_factory=list)
    riffs_requested: bool = False
    riff_messages: list[str] = field(default_factory=list)   # por que não achou riffs (ou avisos)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    waveform: list[float] = field(default_factory=list)      # picos 0..1 para desenhar a onda

    def to_dict(self) -> dict[str, Any]:
        return {
            "track": self.track.to_dict(), "key": self.key, "key_short": self.key_short,
            "tempo": None if self.tempo is None else _r(self.tempo, 1),
            "tuning_cents": _r(self.tuning_cents, 1), "separated": self.separated,
            "instruments": {k: v.to_dict() for k, v in self.instruments.items()},
            "riffs": [r.to_dict() for r in self.riffs], "riffs_requested": self.riffs_requested,
            "riff_messages": list(self.riff_messages),
            "warnings": list(self.warnings),
            "waveform": [round(float(x), 2) for x in self.waveform],
            "timings": {k: _r(v, 1) for k, v in self.timings.items()},
        }
