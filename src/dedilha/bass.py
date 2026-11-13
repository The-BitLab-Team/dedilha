"""Baixo: limpa as notas do basic-pitch (linha monofônica) e extrai a fundamental de cada acorde."""

from __future__ import annotations

import dataclasses
from typing import Iterable, Optional

import numpy as np

from .models import ChordSegment, NoteEvent
from .music import midi_to_name, pc_name

BASS_MIN_MIDI = 23     # B0: corda si grave do baixo de 5 cordas
BASS_MAX_MIDI = 67     # G4: 24ª casa da corda sol
MIN_NOTE_SECONDS = 0.09
MERGE_GAP_SECONDS = 0.03   # notas iguais coladas: podem ser um pedaço solto do detector
FRAGMENT_SECONDS = 0.2
HARMONIC_GAPS = (12, 19, 24)   # oitava, oitava+quinta e 2 oitavas acima de outra nota


def _overlap(a: NoteEvent, b: NoteEvent) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _is_fragment(small: NoteEvent, big: NoteEvent) -> bool:
    return (small.end - small.start < FRAGMENT_SECONDS
            and small.strength < 0.7 * big.strength
            and (big.end - big.start) > 2 * (small.end - small.start))


def clean_bass_line(notes: Iterable[NoteEvent], flats: bool = False) -> list[NoteEvent]:
    """Deixa só a linha de baixo: faixa de notas do instrumento, sem harmônicos fantasmas e monofônica.

    O basic-pitch costuma acrescentar uma nota fraca uma oitava acima (harmônico) ou abaixo da
    nota tocada; como o baixo toca uma nota por vez, mantemos só a mais forte de cada trecho.
    """
    cand = [n for n in notes
            if BASS_MIN_MIDI <= n.midi <= BASS_MAX_MIDI and n.end - n.start >= MIN_NOTE_SECONDS]

    keep: list[NoteEvent] = []
    for n in cand:
        dur = n.end - n.start
        ghost = False
        for m in cand:
            if m is n or _overlap(n, m) < 0.6 * dur:
                continue
            gap = n.midi - m.midi
            if gap in HARMONIC_GAPS and n.strength <= m.strength:
                ghost = True            # harmônico agudo mais fraco que a nota de baixo
            elif gap == -12 and n.strength < 0.7 * m.strength:
                ghost = True            # "oitava abaixo" fantasma
            if ghost:
                break
        if not ghost:
            keep.append(n)

    keep.sort(key=lambda n: (n.start, n.midi))
    out: list[NoteEvent] = []
    for n in keep:
        n = dataclasses.replace(n, name=midi_to_name(n.midi, flats))
        if out:
            prev = out[-1]
            ov = _overlap(prev, n)
            shorter = min(prev.end - prev.start, n.end - n.start)
            if ov > 0.5 * shorter:       # duas notas ao mesmo tempo: fica a de mais energia
                if n.strength * (n.end - n.start) > prev.strength * (prev.end - prev.start):
                    out[-1] = n
                continue
            if prev.midi == n.midi and n.start - prev.end <= MERGE_GAP_SECONDS:
                # mesma nota colada na anterior: se uma das duas é um pedaço curto e fraco (eco do
                # ataque ou cauda), descartamos o pedaço; senão são duas pancadas de verdade.
                if _is_fragment(prev, n):
                    out[-1] = n
                    continue
                if _is_fragment(n, prev):
                    continue
            if ov > 0:                   # legato: a anterior termina onde a nova começa
                prev.end = max(prev.start, n.start)
        out.append(n)
    return out


def bass_roots(notes: list[NoteEvent], segments: Iterable[ChordSegment], flats: bool = False
               ) -> list[ChordSegment]:
    """Nota de baixo predominante em cada acorde da música (a "cifra do baixista")."""
    out: list[ChordSegment] = []
    for seg in segments:
        if seg.label == "N":
            continue
        w = np.zeros(12)
        for n in notes:
            ov = min(seg.end, n.end) - max(seg.start, n.start)
            if ov > 0:
                w[n.midi % 12] += ov * max(n.strength, 0.05)
        total = float(w.sum())
        if total <= 0:
            continue
        pc = int(w.argmax())
        label = pc_name(pc, flats)
        prev: Optional[ChordSegment] = out[-1] if out else None
        if prev and prev.label == label and abs(prev.end - seg.start) < 1e-6:
            prev.end = seg.end
            continue
        out.append(ChordSegment(seg.start, seg.end, label, root=pc, quality="baixo",
                                confidence=float(w[pc] / total)))
    return out


def bass_range(notes: list[NoteEvent]) -> tuple[Optional[str], Optional[str]]:
    if not notes:
        return None, None
    lo = min(notes, key=lambda n: n.midi)
    hi = max(notes, key=lambda n: n.midi)
    return lo.name, hi.name
