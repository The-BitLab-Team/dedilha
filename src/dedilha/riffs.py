"""Riffs: sequências curtas de notas (melodia + ritmo) que se repetem na música.

Como funciona
1. As notas de um instrumento viram uma linha de *eventos*: notas que começam quase juntas formam
   um evento, representado pela nota mais grave (riffs de guitarra e baixo) ou pela mais aguda
   (melodias de piano e solos). As duas leituras são testadas.
2. Cada passo entre eventos vira um par (intervalo em semitons, tempo até o próximo evento). Dois
   trechos são "o mesmo riff" se todos os pares coincidem: o intervalo é exato e o tempo tem pequena
   tolerância. Como só os intervalos contam, um riff tocado em outra altura (transposto) também casa.
3. Uma tabela de comprimentos de repetição (a mesma ideia usada para achar estruturas repetidas em
   sequências) lista, para cada ponto, quantas notas seguidas se repetem em cada outro ponto. Daí
   escolhemos, de forma gulosa, os trechos que cobrem mais notas, sem sobreposição.
4. Descartamos o que não parece riff: acordes batidos, uma nota repetida no mesmo ritmo, alternância
   simples entre duas notas e trechos lentos demais (progressão de acordes sustentados).

Limitação: só reconhece repetições quase idênticas. Um riff com variações é achado pelas partes que
ficam iguais, e uma nota perdida pelo transcritor numa das repetições a deixa de fora.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from bisect import bisect_left, bisect_right

import numpy as np

from .models import NoteEvent, Riff, RiffNote, RiffOccurrence
from .music import midi_to_name
from .tab import make_tab

GROUP_TOL = 0.09          # s: notas que começam com essa diferença formam um evento (acorde; a palhetada leva ~50 ms)
WEAK_RATIO = 0.3          # notas com força abaixo disso (vs. o percentil 90) são ruído
MIN_EVENTS = 4            # menor riff, em eventos (notas ou acordes)
MAX_EVENTS = 24
MAX_SPAN = 10.0           # s: duração máxima de um riff
MAX_MEAN_IOI = 0.9        # s: tempo médio entre notas (mais lento que isso é acorde sustentado)
MAX_IOI = 1.5             # s: pausa maior que isso encerra a frase (o riff não atravessa)
MIN_STEP = 0.09           # s: passos mais curtos que isso são pedaços de uma palhetada, não notas de riff
MAX_FAST_STEPS = 0.25     # fração de passos curtos tolerada (floreios, hammer-ons)
IOI_ABS_TOL = 0.07        # s: tolerância no tempo entre notas
IOI_REL_TOL = 0.15
MAX_TOTAL_EVENTS = 9000
MAX_RIFFS = 6             # por instrumento
LEVELS_PER_START = 10     # comprimentos de trecho avaliados por ponto de partida
HARMONIC_GAPS = (12, 19, 24)   # oitava, oitava+quinta, 2 oitavas: onde o transcritor inventa harmônicos
GHOST_MARGIN = 0.3        # s: um "fantasma de oitava abaixo" aparece enquanto a nota verdadeira soa
GHOST_LOOKBACK = 4.0      # s: até onde olhar para trás (notas longas)
GHOST_RATIO = 0.7


@dataclass
class _Event:
    start: float
    end: float
    notes: list[NoteEvent]
    low: int
    high: int


@dataclass
class _Candidate:
    coverage: int             # eventos cobertos (comprimento × repetições)
    count: int
    k: int                    # comprimento em eventos
    start: int                # evento onde começa a primeira ocorrência considerada
    mode: str                 # "low" | "high"
    positions: np.ndarray


def min_repeats(k: int) -> int:
    """Trechos curtos precisam repetir mais para valer como riff (acaso em músicas longas)."""
    return {4: 4, 5: 3}.get(k, 2)


def _drop_harmonic_ghosts(group: list[NoteEvent]) -> list[NoteEvent]:
    """O transcritor costuma somar harmônicos (oitava, 12ª, 2 oitavas) mais fracos acima da nota tocada."""
    if len(group) < 2:
        return group
    return [n for n in group
            if not any(n.midi - m.midi in HARMONIC_GAPS and n.strength <= m.strength for m in group)]


def _drop_octave_below_ghosts(ns: list[NoteEvent]) -> list[NoteEvent]:
    """Notas fracas 1 ou 2 oitavas abaixo de uma nota bem mais forte e próxima no tempo (artefato típico
    do transcritor em acordes batidos) viram eventos falsos entre os golpes."""
    starts = [n.start for n in ns]
    out = []
    for n in ns:
        lo = bisect_left(starts, n.start - GHOST_LOOKBACK)
        hi = bisect_right(starts, n.start + GHOST_MARGIN)
        if not any(m.midi - n.midi in (12, 24) and n.strength < GHOST_RATIO * m.strength
                   and m.start - GHOST_MARGIN <= n.start <= m.end + GHOST_MARGIN
                   for m in ns[lo:hi]):
            out.append(n)
    return out


def build_events(notes: Iterable[NoteEvent], drop_harmonics: bool = True) -> list[_Event]:
    ns = [n for n in notes if n.end > n.start]
    if not ns:
        return []
    ref = float(np.percentile([n.strength for n in ns], 90))
    ns = [n for n in ns if n.strength >= WEAK_RATIO * ref]
    ns.sort(key=lambda n: (n.start, n.midi))
    if drop_harmonics:
        ns = _drop_octave_below_ghosts(ns)
    groups: list[list[NoteEvent]] = []
    for n in ns:
        if groups and n.start - groups[-1][0].start <= GROUP_TOL:
            groups[-1].append(n)
        else:
            groups.append([n])
    if drop_harmonics:
        groups = [_drop_harmonic_ghosts(g) for g in groups]
    return [_Event(min(n.start for n in g), max(n.end for n in g), g,
                   min(n.midi for n in g), max(n.midi for n in g)) for g in groups]


def _run_lengths(pitch: np.ndarray, onset: np.ndarray) -> np.ndarray:
    """runs[a, b] = quantos passos seguidos (a partir de a e de b) têm o mesmo intervalo e ritmo."""
    iv = np.diff(pitch)
    ioi = np.diff(onset)
    e = len(iv)
    ok = ioi <= MAX_IOI
    runs = np.zeros((e + 1, e + 1), dtype=np.uint8)
    for a in range(e - 1, -1, -1):
        if not ok[a]:
            continue
        tol = np.maximum(IOI_ABS_TOL, IOI_REL_TOL * np.maximum(ioi, ioi[a]))
        same = (iv == iv[a]) & (np.abs(ioi - ioi[a]) <= tol) & ok
        nxt = runs[a + 1, 1:e + 1].astype(np.int16) + 1
        runs[a, :e] = np.where(same, np.minimum(nxt, 255), 0)
    return runs


def _non_overlapping(positions: np.ndarray, k: int) -> np.ndarray:
    out: list[int] = []
    last = -10 ** 9
    for s in positions.tolist():
        if s >= last + k:
            out.append(s)
            last = s
    return np.asarray(out, dtype=int)


def _riff_like(events: list[_Event], pitch: np.ndarray, s: int, k: int) -> bool:
    """O trecho parece um riff (e não um acorde batido, uma nota repetida, uma pausa longa)?"""
    onset = np.array([e.start for e in events[s:s + k]])
    p = pitch[s:s + k]
    ioi = np.diff(onset)
    if onset[-1] - onset[0] > MAX_SPAN or ioi.mean() > MAX_MEAN_IOI:
        return False
    if float(np.mean(ioi < MIN_STEP)) > MAX_FAST_STEPS:
        return False                                  # acorde batido que o transcritor quebrou em pedaços
    uniform = (ioi.max() - ioi.min()) <= max(IOI_ABS_TOL, IOI_REL_TOL * ioi.max())
    density = float(np.mean(np.diff(p) != 0))        # fração dos passos que mudam de nota
    distinct = len(set(p.tolist()))
    if uniform and (distinct <= 2 or density < 0.5):
        return False                                  # nota repetida, trinado ou acordes trocando de tempos em tempos
    sizes = [len(e.notes) for e in events[s:s + k]]
    if float(np.median(sizes)) >= 4 and density < 0.5:
        return False                                  # violão batido: acordes cheios, trocando devagar
    return True


def _candidates(events: list[_Event], mode: str) -> list[_Candidate]:
    pitch = np.array([e.low if mode == "low" else e.high for e in events])
    onset = np.array([e.start for e in events])
    runs = _run_lengths(pitch, onset)
    e = len(events) - 1
    out: list[_Candidate] = []
    for a in range(e):
        row = runs[a, :e].astype(np.int32)
        row[a] = 0
        if row.max(initial=0) < MIN_EVENTS - 1:
            continue
        vals = np.unique(np.minimum(row[row >= MIN_EVENTS - 1], MAX_EVENTS - 1))
        if len(vals) > LEVELS_PER_START:
            vals = np.unique(np.concatenate([vals[-LEVELS_PER_START // 2:],
                                             vals[np.linspace(0, len(vals) - 1, LEVELS_PER_START // 2).astype(int)]]))
        best: Optional[_Candidate] = None
        for r in vals.tolist():
            k = r + 1
            pos = np.union1d(np.flatnonzero(row >= r), [a])
            pos = _non_overlapping(pos, k)
            if len(pos) < min_repeats(k) or not (pos == a).any():
                continue
            cand = _Candidate(k * len(pos), len(pos), k, a, mode, pos)
            if best is None or (cand.coverage, cand.count) > (best.coverage, best.count):
                if _riff_like(events, pitch, a, k):
                    best = cand
        if best is not None:
            out.append(best)
    return out


def _select(cands: list[_Candidate], n_events: int, limit: int) -> list[_Candidate]:
    """Escolha gulosa: o trecho que cobre mais notas primeiro; ocorrências já cobertas são descartadas."""
    flag = np.zeros(n_events, dtype=bool)
    cands = sorted(cands, key=lambda c: (c.coverage, c.count), reverse=True)
    chosen: list[_Candidate] = []
    while cands and len(chosen) < limit:
        c = cands.pop(0)
        pos = np.array([s for s in c.positions.tolist() if not flag[s:s + c.k].any()], dtype=int)
        if len(pos) < min_repeats(c.k):
            continue
        updated = _Candidate(c.k * len(pos), len(pos), c.k, c.start, c.mode, pos)
        if len(pos) < len(c.positions) and cands and \
                (updated.coverage, updated.count) < (cands[0].coverage, cands[0].count):
            cands.append(updated)                       # perdeu ocorrências: reavalia a ordem
            cands.sort(key=lambda x: (x.coverage, x.count), reverse=True)
            continue
        chosen.append(updated)
        for s in pos.tolist():
            flag[s:s + c.k] = True
    return chosen


def _build_riff(instrument: str, instrument_label: str, events: list[_Event], c: _Candidate,
                bpm: Optional[float], flats: bool) -> Riff:
    pitch = np.array([e.low if c.mode == "low" else e.high for e in events])
    k = c.k

    def strength(s: int) -> float:
        return sum(n.strength for e in events[s:s + k] for n in e.notes)

    # Mostra o riff como aparece primeiro na música, a menos que outra execução seja bem mais "limpa"
    # (notas mais fortes = menos chance de ser ruído da transcrição).
    pos = sorted(c.positions.tolist())
    best = max(strength(s) for s in pos)
    rep = int(next(s for s in pos if strength(s) >= 0.85 * best))
    evs = events[rep:rep + k]
    t0 = evs[0].start
    notes: list[RiffNote] = []
    for i, e in enumerate(evs):
        nxt = evs[i + 1].start if i + 1 < k else None
        for n in sorted(e.notes, key=lambda x: x.midi):
            end = n.end if nxt is None else min(n.end, nxt)
            dur = max(0.05, end - e.start)
            if nxt is None:
                dur = min(dur, 1.0)
            notes.append(RiffNote(e.start - t0, dur, n.midi, midi_to_name(n.midi, flats)))
    duration = max(n.offset + n.duration for n in notes)

    occ = []
    for s in sorted(c.positions.tolist()):
        last = events[s + k - 1]
        end = min(max(e.end for e in events[s:s + k]), last.start + 1.0)
        occ.append(RiffOccurrence(events[s].start, end, int(pitch[s] - pitch[rep])))

    summary = " ".join("+".join(midi_to_name(n.midi, flats) for n in sorted(e.notes, key=lambda x: x.midi)[:3])
                       for e in evs)
    beats = None
    if bpm:
        beats = max(0.5, round(duration * bpm / 60.0 * 2) / 2)      # andamento é aproximado: meia batida basta
    tab = make_tab(instrument, [[n.midi for n in e.notes] for e in evs]) if instrument in ("guitar", "bass") else None
    total_events = len(events)
    return Riff(id="", label="", instrument=instrument, instrument_label=instrument_label,
                notes=notes, occurrences=occ, duration=duration, beats=beats, summary=summary,
                coverage=c.coverage / max(1, total_events), tab=tab)


def find_riffs(notes: Iterable[NoteEvent], instrument: str, instrument_label: str, *,
               bpm: Optional[float] = None, flats: bool = False, limit: int = MAX_RIFFS
               ) -> list[Riff]:
    """Riffs de um instrumento, do mais presente para o menos. `id` e `label` ficam vazios."""
    events = build_events(notes, drop_harmonics=instrument != "piano")
    if len(events) < 2 * MIN_EVENTS:
        return []
    events = events[:MAX_TOTAL_EVENTS]
    multi = sum(len(e.notes) > 1 for e in events) / len(events)
    modes = ["low", "high"] if multi >= 0.1 else ["low"]
    cands: list[_Candidate] = []
    for mode in modes:
        cands += _candidates(events, mode)
    chosen = _select(cands, len(events), limit)
    return [_build_riff(instrument, instrument_label, events, c, bpm, flats) for c in chosen]
