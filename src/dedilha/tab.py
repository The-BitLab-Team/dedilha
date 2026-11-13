"""Tablatura simples para um riff (afinação padrão): escolhe corda e casa com o menor movimento da mão.

É uma sugestão: a mesma nota existe em várias cordas, e o violonista pode preferir outra posição.
"""

from __future__ import annotations

import itertools
from typing import Optional, Sequence

# cordas da mais aguda para a mais grave: MIDI das cordas soltas e rótulos da tablatura
TUNINGS: dict[str, tuple[int, ...]] = {
    "guitar": (64, 59, 55, 50, 45, 40),     # E4 B3 G3 D3 A2 E2
    "bass": (43, 38, 33, 28),               # G2 D2 A1 E1
}
STRING_LABELS: dict[str, tuple[str, ...]] = {
    "guitar": ("e", "B", "G", "D", "A", "E"),
    "bass": ("G", "D", "A", "E"),
}
MAX_FRET = 17
MAX_HAND_SPAN = 4          # casas entre o dedo mais agudo e o mais grave num mesmo acorde
MAX_CHORD_NOTES = 4
KEEP_CANDIDATES = 12
MAX_UNPLAYABLE = 0.3       # se mais que isso das notas não existir no instrumento, não gera tablatura


def _candidates(notes: Sequence[int], tuning: Sequence[int]) -> list[tuple[tuple[int, int], ...]]:
    """Todas as formas de tocar as notas de um evento em cordas diferentes (string, fret)."""
    options = [[(s, m - open_) for s, open_ in enumerate(tuning) if 0 <= m - open_ <= MAX_FRET]
               for m in notes]
    out: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.product(*options):
        if len({s for s, _ in combo}) != len(combo):
            continue
        fretted = [f for _, f in combo if f > 0]
        if fretted and max(fretted) - min(fretted) > MAX_HAND_SPAN:
            continue
        out.append(combo)
    out.sort(key=_intrinsic_cost)
    return out[:KEEP_CANDIDATES]


def _position(combo: Sequence[tuple[int, int]]) -> Optional[float]:
    fretted = [f for _, f in combo if f > 0]
    return sum(fretted) / len(fretted) if fretted else None


def _intrinsic_cost(combo: Sequence[tuple[int, int]]) -> float:
    pos = _position(combo)
    return 0.0 if pos is None else 0.08 * pos


def _move_cost(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 0.0
    return abs(a - b)


def assign_frets(events: Sequence[Sequence[int]], instrument: str
                 ) -> Optional[list[tuple[tuple[int, int], ...]]]:
    """Para cada evento (lista de notas MIDI), as posições (corda, casa). None se não for tocável."""
    tuning = TUNINGS.get(instrument)
    if tuning is None or not events:
        return None
    cands: list[list[tuple[tuple[int, int], ...]]] = []
    lost = 0
    for ev in events:
        notes = sorted(set(ev))[:MAX_CHORD_NOTES]
        playable = [m for m in notes if any(0 <= m - o <= MAX_FRET for o in tuning)]
        lost += len(notes) - len(playable)
        c = _candidates(playable, tuning) if playable else []
        cands.append(c or [()])
    total = sum(min(len(set(ev)), MAX_CHORD_NOTES) for ev in events)
    if total == 0 or lost / total > MAX_UNPLAYABLE:
        return None

    # Viterbi sobre as posições candidatas de cada evento
    best = [[_intrinsic_cost(c) for c in cands[0]]]
    back: list[list[int]] = [[-1] * len(cands[0])]
    for i in range(1, len(cands)):
        row, brow = [], []
        for c in cands[i]:
            pc = _position(c)
            options = [best[i - 1][j] + _move_cost(_position(p), pc) for j, p in enumerate(cands[i - 1])]
            j = min(range(len(options)), key=options.__getitem__)
            row.append(options[j] + _intrinsic_cost(c))
            brow.append(j)
        best.append(row)
        back.append(brow)
    k = min(range(len(best[-1])), key=best[-1].__getitem__)
    chosen: list[tuple[tuple[int, int], ...]] = []
    for i in range(len(cands) - 1, -1, -1):
        chosen.append(cands[i][k])
        k = back[i][k]
    chosen.reverse()
    return chosen


def make_tab(instrument: str, events: Sequence[Sequence[int]]) -> Optional[list[str]]:
    """Linhas de tablatura (da corda mais aguda para a mais grave), uma coluna por evento."""
    placed = assign_frets(events, instrument)
    if placed is None:
        return None
    labels = STRING_LABELS[instrument]
    cells = [["-"] * len(placed) for _ in labels]
    for col, combo in enumerate(placed):
        for s, fret in combo:
            cells[s][col] = str(fret)
    width = max(len(c) for row in cells for c in row) + 2
    lines = []
    for label, row in zip(labels, cells):
        body = "".join(c.center(width, "-") if c != "-" else "-" * width for c in row)
        lines.append(f"{label}|{body}|")
    return lines
