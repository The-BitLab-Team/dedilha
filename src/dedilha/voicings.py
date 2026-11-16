"""Como tocar cada acorde: posição no braço do violão/guitarra, teclas do piano e sugestão de capotraste.

Violão/guitarra em afinação padrão (E A D G B E). Para cada acorde escolhemos:
1. a forma aberta tradicional, quando existe (C, G, D, Am, Em...);
2. senão, a forma móvel (com pestana) mais perto do começo do braço: a "forma de E" (fundamental na
   6ª corda) ou a "forma de A" (fundamental na 5ª corda).

É uma sugestão: todo acorde tem várias posições, e o músico da gravação pode ter usado outra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .models import ChordSegment
from .music import CHORD_TYPES, chord_tones

GUITAR_TUNING = (40, 45, 50, 55, 59, 64)     # E2 A2 D3 G3 B3 E4: da 6ª corda (grave) para a 1ª (aguda)
MAX_CAPO = 7
CAPO_FRET_COST = 0.03     # cada casa a mais de capotraste "custa" isso (prefere o capotraste mais baixo)
MIN_CAPO_GAIN = 0.3       # só sugere se o ganho passar disso: trocar um único acorde com pestana não compensa
X = None

Frets = tuple[Optional[int], ...]


@dataclass(frozen=True)
class GuitarShape:
    frets: Frets                       # casa por corda, da 6ª para a 1ª; None = corda não tocada
    barre: Optional[int] = None        # casa da pestana (dedo indicador deitado), se houver

    @property
    def open(self) -> bool:
        """Forma aberta: sem pestana, com cordas soltas e perto do começo do braço (as mais fáceis)."""
        played = [f for f in self.frets if f is not None]
        return self.barre is None and 0 in played and max(played) <= 4

    @property
    def text(self) -> str:
        """Cifra de posição, da 6ª para a 1ª corda: 'x32010' (com casas de dois dígitos: 'x-10-12-12-11-10')."""
        cells = ["x" if f is None else str(f) for f in self.frets]
        return "-".join(cells) if any(len(c) > 1 for c in cells) else "".join(cells)

    def midi(self) -> list[int]:
        return [o + f for o, f in zip(GUITAR_TUNING, self.frets) if f is not None]

    def to_dict(self) -> dict[str, Any]:
        return {"frets": list(self.frets), "barre": self.barre, "open": self.open, "text": self.text}


def _f(text: str) -> Frets:
    return tuple(None if c == "x" else int(c) for c in text)


# Formas abertas tradicionais: (fundamental, tipo) -> casas.
_OPEN: dict[tuple[int, str], Frets] = {
    (0, "maj"): _f("x32010"), (2, "maj"): _f("xx0232"), (4, "maj"): _f("022100"),
    (7, "maj"): _f("320003"), (9, "maj"): _f("x02220"),
    (9, "min"): _f("x02210"), (2, "min"): _f("xx0231"), (4, "min"): _f("022000"),
    (0, "7"): _f("x32310"), (2, "7"): _f("xx0212"), (4, "7"): _f("020100"),
    (7, "7"): _f("320001"), (9, "7"): _f("x02020"), (11, "7"): _f("x21202"),
    (0, "maj7"): _f("x32000"), (2, "maj7"): _f("xx0222"), (4, "maj7"): _f("021100"),
    (5, "maj7"): _f("xx3210"), (7, "maj7"): _f("320002"), (9, "maj7"): _f("x02120"),
    (9, "min7"): _f("x02010"), (2, "min7"): _f("xx0211"), (4, "min7"): _f("020000"),
    (9, "sus4"): _f("x02230"), (2, "sus4"): _f("xx0233"), (4, "sus4"): _f("022200"),
    (9, "sus2"): _f("x02200"), (2, "sus2"): _f("xx0230"),
    (2, "dim"): _f("xx0131"), (0, "aug"): _f("x32110"), (2, "5"): _f("xx023x"),
}

# Formas móveis, em casas relativas à casa da fundamental.
_E_SHAPES: dict[str, Frets] = {          # fundamental na 6ª corda (E solta)
    "maj": (0, 2, 2, 1, 0, 0), "min": (0, 2, 2, 0, 0, 0), "7": (0, 2, 0, 1, 0, 0),
    "maj7": (0, 2, 1, 1, 0, 0), "min7": (0, 2, 0, 0, 0, 0), "sus4": (0, 2, 2, 2, 0, 0),
    "5": (0, 2, 2, X, X, X), "dim": (0, 1, 2, 0, X, X), "aug": (0, 3, 2, 1, 1, 0),
    "m7b5": (0, X, 0, 0, -1, X),
}
_A_SHAPES: dict[str, Frets] = {          # fundamental na 5ª corda (A solta)
    "maj": (X, 0, 2, 2, 2, 0), "min": (X, 0, 2, 2, 1, 0), "7": (X, 0, 2, 0, 2, 0),
    "maj7": (X, 0, 2, 1, 2, 0), "min7": (X, 0, 2, 0, 1, 0), "sus4": (X, 0, 2, 2, 3, 0),
    "sus2": (X, 0, 2, 2, 0, 0), "5": (X, 0, 2, 2, X, X), "dim": (X, 0, 1, 2, 1, X),
    "aug": (X, 0, 3, 2, 2, 1), "m7b5": (X, 0, 1, 0, 1, X),
}
_D_SHAPES: dict[str, Frets] = {          # fundamental na 4ª corda (D solta): evita o sus2 lá no alto do braço
    "sus2": (X, X, 0, 2, 3, 0),
}
_FAMILIES = ((4, _E_SHAPES), (9, _A_SHAPES), (2, _D_SHAPES))     # (classe de altura da corda solta, formas)


def _movable(root: int, quality: str) -> list[tuple[int, GuitarShape]]:
    """Formas móveis possíveis para o acorde: (casa da fundamental, forma)."""
    out: list[tuple[int, GuitarShape]] = []
    for open_pc, shapes in _FAMILIES:
        rel = shapes.get(quality)
        if rel is None:
            continue
        base = (root - open_pc) % 12
        frets = tuple(None if r is None else r + base for r in rel)
        played = [f for f in frets if f is not None]
        if min(played) < 0:
            continue
        on_base = sum(1 for r in rel if r == 0)
        lowest_is_base = min(r for r in rel if r is not None) == 0
        barre = base if base > 0 and on_base >= 2 and lowest_is_base else None
        out.append((base, GuitarShape(frets, barre)))
    return out


def guitar_shape(root: int, quality: str) -> GuitarShape:
    """Posição sugerida no braço para o acorde (fundamental 0..11, tipo de CHORD_TYPES)."""
    root %= 12
    if quality not in CHORD_TYPES:
        raise KeyError(f"tipo de acorde desconhecido: {quality}")
    if (root, quality) in _OPEN:
        return GuitarShape(_OPEN[(root, quality)])
    return min(_movable(root, quality), key=lambda c: c[0])[1]


def piano_keys(root: int, quality: str, bass: Optional[int] = None) -> list[int]:
    """Teclas (MIDI) do acorde em posição fundamental a partir da oitava 3 (C3 = 48).

    Com baixo invertido (C/E), a nota do baixo entra uma oitava abaixo, na mão esquerda."""
    base = 48 + root % 12
    keys = [base + iv for iv in CHORD_TYPES[quality].intervals]
    if bass is not None and bass % 12 != root % 12:
        keys.insert(0, 36 + bass % 12)
    return keys


def all_voicings() -> dict[str, dict[str, Any]]:
    """Tabela completa para a interface: '<fundamental>:<tipo>' -> violão e piano."""
    table: dict[str, dict[str, Any]] = {}
    for quality in CHORD_TYPES:
        for root in range(12):
            table[f"{root}:{quality}"] = {"guitar": guitar_shape(root, quality).to_dict(),
                                          "piano": piano_keys(root, quality)}
    return table


# --------------------------------------------------------------------------- capotraste

_SHAPE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")     # grafia mais usada em cifras


def shape_name(root: int, quality: str) -> str:
    name = _SHAPE_NAMES[root % 12]
    if root % 12 == 8 and chord_tones(0, quality) & {3}:        # com terça menor escreve-se G#m, não Abm
        name = "G#"
    return name + CHORD_TYPES[quality].suffix


@dataclass(frozen=True)
class CapoSuggestion:
    fret: int                 # casa do capotraste
    shapes: tuple[str, ...]   # como os acordes passam a ser tocados, na ordem em que aparecem
    barre_before: float       # fração do tempo de acordes com pestana, sem capotraste
    barre_after: float        # idem, com o capotraste

    def to_dict(self) -> dict[str, Any]:
        return {"fret": self.fret, "shapes": list(self.shapes),
                "barre_before": round(self.barre_before, 2), "barre_after": round(self.barre_after, 2)}


def suggest_capo(chords: Iterable[ChordSegment], max_fret: int = MAX_CAPO) -> Optional[CapoSuggestion]:
    """Capotraste que troca mais acordes com pestana por formas abertas; None se não compensa.

    Com o capotraste na casa N, cada acorde é tocado com a forma do acorde N semitons abaixo."""
    weight: dict[tuple[int, str], float] = {}
    for c in chords:
        if c.root is None or c.quality not in CHORD_TYPES:
            continue
        key = (c.root % 12, c.quality)
        weight[key] = weight.get(key, 0.0) + max(0.0, c.end - c.start)
    total = sum(weight.values())
    if total <= 0:
        return None

    def hard(capo: int) -> float:
        return sum(w for (r, q), w in weight.items() if not guitar_shape(r - capo, q).open) / total

    def cost(capo: int) -> float:
        return hard(capo) + CAPO_FRET_COST * capo

    before = hard(0)
    best = min(range(1, max_fret + 1), key=lambda n: (round(cost(n), 6), n))
    if before - cost(best) < MIN_CAPO_GAIN:
        return None
    return CapoSuggestion(best, tuple(shape_name(r - best, q) for r, q in weight), before, hard(best))
