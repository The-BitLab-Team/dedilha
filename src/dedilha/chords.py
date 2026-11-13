"""Reconhecimento de acordes a partir do posteriorgrama de notas do basic-pitch.

Ideia: o basic-pitch já separa "o que é nota" de "o que é harmônico/ruído" melhor que um
cromagrama clássico. Dobramos as 88 notas em 12 classes de altura (chroma), agrupamos por
meio-tempo (grade de batidas), comparamos com gabaritos de acordes e suavizamos com Viterbi
(acordes mudam pouco). O baixo (notas graves) ajuda a decidir a fundamental.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .models import ChordSegment
from .music import CHORD_TYPES, VOCABULARIES, chord_label, chord_tones, pc_name

_PC_OF_ROW = (21 + np.arange(88)) % 12                 # classe de altura de cada linha (A0 = 21)
_FOLD = np.zeros((88, 12), dtype=np.float32)
_FOLD[np.arange(88), _PC_OF_ROW] = 1.0
_BASS_ROWS = 31                                         # linhas 0..30  =  MIDI 21..51 (até D#3)


def fold_chroma(posteriorgram: np.ndarray, floor: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """(frames,88) -> chroma total (frames,12) e chroma do baixo (frames,12)."""
    p = np.clip(posteriorgram - floor, 0.0, None)
    return p @ _FOLD, p[:, :_BASS_ROWS] @ _FOLD[:_BASS_ROWS]


def make_edges(beat_times: Optional[Sequence[float]], duration: float, *,
               subdivide: int = 2, fallback_hop: float = 0.4, min_seg: float = 0.12) -> np.ndarray:
    """Limites dos segmentos: grade de batidas subdividida; sem batidas, janelas fixas."""
    if duration <= 0:
        return np.array([0.0, 0.0])
    b = np.asarray(beat_times if beat_times is not None else [], dtype=float)
    b = b[(b > 0.05) & (b < duration - 0.05)]
    if len(b) < 4:
        edges = np.arange(0.0, duration, fallback_hop)
    else:
        grid = np.concatenate([[0.0], b])
        parts = [grid]
        for k in range(1, subdivide):
            parts.append(grid[:-1] + (grid[1:] - grid[:-1]) * k / subdivide)
        # último intervalo (depois da última batida) usa o período mediano
        edges = np.sort(np.concatenate(parts))
        last_period = float(np.median(np.diff(b)))
        t = b[-1] + last_period / subdivide
        tail = []
        while t < duration:
            tail.append(t)
            t += last_period / subdivide
        edges = np.concatenate([edges, tail])
    edges = np.concatenate([edges, [duration]])
    # remove segmentos minúsculos
    keep = [edges[0]]
    for e in edges[1:]:
        if e - keep[-1] >= min_seg or e == edges[-1]:
            if e == edges[-1] and e - keep[-1] < min_seg and len(keep) > 1:
                keep[-1] = e
            else:
                keep.append(e)
    return np.asarray(keep)


def _aggregate(frames: np.ndarray, frame_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Média de `frames` (T,12) em cada intervalo [edges[k], edges[k+1])."""
    n = len(frames)
    idx = np.searchsorted(frame_times, edges)
    lo = np.clip(idx[:-1], 0, max(n - 1, 0))
    hi = np.clip(np.maximum(idx[1:], lo + 1), 1, n)
    cs = np.vstack([np.zeros((1, frames.shape[1])), np.cumsum(frames, axis=0)])
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None]


@dataclass
class _Vocab:
    states: list[tuple[int, str]]    # (fundamental, tipo)
    templates: np.ndarray            # (S,12) normalizados em L2
    bias: np.ndarray                 # (S,)


def build_vocab(name: str = "standard") -> _Vocab:
    qualities = VOCABULARIES[name]
    states, rows, bias = [], [], []
    for q in qualities:
        ct = CHORD_TYPES[q]
        for root in range(12):
            t = np.zeros(12)
            for iv, w in zip(ct.intervals, ct.weights):
                t[(root + iv) % 12] = w
            states.append((root, q))
            rows.append(t / np.linalg.norm(t))
            bias.append(ct.bias)
    return _Vocab(states, np.asarray(rows), np.asarray(bias))


def _viterbi(emission: np.ndarray, stay: float) -> np.ndarray:
    """Caminho de máxima probabilidade; mudar de acorde custa log((1-stay)/(S-1))."""
    k_len, s_len = emission.shape
    log_stay = np.log(stay)
    log_move = np.log((1.0 - stay) / (s_len - 1))
    delta = emission[0].copy()
    psi = np.zeros((k_len, s_len), dtype=np.int32)
    ar = np.arange(s_len)
    for k in range(1, k_len):
        best = int(np.argmax(delta))
        stay_sc = delta + log_stay
        move_sc = delta[best] + log_move
        choose_stay = stay_sc >= move_sc
        psi[k] = np.where(choose_stay, ar, best)
        delta = np.where(choose_stay, stay_sc, move_sc) + emission[k]
    path = np.zeros(k_len, dtype=np.int32)
    path[-1] = int(np.argmax(delta))
    for k in range(k_len - 1, 0, -1):
        path[k - 1] = psi[k, path[k]]
    return path


@dataclass
class ChordAnalysis:
    segments: list[ChordSegment]
    chroma_profile: np.ndarray        # (12,) energia média por classe de altura


def detect_chords(posteriorgram: np.ndarray, frame_times: np.ndarray, duration: float, *,
                  beat_times: Optional[Sequence[float]] = None,
                  external_bass: Optional[tuple[np.ndarray, np.ndarray]] = None,
                  vocabulary: str = "standard", flats: bool = False, slash: bool = False,
                  beta: float = 10.0, stay: float = 0.80, bass_weight: float = 0.15,
                  compress: float = 0.5, min_duration: float = 0.45,
                  silence_ratio: float = 0.08) -> ChordAnalysis:
    """Linha do tempo de acordes.

    external_bass: (posteriorgrama_do_baixo, tempos) de um stem de baixo, quando houver.
    """
    if posteriorgram.shape[0] == 0 or duration <= 0:
        return ChordAnalysis([], np.zeros(12))

    chroma, bass_from_signal = fold_chroma(posteriorgram)
    profile = chroma.mean(axis=0)
    edges = make_edges(beat_times, duration)
    seg = _aggregate(chroma, frame_times, edges)                       # (K,12)
    if external_bass is not None:
        bpost, btimes = external_bass
        bass_frames = fold_chroma(bpost)[0]
        bseg = _aggregate(bass_frames, btimes, edges)
    else:
        bseg = _aggregate(bass_from_signal, frame_times, edges)
    bsum = bseg.sum(axis=1, keepdims=True)
    bseg_n = np.divide(bseg, bsum, out=np.zeros_like(bseg), where=bsum > 1e-6)

    energy = seg.sum(axis=1)
    ref = np.percentile(energy, 90) if len(energy) else 0.0
    silent = energy < silence_ratio * max(ref, 1e-9)

    # compressão de amplitude: evita que uma fundamental dobrada em várias oitavas "esmague"
    # as demais notas do acorde
    seg_c = np.power(seg, compress)
    norm = np.linalg.norm(seg_c, axis=1, keepdims=True)
    seg_n = np.divide(seg_c, norm, out=np.zeros_like(seg_c), where=norm > 1e-9)

    vocab = build_vocab(vocabulary)
    cos = seg_n @ vocab.templates.T                                    # (K,S)
    roots = np.array([r for r, _ in vocab.states])
    score = cos - vocab.bias[None, :] + bass_weight * bseg_n[:, roots]

    n_idx = len(vocab.states)                                          # estado "N"
    emission = np.concatenate([score, np.where(silent, 1.0, 0.0)[:, None]], axis=1) * beta
    emission[silent, :n_idx] = -1e3
    path = _viterbi(emission, stay)

    # agrupa segmentos consecutivos com o mesmo estado
    groups: list[dict] = []
    for k, s in enumerate(path):
        dur = float(edges[k + 1] - edges[k])
        c = 0.0 if s == n_idx else float(np.clip(cos[k, s], 0.0, 1.0))
        if groups and groups[-1]["state"] == s:
            g = groups[-1]
            g["end"] = float(edges[k + 1])
            g["conf_w"] += c * dur
            g["rows"].append(k)
        else:
            groups.append({"state": int(s), "start": float(edges[k]), "end": float(edges[k + 1]),
                           "conf_w": c * dur, "rows": [k]})

    groups = _merge_short(groups, min_duration)

    segments: list[ChordSegment] = []
    for g in groups:
        dur = max(g["end"] - g["start"], 1e-6)
        if g["state"] == n_idx:
            segments.append(ChordSegment(g["start"], g["end"], "N", None, "", 0.0))
            continue
        root, qual = vocab.states[g["state"]]
        bass_pc = None
        if slash:
            bvec = bseg_n[g["rows"]].mean(axis=0)
            b = int(np.argmax(bvec))
            if b != root and bvec[b] >= 0.55 and b in chord_tones(root, qual):
                bass_pc = b
        segments.append(ChordSegment(g["start"], g["end"], chord_label(root, qual, flats, bass_pc),
                                     root, qual, float(g["conf_w"] / dur)))
    return ChordAnalysis(_merge_equal(segments), profile)


def _merge_short(groups: list[dict], min_duration: float) -> list[dict]:
    """Funde acordes mais curtos que `min_duration` no vizinho mais longo."""
    while len(groups) > 1:
        durs = [g["end"] - g["start"] for g in groups]
        i = int(np.argmin(durs))
        if durs[i] >= min_duration:
            break
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = i - 1 if durs[i - 1] >= durs[i + 1] else i + 1
        a, b = groups[j], groups[i]
        a["start"], a["end"] = min(a["start"], b["start"]), max(a["end"], b["end"])
        a["rows"] = sorted(a["rows"] + b["rows"])
        del groups[i]
        # fusão de vizinhos iguais
        k = 0
        while k < len(groups) - 1:
            if groups[k]["state"] == groups[k + 1]["state"]:
                x, y = groups[k], groups[k + 1]
                x["end"] = y["end"]
                x["conf_w"] += y["conf_w"]
                x["rows"] = sorted(x["rows"] + y["rows"])
                del groups[k + 1]
            else:
                k += 1
    return groups


def _merge_equal(segs: list[ChordSegment]) -> list[ChordSegment]:
    out: list[ChordSegment] = []
    for s in segs:
        if out and out[-1].label == s.label:
            prev = out[-1]
            d0, d1 = prev.end - prev.start, s.end - s.start
            prev.confidence = (prev.confidence * d0 + s.confidence * d1) / max(d0 + d1, 1e-9)
            prev.end = s.end
        else:
            out.append(s)
    return out


def relabel(segments: list[ChordSegment], flats: bool) -> None:
    """Reescreve os rótulos com a grafia certa (bemóis/sustenidos) depois de descobrir o tom."""
    for s in segments:
        if s.root is None:
            continue
        slash = None
        if "/" in s.label:
            bass_name = s.label.split("/", 1)[1]
            for pc in range(12):
                if bass_name in (pc_name(pc, False), pc_name(pc, True)):
                    slash = pc
                    break
        s.label = chord_label(s.root, s.quality, flats, slash)
