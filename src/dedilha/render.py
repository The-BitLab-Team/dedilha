"""Saída em texto (terminal/arquivo) do resultado da análise."""

from __future__ import annotations

from typing import Iterable

from .models import AnalysisResult, ChordSegment, InstrumentResult, NoteEvent
from .music import fmt_time


def chord_lines(chords: Iterable[ChordSegment], per_line: int = 4) -> list[str]:
    cells = [f"{fmt_time(c.start)}  {c.label:<7}" for c in chords]
    return ["  " + "   ".join(cells[i: i + per_line]).rstrip() for i in range(0, len(cells), per_line)]


def group_events(notes: list[NoteEvent], tol: float = 0.06) -> list[list[NoteEvent]]:
    """Notas que começam quase juntas (um acorde tocado) viram um evento só."""
    events: list[list[NoteEvent]] = []
    for n in sorted(notes, key=lambda x: (x.start, x.midi)):
        if events and n.start - events[-1][0].start <= tol:
            events[-1].append(n)
        else:
            events.append([n])
    return events


def note_lines(notes: list[NoteEvent], *, phrase_gap: float = 0.8, max_per_line: int = 10) -> list[str]:
    """Frases: uma linha por trecho contínuo, ex.:  [00:12]  A4 B4 C5 B4"""
    lines: list[str] = []
    cur: list[str] = []
    cur_start = 0.0
    last_end = -1e9
    for ev in group_events(notes):
        start = ev[0].start
        text = "+".join(n.name for n in ev)
        if cur and (start - last_end > phrase_gap or len(cur) >= max_per_line):
            lines.append(f"  [{fmt_time(cur_start)}]  " + " ".join(cur))
            cur = []
        if not cur:
            cur_start = start
        cur.append(text)
        last_end = max(n.end for n in ev)
    if cur:
        lines.append(f"  [{fmt_time(cur_start)}]  " + " ".join(cur))
    return lines


def _header(r: AnalysisResult) -> list[str]:
    t = r.track
    lines = [f"♪ {t.display_name()}"]
    bits = []
    if r.key:
        bits.append(f"Tom estimado: {r.key}")
    if r.tempo:
        bits.append(f"Andamento: ~{r.tempo:.0f} BPM")
    if t.duration:
        bits.append(f"Duração: {fmt_time(t.duration)}")
    if abs(r.tuning_cents) >= 20:
        bits.append(f"Afinação: {r.tuning_cents:+.0f} cents (corrigida na análise)")
    if bits:
        lines.append("  " + "  |  ".join(bits))
    if t.resolved_url:
        lines.append(f"  Áudio usado (YouTube): {t.resolved_url}")
    return lines


def _section(inst: InstrumentResult, *, show_notes: bool) -> list[str]:
    title = {"mix": "ACORDES DA MÚSICA", "guitar": "VIOLÃO / GUITARRA", "piano": "PIANO",
             "bass": "BAIXO — FUNDAMENTAL DE CADA ACORDE E LINHA DE BAIXO", "voice": "VOZ — NOTAS CANTADAS"}.get(inst.key, inst.label.upper())
    out = ["", f"== {title} " + "=" * max(3, 60 - len(title))]
    if not inst.available:
        out.append(f"  (indisponível: {inst.reason})")
        return out
    if inst.key == "voice":
        if inst.range_low and inst.range_high:
            out.append(f"  Extensão vocal: {inst.range_low} – {inst.range_high}   ({len(inst.notes)} notas)")
        out.extend(note_lines(inst.notes) if inst.notes else ["  (nenhuma nota detectada)"])
        return out
    if inst.key == "bass":
        out.extend(chord_lines(inst.chords) if inst.chords else ["  (nenhuma nota de baixo detectada)"])
        if inst.range_low and inst.range_high:
            out += ["", f"  Extensão do baixo: {inst.range_low} – {inst.range_high}   "
                        f"({len(inst.notes)} notas)"]
        if inst.notes:
            out += ["", "  Linha de baixo:"] + note_lines(inst.notes)
        return out
    out.extend(chord_lines(inst.chords) if inst.chords else ["  (nenhum acorde detectado)"])
    if show_notes and inst.notes:
        out += ["", f"  Notas ({len(inst.notes)}):"] + note_lines(inst.notes)
    return out


def riff_lines(result: AnalysisResult) -> list[str]:
    out = ["", "== RIFFS — TRECHOS QUE SE REPETEM " + "=" * 28]
    if not result.riffs:
        out += [f"  ({m})" for m in result.riff_messages] or ["  (nenhum riff encontrado)"]
        return out
    for r in result.riffs:
        extra = f", ~{r.beats:g} batidas" if r.beats else ""
        out += ["", f"  {r.label} — {r.instrument_label}  ({len(r.occurrences)}× · {r.duration:.1f} s{extra})",
                f"    Notas: {r.summary}"]
        times = [fmt_time(o.start) + (f" ({o.transpose:+d})" if o.transpose else "") for o in r.occurrences]
        for i in range(0, len(times), 8):
            out.append(("    Aparece em: " if i == 0 else "                ") + "  ".join(times[i: i + 8]))
        if r.tab:
            out += ["    Tablatura (afinação padrão; é uma sugestão de posição):"] + ["      " + ln for ln in r.tab]
    if any(o.transpose for r in result.riffs for o in r.occurrences):
        out += ["", "  (+N / -N = o riff aparece N semitons acima/abaixo da versão mostrada)"]
    return out


def render_text(result: AnalysisResult, *, show_notes: bool = False) -> str:
    lines = _header(result)
    for key in ("mix", "guitar", "piano", "bass", "voice"):
        if key in result.instruments:
            lines += _section(result.instruments[key], show_notes=show_notes)
    if result.riffs_requested:
        lines += riff_lines(result)
    if result.warnings:
        lines += ["", "Avisos:"] + [f"  - {w}" for w in result.warnings]
    return "\n".join(lines) + "\n"
