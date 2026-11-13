import librosa
import numpy as np
import pytest

from dedilha.audio import ANALYSIS_SR
from dedilha.bass import bass_range, bass_roots, clean_bass_line
from dedilha.models import ChordSegment, NoteEvent
from dedilha.music import midi_to_name, pc_name


def note(midi, start, end, strength=0.7):
    return NoteEvent(start, end, midi, midi_to_name(midi), strength)


def test_drops_notes_outside_bass_range_and_too_short():
    line = clean_bass_line([note(12, 0, 1), note(40, 0, 1), note(80, 2, 3), note(43, 3, 3.05)])
    assert [n.midi for n in line] == [40]


def test_drops_weaker_octave_ghosts_but_keeps_real_notes():
    line = clean_bass_line([note(40, 0, 1, 0.8), note(52, 0, 1, 0.4),      # harmônico 1 oitava acima
                            note(59, 0, 1, 0.3),                           # oitava + quinta acima
                            note(45, 1, 2, 0.8), note(33, 1, 2, 0.3)])      # "oitava abaixo" fantasma
    assert [n.midi for n in line] == [40, 45]


def test_keeps_strong_upper_note_when_it_is_the_one_played():
    line = clean_bass_line([note(40, 0, 1, 0.3), note(52, 0, 1, 0.8)])
    assert [n.midi for n in line] == [52]


def test_monophonic_and_legato():
    line = clean_bass_line([note(40, 0, 1.0, 0.7), note(43, 0.9, 2.0, 0.7)])
    assert [n.midi for n in line] == [40, 43] and line[0].end == pytest.approx(0.9)
    both = clean_bass_line([note(40, 0, 1, 0.9), note(43, 0.05, 1, 0.3)])      # ao mesmo tempo
    assert [n.midi for n in both] == [40]


def test_fragment_next_to_same_note_is_dropped_but_repeated_notes_stay():
    frag = clean_bass_line([note(43, 2.23, 2.38, 0.25), note(43, 2.39, 3.6, 0.7)])
    assert len(frag) == 1 and frag[0].start == pytest.approx(2.39)
    repeated = clean_bass_line([note(43, 0, 0.5, 0.7), note(43, 0.51, 1.0, 0.7)])
    assert len(repeated) == 2


def test_does_not_mutate_input_and_respects_flats():
    original = [note(46, 0, 1)]
    out = clean_bass_line(original, flats=True)
    assert out[0].name == "Bb2" and original[0].name == "A#2"


def test_roots_per_chord_weighted_and_merged():
    notes = [note(40, 0, 1.8, 0.8), note(43, 1.8, 2.0, 0.8), note(45, 2.0, 4.0, 0.8)]
    segs = [ChordSegment(0, 2, "C"), ChordSegment(2, 3, "Am"), ChordSegment(3, 4, "Am"),
            ChordSegment(4, 5, "N")]
    roots = bass_roots(notes, segs)
    assert [(r.label, r.start, r.end) for r in roots] == [("E", 0, 2), ("A", 2, 4)]
    assert all(r.quality == "baixo" and 0 < r.confidence <= 1 for r in roots)
    assert bass_roots(notes, [ChordSegment(10, 11, "C")]) == []


def test_range():
    assert bass_range([]) == (None, None)
    assert bass_range([note(45, 0, 1), note(33, 1, 2), note(52, 2, 3)]) == ("A1", "E3")


@pytest.mark.slow
def test_bass_line_from_synthetic_bass_stem(song_ctx, transcriber):
    y = librosa.resample(song_ctx.stems["bass"].astype("float32"), orig_sr=song_ctx.sr,
                         target_sr=ANALYSIS_SR)
    y = (y * 0.9 / np.max(np.abs(y))).astype("float32")
    line = clean_bass_line(transcriber.transcribe(y).notes)
    truth = [(s + h * 2 * 60 / song_ctx.bpm, 36 + r % 12)
             for s, _, r, _ in song_ctx.chords for h in range(2)]
    hits = sum(any(n.midi == m and abs(n.start - s) < 0.15 for n in line) for s, m in truth)
    assert hits / len(truth) >= 0.95
    assert len(line) <= 1.15 * len(truth)                      # poucas notas sobrando
    assert {pc_name(n.midi % 12) for n in line} <= {"C", "G", "A", "F"}
