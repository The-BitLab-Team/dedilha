import numpy as np
import pytest
from synth import make_song, voice_note

from dedilha.music import midi_to_name
from dedilha.vocals import extract_vocal_notes, segment_notes, vocal_range

pytestmark = pytest.mark.slow


def test_melody_is_recovered(song_ctx):
    y = song_ctx.stems["vocals"]
    y = (y / np.max(np.abs(y)) * 0.9).astype(np.float32)
    va = extract_vocal_notes(y)
    hits = 0
    for s, e, m in song_ctx.melody:
        if any(n.midi == m and n.start < e - 0.05 and n.end > s + 0.05 for n in va.notes):
            hits += 1
    assert hits / len(song_ctx.melody) >= 0.95
    extras = [n for n in va.notes if not any(s - 0.1 < n.start < e for s, e, _ in song_ctx.melody)]
    assert len(extras) <= 2
    assert va.range_low and va.range_high


def test_single_notes_with_vibrato():
    sr = 22050
    for midi in (53, 69, 79):                       # F3, A4, G5 (exemplos do pedido)
        y = np.concatenate([np.zeros(sr // 4), 0.8 * voice_note(midi, 1.2, sr), np.zeros(sr // 4)]).astype(np.float32)
        va = extract_vocal_notes(y)
        assert [n.name for n in va.notes] == [midi_to_name(midi)]


def test_segmentation_ignores_vibrato_and_glitches():
    t = np.arange(100) * 0.02
    midi = 60 + 0.4 * np.sin(2 * np.pi * 5.5 * t)
    midi[40] = 72                                    # glitch de oitava de 1 quadro
    notes = segment_notes(t, midi)
    assert [n.midi for n in notes] == [60]


def test_two_pitches_split():
    t = np.arange(100) * 0.02
    midi = np.where(t < 1.0, 60.0, 64.0)
    assert [n.midi for n in segment_notes(t, midi)] == [60, 64]


def test_silence_and_range():
    assert extract_vocal_notes(np.zeros(22050 * 2, dtype=np.float32)).notes == []
    assert vocal_range([]) == (None, None)
