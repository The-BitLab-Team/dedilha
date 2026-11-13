import librosa
import numpy as np
import pytest
from synth import SR, chord_accuracy, make_song

from dedilha.audio import estimate_tuning, retune
from dedilha.chords import build_vocab, detect_chords, make_edges
from dedilha.music import estimate_key

pytestmark = pytest.mark.slow


def _run(transcriber, song, signal=None, vocabulary="standard", flats=False):
    y = song.mix() if signal is None else signal
    y = (y / np.max(np.abs(y)) * 0.9).astype(np.float32)
    _, beats = librosa.beat.beat_track(y=song.mix(), sr=SR, units="time")
    t = transcriber.transcribe(y)
    return detect_chords(t.posteriorgram, t.frame_times, song.duration, beat_times=beats,
                         vocabulary=vocabulary, flats=flats)


def test_triads_on_full_mix(transcriber, song_ctx):
    a = _run(transcriber, song_ctx)
    assert chord_accuracy(a.segments, song_ctx) >= 0.95
    assert [s.label for s in a.segments[:4]] == ["C", "G", "Am", "F"]
    assert len(a.segments) == 16                      # sem acordes "fantasma" entre as trocas


def test_each_instrument_alone(transcriber, song_ctx):
    for stem in ("guitar", "piano"):
        a = _run(transcriber, song_ctx, song_ctx.stems[stem])
        assert chord_accuracy(a.segments, song_ctx) >= 0.95, stem


def test_seventh_chords(transcriber):
    song = make_song([(0, "maj7"), (9, "min7"), (2, "min7"), (7, "7")], repeats=3, seed=3)
    a = _run(transcriber, song)
    assert chord_accuracy(a.segments, song) >= 0.9


def test_simple_vocabulary_never_outputs_sevenths(transcriber):
    song = make_song([(0, "maj7"), (9, "min7"), (2, "min7"), (7, "7")], repeats=2, seed=3)
    a = _run(transcriber, song, vocabulary="simple")
    assert all(s.quality in ("maj", "min", "") for s in a.segments)


def test_flat_key_spelling_and_key_detection(transcriber):
    song = make_song([(10, "maj"), (7, "min"), (3, "maj"), (5, "maj")], repeats=3, seed=2)
    a = _run(transcriber, song, flats=True)
    assert [s.label for s in a.segments[:4]] == ["Bb", "Gm", "Eb", "F"]
    k = estimate_key(a.chroma_profile)
    assert k.uses_flats


def test_detuned_recording_is_corrected(transcriber):
    song = make_song([(4, "min"), (0, "maj"), (7, "maj"), (2, "maj")], repeats=3, seed=4,
                     detune_cents=35)
    y = song.mix()
    tuning = estimate_tuning(y, SR)
    assert tuning == pytest.approx(0.35, abs=0.08)
    y2, factor = retune(y, SR, tuning)
    t = transcriber.transcribe(y2)
    _, beats = librosa.beat.beat_track(y=y2, sr=SR, units="time")
    a = detect_chords(t.posteriorgram, t.frame_times, len(y2) / SR, beat_times=beats)
    for s in a.segments:
        s.start, s.end = s.start / factor, s.end / factor
    assert chord_accuracy(a.segments, song) >= 0.9


def test_silence_gives_no_chord(transcriber):
    y = np.zeros(SR * 4, dtype=np.float32)
    t = transcriber.transcribe(y)
    a = detect_chords(t.posteriorgram, t.frame_times, 4.0)
    assert all(s.label == "N" for s in a.segments)


def test_edges_and_vocab_helpers():
    e = make_edges([0.5, 1.0, 1.5, 2.0, 2.5], 3.0)
    assert e[0] == 0.0 and e[-1] == 3.0 and np.all(np.diff(e) > 0)
    e2 = make_edges(None, 2.0)
    assert np.allclose(np.diff(e2)[:-1], 0.4)
    v = build_vocab("extended")
    assert len(v.states) == 11 * 12 and np.allclose(np.linalg.norm(v.templates, axis=1), 1.0)
