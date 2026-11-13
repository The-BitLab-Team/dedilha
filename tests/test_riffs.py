import librosa
import numpy as np
import pytest
import soundfile as sf
from conftest import GroundTruthSeparator
from synth import RIFF_120, make_riff_song

from dedilha.audio import ANALYSIS_SR
from dedilha.cli import _parse_instruments
from dedilha.models import NoteEvent
from dedilha.music import midi_to_name
from dedilha.pipeline import AnalysisOptions, Analyzer
from dedilha.render import render_text
from dedilha.riffs import build_events, find_riffs
from dedilha.tab import make_tab


def note(start, dur, midi, strength=0.8):
    return NoteEvent(start, start + dur, midi, midi_to_name(midi), strength)


def riff_at(t0, shift=0, pattern=RIFF_120):
    return [note(t0 + o, d * 0.9, m + shift) for o, d, m in pattern]


def melody(t0, count, seed, pitches=(52, 55, 57, 59, 62, 64), steps=(0.25, 0.5, 0.75)):
    rng = np.random.default_rng(seed)
    out, t = [], t0
    for _ in range(count):
        out.append(note(t, 0.2, int(rng.choice(pitches))))
        t += float(rng.choice(steps))
    return out, t


def find(notes, instrument="guitar", **kw):
    return find_riffs(notes, instrument, "Guitarra", **kw)


# ------------------------------------------------------------------ o que É riff

def test_finds_riff_and_its_transposed_repeat():
    notes = [n for i in range(4) for n in riff_at(i * 2.0)]
    verse, t = melody(10.0, 40, seed=3)
    notes += verse
    notes += [n for i in range(4) for n in riff_at(t + 2 + i * 2.0, shift=5)]
    riffs = find(notes, bpm=120)
    assert len(riffs) == 1
    r = riffs[0]
    assert r.summary == "E2 G2 A2 A#2 A2 G2"
    assert [o.transpose for o in r.occurrences] == [0] * 4 + [5] * 4
    assert [o.start for o in r.occurrences[:4]] == pytest.approx([0, 2, 4, 6])
    assert r.beats == pytest.approx(4.0, abs=0.5)
    assert [n.offset for n in r.notes] == pytest.approx([0, 0.25, 0.5, 0.75, 1.0, 1.5])
    assert r.coverage > 0.5 and r.tab and len(r.tab) == 6


def test_tolerates_small_timing_jitter():
    rng = np.random.default_rng(0)
    notes = []
    for i in range(6):
        for o, d, m in RIFF_120:
            notes.append(note(i * 2.0 + o + float(rng.normal(0, 0.015)), d * 0.9, m))
    assert len(find(notes)) == 1


def test_rhythmic_chug_on_power_chords_is_a_riff():
    pat = [(0, 0.25, 40), (0.25, 0.25, 40), (0.5, 0.5, 40), (1.0, 0.25, 43), (1.25, 0.25, 40), (1.5, 0.5, 45)]
    notes = []
    for rep in range(8):
        for o, d, m in pat:
            notes += [note(rep * 2.0 + o, d * 0.9, m), note(rep * 2.0 + o + 0.005, d * 0.9, m + 7)]
    riffs = find(notes)
    assert len(riffs) == 1 and riffs[0].summary.startswith("E2+B2 E2+B2")
    assert len(riffs[0].occurrences) == 8


def test_four_note_pattern_needs_four_repeats():
    pat = [(0, 0.25, 40), (0.25, 0.25, 43), (0.5, 0.25, 47), (0.75, 0.5, 45)]
    for repeats, expected in ((3, 0), (5, 1)):
        notes, t = melody(0.0, 30, seed=7)
        for i in range(repeats):
            notes += riff_at(t + 3 + i * 5.0, pattern=pat)
            filler, _ = melody(t + 3 + i * 5.0 + 1.2, 6, seed=20 + i, pitches=(60, 62, 65))
            notes += filler
        got = find(notes)
        assert len(got) == expected, (repeats, [r.summary for r in got])


# ------------------------------------------------------------------ o que NÃO é riff

def test_strummed_chords_are_not_riffs():
    prog = [[48, 55, 60, 64, 67], [43, 50, 55, 59, 62], [45, 52, 57, 60, 64], [41, 48, 53, 57, 60]]
    slow, fast = [], []
    for bar in range(16):
        ch = prog[bar % 4]
        for half in range(2):                                  # dois golpes longos por compasso
            slow += [note(bar * 2.4 + half * 1.2 + j * 0.012, 1.0, m) for j, m in enumerate(ch)]
        for k in range(8):                                     # colcheias: 8 golpes por compasso
            fast += [note(bar * 2.0 + k * 0.25 + j * 0.01, 0.2, m) for j, m in enumerate(ch)]
    assert find(slow) == [] and find(fast) == []


def test_repeated_note_trill_and_sustained_bass_are_not_riffs():
    assert find([note(i * 0.25, 0.2, 40) for i in range(80)]) == []
    assert find([note(i * 0.2, 0.18, 60 + (i % 2) * 2) for i in range(80)]) == []
    assert find([note(bar * 2.0, 1.9, [36, 43, 45, 41][bar % 4]) for bar in range(32)]) == []


def test_random_melodies_do_not_produce_riffs():
    for seed in range(8):
        notes, _ = melody(0.0, 500, seed=seed, pitches=(52, 55, 57, 59, 62, 64, 67))
        assert find(notes) == [], seed


def test_pattern_does_not_cross_long_pauses():
    # as 6 notas só se repetem se atravessarem uma pausa de 3 s no meio: não vale
    notes = []
    for i in range(6):
        t = i * 10.0
        notes += [note(t, 0.2, 40), note(t + 0.25, 0.2, 43), note(t + 0.5, 0.2, 45),
                  note(t + 3.5, 0.2, 46), note(t + 3.75, 0.2, 45), note(t + 4.0, 0.2, 43)]
    assert find(notes) == []


# ------------------------------------------------------------------ detalhes

def test_harmonic_ghosts_are_removed_for_guitar_but_not_piano():
    notes = []
    for i in range(4):
        for o, d, m in RIFF_120:
            notes += [note(i * 2.0 + o, d * 0.9, m, 0.8), note(i * 2.0 + o, d * 0.9, m + 12, 0.4),
                      note(i * 2.0 + o, d * 0.9, m + 24, 0.3)]
    assert find(notes)[0].summary == "E2 G2 A2 A#2 A2 G2"
    assert all(len(e.notes) == 3 for e in build_events(notes, drop_harmonics=False))
    assert all(len(e.notes) == 1 for e in build_events(notes, drop_harmonics=True))


def test_high_line_finds_a_lead_over_a_changing_bass():
    # melodia aguda repetida por cima de um baixo que muda a cada compasso
    lead = [(0.0, 0.25, 72), (0.25, 0.25, 76), (0.5, 0.25, 79), (0.75, 0.25, 76), (1.0, 0.5, 74), (1.5, 0.5, 72)]
    notes = []
    for bar in range(8):
        notes += riff_at(bar * 2.0, pattern=lead)
        notes.append(note(bar * 2.0, 0.2, [36, 41, 43, 38][bar % 4]))
    riffs = find(notes, "piano")
    assert len(riffs) == 1 and len(riffs[0].occurrences) == 8
    assert riffs[0].summary.split()[1:] == ["E5", "G5", "E5", "D5", "C5"]


def test_labels_are_left_to_the_pipeline_and_no_bpm_means_no_beats():
    notes = [n for i in range(4) for n in riff_at(i * 2.0)]
    r = find(notes)[0]
    assert r.id == "" and r.label == "" and r.beats is None


# ------------------------------------------------------------------ tablatura

def test_tab_picks_low_positions_and_keeps_chord_notes_on_different_strings():
    assert make_tab("guitar", [[40], [43], [45], [46]]) == [
        "e|------------|", "B|------------|", "G|------------|", "D|------------|",
        "A|-------0--1-|", "E|-0--3-------|"]
    assert make_tab("guitar", [[40, 47], [43, 50]]) == [          # power chords, uma nota por corda
        "e|------|", "B|------|", "G|------|", "D|----0-|", "A|-2----|", "E|-0--3-|"]


def test_tab_for_bass_and_unplayable_notes():
    assert make_tab("bass", [[28], [33], [38], [43]]) == [
        "G|----------0-|", "D|-------0----|", "A|----0-------|", "E|-0----------|"]
    assert make_tab("guitar", [[30], [31], [32], [33]]) is None       # abaixo da corda mi grave
    assert make_tab("piano", [[60]]) is None


# ------------------------------------------------------------------ com áudio

@pytest.fixture(scope="module")
def riff_audio():
    def build(**kw):
        song = make_riff_song(**kw)
        g = librosa.resample(song.stems["guitar"].astype("float32"), orig_sr=song.sr, target_sr=ANALYSIS_SR)
        return song, g / np.max(np.abs(g)) * 0.9
    return build


@pytest.mark.slow
def test_audio_guitar_riff_is_found_at_the_right_times(riff_audio, transcriber):
    song, g = riff_audio()
    riffs = find_riffs(transcriber.transcribe(g).notes, "guitar", "Guitarra", bpm=song.bpm)
    assert len(riffs) == 1
    r = riffs[0]
    assert r.summary == "E2 G2 A2 A#2 A2 G2"
    got = [o.start for o in r.occurrences]
    assert len(got) == len(song.starts)
    assert max(abs(a - b) for a, b in zip(got, song.starts)) < 0.15
    assert not any(o.transpose for o in r.occurrences)
    assert r.beats == pytest.approx(4.0, abs=0.5)


@pytest.mark.slow
def test_audio_transposed_repeat_is_grouped_with_the_original(riff_audio, transcriber):
    song, g = riff_audio(transpose_b=5)
    riffs = find_riffs(transcriber.transcribe(g).notes, "guitar", "Guitarra")
    assert len(riffs) == 1
    shifts = sorted(o.transpose for o in riffs[0].occurrences)
    assert len(shifts) == 8 and len(set(shifts)) == 2 and abs(max(shifts) - min(shifts)) == 5


@pytest.mark.slow
def test_pipeline_delivers_riffs_json_and_text(tmp_path, transcriber):
    song = make_riff_song()
    wav = tmp_path / "riff.wav"
    sf.write(str(wav), song.mix(), song.sr)
    an = Analyzer(separator=GroundTruthSeparator(song), transcriber=transcriber)
    res = an.analyze(str(wav), tmp_path, AnalysisOptions(instruments=("riffs",)))

    assert res.riffs_requested and not res.instruments.keys() - {"mix"}
    guitar = [r for r in res.riffs if r.instrument == "guitar"]
    assert len(guitar) == 1 and guitar[0].label.startswith("Riff ") and guitar[0].id.startswith("riff-")
    assert len(guitar[0].occurrences) == 8 and guitar[0].summary == "E2 G2 A2 A#2 A2 G2"
    assert guitar[0].beats == pytest.approx(4.0, abs=0.5)
    assert all(r.instrument != "bass" for r in res.riffs)             # o baixo só sustenta notas

    d = res.to_dict()
    assert d["riffs_requested"] and d["riffs"][0]["occurrences"] and d["riffs"][0]["notes"]
    text = render_text(res)
    assert "RIFFS" in text and "E2 G2 A2 A#2 A2 G2" in text and "Aparece em:" in text and "E|-0-" in text


@pytest.mark.slow
def test_no_riffs_in_plain_chord_song_and_message_without_separation(tmp_path, song_ctx, transcriber):
    wav = tmp_path / "acordes.wav"
    sf.write(str(wav), song_ctx.mix(), song_ctx.sr)
    res = Analyzer(separator=GroundTruthSeparator(song_ctx), transcriber=transcriber).analyze(
        str(wav), tmp_path, AnalysisOptions(instruments=("guitar", "piano", "bass", "riffs")))
    assert res.riffs == [] and res.riff_messages and "Não encontrei" in res.riff_messages[0]
    assert "RIFFS" in render_text(res)

    res2 = Analyzer(transcriber=transcriber).analyze(
        str(wav), tmp_path, AnalysisOptions(separate=False, instruments=("riffs",)))
    assert res2.riffs == [] and "separar os instrumentos" in res2.riff_messages[0]


def test_cli_accepts_riffs():
    assert _parse_instruments("riffs") == ("riffs",)
    assert _parse_instruments("riff, baixo") == ("riffs", "bass")
    assert _parse_instruments("tudo")[-1] == "riffs"
