from dedilha.music import (chord_label, estimate_key, fmt_time, midi_to_name, name_to_midi, pc_name)
import numpy as np


def test_note_names_match_the_user_examples():
    assert midi_to_name(79) == "G5"
    assert midi_to_name(53) == "F3"
    assert midi_to_name(69) == "A4"
    assert midi_to_name(60) == "C4"          # dó central
    assert midi_to_name(61, flats=True) == "Db4"
    assert midi_to_name(61, flats=False) == "C#4"


def test_name_roundtrip():
    for m in range(24, 100):
        assert name_to_midi(midi_to_name(m)) == m
        assert name_to_midi(midi_to_name(m, flats=True)) == m


def test_chord_labels():
    assert chord_label(0, "maj") == "C"
    assert chord_label(9, "min") == "Am"
    assert chord_label(7, "7") == "G7"
    assert chord_label(10, "min7", flats=True) == "Bbm7"
    assert chord_label(10, "min7", flats=False) == "A#m7"
    assert chord_label(0, "maj", bass=4) == "C/E"
    assert chord_label(0, "maj", bass=0) == "C"


def test_key_estimation_and_spelling():
    c_major = np.zeros(12)
    for pc, w in {0: 5, 2: 2, 4: 3, 5: 2, 7: 4, 9: 2, 11: 1}.items():
        c_major[pc] = w
    k = estimate_key(c_major)
    assert (k.tonic, k.mode) == (0, "major") and k.name() == "C maior"

    bb_major = np.roll(c_major, 10)
    kb = estimate_key(bb_major)
    assert kb.tonic == 10 and kb.uses_flats and kb.short() == "Bb"
    assert pc_name(kb.tonic, kb.uses_flats) == "Bb"
    assert estimate_key(np.zeros(12)) is None


def test_fmt_time():
    assert fmt_time(0) == "00:00" and fmt_time(75.9) == "01:15" and fmt_time(3725) == "1:02:05"
