import pytest

from dedilha.models import ChordSegment
from dedilha.music import CHORD_TYPES, chord_tones
from dedilha.voicings import (GUITAR_TUNING, GuitarShape, all_voicings, guitar_shape, piano_keys, shape_name,
                              suggest_capo)

ALL_CHORDS = [(root, quality) for quality in CHORD_TYPES for root in range(12)]


def song(*chords, seconds=2.0):
    """Acordes em sequência, cada um com `seconds` de duração."""
    return [ChordSegment(i * seconds, (i + 1) * seconds, "?", root, quality, 1.0) for i, (root, quality) in enumerate(chords)]


@pytest.mark.parametrize("root,quality", ALL_CHORDS)
def test_every_guitar_shape_plays_the_right_chord(root, quality):
    shape = guitar_shape(root, quality)
    assert len(shape.frets) == len(GUITAR_TUNING)
    pcs = [m % 12 for m in shape.midi()]
    tones = chord_tones(root, quality)
    assert set(pcs) <= tones                                    # nenhuma nota de fora do acorde
    fifth = {(root + 7) % 12} if len(tones) == 4 else set()     # em acordes de 4 notas a quinta pode faltar
    assert tones - set(pcs) <= fifth
    assert pcs[0] == root                                       # a nota mais grave é a fundamental
    played = [f for f in shape.frets if f is not None]
    fretted = [f for f in played if f > 0]
    assert max(played) <= 12
    assert not fretted or max(fretted) - min(fretted) <= 3      # cabe na mão (4 casas)
    if shape.barre is not None:
        assert min(fretted) == shape.barre and sum(f == shape.barre for f in played) >= 2


@pytest.mark.parametrize("root,quality,text", [
    (0, "maj", "x32010"), (7, "maj", "320003"), (2, "maj", "xx0232"), (9, "min", "x02210"), (4, "min", "022000"),
    (5, "maj", "133211"), (11, "min", "x24432"), (6, "min", "244222"), (10, "maj", "x13331"), (11, "7", "x21202"),
    (4, "5", "022xxx"), (3, "maj", "x68886"),
])
def test_familiar_shapes(root, quality, text):
    assert guitar_shape(root, quality).text == text


def test_open_and_barre_flags():
    assert guitar_shape(0, "maj").open and guitar_shape(0, "maj").barre is None
    f = guitar_shape(5, "maj")
    assert not f.open and f.barre == 1
    assert not guitar_shape(6, "5").open                        # power chord fora da posição aberta: sem cordas soltas
    assert GuitarShape((None, 10, 12, 12, 11, 10)).text == "x-10-12-12-11-10"
    assert guitar_shape(12, "maj") == guitar_shape(0, "maj")    # a fundamental é tratada módulo 12
    with pytest.raises(KeyError):
        guitar_shape(0, "inexistente")


def test_piano_keys():
    assert piano_keys(0, "maj") == [48, 52, 55]                 # C3 E3 G3
    assert piano_keys(9, "min7") == [57, 60, 64, 67]
    assert piano_keys(0, "maj", bass=4) == [40, 48, 52, 55]     # C/E: mi uma oitava abaixo
    assert piano_keys(0, "maj", bass=0) == [48, 52, 55]
    for root, quality in ALL_CHORDS:
        keys = piano_keys(root, quality)
        assert {k % 12 for k in keys} == chord_tones(root, quality) and 48 <= min(keys) and max(keys) <= 71


def test_all_voicings_table():
    table = all_voicings()
    assert len(table) == 12 * len(CHORD_TYPES)
    c = table["0:maj"]
    assert c["guitar"] == {"frets": [None, 3, 2, 0, 1, 0], "barre": None, "open": True, "text": "x32010"}
    assert c["piano"] == [48, 52, 55]


def test_capo_suggested_when_most_chords_need_a_barre():
    s = suggest_capo(song((10, "maj"), (3, "maj"), (5, "maj"), (7, "min")))        # Bb Eb F Gm
    assert s.fret == 3 and s.shapes == ("G", "C", "D", "Em")
    assert s.barre_before == 1.0 and s.barre_after == 0.0
    assert s.to_dict() == {"fret": 3, "shapes": ["G", "C", "D", "Em"], "barre_before": 1.0, "barre_after": 0.0}
    s = suggest_capo(song((11, "maj"), (4, "maj"), (6, "maj"), (8, "min")))        # B E F# G#m
    assert s.fret == 4 and s.shapes == ("G", "C", "D", "Em")


def test_capo_prefers_the_lowest_fret_that_helps():
    s = suggest_capo(song((3, "maj"), (10, "maj"), (0, "min"), (8, "maj")))        # Eb Bb Cm Ab
    assert s.fret == 1 and s.shapes == ("D", "A", "Bm", "G")


def test_no_capo_when_it_barely_helps():
    assert suggest_capo(song((0, "maj"), (7, "maj"), (9, "min"), (5, "maj"))) is None      # C G Am F: só o F tem pestana
    assert suggest_capo(song((6, "min"), (2, "maj"), (9, "maj"), (4, "maj"))) is None      # F#m D A E
    assert suggest_capo(song((7, "maj"), (0, "maj"), (2, "maj"))) is None                  # já é tudo aberto
    assert suggest_capo([]) is None
    assert suggest_capo([ChordSegment(0, 4, "N")]) is None


def test_capo_weighs_chords_by_duration():
    brief_barre = song((0, "maj"), (7, "maj")) + [ChordSegment(4.0, 4.2, "?", 10, "maj", 1.0)]
    assert suggest_capo(brief_barre) is None                    # um Bb de passagem não justifica capotraste
    mostly_barre = [ChordSegment(0, 30, "?", 10, "maj", 1.0), ChordSegment(30, 31, "?", 0, "maj", 1.0)]
    assert suggest_capo(mostly_barre) is not None


def test_shape_names_follow_common_spelling():
    assert shape_name(8, "min") == "G#m" and shape_name(8, "maj") == "Ab" and shape_name(8, "min7") == "G#m7"
    assert shape_name(10, "7") == "Bb7" and shape_name(-1, "maj") == "B"
