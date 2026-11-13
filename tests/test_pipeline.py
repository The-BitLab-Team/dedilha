import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import GroundTruthSeparator
from synth import chord_accuracy

from dedilha.cli import main as cli_main
from dedilha.music import pc_name
from dedilha.pipeline import AnalysisOptions, Analyzer
from dedilha.render import render_text
from dedilha.separation import SeparationError
from dedilha.sources import InputError

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def wav(tmp_path_factory, song_ctx):
    p = tmp_path_factory.mktemp("audio") / "musica de teste.wav"
    sf.write(str(p), song_ctx.mix(), song_ctx.sr)
    return p


def test_full_pipeline_with_stems(tmp_path, song_ctx, wav, transcriber):
    stages = []
    an = Analyzer(separator=GroundTruthSeparator(song_ctx), transcriber=transcriber)
    res = an.analyze(str(wav), tmp_path, AnalysisOptions(), lambda s, f, m: stages.append((s, f)))

    assert res.separated and res.key == "C maior" and res.key_short == "C"
    assert res.tempo == pytest.approx(100, abs=4)
    assert res.track.title == "musica de teste" and res.track.duration == pytest.approx(song_ctx.duration, abs=0.5)
    assert len(res.waveform) == 240

    assert chord_accuracy(res.instruments["mix"].chords, song_ctx) >= 0.95
    assert chord_accuracy(res.instruments["guitar"].chords, song_ctx) >= 0.95
    assert chord_accuracy(res.instruments["piano"].chords, song_ctx) >= 0.95
    assert res.instruments["guitar"].notes and res.instruments["piano"].notes
    voice = res.instruments["voice"]
    hits = sum(any(n.midi == m and n.start < e - 0.05 and n.end > s + 0.05 for n in voice.notes)
               for s, e, m in song_ctx.melody)
    assert hits / len(song_ctx.melody) >= 0.95
    assert voice.range_low and voice.range_high

    bass = res.instruments["bass"]
    assert bass.available and bass.notes and bass.range_low and bass.range_high
    ok = tot = 0
    for s, e, r, _ in song_ctx.chords:
        mid = (s + e) / 2
        got = next((c.label for c in bass.chords if c.start <= mid < c.end), None)
        ok += got == pc_name(r)
        tot += 1
    assert ok / tot >= 0.9                                         # fundamental certa em cada acorde

    fr = [f for _, f in stages]
    assert fr == sorted(fr) and fr[-1] == 1.0                      # progresso só avança
    assert {"input", "separate", "transcribe", "done"} <= {s for s, _ in stages}

    d = json.loads(json.dumps(res.to_dict()))                      # serializável
    assert set(d["instruments"]) == {"mix", "guitar", "piano", "bass", "voice"}
    assert d["riffs_requested"] and d["riffs"] == [] and d["riff_messages"]   # só acordes batidos: sem riff


def test_without_separation_only_mix(tmp_path, wav, transcriber):
    res = Analyzer(transcriber=transcriber).analyze(
        str(wav), tmp_path, AnalysisOptions(separate=False))
    assert not res.separated
    assert res.instruments["mix"].chords
    assert not res.instruments["guitar"].available and not res.instruments["voice"].available
    assert not res.instruments["bass"].available
    assert any("desligada" in w for w in res.warnings)


def test_voice_from_mix_option(tmp_path, wav, transcriber):
    res = Analyzer(transcriber=transcriber).analyze(
        str(wav), tmp_path, AnalysisOptions(separate=False, instruments=("voice",), voice_from_mix=True))
    assert res.instruments["voice"].available


def test_separation_failure_degrades_gracefully(tmp_path, wav, transcriber):
    class Broken:
        name = "broken"

        def separate(self, *a, **k):
            raise SeparationError("sem internet")

    res = Analyzer(separator=Broken(), transcriber=transcriber).analyze(str(wav), tmp_path, AnalysisOptions())
    assert res.instruments["mix"].chords and not res.instruments["piano"].available
    assert any("sem internet" in w for w in res.warnings)


def test_absent_instrument_is_flagged(tmp_path, song_ctx, wav, transcriber):
    class NoPiano(GroundTruthSeparator):
        def separate(self, wav_path, stems, progress):
            out = super().separate(wav_path, stems, progress)
            out["piano"] = out["piano"] * 1e-4
            return out

    res = Analyzer(separator=NoPiano(song_ctx), transcriber=transcriber).analyze(
        str(wav), tmp_path, AnalysisOptions(instruments=("piano",)))
    assert not res.instruments["piano"].available and "Quase não há" in res.instruments["piano"].reason


def test_detuned_file_timestamps_stay_correct(tmp_path, transcriber):
    from synth import make_song

    song = make_song([(4, "min"), (0, "maj"), (7, "maj"), (2, "maj")], repeats=3, seed=4, detune_cents=-35)
    p = tmp_path / "detune.wav"
    sf.write(str(p), song.mix(), song.sr)
    res = Analyzer(transcriber=transcriber).analyze(
        str(p), tmp_path / "w", AnalysisOptions(separate=False, instruments=()))
    assert res.tuning_cents == pytest.approx(-35, abs=8)
    assert chord_accuracy(res.instruments["mix"].chords, song) >= 0.9
    assert res.key_short == "G"                                  # Em-C-G-D  ->  G maior
    assert res.tempo == pytest.approx(100, abs=4)


def test_input_errors_are_friendly(tmp_path):
    with pytest.raises(InputError):
        Analyzer().analyze(str(tmp_path / "nao_existe.mp3"), tmp_path / "w")
    junk = tmp_path / "lixo.mp3"
    junk.write_bytes(b"isso nao e um mp3" * 100)
    with pytest.raises(InputError):
        Analyzer().analyze(str(junk), tmp_path / "w2")


def test_max_minutes_limit(tmp_path, wav):
    with pytest.raises(InputError, match="limite"):
        Analyzer().analyze(str(wav), tmp_path, AnalysisOptions(max_minutes=0.1))


def test_text_rendering(tmp_path, song_ctx, wav, transcriber):
    res = Analyzer(separator=GroundTruthSeparator(song_ctx), transcriber=transcriber).analyze(
        str(wav), tmp_path, AnalysisOptions())
    txt = render_text(res, show_notes=True)
    assert "ACORDES DA MÚSICA" in txt and "VIOLÃO / GUITARRA" in txt and "PIANO" in txt
    assert "Tom estimado: C maior" in txt and "00:00  C" in txt and "Am" in txt
    assert "Extensão vocal" in txt


def test_cli_end_to_end(tmp_path, wav, capsys):
    out = tmp_path / "r.json"
    code = cli_main(["analyze", str(wav), "--no-separation", "--json", str(out)])
    captured = capsys.readouterr()
    assert code == 0 and "ACORDES DA MÚSICA" in captured.out
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["instruments"]["mix"]["chords"][0]["label"] == "C"
    assert cli_main(["analyze", str(tmp_path / "nada.mp3")]) == 2
    assert "Erro:" in capsys.readouterr().err
