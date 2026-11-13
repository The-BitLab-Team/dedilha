"""Orquestra a análise completa: entrada -> (separação) -> transcrição -> acordes/voz."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import audio
from .audio import ANALYSIS_SR
from .bass import bass_range, bass_roots, clean_bass_line
from .chords import ChordAnalysis, detect_chords, relabel
from .models import AnalysisResult, ChordSegment, InstrumentResult, NoteEvent, Riff, TrackInfo
from .music import estimate_key
from .riffs import find_riffs
from .separation import DemucsOnnxSeparator, SeparationError, Separator
from .sources import ResolvedInput, resolve_source
from .transcribe import Transcriber, Transcription
from .util import ProgressFn, Window, noop_progress, timed
from .vocals import extract_vocal_notes

LABELS = {"mix": "Música (mix)", "guitar": "Violão / Guitarra", "piano": "Piano", "bass": "Baixo",
          "voice": "Voz"}
INSTRUMENTS = ("guitar", "piano", "bass", "voice")
RIFF_SOURCES = ("guitar", "piano", "bass")        # stems onde procuramos riffs
ALL_PARTS = INSTRUMENTS + ("riffs",)              # o que o usuário pode pedir
# stems cuja energia fica abaixo disto (relativa à mix) são tratados como "ausentes"
MIN_STEM_RMS_RATIO = 0.03


@dataclass
class AnalysisOptions:
    instruments: tuple[str, ...] = ALL_PARTS
    separate: bool = True                 # separar instrumentos (precisa do modelo Demucs)
    vocabulary: str = "standard"          # simple | standard | extended
    slash_chords: bool = False            # acordes com baixo invertido (C/E)
    max_minutes: float = 15.0
    cookies_from_browser: Optional[str] = None
    allow_other_urls: bool = False
    voice_from_mix: bool = False          # rastrear a voz direto da mix (ex.: gravação a cappella)
    auto_tune_correction: bool = True     # corrige gravações fora de A=440
    keep_stems_dir: Optional[Path] = None


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0


def _norm(y: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return (y * (peak / m)).astype(np.float32) if m > 1e-6 else y


def waveform_peaks(y: np.ndarray, bins: int = 240) -> list[float]:
    """Envelope (RMS por trecho, normalizado) para desenhar a forma de onda no player."""
    if y.size < bins:
        return []
    chunks = np.array_split(np.abs(y), bins)
    env = np.array([float(np.sqrt(np.mean(c ** 2))) for c in chunks])
    ref = float(np.percentile(env, 98)) or 1.0
    return [float(x) for x in np.clip(env / ref, 0.04, 1.0)]


def _beats(y: np.ndarray, sr: int) -> tuple[np.ndarray, Optional[float]]:
    import librosa

    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        bpm = float(np.atleast_1d(tempo)[0])
        return np.asarray(beats, dtype=float), (bpm if 40 <= bpm <= 240 else None)
    except Exception:
        return np.zeros(0), None


class Analyzer:
    def __init__(self, separator: Optional[Separator] = None, transcriber: Optional[Transcriber] = None):
        self._separator = separator
        self._transcriber = transcriber or Transcriber()

    # ------------------------------------------------------------------ API pública

    def analyze(self, source: str, workdir: Path, options: Optional[AnalysisOptions] = None,
                progress: ProgressFn = noop_progress) -> AnalysisResult:
        opt = options or AnalysisOptions()
        timings: dict[str, float] = {}
        t_start = time.perf_counter()
        with timed(timings, "entrada"):
            resolved = resolve_source(source, workdir, max_minutes=opt.max_minutes,
                                      cookies_from_browser=opt.cookies_from_browser,
                                      allow_other_urls=opt.allow_other_urls, progress=progress,
                                      lo=0.0, hi=0.2)
        progress("load", 0.2, "Lendo o áudio…")
        y_mix = audio.load_mono(resolved.wav_path, ANALYSIS_SR)
        result = self.analyze_signal(y_mix, resolved, opt, progress, wav_path=resolved.wav_path,
                                     timings=timings)
        timings["total"] = time.perf_counter() - t_start
        result.timings = timings
        progress("done", 1.0, "Pronto!")
        return result

    # ------------------------------------------------------------------ núcleo

    def analyze_signal(self, y_mix: np.ndarray, resolved: ResolvedInput, opt: AnalysisOptions,
                       progress: ProgressFn = noop_progress, *, wav_path: Optional[Path] = None,
                       stems: Optional[dict[str, np.ndarray]] = None,
                       timings: Optional[dict[str, float]] = None) -> AnalysisResult:
        """`stems` permite injetar stems já separados (testes, ou separação externa)."""
        timings = timings if timings is not None else {}
        wanted = [i for i in opt.instruments if i in INSTRUMENTS]
        want_riffs = "riffs" in opt.instruments
        result = AnalysisResult(track=resolved.track, warnings=list(resolved.warnings),
                                riffs_requested=want_riffs)
        result.track.duration = result.track.duration or len(y_mix) / ANALYSIS_SR
        result.waveform = waveform_peaks(y_mix)

        # 1) afinação -------------------------------------------------------------------
        tuning = audio.estimate_tuning(y_mix, ANALYSIS_SR)
        result.tuning_cents = tuning * 100.0
        factor = 1.0
        if opt.auto_tune_correction and abs(tuning) >= 0.2:
            y_work, factor = audio.retune(y_mix, ANALYSIS_SR, tuning)
        else:
            y_work = y_mix

        # 2) separação ------------------------------------------------------------------
        needs_stems = bool(wanted or want_riffs) and opt.separate
        if stems is None and needs_stems:
            stems = self._separate(resolved, wav_path, opt, progress, result, timings)
        if stems:
            result.separated = True
            stems = {k: (audio.retune(v, ANALYSIS_SR, tuning)[0] if factor != 1.0 else v)
                     for k, v in stems.items()}
        elif (wanted or want_riffs) and not opt.separate:
            result.warnings.append("Separação de instrumentos desligada: só a mix foi analisada.")

        # 3) batidas (para segmentar os acordes) -------------------------------------------
        progress("beats", 0.56, "Detectando o ritmo…")
        with timed(timings, "ritmo"):
            beat_src = stems["drums"] if stems and "drums" in stems and _rms(stems["drums"]) > 1e-4 else y_work
            beats, bpm = _beats(beat_src, ANALYSIS_SR)
        if bpm:
            result.tempo = bpm * factor
        duration = len(y_work) / ANALYSIS_SR

        # 4) acordes da música (mix harmônica) -----------------------------------------------
        win = Window(progress, "transcribe", 0.58, 0.9)
        harmonic = self._harmonic_mix(y_work, stems)
        win(0.0, "Transcrevendo as notas da música…")
        with timed(timings, "notas_mix"):
            t_mix = self._transcriber.transcribe(_norm(harmonic))
            ext_bass = None
            t_bass: Optional[Transcription] = None
            if stems and "bass" in stems and _rms(stems["bass"]) > MIN_STEM_RMS_RATIO * _rms(y_work):
                t_bass = self._transcriber.transcribe(_norm(stems["bass"]))
                ext_bass = (t_bass.posteriorgram, t_bass.frame_times)
        a_mix = detect_chords(t_mix.posteriorgram, t_mix.frame_times, duration, beat_times=beats,
                              external_bass=ext_bass, vocabulary=opt.vocabulary,
                              slash=opt.slash_chords)
        key = estimate_key(a_mix.chroma_profile)
        flats = bool(key and key.uses_flats)
        if key:
            result.key, result.key_short = key.name(), key.short()
        relabel(a_mix.segments, flats)
        result.instruments["mix"] = InstrumentResult("mix", LABELS["mix"], chords=a_mix.segments)

        # 5) violão/guitarra e piano (os riffs também precisam das notas desses stems) -----------
        riff_notes: dict[str, list[NoteEvent]] = {}
        todo = [i for i in ("guitar", "piano")
                if i in wanted or (want_riffs and stems and i in stems)]
        for n, inst in enumerate(todo):
            win(0.4 + 0.25 * n, f"Transcrevendo {LABELS[inst].lower()}…")
            ir = self._instrument(inst, y_work, stems, beats, duration, opt, flats, timings)
            if inst in wanted:
                result.instruments[inst] = ir
            if ir.available:
                riff_notes[inst] = ir.notes

        # 6) baixo -----------------------------------------------------------------------------
        if "bass" in wanted:
            win(0.8, "Montando a linha de baixo…")
            result.instruments["bass"] = self._bass(y_work, stems, t_bass, a_mix.segments, flats)
        if want_riffs and t_bass is not None:
            riff_notes["bass"] = (result.instruments["bass"].notes if "bass" in result.instruments
                                  else clean_bass_line(t_bass.notes, flats))

        # 6b) riffs --------------------------------------------------------------------------
        if want_riffs:
            win(0.88, "Procurando riffs…")
            with timed(timings, "riffs"):
                result.riffs, result.riff_messages = self._riffs(riff_notes, stems, bpm, flats)

        # 7) voz -------------------------------------------------------------------------
        if "voice" in wanted:
            win(0.9, "Detectando as notas cantadas…")
            result.instruments["voice"] = self._voice(y_work, stems, opt, flats, timings)

        # 8) volta os tempos para a escala original (se corrigimos a afinação) -----------------
        if factor != 1.0:
            for inst in result.instruments.values():
                for c in inst.chords:
                    c.start, c.end = c.start / factor, c.end / factor
                for nt in inst.notes:
                    nt.start, nt.end = nt.start / factor, nt.end / factor
            for rf in result.riffs:
                rf.duration /= factor
                for o in rf.occurrences:
                    o.start, o.end = o.start / factor, o.end / factor
                for rn in rf.notes:
                    rn.offset, rn.duration = rn.offset / factor, rn.duration / factor
        progress("finish", 0.98, "Finalizando…")
        return result

    # ------------------------------------------------------------------ partes

    def _separate(self, resolved: ResolvedInput, wav_path: Optional[Path], opt: AnalysisOptions,
                  progress: ProgressFn, result: AnalysisResult, timings: dict[str, float]
                  ) -> Optional[dict[str, np.ndarray]]:
        win = Window(progress, "separate", 0.2, 0.55)
        sep = self._separator or DemucsOnnxSeparator(save_dir=opt.keep_stems_dir)
        try:
            if wav_path is None:
                raise SeparationError("sem arquivo de áudio para separar")
            with timed(timings, "separacao"):
                return sep.separate(wav_path, ("vocals", "drums", "bass", "other", "guitar", "piano"),
                                    lambda f, m: win(f, m))
        except SeparationError as exc:
            result.warnings.append(
                f"Não foi possível separar os instrumentos ({exc}). Mostrando só os acordes da "
                "música (mix); violão/guitarra, piano e voz ficam indisponíveis.")
        return None

    @staticmethod
    def _harmonic_mix(y_work: np.ndarray, stems: Optional[dict[str, np.ndarray]]) -> np.ndarray:
        """Mix sem bateria e sem voz (quando há stems): menos ruído para os acordes."""
        if not stems:
            return y_work
        parts = [stems[k] for k in ("other", "guitar", "piano", "bass") if k in stems]
        if not parts:
            return y_work
        n = max(len(p) for p in parts)
        out = np.zeros(n, dtype=np.float32)
        for p in parts:
            out[: len(p)] += p
        if _rms(out) < 0.2 * _rms(y_work):     # separação suspeita: volta para a mix inteira
            return y_work
        return out

    def _instrument(self, inst: str, y_work: np.ndarray, stems: Optional[dict[str, np.ndarray]],
                    beats: np.ndarray, duration: float, opt: AnalysisOptions, flats: bool,
                    timings: dict[str, float]) -> InstrumentResult:
        label = LABELS[inst]
        if not stems or inst not in stems:
            return InstrumentResult(inst, label, available=False,
                                    reason="Precisa da separação de instrumentos (modelo Demucs).")
        y = stems[inst]
        if _rms(y) < MIN_STEM_RMS_RATIO * _rms(y_work):
            return InstrumentResult(inst, label, available=False,
                                    reason="Quase não há esse instrumento na música.")
        with timed(timings, f"notas_{inst}"):
            t = self._transcriber.transcribe(_norm(y), flats=flats)
            a = detect_chords(t.posteriorgram, t.frame_times, duration, beat_times=beats,
                              vocabulary=opt.vocabulary, flats=flats, slash=opt.slash_chords)
        return InstrumentResult(inst, label, chords=a.segments, notes=t.notes)

    @staticmethod
    def _bass(y_work: np.ndarray, stems: Optional[dict[str, np.ndarray]],
              t_bass: Optional[Transcription], mix_chords: list[ChordSegment], flats: bool
              ) -> InstrumentResult:
        label = LABELS["bass"]
        if not stems or "bass" not in stems:
            return InstrumentResult("bass", label, available=False,
                                    reason="Precisa da separação de instrumentos (modelo Demucs).")
        if t_bass is None:
            return InstrumentResult("bass", label, available=False,
                                    reason="Quase não há baixo na música.")
        line = clean_bass_line(t_bass.notes, flats)
        low, high = bass_range(line)
        return InstrumentResult("bass", label, chords=bass_roots(line, mix_chords, flats), notes=line,
                                range_low=low, range_high=high)

    @staticmethod
    def _riffs(riff_notes: dict[str, list[NoteEvent]], stems: Optional[dict[str, np.ndarray]],
               bpm: Optional[float], flats: bool) -> tuple[list[Riff], list[str]]:
        """Procura riffs em cada stem e numera todos juntos (o mais presente primeiro)."""
        if not stems:
            return [], ["Para achar riffs é preciso separar os instrumentos (modelo Demucs): o app "
                        "procura as repetições em cada instrumento isolado."]
        found: list[Riff] = []
        for inst in RIFF_SOURCES:
            if inst in riff_notes:
                found += find_riffs(riff_notes[inst], inst, LABELS[inst], bpm=bpm, flats=flats)
        found.sort(key=lambda r: (-r.coverage, r.occurrences[0].start))
        for i, r in enumerate(found, 1):
            r.id, r.label = f"riff-{i}", f"Riff {i}"
        if not found:
            return [], ["Não encontrei nenhum riff que se repita. O app procura trechos curtos "
                        "(4 a 24 notas) tocados de forma quase idêntica mais de uma vez."]
        return found, []

    def _voice(self, y_work: np.ndarray, stems: Optional[dict[str, np.ndarray]],
               opt: AnalysisOptions, flats: bool, timings: dict[str, float]) -> InstrumentResult:
        label = LABELS["voice"]
        if stems and "vocals" in stems:
            y = stems["vocals"]
            if _rms(y) < MIN_STEM_RMS_RATIO * _rms(y_work):
                return InstrumentResult("voice", label, available=False,
                                        reason="Não detectei voz cantada nessa música.")
        elif opt.voice_from_mix:
            y = y_work
        else:
            return InstrumentResult("voice", label, available=False,
                                    reason="Precisa da separação de instrumentos para isolar a voz "
                                           "(ou use a opção 'voz direto da mix' se for só voz).")
        with timed(timings, "voz"):
            va = extract_vocal_notes(_norm(y), ANALYSIS_SR, flats=flats)
        for n in va.notes:
            n.start = max(0.0, n.start)
        return InstrumentResult("voice", label, notes=va.notes, range_low=va.range_low,
                                range_high=va.range_high)
