"""Interface web local (FastAPI). Roda no seu computador; o áudio não sai da sua máquina,
exceto os downloads do YouTube e o download inicial do modelo de separação."""

from __future__ import annotations

import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..audio import AUDIO_EXTENSIONS
from ..pipeline import ALL_PARTS, AnalysisOptions, Analyzer
from ..sources import InputError, classify_source
from ..transcribe import DependencyError

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
JOB_TTL_SECONDS = 3 * 3600


@dataclass
class Job:
    id: str
    workdir: Path
    status: str = "queued"           # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Na fila…"
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    audio_path: Optional[Path] = None


def _parse_instruments(value: str) -> tuple[str, ...]:
    items = tuple(dict.fromkeys(p.strip() for p in value.split(",") if p.strip()))
    bad = [i for i in items if i not in ALL_PARTS]
    if bad:
        raise HTTPException(400, f"Instrumento inválido: {', '.join(bad)}")
    return items


def create_app(jobs_root: Optional[Path] = None,
               analyzer_factory: Callable[[], Analyzer] = Analyzer) -> FastAPI:
    app = FastAPI(title="Dedilha", docs_url=None, redoc_url=None)
    root = jobs_root or Path.home() / ".cache" / "dedilha" / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, Job] = {}
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=1)      # análise é pesada: uma por vez
    analyzer_holder: dict[str, Analyzer] = {}

    def get_analyzer() -> Analyzer:
        if "a" not in analyzer_holder:
            analyzer_holder["a"] = analyzer_factory()
        return analyzer_holder["a"]

    def cleanup_old() -> None:
        now = time.time()
        with lock:
            old = [j for j in jobs.values() if now - j.created > JOB_TTL_SECONDS
                   and j.status in ("done", "error")]
            for j in old:
                shutil.rmtree(j.workdir, ignore_errors=True)
                jobs.pop(j.id, None)

    def run_job(job: Job, source: str, options: AnalysisOptions) -> None:
        job.status = "running"

        def progress(stage: str, frac: float, msg: str) -> None:
            job.stage, job.progress, job.message = stage, float(frac), msg

        try:
            result = get_analyzer().analyze(source, job.workdir, options, progress)
            wav = job.workdir / "input.wav"
            job.audio_path = _playback_copy(wav) if wav.exists() else None
            job.result = result.to_dict()
            job.status, job.progress, job.message = "done", 1.0, "Pronto!"
        except (InputError, DependencyError) as exc:
            job.status, job.error = "error", str(exc)
        except Exception as exc:  # erro inesperado: não derruba o servidor
            traceback.print_exc()
            job.status, job.error = "error", f"Erro inesperado: {exc}"

    @app.post("/api/analyze")
    async def analyze(url: Optional[str] = Form(None), file: Optional[UploadFile] = File(None),
                      instruments: str = Form("guitar,piano,bass,voice,riffs"), separate: bool = Form(True),
                      vocabulary: str = Form("standard"), voice_from_mix: bool = Form(False),
                      slash_chords: bool = Form(False), cookies_from_browser: Optional[str] = Form(None)
                      ) -> JSONResponse:
        cleanup_old()
        url = (url or "").strip()
        if bool(url) == bool(file and file.filename):
            raise HTTPException(400, "Informe um link OU envie um arquivo (apenas um dos dois).")
        if vocabulary not in ("simple", "standard", "extended"):
            raise HTTPException(400, "Vocabulário de acordes inválido.")
        job_id = uuid.uuid4().hex
        workdir = root / job_id
        workdir.mkdir(parents=True)

        try:
            if url:
                classify_source(url)                     # falha rápido se o link não for aceito
                source = url
            else:
                ext = Path(file.filename or "").suffix.lower()          # type: ignore[union-attr]
                if ext not in AUDIO_EXTENSIONS:
                    raise InputError("Formato não suportado. Envie mp3, wav, flac, m4a, ogg, opus...")
                dest = workdir / f"upload{ext}"
                size = 0
                with dest.open("wb") as fh:
                    while chunk := await file.read(1024 * 1024):        # type: ignore[union-attr]
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise InputError("Arquivo grande demais (limite de 300 MB).")
                        fh.write(chunk)
                source = str(dest)
        except InputError as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(400, str(exc)) from exc

        options = AnalysisOptions(
            instruments=_parse_instruments(instruments), separate=separate, vocabulary=vocabulary,
            voice_from_mix=voice_from_mix, slash_chords=slash_chords,
            cookies_from_browser=(cookies_from_browser or None))
        job = Job(id=job_id, workdir=workdir)
        if file and file.filename:
            job.message = "Arquivo recebido. Na fila…"
        with lock:
            jobs[job_id] = job
        pool.submit(run_job, job, source, options)
        return JSONResponse({"job_id": job_id})

    def _get(job_id: str) -> Job:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Análise não encontrada (talvez já tenha expirado).")
        return job

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> JSONResponse:
        job = _get(job_id)
        body: dict[str, Any] = {"status": job.status, "stage": job.stage, "progress": job.progress,
                                "message": job.message}
        if job.status == "done":
            body["result"] = job.result
            body["has_audio"] = job.audio_path is not None
        if job.status == "error":
            body["error"] = job.error
        return JSONResponse(body)

    @app.get("/api/jobs/{job_id}/audio")
    def job_audio(job_id: str) -> FileResponse:
        job = _get(job_id)
        if not job.audio_path or not job.audio_path.exists():
            raise HTTPException(404, "Áudio indisponível.")
        media = "audio/mpeg" if job.audio_path.suffix == ".mp3" else "audio/wav"
        return FileResponse(job.audio_path, media_type=media)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    return app


def _playback_copy(wav: Path) -> Path:
    """MP3 leve para tocar no navegador; se o ffmpeg não tiver MP3, usa o próprio WAV."""
    import subprocess

    from ..audio import find_ffmpeg

    mp3 = wav.with_name("play.mp3")
    try:
        proc = subprocess.run(
            [find_ffmpeg(), "-nostdin", "-y", "-v", "error", "-i", str(wav), "-c:a", "libmp3lame",
             "-b:a", "128k", str(mp3)], capture_output=True, timeout=300)
        if proc.returncode == 0 and mp3.exists() and mp3.stat().st_size > 1000:
            return mp3
    except Exception:
        pass
    return wav
