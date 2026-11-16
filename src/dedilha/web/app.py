"""Interface web local (FastAPI). Roda no seu computador; o áudio não sai da sua máquina,
exceto os downloads do YouTube e o download inicial do modelo de separação."""

from __future__ import annotations

import os
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..audio import AUDIO_EXTENSIONS
from ..pipeline import ALL_PARTS, AnalysisOptions, Analyzer
from ..sources import InputError, classify_source, looks_like_link
from ..transcribe import DependencyError
from ..voicings import all_voicings

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
JOB_TTL_SECONDS = 3 * 3600
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_JOB_DIR_NAME = re.compile(r"^[0-9a-f]{32}$")


class AnalysisCancelled(BaseException):
    """O usuário cancelou a análise. Deriva de BaseException para atravessar os `except Exception` do caminho
    (download, separação), que de outro modo transformariam o cancelamento num erro comum."""


@dataclass
class Job:
    id: str
    workdir: Path
    status: str = "queued"           # queued | running | done | error | cancelled
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Na fila…"
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    audio_path: Optional[Path] = None
    cancel_requested: bool = False


def _parse_instruments(value: str) -> tuple[str, ...]:
    items = tuple(dict.fromkeys(p.strip() for p in value.split(",") if p.strip()))
    bad = [i for i in items if i not in ALL_PARTS]
    if bad:
        raise HTTPException(400, f"Instrumento inválido: {', '.join(bad)}")
    return items


def default_jobs_root() -> Path:
    """Onde ficam os arquivos temporários das análises (DEDILHA_JOBS_DIR muda o local)."""
    env = os.environ.get("DEDILHA_JOBS_DIR")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "dedilha" / "jobs"


def purge_stale_jobs(root: Path, max_age: float = JOB_TTL_SECONDS) -> int:
    """Apaga pastas de análises antigas deixadas por execuções anteriores. Devolve quantas apagou."""
    removed = 0
    now = time.time()
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0
    for d in entries:
        try:
            if d.is_dir() and _JOB_DIR_NAME.match(d.name) and now - d.stat().st_mtime > max_age:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def _trim_workdir(workdir: Path, keep: Optional[Path]) -> None:
    """Depois da análise só o áudio do player precisa ficar (o WAV de entrada passa de 100 MB)."""
    try:
        entries = list(workdir.iterdir())
    except OSError:
        return
    for p in entries:
        if keep is not None and p == keep:
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
        except OSError:
            pass                     # arquivo em uso: a limpeza de pastas antigas cuida depois


def _upload_name(filename: str) -> str:
    """Só o nome do arquivo enviado (sem pastas nem caracteres de controle), para mostrar na tela."""
    name = re.split(r"[\\/]", filename)[-1]
    name = "".join(c for c in name if c.isprintable()).strip()
    return name[:200] or "arquivo"


def _hostname(host_header: str) -> str:
    """'127.0.0.1:8765' -> '127.0.0.1';  '[::1]:8765' -> '::1'."""
    h = host_header.strip().lower()
    if h.startswith("["):
        end = h.find("]")
        return h[1:end] if end > 0 else h
    return h.rsplit(":", 1)[0] if ":" in h else h


class _RequestGuard:
    """Recusa pedidos que não vieram da própria página do Dedilha.

    Um servidor local pode ser chamado por qualquer site aberto no navegador. Duas checagens:
    - Host: com o servidor só em 127.0.0.1, o cabeçalho Host precisa ser local (impede que um site
      aponte o próprio domínio para 127.0.0.1 e leia as respostas: "DNS rebinding");
    - Origin: pedidos que alteram algo (POST) vindos de outro site são recusados.
    """

    def __init__(self, app: Any, allowed_hosts: Optional[frozenset[str]] = None) -> None:
        self.app = app
        self.allowed_hosts = allowed_hosts

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
            host = headers.get("host", "")
            problem = None
            if self.allowed_hosts is not None and _hostname(host) not in self.allowed_hosts:
                problem = "Endereço não permitido. Abra o Dedilha pelo endereço mostrado no terminal."
            elif scope["method"] not in ("GET", "HEAD", "OPTIONS"):
                origin = headers.get("origin")
                if origin is not None and urlsplit(origin).netloc.lower() != host.strip().lower():
                    problem = "Pedido vindo de outro site: recusado."
            if problem:
                await JSONResponse({"detail": problem}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(jobs_root: Optional[Path] = None,
               analyzer_factory: Callable[[], Analyzer] = Analyzer,
               allowed_hosts: Optional[Iterable[str]] = None) -> FastAPI:
    """`allowed_hosts`: nomes aceitos no cabeçalho Host (None = qualquer um; use ao expor na rede)."""
    root = Path(jobs_root) if jobs_root is not None else default_jobs_root()
    root.mkdir(parents=True, exist_ok=True)
    purge_stale_jobs(root)
    jobs: dict[str, Job] = {}
    lock = threading.Lock()
    pending: queue.Queue[tuple[Job, str, AnalysisOptions]] = queue.Queue()
    worker: dict[str, threading.Thread] = {}
    analyzer_holder: dict[str, Analyzer] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        with lock:                                   # ao fechar: não deixa áudio para trás
            for j in jobs.values():
                shutil.rmtree(j.workdir, ignore_errors=True)
            jobs.clear()

    app = FastAPI(title="Dedilha", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(_RequestGuard,
                       allowed_hosts=None if allowed_hosts is None else frozenset(h.lower() for h in allowed_hosts))

    def get_analyzer() -> Analyzer:
        if "a" not in analyzer_holder:
            analyzer_holder["a"] = analyzer_factory()
        return analyzer_holder["a"]

    def cleanup_old() -> None:
        now = time.time()
        with lock:
            old = [j for j in jobs.values() if now - j.created > JOB_TTL_SECONDS
                   and j.status in ("done", "error", "cancelled")]
            for j in old:
                shutil.rmtree(j.workdir, ignore_errors=True)
                jobs.pop(j.id, None)

    def run_job(job: Job, source: str, options: AnalysisOptions) -> None:
        if job.cancel_requested:                     # cancelada ainda na fila
            job.status, job.message = "cancelled", "Análise cancelada."
            _trim_workdir(job.workdir, None)
            return
        job.status = "running"

        def progress(stage: str, frac: float, msg: str) -> None:
            # O cancelamento é cooperativo: vale na próxima vez que a análise avisa do andamento. Etapas
            # longas que não avisam (a separação de instrumentos) terminam antes.
            if job.cancel_requested:
                raise AnalysisCancelled
            job.stage, job.progress, job.message = stage, float(frac), msg

        try:
            result = get_analyzer().analyze(source, job.workdir, options, progress)
            wav = job.workdir / "input.wav"
            job.audio_path = _playback_copy(wav) if wav.exists() else None
            _trim_workdir(job.workdir, job.audio_path)
            job.result = result.to_dict()
            job.status, job.progress, job.message = "done", 1.0, "Pronto!"
        except AnalysisCancelled:
            job.status, job.message = "cancelled", "Análise cancelada."
        except (InputError, DependencyError) as exc:
            job.status, job.error = "error", str(exc)
        except Exception as exc:  # erro inesperado: não derruba o servidor
            traceback.print_exc()
            job.status, job.error = "error", f"Erro inesperado: {exc}"
        if job.status in ("error", "cancelled"):
            _trim_workdir(job.workdir, None)

    def work() -> None:                              # análise é pesada: uma por vez
        while True:
            job, source, options = pending.get()
            try:
                run_job(job, source, options)
            except Exception:                        # nunca deixa a fila parar
                traceback.print_exc()
                job.status, job.error = "error", "Erro inesperado."

    def submit(job: Job, source: str, options: AnalysisOptions) -> None:
        with lock:
            jobs[job.id] = job
            if "t" not in worker or not worker["t"].is_alive():
                # daemon: fechar o programa no meio de uma análise não fica esperando ela terminar
                worker["t"] = threading.Thread(target=work, name="dedilha-analise", daemon=True)
                worker["t"].start()
        pending.put((job, source, options))

    @app.post("/api/analyze")
    async def analyze(url: Optional[str] = Form(None), file: Optional[UploadFile] = File(None),  # noqa: B008
                      instruments: str = Form("guitar,piano,bass,voice,riffs"), separate: bool = Form(True),
                      vocabulary: str = Form("standard"), voice_from_mix: bool = Form(False),
                      slash_chords: bool = Form(False), cookies_from_browser: Optional[str] = Form(None)
                      ) -> JSONResponse:
        cleanup_old()
        url = (url or "").strip()
        has_file = bool(file and file.filename)
        if bool(url) == has_file:
            raise HTTPException(400, "Informe um link OU envie um arquivo (apenas um dos dois).")
        if vocabulary not in ("simple", "standard", "extended"):
            raise HTTPException(400, "Vocabulário de acordes inválido.")
        parts = _parse_instruments(instruments)
        job_id = uuid.uuid4().hex
        workdir = root / job_id
        display_title: Optional[str] = None

        try:
            if url:
                # Só links: um caminho de arquivo aqui faria o servidor ler arquivos do computador a
                # pedido de quem chamou a API. Arquivos entram pelo upload.
                if not looks_like_link(url):
                    raise InputError("Isso não parece um link. Cole um link do YouTube ou do Spotify, "
                                     "ou envie o arquivo de áudio.")
                spec = classify_source(url)              # falha rápido se o link não for aceito
                if spec.kind == "file":
                    raise InputError("Cole um link do YouTube ou do Spotify, ou envie o arquivo de áudio.")
                source = spec.value
            else:
                assert file is not None
                display_title = _upload_name(file.filename or "")
                ext = Path(display_title).suffix.lower()
                if ext not in AUDIO_EXTENSIONS:
                    raise InputError("Formato não suportado. Envie mp3, wav, flac, m4a, ogg, opus...")
                workdir.mkdir(parents=True)
                dest = workdir / f"upload{ext}"
                size = 0
                with dest.open("wb") as fh:
                    while chunk := await file.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise InputError("Arquivo grande demais (limite de 300 MB).")
                        fh.write(chunk)
                source = str(dest)
        except InputError as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(400, str(exc)) from exc

        workdir.mkdir(parents=True, exist_ok=True)
        options = AnalysisOptions(
            instruments=parts, separate=separate, vocabulary=vocabulary,
            voice_from_mix=voice_from_mix, slash_chords=slash_chords,
            cookies_from_browser=(cookies_from_browser or None), display_title=display_title)
        job = Job(id=job_id, workdir=workdir)
        if has_file:
            job.message = "Arquivo recebido. Na fila…"
        submit(job, source, options)
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

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str) -> JSONResponse:
        job = _get(job_id)
        if job.status in ("queued", "running"):
            job.cancel_requested = True
            if job.status == "queued":
                job.status, job.message = "cancelled", "Análise cancelada."
        return JSONResponse({"status": job.status, "cancel_requested": job.cancel_requested})

    @app.get("/api/jobs/{job_id}/audio")
    def job_audio(job_id: str) -> FileResponse:
        job = _get(job_id)
        if not job.audio_path or not job.audio_path.exists():
            raise HTTPException(404, "Áudio indisponível.")
        media = "audio/mpeg" if job.audio_path.suffix == ".mp3" else "audio/wav"
        return FileResponse(job.audio_path, media_type=media)

    voicings_cache: dict[str, Any] = {}

    @app.get("/api/voicings")
    def voicings() -> JSONResponse:
        """Posição de cada acorde no violão e no piano (tabela fixa; a página busca uma vez)."""
        if not voicings_cache:
            voicings_cache["chords"] = all_voicings()
        return JSONResponse(voicings_cache, headers={"Cache-Control": "max-age=3600"})

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
