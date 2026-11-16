import os
import time

import pytest
import soundfile as sf
from conftest import GroundTruthSeparator
from fastapi.testclient import TestClient
from synth import make_riff_song

from dedilha.pipeline import Analyzer
from dedilha.web.app import LOOPBACK_HOSTS, _hostname, _upload_name, create_app, default_jobs_root, purge_stale_jobs

pytestmark = pytest.mark.slow


@pytest.fixture()
def client(tmp_path, song_ctx, transcriber):
    app = create_app(jobs_root=tmp_path / "jobs",
                     analyzer_factory=lambda: Analyzer(separator=GroundTruthSeparator(song_ctx), transcriber=transcriber))
    return TestClient(app)


def _wait(client, job_id, timeout=120, done=("done", "error")):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in done:
            return j
        time.sleep(0.5)
    raise TimeoutError


def test_index_and_health(client):
    html = client.get("/").text
    assert "Dedilha" in html and "<canvas" in html and 'id="i-riffs"' in html
    assert client.get("/api/health").json() == {"status": "ok"}


def test_upload_flow(client, tmp_path, song_ctx):
    wav = tmp_path / "m.wav"
    sf.write(str(wav), song_ctx.mix(), song_ctx.sr)
    with wav.open("rb") as fh:
        r = client.post("/api/analyze", files={"file": ("Minha Música (demo).wav", fh, "audio/wav")},
                        data={"instruments": "guitar,piano,voice", "vocabulary": "standard"})
    assert r.status_code == 200
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done" and job["has_audio"]
    res = job["result"]
    # mostra o nome do arquivo enviado, não "upload" nem o caminho temporário do servidor
    assert res["track"]["display_name"] == "Minha Música (demo)" and res["track"]["source"] == "Minha Música (demo).wav"
    # depois da análise só o áudio do player fica no disco (o WAV de entrada e o upload são apagados)
    left = sorted(p.name for p in (tmp_path / "jobs" / r.json()["job_id"]).iterdir())
    assert left == ["play.mp3"] or left == ["input.wav"]
    assert res["key"] == "C maior" and res["instruments"]["mix"]["chords"][0]["label"] == "C"
    assert res["instruments"]["voice"]["notes"] and len(res["waveform"]) == 240
    audio = client.get(f"/api/jobs/{r.json()['job_id']}/audio")
    assert audio.status_code == 200 and len(audio.content) > 5000


def test_riffs_flow(tmp_path, transcriber):
    song = make_riff_song()
    app = create_app(jobs_root=tmp_path / "jobs",
                     analyzer_factory=lambda: Analyzer(separator=GroundTruthSeparator(song), transcriber=transcriber))
    c = TestClient(app)
    wav = tmp_path / "riff.wav"
    sf.write(str(wav), song.mix(), song.sr)
    with wav.open("rb") as fh:
        r = c.post("/api/analyze", files={"file": ("riff.wav", fh, "audio/wav")}, data={"instruments": "guitar,riffs"})
    assert r.status_code == 200
    job = _wait(c, r.json()["job_id"])
    assert job["status"] == "done", job.get("error")
    res = job["result"]
    assert res["riffs_requested"] and res["riffs"]
    riff = res["riffs"][0]
    assert riff["label"] == "Riff 1" and riff["instrument"] == "guitar" and len(riff["occurrences"]) >= 4
    assert riff["notes"] and riff["tab"] and riff["beats"]
    # sem "riffs" na lista, a seção não é pedida
    with wav.open("rb") as fh:
        r2 = c.post("/api/analyze", files={"file": ("riff.wav", fh, "audio/wav")}, data={"instruments": "guitar"})
    job2 = _wait(c, r2.json()["job_id"])
    assert job2["status"] == "done" and not job2["result"]["riffs_requested"] and job2["result"]["riffs"] == []


def test_validation_errors(client, tmp_path):
    assert client.post("/api/analyze", data={}).status_code == 400
    r = client.post("/api/analyze", data={"url": "https://evil.example.com/x"})
    assert r.status_code == 400 and "não é aceito" in r.json()["detail"]
    r = client.post("/api/analyze", data={"url": "https://open.spotify.com/album/4aawyAB9vmqN3uQ7FjRGTy"})
    assert r.status_code == 400
    r = client.post("/api/analyze", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400
    assert client.post("/api/analyze", data={"url": "https://youtu.be/abc", "instruments": "banjo"}).status_code == 400
    assert client.get("/api/jobs/naoexiste").status_code == 404
    both = client.post("/api/analyze", data={"url": "https://youtu.be/abc"}, files={"file": ("a.mp3", b"x", "audio/mpeg")})
    assert both.status_code == 400


def test_bad_audio_reports_error(client):
    r = client.post("/api/analyze", files={"file": ("lixo.mp3", b"nao e audio" * 200, "audio/mpeg")})
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "error" and job["error"]


# ------------------------------------------------------------------ segurança e limpeza

def test_url_field_only_accepts_links(client, tmp_path, song_ctx):
    """O campo de link não pode servir para mandar o servidor ler arquivos do computador."""
    secret = tmp_path / "particular.wav"
    sf.write(str(secret), song_ctx.mix(), song_ctx.sr)
    for value in (str(secret), f'"{secret}"', "particular.wav", str(tmp_path / "nao_existe.mp3")):
        r = client.post("/api/analyze", data={"url": value})
        assert r.status_code == 400 and "não parece um link" in r.json()["detail"], value
    assert list((tmp_path / "jobs").iterdir()) == []


def test_rejected_requests_leave_nothing_on_disk(client, tmp_path):
    r = client.post("/api/analyze", files={"file": ("a.wav", b"x" * 2000, "audio/wav")}, data={"instruments": "banjo"})
    assert r.status_code == 400
    r = client.post("/api/analyze", files={"file": ("virus.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400
    assert list((tmp_path / "jobs").iterdir()) == []


def test_failed_job_leaves_no_audio_behind(client, tmp_path):
    r = client.post("/api/analyze", files={"file": ("lixo.mp3", b"nao e audio" * 200, "audio/mpeg")})
    job_id = r.json()["job_id"]
    assert _wait(client, job_id)["status"] == "error"
    assert list((tmp_path / "jobs" / job_id).iterdir()) == []


def test_host_header_is_checked_when_serving_locally(tmp_path):
    app = create_app(jobs_root=tmp_path / "jobs", allowed_hosts=LOOPBACK_HOSTS)
    for base in ("http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:8765"):
        assert TestClient(app, base_url=base).get("/api/health").status_code == 200, base
    evil = TestClient(app, base_url="http://site-malicioso.example:8765")      # DNS apontado para 127.0.0.1
    assert evil.get("/api/health").status_code == 403
    assert evil.get("/api/jobs/qualquer").status_code == 403
    # sem a lista (servidor aberto para a rede com --host), qualquer Host é aceito
    open_app = create_app(jobs_root=tmp_path / "jobs2")
    assert TestClient(open_app, base_url="http://192.168.0.10:8765").get("/api/health").status_code == 200


def test_posts_from_other_sites_are_refused(tmp_path):
    c = TestClient(create_app(jobs_root=tmp_path / "jobs", allowed_hosts=LOOPBACK_HOSTS), base_url="http://127.0.0.1:8765")
    data = {"url": "https://evil.example.com/x"}
    r = c.post("/api/analyze", data=data, headers={"Origin": "http://site-malicioso.example"})
    assert r.status_code == 403 and "outro site" in r.json()["detail"]
    assert c.post("/api/analyze", data=data, headers={"Origin": "null"}).status_code == 403
    # a própria página (mesma origem) e clientes sem Origin (curl, scripts) passam pela checagem
    assert c.post("/api/analyze", data=data, headers={"Origin": "http://127.0.0.1:8765"}).status_code == 400
    assert c.post("/api/analyze", data=data).status_code == 400
    assert c.get("/api/health", headers={"Origin": "http://site-malicioso.example"}).status_code == 200


def test_old_job_folders_are_purged_at_startup(tmp_path):
    root = tmp_path / "jobs"
    old, new, other = root / ("a" * 32), root / ("b" * 32), root / "minhas-coisas"
    for d in (old, new, other):
        d.mkdir(parents=True)
        (d / "input.wav").write_bytes(b"x" * 100)
    long_ago = time.time() - 4 * 3600
    os.utime(old, (long_ago, long_ago))
    os.utime(other, (long_ago, long_ago))
    create_app(jobs_root=root)
    assert not old.exists() and new.exists() and other.exists()       # só pastas de análise, e só as antigas
    assert purge_stale_jobs(tmp_path / "nao_existe") == 0


def test_jobs_are_removed_when_the_server_stops(tmp_path, song_ctx, transcriber):
    app = create_app(jobs_root=tmp_path / "jobs",
                     analyzer_factory=lambda: Analyzer(separator=GroundTruthSeparator(song_ctx), transcriber=transcriber))
    wav = tmp_path / "m.wav"
    sf.write(str(wav), song_ctx.mix(), song_ctx.sr)
    with TestClient(app) as c:
        with wav.open("rb") as fh:
            r = c.post("/api/analyze", files={"file": ("m.wav", fh, "audio/wav")},
                       data={"instruments": "", "separate": "false"})
        assert _wait(c, r.json()["job_id"])["status"] == "done"
        assert (tmp_path / "jobs" / r.json()["job_id"]).exists()
    assert list((tmp_path / "jobs").iterdir()) == []


def test_small_helpers(tmp_path, monkeypatch):
    assert _hostname("127.0.0.1:8765") == "127.0.0.1" and _hostname("LOCALHOST") == "localhost"
    assert _hostname("[::1]:8765") == "::1" and _hostname("[::1]") == "::1" and _hostname("") == ""
    assert _upload_name("C:\\Users\\eu\\Músicas\\faixa 1.mp3") == "faixa 1.mp3"
    assert _upload_name("../../etc/passwd") == "passwd" and _upload_name("a\x00b.wav") == "ab.wav"
    assert _upload_name("") == "arquivo"
    monkeypatch.setenv("DEDILHA_JOBS_DIR", str(tmp_path / "x"))
    assert default_jobs_root() == tmp_path / "x"
    monkeypatch.delenv("DEDILHA_JOBS_DIR")
    assert default_jobs_root().name == "jobs"


# ------------------------------------------------------------------ diagramas e cancelamento

def test_voicings_endpoint(client):
    r = client.get("/api/voicings")
    assert r.status_code == 200
    chords = r.json()["chords"]
    assert len(chords) == 132 and chords["7:maj"]["guitar"]["text"] == "320003" and chords["7:maj"]["piano"] == [55, 59, 62]


def test_capo_hint_reaches_the_api(tmp_path, transcriber):
    from synth import make_song

    song = make_song([(10, "maj"), (3, "maj"), (5, "maj"), (7, "min")], repeats=2, seed=5)
    c = TestClient(create_app(jobs_root=tmp_path / "jobs", analyzer_factory=lambda: Analyzer(transcriber=transcriber)))
    wav = tmp_path / "bb.wav"
    sf.write(str(wav), song.mix(), song.sr)
    with wav.open("rb") as fh:
        r = c.post("/api/analyze", files={"file": ("bb.wav", fh, "audio/wav")}, data={"instruments": "", "separate": "false"})
    job = _wait(c, r.json()["job_id"])
    assert job["status"] == "done" and job["result"]["instruments"]["mix"]["capo"]["fret"] == 3


class _SlowAnalyzer:
    """Análise falsa que avisa do andamento várias vezes (para testar o cancelamento sem depender de tempo real)."""

    def __init__(self):
        self.started = 0

    def analyze(self, source, workdir, options, progress):
        self.started += 1
        for i in range(400):
            progress("separate", i / 400, "trabalhando…")
            time.sleep(0.02)
        raise AssertionError("a análise devia ter sido cancelada")


def test_cancel_running_and_queued_jobs(tmp_path, song_ctx):
    slow = _SlowAnalyzer()
    c = TestClient(create_app(jobs_root=tmp_path / "jobs", analyzer_factory=lambda: slow))
    wav = tmp_path / "m.wav"
    sf.write(str(wav), song_ctx.mix(), song_ctx.sr)

    def start():
        with wav.open("rb") as fh:
            return c.post("/api/analyze", files={"file": ("m.wav", fh, "audio/wav")}).json()["job_id"]

    first, second = start(), start()
    t0 = time.time()
    while c.get(f"/api/jobs/{first}").json()["status"] != "running" and time.time() - t0 < 10:
        time.sleep(0.05)
    assert c.get(f"/api/jobs/{second}").json()["status"] == "queued"
    # a que está na fila é cancelada na hora; a que está rodando, no próximo aviso de andamento
    assert c.post(f"/api/jobs/{second}/cancel").json()["status"] == "cancelled"
    assert c.post(f"/api/jobs/{first}/cancel").json()["cancel_requested"] is True
    job = _wait(c, first, timeout=10, done=("done", "error", "cancelled"))
    assert job["status"] == "cancelled" and job["message"] == "Análise cancelada."
    assert _wait(c, second, timeout=10, done=("cancelled",))["status"] == "cancelled"
    time.sleep(0.3)
    assert slow.started == 1                                       # a da fila nem começou
    for job_id in (first, second):                                 # e nada fica no disco
        assert list((tmp_path / "jobs" / job_id).iterdir()) == []
    assert c.post("/api/jobs/naoexiste/cancel").status_code == 404
    assert c.post(f"/api/jobs/{first}/cancel").json()["status"] == "cancelled"     # cancelar de novo não muda nada
