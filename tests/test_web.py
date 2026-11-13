import time

import pytest
import soundfile as sf
from conftest import GroundTruthSeparator
from fastapi.testclient import TestClient
from synth import make_riff_song

from dedilha.pipeline import Analyzer
from dedilha.web.app import create_app

pytestmark = pytest.mark.slow


@pytest.fixture()
def client(tmp_path, song_ctx, transcriber):
    app = create_app(jobs_root=tmp_path / "jobs",
                     analyzer_factory=lambda: Analyzer(separator=GroundTruthSeparator(song_ctx), transcriber=transcriber))
    return TestClient(app)


def _wait(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
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
        r = client.post("/api/analyze", files={"file": ("m.wav", fh, "audio/wav")},
                        data={"instruments": "guitar,piano,voice", "vocabulary": "standard"})
    assert r.status_code == 200
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done" and job["has_audio"]
    res = job["result"]
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
