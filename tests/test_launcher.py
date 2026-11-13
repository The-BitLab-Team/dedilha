import hashlib
import http.server
import io
import sys
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "launcher"))
import dedilha_launcher as L  # noqa: E402


# ------------------------------------------------------------------ funções puras

@pytest.mark.parametrize("system,machine,uv,deno", [
    ("Windows", "AMD64", "uv-x86_64-pc-windows-msvc.zip", "deno-x86_64-pc-windows-msvc.zip"),
    ("Windows", "ARM64", "uv-aarch64-pc-windows-msvc.zip", None),
    ("Linux", "x86_64", "uv-x86_64-unknown-linux-gnu.tar.gz", "deno-x86_64-unknown-linux-gnu.zip"),
    ("Darwin", "arm64", "uv-aarch64-apple-darwin.tar.gz", "deno-aarch64-apple-darwin.zip"),
])
def test_platform_assets(system, machine, uv, deno):
    assert L.platform_assets(system, machine) == {"uv": uv, "deno": deno}


def test_platform_assets_unsupported():
    with pytest.raises(L.SetupError):
        L.platform_assets("Windows", "ppc64")
    with pytest.raises(L.SetupError):
        L.platform_assets("Plan9", "x86_64")


def test_parse_sha256_accepts_common_formats():
    h = "ab" * 32
    assert L.parse_sha256(f"{h}  uv-x86_64-pc-windows-msvc.zip\n") == h               # uv
    assert L.parse_sha256(f"Algorithm : SHA256\nHash      : {h.upper()}\nPath : x.zip") == h  # deno
    assert L.parse_sha256("nada aqui") is None


def test_data_dir_by_platform(tmp_path):
    assert L.data_dir("Windows", {"LOCALAPPDATA": r"C:\Users\ana\AppData\Local"}) == \
        Path(r"C:\Users\ana\AppData\Local") / "Dedilha"
    assert L.data_dir("Linux", {"XDG_DATA_HOME": str(tmp_path)}) == tmp_path / "dedilha"
    assert L.data_dir("Linux", {"DEDILHA_HOME": str(tmp_path / "x")}) == tmp_path / "x"
    assert "Application Support" in str(L.data_dir("Darwin", {}))


def test_venv_python_path():
    assert L.venv_python(Path("/h"), "Windows") == Path("/h/venv/Scripts/python.exe")
    assert L.venv_python(Path("/h"), "Linux") == Path("/h/venv/bin/python")


def test_split_launcher_args():
    assert L.split_launcher_args([]) == ([], [])
    assert L.split_launcher_args(["--setup-only", "--no-model"]) == (["--setup-only", "--no-model"], [])
    assert L.split_launcher_args(["--no-model", "analyze", "x.mp3", "--no-separation"]) == \
        (["--no-model"], ["analyze", "x.mp3", "--no-separation"])
    assert L.split_launcher_args(["analyze", "--where"]) == ([], ["analyze", "--where"])


def _plan(state=None, wheel="w1", **kw):
    base = dict(uv_exists=True, venv_python_exists=True, deno_exists=True, deno_supported=True,
                no_model=False)
    base.update(kw)
    return L.pending_steps(state if state is not None else {}, wheel, **base)


def test_pending_steps_fresh_install():
    assert _plan(uv_exists=False, venv_python_exists=False, deno_exists=False) == \
        ["uv", "python", "package", "deno", "model"]


def test_pending_steps_when_everything_is_installed():
    done = {"schema": L.STATE_SCHEMA, "wheel_sha256": "w1", "model_tried": True}
    assert _plan(done) == []


def test_pending_steps_new_wheel_only_reinstalls_package():
    done = {"schema": L.STATE_SCHEMA, "wheel_sha256": "w1", "model_tried": True}
    assert _plan(done, wheel="w2") == ["python", "package"]


def test_pending_steps_model_options_and_deno_support():
    done = {"schema": L.STATE_SCHEMA, "wheel_sha256": "w1"}
    assert _plan(done) == ["model"]
    assert _plan(done, no_model=True) == []
    assert _plan(done, deno_exists=False, deno_supported=False, no_model=True) == []
    assert _plan(done, deno_exists=False, no_model=True) == ["deno"]
    assert _plan({"schema": 0, "wheel_sha256": "w1", "model_tried": True}) == ["python", "package"]


# ------------------------------------------------------------------ arquivos e rede

def test_extract_binary_from_zip_and_tar(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("pasta/uv.exe", b"ZIPBIN")
        zf.writestr("pasta/outro.txt", b"x")
    out = L.extract_binary(z, "uv.exe", tmp_path / "t1")
    assert out.read_bytes() == b"ZIPBIN"

    t = tmp_path / "a.tar.gz"
    with tarfile.open(t, "w:gz") as tf:
        info = tarfile.TarInfo("uv-x86_64/uv")
        data = b"TARBIN"
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    out = L.extract_binary(t, "uv", tmp_path / "t2")
    assert out.read_bytes() == b"TARBIN" and out.stat().st_mode & 0o100

    with pytest.raises(L.SetupError):
        L.extract_binary(z, "deno.exe", tmp_path / "t3")


@pytest.fixture()
def http_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    root = tmp_path / "www"
    root.mkdir()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()


def test_download_verified_ok_and_checksum_mismatch(http_dir, tmp_path):
    root, base = http_dir
    payload = b"conteudo" * 1000
    (root / "f.zip").write_bytes(payload)
    (root / "f.zip.sha256").write_text(f"{hashlib.sha256(payload).hexdigest()}  f.zip\n")
    dest = tmp_path / "d" / "f.zip"
    L.download_verified(base + "f.zip", dest, "teste", base + "f.zip.sha256")
    assert dest.read_bytes() == payload

    (root / "bad.zip").write_bytes(payload)
    (root / "bad.zip.sha256").write_text(("00" * 32) + "  bad.zip\n")
    with pytest.raises(L.SetupError, match="checksum"):
        L.download_verified(base + "bad.zip", tmp_path / "bad.zip", "teste", base + "bad.zip.sha256")
    assert not (tmp_path / "bad.zip").exists() and not list(tmp_path.glob("*.part"))


def test_download_network_failure_is_friendly(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    with pytest.raises(L.SetupError, match="internet"):
        L.download("http://127.0.0.1:9/x.zip", tmp_path / "x.zip", "x")


# ------------------------------------------------------------------ instalador (sem rede)

def test_installer_plan_and_missing_wheel(tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    inst = L.Installer(tmp_path / "home", payload, system="Linux", machine="x86_64")
    with pytest.raises(L.SetupError, match="pacote"):
        inst.plan()

    (payload / "dedilha-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    inst = L.Installer(tmp_path / "home", payload, system="Linux", machine="x86_64")
    assert inst.plan() == ["uv", "python", "package", "deno", "model"]
    assert inst.plan(no_model=True) == ["uv", "python", "package", "deno"]

    inst.python.parent.mkdir(parents=True)
    inst.python.write_text("")
    inst.tools.mkdir(parents=True)
    inst.uv.write_text("")
    inst.deno.write_text("")
    inst.save_state({"wheel_sha256": inst.wheel_sha(), "model_tried": True})
    assert inst.plan() == []
    assert inst.command(["doctor"])[-2:] == ["dedilha.cli", "doctor"]


def test_installer_env_isolated(tmp_path):
    (tmp_path / "p").mkdir()
    inst = L.Installer(tmp_path / "h", tmp_path / "p", system="Windows", machine="AMD64")
    env = inst.env()
    assert env["UV_PYTHON_PREFERENCE"] == "only-managed"
    assert env["PATH"].startswith(str(inst.tools)) and inst.uv.name == "uv.exe"
    assert env["HF_HOME"].startswith(str(tmp_path / "h"))


def test_uninstall(tmp_path):
    home = tmp_path / "h"
    (home / "venv").mkdir(parents=True)
    assert L.uninstall(home, assume_yes=True) == 0 and not home.exists()
    assert L.uninstall(home, assume_yes=True) == 0          # já removido
