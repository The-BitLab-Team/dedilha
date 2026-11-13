"""Dedilha — executável único: instala o que faltar na primeira vez e abre o programa.

Este arquivo usa só a biblioteca padrão do Python (vira o Dedilha.exe via PyInstaller).
Na primeira execução ele baixa, em uma pasta própria do usuário (sem precisar de administrador):

  1. o `uv` (gerenciador de pacotes da Astral), de github.com/astral-sh/uv;
  2. um Python 3.12 isolado (o próprio uv baixa);
  3. o Dedilha e as dependências, a partir do pacote embutido neste executável + PyPI;
  4. o Deno (runtime JavaScript que o yt-dlp exige para o YouTube), de github.com/denoland/deno;
  5. o modelo de separação de instrumentos (~100 MB, HuggingFace).

Nas execuções seguintes só abre o programa. Tudo fica em uma única pasta, que pode ser
apagada com `Dedilha.exe --uninstall`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Sequence

APP_NAME = "Dedilha"
PYTHON_SPEC = "3.12"
STATE_SCHEMA = 1
UV_BASE = "https://github.com/astral-sh/uv/releases/latest/download/"
DENO_BASE = "https://github.com/denoland/deno/releases/latest/download/"
USER_AGENT = f"{APP_NAME}-launcher"
LAUNCHER_FLAGS = ("--setup-only", "--no-model", "--reinstall", "--uninstall", "--yes", "--where",
                  "--launcher-help")

HELP = f"""{APP_NAME} — instalador e inicializador

  {APP_NAME}.exe                      instala o que faltar e abre o programa no navegador
  {APP_NAME}.exe analyze <link|arquivo> [opções]    analisa pelo terminal (veja --help do programa)
  {APP_NAME}.exe doctor | models | serve            demais comandos do programa

Opções do instalador (devem vir antes do comando):
  --launcher-help  mostra esta ajuda
  --setup-only   só instala e termina
  --no-model     não baixa o modelo de separação agora (será baixado na primeira análise)
  --reinstall    apaga o ambiente e instala tudo de novo
  --uninstall    apaga tudo o que o {APP_NAME} instalou (peça --yes para não perguntar)
  --where        mostra a pasta onde o {APP_NAME} instalou seus arquivos
"""


class SetupError(RuntimeError):
    """Falha de instalação com mensagem já pronta para o usuário."""


# ----------------------------------------------------------------------------- funções puras

def platform_assets(system: str, machine: str) -> dict[str, Optional[str]]:
    """Nomes dos arquivos do uv e do Deno para esta plataforma (None = não existe release)."""
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "x64": "x86_64",
            "arm64": "aarch64", "aarch64": "aarch64"}.get(machine.lower())
    if arch is None:
        raise SetupError(f"Arquitetura de processador não suportada: {machine}")
    s = system.lower()
    if s == "windows":
        return {"uv": f"uv-{arch}-pc-windows-msvc.zip",
                "deno": f"deno-{arch}-pc-windows-msvc.zip" if arch == "x86_64" else None}
    if s == "linux":
        return {"uv": f"uv-{arch}-unknown-linux-gnu.tar.gz", "deno": f"deno-{arch}-unknown-linux-gnu.zip"}
    if s == "darwin":
        return {"uv": f"uv-{arch}-apple-darwin.tar.gz", "deno": f"deno-{arch}-apple-darwin.zip"}
    raise SetupError(f"Sistema operacional não suportado: {system}")


def parse_sha256(text: str) -> Optional[str]:
    """Primeiro hash SHA-256 (64 hex) de um arquivo de checksum, em qualquer formato comum."""
    m = re.search(r"\b[0-9a-fA-F]{64}\b", text)
    return m.group(0).lower() if m else None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def data_dir(system: Optional[str] = None, env: Optional[dict[str, str]] = None) -> Path:
    env = os.environ if env is None else env
    if env.get("DEDILHA_HOME"):
        return Path(env["DEDILHA_HOME"]).expanduser()
    s = (system or platform.system()).lower()
    if s == "windows":
        base = env.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    if s == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(env.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / APP_NAME.lower()


def pending_steps(state: dict, wheel_sha: str, *, uv_exists: bool, venv_python_exists: bool,
                  deno_exists: bool, deno_supported: bool, no_model: bool) -> list[str]:
    """Quais etapas faltam, em ordem. Lista vazia = já está tudo instalado."""
    steps: list[str] = []
    package_ok = (venv_python_exists and state.get("schema") == STATE_SCHEMA
                  and state.get("wheel_sha256") == wheel_sha)
    if not package_ok:
        if not uv_exists:
            steps.append("uv")
        steps += ["python", "package"]
    if deno_supported and not deno_exists:
        steps.append("deno")
    if not no_model and not state.get("model_tried"):
        steps.append("model")
    return steps


def split_launcher_args(args: Sequence[str]) -> tuple[list[str], list[str]]:
    """Separa as opções do instalador (no início) do comando que vai para o programa."""
    opts: list[str] = []
    rest = list(args)
    while rest and rest[0] in LAUNCHER_FLAGS:
        opts.append(rest.pop(0))
    return opts, rest


def find_wheel(payload: Path) -> Optional[Path]:
    wheels = sorted(payload.glob("*.whl")) if payload.is_dir() else []
    return wheels[-1] if wheels else None


def venv_python(home: Path, system: Optional[str] = None) -> Path:
    win = (system or platform.system()).lower() == "windows"
    return home / "venv" / ("Scripts/python.exe" if win else "bin/python")


# ----------------------------------------------------------------------------- rede e arquivos

def _open(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_text(url: str) -> str:
    try:
        with _open(url) as resp:
            return resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        raise SetupError(_net_message(url, exc)) from exc


def _net_message(url: str, exc: Exception) -> str:
    return (f"Não consegui baixar {url}\n  Motivo: {exc}\n  Verifique a conexão com a internet "
            "(e se um antivírus/firewall não está bloqueando) e abra o programa de novo.")


def download(url: str, dest: Path, label: str, expected_sha: Optional[str] = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  Baixando {label}…", flush=True)
    try:
        with _open(url, timeout=120) as resp, part.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done, last = 0, -10
            while chunk := resp.read(1024 * 256):
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    if pct >= last + 10:
                        last = pct
                        print(f"    {pct}% ({done // 1048576} MB)", flush=True)
    except (urllib.error.URLError, OSError) as exc:
        part.unlink(missing_ok=True)
        raise SetupError(_net_message(url, exc)) from exc
    if expected_sha and sha256_of(part) != expected_sha:
        part.unlink(missing_ok=True)
        raise SetupError(f"O arquivo baixado de {url} não confere com o checksum SHA-256 publicado. "
                         "Tente de novo; se persistir, a rede pode estar alterando o download.")
    part.replace(dest)


def download_verified(url: str, dest: Path, label: str, checksum_url: str) -> None:
    expected = parse_sha256(fetch_text(checksum_url))
    if not expected:
        raise SetupError(f"Checksum ilegível em {checksum_url}")
    download(url, dest, label, expected)


def extract_binary(archive: Path, binary: str, dest_dir: Path) -> Path:
    """Extrai de um .zip/.tar.gz o arquivo cujo nome é `binary`, onde quer que esteja."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / binary
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            member = next((m for m in zf.namelist() if Path(m).name == binary), None)
            if member is None:
                raise SetupError(f"{binary} não está em {archive.name}")
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
    else:
        with tarfile.open(archive) as tf:
            member_t = next((m for m in tf.getmembers() if m.isfile() and Path(m.name).name == binary), None)
            if member_t is None:
                raise SetupError(f"{binary} não está em {archive.name}")
            src_f = tf.extractfile(member_t)
            assert src_f is not None
            with src_f, target.open("wb") as out:
                shutil.copyfileobj(src_f, out)
    target.chmod(target.stat().st_mode | 0o111)
    return target


# ----------------------------------------------------------------------------- instalação

class Installer:
    def __init__(self, home: Path, payload: Path, system: Optional[str] = None,
                 machine: Optional[str] = None) -> None:
        self.home = home
        self.payload = payload
        self.system = system or platform.system()
        self.assets = platform_assets(self.system, machine or platform.machine())
        self.exe = ".exe" if self.system.lower() == "windows" else ""
        self.tools = home / "tools"
        self.state_path = home / "state.json"
        self.log_path = home / "logs" / "setup.log"
        self.wheel = find_wheel(payload)

    # -- estado
    @property
    def uv(self) -> Path:
        return self.tools / f"uv{self.exe}"

    @property
    def deno(self) -> Path:
        return self.tools / f"deno{self.exe}"

    @property
    def python(self) -> Path:
        return venv_python(self.home, self.system)

    def load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_state(self, state: dict) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        state["schema"] = STATE_SCHEMA
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def wheel_sha(self) -> str:
        if not self.wheel:
            raise SetupError("O pacote do Dedilha não veio dentro deste executável (pasta 'payload' vazia). "
                             "Baixe o executável de novo na página de Releases.")
        return sha256_of(self.wheel)

    def plan(self, no_model: bool = False) -> list[str]:
        return pending_steps(self.load_state(), self.wheel_sha(), uv_exists=self.uv.exists(),
                             venv_python_exists=self.python.exists(), deno_exists=self.deno.exists(),
                             deno_supported=self.assets["deno"] is not None, no_model=no_model)

    # -- ambiente
    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update({
            "UV_CACHE_DIR": str(self.home / "cache"),
            "UV_PYTHON_INSTALL_DIR": str(self.home / "python"),
            "UV_PYTHON_PREFERENCE": "only-managed",      # nunca usa o Python do sistema
            "UV_LINK_MODE": "copy",
            "HF_HOME": str(self.home / "huggingface"),
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        })
        env["PATH"] = str(self.tools) + os.pathsep + env.get("PATH", "")
        return env

    def run(self, cmd: Sequence[object], what: str) -> None:
        """Roda um comando mostrando a saída e guardando no log."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(map(str, cmd))}\n")
            try:
                proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                        errors="replace", env=self.env(), cwd=str(self.home))
            except OSError as exc:
                raise SetupError(f"Não consegui executar a etapa '{what}': {exc}") from exc
            assert proc.stdout is not None
            for line in proc.stdout:
                print("    " + line.rstrip(), flush=True)
                log.write(line)
            code = proc.wait()
        if code != 0:
            raise SetupError(f"A etapa '{what}' falhou (código {code}). Detalhes em: {self.log_path}")

    # -- etapas
    def step_uv(self) -> None:
        name = self.assets["uv"]
        assert name
        archive = self.home / "downloads" / name
        download_verified(UV_BASE + name, archive, "o gerenciador de pacotes (uv)", UV_BASE + name + ".sha256")
        extract_binary(archive, self.uv.name, self.tools)
        archive.unlink(missing_ok=True)

    def step_python(self, reinstall: bool = False) -> None:
        self.run([self.uv, "python", "install", PYTHON_SPEC], f"instalar o Python {PYTHON_SPEC}")
        if reinstall and (self.home / "venv").exists():
            shutil.rmtree(self.home / "venv", ignore_errors=True)
        if not self.python.exists():
            self.run([self.uv, "venv", self.home / "venv", "--python", PYTHON_SPEC],
                     "criar o ambiente do programa")

    def step_package(self) -> None:
        assert self.wheel
        cmd: list[object] = [self.uv, "pip", "install", "--python", self.python,
                             "--reinstall-package", "dedilha"]
        constraints = self.payload / "constraints.txt"
        if constraints.exists():
            cmd += ["-c", constraints]
        self.run(cmd + [self.wheel], "instalar o Dedilha e as dependências")
        self.run([self.uv, "pip", "install", "--python", self.python, "--no-deps", "basic-pitch"],
                 "instalar o basic-pitch (transcrição de notas)")

    def step_deno(self) -> None:
        name = self.assets["deno"]
        assert name
        archive = self.home / "downloads" / name
        download_verified(DENO_BASE + name, archive, "o Deno (necessário para o YouTube)",
                          DENO_BASE + name + ".sha256sum")
        extract_binary(archive, self.deno.name, self.tools)
        archive.unlink(missing_ok=True)

    def step_model(self) -> bool:
        try:
            self.run([self.python, "-m", "dedilha.cli", "models"], "baixar o modelo de separação")
            return True
        except SetupError as exc:
            print(f"\n  Aviso: {exc}\n  Sem problema: o modelo será baixado na primeira análise.")
            return False

    def setup(self, *, no_model: bool = False, reinstall: bool = False) -> None:
        if reinstall:
            shutil.rmtree(self.home / "venv", ignore_errors=True)
            state = self.load_state()
            state.pop("wheel_sha256", None)
            self.save_state(state)
        steps = self.plan(no_model)
        if not steps:
            return
        labels = {"uv": "Gerenciador de pacotes", "python": f"Python {PYTHON_SPEC}",
                  "package": "Dedilha e dependências", "deno": "Deno (YouTube)", "model": "Modelo de separação"}
        print(f"\nPreparando o {APP_NAME} (só na primeira vez; pode levar vários minutos e baixar uns 500 MB).")
        print(f"Os arquivos ficam em: {self.home}\n", flush=True)
        state = self.load_state()
        t0 = time.time()
        for n, step in enumerate(steps, 1):
            print(f"[{n}/{len(steps)}] {labels[step]}", flush=True)
            if step == "uv":
                self.step_uv()
            elif step == "python":
                self.step_python(reinstall)
            elif step == "package":
                self.step_package()
                state["wheel_sha256"] = self.wheel_sha()
                self.save_state(state)
            elif step == "deno":
                self.step_deno()
            elif step == "model":
                state["model_tried"] = True
                state["model_ok"] = self.step_model()
                self.save_state(state)
        self.save_state(state)
        print(f"\nInstalação concluída em {int(time.time() - t0)} s.\n", flush=True)

    def command(self, args: Sequence[str]) -> list[str]:
        return [str(self.python), "-m", "dedilha.cli", *args]


# ----------------------------------------------------------------------------- programa

def payload_dir() -> Path:
    if os.environ.get("DEDILHA_PAYLOAD"):
        return Path(os.environ["DEDILHA_PAYLOAD"])
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / "payload"


def _pause(msg: str = "\nPressione Enter para fechar esta janela…") -> None:
    if sys.platform == "win32" and getattr(sys, "frozen", False) and sys.stdin and sys.stdin.isatty():
        try:
            input(msg)
        except EOFError:
            pass


def _setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")      # type: ignore[union-attr]
        except Exception:
            pass
    if sys.platform == "win32":
        os.system("chcp 65001 >nul")


def uninstall(home: Path, assume_yes: bool) -> int:
    if not home.exists():
        print("Nada para remover: o Dedilha não está instalado.")
        return 0
    if not assume_yes:
        ans = input(f"Apagar tudo em {home}? (inclui o modelo baixado) [s/N] ").strip().lower()
        if ans not in ("s", "sim", "y", "yes"):
            print("Cancelado.")
            return 1
    shutil.rmtree(home, ignore_errors=True)
    print("Removido." if not home.exists() else f"Não consegui apagar tudo; remova a pasta manualmente: {home}")
    return 0 if not home.exists() else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    _setup_console()
    args = list(sys.argv[1:] if argv is None else argv)
    opts, rest = split_launcher_args(args)
    interactive = not args                       # abriu com dois cliques

    if "--launcher-help" in opts:
        print(HELP)
        return 0
    home = data_dir()
    if "--where" in opts:
        print(home)
        return 0
    if "--uninstall" in opts:
        return uninstall(home, "--yes" in opts)

    try:
        inst = Installer(home, payload_dir())
        # o modelo só é baixado antes de abrir a interface; nos demais comandos o programa
        # baixa sozinho quando precisar (ou use o comando `models`)
        pre_download = "--no-model" not in opts and (not rest or rest[0] == "serve")
        inst.setup(no_model=not pre_download, reinstall="--reinstall" in opts)
    except SetupError as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)
        _pause()
        return 1
    except KeyboardInterrupt:
        print("\nInstalação interrompida. Abra o programa de novo para continuar de onde parou.")
        return 130

    if "--setup-only" in opts:
        print("Pronto. Abra o Dedilha.exe para usar.")
        return 0

    cmd = inst.command(rest or ["serve"])
    if not rest:
        print(f"Abrindo o {APP_NAME} no navegador… (feche esta janela para encerrar o programa)\n", flush=True)
    try:
        code = subprocess.call(cmd, env=inst.env())
    except KeyboardInterrupt:
        return 0
    except OSError as exc:
        print(f"\nERRO: não consegui iniciar o programa: {exc}\nTente: {sys.argv[0]} --reinstall",
              file=sys.stderr)
        _pause()
        return 1
    if code not in (0, 130) and interactive:
        _pause()
    return code


if __name__ == "__main__":
    sys.exit(main())
