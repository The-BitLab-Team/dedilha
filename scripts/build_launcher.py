"""Gera o executável único (Dedilha.exe no Windows) com o PyInstaller.

Uso (na raiz do repositório):
    pip install build pyinstaller uv
    python scripts/build_launcher.py

O que faz:
  1. monta o pacote (wheel) do Dedilha em build/payload/;
  2. gera build/payload/constraints.txt com as versões exatas das dependências para Windows/Python 3.12
     (assim todo executável instala exatamente o que foi testado; se falhar, segue sem travar versões);
  3. empacota launcher/dedilha_launcher.py + payload em dist/Dedilha(.exe).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
PAYLOAD = BUILD / "payload"


def run(cmd: list[str]) -> int:
    print("\n$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="Dedilha", help="nome do executável (padrão: Dedilha)")
    ap.add_argument("--no-constraints", action="store_true", help="não trava as versões das dependências")
    args = ap.parse_args()

    shutil.rmtree(BUILD, ignore_errors=True)
    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    PAYLOAD.mkdir(parents=True)

    if run([sys.executable, "-m", "build", "--wheel", "--outdir", str(PAYLOAD)]) != 0:
        return 1
    wheels = list(PAYLOAD.glob("*.whl"))
    if len(wheels) != 1:
        print(f"Esperava 1 wheel em {PAYLOAD}, achei {len(wheels)}")
        return 1

    if not args.no_constraints:
        uv = shutil.which("uv")
        cmd = ([uv] if uv else [sys.executable, "-m", "uv"]) + [
            "pip", "compile", "pyproject.toml", "--python-version", "3.12",
            "--python-platform", "x86_64-pc-windows-msvc", "--no-header", "--quiet",
            "-o", str(PAYLOAD / "constraints.txt")]
        if run(cmd) != 0:
            print("AVISO: não consegui travar as versões; o executável vai instalar as mais novas.")
            (PAYLOAD / "constraints.txt").unlink(missing_ok=True)

    sep = os.pathsep
    code = run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--console",
                "--name", args.name, "--distpath", str(ROOT / "dist"),
                "--workpath", str(BUILD / "pyi"), "--specpath", str(BUILD),
                "--add-data", f"{PAYLOAD}{sep}payload",
                "--exclude-module", "tkinter",
                str(ROOT / "launcher" / "dedilha_launcher.py")])
    if code == 0:
        out = next((ROOT / "dist").iterdir())
        print(f"\nPronto: {out}  ({out.stat().st_size / 1048576:.1f} MB)")
    return code


if __name__ == "__main__":
    sys.exit(main())
