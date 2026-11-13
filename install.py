"""Instalador do Dedilha (Windows, macOS e Linux).

Uso:   python install.py            # cria a pasta .venv e instala tudo
       python install.py --no-model # não baixa o modelo de separação agora

Por que não basta "pip install"? O pacote basic-pitch exige TensorFlow, que não existe para
Python 3.12+ e quebra a instalação. Aqui instalamos o basic-pitch com --no-deps: ele já traz o
modelo ONNX dentro do pacote, e usamos o onnxruntime no lugar do TensorFlow.
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def run(cmd: list[str], check: bool = True) -> int:
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    code = subprocess.call([str(c) for c in cmd], cwd=ROOT)
    if check and code != 0:
        print(f"\nO comando acima falhou (código {code}).")
        sys.exit(code)
    return code


def main() -> None:
    if sys.version_info < (3, 11):
        sys.exit(f"Python {sys.version_info.major}.{sys.version_info.minor} é antigo demais. "
                 "Instale o Python 3.11, 3.12 ou 3.13 (python.org) e rode de novo.")
    if sys.version_info >= (3, 14):
        print("Aviso: o Python 3.14 é muito novo e alguns pacotes (numba/librosa) podem não ter "
              "versão pronta. Se a instalação falhar, use o Python 3.12 ou 3.13.\n")

    if not VENV.exists():
        print(f"Criando o ambiente virtual em {VENV} …")
        venv.create(VENV, with_pip=True)
    py = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "-e", ROOT])
    run([py, "-m", "pip", "install", "--no-deps", "basic-pitch"])

    if "--no-model" not in sys.argv:
        print("\nBaixando o modelo de separação de instrumentos (~100 MB, só uma vez)…")
        if run([py, "-m", "dedilha.cli", "models"], check=False) != 0:
            print("Não consegui baixar agora. Sem problema: ele será baixado na primeira análise.")

    print("\n" + "=" * 60)
    run([py, "-m", "dedilha.cli", "doctor"], check=False)
    print("=" * 60)
    if os.name == "nt":
        print("\nPronto! Para abrir o programa, dê dois cliques em  iniciar.bat")
    else:
        print(f"\nPronto! Para abrir:  {py} -m dedilha.cli serve")


if __name__ == "__main__":
    main()
