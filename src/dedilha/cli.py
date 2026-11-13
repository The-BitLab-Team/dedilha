"""Linha de comando: dedilha analyze | serve | doctor | models."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from . import __version__


def _setup_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")   # console do Windows
        except Exception:
            pass


def _parse_instruments(value: str) -> tuple[str, ...]:
    aliases = {"violao": "guitar", "violão": "guitar", "guitarra": "guitar", "guitar": "guitar",
               "piano": "piano", "baixo": "bass", "bass": "bass", "contrabaixo": "bass",
               "voz": "voice", "voice": "voice", "vocal": "voice",
               "riffs": "riffs", "riff": "riffs",
               "tudo": "all", "all": "all", "nenhum": "none", "none": "none"}
    out: list[str] = []
    for part in value.replace(";", ",").split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p not in aliases:
            raise argparse.ArgumentTypeError(
                f"instrumento desconhecido: '{part}'. Use: violao, guitarra, piano, baixo, voz, riffs, tudo, nenhum")
        k = aliases[p]
        if k == "all":
            return ("guitar", "piano", "bass", "voice", "riffs")
        if k == "none":
            return ()
        if k not in out:
            out.append(k)
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dedilha",
        description="Detecta acordes (violão, guitarra, piano), a linha de baixo, os riffs e as notas cantadas de uma música.")
    p.add_argument("--version", action="version", version=f"dedilha {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analisa um link do YouTube/Spotify ou um arquivo de áudio")
    a.add_argument("source", help="link do YouTube, link de faixa do Spotify ou caminho do arquivo")
    a.add_argument("-i", "--instruments", type=_parse_instruments, default=("guitar", "piano", "bass", "voice", "riffs"),
                   help="violao,guitarra,piano,baixo,voz,riffs | tudo | nenhum (padrão: tudo)")
    a.add_argument("--no-separation", action="store_true",
                   help="não separa instrumentos (rápido, mas só mostra os acordes da mix)")
    a.add_argument("--voice-from-mix", action="store_true",
                   help="detecta a voz direto da mix (para gravações só com voz/a cappella)")
    a.add_argument("--vocabulary", choices=("simple", "standard", "extended"), default="standard",
                   help="simple=maiores/menores; standard=+7, maj7, m7; extended=+sus, dim, aug, 5...")
    a.add_argument("--slash", action="store_true", help="mostra acordes com baixo (ex.: C/E)")
    a.add_argument("--notes", action="store_true", help="lista também as notas de cada instrumento")
    a.add_argument("--json", metavar="ARQUIVO", help="salva o resultado completo em JSON")
    a.add_argument("--max-minutes", type=float, default=15.0, help="duração máxima aceita (padrão 15)")
    a.add_argument("--cookies-from-browser", metavar="NAVEGADOR",
                   help="chrome, firefox, edge... (se o YouTube pedir login)")
    a.add_argument("--no-tuning-fix", action="store_true", help="não corrige gravações fora de A=440")
    a.add_argument("--keep-stems", metavar="PASTA", help="salva os stems separados nessa pasta")
    a.add_argument("--allow-other-urls", action="store_true",
                   help="aceita outros sites suportados pelo yt-dlp (SoundCloud, Bandcamp...)")

    s = sub.add_parser("serve", help="abre a interface web local")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true", help="não abre o navegador sozinho")

    sub.add_parser("doctor", help="verifica se tudo que o programa precisa está instalado")
    sub.add_parser("models", help="baixa antecipadamente o modelo de separação (~100 MB)")
    return p


def cmd_analyze(args: argparse.Namespace) -> int:
    from .pipeline import AnalysisOptions, Analyzer
    from .render import render_text
    from .sources import InputError
    from .transcribe import DependencyError

    opts = AnalysisOptions(
        instruments=args.instruments, separate=not args.no_separation,
        vocabulary=args.vocabulary, slash_chords=args.slash, max_minutes=args.max_minutes,
        cookies_from_browser=args.cookies_from_browser, allow_other_urls=args.allow_other_urls,
        voice_from_mix=args.voice_from_mix, auto_tune_correction=not args.no_tuning_fix,
        keep_stems_dir=Path(args.keep_stems) if args.keep_stems else None)

    last = {"stage": ""}

    def progress(stage: str, frac: float, msg: str) -> None:
        if stage != last["stage"] or stage in ("done",):
            print(f"[{frac * 100:3.0f}%] {msg}", file=sys.stderr, flush=True)
            last["stage"] = stage

    workdir = Path(tempfile.mkdtemp(prefix="dedilha_"))
    try:
        result = Analyzer().analyze(args.source, workdir, opts, progress)
    except (InputError, DependencyError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    print(render_text(result, show_notes=args.notes))
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"JSON salvo em {args.json}", file=sys.stderr)
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    import importlib.util
    import platform

    from .audio import find_ffmpeg
    from .sources import _js_runtimes

    ok_all = True

    def line(ok: bool, name: str, detail: str = "", required: bool = True) -> None:
        nonlocal ok_all
        mark = "OK " if ok else ("FALTA" if required else "aviso")
        if not ok and required:
            ok_all = False
        print(f"  [{mark:>5}] {name}" + (f" — {detail}" if detail else ""))

    print(f"Python {platform.python_version()} em {platform.system()}")
    line(sys.version_info >= (3, 11), "Python >= 3.11", "necessário para o demucs-onnx")
    for mod, name in (("numpy", "numpy"), ("scipy", "scipy"), ("librosa", "librosa"),
                      ("soundfile", "soundfile"), ("onnxruntime", "onnxruntime"),
                      ("yt_dlp", "yt-dlp"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
                      ("demucs_onnx", "demucs-onnx (separação de instrumentos)"),
                      ("basic_pitch", "basic-pitch (transcrição de notas)")):
        found = importlib.util.find_spec(mod) is not None
        hint = "" if found else "rode: python install.py"
        line(found, name, hint)
    try:
        line(True, "ffmpeg", find_ffmpeg())
    except Exception as exc:
        line(False, "ffmpeg", str(exc))
    rt = _js_runtimes()
    line(bool(rt), "Deno ou Node.js (YouTube via yt-dlp)",
         ", ".join(rt) if rt else "instale o Node.js (nodejs.org) ou o Deno (deno.com)", required=False)
    try:
        import onnxruntime as ort

        print("  Aceleradores ONNX disponíveis:", ", ".join(ort.get_available_providers()))
    except Exception:
        pass
    print("\nTudo certo!" if ok_all else "\nCorrija os itens marcados como FALTA e rode de novo.")
    return 0 if ok_all else 1


def cmd_models(_: argparse.Namespace) -> int:
    from .separation import DemucsOnnxSeparator

    ok, why = DemucsOnnxSeparator.is_available()
    if not ok:
        print(f"Erro: {why}", file=sys.stderr)
        return 2
    print("Baixando o modelo htdemucs_6s (~100 MB)…")
    try:
        DemucsOnnxSeparator.prefetch()
    except Exception as exc:
        print(f"Falhou: {exc}", file=sys.stderr)
        return 1
    print("Modelo pronto.")
    return 0


def _free_port(host: str, start: int, tries: int = 20) -> int:
    """Primeira porta livre a partir de `start` (quem abre duas vezes não leva erro de porta ocupada)."""
    import socket

    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    return start


def cmd_serve(args: argparse.Namespace) -> int:
    import threading
    import webbrowser

    import uvicorn

    from .web.app import create_app

    port = _free_port(args.host, args.port)
    if port != args.port:
        print(f"A porta {args.port} está ocupada (talvez o Dedilha já esteja aberto); usando a {port}.")
    url = f"http://{args.host}:{port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Dedilha rodando em {url}  (Ctrl+C para parar)")
    uvicorn.run(create_app(), host=args.host, port=port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    _setup_stdout()
    args = build_parser().parse_args(argv)
    return {"analyze": cmd_analyze, "doctor": cmd_doctor, "models": cmd_models,
            "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
