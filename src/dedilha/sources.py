"""Resolve a entrada do usuário (link do YouTube, link do Spotify ou arquivo) em um WAV local.

Spotify: o áudio do Spotify é protegido por DRM e NÃO é baixado. O que fazemos (como o spotDL):
lemos título/artista da faixa e procuramos a mesma música no YouTube.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .audio import AUDIO_EXTENSIONS, AudioError, find_ffmpeg, probe_duration, to_wav
from .models import TrackInfo
from .util import ProgressFn, Window, noop_progress


class InputError(RuntimeError):
    """Problema com a entrada do usuário (mensagem já em português e acionável)."""


# --------------------------------------------------------------------------- classificação

_SPOTIFY_ID = r"[A-Za-z0-9]{22}"
_SPOTIFY_TRACK_URL = re.compile(
    rf"open\.spotify\.com/(?:intl-[A-Za-z-]+/)?(?:embed/)?track/({_SPOTIFY_ID})")
_SPOTIFY_TRACK_URI = re.compile(rf"^spotify:track:({_SPOTIFY_ID})$")
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
                  "youtu.be", "www.youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com"}


@dataclass
class SourceSpec:
    kind: str                       # "file" | "youtube" | "spotify" | "url"
    value: str                      # caminho ou URL normalizado
    spotify_id: Optional[str] = None


def classify_source(source: str, allow_other_urls: bool = False) -> SourceSpec:
    s = source.strip().strip('"').strip("'").strip()
    if not s:
        raise InputError("Informe um link do YouTube/Spotify ou o caminho de um arquivo de áudio.")

    m = _SPOTIFY_TRACK_URI.match(s)
    if m:
        return SourceSpec("spotify", s, m.group(1))
    if s.lower().startswith("spotify:"):
        raise InputError("Só aceito links de FAIXAS (track) do Spotify, não álbuns/playlists/artistas.")

    if re.match(r"^https?://", s, re.I):
        parsed = urllib.parse.urlparse(s)
        host = (parsed.hostname or "").lower()
        if host.endswith("spotify.com"):
            m = _SPOTIFY_TRACK_URL.search(s)
            if not m:
                raise InputError("Só aceito links de FAIXAS (track) do Spotify, não álbuns, "
                                 "playlists, artistas ou podcasts.")
            return SourceSpec("spotify", f"https://open.spotify.com/track/{m.group(1)}", m.group(1))
        if host in _YOUTUBE_HOSTS:
            if parsed.path.rstrip("/") == "/playlist":
                raise InputError("Esse é um link de playlist do YouTube. Cole o link de um vídeo.")
            return SourceSpec("youtube", s)
        if allow_other_urls:
            return SourceSpec("url", s)
        raise InputError(
            f"O site '{host}' não é aceito. Use links do YouTube ou do Spotify, ou envie o "
            "arquivo de áudio (mp3, wav, flac, m4a, ogg...).")

    path = Path(os.path.expanduser(s))
    if not path.exists():
        raise InputError(f"Não encontrei o arquivo: {path}")
    if not path.is_file():
        raise InputError(f"Isso é uma pasta, não um arquivo de áudio: {path}")
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise InputError(
            f"Formato '{path.suffix}' não suportado. Aceitos: "
            + ", ".join(sorted(e.lstrip('.') for e in AUDIO_EXTENSIONS)))
    return SourceSpec("file", str(path))


@dataclass
class ResolvedInput:
    wav_path: Path
    track: TrackInfo
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- YouTube


def _js_runtimes() -> dict[str, dict[str, str]]:
    """JS runtimes disponíveis para o yt-dlp (necessário para o YouTube atual)."""
    found: dict[str, dict[str, str]] = {}
    for name, binary in (("deno", "deno"), ("node", "node"), ("quickjs", "qjs"), ("bun", "bun")):
        p = shutil.which(binary)
        if p:
            found[name] = {"path": p}
    return found


def _friendly_download_error(msg: str) -> str:
    low = msg.lower()
    hint = " Se persistir, atualize o yt-dlp: pip install -U yt-dlp"
    if "not a bot" in low or "sign in to confirm" in low:
        return ("O YouTube pediu login para liberar esse vídeo. Tente informar o navegador onde "
                "você está logado (ex.: --cookies-from-browser chrome).")
    if "private video" in low:
        return "Esse vídeo é privado."
    if "video unavailable" in low or "this video is not available" in low:
        return "Esse vídeo não está disponível (removido ou bloqueado na sua região)."
    if "age" in low and "restrict" in low:
        return ("Vídeo com restrição de idade: informe o navegador onde você está logado "
                "(--cookies-from-browser chrome).")
    if "no supported javascript runtime" in low or "javascript runtime" in low:
        return "O yt-dlp precisa do Deno ou do Node.js instalado para baixar do YouTube." + hint
    return "Falha ao baixar do YouTube: " + msg.splitlines()[-1][:300] + "." + hint


def _channel_artist(info: dict[str, Any]) -> Optional[str]:
    artist = info.get("artist") or info.get("creator")
    if artist:
        return str(artist).split(",")[0].strip()
    chan = info.get("channel") or info.get("uploader")
    if chan:
        return re.sub(r"\s*-\s*Topic$", "", str(chan)).strip()
    return None


def _ydl_base_opts(cookies_from_browser: Optional[str]) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "socket_timeout": 30, "retries": 3, "restrictfilenames": True,
        "ffmpeg_location": find_ffmpeg(),
    }
    runtimes = _js_runtimes()
    if runtimes:
        opts["js_runtimes"] = runtimes
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return opts


def download_youtube(url: str, workdir: Path, *, max_seconds: float,
                     cookies_from_browser: Optional[str] = None,
                     progress: Callable[[float, str], None] = lambda f, m: None
                     ) -> tuple[Path, dict[str, Any]]:
    """Baixa só o áudio de um vídeo do YouTube. Retorna (arquivo, info do yt-dlp)."""
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise InputError("O pacote yt-dlp não está instalado (pip install yt-dlp).") from exc

    workdir.mkdir(parents=True, exist_ok=True)
    base = _ydl_base_opts(cookies_from_browser)

    try:
        progress(0.02, "Consultando o vídeo no YouTube…")
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=False)
        if info and info.get("_type") == "playlist":
            entries = [e for e in (info.get("entries") or []) if e]
            if not entries:
                raise InputError("Não encontrei nenhum vídeo nesse link.")
            info = entries[0]
        duration = float(info.get("duration") or 0)
        if duration and duration > max_seconds:
            raise InputError(
                f"O vídeo tem {duration / 60:.0f} min; o limite é {max_seconds / 60:.0f} min. "
                "Aumente o limite (--max-minutes) ou use um trecho menor.")

        def hook(d: dict[str, Any]) -> None:
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                if total:
                    progress(0.1 + 0.9 * d.get("downloaded_bytes", 0) / total, "Baixando áudio…")

        opts = dict(base, format="bestaudio/best", outtmpl=str(workdir / "download.%(ext)s"),
                    progress_hooks=[hook])
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([info.get("webpage_url") or url])
    except InputError:
        raise
    except Exception as exc:  # yt_dlp.utils.DownloadError e afins
        raise InputError(_friendly_download_error(str(exc))) from exc

    files = [p for p in workdir.glob("download.*")
             if p.suffix not in (".part", ".ytdl", ".temp") and p.is_file()]
    if not files:
        raise InputError("O download terminou, mas não encontrei o arquivo de áudio.")
    return max(files, key=lambda p: p.stat().st_size), info


# --------------------------------------------------------------------------- Spotify


@dataclass
class SpotifyMeta:
    title: Optional[str]
    artist: Optional[str]
    duration: Optional[float] = None    # segundos, quando disponível
    origin: str = ""                    # de onde veio (para depuração)


HttpGet = Callable[..., str]


def _http_get(url: str, *, headers: Optional[dict[str, str]] = None, data: Optional[bytes] = None,
              timeout: float = 15.0) -> str:
    h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
         "Accept-Language": "en-US,en;q=0.9"}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h, data=data)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (URLs controladas)
        return resp.read().decode("utf-8", errors="replace")


def _artist_names(artists: Any) -> list[str]:
    """Aceita os formatos: [{'name':..}], {'items':[{'profile':{'name':..}}]}."""
    if isinstance(artists, dict):
        artists = artists.get("items") or []
    names: list[str] = []
    for a in artists or []:
        if not isinstance(a, dict):
            continue
        n = a.get("name") or (a.get("profile") or {}).get("name")
        if isinstance(n, str) and n.strip():
            names.append(n.strip())
    return names


def _find_track_entity(obj: Any) -> Optional[dict[str, Any]]:
    """Procura, recursivamente, um dict com 'name' e 'artists' (a entidade da faixa)."""
    if isinstance(obj, dict):
        if isinstance(obj.get("name"), str) and _artist_names(obj.get("artists")):
            return obj
        for v in obj.values():
            r = _find_track_entity(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_track_entity(v)
            if r:
                return r
    return None


def parse_spotify_embed_html(html: str) -> Optional[SpotifyMeta]:
    """Lê título/artista do JSON embutido na página de embed (__NEXT_DATA__)."""
    for m in re.finditer(r"<script[^>]*type=[\"']application/json[\"'][^>]*>(.*?)</script>",
                         html, re.S | re.I):
        try:
            data = json.loads(m.group(1))
        except ValueError:
            continue
        ent = _find_track_entity(data)
        if ent:
            dur = ent.get("duration") or ent.get("duration_ms") or ent.get("durationMs")
            if isinstance(dur, dict):
                dur = dur.get("totalMilliseconds")
            seconds = float(dur) / 1000.0 if isinstance(dur, (int, float)) and dur else None
            return SpotifyMeta(ent["name"].strip(), ", ".join(_artist_names(ent["artists"])[:2]),
                               seconds, "embed-json")
    # fallback: padrão "artists":[{"name":"X"
    m = re.search(r'"artists"\s*:\s*\[\s*\{[^\]]*?"name"\s*:\s*"((?:[^"\\]|\\.)+)"', html)
    if m:
        try:
            artist = json.loads(f'"{m.group(1)}"')
        except ValueError:
            artist = m.group(1)
        return SpotifyMeta(None, artist, None, "embed-regex")
    return None


_NOISE_PARTS = {"song", "single", "música", "musica", "canción", "cancion", "chanson", "lied",
                "brano", "spotify", "ep", "album", "álbum"}


def parse_spotify_page_html(html: str) -> Optional[SpotifyMeta]:
    """Fallback: meta tags OpenGraph da página pública da faixa."""
    def meta(prop: str) -> Optional[str]:
        m = re.search(rf'<meta[^>]+(?:property|name)=["\']{re.escape(prop)}["\'][^>]*content=["\']([^"\']*)["\']', html, re.I)
        if not m:
            m = re.search(rf'<meta[^>]+content=["\']([^"\']*)["\'][^>]*(?:property|name)=["\']{re.escape(prop)}["\']', html, re.I)
        return _unescape(m.group(1)) if m else None

    title = meta("og:title")
    desc = meta("og:description") or ""
    artist = None
    if desc:
        parts = [p.strip() for p in re.split(r"\s*[·•|]\s*", desc) if p.strip()]
        parts = [p for p in parts if not re.fullmatch(r"\d{4}", p)
                 and p.lower() not in _NOISE_PARTS
                 and not (title and p.lower() == title.lower())]
        if parts:
            artist = parts[0]
    if not artist:
        t = re.search(r"<title>(.*?)</title>", html, re.S | re.I)
        if t:
            mt = re.search(r"-\s*song(?: and lyrics)?\s+by\s+(.+?)\s*\|\s*Spotify", _unescape(t.group(1)), re.I)
            if mt:
                artist = mt.group(1).strip()
    if title or artist:
        return SpotifyMeta(title, artist, None, "page-og")
    return None


def _unescape(s: str) -> str:
    import html as _html
    return _html.unescape(s).strip()


def _spotify_api_meta(track_id: str, http_get: HttpGet) -> Optional[SpotifyMeta]:
    cid, secret = os.environ.get("SPOTIFY_CLIENT_ID"), os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not (cid and secret):
        return None
    import base64
    basic = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    tok = json.loads(http_get(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
        data=b"grant_type=client_credentials"))
    data = json.loads(http_get(f"https://api.spotify.com/v1/tracks/{track_id}",
                               headers={"Authorization": f"Bearer {tok['access_token']}"}))
    names = [a["name"] for a in data.get("artists", []) if a.get("name")]
    return SpotifyMeta(data.get("name"), ", ".join(names[:2]) or None,
                       (data.get("duration_ms") or 0) / 1000.0 or None, "api")


def fetch_spotify_meta(track_id: str, http_get: HttpGet = _http_get) -> SpotifyMeta:
    """Descobre título/artista de uma faixa do Spotify, tentando várias fontes em ordem."""
    errors: list[str] = []
    meta: Optional[SpotifyMeta] = None

    # 1) API oficial (se o usuário configurou SPOTIFY_CLIENT_ID/SECRET): a mais confiável.
    try:
        meta = _spotify_api_meta(track_id, http_get)
    except Exception as exc:
        errors.append(f"api: {exc}")
    if meta and meta.title and meta.artist:
        return meta

    # 2) oEmbed (oficial, sem chave): dá o título.
    title = meta.title if meta else None
    try:
        url = "https://open.spotify.com/oembed?url=" + urllib.parse.quote(
            f"https://open.spotify.com/track/{track_id}", safe="")
        title = title or json.loads(http_get(url)).get("title")
    except Exception as exc:
        errors.append(f"oembed: {exc}")

    # 3) página de embed (JSON embutido) e 4) página pública (meta tags): dão o artista.
    artist, duration = None, None
    for fetch_url, parser in (
        (f"https://open.spotify.com/embed/track/{track_id}", parse_spotify_embed_html),
        (f"https://open.spotify.com/track/{track_id}", parse_spotify_page_html),
    ):
        try:
            found = parser(http_get(fetch_url))
        except Exception as exc:
            errors.append(f"{fetch_url.split('/')[-2]}: {exc}")
            continue
        if found:
            title = title or found.title
            artist = artist or found.artist
            duration = duration or found.duration
        if artist:
            break

    if not title and not artist:
        raise InputError(
            "Não consegui ler os dados dessa faixa do Spotify (" + "; ".join(errors)[:200] + "). "
            "Cole o link do YouTube da música, ou envie o arquivo de áudio.")
    return SpotifyMeta(title, artist, duration, "oembed+page")


# --------------------------------------------------------------------------- busca no YouTube

_BAD_TERMS = ("live", "ao vivo", "cover", "karaoke", "karaokê", "remix", "reaction", "reação",
              "tutorial", "lesson", "aula", "sped up", "slowed", "reverb", "8d", "nightcore",
              "mashup", "instrumental", "acoustic version", "tribute", "parody", "paródia")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", s)


def _tokens(s: str) -> set[str]:
    return {t for t in _norm(s).split() if len(t) > 1}


def score_candidate(cand: dict[str, Any], title: Optional[str], artist: Optional[str],
                    duration: Optional[float]) -> float:
    """Pontua um resultado de busca do YouTube quanto à chance de ser a faixa certa."""
    ctitle = cand.get("title") or ""
    chan = cand.get("channel") or cand.get("uploader") or ""
    hay = _tokens(ctitle) | _tokens(chan)
    score = 0.0

    if title:
        tt = _tokens(re.sub(r"\(.*?\)|\[.*?\]", " ", title))
        if tt:
            score += 4.0 * len(tt & hay) / len(tt)
    if artist:
        at = _tokens(artist)
        if at:
            score += 3.0 * len(at & hay) / len(at)

    cdur = cand.get("duration")
    if duration and cdur:
        diff = abs(float(cdur) - duration)
        score += 3.0 if diff <= 3 else 1.5 if diff <= 8 else 0.0 if diff <= 15 else -min(5.0, diff / 10)
    elif cdur and cdur > 15 * 60:
        score -= 3.0

    if re.search(r"-\s*topic\s*$", chan, re.I):
        score += 1.5
    if "vevo" in chan.lower():
        score += 1.0
    nt, nreq = _norm(ctitle), _norm(title or "")
    if "official" in nt or "oficial" in nt:
        score += 0.5
    for bad in _BAD_TERMS:
        b = _norm(bad)
        if re.search(rf"\b{re.escape(b)}\b", nt) and not re.search(rf"\b{re.escape(b)}\b", nreq):
            score -= 3.0
    return score


def search_youtube_for_track(meta: SpotifyMeta, cookies_from_browser: Optional[str] = None,
                             search: Optional[Callable[[str], list[dict[str, Any]]]] = None
                             ) -> dict[str, Any]:
    """Procura no YouTube a faixa descrita por `meta`; devolve o melhor candidato."""
    import yt_dlp

    def default_search(query: str) -> list[dict[str, Any]]:
        opts = dict(_ydl_base_opts(cookies_from_browser), extract_flat=True)
        with yt_dlp.YoutubeDL(opts) as ydl:
            res = ydl.extract_info(query, download=False) or {}
        return [e for e in (res.get("entries") or []) if e]

    run = search or default_search
    base = " ".join(x for x in (meta.artist, meta.title) if x)
    seen: dict[str, dict[str, Any]] = {}
    last_error: Optional[Exception] = None
    for q in (f"ytsearch6:{base}", f"ytsearch4:{base} official audio"):
        try:
            for e in run(q):
                vid = e.get("id") or e.get("url")
                if vid and vid not in seen:
                    seen[vid] = e
        except Exception as exc:
            last_error = exc
    if not seen:
        if last_error:
            raise InputError(_friendly_download_error(str(last_error)))
        raise InputError(f"Não achei '{base}' no YouTube. Cole o link do YouTube da música.")
    best = max(seen.values(), key=lambda c: score_candidate(c, meta.title, meta.artist, meta.duration))
    vid = best.get("id") or best.get("url")
    best["webpage_url"] = best.get("url") if str(best.get("url", "")).startswith("http") \
        else f"https://www.youtube.com/watch?v={vid}"
    return best


# --------------------------------------------------------------------------- orquestração


def resolve_source(source: str, workdir: Path, *, max_minutes: float = 15.0,
                   cookies_from_browser: Optional[str] = None, allow_other_urls: bool = False,
                   progress: ProgressFn = noop_progress,
                   lo: float = 0.0, hi: float = 0.25) -> ResolvedInput:
    """Entrada do usuário -> WAV local (44,1 kHz estéreo) + metadados."""
    win = Window(progress, "input", lo, hi)
    spec = classify_source(source, allow_other_urls)
    workdir.mkdir(parents=True, exist_ok=True)
    max_seconds = max_minutes * 60.0
    warnings: list[str] = []
    track = TrackInfo(source=source, source_kind=spec.kind)

    if spec.kind == "file":
        win(0.05, "Lendo o arquivo…")
        path = Path(spec.value)
        try:
            duration = probe_duration(path)
            if duration > max_seconds:
                raise InputError(f"O arquivo tem {duration / 60:.0f} min; o limite é "
                                 f"{max_seconds / 60:.0f} min (--max-minutes).")
            win(0.3, "Convertendo para WAV…")
            wav = to_wav(path, workdir / "input.wav")
        except AudioError as exc:
            raise InputError(str(exc)) from exc
        track.title = path.stem
        track.duration = duration
        return ResolvedInput(wav, track, warnings)

    media_url = spec.value
    if spec.kind == "spotify":
        win(0.03, "Lendo dados da faixa no Spotify…")
        meta = fetch_spotify_meta(spec.spotify_id or "")
        track.title, track.artist = meta.title, meta.artist
        if not meta.artist:
            warnings.append("Não consegui descobrir o artista no Spotify; a busca no YouTube usou só "
                            "o título e pode ter achado outra música. Confira o link usado.")
        win(0.08, f"Procurando '{meta.artist or ''} {meta.title or ''}' no YouTube…".replace("  ", " "))
        best = search_youtube_for_track(meta, cookies_from_browser)
        media_url = best["webpage_url"]
        track.resolved_url = media_url
        warnings.append("O áudio do Spotify tem DRM e não pode ser baixado; usei a versão "
                        f"encontrada no YouTube ({media_url}). Se não for a música certa, cole o link "
                        "correto do YouTube.")

    sub = lambda f, m: win(0.15 + 0.7 * f, m)  # noqa: E731
    raw, info = download_youtube(media_url, workdir, max_seconds=max_seconds,
                                 cookies_from_browser=cookies_from_browser, progress=sub)
    if spec.kind != "spotify":
        track.title = info.get("track") or info.get("title")
        track.artist = _channel_artist(info)
    track.duration = float(info.get("duration") or 0)
    try:
        win(0.9, "Convertendo para WAV…")
        wav = to_wav(raw, workdir / "input.wav")
    except AudioError as exc:
        raise InputError(str(exc)) from exc
    finally:
        try:
            raw.unlink()
        except OSError:
            pass
    return ResolvedInput(wav, track, warnings)
