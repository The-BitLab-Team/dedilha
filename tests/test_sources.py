import json
import sys
import types
from pathlib import Path

import pytest

from dedilha import sources
from dedilha.sources import (InputError, SpotifyMeta, classify_source, download_youtube,
                              fetch_spotify_meta, parse_spotify_embed_html, parse_spotify_page_html,
                              score_candidate, search_youtube_for_track)

SPOTIFY_ID = "4cOdK2wGLETKBW3PvgPWqT"


# ------------------------------------------------------------------ classificação
@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?t=10",
    "https://music.youtube.com/watch?v=dQw4w9WgXcQ&list=RDAMVM",
    "https://m.youtube.com/shorts/abc123",
])
def test_youtube_urls(url):
    assert classify_source(url).kind == "youtube"


@pytest.mark.parametrize("url", [
    f"https://open.spotify.com/track/{SPOTIFY_ID}",
    f"https://open.spotify.com/track/{SPOTIFY_ID}?si=abcdef",
    f"https://open.spotify.com/intl-pt/track/{SPOTIFY_ID}",
    f"https://open.spotify.com/embed/track/{SPOTIFY_ID}",
    f"spotify:track:{SPOTIFY_ID}",
])
def test_spotify_urls(url):
    spec = classify_source(url)
    assert spec.kind == "spotify" and spec.spotify_id == SPOTIFY_ID


@pytest.mark.parametrize("url", [
    "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
    "https://open.spotify.com/album/4aawyAB9vmqN3uQ7FjRGTy",
    "spotify:album:4aawyAB9vmqN3uQ7FjRGTy",
    "https://www.youtube.com/playlist?list=PL123",
    "https://evil.example.com/song.mp3",
    "http://169.254.169.254/latest/meta-data",
])
def test_rejected_inputs(url):
    with pytest.raises(InputError):
        classify_source(url)


def test_other_urls_only_when_allowed():
    assert classify_source("https://soundcloud.com/a/b", allow_other_urls=True).kind == "url"


def test_files(tmp_path):
    good = tmp_path / "musica.mp3"
    good.write_bytes(b"x")
    assert classify_source(f'"{good}"').kind == "file"            # aspas do "copiar como caminho" do Windows
    with pytest.raises(InputError, match="Não encontrei"):
        classify_source(str(tmp_path / "nao_existe.mp3"))
    bad = tmp_path / "nota.txt"
    bad.write_text("oi")
    with pytest.raises(InputError, match="não suportado"):
        classify_source(str(bad))
    with pytest.raises(InputError):
        classify_source("   ")
    with pytest.raises(InputError, match="pasta"):
        classify_source(str(tmp_path))


# ------------------------------------------------------------------ Spotify
EMBED_HTML = """<html><head></head><body>
<script id="__NEXT_DATA__" type="application/json">{"props":{"pageProps":{"state":{"data":{"entity":
{"type":"track","name":"Never Gonna Give You Up","duration":213573,
 "artists":[{"name":"Rick Astley","uri":"spotify:artist:0gxyHStUsqpMadRV0Di1Qt"}]}}}}}}</script></body></html>"""

GRAPHQL_HTML = """<script type="application/json">{"a":[{"b":{"name":"Faixa X","artists":{"items":[{"profile":{"name":"Banda Y"}}]},"duration":{"totalMilliseconds":200000}}}]}</script>"""

PAGE_HTML = """<html><head><title>Never Gonna Give You Up - song and lyrics by Rick Astley | Spotify</title>
<meta property="og:title" content="Never Gonna Give You Up"/>
<meta property="og:description" content="Rick Astley · Whenever You Need Somebody · Song · 1987"/></head></html>"""

PAGE_HTML_NEW = """<meta property="og:title" content="Evidências"/>
<meta property="og:description" content="Song · Chitãozinho &amp; Xororó · 1990"/>"""


def test_parse_embed_json():
    m = parse_spotify_embed_html(EMBED_HTML)
    assert (m.title, m.artist) == ("Never Gonna Give You Up", "Rick Astley")
    assert m.duration == pytest.approx(213.573)
    g = parse_spotify_embed_html(GRAPHQL_HTML)
    assert (g.title, g.artist, g.duration) == ("Faixa X", "Banda Y", 200.0)
    assert parse_spotify_embed_html("<html>nada</html>") is None


def test_parse_embed_regex_fallback():
    m = parse_spotify_embed_html('<div>{"foo":1,"artists":[{"name":"Fulano","uri":"x"}]}</div>')
    assert m.artist == "Fulano" and m.title is None


def test_parse_page_og():
    m = parse_spotify_page_html(PAGE_HTML)
    assert (m.title, m.artist) == ("Never Gonna Give You Up", "Rick Astley")
    n = parse_spotify_page_html(PAGE_HTML_NEW)
    assert (n.title, n.artist) == ("Evidências", "Chitãozinho & Xororó")
    only_title = parse_spotify_page_html("<title>Foo - song and lyrics by Bar | Spotify</title>")
    assert only_title.artist == "Bar"


def test_fetch_spotify_meta_layers(monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        if "oembed" in url:
            return json.dumps({"title": "Never Gonna Give You Up"})
        if "/embed/track/" in url:
            return EMBED_HTML
        raise AssertionError(url)

    meta = fetch_spotify_meta(SPOTIFY_ID, fake_get)
    assert (meta.title, meta.artist) == ("Never Gonna Give You Up", "Rick Astley")
    assert f"https://open.spotify.com/track/{SPOTIFY_ID}" not in calls   # não precisou da página pública

    def embed_fails(url, **kw):
        if "oembed" in url:
            return json.dumps({"title": "Never Gonna Give You Up"})
        if "/embed/" in url:
            raise OSError("boom")
        return PAGE_HTML

    assert fetch_spotify_meta(SPOTIFY_ID, embed_fails).artist == "Rick Astley"

    def all_fail(url, **kw):
        raise OSError("sem rede")

    with pytest.raises(InputError, match="YouTube"):
        fetch_spotify_meta(SPOTIFY_ID, all_fail)


def test_fetch_spotify_meta_official_api(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "id")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")

    def fake_get(url, **kw):
        if "token" in url:
            assert b"client_credentials" in kw["data"]
            return json.dumps({"access_token": "tok"})
        assert kw["headers"]["Authorization"] == "Bearer tok"
        return json.dumps({"name": "Faixa", "artists": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
                           "duration_ms": 180000})

    meta = fetch_spotify_meta(SPOTIFY_ID, fake_get)
    assert (meta.title, meta.artist, meta.duration, meta.origin) == ("Faixa", "A, B", 180.0, "api")


# ------------------------------------------------------------------ escolha do vídeo
def test_candidate_scoring_prefers_the_studio_version():
    cands = [
        {"id": "live", "title": "Rick Astley - Never Gonna Give You Up (Live 2019)", "channel": "Rick Astley", "duration": 230},
        {"id": "cover", "title": "Never Gonna Give You Up (cover)", "channel": "Some Guy", "duration": 214},
        {"id": "karaoke", "title": "Never Gonna Give You Up - Karaoke", "channel": "KaraokeHits", "duration": 214},
        {"id": "topic", "title": "Never Gonna Give You Up", "channel": "Rick Astley - Topic", "duration": 213},
        {"id": "long", "title": "Rick Astley Greatest Hits Never Gonna Give You Up", "channel": "Hits", "duration": 3600},
    ]
    best = max(cands, key=lambda c: score_candidate(c, "Never Gonna Give You Up", "Rick Astley", 213.5))
    assert best["id"] == "topic"


def test_search_picks_best_and_builds_url():
    def fake_search(q):
        return [{"id": "aaa", "title": "Coisa qualquer", "channel": "X", "duration": 100},
                {"id": "bbb", "title": "Evidências", "channel": "Chitãozinho & Xororó - Topic", "duration": 270}]

    best = search_youtube_for_track(SpotifyMeta("Evidências", "Chitãozinho & Xororó", 270.0), search=fake_search)
    assert best["webpage_url"] == "https://www.youtube.com/watch?v=bbb"

    with pytest.raises(InputError, match="Não achei"):
        search_youtube_for_track(SpotifyMeta("x", "y"), search=lambda q: [])


# ------------------------------------------------------------------ yt-dlp (simulado)
def _fake_ytdlp(monkeypatch, info, fail=None):
    state = {"downloads": 0}

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            if fail:
                raise RuntimeError(fail)
            return info

        def download(self, urls):
            state["downloads"] += 1
            Path(self.opts["outtmpl"].replace("%(ext)s", "webm")).write_bytes(b"audio" * 100)

    mod = types.ModuleType("yt_dlp")
    mod.YoutubeDL = FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", mod)
    return state


def test_download_youtube_ok(monkeypatch, tmp_path):
    state = _fake_ytdlp(monkeypatch, {"title": "T", "duration": 120, "webpage_url": "https://youtu.be/x"})
    path, info = download_youtube("https://youtu.be/x", tmp_path, max_seconds=600)
    assert path.name == "download.webm" and state["downloads"] == 1 and info["title"] == "T"


def test_download_youtube_too_long(monkeypatch, tmp_path):
    state = _fake_ytdlp(monkeypatch, {"title": "T", "duration": 7200})
    with pytest.raises(InputError, match="limite"):
        download_youtube("https://youtu.be/x", tmp_path, max_seconds=900)
    assert state["downloads"] == 0                       # nem baixou


@pytest.mark.parametrize("raw, expected", [
    ("ERROR: Sign in to confirm you’re not a bot", "cookies-from-browser"),
    ("ERROR: Private video", "privado"),
    ("ERROR: Video unavailable", "não está disponível"),
    ("ERROR: some weird failure", "pip install -U yt-dlp"),
])
def test_download_errors_are_friendly(monkeypatch, tmp_path, raw, expected):
    _fake_ytdlp(monkeypatch, {}, fail=raw)
    with pytest.raises(InputError, match=expected):
        download_youtube("https://youtu.be/x", tmp_path, max_seconds=900)
