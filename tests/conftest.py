import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def transcriber():
    from dedilha.transcribe import Transcriber

    return Transcriber()


@pytest.fixture(scope="session")
def song_ctx():
    """C - G - Am - F, 4 repetições, com todos os instrumentos sintéticos."""
    from synth import make_song

    return make_song([(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")], repeats=4, seed=11)


class GroundTruthSeparator:
    """Separador falso: devolve os stems sintéticos (permite testar o fluxo completo sem o Demucs)."""

    name = "teste"

    def __init__(self, song):
        self.song = song

    def separate(self, wav_path, stems, progress):
        progress(0.5, "separando (falso)")
        out = {k: v.astype("float32") for k, v in self.song.stems.items() if k in stems}
        out["other"] = out["guitar"] * 0.0 + 1e-5        # "other" vazio
        return out
