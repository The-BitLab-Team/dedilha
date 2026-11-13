"""Gera um áudio curto (acorde de dó maior, 6 s) só com a biblioteca padrão — usado no teste de fumaça do CI."""

import math
import struct
import sys
import wave

SR = 22050


def main(path: str) -> None:
    freqs = [130.81, 164.81, 196.00, 261.63, 329.63, 392.00]       # C3 E3 G3 C4 E4 G4
    n = SR * 6
    frames = bytearray()
    for i in range(n):
        t = i / SR
        env = min(1.0, t / 0.02) * (0.6 + 0.4 * math.exp(-t * 0.6))
        v = sum(math.sin(2 * math.pi * f * t) / (1 + k * 0.3) for k, f in enumerate(freqs))
        frames += struct.pack("<h", int(max(-1.0, min(1.0, 0.18 * env * v)) * 32767))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(frames))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smoke.wav")
