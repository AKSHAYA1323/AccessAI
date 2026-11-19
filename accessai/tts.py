"""Text to speech (report section 5.8).

Speech is rendered to an audio file and played by the browser's audio player, which gives the
user play, pause, stop and repeat controls. pyttsx3 runs in a subprocess because its Windows SAPI
driver is unreliable on Streamlit's worker threads.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
from pathlib import Path


class SpeechError(RuntimeError):
    pass


def synthesize_offline(text: str, rate: int = 170) -> tuple[bytes, str]:
    """Offline speech with pyttsx3. Returns (audio bytes, MIME type)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "speech.wav"
        try:
            subprocess.run(
                [sys.executable, "-m", "accessai.tts", str(out), str(rate)],
                input=text, encoding="utf-8", capture_output=True, timeout=120, check=True,
                cwd=Path(__file__).resolve().parent.parent,
            )
        except subprocess.SubprocessError as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            raise SpeechError(f"Offline speech failed: {detail.strip()[-300:]}") from exc
        if not out.exists() or out.stat().st_size == 0:
            raise SpeechError("Offline speech produced no audio.")
        return out.read_bytes(), "audio/wav"


def synthesize_online(text: str) -> tuple[bytes, str]:
    """Online speech with gTTS. Sends the text to Google. Returns (audio bytes, MIME type)."""
    from gtts import gTTS
    from gtts.tts import gTTSError

    buffer = io.BytesIO()
    try:
        gTTS(text=text, lang="en").write_to_fp(buffer)
    except gTTSError as exc:
        raise SpeechError(f"Online speech failed: {exc}") from exc
    return buffer.getvalue(), "audio/mp3"


def _worker() -> None:
    import pyttsx3

    out, rate = sys.argv[1], int(sys.argv[2])
    sys.stdin.reconfigure(encoding="utf-8")
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.save_to_file(sys.stdin.read(), out)
    engine.runAndWait()


if __name__ == "__main__":
    _worker()
