"""Speech-to-text for video assets, via faster-whisper running in-container.

Design constraints, in order of priority:

1. **Nothing leaves the host.** Transcription runs locally against a model
   cached in the app data dir. There is no remote-endpoint fallback here on
   purpose: audio is far more personally identifying than a still frame, so the
   privacy/cost trade-off that ``--skip-sourced`` makes for vision is NOT
   automatically the right one for speech.
2. **Read-only.** Audio is decoded to a temp file outside the Immich mount. The
   library mount is never written to, exactly as with frame extraction.
3. **Off by default.** ``WHISPER_ENABLED`` gates the whole module. A silent
   opt-in would change the cost of every video in an existing install.
4. **Absence of speech is not an error.** Silent clips, music-only clips and
   videos with no audio stream at all are normal in a saved-social library.
   Those return ``None``, not an exception, so the caller carries on and
   classifies from frames as before.

The model is loaded once and reused for the whole run — loading per asset would
dominate the runtime on a large batch.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import WhisperConfig

# 16 kHz mono PCM is what Whisper resamples to internally; handing it that
# directly avoids a second resample and keeps the temp file small.
_SAMPLE_RATE = 16000
_FFMPEG_TIMEOUT = 300  # seconds — audio extraction only, not inference


class TranscribeError(RuntimeError):
    """Raised when transcription fails for a reason worth reporting.

    NOT raised for "this video has no speech" — that is an expected outcome and
    returns None instead.
    """


@dataclass(frozen=True)
class Transcript:
    """A finished transcript for one asset."""

    text: str
    language: Optional[str]
    language_probability: Optional[float]
    duration: Optional[float]
    segments: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "word_count": self.word_count,
            "model": self.model,
            "segments": self.segments,
        }


# --- audio probing / extraction -------------------------------------------


def _ffprobe_audio(local_path: str) -> tuple[bool, Optional[float]]:
    """Return (has_audio_stream, duration_seconds).

    A missing ffprobe binary is not fatal: we fall back to "assume audio, let
    ffmpeg decide", so the module still works on a host where only ffmpeg is
    installed.
    """
    if not shutil.which("ffprobe"):
        return True, None
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type:format=duration",
        "-of", "json", local_path,
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT, check=False
        )
        data = json.loads(out.stdout or b"{}")
    except (subprocess.SubprocessError, ValueError, OSError):
        return True, None

    streams = data.get("streams") or []
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    try:
        duration = float((data.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    return has_audio, duration


def _extract_audio(local_path: str, dest_wav: str) -> None:
    """Decode the asset's audio to 16 kHz mono WAV at dest_wav."""
    if not shutil.which("ffmpeg"):
        raise TranscribeError(
            "ffmpeg is not installed in this container; it is required to decode "
            "audio for transcription."
        )
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-i", local_path,
        "-vn",  # drop video
        "-ac", "1",  # mono
        "-ar", str(_SAMPLE_RATE),
        "-f", "wav", dest_wav,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise TranscribeError(
            f"ffmpeg timed out after {_FFMPEG_TIMEOUT}s extracting audio from "
            f"{os.path.basename(local_path)}."
        ) from exc
    if proc.returncode != 0:
        raise TranscribeError(
            f"ffmpeg failed extracting audio from {os.path.basename(local_path)}: "
            f"{proc.stderr.decode('utf-8', 'replace')[:400]}"
        )


# --- the transcriber -------------------------------------------------------


class WhisperTranscriber:
    """Wraps a faster-whisper model, loaded lazily and reused across a run."""

    def __init__(self, cfg: WhisperConfig) -> None:
        if not cfg.enabled:
            raise TranscribeError(
                "Whisper is disabled. Set WHISPER_ENABLED=true to use it."
            )
        self._cfg = cfg
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._cfg.model

    def _load(self) -> Any:
        """Import and construct the model on first use.

        Lazy so that installs which never enable Whisper are not forced to carry
        a working faster-whisper/CTranslate2 stack, and so `--classify` on an
        image asset never pays the model-load cost.
        """
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on install
            raise TranscribeError(
                "faster-whisper is not installed. Add it to requirements.txt and "
                "rebuild the image, or set WHISPER_ENABLED=false."
            ) from exc

        os.makedirs(self._cfg.download_root, exist_ok=True)
        try:
            self._model = WhisperModel(
                self._cfg.model,
                device=self._cfg.device,
                compute_type=self._cfg.compute_type,
                download_root=self._cfg.download_root,
            )
        except Exception as exc:  # noqa: BLE001 - surfaces as a readable error
            raise TranscribeError(
                f"Could not load Whisper model {self._cfg.model!r} on device "
                f"{self._cfg.device!r} (compute_type={self._cfg.compute_type!r}): {exc}"
            ) from exc
        return self._model

    def transcribe(self, local_path: str) -> Optional[Transcript]:
        """Transcribe one media file. Returns None when there is no useful speech.

        None (not an exception) is returned when: the file has no audio stream,
        the video is longer than ``WHISPER_MAX_DURATION``, or the decoded text is
        shorter than ``WHISPER_MIN_CHARS``. Those are all ordinary outcomes.
        """
        if not os.path.isfile(local_path):
            raise TranscribeError(f"Media file not found for transcription: {local_path}")

        has_audio, duration = _ffprobe_audio(local_path)
        if not has_audio:
            return None
        if self._cfg.max_duration and duration and duration > self._cfg.max_duration:
            return None

        model = self._load()

        tmp_dir = tempfile.mkdtemp(prefix="imgclass-audio-")
        wav_path = os.path.join(tmp_dir, "audio.wav")
        try:
            _extract_audio(local_path, wav_path)
            if os.path.getsize(wav_path) == 0:
                return None

            kwargs: dict[str, Any] = {
                "beam_size": self._cfg.beam_size,
                # VAD keeps Whisper off long silences, which is where it is most
                # prone to hallucinating repeated filler text.
                "vad_filter": True,
            }
            if self._cfg.language:
                kwargs["language"] = self._cfg.language

            try:
                segments, info = model.transcribe(wav_path, **kwargs)
                collected = [
                    {
                        "start": round(float(s.start), 2),
                        "end": round(float(s.end), 2),
                        "text": (s.text or "").strip(),
                    }
                    for s in segments
                ]
            except Exception as exc:  # noqa: BLE001 - readable failure per asset
                raise TranscribeError(
                    f"Whisper failed on {os.path.basename(local_path)}: {exc}"
                ) from exc

            text = " ".join(s["text"] for s in collected if s["text"]).strip()
            if len(text) < self._cfg.min_chars:
                return None

            return Transcript(
                text=text,
                language=getattr(info, "language", None),
                language_probability=getattr(info, "language_probability", None),
                duration=getattr(info, "duration", duration),
                segments=collected,
                model=self._cfg.model,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def build_transcriber(cfg: WhisperConfig) -> Optional[WhisperTranscriber]:
    """Return a transcriber, or None when Whisper is switched off."""
    if not cfg.enabled:
        return None
    return WhisperTranscriber(cfg)
