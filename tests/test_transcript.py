"""Tests for the Whisper transcript stage (IMGCLASS-8).

Everything here runs WITHOUT faster-whisper installed and without touching
Immich. The transcriber is stubbed; what is under test is the wiring, the
degradation behaviour, and the promise that a transcription failure never costs
you the asset.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.classifier import TRANSCRIPT_HINT_CHARS, build_user_text  # noqa: E402
from app.config import Config, InferenceRole, WhisperConfig  # noqa: E402
from app.summarise import Summary, parse_summary  # noqa: E402
from app.transcribe import Transcript, TranscribeError, build_transcriber  # noqa: E402


# --- config ---------------------------------------------------------------


def test_whisper_defaults_to_disabled():
    """A Config built without Whisper must behave exactly as it did before."""
    cfg = WhisperConfig()
    assert cfg.enabled is False
    assert build_transcriber(cfg) is None


def test_transcriber_refuses_to_build_when_disabled():
    from app.transcribe import WhisperTranscriber

    with pytest.raises(TranscribeError):
        WhisperTranscriber(WhisperConfig(enabled=False))


# --- prompt assembly ------------------------------------------------------


def test_user_text_without_hints_is_unchanged():
    text = build_user_text(None, None)
    assert "OCR text" not in text
    assert "transcribed" not in text


def test_user_text_includes_both_hints():
    text = build_user_text("some ocr", "some speech")
    assert "some ocr" in text
    assert "some speech" in text
    # Both must be framed as hints, never as ground truth.
    assert text.count("hint only") == 2


def test_long_transcript_is_truncated_in_the_prompt():
    long_text = "word " * 5000
    text = build_user_text(None, long_text)
    # The excerpt is capped; the full transcript must not reach the vision model.
    assert len(text) < TRANSCRIPT_HINT_CHARS + 500


# --- summary parsing ------------------------------------------------------


def test_parse_summary_happy_path():
    raw = (
        '{"summary": "A walkthrough of a moving-average crossover entry rule.", '
        '"topics": ["trading", "Moving Average", "trading"], '
        '"spoken_language": "EN", "has_useful_content": true}'
    )
    s = parse_summary(raw)
    assert s.ok is True
    assert s.spoken_language == "en"
    # Topics are lowercased and de-duplicated.
    assert s.topics == ["trading", "moving average"]


def test_parse_summary_strips_markdown_fences():
    raw = '```json\n{"summary": "Hello.", "topics": [], "has_useful_content": true}\n```'
    assert parse_summary(raw).ok is True


def test_parse_summary_degrades_to_transcript_excerpt():
    """A garbled reply must not lose the transcript we already paid to produce."""
    s = parse_summary("the model rambled instead of returning json", fallback_text="a" * 900)
    assert s.ok is False
    assert s.summary == "a" * 300


def test_parse_summary_strips_qwen3_thinking_block():
    """Qwen3 has thinking ON by default in Ollama, and the block contains braces."""
    raw = (
        '<think>The user wants JSON. I should use {"summary": ...} with a topics '
        'array. Let me think about what { and } to emit.</think>\n'
        '{"summary": "A clip about position sizing.", "topics": ["trading"], '
        '"has_useful_content": true}'
    )
    s = parse_summary(raw)
    assert s.ok is True
    assert s.summary == "A clip about position sizing."
    assert s.topics == ["trading"]


def test_parse_summary_strips_thinking_wrapped_in_fences():
    raw = (
        "<thinking>reasoning with a { brace }</thinking>\n"
        '```json\n{"summary": "Hello.", "topics": [], "has_useful_content": true}\n```'
    )
    assert parse_summary(raw).ok is True


def test_truncated_thinking_block_falls_back_instead_of_parsing_garbage():
    """max_tokens cut the reply mid-reasoning: there is no answer to recover."""
    raw = '<think>I am reasoning about {this} and never finish'
    s = parse_summary(raw, fallback_text="the raw transcript text")
    assert s.ok is False
    assert s.summary == "the raw transcript text"


def test_parse_summary_caps_topics_at_eight():
    raw = '{"summary": "x", "topics": %s}' % str([f"t{i}" for i in range(20)]).replace("'", '"')
    assert len(parse_summary(raw).topics) == 8


# --- signal gathering -----------------------------------------------------


class _StubTranscriber:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.calls = 0

    def transcribe(self, path):
        self.calls += 1
        if self._error:
            raise TranscribeError(self._error)
        return self._result


def _cfg() -> Config:
    return Config(
        immich_url="http://stub", immich_api_key="k",
        immich_internal_prefix="/usr/src/app/upload", local_mount="/immich-library",
        vision=InferenceRole("http://stub/v1", "stub-vlm", ""),
        text=InferenceRole("", "", ""),
        tag_verify_max_retries=1, tag_verify_delay=0.0,
        source_album="Unsorted", app_data_dir="/tmp",
        batch_group_size=25, batch_pause=0.0,
    )


def _patch_video_reads(monkeypatch):
    import app.signals as signals

    monkeypatch.setattr(signals, "_read_local_file", lambda a, c: "/fake/video.mp4")
    monkeypatch.setattr(signals, "_extract_frames", lambda p, n=5: ["Zm9v"] * 5)


def test_image_asset_never_invokes_the_transcriber(monkeypatch):
    import app.signals as signals

    monkeypatch.setattr(signals, "_read_local_file", lambda a, c: "/fake/img.jpg")
    monkeypatch.setattr(signals, "_image_b64", lambda p: "Zm9v")
    stub = _StubTranscriber(result=Transcript("nope", "en", 0.9, 1.0))

    out = signals.gather_signals({"id": "a1", "type": "IMAGE"}, _cfg(), None, stub)
    assert stub.calls == 0
    assert out["transcript"] is None


def test_video_asset_carries_the_transcript(monkeypatch):
    import app.signals as signals

    _patch_video_reads(monkeypatch)
    t = Transcript("this clip explains risk management", "en", 0.98, 31.0, model="base")
    out = signals.gather_signals({"id": "a2", "type": "VIDEO"}, _cfg(), None,
                                 _StubTranscriber(result=t))
    assert out["transcript"].text == "this clip explains risk management"
    assert out["transcript"].word_count == 5
    assert out["transcript_error"] is None
    assert len(out["frames"]) == 5


def test_transcription_failure_does_not_lose_the_asset(monkeypatch):
    """The whole point: a broken soundtrack must not cost you the frames."""
    import app.signals as signals

    _patch_video_reads(monkeypatch)
    out = signals.gather_signals({"id": "a3", "type": "VIDEO"}, _cfg(), None,
                                 _StubTranscriber(error="ffmpeg exploded"))
    assert out["transcript"] is None
    assert "ffmpeg exploded" in out["transcript_error"]
    # Frames survived, so the asset is still classifiable.
    assert len(out["frames"]) == 5


def test_no_speech_is_not_an_error(monkeypatch):
    import app.signals as signals

    _patch_video_reads(monkeypatch)
    out = signals.gather_signals({"id": "a4", "type": "VIDEO"}, _cfg(), None,
                                 _StubTranscriber(result=None))
    assert out["transcript"] is None
    assert out["transcript_error"] is None


def test_transcriber_is_optional(monkeypatch):
    """Passing no transcriber reproduces the exact pre-Whisper signal bundle."""
    import app.signals as signals

    _patch_video_reads(monkeypatch)
    out = signals.gather_signals({"id": "a5", "type": "VIDEO"}, _cfg(), None)
    assert out["transcript"] is None
    assert out["transcript_error"] is None


# --- transcript shape -----------------------------------------------------


def test_transcript_serialises_for_the_spike_dump():
    t = Transcript("hello there world", "en", 0.9, 12.5,
                   segments=[{"start": 0.0, "end": 1.2, "text": "hello there world"}],
                   model="base")
    d = t.to_dict()
    assert d["word_count"] == 3
    assert d["segments"][0]["text"] == "hello there world"
    assert d["model"] == "base"
