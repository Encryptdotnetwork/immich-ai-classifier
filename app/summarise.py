"""Transcript summarisation via the TEXT_* inference role.

This is the first consumer of the ``TEXT_ENDPOINT`` / ``TEXT_MODEL`` role, which
until now was captured in config and unused. Same contract as the vision role:
endpoint-agnostic, so it points at a local Ollama box or a remote provider
without the code knowing which.

Why summarise at all, rather than storing the raw transcript alone: a 3-minute
saved clip produces several hundred words of speech-to-text with no punctuation
discipline and no structure. That is fine as a *search* corpus and useless as a
*description*. The summary is what a human or an agent reads; the raw transcript
is what full-text search matches on. They serve different jobs, which is exactly
why the storage question has two halves.

The model returns JSON. Parse failures degrade to a truncated transcript excerpt
rather than raising: a missing summary must never fail a classification run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests

from .config import InferenceRole

TEXT_TIMEOUT = 120  # seconds
# Whisper output for a long clip can exceed a small local model's context. Trim
# from the middle rather than the tail: the opening states the topic and the
# close usually carries the takeaway, so both ends matter more than the middle.
MAX_TRANSCRIPT_CHARS = 12000

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = (
    "You summarise transcripts of short videos. You are given the speech-to-text "
    "output of one video. It may be noisy, unpunctuated or partial.\n\n"
    "Return ONLY a JSON object, no prose and no markdown fences:\n"
    "{\n"
    '  "summary": "<1-2 sentences, max 300 characters, describing what the video '
    'is ABOUT and what it claims or teaches. Write it as a factual description, '
    'not as \'this video discusses...\'>",\n'
    '  "topics": ["<lowercase topic keyword>", "..."],\n'
    '  "spoken_language": "<ISO 639-1 code>",\n'
    '  "has_useful_content": <true|false>\n'
    "}\n\n"
    "Set has_useful_content to false when the transcript is only music, filler, "
    "background chatter, or too garbled to be meaningful. Give at most 8 topics. "
    "Do not invent detail that is not in the transcript."
)


class SummariseError(RuntimeError):
    """Raised when the text endpoint cannot be reached or is misconfigured."""


@dataclass(frozen=True)
class Summary:
    """A transcript summary. ``ok`` is False when the model output was unusable."""

    summary: str
    topics: list[str]
    spoken_language: Optional[str]
    has_useful_content: bool
    ok: bool
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "topics": self.topics,
            "spoken_language": self.spoken_language,
            "has_useful_content": self.has_useful_content,
            "ok": self.ok,
        }


def _trim(text: str, limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Keep both ends of an over-long transcript, marking the elision."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n\n[... transcript trimmed ...]\n\n{text[-half:]}"


class TextClient:
    """OpenAI-compatible text-only chat client for the TEXT_* role."""

    def __init__(self, role: InferenceRole, timeout: int = TEXT_TIMEOUT) -> None:
        if not role.configured:
            raise SummariseError(
                "Text role is not configured (TEXT_ENDPOINT / TEXT_MODEL missing). "
                "Transcription still works without it; only summarisation needs it."
            )
        self._url = self._completions_url(role.endpoint)
        self._model = role.model
        self._timeout = timeout
        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if role.key:
            headers["Authorization"] = f"Bearer {role.key}"
        self._session.headers.update(headers)

    @staticmethod
    def _completions_url(endpoint: str) -> str:
        e = endpoint.rstrip("/")
        if e.endswith("/chat/completions"):
            return e
        return f"{e}/chat/completions"

    def complete(self, system_prompt: str, user_text: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0,
            "max_tokens": 800,
            "stream": False,
        }
        try:
            resp = self._session.post(self._url, json=payload, timeout=self._timeout)
        except requests.exceptions.Timeout as exc:
            raise SummariseError(
                f"Text request timed out after {self._timeout}s ({self._url})."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise SummariseError(f"Text request failed ({self._url}): {exc}") from exc

        if resp.status_code != 200:
            raise SummariseError(
                f"Text endpoint returned HTTP {resp.status_code}: {resp.text[:500]}"
            )
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise SummariseError(
                f"Unexpected text response shape: {resp.text[:500]}"
            ) from exc
        if not isinstance(content, str):
            raise SummariseError(f"Text response content was not text: {content!r}")
        return content


def parse_summary(raw: str, fallback_text: str = "") -> Summary:
    """Parse the model's JSON reply, degrading gracefully.

    A bad reply is not fatal. We fall back to a truncated transcript excerpt so
    the caller always has *something* human-readable, flagged with ok=False.
    """
    fallback = Summary(
        summary=fallback_text[:300].strip(),
        topics=[],
        spoken_language=None,
        has_useful_content=bool(fallback_text.strip()),
        ok=False,
        raw=raw if isinstance(raw, str) else "",
    )
    if not isinstance(raw, str):
        return fallback

    text = raw.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1 and j > i:
            text = text[i : j + 1]

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return fallback
    if not isinstance(data, dict):
        return fallback

    summary = str(data.get("summary") or "").strip()
    if not summary:
        return fallback

    topics: list[str] = []
    for t in data.get("topics") or []:
        if t is None:
            continue
        s = str(t).strip().lower()
        if s and s not in topics:
            topics.append(s)

    lang = data.get("spoken_language")
    lang = str(lang).strip().lower() if lang else None

    return Summary(
        summary=summary[:300],
        topics=topics[:8],
        spoken_language=lang,
        has_useful_content=bool(data.get("has_useful_content", True)),
        ok=True,
        raw=raw,
    )


def summarise_transcript(transcript_text: str, text_client: TextClient) -> Summary:
    """Summarise one transcript. Never raises on a bad model reply."""
    user_text = (
        "Summarise this video transcript according to the system instructions.\n\n"
        '"""\n' + _trim(transcript_text) + '\n"""'
    )
    raw = text_client.complete(SYSTEM_PROMPT, user_text)
    return parse_summary(raw, fallback_text=transcript_text)
