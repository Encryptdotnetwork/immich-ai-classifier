"""OpenAI-compatible vision client.

Endpoint-agnostic by design: the same code talks to a local Ollama box on the
LAN or a remote provider. Nothing here hardcodes which — it only knows the
configured endpoint/model/key (the VISION_* role, loaded once at startup).

The single public entry point is ``VisionClient.vision_complete``.
"""

from __future__ import annotations

import base64
import json

import requests

from .config import InferenceRole

VISION_TIMEOUT = 180  # seconds — local model load + inference can be slow


class VisionError(RuntimeError):
    """Any failure talking to the vision endpoint, with a human-readable message."""


def _detect_media_type(image_bytes: bytes) -> str:
    """Best-effort image MIME sniff from magic bytes; defaults to PNG."""
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


class VisionClient:
    """Sends base64 image + text chat completions to an OpenAI-compatible server."""

    def __init__(self, role: InferenceRole, timeout: int = VISION_TIMEOUT) -> None:
        if not role.endpoint or not role.model:
            raise VisionError(
                "Vision role is not configured (VISION_ENDPOINT / VISION_MODEL missing)."
            )
        self._url = self._completions_url(role.endpoint)
        self._model = role.model
        self._timeout = timeout
        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if role.key:  # local servers often need no auth; only send if provided
            headers["Authorization"] = f"Bearer {role.key}"
        self._session.headers.update(headers)

    @staticmethod
    def _completions_url(endpoint: str) -> str:
        """Resolve the chat-completions URL from a forgiving endpoint value.

        Accepts either a base ending in '/v1' (the OpenAI convention) or a full
        '.../chat/completions' URL.
        """
        e = endpoint.rstrip("/")
        if e.endswith("/chat/completions"):
            return e
        return f"{e}/chat/completions"

    def vision_complete(
        self, image_bytes: bytes, system_prompt: str, user_text: str | None = None
    ) -> str:
        """Return the assistant's text reply for (image + optional text).

        Raises VisionError on connection/timeout/non-200/bad-JSON/odd-shape.
        """
        data_url = (
            f"data:{_detect_media_type(image_bytes)};base64,"
            + base64.b64encode(image_bytes).decode("ascii")
        )
        user_content: list[dict] = []
        if user_text:
            user_content.append({"type": "text", "text": user_text})
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "max_tokens": 1024,
            "stream": False,
        }

        try:
            resp = self._session.post(self._url, json=payload, timeout=self._timeout)
        except requests.exceptions.Timeout as exc:
            raise VisionError(
                f"Vision request timed out after {self._timeout}s ({self._url})."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise VisionError(
                f"Could not connect to vision endpoint {self._url}: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise VisionError(f"Vision request failed ({self._url}): {exc}") from exc

        if resp.status_code != 200:
            raise VisionError(
                f"Vision endpoint returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise VisionError(
                f"Vision endpoint returned non-JSON body: {resp.text[:500]}"
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionError(
                f"Unexpected vision response shape: {json.dumps(data)[:500]}"
            ) from exc

        if not isinstance(content, str):
            raise VisionError(f"Vision response content was not text: {content!r}")
        return content
