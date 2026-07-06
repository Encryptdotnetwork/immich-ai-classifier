"""Classification (dry-run): image -> {category, tags, confidence}.

This module ONLY classifies. It performs no Immich writes — no album moves, no
tags, no description updates. Those are later stages.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

from .config import Config
from .signals import SignalError, gather_signals
from .taxonomy import load_taxonomy, normalize_source

# Categories, the system prompt, and the catch-all all come from the loaded
# taxonomy (config/categories.yaml) — see app/taxonomy.py. Nothing here is
# hardcoded; edit the YAML to change the taxonomy.

_BASE_USER_TEXT = (
    "Classify the attached image according to the system instructions. "
    "Respond with ONLY the JSON object — no prose, no markdown fences."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def build_user_text(ocr_text: str | None) -> str:
    """User message. Includes OCR as an optional hint when present (never required)."""
    if ocr_text:
        return (
            _BASE_USER_TEXT
            + "\n\nOCR text was detected in the image. Treat it as a hint only; "
            "it may be partial or noisy:\n\"\"\"\n" + ocr_text + "\n\"\"\""
        )
    return _BASE_USER_TEXT


def parse_classification(raw: str) -> dict[str, Any]:
    """Parse the model's reply into a normalised result.

    Strips markdown fences and tolerates surrounding prose. On parse failure,
    defaults to the catch-all category, confidence 0.0, source 'unknown', and
    flags it (parse_ok=False). Valid categories come from the loaded taxonomy.
    """
    tax = load_taxonomy()
    result: dict[str, Any] = {
        "category": tax.catch_all_name,
        "source": "unknown",
        "tags": [],
        "confidence": 0.0,
        "parse_ok": False,
        "category_valid": False,
        "raw_category": None,
    }
    if not isinstance(raw, str):
        return result

    text = raw.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):  # slice to outermost braces if prose wraps it
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1 and j > i:
            text = text[i : j + 1]

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return result
    if not isinstance(data, dict):
        return result

    raw_category = data.get("category")
    valid = raw_category in tax.names_set

    tags: list[str] = []
    raw_tags = data.get("tags")
    if isinstance(raw_tags, list):  # ignore non-list (e.g. a bare string)
        for t in raw_tags:
            if t is None:
                continue
            s = str(t).strip().lower()
            if s and s not in tags:
                tags.append(s)

    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    result.update(
        category=raw_category if valid else tax.catch_all_name,
        source=normalize_source(data.get("source")),
        tags=tags,
        confidence=confidence,
        parse_ok=True,
        category_valid=valid,
        raw_category=raw_category,
    )
    return result


def _select_image_bytes(signals: dict[str, Any]) -> bytes:
    """Pick the bytes to send: the image, or a representative middle frame."""
    if signals.get("image_b64"):
        return base64.b64decode(signals["image_b64"])
    frames = signals.get("frames") or []
    if frames:
        # Single representative still for this dry-run; multi-frame video
        # reasoning is a later refinement.
        return base64.b64decode(frames[len(frames) // 2])
    raise SignalError(
        f"No image data to classify for asset {signals.get('asset_id')} "
        f"(type {signals.get('type')})."
    )


def classify_asset(
    asset: dict[str, Any], cfg: Config, vision_client: Any, client: Any
) -> dict[str, Any]:
    """Gather signals, run the vision model, parse the result. NO writes.

    ``client`` is the ImmichClient — needed so signal gathering can fetch the
    JPEG preview for HEIC/HEIF assets the vision endpoint can't decode.
    """
    signals = gather_signals(asset, cfg, client)
    image_bytes = _select_image_bytes(signals)
    ocr_text = signals.get("ocr_text")

    raw = vision_client.vision_complete(
        image_bytes, load_taxonomy().system_prompt, build_user_text(ocr_text)
    )

    result = parse_classification(raw)
    result["raw"] = raw
    result["ocr_available"] = bool(ocr_text)
    result["asset_type"] = signals.get("type")
    result["num_frames"] = len(signals["frames"]) if signals.get("frames") else None
    return result