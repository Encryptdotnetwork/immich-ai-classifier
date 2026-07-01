"""Signal gathering: turn an Immich asset into model-ready inputs.

For an image: read the file off the read-only mount (via path translation) and
base64-encode it. For a video: extract a few evenly-spaced frames (MoviePy).
OCR text is included only when Immich already has it on the asset — never
assumed (see [[immich-ocr-via-exifinfo]]). Whisper is stubbed.

This module reads only; it never writes to Immich storage.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Optional

from .config import Config
from .paths import translate_path

NUM_VIDEO_FRAMES = 5


class SignalError(RuntimeError):
    """Raised when an asset's inputs cannot be gathered (missing file, etc.)."""


def find_ocr_text(asset: dict[str, Any]) -> tuple[Optional[str], str]:
    """Locate OCR text in an asset payload.

    In Immich 2.7.5 OCR lands at ``exifInfo.ocrText`` once the OCR job has run;
    we also check the top level and fall back to a recursive scan. Returns
    (value, where_found); value is None if no ocrText key exists at all.
    """
    if "ocrText" in asset:
        return asset.get("ocrText"), "asset.ocrText"

    exif = asset.get("exifInfo")
    if isinstance(exif, dict) and "ocrText" in exif:
        return exif.get("ocrText"), "asset.exifInfo.ocrText"

    stack: list[tuple[str, Any]] = [("asset", asset)]
    while stack:
        path, node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "ocrText":
                    return v, f"{path}.{k}"
                stack.append((f"{path}.{k}", v))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                stack.append((f"{path}[{i}]", v))
    return None, "<not present in payload>"


def _read_local_file(asset: dict[str, Any], cfg: Config) -> str:
    """Translate originalPath -> local mount path; return it, or raise."""
    original = asset.get("originalPath")
    if not original:
        raise SignalError(f"Asset {asset.get('id')} has no originalPath.")
    local_path = translate_path(original, cfg.immich_internal_prefix, cfg.local_mount)
    if not os.path.isfile(local_path):
        raise SignalError(
            f"Asset file not found at {local_path}. Check the read-only mount "
            f"and IMMICH_INTERNAL_PREFIX (raw path: {original})."
        )
    return local_path


def _image_b64(local_path: str) -> str:
    with open(local_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _extract_frames(local_path: str, num_frames: int = NUM_VIDEO_FRAMES) -> list[str]:
    """Extract evenly-spaced JPEG frames as base64. Lazy imports so the image
    path never depends on MoviePy/Pillow/ffmpeg being present."""
    try:  # MoviePy 2.x layout, then 1.x
        from moviepy import VideoFileClip  # type: ignore
    except ImportError:  # pragma: no cover - depends on installed version
        from moviepy.editor import VideoFileClip  # type: ignore
    from PIL import Image

    clip = VideoFileClip(local_path)
    try:
        duration = clip.duration or 0.0
        if duration <= 0:
            times = [0.0]
        else:
            fractions = (0.1, 0.3, 0.5, 0.7, 0.9)[:num_frames]
            times = [duration * f for f in fractions]
        frames_b64: list[str] = []
        for t in times:
            frame = clip.get_frame(t)  # HxWx3 RGB uint8 ndarray
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="JPEG", quality=85)
            frames_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        return frames_b64
    finally:
        clip.close()


def gather_signals(asset: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """Build the model-ready signal bundle for one asset.

    Returns:
        {asset_id, type, image_b64 | frames, ocr_text, transcript}
        - image assets: ``image_b64`` set, ``frames`` None.
        - video assets: ``frames`` set (list of base64 JPEGs), ``image_b64`` None.
    """
    asset_type = asset.get("type")
    local_path = _read_local_file(asset, cfg)

    ocr_value, _where = find_ocr_text(asset)
    ocr_text = ocr_value or None  # treat empty string as absent

    signals: dict[str, Any] = {
        "asset_id": asset.get("id"),
        "type": asset_type,
        "image_b64": None,
        "frames": None,
        "ocr_text": ocr_text,
        "transcript": None,  # TODO: whisper — stubbed, do not block this stage
    }

    if asset_type == "VIDEO":
        signals["frames"] = _extract_frames(local_path)
    else:  # treat anything non-VIDEO as an image
        signals["image_b64"] = _image_b64(local_path)

    return signals
