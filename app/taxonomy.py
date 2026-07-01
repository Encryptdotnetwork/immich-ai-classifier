"""Taxonomy: the user-editable category config (YAML), loaded ONCE at startup.

Single source of truth for categories, the review machinery, the marker tag, and
source detection. The prompt MACHINERY (JSON output contract, confidence/tag
rules, source-detection instruction) lives here in code as a fixed TEMPLATE; the
user's categories are injected into it. Users edit the YAML, never the template.

Loaded via load_taxonomy() (cached singleton, like load_config()). Path comes
from the CATEGORIES_FILE env var, else the shipped config/categories.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

try:  # PyYAML is required; surface a clear error if missing.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

# The fixed set of source platforms the model may return.
SOURCES = ("tiktok", "instagram", "facebook", "youtube", "twitter", "reddit", "other", "unknown")
_SOURCE_ALIASES = {"x": "twitter", "yt": "youtube", "ig": "instagram", "fb": "facebook"}

# <repo>/config/categories.yaml  (==> /app/config/categories.yaml in the container)
DEFAULT_CATEGORIES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "categories.yaml"
)


class TaxonomyError(RuntimeError):
    """Raised on an invalid category config. Message names the offending item."""


def normalize_source(raw: Any) -> str:
    """Map a model's source string onto the fixed SOURCES set."""
    if not isinstance(raw, str):
        return "unknown"
    s = raw.strip().lower()
    if not s:
        return "unknown"
    if s in SOURCES:
        return s
    if s in _SOURCE_ALIASES:
        return _SOURCE_ALIASES[s]
    return "other"


@dataclass(frozen=True)
class Category:
    name: str
    description: str
    notes: str = ""
    examples: tuple[str, ...] = ()
    catch_all: bool = False


@dataclass(frozen=True)
class Review:
    threshold: float
    album_name: str
    tag_name: str


@dataclass(frozen=True)
class SourceDetection:
    enabled: bool
    tag_sources: bool
    process_only: Optional[str]


@dataclass(frozen=True)
class Taxonomy:
    categories: tuple[Category, ...]
    marker_tag: str
    review: Review
    source_detection: SourceDetection
    system_prompt: str = field(repr=False)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.categories]

    @property
    def names_set(self) -> set[str]:
        return {c.name for c in self.categories}

    @property
    def catch_all_name(self) -> str:
        return next(c.name for c in self.categories if c.catch_all)


# --------------------------------------------------------------------------
# Prompt template (machinery; users never edit this). Categories injected.
# --------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are classifying a single image (a social-media video frame or screenshot)
into exactly ONE category, plus free-form tags and the source platform.

Categories (choose exactly one):
{categories_block}

Decide by INTENT and read each category's Notes carefully — a single on-screen
element or a surface keyword does NOT decide the category. When two categories
seem close, the Notes tell you how to choose. If nothing clearly fits, use
"{catch_all}".

Identify the SOURCE platform if recognisable from on-screen UI, watermarks,
logos, or caption/comment styling. Return exactly one of: {sources}. Use
"unknown" if you genuinely cannot tell, and "other" for a platform not in that
list.

Also produce:
- tags: short lowercase free-form tags for specifics (e.g. "bitcoin",
  "chart-pattern", "5min", "breakout", "recipe", "keto"). No fixed vocabulary.
- confidence: 0.0-1.0 for the CATEGORY. Be honest; if the category is genuinely
  ambiguous, score below {threshold}.

Return ONLY this JSON, nothing else:
{{"category": "<one of the {n} category names above>", "source": "<one platform>", "tags": ["..."], "confidence": 0.0}}"""


def _render_category(c: Category) -> str:
    lines = [f"- {c.name}: {c.description}"]
    if c.notes:
        lines.append(f"  Notes: {c.notes}")
    if c.examples:
        lines.append(f"  Examples: {', '.join(c.examples)}")
    return "\n".join(lines)


def _assemble_prompt(categories: list[Category], catch_all_name: str, threshold: float) -> str:
    # Non-catch-all first, catch-all rendered LAST as the fallback.
    ordered = [c for c in categories if not c.catch_all] + [c for c in categories if c.catch_all]
    block = "\n".join(_render_category(c) for c in ordered)
    return _PROMPT_TEMPLATE.format(
        categories_block=block, catch_all=catch_all_name,
        sources=", ".join(SOURCES), threshold=f"{threshold:.2f}", n=len(categories),
    )


# --------------------------------------------------------------------------
# Validation + build. Every failure names the offending item and refuses to run.
# --------------------------------------------------------------------------

def _build(data: Any, src: str) -> Taxonomy:
    if not isinstance(data, dict):
        raise TaxonomyError(f"{src}: top level must be a mapping (key: value).")

    raw_cats = data.get("categories")
    if not isinstance(raw_cats, list) or len(raw_cats) < 2:
        raise TaxonomyError(f"{src}: 'categories' must be a list of at least 2 categories.")

    categories: list[Category] = []
    seen: set[str] = set()
    catch_all_count = 0
    for i, rc in enumerate(raw_cats, 1):
        if not isinstance(rc, dict):
            raise TaxonomyError(f"{src}: category #{i} must be a mapping.")
        name = str(rc.get("name") or "").strip()
        if not name:
            raise TaxonomyError(f"{src}: category #{i} is missing a 'name'.")
        if name in seen:
            raise TaxonomyError(f"{src}: category '{name}' defined twice.")
        seen.add(name)
        desc = str(rc.get("description") or "").strip()
        if not desc:
            raise TaxonomyError(f"{src}: category '{name}' is missing a 'description'.")
        examples = rc.get("examples") or []
        if not isinstance(examples, list):
            raise TaxonomyError(f"{src}: category '{name}' 'examples' must be a list.")
        is_catch = bool(rc.get("catch_all", False))
        catch_all_count += int(is_catch)
        categories.append(Category(
            name=name, description=desc, notes=str(rc.get("notes") or "").strip(),
            examples=tuple(str(e).strip() for e in examples), catch_all=is_catch,
        ))

    if catch_all_count != 1:
        raise TaxonomyError(
            f"{src}: exactly one category must have 'catch_all: true' (found {catch_all_count})."
        )

    review_raw = data.get("review") or {}
    if not isinstance(review_raw, dict):
        raise TaxonomyError(f"{src}: 'review' must be a mapping.")
    try:
        threshold = float(review_raw.get("threshold", 0.70))
    except (TypeError, ValueError):
        raise TaxonomyError(f"{src}: review.threshold must be a number.") from None
    if not 0.0 <= threshold <= 1.0:
        raise TaxonomyError(f"{src}: review.threshold must be between 0.0 and 1.0 (got {threshold}).")
    review = Review(
        threshold=threshold,
        album_name=str(review_raw.get("album_name") or "_Review").strip(),
        tag_name=str(review_raw.get("tag_name") or "needs-review").strip(),
    )
    marker_tag = str(data.get("marker_tag") or "ai-classified").strip()

    names = {c.name for c in categories}
    for label, value in (
        ("review.album_name", review.album_name),
        ("review.tag_name", review.tag_name),
        ("marker_tag", marker_tag),
    ):
        if value in names:
            raise TaxonomyError(f"{src}: {label} '{value}' collides with a category name.")

    sd_raw = data.get("source_detection") or {}
    if not isinstance(sd_raw, dict):
        raise TaxonomyError(f"{src}: 'source_detection' must be a mapping.")
    process_only = sd_raw.get("process_only")
    if process_only is not None:
        process_only = str(process_only).strip().lower()
        if process_only not in SOURCES:
            raise TaxonomyError(
                f"{src}: source_detection.process_only '{process_only}' is not a known source "
                f"(one of: {', '.join(SOURCES)})."
            )
    source_detection = SourceDetection(
        enabled=bool(sd_raw.get("enabled", True)),
        tag_sources=bool(sd_raw.get("tag_sources", False)),
        process_only=process_only,
    )

    catch_all_name = next(c.name for c in categories if c.catch_all)
    return Taxonomy(
        categories=tuple(categories), marker_tag=marker_tag, review=review,
        source_detection=source_detection,
        system_prompt=_assemble_prompt(categories, catch_all_name, threshold),
    )


_TAXONOMY: Optional[Taxonomy] = None


def load_taxonomy(path: Optional[str] = None) -> Taxonomy:
    """Return the process-wide Taxonomy, building it from YAML once.

    If `path` is given it is (re)loaded and cached (used by tests); otherwise the
    cached instance is returned, or built from CATEGORIES_FILE / the default.
    """
    global _TAXONOMY
    if path is None and _TAXONOMY is not None:
        return _TAXONOMY
    if yaml is None:
        raise TaxonomyError("PyYAML is not installed; cannot load the category config.")
    resolved = path or os.environ.get("CATEGORIES_FILE", "").strip() or DEFAULT_CATEGORIES_FILE
    if not os.path.isfile(resolved):
        raise TaxonomyError(
            f"Category config not found at {resolved!r}. "
            f"Set CATEGORIES_FILE or provide config/categories.yaml."
        )
    try:
        with open(resolved, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # type: ignore[union-attr]
        raise TaxonomyError(f"{resolved}: invalid YAML — {exc}") from exc
    tax = _build(data, resolved)
    _TAXONOMY = tax
    return tax


def reset_taxonomy() -> None:
    """Clear the cached taxonomy (tests)."""
    global _TAXONOMY
    _TAXONOMY = None
