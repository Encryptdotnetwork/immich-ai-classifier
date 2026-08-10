"""Tests for cache-scoped stale tag removal (--prune-tags).

This is the only destructive tag operation in the tool, and IMGCLASS-15 was a
silent tag-loss incident. So these tests are written from the angle of "what
must NEVER be removed" rather than "what gets removed".
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.reprocess import stale_tags_to_remove  # noqa: E402


@dataclass
class _Review:
    tag_name: str = "needs-review"
    album_name: str = "_Review"
    threshold: float = 0.55


@dataclass
class _Tax:
    marker_tag: str = "ai-classified"
    review: _Review = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.review is None:
            self.review = _Review()


TAX = _Tax()


# --- what must never be removed -------------------------------------------


def test_human_added_tags_are_never_touched():
    """A hand-added tag was never in the cache, so it can never be proposed."""
    # 'my-favourite' exists on the asset but not in the cache.
    assert stale_tags_to_remove(["talk"], ["geopolitics"], TAX) == ["talk"]
    # Nothing the caller didn't pass in cached_tags can appear in the output.
    out = stale_tags_to_remove([], ["geopolitics"], TAX)
    assert out == []


def test_source_tags_survive_even_when_cached_and_unplanned():
    """The exact damage from IMGCLASS-15. Never prune these."""
    out = stale_tags_to_remove(
        ["source:tiktok", "talk"], ["geopolitics"], TAX,
    )
    assert "source:tiktok" not in out
    assert out == ["talk"]


def test_source_prefix_match_is_case_insensitive():
    out = stale_tags_to_remove(["Source:TikTok"], ["geopolitics"], TAX)
    assert out == []


def test_marker_tag_survives():
    out = stale_tags_to_remove(["ai-classified", "talk"], ["geopolitics"], TAX)
    assert out == ["talk"]


def test_review_tag_survives_here():
    """reprocess strips needs-review separately, after the album move."""
    out = stale_tags_to_remove(["needs-review", "talk"], ["geopolitics"], TAX)
    assert out == ["talk"]


def test_tag_still_in_the_new_plan_is_kept():
    out = stale_tags_to_remove(["talk", "censorship"], ["censorship", "geopolitics"], TAX)
    assert out == ["talk"]


# --- what does get removed ------------------------------------------------


def test_the_real_world_case():
    """Dan's asset 34a363da: frame-derived tags replaced by transcript-derived."""
    cached = ["talk", "presentation", "youtube", "source:tiktok", "ai-classified"]
    planned = ["geopolitics", "richard-day", "population-control",
               "source:tiktok", "ai-classified"]
    out = stale_tags_to_remove(cached, planned, TAX)
    assert out == ["talk", "presentation", "youtube"]
    assert "source:tiktok" not in out
    assert "ai-classified" not in out


def test_empty_cache_removes_nothing():
    """First-ever pass on an asset: we own no tags, so we remove none."""
    assert stale_tags_to_remove([], ["geopolitics", "talk"], TAX) == []


def test_output_is_deduplicated():
    out = stale_tags_to_remove(["talk", "talk"], ["geopolitics"], TAX)
    assert out == ["talk"]


def test_identical_plan_removes_nothing():
    tags = ["geopolitics", "censorship"]
    assert stale_tags_to_remove(tags, tags, TAX) == []


def test_custom_taxonomy_names_are_honoured():
    """marker_tag and review.tag_name are user-configurable in categories.yaml."""
    tax = _Tax(marker_tag="sorted-by-bot", review=_Review(tag_name="check-me"))
    out = stale_tags_to_remove(
        ["sorted-by-bot", "check-me", "talk"], ["geopolitics"], tax,
    )
    assert out == ["talk"]
