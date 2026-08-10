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


# --- marker-scoped prune (--prune-all-tags) -------------------------------

# Dan's real asset c23f7647 after the cache-scoped run: the residue the cache
# could not see, alongside the tags the transcript produced.
LIVE = ["censorship", "source:tiktok", "postmodernism", "social-commentary",
        "animation", "free-speech", "political-commentary", "freedom-of-speech",
        "ai-classified", "politics"]
PLANNED = ["politics", "social-commentary", "free-speech", "postmodernism",
           "source:tiktok", "ai-classified"]


def test_marker_scope_catches_the_residue_the_cache_missed():
    """The whole point of widening: kill 'animation' and 'freedom-of-speech'."""
    out = stale_tags_to_remove(
        ["joke", "meme"], PLANNED, TAX, current_tags=LIVE, marker_scoped=True,
    )
    assert set(out) == {"censorship", "animation", "political-commentary",
                        "freedom-of-speech"}
    assert "source:tiktok" not in out
    assert "ai-classified" not in out
    assert "free-speech" not in out  # in the new plan, stays


def test_marker_scope_leaves_the_asset_matching_the_plan_exactly():
    """After removal the asset should hold plan tags plus protected ones only."""
    out = stale_tags_to_remove(
        [], PLANNED, TAX, current_tags=LIVE, marker_scoped=True,
    )
    remaining = [t for t in LIVE if t not in out]
    assert sorted(remaining) == sorted(PLANNED)


def test_no_marker_falls_back_to_cache_scope():
    """An asset we never filed is not ours to tidy, whatever the flag says."""
    live_without_marker = [t for t in LIVE if t != "ai-classified"]
    out = stale_tags_to_remove(
        ["joke"], PLANNED, TAX,
        current_tags=live_without_marker, marker_scoped=True,
    )
    assert out == ["joke"]  # cache-scoped result, not the wide one


def test_marker_scope_still_protects_source_tags():
    out = stale_tags_to_remove(
        [], ["politics"], TAX,
        current_tags=["source:tiktok", "source:youtube", "talk", "ai-classified"],
        marker_scoped=True,
    )
    assert out == ["talk"]


def test_keep_tags_are_protected_under_marker_scope():
    out = stale_tags_to_remove(
        [], ["politics"], TAX,
        current_tags=["favourite", "animation", "ai-classified"],
        marker_scoped=True, keep_tags=("favourite",),
    )
    assert out == ["animation"]


def test_keep_tags_also_apply_to_cache_scope():
    out = stale_tags_to_remove(
        ["favourite", "talk"], ["politics"], TAX, keep_tags=("favourite",),
    )
    assert out == ["talk"]


def test_marker_scope_off_by_default_reproduces_old_behaviour():
    out = stale_tags_to_remove(["joke", "meme"], PLANNED, TAX, current_tags=LIVE)
    assert out == ["joke", "meme"]


def test_empty_live_tags_falls_back_to_cache():
    out = stale_tags_to_remove(["talk"], PLANNED, TAX,
                               current_tags=[], marker_scoped=True)
    assert out == ["talk"]


def test_custom_taxonomy_names_are_honoured():
    """marker_tag and review.tag_name are user-configurable in categories.yaml."""
    tax = _Tax(marker_tag="sorted-by-bot", review=_Review(tag_name="check-me"))
    out = stale_tags_to_remove(
        ["sorted-by-bot", "check-me", "talk"], ["geopolitics"], tax,
    )
    assert out == ["talk"]
