"""Regression tests for the Immich 3.0 breaking changes.

Both bugs were silent: neither raised, both just quietly did the wrong thing.
These tests assert on the request bodies we send and on the refusal behaviour,
because that is where the silence was.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.batch import DEFAULT_SEARCH_VISIBILITY, search_paginated  # noqa: E402
from app.config import Config, InferenceRole  # noqa: E402
from app.dedupe_albums import EnumerationMismatch, _album_asset_ids  # noqa: E402


class _RecordingClient:
    """Captures every search body, returns a fixed page of assets."""

    def __init__(self, items=None, total=None):
        self.bodies = []
        self._items = items if items is not None else [{"id": "a1"}, {"id": "a2"}]
        self._total = total if total is not None else len(self._items)

    def search_metadata(self, body):
        self.bodies.append(body)
        return {"assets": {"total": self._total, "items": self._items, "nextPage": None}}


# --- v3 change 1: search visibility default --------------------------------


def test_visibility_is_sent_explicitly_by_default():
    """The whole bug: omitting this let v3 widen a run to archived assets."""
    c = _RecordingClient()
    search_paginated(c, {"albumIds": ["alb1"]}, None)
    assert c.bodies[0]["visibility"] == DEFAULT_SEARCH_VISIBILITY
    assert DEFAULT_SEARCH_VISIBILITY == "timeline"


def test_visibility_is_sent_on_every_page_not_just_the_first():
    items = [{"id": f"a{i}"} for i in range(3)]

    class Paging(_RecordingClient):
        def search_metadata(self, body):
            self.bodies.append(body)
            page = body["page"]
            return {"assets": {"total": 6, "items": items,
                               "nextPage": str(page + 1) if page < 2 else None}}

    c = Paging()
    search_paginated(c, {"albumIds": ["alb1"]}, None)
    assert len(c.bodies) == 2
    assert all(b["visibility"] == "timeline" for b in c.bodies)


class _Paged:
    """Serves `total` assets in pages, reporting Immich's per-page 'total'."""

    def __init__(self, total, page_size=500):
        self.total, self.page_size, self.calls = total, page_size, 0

    def search_metadata(self, body):
        self.calls += 1
        page, size = body["page"], body["size"]
        start = (page - 1) * size
        items = [{"id": f"a{i}"} for i in range(start, min(start + size, self.total))]
        more = start + size < self.total
        return {"assets": {
            "total": len(items),           # per-page, which was the whole bug
            "items": items,
            "nextPage": str(page + 1) if more else None,
        }}


def test_scope_size_is_the_real_count_not_the_page_size():
    """Reported 500 for a 1,300-asset scope, so a --limit run couldn't size the job."""
    c = _Paged(total=1300)
    assets, total = search_paginated(c, {"albumIds": ["a"]}, limit=20)
    assert len(assets) == 20
    assert total == 1300          # not 500, and not 20
    assert c.calls == 3           # walked to the end, cheaply


def test_counting_pages_costs_no_extra_calls_when_unlimited():
    c = _Paged(total=1300)
    assets, total = search_paginated(c, {"albumIds": ["a"]}, limit=None)
    assert (len(assets), total, c.calls) == (1300, 1300, 3)


def test_count_total_false_stops_at_the_limit():
    c = _Paged(total=1300)
    assets, total = search_paginated(c, {"albumIds": ["a"]}, limit=20, count_total=False)
    assert len(assets) == 20
    assert c.calls == 1           # no extra paging


def test_single_page_scope_is_exact():
    c = _Paged(total=114)
    assert search_paginated(c, {"albumIds": ["a"]}, limit=None)[1] == 114


def test_explicit_visibility_argument_overrides_the_default():
    c = _RecordingClient()
    search_paginated(c, {"albumIds": ["alb1"]}, None, visibility="archive")
    assert c.bodies[0]["visibility"] == "archive"


def test_body_filter_wins_over_the_argument():
    c = _RecordingClient()
    search_paginated(c, {"albumIds": ["a"], "visibility": "archive"}, None,
                     visibility="timeline")
    assert c.bodies[0]["visibility"] == "archive"


def test_none_opts_into_v3_any_visibility():
    """Deliberate omission must stay possible, e.g. for album membership."""
    c = _RecordingClient()
    search_paginated(c, {"albumIds": ["alb1"]}, None, visibility=None)
    assert "visibility" not in c.bodies[0]


def test_config_default_is_timeline_and_is_overridable(monkeypatch):
    cfg = Config(
        immich_url="http://s", immich_api_key="k", immich_internal_prefix="/u",
        local_mount="/m", vision=InferenceRole("", "", ""), text=InferenceRole("", "", ""),
        tag_verify_max_retries=1, tag_verify_delay=0.0, source_album="Unsorted",
        app_data_dir="/tmp", batch_group_size=25, batch_pause=0.0,
    )
    assert cfg.search_visibility == "timeline"

    for var in ("IMMICH_URL", "IMMICH_API_KEY", "IMMICH_INTERNAL_PREFIX", "LOCAL_MOUNT"):
        monkeypatch.setenv(var, "x")
    monkeypatch.setenv("SEARCH_VISIBILITY", "archive")
    assert Config.from_env().search_visibility == "archive"


# --- v3 change 2: AlbumResponseDto.assets removed ---------------------------


def test_album_membership_no_longer_reads_the_removed_assets_key():
    """Must enumerate via search, not via get_album()['assets']."""
    c = _RecordingClient(items=[{"id": "x1"}, {"id": "x2"}, {"id": "x3"}])
    ids = _album_asset_ids(c, "alb1", expected=3)
    assert ids == ["x1", "x2", "x3"]
    assert c.bodies[0]["albumIds"] == ["alb1"]
    # Membership is membership: archived assets are still in the album.
    assert "visibility" not in c.bodies[0]


def test_undercount_refuses_rather_than_stranding_assets():
    """The dangerous case. On v3 this used to return [] and delete the album."""
    c = _RecordingClient(items=[])
    with pytest.raises(EnumerationMismatch):
        _album_asset_ids(c, "alb1", expected=146)


def test_partial_enumeration_also_refuses():
    c = _RecordingClient(items=[{"id": "x1"}])
    with pytest.raises(EnumerationMismatch):
        _album_asset_ids(c, "alb1", expected=5)


def test_overcount_is_tolerated():
    """assetCount can lag behind reality; only UNDER-counting strands assets."""
    c = _RecordingClient(items=[{"id": "x1"}, {"id": "x2"}])
    assert len(_album_asset_ids(c, "alb1", expected=1)) == 2


def test_genuinely_empty_album_is_fine():
    c = _RecordingClient(items=[])
    assert _album_asset_ids(c, "alb1", expected=0) == []
