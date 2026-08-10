"""Tests for the transcript store (cache + corpus).

The store's job is to make Whisper run once per asset. The tests that matter are
the invalidation rules and the no-speech case, because getting either wrong
means either serving stale text or silently re-paying the most expensive step in
the pipeline.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.transcribe import Transcript  # noqa: E402
from app.transcript_store import TranscriptStore  # noqa: E402

T = Transcript(
    text="this clip explains position sizing",
    language="en", language_probability=0.98, duration=31.0,
    segments=[{"start": 0.0, "end": 2.0, "text": "this clip explains position sizing"}],
    model="base",
)


def _store():
    return TranscriptStore(tempfile.mkdtemp(prefix="tstore-"))


# --- round trip -----------------------------------------------------------


def test_round_trip_preserves_everything():
    s = _store()
    s.put("a1", "hash1", "base", T)
    hit, got = s.get("a1", "hash1", "base")
    assert hit is True
    assert got.text == T.text
    assert got.language == "en"
    assert got.duration == 31.0
    assert got.segments == T.segments
    assert got.word_count == 5


def test_miss_on_unknown_asset():
    s = _store()
    hit, got = s.get("nope", "hash1", "base")
    assert (hit, got) == (False, None)


# --- invalidation ---------------------------------------------------------


def test_changed_checksum_invalidates():
    """Same rule as the asset cache: Immich's checksum is identity."""
    s = _store()
    s.put("a1", "hash1", "base", T)
    hit, got = s.get("a1", "hash2", "base")
    assert (hit, got) == (False, None)


def test_changed_model_invalidates():
    """Switching base -> small is a request for better text, not a cache hit."""
    s = _store()
    s.put("a1", "hash1", "base", T)
    hit, got = s.get("a1", "hash1", "small")
    assert (hit, got) == (False, None)


def test_reput_overwrites_rather_than_duplicating():
    s = _store()
    s.put("a1", "hash1", "base", T)
    better = Transcript(text="a better transcript", language="en",
                        language_probability=0.99, duration=31.0, model="small")
    s.put("a1", "hash1", "small", better)
    hit, got = s.get("a1", "hash1", "small")
    assert hit and got.text == "a better transcript"
    assert s.stats()["total"] == 1


# --- the no-speech case ---------------------------------------------------


def test_no_speech_is_cached_as_a_hit():
    """Establishing 'no speech' costs a full Whisper pass. Never redo it."""
    s = _store()
    s.put("a1", "hash1", "base", None)
    hit, got = s.get("a1", "hash1", "base")
    assert hit is True      # a HIT...
    assert got is None      # ...whose answer is "nothing to transcribe"


def test_no_speech_still_invalidates_on_checksum_change():
    s = _store()
    s.put("a1", "hash1", "base", None)
    hit, _ = s.get("a1", "hash2", "base")
    assert hit is False


# --- corpus ---------------------------------------------------------------


def test_iter_with_speech_excludes_silent_assets():
    s = _store()
    s.put("a1", "h", "base", T)
    s.put("a2", "h", "base", None)
    rows = list(s.iter_with_speech())
    assert [r["asset_id"] for r in rows] == ["a1"]


def test_iter_with_speech_min_words_filter():
    s = _store()
    s.put("a1", "h", "base", T)                                   # 5 words
    s.put("a2", "h", "base", Transcript(text="hi", language="en",
                                        language_probability=1.0, duration=1.0))
    assert len(list(s.iter_with_speech(min_words=3))) == 1


def test_stats_counts_both_kinds():
    s = _store()
    s.put("a1", "h", "base", T)
    s.put("a2", "h", "base", None)
    st = s.stats()
    assert st["total"] == 2
    assert st["with_speech"] == 1
    assert st["no_speech"] == 1
    assert st["words"] == 5


def test_hit_and_miss_counters_track_usage():
    s = _store()
    s.put("a1", "h", "base", T)
    s.get("a1", "h", "base")     # hit
    s.get("a1", "other", "base")  # miss, checksum
    s.get("a2", "h", "base")     # miss, unknown
    assert (s.hits, s.misses) == (1, 2)


def test_survives_reopen():
    """It is a cache on disk; a new process must see prior results."""
    d = tempfile.mkdtemp(prefix="tstore-")
    TranscriptStore(d).put("a1", "h", "base", T)
    hit, got = TranscriptStore(d).get("a1", "h", "base")
    assert hit and got.text == T.text


def test_shares_the_file_with_the_asset_cache():
    """One file to back up. Both tables must coexist."""
    from app.cache import Cache

    d = tempfile.mkdtemp(prefix="tstore-")
    cache = Cache(d)
    store = TranscriptStore(d)
    store.put("a1", "h", "base", T)
    cache.upsert(asset_id="a1", content_hash="h", category="Tech", album="Tech",
                 tags=["a"], confidence=0.9, needs_review=False)
    assert store.get("a1", "h", "base")[0] is True
    assert cache.get("a1")["category"] == "Tech"
    assert store.path == cache.path
