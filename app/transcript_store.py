"""Persistent store for Whisper transcripts.

Two jobs, one table:

1. **Cache.** Transcription is the most expensive step in a run — roughly 7
   seconds per video on CPU, against sub-second for everything else the tool
   does locally. Without persistence every re-run pays it again, so a prompt
   tweak or a re-tag across a few thousand videos means hours of recomputing
   text that has not changed.
2. **Corpus.** The transcript itself is the deliverable for the Obsidian
   pipeline and for any future full-text search. Immich has nowhere to put it
   (measured: p50 1,039 chars, far past a usable description), so it lives here.

Invalidation follows the same rule as the asset cache in app/cache.py: identity
is Immich's own ``checksum``. If the file changes, Immich's checksum changes,
and we re-transcribe. Never recompute a hash ourselves.

A *model* change also invalidates. Switching WHISPER_MODEL from base to small is
an explicit request for better text, so silently serving the old, worse
transcript would defeat the point.

NO-SPEECH RESULTS ARE CACHED TOO. "This clip has no usable speech" costs a full
Whisper pass to establish, so forgetting it means re-running the expensive part
on exactly the assets that yield nothing. Those rows are stored with empty text
and ``has_speech = 0``.

Lives in the same SQLite file as the asset cache so there is one thing to back
up, on its own connection.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from .transcribe import Transcript

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    asset_id     TEXT PRIMARY KEY,
    content_hash TEXT,      -- Immich checksum; mismatch => re-transcribe
    model        TEXT,      -- Whisper model that produced it; change => redo
    has_speech   INTEGER,   -- 0 = checked and found nothing usable (still a hit)
    text         TEXT,
    language     TEXT,
    language_probability REAL,
    duration     REAL,
    word_count   INTEGER,
    segments     TEXT,      -- JSON array of {start, end, text}
    created_at   TEXT       -- ISO-8601 UTC
);
"""
# Lookups during a batch run are by asset_id (the primary key), but the Obsidian
# pipeline will want "everything with speech" in one sweep.
_INDEX = "CREATE INDEX IF NOT EXISTS idx_transcripts_speech ON transcripts(has_speech);"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TranscriptStore:
    """SQLite-backed transcript cache and corpus."""

    def __init__(self, data_dir: str, filename: str = "cache.db") -> None:
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, filename)
        # timeout guards against a lock if the asset cache is mid-write on its
        # own connection to the same file.
        self._conn = sqlite3.connect(self.path, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.execute(_INDEX)
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    # --- read ------------------------------------------------------------

    def get(
        self, asset_id: str, content_hash: Optional[str], model: str
    ) -> tuple[bool, Optional[Transcript]]:
        """Return (is_hit, transcript).

        ``(True, None)`` is a genuine hit meaning "we transcribed this and it
        has no usable speech" — distinct from ``(False, None)``, a miss. The
        caller must not re-transcribe on the former.
        """
        row = self._conn.execute(
            "SELECT * FROM transcripts WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        if row is None:
            self.misses += 1
            return False, None

        # Content changed under us, or the user asked for a different model.
        if row["content_hash"] != content_hash or row["model"] != model:
            self.misses += 1
            return False, None

        self.hits += 1
        if not row["has_speech"]:
            return True, None

        try:
            segments = json.loads(row["segments"]) if row["segments"] else []
        except (json.JSONDecodeError, TypeError):
            segments = []
        return True, Transcript(
            text=row["text"] or "",
            language=row["language"],
            language_probability=row["language_probability"],
            duration=row["duration"],
            segments=segments,
            model=row["model"] or "",
        )

    # --- write -----------------------------------------------------------

    def put(
        self, asset_id: str, content_hash: Optional[str], model: str,
        transcript: Optional[Transcript],
    ) -> None:
        """Record a transcript, or a verified absence of speech (transcript=None)."""
        if transcript is None:
            values = (asset_id, content_hash, model, 0, "", None, None, None, 0, "[]", _now())
        else:
            values = (
                asset_id, content_hash, model, 1, transcript.text,
                transcript.language, transcript.language_probability,
                transcript.duration, transcript.word_count,
                json.dumps(transcript.segments, ensure_ascii=False), _now(),
            )
        self._conn.execute(
            """INSERT INTO transcripts
               (asset_id, content_hash, model, has_speech, text, language,
                language_probability, duration, word_count, segments, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(asset_id) DO UPDATE SET
                 content_hash=excluded.content_hash, model=excluded.model,
                 has_speech=excluded.has_speech, text=excluded.text,
                 language=excluded.language,
                 language_probability=excluded.language_probability,
                 duration=excluded.duration, word_count=excluded.word_count,
                 segments=excluded.segments, created_at=excluded.created_at""",
            values,
        )
        self._conn.commit()

    # --- corpus access (for the Obsidian pipeline) -----------------------

    def iter_with_speech(self, min_words: int = 0):
        """Yield every stored transcript that has usable speech.

        This is the read path the Obsidian note generator will use — it wants
        the corpus, not a per-asset cache lookup.
        """
        rows = self._conn.execute(
            "SELECT * FROM transcripts WHERE has_speech = 1 AND word_count >= ? "
            "ORDER BY created_at",
            (min_words,),
        )
        for row in rows:
            yield dict(row)

    def stats(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(has_speech) AS with_speech, "
            "SUM(word_count) AS words FROM transcripts"
        ).fetchone()
        return {
            "total": row["total"] or 0,
            "with_speech": row["with_speech"] or 0,
            "no_speech": (row["total"] or 0) - (row["with_speech"] or 0),
            "words": row["words"] or 0,
        }

    def close(self) -> None:
        self._conn.close()


def build_transcript_store(cfg: Any) -> Optional[TranscriptStore]:
    """Return a store, or None when Whisper is switched off (nothing to cache)."""
    if not cfg.whisper.enabled:
        return None
    return TranscriptStore(cfg.app_data_dir)
