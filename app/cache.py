"""SQLite cache of classified assets.

One row per asset. The cache lets a re-run skip assets that are unchanged
(no inference, no writes) and lets the manual-fix guard notice when a human has
moved an asset to a different album.

Content identity uses Immich's own ``checksum`` field (do NOT recompute) — if
the file changes, Immich's checksum changes, and we re-process.

DB lives under the app data dir (container path, e.g. /data/cache.db, which
compose maps to the host dir set by APP_DATA_HOST_PATH).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    asset_id     TEXT PRIMARY KEY,
    content_hash TEXT,
    category     TEXT,
    album        TEXT,
    tags         TEXT,      -- JSON array of slug tags applied
    confidence   REAL,
    needs_review INTEGER,   -- 0/1
    processed_at TEXT       -- ISO-8601 UTC
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Cache:
    def __init__(self, data_dir: str, filename: str = "cache.db") -> None:
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, filename)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, asset_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        if row is None:
            return None
        rec = dict(row)
        rec["tags"] = json.loads(rec["tags"]) if rec["tags"] else []
        rec["needs_review"] = bool(rec["needs_review"])
        return rec

    def upsert(
        self, asset_id: str, content_hash: Optional[str], category: str,
        album: str, tags: list[str], confidence: float, needs_review: bool,
    ) -> None:
        self._conn.execute(
            """INSERT INTO assets
                 (asset_id, content_hash, category, album, tags, confidence,
                  needs_review, processed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(asset_id) DO UPDATE SET
                 content_hash=excluded.content_hash, category=excluded.category,
                 album=excluded.album, tags=excluded.tags,
                 confidence=excluded.confidence, needs_review=excluded.needs_review,
                 processed_at=excluded.processed_at""",
            (asset_id, content_hash, category, album, json.dumps(tags),
             float(confidence), 1 if needs_review else 0, _now()),
        )
        self._conn.commit()

    def update_album(self, asset_id: str, album: str) -> None:
        """Record that a human moved the asset to a different album."""
        self._conn.execute(
            "UPDATE assets SET album = ?, processed_at = ? WHERE asset_id = ?",
            (album, _now(), asset_id),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
