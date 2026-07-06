"""Minimal, read-only Immich API client.

Scope for this task: the read path only. No writes (descriptions, tags, albums)
are implemented here yet — that's a later task. Authentication is the Immich
``x-api-key`` header.

Endpoints used (the correct ones for Immich 2.7.5):
    GET /api/assets          -> list_assets()
    GET /api/assets/{id}     -> get_asset(id)   (incl. originalPath, exif, OCR)
    GET /api/tags            -> get_tags()
    GET /api/albums          -> get_albums()
"""

from __future__ import annotations

from typing import Any, Optional

import requests

from .config import Config

DEFAULT_TIMEOUT = 30  # seconds


class ImmichClient:
    """Thin wrapper over the Immich REST API, authenticated by API key."""

    def __init__(self, config: Config, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._base = config.api_base  # '<host>/api', no trailing slash
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "x-api-key": config.immich_api_key,
                "Accept": "application/json",
            }
        )

    # --- internal helpers ------------------------------------------------

    def _request(
        self, method: str, path: str, *, params: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
    ) -> Any:
        url = f"{self._base}/{path.lstrip('/')}"
        resp = self._session.request(
            method, url, params=params, json=json, timeout=self._timeout
        )
        resp.raise_for_status()
        if not resp.content:
            return None
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return None

    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params=params)

    # --- read path -------------------------------------------------------

    def get_asset(self, asset_id: str) -> dict[str, Any]:
        """GET /api/assets/{id} — full asset detail incl. originalPath, OCR, tags."""
        return self._get(f"assets/{asset_id}")

    def list_assets(self, **params: Any) -> Any:
        """GET /api/assets — asset list. Optional query params pass through."""
        return self._get("assets", params=params or None)

    def get_thumbnail(self, asset_id: str, size: str = "preview") -> bytes:
        """GET /api/assets/{id}/thumbnail?size=preview — the Immich-rendered JPEG.

        Returns raw bytes (NOT JSON, so it bypasses _request's JSON handling).
        Immich generates a JPEG preview for every asset regardless of the
        original format, so this is the reliable way to feed the vision endpoint
        formats it cannot decode itself — notably iPhone HEIC/HEIF originals,
        which fail with HTTP 400 'Failed to load image or audio file'.
        """
        url = f"{self._base}/assets/{asset_id}/thumbnail"
        resp = self._session.get(url, params={"size": size}, timeout=self._timeout)
        resp.raise_for_status()
        return resp.content

    def get_tags(self) -> Any:
        """GET /api/tags — all tags (flat list with id/name/value)."""
        return self._get("tags")

    def get_albums(self) -> Any:
        """GET /api/albums — all albums."""
        return self._get("albums")

    def get_albums_for_asset(self, asset_id: str) -> Any:
        """GET /api/albums?assetId={id} — albums that CONTAIN this asset.

        Used by the manual-fix guard to see where a human has filed/moved an
        asset (the asset payload itself does not list its albums)."""
        return self._get("albums", params={"assetId": asset_id})

    def get_album(self, album_id: str) -> dict[str, Any]:
        """GET /api/albums/{id} — album incl. its assets (for membership re-read)."""
        return self._get(f"albums/{album_id}")

    def search_metadata(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /api/search/metadata — paginated asset search (searchAssets v2).

        The ONLY safe way to enumerate a large album: a GET /api/albums/{id}
        silently caps at ~1000. Body supports albumIds, page, size, withExif,
        tagIds, takenAfter/Before, etc. Response: {assets: {total, items,
        nextPage, ...}}."""
        return self._request("POST", "search/metadata", json=body)

    # --- write path (Stage 3) -------------------------------------------
    # Reminder: the HTTP status is NOT the success signal for tagging — the
    # caller must re-read the asset. See app/writer.py.

    def upsert_tags(self, names: list[str]) -> Any:
        """PUT /api/tags — upsert tags by name; returns the tag objects (idempotent)."""
        return self._request("PUT", "tags", json={"tags": names})

    def bulk_tag_assets(self, tag_ids: list[str], asset_ids: list[str]) -> Any:
        """PUT /api/tags/assets — bulkTagAssets, many tags x many assets in ONE call."""
        return self._request(
            "PUT", "tags/assets", json={"tagIds": tag_ids, "assetIds": asset_ids}
        )

    def create_album(self, name: str) -> dict[str, Any]:
        """POST /api/albums — create an album by name; returns it (incl. new id)."""
        return self._request("POST", "albums", json={"albumName": name})

    def add_assets_to_album(self, album_id: str, asset_ids: list[str]) -> Any:
        """PUT /api/albums/{id}/assets — add assets; duplicates are reported, not errors."""
        return self._request(
            "PUT", f"albums/{album_id}/assets", json={"ids": asset_ids}
        )

    def update_asset(self, asset_id: str, **fields: Any) -> dict[str, Any]:
        """PUT /api/assets/{id} — updateAsset (SINGULAR). e.g. description=..."""
        return self._request("PUT", f"assets/{asset_id}", json=fields)

    # --- removals (Stage 5 move-not-add) --------------------------------
    # These remove ALBUM MEMBERSHIP / a TAG — they never delete an asset.
    # Paths are for Immich 2.7.5. NOTE: if Immich is upgraded to v3.x the
    # album-asset routes change (e.g. /api/albums/{id}/assets shape differs) —
    # revisit these two methods then.

    def remove_assets_from_album(self, album_id: str, asset_ids: list[str]) -> Any:
        """DELETE /api/albums/{id}/assets — remove assets from an album (NOT delete)."""
        return self._request(
            "DELETE", f"albums/{album_id}/assets", json={"ids": asset_ids}
        )

    def untag_assets(self, tag_id: str, asset_ids: list[str]) -> Any:
        """DELETE /api/tags/{id}/assets — remove ONE tag from many assets (inverse of tagAssets)."""
        return self._request(
            "DELETE", f"tags/{tag_id}/assets", json={"ids": asset_ids}
        )
