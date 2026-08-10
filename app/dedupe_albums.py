"""Merge duplicate same-named albums, then delete the emptied duplicates.

Immich permits several albums to share a NAME (uniqueness is by id), so a
caller that fails to reuse an album it created earlier in the same run will
make a fresh album per asset. This consolidates every group of same-named
albums into one canonical album and removes the leftovers.

NO ASSET DELETION, EVER. This removes album CONTAINERS only, and only after a
re-read confirms each asset already sits in the canonical album. Deleting an
Immich album never deletes its photos — they remain in the library and in any
other album they belong to.

    python -m app.dedupe_albums                 # dry-run (default, writes nothing)
    python -m app.dedupe_albums --commit        # merge, then delete duplicates
    python -m app.dedupe_albums --name Pets     # scope to a single album name

Canonical choice: the album holding the MOST assets; ties broken by the
earliest createdAt (i.e. the original album rather than a later clone).
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from typing import Any, Optional

from .batch import search_paginated
from .config import ConfigError, load_config
from .immich_client import ImmichClient
from .writer import _asset_in_album


class EnumerationMismatch(RuntimeError):
    """Album membership could not be enumerated in full. Never delete on this."""


def _album_asset_ids(
    client: ImmichClient, album_id: str, expected: Optional[int] = None,
    visibility: Optional[str] = None,
) -> list[str]:
    """Every asset id in an album, via search/metadata.

    v3 BREAKING: this used to read ``album["assets"]`` from GET /api/albums/{id},
    but Immich 3.0 REMOVED the ``assets`` property from AlbumResponseDto. On v3
    that read silently returns [], which in --commit mode would have deleted a
    duplicate album WITHOUT moving its assets to the canonical first. The assets
    would survive (deleting an album never deletes photos) but they would be
    stranded outside the canonical album. Hence search/metadata plus the count
    cross-check below.

    ``visibility`` is None here on purpose: album membership is membership,
    regardless of whether an asset is archived. A dedupe must move EVERY asset
    or it must move none, otherwise it strands exactly the ones it can't see.
    """
    ids = [a.get("id") for a in search_paginated(
        client, {"albumIds": [album_id]}, None, visibility=visibility
    )[0] if a.get("id")]

    # Re-read is truth (writer.py's rule). If the album claims more assets than
    # we could enumerate, we are about to under-count, and under-counting is
    # what strands assets. Refuse rather than guess.
    if expected is not None and len(ids) < int(expected):
        raise EnumerationMismatch(
            f"album {album_id} reports {expected} assets but enumeration returned "
            f"{len(ids)}"
        )
    return ids


def _pick_canonical(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Most assets wins; ties broken by earliest createdAt."""
    return sorted(
        group,
        key=lambda a: (-int(a.get("assetCount") or 0), str(a.get("createdAt") or "")),
    )[0]


def run(argv: list[str]) -> int:
    commit = "--commit" in argv
    name_filter: Optional[str] = None
    if "--name" in argv:
        i = argv.index("--name")
        if i + 1 < len(argv):
            name_filter = argv[i + 1]

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2
    client = ImmichClient(cfg)

    albums = client.get_albums() or []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in albums:
        name = a.get("albumName")
        if name and (name_filter is None or name == name_filter):
            groups[name].append(a)
    dupes = {n: g for n, g in groups.items() if len(g) > 1}

    mode = "COMMIT — MERGES AND DELETES ALBUMS" if commit else "DRY-RUN — writes NOTHING"
    print("=" * 72)
    print(f"Immich AI Classifier — ALBUM DEDUPE  [{mode}]")
    print("=" * 72)
    print(f"Albums total    : {len(albums)}")
    print(f"Duplicate names : {len(dupes)}")
    if name_filter:
        print(f"Name filter     : {name_filter}")
    if not dupes:
        print("Nothing to do — no duplicate album names.")
        return 0
    print("-" * 72)

    merged = deleted = failed = 0
    for name, group in sorted(dupes.items(), key=lambda kv: -len(kv[1])):
        canonical = _pick_canonical(group)
        canonical_id = canonical.get("id")
        others = [a for a in group if a.get("id") != canonical_id]
        print(f"\n{name}: {len(group)} albums -> keep {canonical_id} "
              f"({canonical.get('assetCount')} assets), merge {len(others)}")

        for dup in others:
            dup_id = dup.get("id")
            try:
                # visibility=None always: the dry-run plan must enumerate the
                # exact same set the commit will act on.
                asset_ids = _album_asset_ids(
                    client, dup_id, expected=dup.get("assetCount"), visibility=None,
                )
            except EnumerationMismatch as exc:
                failed += 1
                print(f"  [FAIL] {dup_id}  {exc} — album KEPT, nothing moved or deleted")
                continue

            if not commit:
                print(f"  [plan] {dup_id}  move {len(asset_ids)} asset(s) then delete album")
                continue

            if asset_ids:
                client.add_assets_to_album(canonical_id, asset_ids)
                time.sleep(cfg.tag_verify_delay)

            # Re-read is truth (same rule as writer.py): never delete a source
            # album until every one of its assets is confirmed in the canonical.
            stragglers = [
                aid for aid in asset_ids
                if not _asset_in_album(client, aid, canonical_id)
            ]
            if stragglers:
                failed += 1
                print(f"  [FAIL] {dup_id}  {len(stragglers)} asset(s) unconfirmed "
                      f"in canonical — album KEPT, nothing deleted")
                continue

            client.delete_album(dup_id)
            merged += len(asset_ids)
            deleted += 1
            print(f"  [OK ] {dup_id}  {len(asset_ids)} asset(s) merged; album deleted")

    print("-" * 72)
    print("DEDUPE SUMMARY:")
    print(f"  duplicate names    : {len(dupes)}")
    print(f"  assets merged      : {merged}")
    print(f"  albums deleted     : {deleted}")
    print(f"  albums kept (unverified): {failed}")
    if not commit:
        print("  (dry-run — re-run with --commit to act)")
    print("-" * 72)
    return 1 if failed else 0


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
