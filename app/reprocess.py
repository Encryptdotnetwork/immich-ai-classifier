"""Stage 5 — scoped reprocessing with MOVE-NOT-ADD semantics.

Re-classify a targeted subset of already-filed assets, overriding the cache (but
NOT the human — see the guard interaction). When an asset's new category differs
from where it currently sits, it is ADDED to the new album AND REMOVED from the
old one (and the needs-review tag is removed when it leaves _Review). Removals
are verified by RE-READING, exactly like Stage 3 tagging — never the HTTP status.

NO ASSET DELETION, EVER — only album-membership / tag removals.
Whisper and multi-frame video are out of scope.

The four move cases (all remove-from-old except the no-op):
  1. macro -> different macro : add new, remove old album.
  2. _Review -> macro         : add macro, remove _Review + needs-review tag.
  3. macro -> _Review         : add _Review (+needs-review), remove old macro.
  4. same destination         : no album churn (tags/desc refreshed only).
"""

from __future__ import annotations

import sys
from typing import Any, Optional

from .batch import (
    _filed_albums,
    _resolve_album_id,
    _snapshot,
    has_source_tag,
    human_edit_reason,
    route,
    search_paginated,
    source_allowed,
)
from .cache import Cache
from .classifier import classify_asset
from .config import Config
from .immich_client import ImmichClient
from .signals import SignalError
from .taxonomy import load_taxonomy
from .transcribe import build_transcriber
from .vision import VisionClient, VisionError
from .writer import (
    build_plan,
    execute_plan,
    format_plan,
    remove_from_album_verified,
    remove_tag_verified,
)


SOURCE_TAG_PREFIX = "source:"


def stale_tags_to_remove(
    cached_tags: list[str], planned_tags: list[str], tax: Any
) -> list[str]:
    """Tags THIS TOOL wrote on a previous run that the new plan no longer wants.

    The whole safety property lives here. We only ever propose removing tags the
    cache records us writing, because that is the only set we can prove we own.
    A tag added by hand in the Immich UI was never in the cache, so it can never
    appear in this list. IMGCLASS-15 was a silent tag-loss incident; the lesson
    taken from it is that this tool removes only what it can prove is its own.

    Never removed, even when the cache claims them:
      - the marker tag: it records that we filed this asset at all
      - the review tag: reprocess strips that separately, after the album move
      - source:<platform>: expensive to derive, needs a frontier model to get
        right, and a weak local pass routinely fails to re-emit it. Losing these
        was the exact damage in IMGCLASS-15.

    Known limitation: the cache holds only the MOST RECENT run's tag list, so
    tags from two runs ago (e.g. a June pass later overwritten by an August one)
    are invisible here and will not be proposed for removal.
    """
    protected = {tax.marker_tag, tax.review.tag_name}
    planned = set(planned_tags)
    out: list[str] = []
    for tag in cached_tags:
        if tag in planned or tag in protected:
            continue
        if tag.lower().startswith(SOURCE_TAG_PREFIX):
            continue
        if tag not in out:
            out.append(tag)
    return out


def _resolve_scope(
    cfg: Config, client: ImmichClient, *,
    album: Optional[str], tag: Optional[str], asset_ids: list[str], limit: Optional[int],
) -> tuple[Optional[str], Optional[list[dict[str, Any]]], int]:
    """Return (label, assets, total). assets is None on error (message printed)."""
    if asset_ids:
        ids = asset_ids[:limit] if limit else asset_ids
        return f"ids[{len(ids)}]", [client.get_asset(a) for a in ids], len(asset_ids)
    if album:
        album_id = _resolve_album_id(client, album)
        if not album_id:
            print(f"!! Scope album {album!r} not found.", file=sys.stderr)
            return None, None, 0
        items, total = search_paginated(
            client, {"albumIds": [album_id]}, limit, visibility=cfg.search_visibility
        )
        return f"album:{album}", items, total
    if tag:
        tag_id = next(
            (t["id"] for t in client.get_tags() if (t.get("value") or t.get("name")) == tag), None
        )
        if not tag_id:
            print(f"!! Scope tag {tag!r} not found.", file=sys.stderr)
            return None, None, 0
        items, total = search_paginated(
            client, {"tagIds": [tag_id]}, limit, visibility=cfg.search_visibility
        )
        return f"tag:{tag}", items, total
    print("!! No scope given. Use --album <name>, --tag <name>, or asset ids.", file=sys.stderr)
    return None, None, 0


def run_reprocess(
    cfg: Config, client: ImmichClient, *, commit: bool, limit: Optional[int],
    album: Optional[str], tag: Optional[str], asset_ids: list[str],
    include_human_edited: bool, skip_sourced: bool = False,
    prune_tags: bool = False,
) -> int:
    if not cfg.vision.configured:
        print("[config] Reprocess needs VISION_ENDPOINT and VISION_MODEL set.", file=sys.stderr)
        return 2

    # Loaded once for the whole run — a per-asset model load would dominate.
    # None when WHISPER_ENABLED is false, which reproduces the pre-Whisper path.
    transcriber = build_transcriber(cfg.whisper)
    if transcriber is not None:
        print(f"Transcripts     : {cfg.whisper.model} on {cfg.whisper.device} "
              f"(video assets only)")

    label, assets, total = _resolve_scope(
        cfg, client, album=album, tag=tag, asset_ids=asset_ids, limit=limit
    )
    if assets is None:
        return 2

    tax = load_taxonomy()
    mode = "COMMIT — WRITES/MOVES IN IMMICH" if commit else "DRY-RUN — writes NOTHING"
    print("=" * 72)
    print(f"Immich AI Classifier — REPROCESS (move-not-add)  [{mode}]")
    print("=" * 72)
    print(f"Scope           : {label}")
    print(f"Scope size      : {total}")
    print(f"Processing      : {len(assets)}{'  (--limit)' if limit else ''}")
    print(f"Review threshold: {tax.review.threshold}")
    print(f"Cache override  : ON (reprocess re-classifies even if checksum unchanged)")
    if tax.source_detection.process_only:
        print(f"Source filter   : only filing source={tax.source_detection.process_only}")
    if include_human_edited:
        print("-" * 72)
        print("!!  --include-human-edited: human-filed / human-moved assets in scope WILL")
        print("!!  be reclassified. Your manual corrections in this scope can be OVERWRITTEN.")
    print("-" * 72)

    cache = Cache(cfg.app_data_dir)
    vision = VisionClient(cfg.vision)
    macro_names = set(tax.names)
    known_tags = client.get_tags()
    known_albums = client.get_albums()
    album_id_by_name = {a.get("albumName"): a.get("id") for a in known_albums}
    needs_review_tag_id = next(
        (t["id"] for t in known_tags if (t.get("value") or t.get("name")) == tax.review.tag_name), None
    )

    tag_id_by_name = {
        (t.get("value") or t.get("name")): t.get("id") for t in known_tags
    }

    summary = {
        "scope": total, "reprocessed": 0, "unchanged": 0, "skipped_human": 0,
        "skipped_source": 0, "skipped_sourced": 0, "human_overridden": 0,
        "failed": [], "verify_retries": 0, "tags_pruned": 0, "tags_prune_stuck": 0,
    }
    moves: dict[str, int] = {}
    dry_probe: Optional[tuple[str, dict[str, Any]]] = None

    for idx, asset in enumerate(assets, 1):
        asset_id = asset.get("id")
        try:
            reason, current_filed = human_edit_reason(asset, cfg, client, cache, macro_names)
            if reason:
                if not include_human_edited:
                    summary["skipped_human"] += 1
                    print(f"  [skip] {asset_id}  human-touched ({reason}) — not reprocessed")
                    continue
                summary["human_overridden"] += 1
                print(f"  [WARN] overriding human-edited ({reason}): {asset_id}")

            # Pre-inference: leave already-source-tagged (saved social) content
            # alone so a local model does not degrade a remote model's earlier,
            # better classification. Costs no GPU time.
            if skip_sourced and has_source_tag(asset, client):
                summary["skipped_sourced"] += 1
                print(f"  [skip] {asset_id}  already source-tagged — left for a remote-model pass")
                continue

            # Cache override: always classify within scope.
            result = classify_asset(asset, cfg, vision, client, transcriber)

            # Source filter: visibly skip assets whose detected source doesn't
            # match process_only (never silently dropped).
            if not source_allowed(result, tax):
                summary["skipped_source"] += 1
                print(f"  [skip] {asset_id}  source={result['source']} "
                      f"!= process_only={tax.source_detection.process_only}")
                continue

            r = route(result, tax)
            new_needs_review = r["needs_review"]
            new_album = r["album"]

            cached = cache.get(asset_id)
            from_label = (cached.get("album") if cached else None) or (
                current_filed[0] if current_filed else "(unfiled)")
            albums_to_remove = [a for a in current_filed if a != new_album]
            is_no_op = (new_album in current_filed) and not albums_to_remove
            move_key = f"{new_album} -> {new_album} (no move)" if is_no_op else f"{from_label} -> {new_album}"

            plan = build_plan(
                asset, result, cfg, client,
                album_override=tax.review.album_name if new_needs_review else None,
                extra_tags=r["extra_tags"], literal_tags=r["literal_tags"],
                known_tags=known_tags, known_albums=known_albums,
            )

            # Stale tags: what we wrote last run that this run no longer wants.
            # Computed even in dry-run so the report shows the real proposal.
            planned_tag_values = [tp.normalized for tp in plan.tags]
            stale = stale_tags_to_remove(
                (cached.get("tags") if cached else []) or [], planned_tag_values, tax,
            ) if prune_tags else []

            # --- DRY-RUN: print the planned move, write nothing ---
            if not commit:
                if dry_probe is None:
                    dry_probe = (asset_id, _snapshot(client.get_asset(asset_id)))
                print(f"\n[{idx}/{len(assets)}] {asset_id} ({asset.get('type')})  {move_key}"
                      f"   conf={result['confidence']:.2f}  source={result['source']}")
                if albums_to_remove:
                    extra = " + needs-review tag" if tax.review.album_name in albums_to_remove else ""
                    print(f"  would remove from: {albums_to_remove}{extra}")
                print(format_plan(plan, commit=False))
                if result.get("transcript_available"):
                    print(f"  transcript      : {result['transcript'].word_count} words "
                          f"(fed to the classifier as a hint)")
                elif result.get("transcript_error"):
                    print(f"  transcript      : FAILED — {result['transcript_error']}")
                if prune_tags:
                    prior = (cached.get("tags") if cached else []) or []
                    added = [t for t in planned_tag_values if t not in prior]
                    if added:
                        print(f"  tags ADD        : {added}")
                    if stale:
                        print(f"  tags REMOVE     : {stale}   "
                              f"(ours from a prior run, not in the new plan)")
                    if not added and not stale:
                        print("  tags            : unchanged")
                summary["reprocessed"] += 1
                if is_no_op:
                    summary["unchanged"] += 1
                else:
                    moves[move_key] = moves.get(move_key, 0) + 1
                continue

            # --- COMMIT: add new (Stage 3 verify) FIRST. Only after the add is
            # verified do we remove from the old album / strip needs-review. A
            # failed add must leave the asset exactly where it was (still in
            # _Review, still tagged) and must NOT be cached — otherwise it is
            # stranded: pulled from the queue but filed nowhere, and invisible to
            # a needs-review-scoped rerun. ---
            # album_id_by_name is live: execute_plan records any album it creates
            # so the next asset of the same category reuses it (see writer step 2).
            add_outcome = execute_plan(plan, cfg, client, album_ids=album_id_by_name)
            summary["verify_retries"] += add_outcome.retags

            if not add_outcome.ok:
                # Add did not persist. Touch nothing else — no album removal, no
                # tag strip, no cache write — so the asset stays recoverable.
                summary["failed"].append(asset_id)
                print(f"  [FAIL] {asset_id}  {move_key}  add_ok=False "
                      f"(left in place; nothing removed, not cached)  "
                      f"missing={add_outcome.missing_tag_names}")
                continue

            ok = True
            removed: list[tuple[str, bool]] = []
            for old in albums_to_remove:
                old_id = album_id_by_name.get(old) or _resolve_album_id(client, old)
                if old_id:
                    rok, rretries = remove_from_album_verified(old_id, asset_id, cfg, client)
                    summary["verify_retries"] += rretries
                    removed.append((old, rok))
                    ok = ok and rok

            review_tag_removed: Optional[bool] = None
            if tax.review.album_name in albums_to_remove and needs_review_tag_id:
                rok, rretries = remove_tag_verified(needs_review_tag_id, asset_id, cfg, client)
                summary["verify_retries"] += rretries
                review_tag_removed = rok
                ok = ok and rok

            # Stale-tag prune. Deliberately LAST, after the add verified and
            # after the album/needs-review removals — the same ordering the
            # review-tag strip uses. execute_plan's clobber-repair loop unions
            # the asset's CURRENT tags to restore them, so pruning any earlier
            # would just see them handed straight back (IMGCLASS-11/15).
            pruned: list[str] = []
            prune_stuck: list[str] = []
            for stale_tag in stale:
                stale_id = tag_id_by_name.get(stale_tag)
                if not stale_id:
                    continue  # tag no longer exists server-side; nothing to do
                pok, pretries = remove_tag_verified(stale_id, asset_id, cfg, client)
                summary["verify_retries"] += pretries
                if pok:
                    pruned.append(stale_tag)
                else:
                    prune_stuck.append(stale_tag)
            summary["tags_pruned"] += len(pruned)
            summary["tags_prune_stuck"] += len(prune_stuck)

            cache.upsert(
                asset_id=asset_id, content_hash=asset.get("checksum"),
                category=result["category"], album=new_album,
                tags=[tp.normalized for tp in plan.tags],
                confidence=result["confidence"], needs_review=new_needs_review,
            )

            summary["reprocessed"] += 1
            if is_no_op:
                summary["unchanged"] += 1
            else:
                moves[move_key] = moves.get(move_key, 0) + 1

            if ok:
                extras = []
                if removed:
                    extras.append("removed " + ",".join(a for a, _ in removed))
                if review_tag_removed is not None:
                    extras.append("needs-review " + ("removed" if review_tag_removed else "STUCK"))
                if pruned:
                    extras.append("pruned " + ",".join(pruned))
                if prune_stuck:
                    extras.append("prune STUCK " + ",".join(prune_stuck))
                tail = ("; " + "; ".join(extras)) if extras else ""
                print(f"  [OK ] {asset_id}  {move_key}  "
                      f"({add_outcome.present_count}/{add_outcome.intended_count} tags{tail})")
            else:
                # Add succeeded (asset is safely in new_album, cached as such); a
                # removal or the tag strip did not fully verify, so it may linger
                # in the old album. Not stranded — recoverable on a plain rerun.
                summary["failed"].append(asset_id)
                stuck = [a for a, o in removed if not o]
                print(f"  [FAIL] {asset_id}  {move_key}  add_ok=True "
                      f"stuck_albums={stuck} review_tag_removed={review_tag_removed}")

        except (VisionError, SignalError) as exc:
            summary["failed"].append(asset_id)
            print(f"  [FAIL] {asset_id}  {exc}")
        except Exception as exc:  # noqa: BLE001 - resilience over a long run
            summary["failed"].append(asset_id)
            print(f"  [FAIL] {asset_id}  unexpected: {exc}")

    if not commit and dry_probe:
        aid, before = dry_probe
        after = _snapshot(client.get_asset(aid))
        print("-" * 72)
        print("IMMICH WRITE CHECK (dry-run must change nothing):")
        print(f"  probe asset {aid}: unchanged = {before == after}")

    cache.close()

    print("-" * 72)
    print("REPROCESS SUMMARY:")
    print(f"  scope size             : {summary['scope']}")
    print(f"  reprocessed            : {summary['reprocessed']}")
    print(f"  moved                  : {sum(moves.values())}")
    for k, v in sorted(moves.items(), key=lambda kv: -kv[1]):
        print(f"      {k} : {v}")
    print(f"  unchanged (no move)    : {summary['unchanged']}")
    print(f"  skipped (human-touched): {summary['skipped_human']}")
    print(f"  skipped (source filter): {summary['skipped_source']}")
    if skip_sourced:
        print(f"  skipped (already sourced): {summary['skipped_sourced']}")
    if include_human_edited:
        print(f"  human-edited OVERRIDDEN : {summary['human_overridden']}")
    if prune_tags:
        print(f"  stale tags pruned      : {summary['tags_pruned']}")
        if summary["tags_prune_stuck"]:
            print(f"  prune STUCK (kept)     : {summary['tags_prune_stuck']}")
    print(f"  failed                 : {len(summary['failed'])}  {summary['failed']}")
    print(f"  total verify-retries   : {summary['verify_retries']}")
    print("-" * 72)
    return 1 if summary["failed"] else 0
