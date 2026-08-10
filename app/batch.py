"""Stage 4 — batch processing across many assets.

Pipeline per asset: enumerate -> cache-skip check -> manual-fix guard ->
gather_signals + classify (Stage 2) -> build_plan with review routing ->
print (dry-run) or execute_plans_batch (Stage 3 verify-and-retry, group-paced)
-> cache the result.

Human decisions always win (manual-fix guard). --commit defaults OFF.

Whisper is out of scope; video frames already work via signals.gather_signals.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

from . import writer
from .cache import Cache
from .classifier import classify_asset
from .config import Config
from .immich_client import ImmichClient
from .signals import SignalError
from .taxonomy import Taxonomy, load_taxonomy
from .transcribe import build_transcriber
from .transcript_store import build_transcript_store
from .vision import VisionClient, VisionError
from .writer import Plan, build_plan, execute_plans_batch, format_plan

_PAGE_SIZE = 500


def route(result: dict[str, Any], tax: Taxonomy) -> dict[str, Any]:
    """From a classification, decide where it files and which marker tags apply.

    Returns {album, extra_tags, literal_tags, needs_review, source}. Shared by
    batch, reprocess, and the single-asset path so routing can't diverge.
    """
    source = result.get("source", "unknown")
    needs_review = result["confidence"] < tax.review.threshold
    album = tax.review.album_name if needs_review else result["category"]
    extra_tags = [tax.review.tag_name] if needs_review else []
    literal_tags: list[str] = []
    sd = tax.source_detection
    if sd.enabled and sd.tag_sources and source != "unknown":
        literal_tags.append(f"source:{source}")
    return {
        "album": album, "extra_tags": extra_tags, "literal_tags": literal_tags,
        "needs_review": needs_review, "source": source,
    }


def source_allowed(result: dict[str, Any], tax: Taxonomy) -> bool:
    """False only when source_detection.process_only is set and the detected
    source does not match (such assets are skipped + reported, never silent)."""
    sd = tax.source_detection
    if not (sd.enabled and sd.process_only):
        return True
    return result.get("source", "unknown") == sd.process_only


@dataclass
class _Item:
    asset: dict[str, Any]
    classification: dict[str, Any]
    plan: Plan
    needs_review: bool


# --- enumeration ----------------------------------------------------------

# What a classification run considers "the library". Immich 2.x used this as the
# search default; Immich 3.0 dropped that default, so we send it ourselves.
# Overridable per-run via SEARCH_VISIBILITY (see app/config.py).
DEFAULT_SEARCH_VISIBILITY = "timeline"


def _resolve_album_id(client: ImmichClient, name: str) -> Optional[str]:
    for a in client.get_albums() or []:
        if a.get("albumName") == name:
            return a.get("id")
    return None


def search_paginated(
    client: ImmichClient, body_filter: dict[str, Any], limit: Optional[int],
    page_size: int = _PAGE_SIZE, visibility: Optional[str] = DEFAULT_SEARCH_VISIBILITY,
    count_total: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """Paginate POST /api/search/metadata with the given filter (albumIds /
    tagIds / ...). Returns (assets, total).

    'total' is counted by walking the pages, because Immich's ``assets.total``
    is per-page, not the match count. With ``count_total`` (the default) a
    --limit run still pages to the end to establish the true scope size; those
    extra calls are search only, no inference. Pass False to stop at the limit.

    ``visibility`` is ALWAYS sent explicitly. Immich 2.x defaulted to 'timeline'
    when it was omitted; Immich 3.0 changed that default to ANY visibility, so
    an omitted filter means a server upgrade silently widens every run to
    include archived and hidden assets. Pass None to genuinely omit it (i.e. to
    opt into the v3 any-visibility behaviour); a 'visibility' key already
    present in body_filter always wins.
    """
    collected: list[dict[str, Any]] = []
    seen = 0
    page = 1
    truncated_at: Optional[int] = None

    while True:
        body = {**body_filter, "page": page, "size": page_size, "withExif": True}
        if visibility is not None and "visibility" not in body_filter:
            body["visibility"] = visibility
        resp = client.search_metadata(body) or {}
        block = resp.get("assets") or {}
        items = block.get("items") or []

        # Immich's assets.total is the count IN THIS PAGE, not the match count.
        # Reading it once off page 1 reported "500" for any album bigger than a
        # page, which made a --limit run unable to tell you how big the job
        # actually is. Count what we actually walk instead.
        seen += len(items)
        if truncated_at is None:
            collected.extend(items)
            if limit and len(collected) >= limit:
                collected = collected[:limit]
                truncated_at = page

        next_page = block.get("nextPage")
        if not items or not next_page:
            break
        page = int(next_page)
        # Past the limit we keep paging purely to finish counting. No inference
        # happens here, so it costs a couple of cheap search calls and buys an
        # honest scope size to plan a long run against.
        if truncated_at is not None and not count_total:
            break

    return collected, (seen if (count_total or truncated_at is None) else len(collected))


def _enumerate(
    client: ImmichClient, album_id: str, limit: Optional[int],
    page_size: int = _PAGE_SIZE, visibility: Optional[str] = DEFAULT_SEARCH_VISIBILITY,
) -> tuple[list[dict[str, Any]], int]:
    """Stage 4 album enumeration (thin wrapper over search_paginated)."""
    return search_paginated(
        client, {"albumIds": [album_id]}, limit, page_size, visibility
    )


# --- per-asset decisions --------------------------------------------------

def has_source_tag(asset: dict[str, Any], client: ImmichClient) -> bool:
    """True if the asset already carries a `source:<platform>` tag.

    Used by --skip-sourced to leave saved social content alone. Source tags are
    only ever written by a prior classification run, so their presence marks an
    asset as saved-from-a-platform rather than personal camera-roll. This is a
    PRE-INFERENCE check — it reads existing tags rather than asking the model,
    so a skipped asset costs no GPU time (and does not depend on the local
    model's own weak source detection).

    NOTE: search/metadata items omit tags on 2.7.5, so fall back to a full
    get_asset when the passed-in dict carries none.
    """
    tags = asset.get("tags")
    if not tags:
        tags = (client.get_asset(asset["id"]) or {}).get("tags") or []
    for t in tags:
        if isinstance(t, dict):
            name = t.get("value") or t.get("name") or ""
            if str(name).strip().lower().startswith("source:"):
                return True
    return False


def _has_marker(asset: dict[str, Any]) -> bool:
    marker = load_taxonomy().marker_tag
    for t in asset.get("tags") or []:
        if isinstance(t, dict) and (t.get("value") == marker or t.get("name") == marker):
            return True
    return False


def _filed_albums(client: ImmichClient, asset_id: str) -> list[str]:
    albums = client.get_albums_for_asset(asset_id) or []
    return [a.get("albumName") for a in albums if a.get("albumName")]


def human_edit_reason(
    asset: dict[str, Any], cfg: Config, client: ImmichClient, cache: Cache,
    macro_names: set[str],
) -> tuple[Optional[str], list[str]]:
    """Decide whether a human edited this asset. Returns (reason, current_filed).

    reason is 'locked' | 'moved' | 'filed' | None. current_filed is the asset's
    membership among macro/_Review albums (re-read truth). Shared by the Stage 4
    guard and Stage 5 reprocess so they can never diverge.

    NOTE: search/metadata items omit tags on 2.7.5, so the marker is only read
    from a full get_asset; ownership is otherwise decided by cache presence.
    """
    asset_id = asset["id"]
    if asset.get("lockedProperties"):
        return "locked", []

    filed = _filed_albums(client, asset_id)
    review_album = load_taxonomy().review.album_name
    current_filed = [n for n in filed if n in (macro_names | {review_album})]
    cached = cache.get(asset_id)

    if cached:
        cached_album = cached.get("album")
        # Ours, but a human moved it out of the album we filed it to.
        if cached_album and cached_album not in current_filed:
            return "moved", current_filed
        return None, current_filed

    # Not in our cache. If it already sits in a macro-album, confirm whether it's
    # a human's manual filing by reading the REAL tags (search items lack them).
    if any(n in macro_names for n in filed):
        if not _has_marker(client.get_asset(asset_id)):
            return "filed", current_filed
        # Has our marker but missing from cache (e.g. cache reset): not human.

    return None, current_filed


def decide(
    asset: dict[str, Any], cfg: Config, client: ImmichClient, cache: Cache,
    macro_names: set[str],
) -> str:
    """Return 'skip_cache' | 'skip_human' | 'process'. Human filing/moves win."""
    asset_id = asset["id"]
    reason, current_filed = human_edit_reason(asset, cfg, client, cache, macro_names)

    if reason == "moved":
        # Sync the cache to the new location (never move it back).
        cache.update_album(asset_id, current_filed[0] if current_filed else "")
        return "skip_human"
    if reason:  # locked / filed
        return "skip_human"

    cached = cache.get(asset_id)
    checksum = asset.get("checksum")
    if cached and checksum and cached.get("content_hash") == checksum:
        return "skip_cache"
    return "process"  # new, or our own asset with a changed file


def _snapshot(asset: dict[str, Any]) -> dict[str, Any]:
    tags = sorted(
        (t.get("name") or t.get("value") or t.get("id"))
        for t in asset.get("tags", []) if isinstance(t, dict)
    )
    return {"tags": tags, "description": (asset.get("exifInfo") or {}).get("description")}


# --- main batch run -------------------------------------------------------

def run_batch(
    cfg: Config, client: ImmichClient, *, commit: bool, limit: Optional[int],
    skip_sourced: bool = False,
) -> int:
    if not cfg.vision.configured:
        print(
            "[config] Batch needs VISION_ENDPOINT and VISION_MODEL set. See .env.example.",
            file=sys.stderr,
        )
        return 2

    tax = load_taxonomy()
    src_id = _resolve_album_id(client, cfg.source_album)
    mode = "COMMIT — WRITES TO IMMICH" if commit else "DRY-RUN — writes NOTHING"
    print("=" * 72)
    print(f"Immich AI Classifier — BATCH  [{mode}]")
    print("=" * 72)
    print(f"Source album    : {cfg.source_album}")
    if not src_id:
        print(f"!! Source album {cfg.source_album!r} not found.", file=sys.stderr)
        return 2

    # Loaded once for the whole run; None when WHISPER_ENABLED is false, which
    # reproduces the exact pre-Whisper behaviour.
    transcriber = build_transcriber(cfg.whisper)
    transcript_store = build_transcript_store(cfg)
    if transcriber is not None:
        print(f"Transcripts     : {cfg.whisper.model} on {cfg.whisper.device} "
              f"(video assets only)")

    assets, total_found = _enumerate(
        client, src_id, limit, visibility=cfg.search_visibility
    )
    print(f"Total found     : {total_found}  (paginated via search/metadata, "
          f"visibility={cfg.search_visibility})")
    print(f"Processing      : {len(assets)}{'  (--limit)' if limit else ''}")
    print(f"Review threshold: {tax.review.threshold}   group size: {cfg.batch_group_size}")
    if tax.source_detection.process_only:
        print(f"Source filter   : only filing source={tax.source_detection.process_only}")
    print("-" * 72)

    cache = Cache(cfg.app_data_dir)
    vision = VisionClient(cfg.vision)
    macro_names = set(tax.names)
    known_tags = client.get_tags()      # prefetched for build_plan (avoids per-asset GETs)
    known_albums = client.get_albums()

    summary = {
        "found": total_found, "processed": 0, "skipped_cache": 0,
        "skipped_human": 0, "skipped_source": 0, "skipped_sourced": 0,
        "review": 0, "failed": [], "verify_retries": 0,
    }
    dry_probe: Optional[tuple[str, dict[str, Any]]] = None
    pending: list[_Item] = []

    def flush() -> None:
        if not pending:
            return
        try:
            outcomes = execute_plans_batch([it.plan for it in pending], cfg, client)
        except Exception as exc:  # noqa: BLE001 - one bad group must not kill the run
            print(f"  [FAIL] group write errored: {exc}")
            summary["failed"].extend(it.asset["id"] for it in pending)
            pending.clear()
            return
        for it, oc in zip(pending, outcomes):
            summary["verify_retries"] += oc.retags
            if oc.ok:
                summary["processed"] += 1
                if it.needs_review:
                    summary["review"] += 1
                cache.upsert(
                    asset_id=it.asset["id"], content_hash=it.asset.get("checksum"),
                    category=it.classification["category"], album=it.plan.album_name,
                    tags=[tp.normalized for tp in it.plan.tags],
                    confidence=it.classification["confidence"], needs_review=it.needs_review,
                )
                print(f"  [OK ] {it.asset['id']} -> {it.plan.album_name} "
                      f"({oc.present_count}/{oc.intended_count} tags, retags {oc.retags})")
            else:
                summary["failed"].append(it.asset["id"])
                print(f"  [FAIL] {it.asset['id']} -> {it.plan.album_name} "
                      f"missing={oc.missing_tag_names} unresolved={oc.unresolved_tag_names}")
        pending.clear()

    for idx, asset in enumerate(assets, 1):
        asset_id = asset.get("id")
        try:
            action = decide(asset, cfg, client, cache, macro_names)
            if action == "skip_cache":
                summary["skipped_cache"] += 1
                continue
            if action == "skip_human":
                summary["skipped_human"] += 1
                print(f"  [skip] {asset_id}  human-touched — left untouched")
                continue

            # Pre-inference: leave already-source-tagged (saved social) content
            # for a remote-model pass. Costs no GPU time.
            if skip_sourced and has_source_tag(asset, client):
                summary["skipped_sourced"] += 1
                print(f"  [skip] {asset_id}  already source-tagged — left for a remote-model pass")
                continue

            result = classify_asset(
                asset, cfg, vision, client, transcriber, transcript_store,
            )

            # Source filter: visibly skip assets whose detected source doesn't
            # match process_only (never silently dropped).
            if not source_allowed(result, tax):
                summary["skipped_source"] += 1
                print(f"  [skip] {asset_id}  source={result['source']} "
                      f"!= process_only={tax.source_detection.process_only}")
                continue

            r = route(result, tax)
            needs_review = r["needs_review"]
            plan = build_plan(
                asset, result, cfg, client,
                album_override=tax.review.album_name if needs_review else None,
                extra_tags=r["extra_tags"], literal_tags=r["literal_tags"],
                known_tags=known_tags, known_albums=known_albums,
            )

            if not commit:
                if dry_probe is None:
                    # Snapshot via get_asset (search items omit tags, which would
                    # make an unchanged asset look 'changed' vs the after re-read).
                    dry_probe = (asset_id, _snapshot(client.get_asset(asset_id)))
                print(f"\n[{idx}/{len(assets)}] {asset_id}  ({asset.get('type')})  "
                      f"-> {result['category']}  conf={result['confidence']:.2f}  "
                      f"source={result['source']}  review={'YES' if needs_review else 'no'}")
                print(format_plan(plan, commit=False))
                summary["processed"] += 1
                if needs_review:
                    summary["review"] += 1
            else:
                pending.append(_Item(asset, result, plan, needs_review))
                if len(pending) >= cfg.batch_group_size:
                    flush()

            if cfg.batch_pause:
                time.sleep(cfg.batch_pause)
        except (VisionError, SignalError) as exc:
            summary["failed"].append(asset_id)
            print(f"  [FAIL] {asset_id}  {exc}")
        except Exception as exc:  # noqa: BLE001 - resilience over a long batch
            summary["failed"].append(asset_id)
            print(f"  [FAIL] {asset_id}  unexpected: {exc}")

    if commit:
        flush()

    # Dry-run write proof: re-fetch the first planned asset; it must be unchanged.
    if not commit and dry_probe:
        aid, before = dry_probe
        after = _snapshot(client.get_asset(aid))
        print("-" * 72)
        print("IMMICH WRITE CHECK (dry-run must change nothing):")
        print(f"  probe asset {aid}: unchanged = {before == after}")

    cache.close()

    print("-" * 72)
    print("RUN SUMMARY:")
    print(f"  total found            : {summary['found']}")
    print(f"  processed (filed/plan) : {summary['processed']}")
    print(f"  sent to review         : {summary['review']}")
    print(f"  skipped (cache)        : {summary['skipped_cache']}")
    print(f"  skipped (human-touched): {summary['skipped_human']}")
    print(f"  skipped (source filter): {summary['skipped_source']}")
    if skip_sourced:
        print(f"  skipped (already sourced): {summary['skipped_sourced']}")
    print(f"  failed                 : {len(summary['failed'])}  {summary['failed']}")
    print(f"  total verify-retries   : {summary['verify_retries']}")
    print("-" * 72)
    return 1 if summary["failed"] else 0
