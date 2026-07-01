"""Entry point: foundation verify (default) and classification dry-run.

Verify / acceptance (Stage 1):
    python -m app.main <asset_id> [--raw]
        (a) raw originalPath  (b) translated path  (c) opens off disk  (d) ocrText

Classification DRY-RUN (Stage 2) — reads the image, runs the vision model,
prints a classification, and writes NOTHING back to Immich:
    python -m app.main --classify <asset_id>
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from .batch import route, run_batch
from .reprocess import run_reprocess
from .classifier import classify_asset
from .config import Config, ConfigError, load_config
from .immich_client import ImmichClient
from .paths import translate_path
from .signals import SignalError, find_ocr_text
from .taxonomy import TaxonomyError, load_taxonomy
from .vision import VisionClient, VisionError
from .writer import build_plan, execute_plan, format_outcome, format_plan

PREVIEW_BYTES = 16


def _resolve_asset_id(positionals: list[str]) -> str:
    if positionals and positionals[0].strip():
        return positionals[0].strip()
    env_id = os.environ.get("ASSET_ID", "").strip()
    if env_id:
        return env_id
    raise SystemExit(
        "No asset id provided. Pass one as an argument "
        "(python -m app.main <asset_id>) or set ASSET_ID."
    )


# --------------------------------------------------------------------------
# Stage 1: foundation verify / acceptance test
# --------------------------------------------------------------------------

def _run_verify(cfg: Config, client: ImmichClient, asset_id: str, dump_raw: bool) -> int:
    print("=" * 72)
    print("Immich AI Classifier — foundation verify / acceptance test")
    print("=" * 72)
    print(f"Immich API base : {cfg.api_base}")
    print(f"Internal prefix : {cfg.immich_internal_prefix}")
    print(f"Local mount     : {cfg.local_mount}")
    print(f"Asset id        : {asset_id}")
    print("-" * 72)

    asset = client.get_asset(asset_id)

    # (a) raw originalPath — printed BEFORE translation, so the real prefix is
    #     always visible even if translation fails.
    raw = asset.get("originalPath")
    print(f"(a) raw originalPath   : {raw}")
    if not raw:
        print("    !! Asset has no originalPath — cannot translate.", file=sys.stderr)
        return 1

    # (b) translated local path
    try:
        local_path = translate_path(raw, cfg.immich_internal_prefix, cfg.local_mount)
    except ValueError as exc:
        print("(b) translated path    : <FAILED>")
        print(f"    !! {exc}", file=sys.stderr)
        return 1
    print(f"(b) translated path    : {local_path}")

    # (c) confirm the file exists and opens off disk
    exists = os.path.isfile(local_path)
    print(f"(c) exists off disk    : {exists}")
    if not exists:
        print(
            "    !! File not found at the translated path. Check the read-only "
            "bind-mount and that IMMICH_INTERNAL_PREFIX matches the raw path.",
            file=sys.stderr,
        )
        return 1
    size = os.path.getsize(local_path)
    with open(local_path, "rb") as fh:
        head = fh.read(PREVIEW_BYTES)
    print(f"    opened OK, size    : {size:,} bytes")
    print(f"    first {PREVIEW_BYTES} bytes    : {head.hex()}")

    # (d) the asset's ocrText field
    ocr, where = find_ocr_text(asset)
    print(f"(d) ocrText ({where}):")
    if ocr:
        preview = ocr if len(ocr) <= 500 else ocr[:500] + " …[truncated]"
        print(f"    {preview!r}")
    else:
        print("    <empty or not set>")

    if dump_raw:
        print("-" * 72)
        print("--raw: top-level keys:")
        print("    " + ", ".join(sorted(asset.keys())))
        print("--raw: full asset JSON:")
        print(json.dumps(asset, indent=2, ensure_ascii=False))

    print("-" * 72)
    print("RESULT: PASS — file opened off disk, path translation proven.")
    return 0


# --------------------------------------------------------------------------
# Stage 2: classification DRY-RUN (no writes)
# --------------------------------------------------------------------------

def _immich_snapshot(asset: dict[str, Any]) -> dict[str, Any]:
    """The fields a classifier would ever touch — used to prove dry-run is inert."""
    tags = sorted(
        (t.get("name") or t.get("value") or t.get("id"))
        for t in asset.get("tags", [])
        if isinstance(t, dict)
    )
    description = (asset.get("exifInfo") or {}).get("description")
    return {"tags": tags, "description": description}


def _run_classify(cfg: Config, client: ImmichClient, asset_id: str, commit: bool) -> int:
    if not cfg.vision.configured:
        print(
            "[config] Classification needs VISION_ENDPOINT and VISION_MODEL set "
            "(VISION_KEY optional for local servers). See .env.example.",
            file=sys.stderr,
        )
        return 2

    tax = load_taxonomy()
    asset = client.get_asset(asset_id)
    before = _immich_snapshot(asset)
    mode = "COMMIT — WRITES TO IMMICH" if commit else "DRY-RUN — writes NOTHING"

    print("=" * 72)
    print(f"Immich AI Classifier — classify + file  [{mode}]")
    print("=" * 72)
    print(f"Vision endpoint : {cfg.vision.endpoint}")
    print(f"Vision model    : {cfg.vision.model}")
    print(f"Asset id        : {asset_id}")
    print(f"Asset type      : {asset.get('type')}")
    print("-" * 72)

    try:
        result = classify_asset(asset, cfg, VisionClient(cfg.vision))
    except (VisionError, SignalError) as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 1

    # Same routing the batch/reprocess paths use, so the single-asset flow files
    # identically (review redirect + source tag) instead of diverging.
    r = route(result, tax)
    needs_review = r["needs_review"]

    flags: list[str] = []
    if not result["parse_ok"]:
        flags.append(f"PARSE FAILED -> defaulted to {tax.catch_all_name}/0.0")
    elif not result["category_valid"]:
        flags.append(f"off-list category {result['raw_category']!r} -> {tax.catch_all_name}")
    if result["parse_ok"] and needs_review:
        flags.append(f"low confidence (<{tax.review.threshold:.2f}) -> routed to {tax.review.album_name}")

    print("CLASSIFICATION (raw model output; see FILING PLAN for the slugged tags written):")
    print(f"  category    : {result['category']}")
    print(f"  source      : {result['source']}")
    print(f"  tags (raw)  : {result['tags']}")
    print(f"  confidence  : {result['confidence']:.2f}")
    print(f"  ocr hint    : {'yes' if result['ocr_available'] else 'no (none on asset)'}")
    if result["asset_type"] == "VIDEO":
        print(f"  frames used : {result['num_frames']} (classified the middle frame)")
    if flags:
        print(f"  flags       : {'; '.join(flags)}")
    print("-" * 72)
    print("RAW MODEL OUTPUT:")
    print(result["raw"])
    print("-" * 72)

    # Stage 3: resolve what filing this classification would entail (read-only).
    # Honour review routing + source tagging exactly like batch/reprocess.
    try:
        plan = build_plan(
            asset, result, cfg, client,
            album_override=tax.review.album_name if needs_review else None,
            extra_tags=r["extra_tags"], literal_tags=r["literal_tags"],
        )
    except SignalError as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 1
    print(format_plan(plan, commit))
    print("-" * 72)

    if not commit:
        # Prove Immich is untouched: re-fetch and compare the touchable fields.
        after = _immich_snapshot(client.get_asset(asset_id))
        unchanged = before == after
        print("IMMICH WRITE CHECK (dry-run must change nothing):")
        print("  write calls issued : 0")
        print(f"  tags  before/after : {before['tags']} / {after['tags']}")
        print(f"  desc  before/after : {before['description']!r} / {after['description']!r}")
        print(f"  unchanged          : {unchanged}")
        print("-" * 72)
        if not unchanged:
            print("RESULT: WARN — snapshot differs although no writes were issued; investigate.")
            return 1
        print("RESULT: PASS — classification + plan produced, Immich left UNCHANGED.")
        print("        (re-run with --commit to write.)")
        return 0

    outcome = execute_plan(plan, cfg, client)
    print(format_outcome(outcome))
    print("-" * 72)
    return 0 if outcome.ok else 1


# --------------------------------------------------------------------------

def run(argv: list[str]) -> int:
    try:
        cfg: Config = load_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    # Load + validate the category taxonomy ONCE at startup. A bad config (bad
    # YAML, missing/duplicate categories, no/2+ catch-all, name collisions, an
    # out-of-range threshold, an unknown process_only source, ...) names the
    # offending item and refuses to run rather than misfiling assets.
    try:
        load_taxonomy()
    except TaxonomyError as exc:
        print(f"[taxonomy] {exc}", file=sys.stderr)
        return 2

    # Parse flags and positionals; --limit/--album/--tag take a value.
    flags: set[str] = set()
    positionals: list[str] = []
    limit: int | None = None
    album: str | None = None
    tag: str | None = None
    args = argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--limit":
            i += 1
            if i < len(args):
                try:
                    limit = int(args[i])
                except ValueError:
                    print(f"[args] --limit expects an integer, got {args[i]!r}", file=sys.stderr)
                    return 2
        elif a == "--album":
            i += 1
            album = args[i] if i < len(args) else None
        elif a == "--tag":
            i += 1
            tag = args[i] if i < len(args) else None
        elif a.startswith("-"):
            flags.add(a)
        else:
            positionals.append(a)
        i += 1

    client = ImmichClient(cfg)
    commit = "--commit" in flags

    # Reprocess mode: re-classify a scope with move-not-add (cache-override).
    if "--reprocess" in flags:
        return run_reprocess(
            cfg, client, commit=commit, limit=limit, album=album, tag=tag,
            asset_ids=positionals, include_human_edited="--include-human-edited" in flags,
        )

    # Batch mode sources assets from the album; no asset id needed.
    if "--batch" in flags:
        return run_batch(cfg, client, commit=commit, limit=limit)

    asset_id = _resolve_asset_id(positionals)
    # --commit implies the classify+file flow (you can't file without classifying).
    if "--classify" in flags or commit:
        return _run_classify(cfg, client, asset_id, commit)

    dump_raw = bool(flags & {"--raw", "--dump-asset"})
    return _run_verify(cfg, client, asset_id, dump_raw)


def main() -> None:
    raise SystemExit(run(sys.argv))


if __name__ == "__main__":
    main()
