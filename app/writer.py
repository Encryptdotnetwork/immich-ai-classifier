"""Stage 3 — first WRITE-BACK: file one classification to Immich.

Takes a {category, tags, confidence} from the classifier and writes album +
tags + a short description for a SINGLE asset. Single asset only — no batching,
no caching, no review-queue, no deletion.

THE GOLDEN RULE (two confirmed, still-open Immich bugs):
- Tagging returns HTTP 200 BEFORE the tag persists (issue #23861).
- bulkTagAssets sometimes tags only SOME assets while reporting success
  (issue #16747).
So the SUCCESS SIGNAL IS A RE-READ OF THE ASSET, never the HTTP status. We
apply tags in one bulk call, then re-read and re-tag the misses until the
asset itself confirms them (or we report FAIL).

Write ordering also matters: a description write right before tagging can wipe
tags, so we batch tagging LAST and keep it separate from the description write.

A --commit flag (handled in main.py) gates all of this. Without it, build_plan
runs read-only and we print what WOULD happen; nothing is written.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import Config
from .immich_client import ImmichClient
from .taxonomy import load_taxonomy

# Marker tag, review tag, and the macro-album names all come from the loaded
# taxonomy now (config/categories.yaml). Nothing about the taxonomy is hardcoded.

_SLUG_RE = re.compile(r"[^a-z0-9]+")

def normalize_tag(raw: str) -> str:
    """Normalisation rule: SLUG form. Lowercase, then every run of
    non-alphanumeric characters (spaces, slashes, punctuation, underscores)
    becomes a single hyphen; leading/trailing hyphens are stripped.
    E.g. 'Bang Bang Chicken' -> 'bang-bang-chicken', 'buy/sell' -> 'buy-sell'
    (slugging '/' also avoids Immich treating it as tag hierarchy)."""
    return _SLUG_RE.sub("-", str(raw).lower()).strip("-")


def _display_tag(raw: str) -> str:
    """Readable form used only in the description text: lowercase + collapse
    whitespace (keeps spaces, unlike the slug used for the actual tag)."""
    return " ".join(str(raw).strip().lower().split())


def _normalize_literal(raw: str) -> str:
    """Lowercase + collapse whitespace but DO NOT slug — keeps structured tags
    like 'source:tiktok' intact (the ':' would otherwise become a hyphen)."""
    return " ".join(str(raw).strip().lower().split())


def compose_description(category: str, topic_tags: list[str], confidence: float) -> str:
    """Short description: category + a one-line topic summary. NOT the OCR/transcript."""
    text = f"AI-classified as {category}."
    if topic_tags:
        text += " Key topics: " + ", ".join(topic_tags[:8]) + "."
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0
    return text + f" (confidence {conf:.2f})"


def _tag_index(tags: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (key->id, id->name) maps. key is the lowercased value or name."""
    by_key: dict[str, str] = {}
    id_to_name: dict[str, str] = {}
    for t in tags:
        tid = t.get("id")
        if not tid:
            continue
        name = t.get("value") or t.get("name") or tid
        id_to_name[tid] = name
        for key in (t.get("value"), t.get("name")):
            if key:
                by_key.setdefault(key.strip().lower(), tid)
    return by_key, id_to_name


@dataclass
class TagPlan:
    normalized: str
    tag_id: Optional[str]
    exists: bool
    is_marker: bool = False


@dataclass
class Plan:
    asset_id: str
    album_name: str
    album_id: Optional[str]
    album_exists: bool
    tags: list[TagPlan]
    description: str


@dataclass
class Outcome:
    album_name: str
    album_id: str
    album_member: bool
    description: str
    description_confirmed: bool
    intended_count: int
    present_count: int
    missing_tag_names: list[str] = field(default_factory=list)
    unresolved_tag_names: list[str] = field(default_factory=list)
    verify_passes: int = 0
    retags: int = 0

    @property
    def ok(self) -> bool:
        return (
            self.album_member
            and self.description_confirmed
            and not self.missing_tag_names
            and not self.unresolved_tag_names
        )


def build_plan(
    asset: dict[str, Any], classification: dict[str, Any],
    cfg: Config, client: ImmichClient, *,
    album_override: Optional[str] = None,
    extra_tags: Optional[list[str]] = None,
    literal_tags: Optional[list[str]] = None,
    known_tags: Optional[list[dict[str, Any]]] = None,
    known_albums: Optional[list[dict[str, Any]]] = None,
) -> Plan:
    """Read-only: resolve album + tags + description against current Immich state.

    Creates nothing. Tags/album not present yet are marked exists=False so the
    dry-run report can show what WOULD be created.

    Review routing: pass album_override='_Review' and extra_tags=['needs-review']
    to file an ambiguous/low-confidence asset into the review bucket. The
    description still names the PREDICTED category (so a human reviewer sees the
    model's guess). extra_tags are applied as real tags but kept out of the
    description's topic summary.

    known_tags / known_albums let a batch prefetch these once and avoid a
    per-asset GET; if omitted they are fetched.
    """
    asset_id = asset["id"]
    category = classification["category"]
    album_name = album_override or category

    # Ordered, de-duplicated slug tags: model topics, then extra (e.g.
    # needs-review), then marker last. topic_display (readable) is model topics
    # ONLY — extras/marker do not pollute the description sentence.
    ordered: list[tuple[str, bool]] = []
    topic_display: list[str] = []
    seen: set[str] = set()
    for raw in list(classification.get("tags") or []):
        slug = normalize_tag(raw)
        if slug and slug not in seen:
            seen.add(slug)
            ordered.append((slug, False))
            topic_display.append(_display_tag(raw))
    for raw in extra_tags or []:
        slug = normalize_tag(raw)
        if slug and slug not in seen:
            seen.add(slug)
            ordered.append((slug, False))
    # Literal tags (e.g. 'source:tiktok') keep their punctuation — NOT slugged.
    for raw in literal_tags or []:
        lit = _normalize_literal(raw)
        if lit and lit not in seen:
            seen.add(lit)
            ordered.append((lit, False))
    marker_tag = load_taxonomy().marker_tag
    if marker_tag not in seen:
        ordered.append((marker_tag, True))

    tags_source = known_tags if known_tags is not None else client.get_tags()
    by_key, _ = _tag_index(tags_source)
    tag_plans = [
        TagPlan(normalized=n, tag_id=by_key.get(n), exists=n in by_key, is_marker=marker)
        for n, marker in ordered
    ]

    albums_source = known_albums if known_albums is not None else client.get_albums()
    album_id, album_exists = None, False
    for a in albums_source:
        if a.get("albumName") == album_name:
            album_id, album_exists = a.get("id"), True
            break

    description = compose_description(category, topic_display, classification.get("confidence", 0.0))

    return Plan(asset_id, album_name, album_id, album_exists, tag_plans, description)


def format_plan(plan: Plan, commit: bool) -> str:
    mode = "COMMIT" if commit else "DRY-RUN (writes nothing)"
    lines = [f"FILING PLAN [{mode}]:"]
    lines.append(f"  album       : {plan.album_name}  [{'reuse' if plan.album_exists else 'CREATE'}]")
    lines.append(f"  description : {plan.description!r}")
    lines.append(f"  tags ({len(plan.tags)}):")
    for tp in plan.tags:
        state = "reuse" if tp.exists else "create"
        marker = "  (marker)" if tp.is_marker else ""
        lines.append(f"    - {tp.normalized}  [{state}]{marker}")
    return "\n".join(lines)


def _current_tag_ids(client: ImmichClient, asset_id: str) -> set[str]:
    asset = client.get_asset(asset_id)
    return {t.get("id") for t in asset.get("tags", []) if isinstance(t, dict)}


def execute_plan(plan: Plan, cfg: Config, client: ImmichClient) -> Outcome:
    """Commit the plan, then verify by RE-READING. Tag writes happen LAST.

    Order: create missing tag defs -> add to album -> write description ->
    bulk-tag -> verify/re-tag loop -> final re-read.
    """
    asset_id = plan.asset_id

    # 1. Create any missing tag definitions (idempotent upsert), then resolve
    #    every intended tag id from a fresh read.
    missing_defs = [tp.normalized for tp in plan.tags if not tp.exists]
    if missing_defs:
        client.upsert_tags(missing_defs)
    by_key, id_to_name = _tag_index(client.get_tags())

    intended_ids: list[str] = []
    unresolved: list[str] = []
    for tp in plan.tags:
        tid = by_key.get(tp.normalized)
        if tid and tid not in intended_ids:
            intended_ids.append(tid)
        elif not tid:
            unresolved.append(tp.normalized)

    # 2. Album: create-or-reuse, then add the asset.
    album_id = plan.album_id
    if not album_id:
        album_id = client.create_album(plan.album_name)["id"]
    client.add_assets_to_album(album_id, [asset_id])

    # 3. Description — BEFORE tagging, never interleaved with tag writes.
    client.update_asset(asset_id, description=plan.description)

    # 4. Tag in ONE bulk call, then verify-and-retry against re-reads.
    if intended_ids:
        client.bulk_tag_assets(intended_ids, [asset_id])

    verify_passes = 0
    retags = 0
    missing_ids: list[str] = list(intended_ids)
    for _ in range(cfg.tag_verify_max_retries):
        time.sleep(cfg.tag_verify_delay)
        verify_passes += 1
        current = _current_tag_ids(client, asset_id)
        missing_ids = [tid for tid in intended_ids if tid not in current]
        if not missing_ids:
            break
        client.bulk_tag_assets(missing_ids, [asset_id])  # re-tag only the misses
        retags += 1
    else:
        # Loop exhausted without a clean read — do one final authoritative verify
        # so we never report based on an un-verified re-tag.
        if intended_ids:
            time.sleep(cfg.tag_verify_delay)
            verify_passes += 1
            current = _current_tag_ids(client, asset_id)
            missing_ids = [tid for tid in intended_ids if tid not in current]

    # 5. Final re-reads for the report (asset + album membership).
    final_asset = client.get_asset(asset_id)
    final_tag_ids = {t.get("id") for t in final_asset.get("tags", []) if isinstance(t, dict)}
    final_desc = (final_asset.get("exifInfo") or {}).get("description")
    album = client.get_album(album_id)
    member = asset_id in {a.get("id") for a in album.get("assets", [])}

    present = sum(1 for tid in intended_ids if tid in final_tag_ids)
    missing_now = [tid for tid in intended_ids if tid not in final_tag_ids]

    return Outcome(
        album_name=plan.album_name,
        album_id=album_id,
        album_member=member,
        description=plan.description,
        description_confirmed=(final_desc == plan.description),
        intended_count=len(intended_ids),
        present_count=present,
        missing_tag_names=[id_to_name.get(tid, tid) for tid in missing_now],
        unresolved_tag_names=unresolved,
        verify_passes=verify_passes,
        retags=retags,
    )


def format_outcome(outcome: Outcome) -> str:
    lines = ["WRITE RESULT (confirmed by re-reading the asset):"]
    lines.append(f"  album        : {outcome.album_name}  (member: {outcome.album_member})")
    lines.append(f"  description  : {'confirmed' if outcome.description_confirmed else 'NOT CONFIRMED'}")
    lines.append(f"  tags present : {outcome.present_count}/{outcome.intended_count}")
    lines.append(f"  verify passes: {outcome.verify_passes}   re-tag calls: {outcome.retags}")
    if outcome.unresolved_tag_names:
        lines.append(f"  UNRESOLVED   : {outcome.unresolved_tag_names}")
    if outcome.missing_tag_names:
        lines.append(f"  MISSING      : {outcome.missing_tag_names}")
    lines.append(f"  RESULT: {'PASS — Immich confirms album, tags, description.' if outcome.ok else 'FAIL — see MISSING/UNRESOLVED above.'}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Batch executor (Stage 4): commit many plans, amortising the verify delay.
#
# Same golden rule as execute_plan — re-read is truth — but the 1.5s verify
# sleep is shared across the GROUP rather than paid per asset. Writes go out for
# every asset first (album add, description, one bulk-tag), then a single shared
# verify/re-tag loop re-reads each asset and re-tags only its misses.
# --------------------------------------------------------------------------

def _resolve_plan_tags(
    plan: Plan, by_key: dict[str, str]
) -> tuple[list[str], list[str]]:
    intended: list[str] = []
    unresolved: list[str] = []
    for tp in plan.tags:
        tid = by_key.get(tp.normalized)
        if tid and tid not in intended:
            intended.append(tid)
        elif not tid:
            unresolved.append(tp.normalized)
    return intended, unresolved


def _finalize(
    plan: Plan, album_id: str, intended: list[str], unresolved: list[str],
    id_to_name: dict[str, str], verify_passes: int, retags: int,
    client: ImmichClient,
) -> Outcome:
    final = client.get_asset(plan.asset_id)
    final_ids = {t.get("id") for t in final.get("tags", []) if isinstance(t, dict)}
    desc = (final.get("exifInfo") or {}).get("description")
    album = client.get_album(album_id)
    member = plan.asset_id in {a.get("id") for a in album.get("assets", [])}
    present = sum(1 for tid in intended if tid in final_ids)
    missing = [tid for tid in intended if tid not in final_ids]
    return Outcome(
        album_name=plan.album_name, album_id=album_id, album_member=member,
        description=plan.description, description_confirmed=(desc == plan.description),
        intended_count=len(intended), present_count=present,
        missing_tag_names=[id_to_name.get(tid, tid) for tid in missing],
        unresolved_tag_names=unresolved, verify_passes=verify_passes, retags=retags,
    )


def execute_plans_batch(
    plans: list[Plan], cfg: Config, client: ImmichClient
) -> list[Outcome]:
    """Commit a group of plans, sharing one verify loop. Returns one Outcome each."""
    if not plans:
        return []

    # 1. Create every missing tag definition across the group in ONE upsert,
    #    then resolve all ids from a single fresh read.
    missing_defs = sorted({tp.normalized for p in plans for tp in p.tags if not tp.exists})
    if missing_defs:
        client.upsert_tags(missing_defs)
    by_key, id_to_name = _tag_index(client.get_tags())

    # 2. Resolve album ids once per distinct album name (create-or-reuse).
    album_ids = {a.get("albumName"): a.get("id") for a in client.get_albums()}

    # 3. Write everything EXCEPT the verify: album add, description, initial
    #    bulk-tag. No per-asset sleep here.
    states: list[dict] = []
    for plan in plans:
        album_id = album_ids.get(plan.album_name)
        if not album_id:
            album_id = client.create_album(plan.album_name)["id"]
            album_ids[plan.album_name] = album_id
        client.add_assets_to_album(album_id, [plan.asset_id])
        client.update_asset(plan.asset_id, description=plan.description)
        intended, unresolved = _resolve_plan_tags(plan, by_key)
        if intended:
            client.bulk_tag_assets(intended, [plan.asset_id])
        states.append({"plan": plan, "album_id": album_id, "intended": intended,
                       "unresolved": unresolved, "retags": 0})

    # 4. Shared verify/re-tag loop: ONE sleep per attempt for the whole group.
    verify_passes = 0
    for _ in range(cfg.tag_verify_max_retries):
        time.sleep(cfg.tag_verify_delay)
        verify_passes += 1
        all_clean = True
        for s in states:
            if not s["intended"]:
                continue
            current = _current_tag_ids(client, s["plan"].asset_id)
            missing = [tid for tid in s["intended"] if tid not in current]
            if missing:
                all_clean = False
                client.bulk_tag_assets(missing, [s["plan"].asset_id])
                s["retags"] += 1
        if all_clean:
            break
    else:
        # Final authoritative verify (no re-tag) so reports never rest on an
        # un-verified write.
        if any(s["intended"] for s in states):
            time.sleep(cfg.tag_verify_delay)
            verify_passes += 1

    # 5. Finalize each plan by re-reading.
    return [
        _finalize(s["plan"], s["album_id"], s["intended"], s["unresolved"],
                  id_to_name, verify_passes, s["retags"], client)
        for s in states
    ]


# --------------------------------------------------------------------------
# Removals (Stage 5 move-not-add). Same golden rule as tagging: re-read is
# truth, never the HTTP status. Each returns (ok, retries).
# --------------------------------------------------------------------------

def _album_member_ids(client: ImmichClient, album_id: str) -> set[str]:
    return {a.get("id") for a in client.get_album(album_id).get("assets", [])}


def remove_from_album_verified(
    album_id: str, asset_id: str, cfg: Config, client: ImmichClient
) -> tuple[bool, int]:
    """Remove an asset from an album; confirm by RE-READING the album. Retries."""
    retries = 0
    for attempt in range(cfg.tag_verify_max_retries + 1):
        client.remove_assets_from_album(album_id, [asset_id])
        time.sleep(cfg.tag_verify_delay)
        if asset_id not in _album_member_ids(client, album_id):
            return True, retries
        retries += 1
    return False, retries


def remove_tag_verified(
    tag_id: str, asset_id: str, cfg: Config, client: ImmichClient
) -> tuple[bool, int]:
    """Remove one tag from an asset; confirm by RE-READING the asset. Retries."""
    retries = 0
    for attempt in range(cfg.tag_verify_max_retries + 1):
        client.untag_assets(tag_id, [asset_id])
        time.sleep(cfg.tag_verify_delay)
        tag_ids = {t.get("id") for t in client.get_asset(asset_id).get("tags", []) if isinstance(t, dict)}
        if tag_id not in tag_ids:
            return True, retries
        retries += 1
    return False, retries
