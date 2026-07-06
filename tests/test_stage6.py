"""Stage 6 acceptance + regression tests (offline).

Runs the REAL taxonomy/parse/route/build_plan/batch/reprocess code paths. The
only things stubbed are the two things that need a live deployment: the vision
HTTP call (replaced with canned model JSON) and the Immich client (a small fake
that serves albums/tags/assets from memory). Everything else — validation,
prompt assembly, category routing, source tagging, the manual-fix guard, the
move-not-add dry-run — is exercised for real.

Run:  PYTHONPATH=<repo> python tests/test_stage6.py
"""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import redirect_stdout

from app import batch as batch_mod
from app import reprocess as reprocess_mod
from app.batch import route, source_allowed, run_batch
from app.classifier import parse_classification
from app.config import Config, InferenceRole
from app.reprocess import run_reprocess
from app.taxonomy import (
    DEFAULT_CATEGORIES_FILE,
    TaxonomyError,
    load_taxonomy,
    reset_taxonomy,
)
from app.writer import build_plan

_PASSED = 0
_FAILED = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _PASSED, _FAILED
    if cond:
        _PASSED += 1
        print(f"  PASS  {label}")
    else:
        _FAILED += 1
        print(f"  FAIL  {label}  {detail}")


def expect_error(label: str, path: str, needle: str) -> None:
    """Load a deliberately-broken YAML; require a TaxonomyError naming the item."""
    global _PASSED, _FAILED
    try:
        load_taxonomy(path=path)
    except TaxonomyError as exc:
        msg = str(exc)
        if needle.lower() in msg.lower():
            _PASSED += 1
            print(f"  PASS  {label}  -> {msg.split(':', 1)[-1].strip()[:80]}")
        else:
            _FAILED += 1
            print(f"  FAIL  {label}  message missing {needle!r}: {msg}")
    except Exception as exc:  # noqa: BLE001
        _FAILED += 1
        print(f"  FAIL  {label}  wrong exception type {type(exc).__name__}: {exc}")
    else:
        _FAILED += 1
        print(f"  FAIL  {label}  loaded WITHOUT error (should have refused)")


def write_yaml(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


_VALID = """\
marker_tag: ai-classified
review:
  threshold: 0.70
  album_name: _Review
  tag_name: needs-review
source_detection:
  enabled: true
  tag_sources: true
  process_only: null
categories:
  - name: Alpha
    description: first
  - name: Beta
    description: second
  - name: General
    description: fallback
    catch_all: true
"""


def prime_default() -> None:
    reset_taxonomy()
    load_taxonomy(path=DEFAULT_CATEGORIES_FILE)


# --------------------------------------------------------------------------
# A. Default taxonomy loads + validates
# --------------------------------------------------------------------------

def test_default_load() -> None:
    print("\n[A] default categories.yaml loads + validates")
    reset_taxonomy()
    tax = load_taxonomy(path=DEFAULT_CATEGORIES_FILE)
    check("loads without error", True)
    check("catch-all is 'General'", tax.catch_all_name == "General", tax.catch_all_name)
    check("Geopolitics present", "Geopolitics" in tax.names_set)
    check("History present (distinct from Geopolitics)", "History" in tax.names_set)
    check("review.threshold == 0.70", tax.review.threshold == 0.70, str(tax.review.threshold))
    check("review.album_name == '_Review'", tax.review.album_name == "_Review")
    check("marker_tag == 'ai-classified'", tax.marker_tag == "ai-classified")
    check("source detection enabled + tags sources", tax.source_detection.enabled and tax.source_detection.tag_sources)
    check("process_only unset by default", tax.source_detection.process_only is None)
    check(">= 11 categories shipped", len(tax.names) >= 11, str(len(tax.names)))


# --------------------------------------------------------------------------
# B. Validation matrix — each broken config must refuse to run with a clear msg
# --------------------------------------------------------------------------

def test_validation_matrix() -> None:
    print("\n[B] validation matrix (each must refuse to run, naming the item)")

    dup = _VALID.replace("  - name: Beta\n    description: second", "  - name: Alpha\n    description: dupe")
    expect_error("duplicate category name", write_yaml(dup), "defined twice")

    two_catch = _VALID.replace(
        "  - name: Alpha\n    description: first",
        "  - name: Alpha\n    description: first\n    catch_all: true",
    )
    expect_error("two catch-alls", write_yaml(two_catch), "found 2")

    no_catch = _VALID.replace("    catch_all: true\n", "")
    expect_error("no catch-all", write_yaml(no_catch), "found 0")

    collide = _VALID.replace("album_name: _Review", "album_name: Alpha")
    expect_error("review name collides with category", write_yaml(collide), "collides")

    bad_thresh = _VALID.replace("threshold: 0.70", "threshold: 1.5")
    expect_error("threshold out of range (1.5)", write_yaml(bad_thresh), "between 0.0 and 1.0")

    too_few = """\
marker_tag: ai-classified
categories:
  - name: General
    description: only one
    catch_all: true
"""
    expect_error("fewer than 2 categories", write_yaml(too_few), "at least 2")

    malformed = "categories: [unclosed\n  - name: Alpha\n"
    expect_error("malformed YAML", write_yaml(malformed), "invalid YAML")

    bad_source = _VALID.replace("process_only: null", "process_only: myspace")
    expect_error("unknown process_only source", write_yaml(bad_source), "not a known source")


# --------------------------------------------------------------------------
# C. parse_classification honours the config taxonomy
# --------------------------------------------------------------------------

def test_parse() -> None:
    print("\n[C] parse_classification uses the config categories")
    prime_default()
    geo = parse_classification(
        '{"category": "Geopolitics", "source": "tiktok", "tags": ["conflict-map"], "confidence": 0.88}'
    )
    check("Geopolitics accepted as valid", geo["category"] == "Geopolitics" and geo["category_valid"])
    check("source normalised to tiktok", geo["source"] == "tiktok", geo["source"])

    off = parse_classification('{"category": "Knitting", "tags": [], "confidence": 0.9}')
    check("off-list category -> catch-all General", off["category"] == "General" and not off["category_valid"])
    check("off-list keeps raw_category for the flag", off["raw_category"] == "Knitting")

    alias = parse_classification('{"category": "Tech", "source": "ig", "tags": [], "confidence": 0.5}')
    check("source alias ig -> instagram", alias["source"] == "instagram", alias["source"])


# --------------------------------------------------------------------------
# D. route() + source_allowed()
# --------------------------------------------------------------------------

def test_route() -> None:
    print("\n[D] route() + source_allowed()")
    prime_default()
    tax = load_taxonomy()

    hi = route({"category": "Geopolitics", "confidence": 0.9, "source": "tiktok"}, tax)
    check("high-conf files to its category", hi["album"] == "Geopolitics" and not hi["needs_review"])
    check("source tag added (source:tiktok)", "source:tiktok" in hi["literal_tags"])

    lo = route({"category": "Crypto", "confidence": 0.4, "source": "youtube"}, tax)
    check("low-conf routes to review album", lo["album"] == "_Review" and lo["needs_review"])
    check("review adds needs-review tag", "needs-review" in lo["extra_tags"])

    unk = route({"category": "Food", "confidence": 0.9, "source": "unknown"}, tax)
    check("unknown source -> no source tag", unk["literal_tags"] == [])

    check("source_allowed True when process_only unset", source_allowed({"source": "tiktok"}, tax))

    po = load_taxonomy(path=write_yaml(_VALID.replace("process_only: null", "process_only: tiktok")))
    check("process_only set: matching source allowed", source_allowed({"source": "tiktok"}, po))
    check("process_only set: other source skipped", not source_allowed({"source": "instagram"}, po))


# --------------------------------------------------------------------------
# E. build_plan reflects routing (marker, literal source tag, review)
# --------------------------------------------------------------------------

def test_build_plan() -> None:
    print("\n[E] build_plan produces the right album + tags")
    prime_default()
    tax = load_taxonomy()
    asset = {"id": "asset-1"}

    result = {"category": "Tech", "confidence": 0.92, "source": "tiktok", "tags": ["gpu", "benchmark"]}
    r = route(result, tax)
    plan = build_plan(
        asset, result, None, None,
        album_override=None, extra_tags=r["extra_tags"], literal_tags=r["literal_tags"],
        known_tags=[], known_albums=[],
    )
    names = [tp.normalized for tp in plan.tags]
    check("files to predicted category album", plan.album_name == "Tech", plan.album_name)
    check("marker tag present + flagged", any(tp.is_marker and tp.normalized == "ai-classified" for tp in plan.tags))
    check("topic tags slugged", "gpu" in names and "benchmark" in names)
    check("source tag kept LITERAL (colon preserved)", "source:tiktok" in names, str(names))

    low = {"category": "Crypto", "confidence": 0.3, "source": "unknown", "tags": ["bitcoin"]}
    rl = route(low, tax)
    plan2 = build_plan(
        asset, low, None, None,
        album_override=tax.review.album_name if rl["needs_review"] else None,
        extra_tags=rl["extra_tags"], literal_tags=rl["literal_tags"],
        known_tags=[], known_albums=[],
    )
    names2 = [tp.normalized for tp in plan2.tags]
    check("low-conf plan files to _Review", plan2.album_name == "_Review", plan2.album_name)
    check("low-conf plan carries needs-review", "needs-review" in names2, str(names2))
    check("description still names predicted category", "Crypto" in plan2.description, plan2.description)


# --------------------------------------------------------------------------
# F. prompt-from-config: editing the YAML changes the prompt
# --------------------------------------------------------------------------

def test_prompt_from_config() -> None:
    print("\n[F] editability: a new category in the YAML appears in the prompt")
    reset_taxonomy()
    default = load_taxonomy(path=DEFAULT_CATEGORIES_FILE)
    check("default prompt mentions Geopolitics", "Geopolitics" in default.system_prompt)
    check("a throwaway word is NOT in default prompt", "Underwaterbasketweaving" not in default.system_prompt)

    edited = _VALID.replace(
        "  - name: Beta\n    description: second\n",
        "  - name: Beta\n    description: second\n"
        "  - name: Underwaterbasketweaving\n    description: a clearly novel throwaway category\n",
    )
    tax = load_taxonomy(path=write_yaml(edited))
    check("new category is in names_set", "Underwaterbasketweaving" in tax.names_set)
    check("new category injected into the prompt", "Underwaterbasketweaving" in tax.system_prompt)
    check("its description injected too", "a clearly novel throwaway category" in tax.system_prompt)
    check("catch-all still rendered last in prompt",
          tax.system_prompt.rfind("General") > tax.system_prompt.rfind("Underwaterbasketweaving"))

    # custom marker tag flows through to build_plan
    custom = load_taxonomy(path=write_yaml(_VALID.replace("marker_tag: ai-classified", "marker_tag: sorted-by-bot")))
    check("custom marker respected in build_plan",
          any(tp.is_marker and tp.normalized == "sorted-by-bot"
              for tp in build_plan({"id": "x"}, {"category": "Alpha", "confidence": 0.9, "tags": []},
                                   None, None, known_tags=[], known_albums=[]).tags))


# --------------------------------------------------------------------------
# Fake Immich client for the dry-run batch/reprocess regression
# --------------------------------------------------------------------------

class FakeClient:
    def __init__(self, albums, tags, assets, membership):
        self._albums = albums            # [{albumName, id}]
        self._tags = tags                # [{id, value}]
        self._assets = assets            # {id: asset dict}
        self._membership = membership    # {asset_id: [albumName, ...]}

    def get_albums(self):
        return list(self._albums)

    def get_tags(self):
        return list(self._tags)

    def get_asset(self, asset_id):
        return self._assets[asset_id]

    def get_albums_for_asset(self, asset_id):
        names = self._membership.get(asset_id, [])
        return [{"albumName": n} for n in names]

    def search_metadata(self, body):
        album_ids = set(body.get("albumIds") or [])
        id_to_name = {a["id"]: a["albumName"] for a in self._albums}
        wanted = {id_to_name[i] for i in album_ids if i in id_to_name}
        items = [
            a for a in self._assets.values()
            if wanted & set(self._membership.get(a["id"], []))
        ]
        return {"assets": {"items": items, "total": len(items), "nextPage": None}}


def _cfg(tmp_data: str) -> Config:
    role = InferenceRole(endpoint="http://stub/v1", model="stub-vlm", key="")
    return Config(
        immich_url="http://stub", immich_api_key="k",
        immich_internal_prefix="/usr/src/app/upload", local_mount="/immich-library",
        vision=role, text=InferenceRole("", "", ""),
        tag_verify_max_retries=1, tag_verify_delay=0.0,
        source_album="TikTok.Saved", app_data_dir=tmp_data,
        batch_group_size=25, batch_pause=0.0,
    )


# --------------------------------------------------------------------------
# G. --batch dry-run regression (enumerate, guard, cache, review + source route)
# --------------------------------------------------------------------------

def test_batch_dryrun() -> None:
    print("\n[G] --batch DRY-RUN regression (no writes; guard + route + cache)")
    prime_default()
    tmp = tempfile.mkdtemp()
    cfg = _cfg(tmp)

    assets = {
        "a1": {"id": "a1", "type": "IMAGE", "checksum": "c1", "tags": [], "exifInfo": {"description": None}},
        "a2": {"id": "a2", "type": "IMAGE", "checksum": "c2", "tags": [], "exifInfo": {"description": None}},
    }
    fake = FakeClient(
        albums=[{"albumName": "TikTok.Saved", "id": "src"}],
        tags=[],
        assets=assets,
        membership={"a1": ["TikTok.Saved"], "a2": ["TikTok.Saved"]},
    )

    canned = {
        "a1": {"category": "Tech", "source": "tiktok", "tags": ["gpu"], "confidence": 0.9,
               "parse_ok": True, "category_valid": True, "raw_category": "Tech"},
        "a2": {"category": "Crypto", "source": "unknown", "tags": ["bitcoin"], "confidence": 0.4,
               "parse_ok": True, "category_valid": True, "raw_category": "Crypto"},
    }
    orig = batch_mod.classify_asset
    batch_mod.classify_asset = lambda asset, c, v, cl: dict(canned[asset["id"]])
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run_batch(cfg, fake, commit=False, limit=None)
        out = buf.getvalue()
    finally:
        batch_mod.classify_asset = orig

    check("batch dry-run returns 0", rc == 0, f"rc={rc}\n{out}")
    check("both assets processed", "processed (filed/plan) : 2" in out, out)
    check("one routed to review (low conf)", "sent to review         : 1" in out, out)
    check("source-filter line present in summary", "skipped (source filter): 0" in out, out)
    check("dry-run wrote nothing (probe unchanged)", "unchanged = True" in out, out)
    check("a1 plan shows Tech", "-> Tech" in out, out)
    check("a2 plan shows _Review", "-> _Review" in out or "review=YES" in out, out)


# --------------------------------------------------------------------------
# H. --reprocess dry-run regression (move-not-add + human guard via marker)
# --------------------------------------------------------------------------

def test_reprocess_dryrun() -> None:
    print("\n[H] --reprocess DRY-RUN regression (move-not-add; marker => ours)")
    prime_default()
    tmp = tempfile.mkdtemp()
    cfg = _cfg(tmp)

    # Asset currently filed in Tech, carrying OUR marker (so the guard treats it
    # as ours, not a human filing). Reclassifies to Gaming -> should plan a move.
    assets = {
        "b1": {
            "id": "b1", "type": "IMAGE", "checksum": "h1",
            "tags": [{"id": "m", "value": "ai-classified"}],
            "exifInfo": {"description": None},
        },
    }
    fake = FakeClient(
        albums=[{"albumName": "Tech", "id": "tech"}, {"albumName": "Gaming", "id": "gaming"}],
        tags=[{"id": "m", "value": "ai-classified"}, {"id": "nr", "value": "needs-review"}],
        assets=assets,
        membership={"b1": ["Tech"]},
    )

    orig = reprocess_mod.classify_asset
    reprocess_mod.classify_asset = lambda asset, c, v, cl: {
        "category": "Gaming", "source": "tiktok", "tags": ["fps"], "confidence": 0.9,
        "parse_ok": True, "category_valid": True, "raw_category": "Gaming",
    }
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run_reprocess(
                cfg, fake, commit=False, limit=None, album="Tech", tag=None,
                asset_ids=[], include_human_edited=False,
            )
        out = buf.getvalue()
    finally:
        reprocess_mod.classify_asset = orig

    check("reprocess dry-run returns 0", rc == 0, f"rc={rc}\n{out}")
    check("plans a move Tech -> Gaming", "Tech -> Gaming" in out, out)
    check("would remove from old album (move-not-add)", "would remove from: ['Tech']" in out, out)
    check("not skipped as human (marker => ours)", "human-touched" not in out.split("REPROCESS SUMMARY")[0], out)
    check("source-filter line present in summary", "skipped (source filter): 0" in out, out)
    check("dry-run wrote nothing (probe unchanged)", "unchanged = True" in out, out)


def main() -> int:
    test_default_load()
    test_validation_matrix()
    test_parse()
    test_route()
    test_build_plan()
    test_prompt_from_config()
    test_batch_dryrun()
    test_reprocess_dryrun()
    reset_taxonomy()
    print("\n" + "=" * 60)
    print(f"RESULT: {_PASSED} passed, {_FAILED} failed")
    print("=" * 60)
    return 1 if _FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
