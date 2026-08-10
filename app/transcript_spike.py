"""Transcript spike — transcribe + summarise a sample of videos, WRITE NOTHING.

Purpose: the open question is not "can we transcribe", it is "where should the
transcript live". That decision needs real numbers from a real library — how
long transcripts actually are, how many clips have no usable speech, whether the
summaries are good enough to put in a field a human reads. Guessing at those
produces a schema that has to be redone.

So this module produces evidence, not writes. It enumerates videos, transcribes
them locally, summarises them via the TEXT_* role, and dumps one JSON object per
asset to a JSONL file, then prints the distribution stats that bear on the
storage choice.

It shares batch.py's paginated enumeration, so it sees the same asset set a real
run would. It touches no Immich write endpoint at all — there is no --commit.

Usage:
    python -m app.transcript_spike --limit 20
    python -m app.transcript_spike --album "TikTok.Saved" --limit 50
    python -m app.transcript_spike --asset <asset_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

from .batch import _resolve_album_id, search_paginated
from .config import load_config
from .immich_client import ImmichClient
from .signals import SignalError, gather_signals
from .summarise import SummariseError, TextClient, summarise_transcript
from .transcribe import TranscribeError, build_transcriber

# Description budget candidates, used only to report how many transcripts WOULD
# fit if the "everything in the description field" option were taken.
_DESC_BUDGETS = (300, 1000, 2000, 5000)


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _enumerate_videos(
    client: ImmichClient, album: Optional[str], limit: Optional[int],
    visibility: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Video assets from an album, or from the whole library when album is None."""
    body: dict[str, Any] = {"type": "VIDEO"}
    if album:
        album_id = _resolve_album_id(client, album)
        if not album_id:
            raise SystemExit(f"Album {album!r} not found in Immich.")
        body["albumIds"] = [album_id]
    assets, total = search_paginated(client, body, limit, visibility=visibility)
    print(f"Videos enumerated : {len(assets)} (search reports {total})")
    return assets


def run(
    album: Optional[str],
    asset_ids: list[str],
    limit: Optional[int],
    out_path: str,
    pause: float,
) -> int:
    cfg = load_config()

    if not cfg.whisper.enabled:
        print(
            "WHISPER_ENABLED is false. Set it to true (and set WHISPER_MODEL) "
            "before running the spike.",
            file=sys.stderr,
        )
        return 2

    client = ImmichClient(cfg)
    transcriber = build_transcriber(cfg.whisper)
    assert transcriber is not None  # guarded above

    # The text role is optional: a run without it still yields transcripts and
    # their length distribution, which is most of what the storage decision needs.
    text_client: Optional[TextClient] = None
    if cfg.text.configured:
        text_client = TextClient(
            cfg.text, max_tokens=cfg.text_max_tokens, no_think=cfg.text_no_think,
            structured=cfg.text_structured,
        )
        print(f"Summariser        : {cfg.text.model} "
              f"(max_tokens={cfg.text_max_tokens}"
              f"{', schema-constrained' if cfg.text_structured else ''}"
              f"{', /no_think' if cfg.text_no_think else ''})")
    else:
        print("Summariser        : DISABLED (TEXT_ENDPOINT / TEXT_MODEL not set)")
    print(f"Whisper model     : {cfg.whisper.model} on {cfg.whisper.device}")

    if asset_ids:
        assets = [client.get_asset(a) for a in asset_ids]
    else:
        assets = _enumerate_videos(client, album, limit, cfg.search_visibility)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    counts = {"total": 0, "transcribed": 0, "no_speech": 0, "failed": 0, "summarised": 0}
    char_lengths: list[float] = []
    word_counts: list[float] = []
    started = time.time()

    with open(out_path, "w", encoding="utf-8") as fh:
        for i, asset in enumerate(assets, start=1):
            counts["total"] += 1
            asset_id = asset.get("id")
            record: dict[str, Any] = {
                "asset_id": asset_id,
                "filename": asset.get("originalFileName"),
                "existing_description": (asset.get("exifInfo") or {}).get("description"),
                "transcript": None,
                "summary": None,
                "error": None,
            }

            try:
                signals = gather_signals(asset, cfg, client, transcriber)
            except SignalError as exc:
                record["error"] = f"signals: {exc}"
                counts["failed"] += 1
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{i}/{len(assets)}] {asset_id} FAILED: {exc}")
                continue

            if signals.get("transcript_error"):
                record["error"] = f"transcribe: {signals['transcript_error']}"
                counts["failed"] += 1
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{i}/{len(assets)}] {asset_id} FAILED: {signals['transcript_error']}")
                continue

            transcript = signals.get("transcript")
            if transcript is None:
                counts["no_speech"] += 1
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{i}/{len(assets)}] {asset_id} no usable speech")
                continue

            counts["transcribed"] += 1
            char_lengths.append(float(len(transcript.text)))
            word_counts.append(float(transcript.word_count))
            record["transcript"] = transcript.to_dict()

            if text_client is not None:
                try:
                    summary = summarise_transcript(transcript.text, text_client)
                    record["summary"] = summary.to_dict()
                    if summary.ok:
                        counts["summarised"] += 1
                    else:
                        counts["summary_failed"] = counts.get("summary_failed", 0) + 1
                except SummariseError as exc:
                    record["error"] = f"summarise: {exc}"
                    counts["summary_error"] = counts.get("summary_error", 0) + 1

            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            preview = (record.get("summary") or {}).get("summary") or transcript.text[:80]
            print(
                f"[{i}/{len(assets)}] {asset_id} {transcript.word_count}w "
                f"{len(transcript.text)}c :: {preview[:100]}"
            )

            if pause:
                time.sleep(pause)

    _report(counts, char_lengths, word_counts, out_path, time.time() - started)
    return 0


def _report(
    counts: dict[str, int],
    char_lengths: list[float],
    word_counts: list[float],
    out_path: str,
    elapsed: float,
) -> None:
    print("\n" + "=" * 68)
    print("TRANSCRIPT SPIKE — nothing was written to Immich")
    print("=" * 68)
    print(f"  videos examined   : {counts['total']}")
    print(f"  transcribed       : {counts['transcribed']}")
    print(f"  no usable speech  : {counts['no_speech']}")
    print(f"  failed            : {counts['failed']}")
    print(f"  summarised ok     : {counts['summarised']}")
    if counts.get("summary_failed"):
        print(f"  summary unparsed  : {counts['summary_failed']}  "
              f"(model reply kept as .summary.raw in the JSONL)")
    if counts.get("summary_error"):
        print(f"  summary ERRORED   : {counts['summary_error']}  "
              f"(endpoint raised; reason in .error in the JSONL)")
    accounted = (counts.get("summarised", 0) + counts.get("summary_failed", 0)
                 + counts.get("summary_error", 0))
    if counts["transcribed"] and accounted != counts["transcribed"]:
        print(f"  !! {counts['transcribed'] - accounted} transcript(s) unaccounted for")
    print(f"  elapsed           : {elapsed:.1f}s "
          f"({elapsed / max(counts['total'], 1):.1f}s per video)")

    if char_lengths:
        print("\n  Transcript length (characters)")
        for pct in (50, 75, 90, 95, 100):
            print(f"    p{pct:<3} : {int(_percentile(char_lengths, pct) or 0):>7}")
        print(f"    mean : {int(sum(char_lengths) / len(char_lengths)):>7}")
        print(f"  Words: mean {int(sum(word_counts) / len(word_counts))}, "
              f"max {int(max(word_counts))}")

        # This is the number the storage decision turns on: if most transcripts
        # blow past a readable description length, "put it all in description"
        # stops being viable regardless of whether the field technically accepts it.
        print("\n  Share of transcripts fitting a given description budget")
        for budget in _DESC_BUDGETS:
            fits = sum(1 for c in char_lengths if c <= budget)
            print(f"    <= {budget:>5} chars : {fits}/{len(char_lengths)} "
                  f"({100 * fits / len(char_lengths):.0f}%)")

    print(f"\n  Full output: {out_path}")
    print("  Review it, then pick the storage option. Nothing is committed until you do.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.transcript_spike",
        description="Transcribe and summarise a sample of videos. Writes nothing to Immich.",
    )
    parser.add_argument("--album", help="Album name to sample from (default: whole library).")
    parser.add_argument("--asset", action="append", default=[],
                        help="Specific asset id; repeatable. Overrides --album.")
    parser.add_argument("--limit", type=int, default=20, help="Max videos to process (default 20).")
    parser.add_argument("--out", default="/data/transcript-spike.jsonl",
                        help="JSONL output path (default /data/transcript-spike.jsonl).")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="Seconds to pause between assets.")
    args = parser.parse_args(argv)

    try:
        return run(args.album, args.asset, args.limit, args.out, args.pause)
    except (TranscribeError, SummariseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
