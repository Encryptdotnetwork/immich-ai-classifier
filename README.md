# Immich AI Classifier

Sorts saved social-media videos and screenshots in Immich into a set of
**user-defined** macro-albums plus unlimited free-form micro-tags, and can tag
the source platform it came from. It is a **classifier, not a remover** — it
never deletes assets and never modifies files on Immich storage.

The categories are **not** baked into the code. They live in a single
user-editable YAML (`config/categories.yaml`); the tool ships a sensible default
taxonomy but is a general Immich classifier — edit the YAML to make it sort
whatever you want.

## Architecture (do not deviate)

- **Read-local, write-API.** Asset bytes are read directly off a **read-only**
  bind-mount of the Immich library. All results are written back via the Immich
  API.
- **Never** modify files in place on Immich storage (no ExifTool, no rewrites).
  Immich tracks checksums and sizes in Postgres; rewriting desyncs the DB and
  flags assets corrupt.
- Config and the category taxonomy are each loaded **once** at startup, never
  re-read per call (`app/config.py`, `app/taxonomy.py`).
- **No asset deletion, ever** — only album-membership and tag changes.

## Layout

```
app/
  config.py         # env -> Config, loaded once (Immich + VISION_/TEXT_ roles)
  taxonomy.py       # categories.yaml -> Taxonomy (loaded once); prompt assembly + validation
  immich_client.py  # read-only Immich client: get_asset / list_assets / get_tags / get_albums
  paths.py          # translate_path(): Immich-server path -> this container's mount
  vision.py         # OpenAI-compatible vision client (endpoint-agnostic)
  signals.py        # gather_signals(): image b64 / video frames + OCR hint (Whisper stubbed)
  classifier.py     # classify -> {category, source, tags, confidence}
  writer.py         # file album+tags+description; verify-and-retry on re-reads
  cache.py          # SQLite cache (skip unchanged, detect human moves)
  batch.py          # enumerate album, guard, route (review/source), group-write
  reprocess.py      # scoped re-classify with MOVE-NOT-ADD (remove-from-old)
  main.py           # entrypoint: verify, --classify, --commit, --batch, --reprocess
config/
  categories.yaml   # THE taxonomy: categories, review machinery, marker tag, source detection
Dockerfile
docker-compose.yml  # read-only bind-mount of the Immich library + editable config mount
.env.example
```

## The category taxonomy (`config/categories.yaml`)

This file is **the single source of truth** for what the classifier does. The
prompt *machinery* (the JSON output contract, the confidence rule, the tag
rules, the "return ONLY JSON" instruction, the source-detection instruction)
is fixed in `app/taxonomy.py` and **injects** your categories into a template —
you never edit prompt text in code. Edit the YAML, restart, done.

```yaml
marker_tag: ai-classified          # tag added to every asset this tool files

review:
  threshold: 0.70                  # confidence below this -> review album
  album_name: _Review              # NOT one of the categories
  tag_name: needs-review

source_detection:
  enabled: true
  tag_sources: true                # add a source:<platform> tag, e.g. source:tiktok
  process_only: null               # e.g. tiktok -> only FILE tiktok assets (others skipped+reported)

categories:
  - name: Trading
    description: Teaches a REPEATABLE METHOD — entry/exit rules, indicators, risk...
    notes: Trading vs Crypto is decided by INTENT; a chart on screen does NOT decide it...
    examples: [chart-setup walkthrough, risk-management rules]
  - name: General
    description: Anything that does not clearly fit another category.
    catch_all: true                # EXACTLY ONE category must be the catch-all
```

Each category takes `name` (required), `description` (required), `notes`
(optional — this is where intent/disambiguation guidance lives now, as data),
and `examples` (optional). The catch-all is always rendered last in the prompt
as the fallback.

The shipped default ships 12 categories: Trading, Crypto, Food, Gaming, Tech,
Health, History, **Geopolitics**, Jokes/Memes, Anime, Self-improvement, and
General (catch-all). Geopolitics is distinct from History by **recency /
lifecycle** — current-affairs content that goes stale (ongoing conflicts,
elections, foreign policy) versus settled keep-forever history.

### Startup validation (refuses to run on a bad config)

The taxonomy is validated once at startup; on any problem the tool prints a
message **naming the offending item** and exits without touching Immich:

- malformed YAML, or a top level that isn't a mapping;
- fewer than 2 categories;
- a duplicate category name;
- zero or more than one `catch_all: true`;
- a category missing `name` or `description`;
- `review.album_name`, `review.tag_name`, or `marker_tag` colliding with a
  category name;
- `review.threshold` outside `0.0–1.0`;
- `source_detection.process_only` set to an unknown source.

Override the file path with the `CATEGORIES_FILE` env var; otherwise the shipped
`config/categories.yaml` is used (the container bind-mounts it editable).

## Source detection

The model also returns the **source platform** for each asset, one of:
`tiktok, instagram, facebook, youtube, twitter, reddit, other, unknown`. With
`source_detection.tag_sources: true`, a literal `source:<platform>` tag is added
(the colon is preserved — it is not slugged). Set `process_only: tiktok` (or any
source) to file **only** matching assets; the rest are skipped and **reported**
in the run summary, never silently dropped.

**Accuracy caveats.** Source detection is inferred from on-screen UI,
watermarks, and caption styling, so it is a best-effort hint, not ground truth:
it is good on frontier vision models, patchier on small local models, and
Facebook in particular is hard to distinguish. Treat `source:*` tags as
advisory.

## The path-translation gotcha

`originalPath` from the API is from the **Immich server container's**
filesystem, e.g. `/usr/src/app/upload/upload/<userId>/ab/cd/<uuid>.jpg`. That
string is meaningless in this container. A read-only mount alone does **not**
fix it — the path is rewritten: strip `IMMICH_INTERNAL_PREFIX`, prepend
`LOCAL_MOUNT` (`app/paths.py`).

**Verify the prefix first.** The acceptance test prints the raw `originalPath`
before translating, so you can confirm the real prefix this instance uses and
set `IMMICH_INTERNAL_PREFIX` accordingly. Storage template is off by default,
so expect a UUID-based path, not `library/admin/2025/...`.

## Setup & run

```bash
# Deploy location on the Docker host: /data/compose/immich-ai-classifier/
cp .env.example .env
# edit .env: IMMICH_URL, IMMICH_API_KEY, IMMICH_INTERNAL_PREFIX,
#            LOCAL_MOUNT, IMMICH_LIBRARY_HOST_PATH, ASSET_ID
# edit config/categories.yaml if you want categories other than the defaults

docker compose build

# Acceptance test against a known asset id:
docker compose run --rm immich-ai-classifier python -m app.main <asset_id>
# or, with ASSET_ID set in .env:
docker compose run --rm immich-ai-classifier
```

> The compose project / image / service / container name is now
> `immich-ai-classifier`. If you previously deployed this as `tiktok-classifier`,
> the next `docker compose up` creates a NEW project/container (the old one is
> orphaned — remove it manually). Your cache/app-data carries over as long as
> `APP_DATA_HOST_PATH` still points at the same host dir.

`config/categories.yaml` is bind-mounted into the container, so editing it does
**not** require a rebuild — just re-run. (Editing `app/` code still does, since
the Dockerfile COPYs it.)

## Classification dry-run — `--classify`

Reads the image off disk, sends it to the vision model, and prints a
classification into one of your categories + free-form tags + the detected
source. **Writes nothing back to Immich.** Single asset only.

Set `VISION_ENDPOINT` / `VISION_MODEL` (and `VISION_KEY` if the server needs
auth) in `.env`, then run:

```bash
docker compose build
docker compose run --rm immich-ai-classifier python -m app.main --classify <asset_id>
```

It prints the chosen category, source, tags, confidence, the raw model output,
whether OCR text was available as a hint, the full filing plan, and a
write-check proving Immich is unchanged. Low-confidence assets show that they
would route to the review album.

Notes:
- **Vision-first.** The model does text-reading *and* intent classification in
  one pass. If the asset already has `exifInfo.ocrText`, it's passed as an
  optional hint — never required (Immich OCR is empty until its backfill runs).
- Intent over keywords: e.g. the default taxonomy's hardest call is **Trading**
  (teaches a repeatable method) vs **Crypto** (price hype, no method) — decided
  by intent via each category's `notes`, not by "a chart is on screen".
- Video assets: a few frames are extracted with MoviePy (`ffmpeg` is in the
  image) and the middle frame is classified; Whisper transcript and multi-frame
  reasoning are out of scope.

## Write-back — `--commit`

Files a classification to Immich for a SINGLE asset: album + tags (including the
`marker_tag`, default `ai-classified`) + a short description. **`--commit`
defaults OFF** — without it you get the classification *and* the full filing
plan, but nothing is written.

```bash
# Plan only (writes nothing; re-fetch confirms unchanged):
docker compose run --rm immich-ai-classifier python -m app.main --classify <asset_id>

# Actually write:
docker compose run --rm immich-ai-classifier python -m app.main --commit <asset_id>
```

Built against two confirmed, still-open Immich tagging bugs:
- It returns **HTTP 200 before the tag persists** (#23861).
- bulkTagAssets can **silently tag only some assets** while reporting success
  (#16747).

So the **success signal is a re-read of the asset, never the HTTP status**.
Writer behaviour:
- Tags applied in ONE `bulkTagAssets` call, then a verify-and-retry loop
  (`TAG_VERIFY_MAX_RETRIES`, `TAG_VERIFY_DELAY`): re-fetch the asset, re-tag only
  the misses, repeat; if still missing after the retries it reports **FAIL**
  with the missing tags — it never silently passes.
- Description is written via `PUT /api/assets/{id}` (singular) **before** tagging
  and is not interleaved with tag writes (a description write can wipe tags).
- Album + tags are create-or-reuse by name, so re-running `--commit` produces no
  duplicate tags or album membership and does not error.
- Tag normalisation: **slug-style** — lowercase, and every run of
  non-alphanumeric chars becomes a single hyphen (`Bang Bang Chicken` ->
  `bang-bang-chicken`). Source tags (`source:tiktok`) are kept **literal** (the
  colon is preserved). The description keeps a readable form.

## Batch — `--batch`

Processes many assets from the album named by `SOURCE_ALBUM` — **set this to
your own source album**. It is just a default you point at whatever album holds
the unsorted items (the baked-in default is `Unsorted`), not a fixed
requirement. `--commit` still defaults OFF.

```bash
# Dry-run the first 10 (prints plans, writes nothing):
docker compose run --rm immich-ai-classifier python -m app.main --batch --limit 10

# Commit the first 10:
docker compose run --rm immich-ai-classifier python -m app.main --batch --commit --limit 10

# Whole album:
docker compose run --rm immich-ai-classifier python -m app.main --batch --commit
```

Key behaviours:
- **Pagination:** enumerates via paginated `POST /api/search/metadata` (album
  filter), never the ~1000-capped `GET /api/albums/{id}`. Logs the real total.
- **Cache** (SQLite at `APP_DATA_DIR/cache.db` → host
  `/opt/appdata/immich-ai-classifier`): a re-run skips assets whose Immich
  `checksum` is unchanged — no inference, no writes.
- **Manual-fix guard (humans win):** an unmarked asset already in a macro-album
  (a human filed it) is left alone; a marked asset a human *moved* is left alone
  and the cache is synced to the new album (never moved back).
- **Review routing:** confidence below `review.threshold` (from
  `categories.yaml`) → filed into the review album with the needs-review tag
  instead of a category album. The description still names the predicted category
  for the reviewer.
- **Source filter:** if `source_detection.process_only` is set, assets whose
  detected source doesn't match are skipped and counted (`skipped (source filter)`
  in the summary), never silently dropped.
- **Pacing:** the verify delay is amortised across a group (`BATCH_GROUP_SIZE`)
  rather than paid per asset; re-read is still truth.
- **Run summary:** total found, processed, sent-to-review, skipped(cache),
  skipped(human-touched), skipped(source filter), failed (with IDs), verify-retries.

## Reprocess — `--reprocess`, MOVE-NOT-ADD

Re-classify a targeted SCOPE of already-filed assets (overriding the cache), and
when the new category differs from where the asset currently sits, **add to the
new album AND remove it from the old one** (so re-classification can't leave it
double-filed). Removals are verified by re-reading, same as tagging. `--commit`
defaults OFF. **No asset deletion — only album-membership / tag removals.**

```bash
# Dry-run: print "from -> to" for everything in the General album (writes nothing):
docker compose run --rm immich-ai-classifier python -m app.main --reprocess --album General --limit 10

# Commit the moves (adds new album, REMOVES old):
docker compose run --rm immich-ai-classifier python -m app.main --reprocess --album General --commit --limit 10

# Re-grade the review queue (assets the improved prompt now scores >= threshold leave _Review):
docker compose run --rm immich-ai-classifier python -m app.main --reprocess --tag needs-review --commit --limit 10

# Specific assets:
docker compose run --rm immich-ai-classifier python -m app.main --reprocess <id1> <id2> --commit
```

The four move cases (all remove-from-old except the no-op):
1. **macro → different macro** — add new, remove old album.
2. **review → macro** — add macro, remove the review album AND the needs-review tag.
3. **macro → review** — add review album + needs-review, remove old macro.
4. **same destination** — no album churn (tags/description refreshed only).

Guard interaction: `--reprocess` overrides the **cache**, never the **human**.
Human-filed/moved assets are SKIPPED even under reprocess. To override them, pass
`--include-human-edited` — it warns, reports how many it overrode, then
reclassifies. Scope via `--album <name>`, `--tag <name>`, or explicit asset ids;
all enumerate through the same paginated `search/metadata`. `--limit N` supported.
The `process_only` source filter applies here too.

> Built for Immich **2.7.5** (album object includes its assets; the remove
> endpoints work). On a future v3.x upgrade the album-asset routes change — see
> the note on `remove_assets_from_album` in `app/immich_client.py`.

### Local (no Docker) smoke run

```bash
pip install -r requirements.txt   # includes PyYAML for the taxonomy
# put the same vars in a local .env (python-dotenv loads it)
python -m app.main <asset_id>
```

## Foundation / connectivity check

A first-run sanity check that the container can reach Immich and read asset bytes
off the read-only mount. It tests **path translation and the read path, not
classification**. Running the plain entrypoint against a known asset id prints:

- **(a)** the raw `originalPath`
- **(b)** the translated local path
- **(c)** confirmation the file exists and opens off disk (size + first bytes)
- **(d)** the asset's `ocrText` field — **an empty value here is normal and
  expected.** Immich's OCR is blank until its backfill job runs, which is exactly
  why this tool is vision-first; an empty `ocrText` is not a failure.

If the file opens, path translation is proven.

## Acceptance test — done when

End-to-end proof the finished classifier works, **writing nothing**. Run
`--classify` (dry-run) on a known asset id:

```bash
docker compose run --rm immich-ai-classifier python -m app.main --classify <asset_id>
```

It is working when the output shows:

- a **category** that is one of your configured categories (from
  `config/categories.yaml`);
- a detected **source** (one of the source platforms, or `unknown`);
- free-form **tags**;
- a **FILING PLAN** — album + tags (including the `ai-classified` marker) +
  description;
- **`unchanged: True`** in the write-check — the dry-run touched nothing in Immich.

Re-run with `--commit` to actually file it.

## Notes

- The container runs as root so it can read library files regardless of the
  Immich UID/GID. The mount is `:ro`, so it can never write to Immich storage.
- `IMMICH_URL` must be the base URL **without** `/api`; the client appends it.
- Networking: the compose attaches to Immich's external network `immich_default`
  and reaches the API container-to-container at `http://immich_server:2283` (no
  reverse proxy, no TLS). If the attach fails, confirm the name with
  `docker network ls | grep immich`. Host-IP fallback: `http://<host>:2283`.
- `IMMICH_LIBRARY_HOST_PATH` is the Immich `UPLOAD_LOCATION` on disk
  (e.g. `/path/to/immich/library`); it maps to `/usr/src/app/upload` inside
  Immich, which is why `IMMICH_INTERNAL_PREFIX=/usr/src/app/upload`.
- The `TEXT_*` inference role is a placeholder, captured but unused (the
  classifier is vision-only; Whisper/transcript is out of scope).
```