"""Configuration, loaded ONCE from the environment at startup.

Rules of the project:
- Read every setting from environment variables a single time, at process
  start. Never re-read os.environ per call.
- Inference settings (vision/text endpoints) are placeholders for now. They
  are captured here so the rest of the app can be wired up endpoint-agnostic
  later, but nothing in this task uses them.

Usage:
    from app.config import load_config
    cfg = load_config()        # builds the singleton once
    cfg = load_config()        # returns the same instance, no re-read
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# Best-effort: load a local .env when running outside Docker (dev convenience).
# In the container, docker-compose's env_file already populates os.environ, so
# this is a no-op there. Failure to import python-dotenv is non-fatal.
try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001 - optional dependency, never block startup
    pass


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or invalid."""


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"Required environment variable {name!r} is not set. "
            f"See .env.example."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _optional_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from exc


def _optional_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}.") from exc


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _optional_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return default if raw == "" else False
    raise ConfigError(f"{name} must be a boolean (true/false), got {raw!r}.")


@dataclass(frozen=True)
class InferenceRole:
    """Placeholder config for one OpenAI-compatible inference role.

    Unused in this task. Kept so the backend stays endpoint-agnostic: whether
    a role points at Ollama on the LAN or a remote provider, the code only
    ever sees endpoint + model + key.
    """

    endpoint: str
    model: str
    key: str

    @property
    def configured(self) -> bool:
        return bool(self.endpoint and self.model)


@dataclass(frozen=True)
class WhisperConfig:
    """Local speech-to-text settings (app/transcribe.py).

    Runs faster-whisper INSIDE this container — no audio leaves the host. Off by
    default: enabling it changes what every video asset costs to process, so it
    must be an explicit opt-in rather than a silent behaviour change on upgrade.
    """

    enabled: bool = False
    model: str = "base"  # faster-whisper size or a local model dir
    device: str = "cpu"  # 'cpu' or 'cuda'
    compute_type: str = "int8"  # 'int8' (cpu), 'float16' (gpu), ...
    language: str = ""  # ISO code, or '' to auto-detect
    beam_size: int = 5
    max_duration: float = 900.0  # seconds; skip longer videos (0 = no limit)
    min_chars: int = 20  # below this, treat as "no useful speech"
    download_root: str = "/data/whisper-models"  # model weight cache (container path)


@dataclass(frozen=True)
class Config:
    """Immutable, fully-resolved application configuration."""

    # --- Immich connection ---
    immich_url: str
    immich_api_key: str

    # --- Path translation (see app/paths.py) ---
    # The prefix the Immich *server* container uses internally for its library.
    immich_internal_prefix: str
    # Where that same library is bind-mounted (read-only) in THIS container.
    local_mount: str

    # --- Inference roles ---
    vision: InferenceRole = field(repr=False)
    text: InferenceRole = field(repr=False)

    # --- Write-back tag verify-and-retry (see app/writer.py) ---
    # Immich returns 200 before a tag persists and can silently tag only some
    # assets, so we re-read and retry. These tune that loop.
    tag_verify_max_retries: int  # how many re-tag passes before reporting FAIL
    tag_verify_delay: float  # seconds to wait before each re-read

    # --- Batch processing (see app/batch.py) ---
    # Review machinery (threshold, _Review album name, needs-review tag) lives in
    # the taxonomy YAML now (config/categories.yaml), not here.
    source_album: str  # album to enumerate assets from
    app_data_dir: str  # writable dir for the SQLite cache (container path)
    batch_group_size: int  # assets per tag/verify group (amortises the delay)
    batch_pause: float  # seconds to pause between inference calls (rate-sense)

    # Visibility filter sent with every POST /api/search/metadata.
    # Immich 2.x defaulted to 'timeline' when this was omitted; Immich 3.0
    # changed the default to ANY visibility, which would silently pull archived
    # and hidden assets into a run. We now always send it explicitly.
    # Set to 'archive' to deliberately classify archived assets instead.
    search_visibility: str = "timeline"

    # --- Text role tuning (see app/summarise.py) ---
    # Reasoning models spend max_tokens thinking before they answer, so a budget
    # sized for the answer alone truncates them mid-thought.
    text_max_tokens: int = 2000
    # Append '/no_think' to disable the reasoning block on Qwen3-class models.
    text_no_think: bool = False
    # Constrain generation to the summary JSON schema. Ollama compiles this to a
    # grammar, making non-JSON output impossible. Auto-disables if the endpoint
    # returns HTTP 400 for it.
    text_structured: bool = True

    # --- Speech-to-text (see app/transcribe.py) ---
    # Defaults to a disabled WhisperConfig so existing callers and tests that
    # build a Config without it keep the exact pre-Whisper behaviour.
    whisper: WhisperConfig = field(default_factory=WhisperConfig, repr=False)

    @property
    def api_base(self) -> str:
        """Immich API base, normalised to '<host>/api' with no trailing slash."""
        base = self.immich_url.rstrip("/")
        if base.endswith("/api"):
            return base
        return f"{base}/api"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            immich_url=_require("IMMICH_URL"),
            immich_api_key=_require("IMMICH_API_KEY"),
            immich_internal_prefix=_require("IMMICH_INTERNAL_PREFIX"),
            local_mount=_require("LOCAL_MOUNT"),
            vision=InferenceRole(
                endpoint=_optional("VISION_ENDPOINT"),
                model=_optional("VISION_MODEL"),
                key=_optional("VISION_KEY"),
            ),
            text=InferenceRole(
                endpoint=_optional("TEXT_ENDPOINT"),
                model=_optional("TEXT_MODEL"),
                key=_optional("TEXT_KEY"),
            ),
            tag_verify_max_retries=_optional_int("TAG_VERIFY_MAX_RETRIES", 3),
            tag_verify_delay=_optional_float("TAG_VERIFY_DELAY", 1.5),
            source_album=_optional("SOURCE_ALBUM", "Unsorted"),
            app_data_dir=_optional("APP_DATA_DIR", "/data"),
            batch_group_size=_optional_int("BATCH_GROUP_SIZE", 25),
            batch_pause=_optional_float("BATCH_PAUSE", 0.0),
            search_visibility=_optional("SEARCH_VISIBILITY", "timeline"),
            text_max_tokens=_optional_int("TEXT_MAX_TOKENS", 2000),
            text_no_think=_optional_bool("TEXT_NO_THINK", False),
            text_structured=_optional_bool("TEXT_STRUCTURED", True),
            whisper=WhisperConfig(
                enabled=_optional_bool("WHISPER_ENABLED", False),
                model=_optional("WHISPER_MODEL", "base"),
                device=_optional("WHISPER_DEVICE", "cpu"),
                compute_type=_optional("WHISPER_COMPUTE_TYPE", "int8"),
                language=_optional("WHISPER_LANGUAGE"),
                beam_size=_optional_int("WHISPER_BEAM_SIZE", 5),
                max_duration=_optional_float("WHISPER_MAX_DURATION", 900.0),
                min_chars=_optional_int("WHISPER_MIN_CHARS", 20),
                download_root=_optional("WHISPER_DOWNLOAD_ROOT", "/data/whisper-models"),
            ),
        )


# Module-level singleton. Populated exactly once by the first load_config() call.
_CONFIG: Optional[Config] = None


def load_config() -> Config:
    """Return the process-wide Config, building it from the environment once."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config.from_env()
    return _CONFIG
