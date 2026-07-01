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
        )


# Module-level singleton. Populated exactly once by the first load_config() call.
_CONFIG: Optional[Config] = None


def load_config() -> Config:
    """Return the process-wide Config, building it from the environment once."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config.from_env()
    return _CONFIG
