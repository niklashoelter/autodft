"""TOML-based configuration with sensible defaults.

The storage layout is anchored on a single ``data_path``::

    <data_path>/
        autodft.db          # SQLite database (unless ``database.url`` is set)
        comp_data/          # per-molecule SLURM working directories
        export_data/        # CSV / JSON / file exports

``database.url`` and the individual sub-paths (``comp_data_path``,
``export_data_path``) can still be overridden explicitly in the TOML or
via environment variables.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.toml"

# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DatabaseConfig:
    # If left as None, the URL is derived from ``storage.data_path``.
    url: Optional[str] = None


@dataclass
class StorageConfig:
    # Single root directory holding DB, comp_data/, export_data/.
    data_path: str = "/data/autodft"
    # Explicit overrides; computed from ``data_path`` when None.
    comp_data_path: Optional[str] = None
    export_data_path: Optional[str] = None


@dataclass
class StageConfig:
    time_limit: str = "1-00:00:00"
    default_nprocs: int = 16
    default_mem_per_core: int = 4000
    max_iter: int = 1000
    displacement: float = 0.1


@dataclass
class RetryConfig:
    increased_time_limit: str = "4-00:00:00"
    increased_nprocs: int = 32
    increased_mem_per_core: int = 4000
    # Ceiling on the SLURM --mem of an escalated job, in MB. 0 disables it.
    #
    # SET THIS to the memory of the largest node in your partition. The
    # defaults above multiply out to 32 * (4000 + 50) = 126 GB per job; if no
    # node has that much the job sits PENDING forever with ReqNodeNotAvail,
    # and because the queue-length throttle counts pending jobs, a pile of
    # them stalls entrypoint expansion for the whole campaign.
    #
    # When the ceiling is hit the *rank count* is reduced to fit, never the
    # per-rank %maxcore -- lowering that would starve a job that died
    # needing more memory per rank. Defaults to 0 so that enabling it is a
    # deliberate choice made against real hardware.
    max_mem_per_job_mb: int = 0


@dataclass
class PipelineConfig:
    max_simultaneous_entrypoints: int = 40
    # SLURM queue slots granted per unit of entrypoint priority: a molecule
    # of priority p may hold up to p * queue_slots_per_priority of its jobs
    # in the queue at once. This is the only throttle on submission.
    queue_slots_per_priority: int = 10
    # Ceiling on jobs that exist in the database but have not been submitted
    # yet. Entrypoint expansion pauses above this, so the entrypoint table --
    # not the job table -- absorbs an arbitrarily large campaign. Expansion
    # deliberately does NOT consult squeue: the submission throttle keeps the
    # queue short by design, so gating expansion on queue length either never
    # fired or (when squeue was slow) stalled the whole campaign.
    max_unsubmitted_jobs: int = 500
    # Retained for older configs; no longer consulted.
    max_queue_length: int = 20
    loop_interval_seconds: int = 30
    max_attempts: int = 3
    # Global failure circuit breaker. max_attempts bounds retries per task,
    # but nothing bounded the campaign: a systematic error would fail every
    # molecule in turn, each burning its full escalated retry budget. When
    # more than `failure_breaker_ratio` of the last `failure_breaker_window`
    # judged tasks failed, new job creation and submission stop until an
    # operator resets it from the dashboard.
    failure_breaker_enabled: bool = True
    failure_breaker_ratio: float = 0.9
    failure_breaker_window: int = 100
    failure_breaker_min_samples: int = 20
    # Backstop on sbatch calls in one tick. 0 (the default) disables it:
    # the real throttle is the priority cap on *waiting* jobs, and a count
    # limit here silently became the binding constraint instead -- a tick
    # would stop at 100 with the partition idle, and the next one was 60 s
    # away, so the pipeline could only ever fill the cluster at 100 jobs a
    # minute no matter how much of it was free.
    max_submissions_per_tick: int = 0
    # What actually bounds one tick: wall clock, not a job count. Submission
    # must not run so long that status polling and job creation are starved,
    # so it yields after this many seconds and resumes on the next tick.
    max_submission_seconds_per_tick: float = 30.0
    confsearch: StageConfig = field(default_factory=StageConfig)
    optimization: StageConfig = field(default_factory=StageConfig)
    singlepoint: StageConfig = field(default_factory=StageConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class SlurmConfig:
    partition: str = "CPU"
    nice: int = 1000


@dataclass
class ApiConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class SecurityConfig:
    # The key that SIGNS the session cookie. It is not a credential and
    # nobody types it: access is a username plus a personal API key, and
    # there is no shared password any more.
    #
    # Leave it unset. A random secret is then generated once and kept in
    # `<data_path>/.session_secret`, so sessions survive a restart and no
    # secret has to live in a config file that gets copied around. Set it
    # explicitly only to share sessions across several controllers.
    session_secret: Optional[str] = None
    # How long the session cookie stays valid, in seconds. Default: 7 days.
    session_lifetime_seconds: int = 7 * 24 * 3600


@dataclass
class OrcaConfig:
    # Absolute path to the ORCA executable. Use "orca" if a module system
    # puts it on PATH; on bare-metal clusters set the full path.
    path: str = "orca"
    # Extra string passed as ORCA's second argument (e.g. MPI binding).
    # The old pipeline used "--bind-to none".
    extra_args: str = ""
    # Optional NBO executable; exported as NBOEXE when set.
    nbo_exe: Optional[str] = None
    # Per-job temp directory parent. "" disables the TMP_DIR copy-out
    # pattern and runs ORCA directly inside the job_path.
    tmp_dir: str = "/tmp"
    # Whether to copy the SCF wavefunction (``*.gbw``) back from the scratch
    # dir to the shared job directory. Off by default: nothing in the
    # pipeline reads it and there is no restart path, so on a large campaign
    # it is pure dead weight on the shared filesystem (~26% of it here).
    # Turn on only if you want the .gbw for manual restarts / re-analysis.
    keep_wavefunction: bool = False
    # Whether to copy the SCF electron density scratch files
    # (``*.densities``/``*.densitiesinfo``/``*.cube``) back. Off by default:
    # these are the single largest consumer of shared disk (~68% here) and
    # are never read by the pipeline. Turn on only if you need to re-plot
    # densities (which also needs keep_wavefunction).
    keep_densities: bool = False


@dataclass
class Settings:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    orca: OrcaConfig = field(default_factory=OrcaConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    # ------------------------------------------------------------------
    # Derived path helpers (used everywhere instead of touching the
    # raw config fields, so the data_path default applies uniformly).
    # ------------------------------------------------------------------

    @property
    def data_path(self) -> Path:
        return Path(self.storage.data_path).expanduser().resolve()

    @property
    def comp_data_path(self) -> Path:
        if self.storage.comp_data_path:
            return Path(self.storage.comp_data_path).expanduser().resolve()
        return self.data_path / "comp_data"

    @property
    def export_data_path(self) -> Path:
        if self.storage.export_data_path:
            return Path(self.storage.export_data_path).expanduser().resolve()
        return self.data_path / "export_data"

    @property
    def database_url(self) -> str:
        if self.database.url:
            return self.database.url
        return f"sqlite:///{self.data_path / 'autodft.db'}"

    def ensure_directories(self) -> None:
        """Create ``data_path``, ``comp_data``, and ``export_data`` if missing."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.comp_data_path.mkdir(parents=True, exist_ok=True)
        self.export_data_path.mkdir(parents=True, exist_ok=True)

    def session_secret(self) -> str:
        """The key that signs session cookies.

        Configured value if there is one, otherwise a random secret
        persisted next to the database. Persisted rather than generated
        per process because a fresh key on every restart would log
        everyone out on every restart.
        """
        if self.security.session_secret:
            return self.security.session_secret
        return _persisted_secret(self.data_path / ".session_secret")


# ---------------------------------------------------------------------------
# Session secret
# ---------------------------------------------------------------------------

# Keyed by path: reading the file on every request would put a filesystem
# round-trip (on NFS, here) in front of every authenticated call.
_SECRET_CACHE: dict[str, str] = {}


def _persisted_secret(path: Path) -> str:
    """Read the secret at *path*, creating it on first use.

    Created with O_EXCL and mode 0600, so two controllers starting at once
    cannot end up with different secrets: the loser of the race reads what
    the winner wrote instead of overwriting it.
    """
    key = str(path)
    cached = _SECRET_CACHE.get(key)
    if cached is not None:
        return cached

    import os
    import secrets

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        secret = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        # An unwritable data path must not take the dashboard down. The
        # cost is that sessions do not survive this process.
        logger.warning(
            "Cannot persist the session secret at %s (%s); using a "
            "process-local one, so logins will not survive a restart.",
            path, exc,
        )
        secret = secrets.token_urlsafe(32)
    else:
        secret = secrets.token_urlsafe(32)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(secret)

    if not secret:
        # An empty file would key every HMAC with b"" — regenerate.
        secret = secrets.token_urlsafe(32)
        path.write_text(secret, encoding="utf-8")

    _SECRET_CACHE[key] = secret
    return secret


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _merge(target: dict, source: dict) -> dict:
    """Deep-merge *source* into *target* (mutates target)."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value
    return target


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively convert a dict to a dataclass instance."""
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_types:
            continue
        ft = field_types[key]
        # Resolve string annotations
        if isinstance(ft, str):
            ft = eval(ft)  # noqa: S307 - only our own type names
        if isinstance(value, dict) and hasattr(ft, "__dataclass_fields__"):
            kwargs[key] = _dict_to_dataclass(ft, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


# Settings whose environment override is genuinely an integer. Everything
# else is left as the string it arrived as.
_INT_SETTINGS = {"port", "loop_interval_seconds"}


def _apply_env_overrides(data: dict[str, Any]) -> None:
    """Override config values from environment variables.

    Supported env vars:
        AUTODFT_DB_URL        -> database.url
        AUTODFT_DATA_PATH     -> storage.data_path
        AUTODFT_COMP_DATA     -> storage.comp_data_path
        AUTODFT_EXPORT_DATA   -> storage.export_data_path
        AUTODFT_PARTITION     -> slurm.partition
        AUTODFT_API_PORT      -> api.port
        AUTODFT_LOOP_INTERVAL -> pipeline.loop_interval_seconds
        AUTODFT_SESSION_SECRET-> security.session_secret
    """
    import os

    env_map = [
        ("AUTODFT_DB_URL",         ["database", "url"]),
        ("AUTODFT_DATA_PATH",      ["storage", "data_path"]),
        ("AUTODFT_COMP_DATA",      ["storage", "comp_data_path"]),
        ("AUTODFT_EXPORT_DATA",    ["storage", "export_data_path"]),
        ("AUTODFT_PARTITION",      ["slurm", "partition"]),
        ("AUTODFT_API_PORT",       ["api", "port"]),
        ("AUTODFT_LOOP_INTERVAL",  ["pipeline", "loop_interval_seconds"]),
        ("AUTODFT_ORCA_PATH",      ["orca", "path"]),
        ("AUTODFT_ORCA_EXTRA",     ["orca", "extra_args"]),
        ("AUTODFT_NBO_EXE",        ["orca", "nbo_exe"]),
        ("AUTODFT_TMP_DIR",        ["orca", "tmp_dir"]),
        ("AUTODFT_SESSION_SECRET", ["security", "session_secret"]),
    ]

    for env_key, path in env_map:
        value = os.environ.get(env_key)
        if value is None:
            continue
        target = data
        for part in path[:-1]:
            target = target.setdefault(part, {})
        # Only coerce the keys that are genuinely numeric. Coercing every
        # override meant a numeric-looking secret arrived as an int:
        # issue_token() then raised AttributeError on .encode() and every
        # single request 500'd, with nothing pointing at the cause.
        if path[-1] in _INT_SETTINGS:
            try:
                value = int(value)  # type: ignore[assignment]
            except (ValueError, TypeError):
                logger.warning(
                    "Ignoring non-numeric value %r for %s", value, ".".join(path),
                )
                continue
        target[path[-1]] = value


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from *config_path*, falling back to defaults.

    Priority (highest wins): env vars > user config > default.toml
    """
    # Start with defaults
    with open(_DEFAULT_CONFIG, "rb") as f:
        data = tomllib.load(f)

    # Overlay user-supplied config
    if config_path is not None:
        user_path = Path(config_path)
        if user_path.exists():
            with open(user_path, "rb") as f:
                _merge(data, tomllib.load(f))

    # Environment variable overrides (highest priority)
    _apply_env_overrides(data)

    return _dict_to_dataclass(Settings, data)
