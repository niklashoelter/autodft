"""Global failure circuit breaker.

`max_attempts` bounds retries per task, but nothing bounded the campaign as a
whole. A systematic error -- a header with a basis set that doesn't cover one
element, a wrong charge convention, a cluster misconfiguration -- would fail
every molecule in turn, each burning its full retry budget with escalated
resources, and the pipeline would keep submitting until the queue was empty.
At the scale of a few thousand molecules that is millions of core-hours spent
producing nothing.

So: watch the recent judged tasks, and when the failure ratio crosses a
threshold, stop creating and submitting jobs. Work already in flight is left
alone -- this only stops *new* submissions.

The tripped state is a marker file under the data path rather than in-memory
state, so it survives a controller restart (a breaker you can reset by
restarting is not a breaker) and can be cleared from the dashboard without a
schema change.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlmodel import Session, col, select

from autodft.models.enums import TaskStatus
from autodft.models.task import ComputationTask

logger = logging.getLogger(__name__)

MARKER_FILENAME = "circuit_breaker.json"


def marker_path(data_path: Path) -> Path:
    return Path(data_path) / MARKER_FILENAME


def recent_failure_ratio(session: Session, window: int) -> tuple[float, int, int]:
    """Return ``(ratio, failed, judged)`` over the most recently judged tasks.

    Only ``successful`` / ``failed`` tasks count as judged -- tasks still
    running say nothing about whether the campaign is healthy.
    """
    tasks = session.exec(
        select(ComputationTask.status)
        .where(col(ComputationTask.status).in_([TaskStatus.successful, TaskStatus.failed]))
        .order_by(col(ComputationTask.updated_at).desc())
        .limit(window)
    ).all()

    judged = len(tasks)
    if judged == 0:
        return 0.0, 0, 0
    failed = sum(1 for status in tasks if status == TaskStatus.failed)
    return failed / judged, failed, judged


def read_state(data_path: Path) -> Optional[dict]:
    """Return the tripped-state payload, or None if the breaker is clear."""
    path = marker_path(data_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        # A corrupt marker still means tripped -- fail closed.
        return {"tripped_at": None, "detail": "unreadable marker file"}


def trip(data_path: Path, ratio: float, failed: int, judged: int) -> dict:
    """Write the marker file and return its payload."""
    payload = {
        "tripped_at": datetime.now(timezone.utc).isoformat(),
        "ratio": round(ratio, 4),
        "failed": failed,
        "judged": judged,
        "detail": (
            f"{failed} of the last {judged} judged tasks failed "
            f"({ratio:.0%}). New job creation and submission are halted."
        ),
    }
    path = marker_path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.critical(
        "CIRCUIT BREAKER TRIPPED: %s Reset it from the dashboard Admin page "
        "once the cause is fixed.", payload["detail"],
    )
    return payload


def reset(data_path: Path) -> bool:
    """Clear the tripped state. Returns True if it was tripped."""
    path = marker_path(data_path)
    if not path.exists():
        return False
    path.unlink()
    logger.warning("Circuit breaker reset; job submission resumes")
    return True


def check(session: Session, settings) -> Optional[dict]:
    """Return the tripped payload if new submissions should be blocked.

    Already-tripped stays tripped until explicitly reset: once submission
    stops, no new tasks are judged, so the ratio cannot recover on its own.
    """
    cfg = settings.pipeline
    if not cfg.failure_breaker_enabled:
        return None

    existing = read_state(settings.data_path)
    if existing is not None:
        return existing

    ratio, failed, judged = recent_failure_ratio(session, cfg.failure_breaker_window)
    if judged >= cfg.failure_breaker_min_samples and ratio >= cfg.failure_breaker_ratio:
        return trip(settings.data_path, ratio, failed, judged)
    return None
