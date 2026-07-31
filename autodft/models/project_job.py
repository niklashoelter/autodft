"""ProjectJob model -- a durable, user-attached background operation.

Archiving or exporting a project copies or deletes thousands of files over a
network mount and takes minutes. Running that inside the HTTP request tied its
lifetime to the browser tab and blocked the response for its whole duration.

A ``ProjectJob`` row moves that work off the request:

* it is started on a background thread, so closing the website does not stop
  it -- the row records the outcome regardless;
* the row *is* the per-project lock. A project with a row in ``running`` has
  an operation in flight, and every other mutating operation on that project
  (a second archive/export, a wipe, a new submission) is refused until it
  finishes. The controller also skips advancing that project's tasks while a
  job runs (see ``engine.state_machine``);
* it is attached to a user (``owner_id``) so the dashboard can show whose job
  it is and scope who may read or download its result.

Durability is deliberately "survives the website, not the process": the thread
lives in the controller process, so a controller restart orphans an in-flight
job. Startup reconciles any row left ``running`` to ``failed`` rather than
resuming it, because the destructive archive is not safe to restart blindly.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel

from autodft.models.enums import ProjectJobKind, ProjectJobStatus


class ProjectJob(SQLModel, table=True):
    __tablename__ = "project_jobs"

    id: Optional[int] = Field(default=None, primary_key=True)

    # The ``owner/project`` this operates on, matching ``molecules.project_name``.
    qualified_name: str = Field(index=True, max_length=161)

    # The user the job is attached to. Nullable only for the shared-password
    # admin on a database that predates accounts (identity.user_id is None
    # there); every real account sets it.
    owner_id: Optional[int] = Field(default=None, foreign_key="users.id", index=True)

    kind: ProjectJobKind
    # Indexed because the controller's tick queries "which projects are
    # currently running a job" on every cycle.
    status: ProjectJobStatus = Field(default=ProjectJobStatus.running, index=True)

    # The request options (extensions, all_conformers, ...) as JSON, so the
    # worker thread can reconstruct the call without holding request state.
    params_json: str = Field(default="{}")
    # The summary dict on success (paths produced, counts), as JSON.
    result_json: Optional[str] = None
    # The failure message on failure.
    error: Optional[str] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
