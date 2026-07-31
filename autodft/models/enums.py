"""Enumerations shared across models."""

from enum import Enum


class TaskType(str, Enum):
    confsearch = "confsearch"
    optimization = "optimization"
    singlepoint = "singlepoint"
    singlepoint_vert_ox = "singlepoint_vert_ox"
    singlepoint_vert_red = "singlepoint_vert_red"
    singlepoint_vert_spin_change = "singlepoint_vert_spin_change"
    singlepoint_nbo = "singlepoint_nbo"


class TaskStatus(str, Enum):
    created = "created"
    pending = "pending"
    successful = "successful"
    failed = "failed"


class ProjectJobKind(str, Enum):
    """The long-running, user-triggered project operations that run as a
    durable background job rather than inside the HTTP request."""

    archive = "archive"
    export_csv = "export_csv"
    export_json = "export_json"
    export_files = "export_files"
    export_xlsx = "export_xlsx"


class ProjectJobStatus(str, Enum):
    """Lifecycle of a :class:`~autodft.models.project_job.ProjectJob`.

    There is no ``pending``: the worker thread is spawned the moment the row
    is written, so a job is ``running`` from creation. A row still ``running``
    at controller startup was orphaned by a restart and is reconciled to
    ``failed`` (see ``project_jobs.reconcile_orphaned_jobs``)."""

    running = "running"
    successful = "successful"
    failed = "failed"


class SlurmStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


# `slurm_status` holds the raw sacct string, so these sets -- not the enum --
# decide what the pipeline does with a job.
#
# Transient: the job is not finished. Keep polling; do NOT parse its output.
# COMPLETING in particular is common (epilog, node drain) and can last
# minutes on a network filesystem.
TRANSIENT_SLURM_STATES = frozenset({
    "PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED",
    "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESIZING", "SIGNALING",
    "STAGE_OUT", "STOPPED", "UNKNOWN",
})

# Terminal: the job is over, one way or another, and its output can be read.
TERMINAL_SLURM_STATES = frozenset({
    "COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
    "SPECIAL_EXIT",
})
