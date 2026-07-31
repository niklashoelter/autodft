"""Start the AutoDFT pipeline worker."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import typer
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(name="run", help="Start the pipeline worker.")


@app.callback(invoke_without_command=True)
def run(
    config: Optional[str] = typer.Option(None, "--config", help="Path to config TOML file"),
    scheduler: str = typer.Option("slurm", "--scheduler", help="Scheduler backend: slurm or local"),
    once: bool = typer.Option(False, "--once", help="Run a single pipeline tick then exit"),
) -> None:
    """Start the pipeline worker loop."""
    from autodft.config import load_settings

    settings = load_settings(config)

    # Select scheduler backend
    if scheduler == "local":
        from autodft.engine.scheduler import LocalScheduler
        sched = LocalScheduler()
        console.print("[cyan]Using local scheduler (testing mode)[/cyan]")
    elif scheduler == "slurm":
        from autodft.engine.scheduler import SlurmScheduler
        sched = SlurmScheduler(
            partition=settings.slurm.partition,
            nice=settings.slurm.nice,
        )
        console.print("[cyan]Using SLURM scheduler[/cyan]")
    else:
        console.print(f"[red]Unknown scheduler:[/red] {scheduler}")
        raise typer.Exit(code=1)

    # Create the QM engine (orca config injected for submit.cmd generation)
    from autodft.qm.orca.parser import OrcaParser
    qm_engine = OrcaParser(orca=settings.orca)

    # Initialize the database
    from autodft.db import init_db
    init_db(settings)

    # A project archive/export runs on a background thread in this process. A
    # restart orphans any that were in flight; fail them (and unblock their
    # projects) rather than leaving the rows wedged in `running` forever.
    from autodft.api.project_jobs import reconcile_orphaned_jobs
    reconcile_orphaned_jobs(settings)

    # Optionally start the FastAPI dashboard in a background thread
    if settings.api.enabled:
        _start_api_server(settings)

    from autodft.engine.pipeline import PipelineWorker

    worker = PipelineWorker(settings=settings, scheduler=sched, qm_engine=qm_engine)

    if once:
        console.print("[cyan]Running single pipeline tick...[/cyan]")
        worker.tick()
        console.print("[green]Tick complete.[/green]")
    else:
        console.print(
            f"[cyan]Starting pipeline worker (interval={settings.pipeline.loop_interval_seconds}s)...[/cyan]"
        )
        console.print("Press Ctrl+C to stop.")
        try:
            worker.run_forever()
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            logger.info("Pipeline worker stopped by user")


def _start_api_server(settings) -> None:  # noqa: ANN001
    """Launch the FastAPI dashboard in a daemon thread."""
    try:
        import uvicorn

        from autodft.api.app import create_app  # type: ignore[import-untyped]

        api_app = create_app(settings)

        def _serve() -> None:
            uvicorn.run(
                api_app,
                host=settings.api.host,
                port=settings.api.port,
                log_level="warning",
            )

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        console.print(
            f"[cyan]API dashboard started at http://{settings.api.host}:{settings.api.port}[/cyan]"
        )
    except ImportError:
        logger.warning("uvicorn or autodft.api not available; skipping API dashboard")
    except Exception:
        logger.exception("Failed to start API dashboard")
