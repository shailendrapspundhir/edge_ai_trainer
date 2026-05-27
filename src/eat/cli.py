"""Typer CLI: `eat <subcommand>`.

Subcommands are organised by noun:

    eat registry  ...    inspect models/datasets/recipes/targets
    eat project   ...    list / init projects
    eat run       ...    submit / status / logs
    eat export    ...    one-off export of a finished run
    eat android   ...    bench on a connected device
    eat cloud     ...    build / deploy / bench containers
    eat eval      ...    quality / safety / perf eval on a run

The CLI is a thin wrapper around modules; it should not contain business logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from eat.config import get_settings
from eat.logging_setup import get_logger, setup_logging
from eat.paths import ensure_dirs

app = typer.Typer(
    help="Edge AI Trainer (eat) — train, evaluate, and deploy edge AI models.",
    no_args_is_help=True,
    add_completion=False,
)

registry_app = typer.Typer(help="Inspect the model/dataset/recipe/target registry.", no_args_is_help=True)
project_app = typer.Typer(help="Manage projects (one project = one training pipeline).", no_args_is_help=True)
runs_app = typer.Typer(help="Inspect runs (status, list, logs, cancel).", no_args_is_help=True)
export_app = typer.Typer(help="Export a finished run to a target format.", no_args_is_help=True)
android_app = typer.Typer(help="Benchmark on a connected Android device.", no_args_is_help=True)
cloud_app = typer.Typer(help="Build, deploy, and benchmark cloud serving containers.", no_args_is_help=True)
eval_app = typer.Typer(help="Run quality, safety, or perf evaluation on a run.", no_args_is_help=True)
data_app = typer.Typer(help="Prepare, inspect, and manage training datasets.", no_args_is_help=True)
sweep_app = typer.Typer(help="Hyperparameter sweep engine — ASHA-scheduled multi-objective search.", no_args_is_help=True)

app.add_typer(registry_app, name="registry")
app.add_typer(project_app, name="project")
app.add_typer(runs_app, name="runs")
app.add_typer(export_app, name="export")
app.add_typer(android_app, name="android")
app.add_typer(cloud_app, name="cloud")
app.add_typer(eval_app, name="eval")
app.add_typer(data_app, name="data")
app.add_typer(sweep_app, name="sweep")

console = Console()
log = get_logger("cli")


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging.")) -> None:
    if verbose:
        import os
        os.environ["EAT_LOG_LEVEL"] = "DEBUG"
    setup_logging(force=True)
    ensure_dirs()


@app.command()
def version() -> None:
    """Print version info."""
    from eat import __version__
    console.print(f"edge-ai-trainer [bold]{__version__}[/bold]")


@app.command()
def doctor() -> None:
    """Sanity-check the environment (paths, redis, hf token, GPU)."""
    from eat.doctor import run_doctor
    ok = run_doctor()
    raise typer.Exit(code=0 if ok else 1)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


@registry_app.command("list")
def registry_list(
    kind: str = typer.Argument("all", help="all | models | datasets | recipes | targets"),
) -> None:
    """List registered entries."""
    from eat.registry import datasets, models, recipes, targets

    show = {"all", "models"}
    if kind in show:
        _table("Models", models.all_entries(), ["name", "family", "hf_id", "multimodal", "gated"])
    if kind in {"all", "datasets"}:
        _table("Datasets", datasets.all_entries(), ["name", "kind", "source", "location"])
    if kind in {"all", "recipes"}:
        _table("Recipes", recipes.all_entries(), ["name", "backend", "method", "base_model"])
    if kind in {"all", "targets"}:
        _table("Targets", targets.all_entries(), ["name", "kind", "platform", "runtime", "quantization"])


@registry_app.command("show")
def registry_show(name: str, kind: str = typer.Option(..., help="model|dataset|recipe|target")) -> None:
    from eat.registry import datasets, models, recipes, targets
    mapping = {
        "model": models.get,
        "dataset": datasets.get,
        "recipe": recipes.get,
        "target": targets.get,
    }
    entry = mapping[kind](name)
    console.print_json(entry.model_dump_json(indent=2))


def _table(title: str, items, columns: list[str]) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    for c in columns:
        table.add_column(c)
    for it in items:
        row = []
        for c in columns:
            v = getattr(it, c, "")
            if isinstance(v, list):
                v = ",".join(map(str, v))
            row.append(str(v) if v is not None else "")
        table.add_row(*row)
    console.print(table)


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------


@project_app.command("list")
def project_list() -> None:
    from eat.registry import projects
    table = Table(title="Projects", box=box.SIMPLE_HEAVY)
    for c in ("name", "base_model", "recipes", "targets"):
        table.add_column(c)
    for p in projects.all_specs():
        table.add_row(p.name, p.base_model, ",".join(p.recipes), ",".join(p.targets))
    console.print(table)


@project_app.command("init")
def project_init(name: str) -> None:
    """Scaffold a new project under projects/<name>/."""
    from eat.registry import projects
    path = projects.scaffold(name)
    console.print(f"Created [green]{path}[/green]. Edit `project.yaml` and add prompts/data.")


@project_app.command("show")
def project_show(name: str) -> None:
    from eat.registry import projects
    spec = projects.get(name)
    console.print_json(spec.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command("run")
def run_submit(
    project: str = typer.Option(..., help="Project name."),
    recipe: str = typer.Option(..., help="Recipe name (must be listed under the project)."),
    targets: Optional[list[str]] = typer.Option(None, help="Override export targets."),
    compute: str = typer.Option("local_gpu", help="local_cpu | local_gpu | cloud_gpu"),
    dry_run: bool = typer.Option(False, help="Plan the run but do not enqueue jobs."),
    notes: str = typer.Option("", help="Free-form notes saved with the run."),
) -> None:
    """Submit a new run to the orchestrator."""
    from eat.orchestrator import service
    from eat.types import Compute, RunSpec

    spec = RunSpec(
        project=project,
        recipe=recipe,
        targets=targets or [],
        compute=Compute(compute),
        dry_run=dry_run,
        notes=notes,
    )
    run = service.submit_run(spec)
    console.print(f"Run [bold cyan]{run.id}[/bold cyan] created with {len(run.job_ids)} jobs.")
    if dry_run:
        console.print("[yellow]dry-run[/yellow]: jobs were planned but not enqueued.")
    console.print(f"Watch: [dim]eat runs logs {run.id} --follow[/dim]")


@runs_app.command("status")
def run_status(run_id: str) -> None:
    from eat.orchestrator import service
    run, jobs = service.get_run(run_id)
    console.print(f"[bold]{run.id}[/bold]  project={run.project}  recipe={run.recipe}  "
                  f"compute={run.compute.value}  status=[cyan]{run.status.value}[/cyan]")
    table = Table(box=box.SIMPLE)
    for c in ("id", "kind", "status", "queue", "started", "finished", "error"):
        table.add_column(c)
    for j in jobs:
        table.add_row(
            j.id, j.kind.value, j.status.value, j.queue or "",
            str(j.started_at or ""), str(j.finished_at or ""),
            (j.error or "")[:80],
        )
    console.print(table)


@runs_app.command("list")
def run_list(project: Optional[str] = None, limit: int = 20) -> None:
    from eat.orchestrator import service
    runs = service.list_runs(project=project, limit=limit)
    table = Table(title="Runs", box=box.SIMPLE_HEAVY)
    for c in ("id", "project", "recipe", "status", "created"):
        table.add_column(c)
    for r in runs:
        table.add_row(r.id, r.project, r.recipe, r.status.value, str(r.created_at))
    console.print(table)


@runs_app.command("logs")
def run_logs(run_id: str, follow: bool = typer.Option(False, "--follow", "-f")) -> None:
    from eat.orchestrator import service
    service.tail_logs(run_id, follow=follow, sink=console.print)


@runs_app.command("cancel")
def run_cancel(run_id: str) -> None:
    from eat.orchestrator import service
    service.cancel_run(run_id)
    console.print(f"Cancelled {run_id}")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@export_app.command("gguf")
def export_gguf(run_id: str, quant: str = "Q4_K_M") -> None:
    from eat.export import gguf
    out = gguf.export(run_id, quant=quant)
    console.print(f"GGUF written: [green]{out}[/green]")


@export_app.command("mediapipe")
def export_mediapipe(run_id: str) -> None:
    from eat.export import mediapipe
    out = mediapipe.export(run_id)
    console.print(f"MediaPipe .task written: [green]{out}[/green]")


@export_app.command("onnx")
def export_onnx(run_id: str) -> None:
    from eat.export import onnx_export
    out = onnx_export.export(run_id)
    console.print(f"ONNX written: [green]{out}[/green]")


# ---------------------------------------------------------------------------
# android
# ---------------------------------------------------------------------------


@android_app.command("devices")
def android_devices() -> None:
    from eat.android import adb_runner
    for d in adb_runner.list_devices():
        console.print(f"  {d.serial}  {d.model}  android={d.android_version}  state={d.state}")


@android_app.command("install")
def android_install(apk: Path, device: Optional[str] = None) -> None:
    from eat.android import adb_runner
    adb_runner.install_apk(apk, serial=device)
    console.print(f"Installed {apk}")


@android_app.command("bench")
def android_bench(run_id: str, device: Optional[str] = None, target: str = "gguf_q4km") -> None:
    from eat.android import bench
    result = bench.run_bench(run_id, target=target, device_serial=device)
    console.print_json(json.dumps(result))


# ---------------------------------------------------------------------------
# cloud
# ---------------------------------------------------------------------------


@cloud_app.command("build")
def cloud_build(run_id: str, target: str = "cloud_vllm_gpu") -> None:
    from eat.cloud import containers
    image = containers.build(run_id, target=target)
    console.print(f"Built image: [green]{image}[/green]")


@cloud_app.command("deploy")
def cloud_deploy(run_id: str, target: str = "cloud_modal") -> None:
    from eat.cloud import deploy
    endpoint = deploy.deploy(run_id, target=target)
    console.print(f"Deployed: [green]{endpoint}[/green]")


@cloud_app.command("bench")
def cloud_bench(run_id: str, endpoint: Optional[str] = None) -> None:
    from eat.cloud import bench_remote
    result = bench_remote.bench(run_id, endpoint=endpoint)
    console.print_json(json.dumps(result))


@cloud_app.command("destroy")
def cloud_destroy(deployment_id: str, target: str = "cloud_modal") -> None:
    from eat.cloud import deploy
    deploy.destroy(deployment_id, target=target)
    console.print(f"Destroyed {deployment_id}")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@data_app.command("build")
def data_build(
    project: str = typer.Option(..., help="Project name."),
    dataset: Optional[str] = typer.Option(None, help="Single dataset name; omit to build all datasets for the project."),
) -> None:
    """Materialise training data for a project (downloads HF datasets, runs synthetic gen, etc.)."""
    from eat.data.loaders import prepare
    from eat.registry import datasets as datasets_reg, projects as projects_reg

    if dataset:
        ds_names = [dataset]
    else:
        proj = projects_reg.get(project)
        ds_names = list(proj.datasets or [])
        if proj.eval_set and proj.eval_set not in ds_names:
            ds_names.append(proj.eval_set)

    if not ds_names:
        console.print(f"[yellow]No datasets configured for project '{project}'.[/yellow]")
        raise typer.Exit(1)

    console.print(f"Building [bold]{len(ds_names)}[/bold] dataset(s) for project [cyan]{project}[/cyan]…\n")
    errors: list[str] = []
    for name in ds_names:
        try:
            stats = prepare(project=project, dataset=name)
            count = stats.get("count", "?")
            path = stats.get("path", "")
            console.print(f"  [green]✓[/green] {name:<35} {count:>6} records  →  {path}")
        except Exception as exc:
            console.print(f"  [red]✗[/red] {name:<35} [red]{exc}[/red]")
            errors.append(name)

    console.print()
    if errors:
        console.print(f"[red]Failed:[/red] {', '.join(errors)}")
        raise typer.Exit(1)
    console.print(f"[green]Done.[/green] All datasets ready under projects/{project}/data/processed/")


@data_app.command("list")
def data_list(project: str = typer.Option(..., help="Project name.")) -> None:
    """List all datasets registered for a project and their on-disk status."""
    from eat.registry import datasets as datasets_reg, projects as projects_reg
    from eat.paths import repo_root

    proj = projects_reg.get(project)
    ds_names = list(proj.datasets or [])
    if proj.eval_set and proj.eval_set not in ds_names:
        ds_names.append(proj.eval_set)

    table = Table(title=f"Datasets — {project}", box=box.SIMPLE_HEAVY)
    for col in ("name", "kind", "source", "records", "path", "status"):
        table.add_column(col)

    for name in ds_names:
        try:
            entry = datasets_reg.get(name)
            p = Path(entry.location)
            if not p.is_absolute():
                p = repo_root() / p
            if p.exists():
                n = sum(1 for line in p.open() if line.strip())
                status = "[green]ready[/green]" if n > 0 else "[yellow]empty[/yellow]"
            else:
                n = 0
                status = "[red]missing[/red]"
            table.add_row(name, entry.kind, entry.source, str(n), str(p.relative_to(repo_root())), status)
        except Exception as exc:
            table.add_row(name, "?", "?", "?", "?", f"[red]error: {exc}[/red]")

    console.print(table)


@data_app.command("stats")
def data_stats(
    project: str = typer.Option(..., help="Project name."),
    dataset: str = typer.Option(..., help="Dataset name."),
) -> None:
    """Print record count and a few sample records from a processed JSONL file."""
    from eat.registry import datasets as datasets_reg
    from eat.paths import repo_root

    entry = datasets_reg.get(dataset)
    p = Path(entry.location)
    if not p.is_absolute():
        p = repo_root() / p

    if not p.exists():
        console.print(f"[red]File not found:[/red] {p}\nRun: eat data build --project {project} --dataset {dataset}")
        raise typer.Exit(1)

    lines = [line for line in p.read_text().splitlines() if line.strip()]
    console.print(f"[bold]{dataset}[/bold]  ({entry.kind}, {entry.source})  —  [cyan]{len(lines)} records[/cyan]\n")
    console.print(f"Path: {p}\n")
    for i, line in enumerate(lines[:3], 1):
        rec = json.loads(line)
        console.print(f"[dim]── sample {i} ──────────────────────────────[/dim]")
        console.print_json(json.dumps(rec, indent=2))


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@eval_app.command("quality")
def eval_quality(run_id: str) -> None:
    from eat.eval import quality
    report = quality.evaluate(run_id)
    console.print_json(json.dumps(report))


@eval_app.command("safety")
def eval_safety(run_id: str) -> None:
    from eat.eval import safety
    report = safety.evaluate(run_id)
    console.print_json(json.dumps(report))


@eval_app.command("perf")
def eval_perf(run_id: str, runtime: str = "llama.cpp", quant: str = "Q4_K_M") -> None:
    from eat.eval import perf
    report = perf.evaluate(run_id, runtime=runtime, quant=quant)
    console.print_json(json.dumps(report))


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


@sweep_app.command("run")
def sweep_run(
    project: str = typer.Option(..., help="Project name."),
    recipe: str = typer.Option(..., help="Base recipe to sweep over."),
    search_space: Path = typer.Option(..., help="Path to search_space.yaml."),
    objectives: Path = typer.Option(..., help="Path to objectives.yaml."),
    data_path: Path = typer.Option(..., help="Path to training data JSONL."),
    model_path: Path = typer.Option(..., help="Path to local model directory."),
    output_dir: Path = typer.Option("./sweep_output", help="Output directory for sweep artefacts."),
    max_trials: int = typer.Option(6, help="Maximum number of HP trials."),
    asha_max_rung: int = typer.Option(3, help="ASHA max rung (number of halving rounds)."),
    asha_reduction: int = typer.Option(3, help="ASHA reduction factor (keep 1/N at each rung)."),
    asha_min_epochs: float = typer.Option(1.0, help="Epochs at rung 0."),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility."),
) -> None:
    """Launch an ASHA-scheduled hyperparameter sweep."""
    from eat.sweep.config import load_objectives, load_search_space
    from eat.sweep.engine import run_sweep

    ss = load_search_space(search_space)
    obj = load_objectives(objectives)

    console.print(f"[bold cyan]Sweep[/bold cyan]: {max_trials} trials, "
                  f"{len(ss.params)} HP axes, {len(obj.objectives)} objectives")
    console.print(f"  ASHA: rungs={asha_max_rung} reduction={asha_reduction} min_epochs={asha_min_epochs}")

    import sys
    result = run_sweep(
        project=project,
        recipe=recipe,
        search_space=ss,
        objectives=obj,
        max_trials=max_trials,
        data_path=str(data_path),
        model_path=str(model_path),
        output_dir=str(output_dir),
        asha_max_rung=asha_max_rung,
        asha_reduction_factor=asha_reduction,
        asha_min_epochs=asha_min_epochs,
        log_file=sys.stdout,
        seed=seed,
    )

    console.print(f"\n[bold green]Sweep complete[/bold green]: {result.id}")
    console.print(f"  Status: {result.status.value}")
    console.print(f"  Best trial: {result.best_trial_id}")
    summary = Path(output_dir) / result.id / "sweep_summary.json"
    if summary.exists():
        console.print(f"  Summary: {summary}")


# ---------------------------------------------------------------------------


def main() -> int:
    try:
        app()
    except typer.Exit as exc:
        return exc.exit_code
    except Exception:
        log.exception("cli_unhandled")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
