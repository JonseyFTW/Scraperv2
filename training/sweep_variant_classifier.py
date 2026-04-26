#!/usr/bin/env python3
"""
Variant-classifier sweep runner.

Runs a list of training+eval configurations back-to-back, logs one row per
experiment into a CSV, and is safe to Ctrl-C and resume. Designed to be
kicked off before bed / on a Friday and come back to a ranked CSV.

Usage (from repo root):
    python training/sweep_variant_classifier.py \\
        --configs training/sweep_configs_default.json

Or an explicit output root:
    python training/sweep_variant_classifier.py \\
        --configs my_experiments.json \\
        --output-root ./sweeps/nightly_2026_04_23

Resume semantics:
    Re-running the same command skips any experiment whose name already has a
    completed row in ``results.csv``. Incomplete runs (e.g. interrupted
    mid-training) are re-run from scratch — checkpoints inside each run's
    output dir are overwritten.

Config file format (JSON):
    {
      "experiments": [
        {
          "name": "v1_mlp512_base",
          "arch": "mlp",
          "hidden_dim": 512,
          "dropout": 0.4,
          "feat_dropout": 0.1,
          "weight_decay": 1e-3,
          "label_smoothing": 0.15,
          "epochs": 80,
          "patience": 15,
          "sampler": "set_stratified",
          "sets_per_batch": 8,
          "scheduler": "cosine_warm_restarts",
          "hierarchical": true,
          "seed": 42,
          "finetuned_backbone": "./checkpoints/dinov2_finetuned_best.pt"
        },
        ...
      ]
    }

Every field is optional; defaults come from 05_train_variant_classifier.py.
Only ``name`` is required (used as the output dir + CSV row key).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Repo root importable + expose training/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

SCRIPT_DIR  = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SCRIPT_DIR.parent
TRAIN_SCRIPT = SCRIPT_DIR / "05_train_variant_classifier.py"
EVAL_SCRIPT  = SCRIPT_DIR / "06_eval_variant_classifier.py"


# ---------------------------------------------------------------------------
# Config → CLI translation
# ---------------------------------------------------------------------------

# Map config-dict keys to the training script's CLI flags. Keys missing from a
# given config are omitted from the command line (so training defaults apply).
_FLAG_MAP = {
    "arch":                "--arch",
    "hidden_dim":          "--hidden-dim",
    "dropout":             "--dropout",
    "feat_dropout":        "--feat-dropout",
    "weight_decay":        "--weight-decay",
    "label_smoothing":     "--label-smoothing",
    "epochs":              "--epochs",
    "batch":               "--batch",
    "lr":                  "--lr",
    "workers":             "--workers",
    "seed":                "--seed",
    "patience":            "--patience",
    "sampler":             "--sampler",
    "sets_per_batch":      "--sets-per-batch",
    "scheduler":           "--scheduler",
    "scheduler_t0":        "--scheduler-t0",
    "scheduler_tmult":     "--scheduler-tmult",
    "cache_batch":         "--cache-batch",
    "finetuned_backbone":  "--finetuned-backbone",
}

# Boolean flags — presence-only. If True, append the flag; if False/absent, skip.
_BOOL_FLAGS = {
    "hierarchical": "--hierarchical",
    "no_cache":     "--no-cache",
}


def _build_train_cmd(cfg: dict, output_dir: Path, data_dir: str | None) -> list[str]:
    cmd: list[str] = [sys.executable, str(TRAIN_SCRIPT), "--output-dir", str(output_dir)]
    if data_dir:
        cmd += ["--data-dir", data_dir]
    for key, flag in _FLAG_MAP.items():
        if key in cfg and cfg[key] is not None:
            cmd += [flag, str(cfg[key])]
    for key, flag in _BOOL_FLAGS.items():
        if cfg.get(key):
            cmd += [flag]
    return cmd


def _build_eval_cmd(ckpt_path: Path, eval_dir: Path, data_dir: str | None) -> list[str]:
    cmd: list[str] = [
        sys.executable, str(EVAL_SCRIPT),
        "--checkpoint", str(ckpt_path),
        "--output-dir", str(eval_dir),
    ]
    if data_dir:
        cmd += ["--data-dir", data_dir]
    return cmd


# ---------------------------------------------------------------------------
# Eval report parsing
# ---------------------------------------------------------------------------

def _read_per_class_metrics(path: Path, rare_threshold: int, common_threshold: int):
    """Return (rare_f1_mean, common_f1_mean, n_rare_supported, n_common_supported)."""
    rare_f1s: list[float] = []
    common_f1s: list[float] = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                support = int(row["support"])
                f1 = float(row["f1"])
            except (ValueError, KeyError):
                continue
            if support <= 0:
                continue
            if support < rare_threshold:
                rare_f1s.append(f1)
            if support >= common_threshold:
                common_f1s.append(f1)
    rare_mean = sum(rare_f1s) / len(rare_f1s) if rare_f1s else 0.0
    common_mean = sum(common_f1s) / len(common_f1s) if common_f1s else 0.0
    return rare_mean, common_mean, len(rare_f1s), len(common_f1s)


def _read_eval_summary(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# CSV layout
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "name",
    "status",
    "started_at",
    "finished_at",
    "wall_time_sec",
    "val_top1",
    "val_top3",
    "macro_f1",
    "rare_class_f1",
    "common_class_f1",
    "n_rare",
    "n_common",
    "n_labels",
    "n_sets",
    "arch",
    "hidden_dim",
    "dropout",
    "feat_dropout",
    "weight_decay",
    "label_smoothing",
    "epochs",
    "batch",
    "lr",
    "patience",
    "sampler",
    "sets_per_batch",
    "scheduler",
    "scheduler_t0",
    "hierarchical",
    "seed",
    "finetuned_backbone",
    "error",
]


def _load_existing_results(csv_path: Path) -> dict[str, dict]:
    """Return {name: row_dict} for already-completed experiments."""
    if not csv_path.exists():
        return {}
    rows: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("name")
            status = row.get("status")
            if name and status == "completed":
                rows[name] = row
    return rows


def _append_row(csv_path: Path, row: dict):
    """Atomically append one row; creates the file with a header if missing."""
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


# ---------------------------------------------------------------------------
# Per-experiment runner
# ---------------------------------------------------------------------------

_interrupted = False


def _sigint_handler(signum, frame):  # noqa: ARG001
    global _interrupted
    _interrupted = True
    console.print("\n[yellow]Ctrl-C received — finishing current run then stopping.[/yellow]")


def _stream_and_log(cmd: list[str], log_path: Path, cwd: Path) -> int:
    """Run a subprocess, stream output to both console and log file. Return exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    with open(log_path, "a", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, encoding="utf-8", errors="replace",
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        return proc.wait()


def _run_one_experiment(
    cfg: dict,
    output_root: Path,
    csv_path: Path,
    data_dir: str | None,
    rare_threshold: int,
    common_threshold: int,
):
    name = cfg["name"]
    run_dir  = output_root / "runs" / name
    eval_dir = output_root / "eval" / name
    log_path = run_dir / "stdout.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    row = {
        "name":               name,
        "status":             "running",
        "started_at":         started_at,
        "arch":               cfg.get("arch", "linear"),
        "hidden_dim":         cfg.get("hidden_dim", ""),
        "dropout":            cfg.get("dropout", ""),
        "feat_dropout":       cfg.get("feat_dropout", ""),
        "weight_decay":       cfg.get("weight_decay", ""),
        "label_smoothing":    cfg.get("label_smoothing", ""),
        "epochs":             cfg.get("epochs", ""),
        "batch":              cfg.get("batch", ""),
        "lr":                 cfg.get("lr", ""),
        "patience":           cfg.get("patience", ""),
        "sampler":            cfg.get("sampler", ""),
        "sets_per_batch":     cfg.get("sets_per_batch", ""),
        "scheduler":          cfg.get("scheduler", ""),
        "scheduler_t0":       cfg.get("scheduler_t0", ""),
        "hierarchical":       bool(cfg.get("hierarchical", False)),
        "seed":               cfg.get("seed", ""),
        "finetuned_backbone": cfg.get("finetuned_backbone", ""),
    }

    # Train
    try:
        train_cmd = _build_train_cmd(cfg, run_dir, data_dir)
        rc = _stream_and_log(train_cmd, log_path, cwd=PROJECT_DIR)
        if rc != 0:
            raise RuntimeError(f"training script exited with code {rc}")
    except KeyboardInterrupt:
        row.update(status="interrupted", finished_at=datetime.now(timezone.utc).isoformat(),
                   wall_time_sec=round(time.time() - t0, 1), error="KeyboardInterrupt")
        _append_row(csv_path, row)
        raise
    except Exception as e:
        row.update(status="failed", finished_at=datetime.now(timezone.utc).isoformat(),
                   wall_time_sec=round(time.time() - t0, 1), error=str(e))
        _append_row(csv_path, row)
        return row

    # Locate the best checkpoint. Training script writes
    # variant_classifier_{arch}.pt for the best val@1.
    arch = cfg.get("arch", "linear")
    ckpt_path = run_dir / f"variant_classifier_{arch}.pt"
    if not ckpt_path.exists():
        row.update(status="failed", finished_at=datetime.now(timezone.utc).isoformat(),
                   wall_time_sec=round(time.time() - t0, 1),
                   error=f"expected checkpoint missing: {ckpt_path}")
        _append_row(csv_path, row)
        return row

    # Eval — produces per_class_metrics.csv + top_confusions.csv + summary.json
    try:
        eval_cmd = _build_eval_cmd(ckpt_path, eval_dir, data_dir)
        rc = _stream_and_log(eval_cmd, log_path, cwd=PROJECT_DIR)
        if rc != 0:
            raise RuntimeError(f"eval script exited with code {rc}")
    except KeyboardInterrupt:
        row.update(status="interrupted", finished_at=datetime.now(timezone.utc).isoformat(),
                   wall_time_sec=round(time.time() - t0, 1), error="KeyboardInterrupt (during eval)")
        _append_row(csv_path, row)
        raise
    except Exception as e:
        row.update(status="failed", finished_at=datetime.now(timezone.utc).isoformat(),
                   wall_time_sec=round(time.time() - t0, 1), error=str(e))
        _append_row(csv_path, row)
        return row

    # Gather metrics
    summary = _read_eval_summary(eval_dir / "summary.json")
    per_class_path = eval_dir / "per_class_metrics.csv"
    rare_f1, common_f1, n_rare, n_common = _read_per_class_metrics(
        per_class_path, rare_threshold, common_threshold,
    ) if per_class_path.exists() else (0.0, 0.0, 0, 0)

    row.update(
        status="completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        wall_time_sec=round(time.time() - t0, 1),
        val_top1=summary.get("top1", ""),
        val_top3=summary.get("top3", ""),
        macro_f1=summary.get("macro_f1", ""),
        rare_class_f1=round(rare_f1, 4),
        common_class_f1=round(common_f1, 4),
        n_rare=n_rare,
        n_common=n_common,
        n_labels=summary.get("n_labels", ""),
        n_sets=summary.get("n_sets", ""),
    )
    _append_row(csv_path, row)
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _validate_configs(experiments: list[dict]):
    seen = set()
    for i, cfg in enumerate(experiments):
        name = cfg.get("name")
        if not name:
            raise ValueError(f"experiment #{i} is missing 'name'")
        if name in seen:
            raise ValueError(f"duplicate experiment name: {name}")
        seen.add(name)


def _render_plan_table(experiments: list[dict], completed: set[str]) -> Table:
    table = Table(title="Sweep plan", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Arch")
    table.add_column("Reg (dropout/feat/wd/ls)", style="dim")
    table.add_column("Sampler")
    table.add_column("Epochs")
    table.add_column("Status")
    for cfg in experiments:
        name = cfg["name"]
        reg = f"{cfg.get('dropout','-')}/{cfg.get('feat_dropout','-')}/{cfg.get('weight_decay','-')}/{cfg.get('label_smoothing','-')}"
        status = "[green]done[/green]" if name in completed else "[yellow]pending[/yellow]"
        table.add_row(
            name,
            str(cfg.get("arch", "linear")) + (f" h{cfg['hidden_dim']}" if cfg.get("hidden_dim") else ""),
            reg,
            str(cfg.get("sampler", "set_stratified")),
            str(cfg.get("epochs", 30)),
            status,
        )
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", required=True,
                    help="Path to JSON config file listing experiments.")
    ap.add_argument("--output-root", default="./sweeps/default",
                    help="Root dir for runs/, eval/, results.csv.")
    ap.add_argument("--data-dir", default=None,
                    help="Variant-classifier manifest dir (passed through to 05/06). "
                         "Defaults to each script's own default.")
    ap.add_argument("--rare-threshold", type=int, default=20,
                    help="Classes with support < this count as 'rare' in the report.")
    ap.add_argument("--common-threshold", type=int, default=100,
                    help="Classes with support >= this count as 'common' in the report.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan without running anything.")
    args = ap.parse_args()

    cfg_path = Path(args.configs)
    if not cfg_path.exists():
        console.print(f"[red]Config file not found: {cfg_path}[/red]")
        sys.exit(2)

    cfg_data = json.loads(cfg_path.read_text())
    experiments = cfg_data.get("experiments") or []
    if not experiments:
        console.print("[red]No 'experiments' found in config file.[/red]")
        sys.exit(2)
    _validate_configs(experiments)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "results.csv"

    # Snapshot the config file we're running into the output dir for reproducibility.
    (output_root / "config.json").write_text(json.dumps(cfg_data, indent=2))

    completed = set(_load_existing_results(csv_path).keys())
    console.print(Panel.fit(
        f"[bold]Variant-classifier sweep[/bold]\n"
        f"  configs:       [cyan]{cfg_path}[/cyan]\n"
        f"  output:        [cyan]{output_root}[/cyan]\n"
        f"  experiments:   [cyan]{len(experiments)}[/cyan]  "
        f"([green]{len(completed)} already done[/green])\n"
        f"  rare support:  < {args.rare_threshold}\n"
        f"  common support: >= {args.common_threshold}",
        border_style="cyan",
    ))
    console.print(_render_plan_table(experiments, completed))

    if args.dry_run:
        console.print("[dim]--dry-run — exiting without running.[/dim]")
        return

    signal.signal(signal.SIGINT, _sigint_handler)

    for cfg in experiments:
        if _interrupted:
            break
        name = cfg["name"]
        if name in completed:
            console.print(f"[dim]Skip {name} (already completed)[/dim]")
            continue
        console.print(Panel(f"[bold cyan]Running {name}[/bold cyan]", border_style="cyan"))
        try:
            row = _run_one_experiment(
                cfg, output_root, csv_path, args.data_dir,
                args.rare_threshold, args.common_threshold,
            )
        except KeyboardInterrupt:
            console.print(f"[yellow]Interrupted during {name}. Re-run to resume.[/yellow]")
            break
        status = row.get("status", "?")
        if status == "completed":
            console.print(
                f"[green]✓ {name}[/green]  "
                f"val@1={row['val_top1']:.4f}  val@3={row['val_top3']:.4f}  "
                f"macro_f1={row['macro_f1']:.4f}  "
                f"rare_f1={row['rare_class_f1']:.4f}  common_f1={row['common_class_f1']:.4f}  "
                f"({row['wall_time_sec']:.0f}s)"
            )
        else:
            console.print(f"[red]✗ {name} ({status})[/red] — {row.get('error','')}")

    _print_leaderboard(csv_path)


def _print_leaderboard(csv_path: Path, top_n: int = 10):
    if not csv_path.exists():
        return
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "completed":
                continue
            try:
                rows.append({
                    "name":            row["name"],
                    "val_top1":        float(row["val_top1"]),
                    "val_top3":        float(row["val_top3"]),
                    "macro_f1":        float(row["macro_f1"]),
                    "rare_class_f1":   float(row.get("rare_class_f1") or 0),
                    "common_class_f1": float(row.get("common_class_f1") or 0),
                    "wall_time_sec":   float(row.get("wall_time_sec") or 0),
                })
            except (ValueError, KeyError):
                continue
    if not rows:
        return
    rows.sort(key=lambda r: r["val_top1"], reverse=True)
    table = Table(title=f"Top {min(top_n, len(rows))} by val@1", show_header=True)
    for col in ("name", "val@1", "val@3", "macro_f1", "rare_f1", "common_f1", "wall (s)"):
        table.add_column(col)
    for r in rows[:top_n]:
        table.add_row(
            r["name"],
            f"{r['val_top1']:.4f}",
            f"{r['val_top3']:.4f}",
            f"{r['macro_f1']:.4f}",
            f"{r['rare_class_f1']:.4f}",
            f"{r['common_class_f1']:.4f}",
            f"{r['wall_time_sec']:.0f}",
        )
    console.print(table)


if __name__ == "__main__":
    main()
