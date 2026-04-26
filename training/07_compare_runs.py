#!/usr/bin/env python3
"""
Step 7 (Part B): Compare two or more variant-classifier training runs.

Reads the ``summary.json`` + ``top_confusions.csv`` + ``per_class_metrics.csv``
that step 5's auto-eval (and step 6) drop into ``eval/runs/<run_name>/`` and
prints a side-by-side ablation table.

Three views, in order:

    1. Headline metrics   — top1 / top3 / macro F1 per run, with deltas vs baseline.
    2. Known hard-pair confusion tracker — for the curated list of confusion pairs
       (Topps Now color foils, UD exclusives ladder, OPC borders, mosaic finishes,
       die-cut vs base, etc.), report the count per run and whether it improved.
    3. Top regressions / improvements — confusion pairs whose count changed most
       between baseline and each subsequent run.

The first run on the command line is the baseline; later runs are diffed
against it. If no paths are passed, every immediate subdirectory of
``--root`` (default ``./eval/runs``) is loaded in mtime order, with the
oldest treated as the baseline.

Examples:
    python training/07_compare_runs.py
    python training/07_compare_runs.py ./eval/runs/mlp__20260422 ./eval/runs/mlp__color=lab32__20260427
    python training/07_compare_runs.py --root ./eval/runs --top 30
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


# Confusion pairs the user has flagged as systematic — see the original
# investigation in this branch's commit history. We track these explicitly so
# regressions on the named hard pairs surface even when overall metrics move.
# Each entry is (true_label, pred_label) — we also count the reverse direction
# under the same group so a "purple↔blue" mix-up doesn't get split in half.
KNOWN_HARD_PAIRS: list[tuple[str, str, str]] = [
    # group_name, true_label, pred_label
    ("topps-now-2025 orange/gold foil", "baseball-cards-2025-topps-now::orange foil",   "baseball-cards-2025-topps-now::gold foil"),
    ("topps-now-2025 orange/gold foil", "baseball-cards-2025-topps-now::gold foil",     "baseball-cards-2025-topps-now::orange foil"),
    ("topps-now-2025 black/gold foil",  "baseball-cards-2025-topps-now::black foil",    "baseball-cards-2025-topps-now::gold foil"),
    ("topps-now-2025 red/orange foil",  "baseball-cards-2025-topps-now::red foil",      "baseball-cards-2025-topps-now::orange foil"),
    ("topps-now-2022 purple/blue",      "baseball-cards-2022-topps-now::purple",        "baseball-cards-2022-topps-now::blue"),
    ("topps-now-2022 purple/blue",      "baseball-cards-2022-topps-now::blue",          "baseball-cards-2022-topps-now::purple"),
    ("topps-now-2023 purple/blue",      "baseball-cards-2023-topps-now::purple",        "baseball-cards-2023-topps-now::blue"),
    ("topps-now-2021 purple/blue",      "baseball-cards-2021-topps-now::purple",        "baseball-cards-2021-topps-now::blue"),
    ("topps-now-2021 purple/blue",      "baseball-cards-2021-topps-now::blue",          "baseball-cards-2021-topps-now::purple"),
    ("topps-now-2022 red/blue",         "baseball-cards-2022-topps-now::red",           "baseball-cards-2022-topps-now::blue"),
    ("topps-now-2023 red/blue",         "baseball-cards-2023-topps-now::red",           "baseball-cards-2023-topps-now::blue"),
    ("topps-heritage red/black border", "baseball-cards-2023-topps-heritage::red border", "baseball-cards-2023-topps-heritage::black border"),
    ("topps-heritage red/black border", "baseball-cards-2021-topps-heritage::red",      "baseball-cards-2021-topps-heritage::black border"),
    ("topps-heritage-2024 black/white", "baseball-cards-2024-topps-heritage::black",    "baseball-cards-2024-topps-heritage::white"),
    ("topps-day mother/father ribbon",  "baseball-cards-2024-topps-day::mother's day pink", "baseball-cards-2024-topps-day::father's day blue"),
    ("topps-day mother/father ribbon",  "baseball-cards-2024-topps-day::father's day blue", "baseball-cards-2024-topps-day::mother's day pink"),
    ("topps-2022 rainbow/gold foil",    "baseball-cards-2022-topps::rainbow",           "baseball-cards-2022-topps::gold foil"),
    ("topps-2022 rainbow/gold foil",    "baseball-cards-2022-topps::gold foil",         "baseball-cards-2022-topps::rainbow"),
    ("topps-2023 rainbow/gold foil",    "baseball-cards-2023-topps::gold foil",         "baseball-cards-2023-topps::rainbow foil"),
    ("topps-2023 orange/red foil",      "baseball-cards-2023-topps::orange foil",       "baseball-cards-2023-topps::red foil"),
    ("topps-2024 rainbow/gold foil",    "baseball-cards-2024-topps::gold foil",         "baseball-cards-2024-topps::rainbow foil"),
    ("topps-2024 purple/blue holofoil", "baseball-cards-2024-topps::purple holofoil",   "baseball-cards-2024-topps::blue holofoil"),
    ("topps-2024 purple foil/blue",     "baseball-cards-2024-topps::purple foil",       "baseball-cards-2024-topps::blue holofoil"),
    ("topps-2024 aqua/yellow",          "baseball-cards-2024-topps::aqua",              "baseball-cards-2024-topps::yellow"),
    ("topps-2022 orange foilboard/red", "baseball-cards-2022-topps::orange foilboard",  "baseball-cards-2022-topps::red"),
    ("topps-allen-ginter mini back",    "baseball-cards-2024-topps-allen-&-ginter::mini", "baseball-cards-2024-topps-allen-&-ginter::back mini"),
    ("topps-pristine refractor/pristine", "baseball-cards-2023-topps-pristine::blue refractor", "baseball-cards-2023-topps-pristine::blue pristine"),
    ("opc red/blue border 2021",        "hockey-cards-2021-o-pee-chee::red border",     "hockey-cards-2021-o-pee-chee::blue border"),
    ("opc red/blue border 2022",        "hockey-cards-2022-o-pee-chee::red border",     "hockey-cards-2022-o-pee-chee::blue border"),
    ("upper-deck-2023 exclusive/deluxe", "hockey-cards-2023-upper-deck::exclusive",     "hockey-cards-2023-upper-deck::deluxe"),
    ("upper-deck-2023 exclusive/deluxe", "hockey-cards-2023-upper-deck::deluxe",        "hockey-cards-2023-upper-deck::exclusive"),
    ("upper-deck-2023 high-gloss/deluxe", "hockey-cards-2023-upper-deck::high gloss",   "hockey-cards-2023-upper-deck::deluxe"),
    ("upper-deck-2024 high-gloss/excl", "hockey-cards-2024-upper-deck::high gloss",     "hockey-cards-2024-upper-deck::exclusive"),
    ("upper-deck-2022 high-gloss/excl", "hockey-cards-2022-upper-deck::high gloss",     "hockey-cards-2022-upper-deck::exclusives"),
    ("upper-deck-2018 high-gloss/excl", "hockey-cards-2018-upper-deck::high gloss",     "hockey-cards-2018-upper-deck::exclusives"),
    ("upper-deck-2023 outburst/silver", "hockey-cards-2023-upper-deck::outburst",       "hockey-cards-2023-upper-deck::outburst silver"),
    ("prizm-2025 blue/red shimmer",     "football-cards-2025-panini-prizm::blue shimmer", "football-cards-2025-panini-prizm::red shimmer"),
    ("prizm-2025 purple/red shimmer",   "football-cards-2025-panini-prizm::purple shimmer", "football-cards-2025-panini-prizm::red shimmer"),
    ("prizm-2021 ice purple/red",       "football-cards-2021-panini-prizm::purple ice prizm", "football-cards-2021-panini-prizm::red ice prizm"),
    ("prizm-2023 ice blue/green",       "football-cards-2023-panini-prizm::blue ice",   "football-cards-2023-panini-prizm::green ice"),
    ("prizm-2019 wave blue/red",        "football-cards-2019-panini-prizm::blue wave prizm", "football-cards-2019-panini-prizm::red wave prizm"),
    ("select zebra die-cut/base",       "football-cards-2020-panini-select::zebra prizm",   "football-cards-2020-panini-select::zebra prizm die cut"),
    ("select zebra die-cut/base",       "football-cards-2020-panini-select::zebra prizm die cut", "football-cards-2020-panini-select::zebra prizm"),
    ("select dragon scale gold/base 23","football-cards-2023-panini-select::dragon scale gold prizm", "football-cards-2023-panini-select::dragon scale prizm"),
    ("select dragon scale gold/base 24","football-cards-2024-panini-select::gold dragon scale prizm", "football-cards-2024-panini-select::dragon scale prizm"),
    ("mosaic orange/reactive",          "football-cards-2022-panini-mosaic::orange fluorescent", "football-cards-2022-panini-mosaic::reactive orange"),
    ("mosaic orange/reactive",          "football-cards-2022-panini-mosaic::reactive orange",   "football-cards-2022-panini-mosaic::orange fluorescent"),
    ("mosaic green sparkle/mosaic",     "football-cards-2025-panini-mosaic::green sparkle",     "football-cards-2025-panini-mosaic::mosaic green"),
    ("mosaic white/sparkle",            "football-cards-2021-panini-mosaic::white mosaic",      "football-cards-2021-panini-mosaic::white sparkle"),
    ("donruss press proof red/blue",    "football-cards-2022-panini-donruss::press proof red",  "football-cards-2022-panini-donruss::press proof blue"),
]


def _load_run(run_dir: Path) -> dict | None:
    summary_path = run_dir / "summary.json"
    confusions_path = run_dir / "top_confusions.csv"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())

    # Confusion CSV is optional — skip silently if step6 wasn't re-run.
    confusions: dict[tuple[str, str], int] = {}
    if confusions_path.exists():
        with open(confusions_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    count = int(row["count"])
                except (KeyError, ValueError):
                    continue
                key = (row["true_label"], row["pred_label"])
                confusions[key] = count

    return {
        "name":       run_dir.name,
        "dir":        run_dir,
        "summary":    summary,
        "confusions": confusions,
    }


def _discover_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    children = [c for c in root.iterdir() if c.is_dir() and (c / "summary.json").exists()]
    children.sort(key=lambda p: p.stat().st_mtime)
    return children


def _format_flag_summary(flags: dict | None) -> str:
    if not flags:
        return ""
    parts = []
    if flags.get("color_hist") and flags["color_hist"] != "none":
        parts.append(f"color={flags['color_hist']}")
    if flags.get("supcon_lambda") and float(flags["supcon_lambda"]) > 0:
        parts.append(f"supcon={flags['supcon_lambda']}")
    if flags.get("foil_aux_lambda") and float(flags["foil_aux_lambda"]) > 0:
        parts.append(f"foil={flags['foil_aux_lambda']}")
    if flags.get("edge_channel"):
        parts.append("edge")
    return " ".join(parts) if parts else "(baseline)"


def _print_headline(runs: list[dict]) -> None:
    table = Table(title="Headline metrics")
    table.add_column("Run", style="cyan", no_wrap=False)
    table.add_column("Flags", style="dim")
    table.add_column("n_val", justify="right")
    table.add_column("top1", justify="right", style="green")
    table.add_column("Δtop1", justify="right")
    table.add_column("top3", justify="right", style="green")
    table.add_column("Δtop3", justify="right")
    table.add_column("macro F1", justify="right", style="cyan")
    table.add_column("Δf1", justify="right")

    base = runs[0]["summary"]
    for i, r in enumerate(runs):
        s = r["summary"]
        flags_text = _format_flag_summary(s.get("feature_flags"))
        if i == 0:
            d_top1 = d_top3 = d_f1 = ""
        else:
            d_top1 = _delta(s["top1"], base["top1"])
            d_top3 = _delta(s["top3"], base["top3"])
            d_f1   = _delta(s["macro_f1"], base["macro_f1"])
        table.add_row(
            r["name"], flags_text, f"{s.get('n_val', 0):,}",
            f"{s['top1']:.4f}", d_top1,
            f"{s['top3']:.4f}", d_top3,
            f"{s['macro_f1']:.4f}", d_f1,
        )
    console.print(table)


def _delta(now: float, base: float) -> str:
    d = now - base
    if abs(d) < 1e-5:
        return "+0.0000"
    color = "green" if d > 0 else "red"
    return f"[{color}]{d:+.4f}[/{color}]"


def _delta_int(now: int, base: int, *, lower_is_better: bool) -> str:
    d = now - base
    if d == 0:
        return "+0"
    improved = (d < 0) if lower_is_better else (d > 0)
    color = "green" if improved else "red"
    return f"[{color}]{d:+d}[/{color}]"


def _print_known_pairs(runs: list[dict]) -> None:
    table = Table(title="Known hard-pair confusions (lower = better)")
    table.add_column("Pair group", style="cyan")
    table.add_column("True → Pred", style="dim", overflow="fold")
    for r in runs:
        table.add_column(r["name"][:30], justify="right")
        if r is not runs[0]:
            table.add_column("Δ", justify="right")

    # Aggregate by group_name across both directions of the pair.
    grouped: dict[str, list[tuple[str, str]]] = OrderedDict()
    for group, t, p in KNOWN_HARD_PAIRS:
        grouped.setdefault(group, []).append((t, p))

    # Per-group row: sum the pair counts (in either direction).
    for group, pairs in grouped.items():
        # Build a "True → Pred" display string (just the first pair for display
        # — the count column sums all directions).
        display_pair = " / ".join({f"{t.split('::',1)[1]} ↔ {p.split('::',1)[1]}" for t, p in pairs})
        row: list[str] = [group, display_pair]

        base_count = sum(runs[0]["confusions"].get((t, p), 0) for t, p in pairs)
        row.append(str(base_count))
        for r in runs[1:]:
            cnt = sum(r["confusions"].get((t, p), 0) for t, p in pairs)
            row.append(str(cnt))
            row.append(_delta_int(cnt, base_count, lower_is_better=True))
        table.add_row(*row)
    console.print(table)


def _print_top_movers(runs: list[dict], top_n: int) -> None:
    if len(runs) < 2:
        return
    base = runs[0]
    for r in runs[1:]:
        # Union of all confusion keys present in either run.
        keys = set(base["confusions"].keys()) | set(r["confusions"].keys())
        moves = []
        for k in keys:
            b = base["confusions"].get(k, 0)
            n = r["confusions"].get(k, 0)
            if b == 0 and n == 0:
                continue
            moves.append((k, b, n, n - b))
        # Sort by absolute change, descending.
        moves.sort(key=lambda x: abs(x[3]), reverse=True)

        table = Table(title=f"Biggest confusion-pair shifts: baseline → {r['name']}")
        table.add_column("True", style="cyan", overflow="fold")
        table.add_column("Pred", style="cyan", overflow="fold")
        table.add_column("baseline", justify="right")
        table.add_column(r["name"][:30], justify="right")
        table.add_column("Δ", justify="right")
        for (t, p), b, n, d in moves[:top_n]:
            table.add_row(t, p, str(b), str(n), _delta_int(n, b, lower_is_better=True))
        console.print(table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*",
                    help="Paths to run directories (the first is baseline). "
                         "If empty, all subdirs of --root with summary.json are loaded "
                         "in mtime order.")
    ap.add_argument("--root", default="./eval/runs",
                    help="Root directory holding per-run subdirs. Used when no paths are given.")
    ap.add_argument("--top", type=int, default=25,
                    help="Show this many top movers per non-baseline run.")
    ap.add_argument("--no-known-pairs", action="store_true",
                    help="Skip the hand-curated known-hard-pair tracker.")
    ap.add_argument("--no-movers", action="store_true",
                    help="Skip the per-pair top-movers tables.")
    args = ap.parse_args()

    if args.paths:
        run_dirs = [Path(p) for p in args.paths]
    else:
        run_dirs = _discover_runs(Path(args.root))
        if not run_dirs:
            console.print(f"[red]No runs found under {args.root} (looking for subdirs with summary.json)[/red]")
            sys.exit(1)
        console.print(f"[dim]Discovered {len(run_dirs)} run(s) under {args.root}:[/dim]")
        for p in run_dirs:
            console.print(f"  [dim]- {p.name}[/dim]")

    runs = []
    for d in run_dirs:
        r = _load_run(d)
        if r is None:
            console.print(f"[yellow]Skipping {d} (no summary.json)[/yellow]")
            continue
        runs.append(r)

    if not runs:
        console.print("[red]No usable runs.[/red]")
        sys.exit(1)

    console.print()
    _print_headline(runs)
    if not args.no_known_pairs:
        console.print()
        _print_known_pairs(runs)
    if not args.no_movers:
        console.print()
        _print_top_movers(runs, args.top)


if __name__ == "__main__":
    main()
