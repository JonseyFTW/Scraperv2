#!/usr/bin/env python3
"""
Merge multiple training data exports into a single manifest + hard negatives.

Combines Pokemon, Football, Baseball, etc. into one dataset for training
a single unified model.

Usage:
    python training/merge_training_data.py \
        --inputs ./training_data_pokemon ./training_data_football ./training_data_baseball \
        --output ./training_data_combined

Each input directory must contain manifest.json and hard_negatives.json
from Step 1 (01_export_training_data.py).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

from rich.console import Console

console = Console()


def merge(input_dirs: list[str], output_dir: str):
    combined_manifest = []
    combined_hard_negatives = defaultdict(list)
    seen_ids = set()

    for input_dir in input_dirs:
        manifest_path = os.path.join(input_dir, "manifest.json")
        hn_path = os.path.join(input_dir, "hard_negatives.json")

        if not os.path.exists(manifest_path):
            console.print(f"[red]Missing {manifest_path}[/red]")
            sys.exit(1)
        if not os.path.exists(hn_path):
            console.print(f"[red]Missing {hn_path}[/red]")
            sys.exit(1)

        with open(manifest_path) as f:
            manifest = json.load(f)
        with open(hn_path) as f:
            hard_negatives = json.load(f)

        # Deduplicate by card id
        added = 0
        dupes = 0
        for card in manifest:
            if card["id"] not in seen_ids:
                seen_ids.add(card["id"])
                combined_manifest.append(card)
                added += 1
            else:
                dupes += 1

        # Merge hard negative groups
        for char_name, ids in hard_negatives.items():
            existing = set(combined_hard_negatives[char_name])
            for card_id in ids:
                if card_id not in existing:
                    combined_hard_negatives[char_name].append(card_id)
                    existing.add(card_id)

        console.print(f"  [cyan]{input_dir}[/cyan]: {added:,} cards added, {dupes:,} duplicates skipped")

    # Remove groups with < 2 cards
    combined_hard_negatives = {
        name: ids for name, ids in combined_hard_negatives.items()
        if len(ids) >= 2
    }

    total_in_groups = sum(len(ids) for ids in combined_hard_negatives.values())

    console.print(f"\n[bold]Combined dataset:[/bold]")
    console.print(f"  Total cards:         [green]{len(combined_manifest):,}[/green]")
    console.print(f"  Hard negative groups: [cyan]{len(combined_hard_negatives):,}[/cyan]")
    console.print(f"  Cards in groups:     [cyan]{total_in_groups:,}[/cyan]")
    console.print(f"  Cards without group: [dim]{len(combined_manifest) - total_in_groups:,}[/dim]")

    # Top groups
    sorted_groups = sorted(combined_hard_negatives.items(), key=lambda x: len(x[1]), reverse=True)
    console.print(f"\n  Top 10 hard negative groups:")
    for name, ids in sorted_groups[:10]:
        console.print(f"    {name}: [cyan]{len(ids)}[/cyan] cards")

    # Write output
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(combined_manifest, f, indent=2)
    console.print(f"\n  Wrote [green]{manifest_path}[/green]")

    hn_path = os.path.join(output_dir, "hard_negatives.json")
    with open(hn_path, "w") as f:
        json.dump(combined_hard_negatives, f, indent=2)
    console.print(f"  Wrote [green]{hn_path}[/green]")

    console.print(f"\n[green]Done! Train with:[/green]")
    console.print(f"  python training/02_finetune_dinov2.py --manifest {manifest_path} --hard-negatives {hn_path} --epochs 10 --batch-size 16")


def main():
    parser = argparse.ArgumentParser(description="Merge training data from multiple exports")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input directories from Step 1 exports")
    parser.add_argument("--output", type=str, default="./training_data_combined",
                        help="Output directory for merged data")

    args = parser.parse_args()
    console.print(f"\n[bold]Merging {len(args.inputs)} training data exports[/bold]\n")
    merge(args.inputs, args.output)


if __name__ == "__main__":
    main()
