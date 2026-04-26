#!/usr/bin/env python3
"""
Step 6 (Part B): Evaluate the variant classifier.

Loads the checkpoint produced by 05_train_variant_classifier.py, runs it over
the held-out val manifest, and writes three reports:

    per_class_metrics.csv   (label, support, precision, recall, f1)
    top_confusions.csv      (true_label, pred_label, count)
    summary.json            (top1, top3, macro_f1, etc.)

When the checkpoint was trained with --hierarchical, the evaluator applies the
same per-sample set mask so reported metrics reflect the masked softmax the
model was actually trained to produce (and that the RunPod handler applies at
inference via ``candidate_labels``).

Run from repo root:
    python training/06_eval_variant_classifier.py
    python training/06_eval_variant_classifier.py --checkpoint ./checkpoints/variant_classifier_linear.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Repo root importable + also expose training/ for the shared lib import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console

from embeddings_dinov2 import PREPROCESS_SPEC, build_preprocess
from variant_classifier_lib import (
    ImageDataset,
    collate_image_skip_none,
    setup_backbone,
    extract_features,
    build_head,
    load_jsonl,
    derive_set_labels,
    build_set_class_mask,
    evaluate_head,
    write_eval_outputs,
)

console = Console()


def evaluate(args):
    import torch
    from torch.utils.data import DataLoader

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        console.print(f"[red]Checkpoint not found: {ckpt_path}[/red]")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    labels = ckpt["label_map"]
    arch   = ckpt["arch"]
    input_dim = ckpt["input_dim"]
    stored_spec = ckpt.get("preprocess", {})
    if stored_spec and stored_spec != PREPROCESS_SPEC:
        console.print(f"[red]Preprocess spec mismatch![/red]\n  ckpt: {stored_spec}\n  now:  {PREPROCESS_SPEC}")
        sys.exit(1)

    val_path = Path(args.data_dir) / "variant_manifest_val.jsonl"
    if not val_path.exists():
        console.print(f"[red]Val manifest missing: {val_path}[/red]")
        sys.exit(1)
    val_records = load_jsonl(str(val_path))

    # Resolve set labels: prefer the checkpoint's copy (trained ordering), fall
    # back to the manifest, then derive from records as a last resort.
    set_labels = ckpt.get("set_labels")
    if not set_labels:
        lm_path = Path(args.data_dir) / "label_map.json"
        if lm_path.exists():
            set_labels = json.loads(lm_path.read_text()).get("set_labels")
    if not set_labels:
        set_labels, _ = derive_set_labels(val_records)
    set_to_idx = {s: i for i, s in enumerate(set_labels)}

    # Backfill set_idx for older manifests that didn't carry it.
    for rec in val_records:
        if "set_idx" not in rec:
            rec["set_idx"] = set_to_idx.get((rec.get("set_slug") or "").strip().lower(), 0)

    console.print(f"  Val records: [cyan]{len(val_records):,}[/cyan]   labels: [cyan]{len(labels)}[/cyan]   sets: [cyan]{len(set_labels)}[/cyan]")

    hierarchical = bool(ckpt.get("hierarchical", False)) or args.force_hierarchical
    class_set_mask = None
    if hierarchical:
        class_set_mask = build_set_class_mask(labels, set_labels).to(device)
        console.print(f"[cyan]Hierarchical eval:[/cyan] applying per-sample set mask")

    transform = build_preprocess()
    ds = ImageDataset(
        val_records, transform, set_to_idx,
        color_hist_kind=ckpt.get("color_hist_kind", "none"),
        color_hist_bins=int(ckpt.get("color_hist_bins", 32)),
        edge_channel=bool(ckpt.get("edge_channel", False)),
    )
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=collate_image_skip_none,
        persistent_workers=args.workers > 0,
    )

    backbone = setup_backbone(device, ckpt.get("finetuned_backbone"), console=console)
    # Rebuild the head with the same hyperparams it was trained with, so the
    # state_dict shapes match for MLPs with non-default hidden_dim.
    head = build_head(
        arch, input_dim, len(labels),
        hidden_dim=ckpt.get("hidden_dim", 256),
        dropout=ckpt.get("dropout", 0.1),
    ).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    result = evaluate_head(
        head=head,
        val_loader=loader,
        labels=labels,
        class_set_mask=class_set_mask,
        device=device,
        backbone=backbone,
        use_cache=False,
        top_confusions=args.top_confusions,
    )

    summary_extra = {
        "checkpoint":     str(ckpt_path),
        "n_labels":       len(labels),
        "n_sets":         len(set_labels),
        "arch":           arch,
        "hierarchical":   hierarchical,
        "trained_at":     ckpt.get("trained_at"),
        "backbone":       ckpt.get("backbone"),
        "finetuned_sha":  ckpt.get("finetuned_backbone_sha"),
        "sampler":        ckpt.get("sampler"),
        "scheduler":      ckpt.get("scheduler"),
        "run_name":       ckpt.get("run_name"),
        "feature_flags":  ckpt.get("feature_flags"),
    }
    per_class_path, confusions_path, summary_path = write_eval_outputs(
        args.output_dir, result, summary_extra=summary_extra
    )

    console.print(f"\n[bold]Eval summary[/bold]")
    console.print(f"  samples:  [cyan]{result['n_val']:,}[/cyan]")
    console.print(f"  top-1:    [green]{result['top1']:.4f}[/green]")
    console.print(f"  top-3:    [green]{result['top3']:.4f}[/green]")
    console.print(f"  macro F1: [cyan]{result['macro_f1']:.4f}[/cyan]")
    console.print(f"\nWrote:")
    console.print(f"  [green]{per_class_path}[/green]")
    console.print(f"  [green]{confusions_path}[/green]")
    console.print(f"  [green]{summary_path}[/green]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="./checkpoints/variant_classifier_linear.pt")
    ap.add_argument("--data-dir", default="./training_data/variant_classifier")
    ap.add_argument("--output-dir", default="./eval/variant_classifier")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--top-confusions", type=int, default=50)
    ap.add_argument("--force-hierarchical", action="store_true",
                    help="Apply per-sample set mask at eval time even if the checkpoint "
                         "wasn't trained hierarchically. Simulates the inference-time "
                         "candidate_labels shortlist.")
    args = ap.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
