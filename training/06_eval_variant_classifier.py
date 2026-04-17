#!/usr/bin/env python3
"""
Step 6 (Part B): Evaluate the variant classifier.

Loads the checkpoint produced by 05_train_variant_classifier.py, runs it over
the held-out val manifest, and writes three reports:

    per_class_metrics.csv   (label, support, precision, recall, f1)
    top_confusions.csv      (true_label, pred_label, count)
    summary.json            (top1, top3, macro_f1, etc.)

Run from repo root:
    python training/06_eval_variant_classifier.py
    python training/06_eval_variant_classifier.py --checkpoint ./checkpoints/variant_classifier_linear.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console

from embeddings_dinov2 import PREPROCESS_SPEC, build_preprocess

# 05 is prefixed with a digit so it can't be imported as a normal module —
# use importlib to reach into it.
import importlib.util as _iu
_SPEC = _iu.spec_from_file_location(
    "train_variant",
    os.path.join(os.path.dirname(__file__), "05_train_variant_classifier.py"),
)
_TRAIN_MOD = _iu.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAIN_MOD)
VariantDataset   = _TRAIN_MOD.VariantDataset
collate_skip_none = _TRAIN_MOD.collate_skip_none
setup_backbone   = _TRAIN_MOD.setup_backbone
extract_features = _TRAIN_MOD.extract_features
build_head       = _TRAIN_MOD.build_head
load_jsonl       = _TRAIN_MOD.load_jsonl

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
    console.print(f"  Val records: [cyan]{len(val_records):,}[/cyan]   labels: [cyan]{len(labels)}[/cyan]")

    transform = build_preprocess()
    ds = VariantDataset(val_records, transform)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=collate_skip_none,
                        persistent_workers=args.workers > 0)

    backbone = setup_backbone(device, ckpt.get("finetuned_backbone"))
    head = build_head(arch, input_dim, len(labels)).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    tp = Counter()
    fp = Counter()
    fn = Counter()
    confusions: Counter = Counter()  # (true_idx, pred_idx) -> count
    total = 0
    top1_correct = 0
    top3_correct = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs, gold = batch
            imgs = imgs.to(device, non_blocking=True)
            gold = gold.to(device, non_blocking=True)
            feats = extract_features(backbone, imgs, device)
            logits = head(feats)
            k = min(3, len(labels))
            topk = logits.topk(k, dim=-1).indices
            pred = topk[:, 0]

            total += gold.size(0)
            top1_correct += (pred == gold).sum().item()
            top3_correct += (topk == gold.unsqueeze(-1)).any(-1).sum().item()

            for g, p in zip(gold.cpu().tolist(), pred.cpu().tolist()):
                if g == p:
                    tp[g] += 1
                else:
                    fp[p] += 1
                    fn[g] += 1
                    confusions[(g, p)] += 1

    top1 = top1_correct / max(total, 1)
    top3 = top3_correct / max(total, 1)

    # Per-class metrics
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_class_path = out_dir / "per_class_metrics.csv"
    f1s = []
    with open(per_class_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "support", "precision", "recall", "f1"])
        for idx, label in enumerate(labels):
            support = tp[idx] + fn[idx]
            prec = tp[idx] / (tp[idx] + fp[idx]) if (tp[idx] + fp[idx]) else 0.0
            rec  = tp[idx] / support if support else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            if support:
                f1s.append(f1)
            w.writerow([label, support, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

    macro_f1 = sum(f1s) / max(len(f1s), 1)

    confusions_path = out_dir / "top_confusions.csv"
    with open(confusions_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_label", "pred_label", "count"])
        for (g, p), c in confusions.most_common(args.top_confusions):
            w.writerow([labels[g], labels[p], c])

    summary_path = out_dir / "summary.json"
    summary = {
        "checkpoint":     str(ckpt_path),
        "n_val":          total,
        "n_labels":       len(labels),
        "top1":           top1,
        "top3":           top3,
        "macro_f1":       macro_f1,
        "arch":           arch,
        "trained_at":     ckpt.get("trained_at"),
        "backbone":       ckpt.get("backbone"),
        "finetuned_sha":  ckpt.get("finetuned_backbone_sha"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    console.print(f"\n[bold]Eval summary[/bold]")
    console.print(f"  samples:  [cyan]{total:,}[/cyan]")
    console.print(f"  top-1:    [green]{top1:.4f}[/green]")
    console.print(f"  top-3:    [green]{top3:.4f}[/green]")
    console.print(f"  macro F1: [cyan]{macro_f1:.4f}[/cyan]")
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
    args = ap.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
