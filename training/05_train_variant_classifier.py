#!/usr/bin/env python3
"""
Step 5 (Part B): Train the foil-pattern variant classifier.

Head-on-frozen-DINOv2 architecture. The backbone (DINOv2 ViT-L/14-reg, optionally
with the fine-tuned weights from Step 2) computes 1024-dim features; a small
classification head learns to map those features to variant classes.

V1 — linear probe       (--arch linear)   Linear(1024, C)
V2 — small MLP          (--arch mlp)      Linear(1024,256) -> GELU -> Dropout -> Linear(256,C)

The produced checkpoint embeds the preprocess spec and backbone identifier so
the RunPod handler can refuse mismatched loads.

Run from repo root so config/embeddings_dinov2 import:
    python training/05_train_variant_classifier.py
    python training/05_train_variant_classifier.py --arch mlp --epochs 20 --batch 128
    python training/05_train_variant_classifier.py --finetuned-backbone ./checkpoints/dinov2_finetuned_backbone.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from embeddings_dinov2 import PREPROCESS_SPEC, build_preprocess

console = Console()


def sha256_of_file(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class VariantDataset:
    """PyTorch Dataset returning (image_tensor, label_idx)."""

    def __init__(self, records: list[dict], transform):
        from PIL import Image
        self._Image = Image
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = self._Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            return None
        return self.transform(img), int(rec["label_idx"])


def collate_skip_none(batch):
    import torch
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return imgs, labels


def build_head(arch: str, input_dim: int, num_classes: int):
    import torch.nn as nn
    if arch == "linear":
        return nn.Linear(input_dim, num_classes)
    if arch == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
    raise ValueError(f"Unknown arch: {arch}")


def setup_backbone(device, finetuned_path: str | None):
    import torch
    console.print("[cyan]Loading DINOv2-ViT-L/14-reg (frozen)...[/cyan]")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    if finetuned_path:
        console.print(f"[cyan]Loading fine-tuned backbone from {finetuned_path}...[/cyan]")
        state_dict = torch.load(finetuned_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    return model


def extract_features(backbone, imgs, device):
    import torch
    if device == "cuda":
        imgs = imgs.half()
    with torch.no_grad():
        feats = backbone(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float()


def train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[green]GPU: {gpu} ({vram:.1f}GB VRAM)[/green]")
    else:
        console.print("[yellow]WARNING: CPU training will be very slow[/yellow]")

    data_dir = Path(args.data_dir)
    label_map_path = data_dir / "label_map.json"
    train_path = data_dir / "variant_manifest_train.jsonl"
    val_path   = data_dir / "variant_manifest_val.jsonl"
    for p in (label_map_path, train_path, val_path):
        if not p.exists():
            console.print(f"[red]Missing {p} — run 04_export_variant_training_data.py first[/red]")
            sys.exit(1)

    label_map = json.loads(label_map_path.read_text())
    labels = label_map["labels"]
    num_classes = len(labels)
    console.print(f"  Labels: [cyan]{num_classes}[/cyan]")

    train_records = load_jsonl(str(train_path))
    val_records   = load_jsonl(str(val_path))
    console.print(f"  Train: [cyan]{len(train_records):,}[/cyan]   Val: [cyan]{len(val_records):,}[/cyan]")

    transform = build_preprocess()
    train_ds = VariantDataset(train_records, transform)
    val_ds   = VariantDataset(val_records, transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        collate_fn=collate_skip_none, drop_last=False, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=collate_skip_none, persistent_workers=args.workers > 0,
    )

    backbone = setup_backbone(device, args.finetuned_backbone)
    head = build_head(args.arch, 1024, num_classes).to(device)

    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_top1 = -1.0
    best_top3 = -1.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"variant_classifier_{args.arch}.pt"

    for epoch in range(1, args.epochs + 1):
        head.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        n_correct = 0

        with Progress(TextColumn(f"Epoch {epoch}/{args.epochs}"), BarColumn(),
                      TextColumn("{task.completed}/{task.total}"),
                      TextColumn("loss={task.fields[loss]:.4f}"),
                      TimeRemainingColumn()) as progress:
            task = progress.add_task("train", total=len(train_loader), loss=0.0)
            for batch in train_loader:
                if batch is None:
                    progress.advance(task); continue
                imgs, labels_t = batch
                imgs = imgs.to(device, non_blocking=True)
                labels_t = labels_t.to(device, non_blocking=True)

                feats = extract_features(backbone, imgs, device)
                logits = head(feats)
                loss = criterion(logits, labels_t)

                optim.zero_grad()
                loss.backward()
                optim.step()

                running_loss += loss.item() * imgs.size(0)
                n_seen += imgs.size(0)
                n_correct += (logits.argmax(-1) == labels_t).sum().item()
                progress.update(task, advance=1, loss=running_loss / max(n_seen, 1))

        sched.step()
        train_loss = running_loss / max(n_seen, 1)
        train_top1 = n_correct / max(n_seen, 1)

        # Eval
        head.eval()
        v_seen = 0
        v_top1 = 0
        v_top3 = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                imgs, labels_t = batch
                imgs = imgs.to(device, non_blocking=True)
                labels_t = labels_t.to(device, non_blocking=True)
                feats = extract_features(backbone, imgs, device)
                logits = head(feats)
                topk = logits.topk(min(3, num_classes), dim=-1).indices
                v_seen += imgs.size(0)
                v_top1 += (topk[:, 0] == labels_t).sum().item()
                v_top3 += (topk == labels_t.unsqueeze(-1)).any(-1).sum().item()

        val_top1 = v_top1 / max(v_seen, 1)
        val_top3 = v_top3 / max(v_seen, 1)
        dt = time.time() - t0
        console.print(
            f"  epoch {epoch:>2d}  loss {train_loss:.4f}  "
            f"train@1 {train_top1:.3f}  val@1 {val_top1:.3f}  val@3 {val_top3:.3f}  ({dt:.0f}s)"
        )

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_top3 = val_top3
            torch.save({
                "head_state_dict":        head.state_dict(),
                "label_map":              labels,
                "backbone":               "dinov2_vitl14_reg",
                "finetuned_backbone":     args.finetuned_backbone,
                "finetuned_backbone_sha": sha256_of_file(args.finetuned_backbone),
                "arch":                   args.arch,
                "input_dim":              1024,
                "preprocess":             PREPROCESS_SPEC,
                "trained_at":             datetime.now(timezone.utc).isoformat(),
                "train_samples":          len(train_records),
                "val_samples":            len(val_records),
                "val_top1":               best_top1,
                "val_top3":               best_top3,
                "epoch":                  epoch,
            }, ckpt_path)
            console.print(f"    [green]saved best -> {ckpt_path} (top1={best_top1:.3f})[/green]")

    console.print(f"\n[bold green]Done.[/bold green] best val@1={best_top1:.3f}  val@3={best_top3:.3f}")
    console.print(f"Checkpoint: [cyan]{ckpt_path}[/cyan]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./training_data/variant_classifier")
    ap.add_argument("--output-dir", default="./checkpoints")
    ap.add_argument("--arch", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--finetuned-backbone", default=None,
                    help="Optional path to a fine-tuned DINOv2 backbone .pt from step 2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    train(args)


if __name__ == "__main__":
    main()
