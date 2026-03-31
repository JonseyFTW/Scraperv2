#!/usr/bin/env python3
"""
Step 2: Fine-Tune DINOv2-ViT-L/14-reg with Contrastive Learning

Trains the last 4 transformer blocks using triplet loss with synthetic
augmentations (simulating phone scans) and hard negatives (same character,
different card).

Run from the Scraperv2 directory:
    python training/02_finetune_dinov2.py
    python training/02_finetune_dinov2.py --epochs 15 --batch-size 16
    python training/02_finetune_dinov2.py --manifest ./training_data/manifest.json

VRAM usage on 4070 Ti Super (16GB):
    batch_size=16 → ~6-7GB VRAM
    batch_size=24 → ~9-10GB VRAM
    batch_size=32 → ~12-13GB VRAM
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
)

console = Console()


# ---------------------------------------------------------------------------
# Synthetic augmentations — simulate phone camera conditions
# ---------------------------------------------------------------------------

class ScanAugmentation:
    """Simulate realistic phone-camera scans of trading cards."""

    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: Image.Image, augment: bool = True) -> torch.Tensor:
        if not augment:
            return self.base_transform(img)

        # Random perspective warp (card held at angle)
        if random.random() < 0.7:
            img = transforms.functional.perspective(
                img,
                startpoints=[[0, 0], [img.width, 0], [img.width, img.height], [0, img.height]],
                endpoints=[
                    [random.randint(0, int(img.width * 0.08)), random.randint(0, int(img.height * 0.08))],
                    [img.width - random.randint(0, int(img.width * 0.08)), random.randint(0, int(img.height * 0.08))],
                    [img.width - random.randint(0, int(img.width * 0.08)), img.height - random.randint(0, int(img.height * 0.08))],
                    [random.randint(0, int(img.width * 0.08)), img.height - random.randint(0, int(img.height * 0.08))],
                ],
                interpolation=transforms.InterpolationMode.BICUBIC,
            )

        # Slight rotation (-5 to +5 degrees)
        if random.random() < 0.6:
            angle = random.uniform(-5, 5)
            img = transforms.functional.rotate(img, angle, fill=0)

        # Random crop (92-98% of image)
        if random.random() < 0.5:
            crop_ratio = random.uniform(0.92, 0.98)
            crop_w = int(img.width * crop_ratio)
            crop_h = int(img.height * crop_ratio)
            left = random.randint(0, img.width - crop_w)
            top = random.randint(0, img.height - crop_h)
            img = img.crop((left, top, left + crop_w, top + crop_h))

        # Brightness/contrast variation (different lighting)
        if random.random() < 0.7:
            brightness = random.uniform(0.8, 1.3)
            img = ImageEnhance.Brightness(img).enhance(brightness)
        if random.random() < 0.7:
            contrast = random.uniform(0.8, 1.3)
            img = ImageEnhance.Contrast(img).enhance(contrast)

        # Color temperature shift
        if random.random() < 0.4:
            saturation = random.uniform(0.8, 1.2)
            img = ImageEnhance.Color(img).enhance(saturation)

        # Simulate glare hotspot (common on holo/foil cards)
        if random.random() < 0.3:
            img = self._add_glare(img)

        # Camera blur (slight motion or focus issues)
        if random.random() < 0.4:
            radius = random.uniform(0.3, 1.2)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Background bleed — card on dark/light/wood/fabric surface
        if random.random() < 0.35:
            img = self._add_background_bleed(img)

        # JPEG compression artifacts
        if random.random() < 0.3:
            import io
            buffer = io.BytesIO()
            quality = random.randint(60, 85)
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert("RGB")

        return self.base_transform(img)

    def _add_glare(self, img: Image.Image) -> Image.Image:
        """Add a circular glare hotspot."""
        import numpy as np
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Random glare center
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        radius = random.randint(int(min(h, w) * 0.1), int(min(h, w) * 0.25))

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = np.clip(1.0 - dist / radius, 0, 1) ** 2

        intensity = random.uniform(40, 120)
        arr += mask[:, :, np.newaxis] * intensity
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _add_background_bleed(self, img: Image.Image) -> Image.Image:
        """Place the card on a colored/textured background with visible border."""
        import numpy as np

        w, h = img.size
        # Pad 3-8% on each side
        pad_pct = random.uniform(0.03, 0.08)
        pad_x = int(w * pad_pct)
        pad_y = int(h * pad_pct)
        new_w = w + 2 * pad_x
        new_h = h + 2 * pad_y

        # Pick a background color
        bg_type = random.choice(["dark", "light", "wood", "fabric", "solid"])
        if bg_type == "dark":
            bg_color = (random.randint(10, 50), random.randint(10, 50), random.randint(10, 50))
        elif bg_type == "light":
            bg_color = (random.randint(200, 245), random.randint(200, 245), random.randint(200, 245))
        elif bg_type == "wood":
            # Warm brown tones
            r = random.randint(120, 180)
            bg_color = (r, int(r * 0.7), int(r * 0.4))
        elif bg_type == "fabric":
            # Muted blue/gray/green
            base = random.randint(80, 150)
            shift = random.choice([(0, 0, 30), (0, 20, 0), (0, 0, 0)])
            bg_color = (base + shift[0], base + shift[1], base + shift[2])
        else:
            bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Create background and paste card onto it
        bg = Image.new("RGB", (new_w, new_h), bg_color)

        # Add slight noise to background for realism
        bg_arr = np.array(bg, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(3, 12), bg_arr.shape)
        bg_arr = np.clip(bg_arr + noise, 0, 255).astype(np.uint8)
        bg = Image.fromarray(bg_arr)

        bg.paste(img, (pad_x, pad_y))

        # Resize back to original dimensions
        bg = bg.resize((w, h), Image.LANCZOS)
        return bg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TripletCardDataset(Dataset):
    """Generates (anchor, positive, negative) triplets for contrastive learning.

    Anchor:   clean reference image
    Positive: augmented version of same card (synthetic scan)
    Negative: hard negative (same character, different card) or random card
    """

    def __init__(self, manifest: list[dict], hard_negatives: dict[str, list[str]],
                 augmentor: ScanAugmentation):
        self.cards = manifest
        self.hard_negatives = hard_negatives
        self.augmentor = augmentor

        # Build lookup: id -> card, character_name -> [ids]
        self.id_to_card = {c["id"]: c for c in self.cards}
        self.id_to_idx = {c["id"]: i for i, c in enumerate(self.cards)}

        # Reverse map: card_id -> character group
        self.id_to_char = {}
        for char_name, ids in self.hard_negatives.items():
            for card_id in ids:
                self.id_to_char[card_id] = char_name

        # All valid card ids
        self.valid_ids = [c["id"] for c in self.cards]

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, idx):
        anchor_card = self.cards[idx]
        anchor_id = anchor_card["id"]

        # Load anchor image
        try:
            img = Image.open(anchor_card["image_path"]).convert("RGB")
        except Exception:
            # Return a dummy on error; will be filtered by collate
            return None

        # Anchor: clean (no augmentation)
        anchor = self.augmentor(img, augment=False)

        # Positive: augmented version of same image (synthetic scan)
        positive = self.augmentor(img, augment=True)

        # Negative: hard negative (70% chance) or random (30% chance)
        neg_id = None
        char_name = self.id_to_char.get(anchor_id)

        if char_name and random.random() < 0.7:
            # Hard negative: same character, different card
            group = self.hard_negatives[char_name]
            candidates = [cid for cid in group if cid != anchor_id]
            if candidates:
                neg_id = random.choice(candidates)

        if neg_id is None:
            # Random negative
            neg_id = random.choice(self.valid_ids)
            while neg_id == anchor_id:
                neg_id = random.choice(self.valid_ids)

        neg_card = self.id_to_card[neg_id]
        try:
            neg_img = Image.open(neg_card["image_path"]).convert("RGB")
            negative = self.augmentor(neg_img, augment=False)
        except Exception:
            return None

        return anchor, positive, negative


def triplet_collate(batch):
    """Filter None entries from failed image loads."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    anchors = torch.stack([b[0] for b in batch])
    positives = torch.stack([b[1] for b in batch])
    negatives = torch.stack([b[2] for b in batch])
    return anchors, positives, negatives


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """Small MLP projection head used only during training."""

    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def setup_model(device: str):
    """Load DINOv2-ViT-L/14-reg and freeze all but last 4 blocks."""
    console.print("[cyan]Loading DINOv2-ViT-L/14-reg (with registers)...[/cyan]")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    model = model.to(device)
    model.train()

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 4 transformer blocks + norm layer
    num_blocks = len(model.blocks)
    unfreeze_from = num_blocks - 4
    unfrozen = 0
    for i in range(unfreeze_from, num_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = True
            unfrozen += 1

    # Unfreeze final norm
    if hasattr(model, 'norm'):
        for param in model.norm.parameters():
            param.requires_grad = True
            unfrozen += 1

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    console.print(f"[green]Model loaded — {total_params / 1e6:.0f}M total params[/green]")
    console.print(f"[green]  Trainable: {trainable_params / 1e6:.1f}M ({trainable_params / total_params * 100:.1f}%) — last 4/{num_blocks} blocks[/green]")

    # Projection head (training only, discarded at export)
    proj_head = ProjectionHead(input_dim=1024).to(device)

    return model, proj_head


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(manifest_path: str, hard_negatives_path: str, output_dir: str,
          epochs: int, batch_size: int, lr: float, margin: float):
    """Fine-tune DINOv2 with triplet loss."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[green]GPU: {gpu_name} ({vram:.1f}GB VRAM)[/green]")
    else:
        console.print("[yellow]WARNING: Training on CPU will be very slow![/yellow]")

    # Load data
    console.print(f"\n[bold]Loading training data...[/bold]")
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(hard_negatives_path) as f:
        hard_negatives = json.load(f)

    console.print(f"  Cards: [cyan]{len(manifest):,}[/cyan]")
    console.print(f"  Hard negative groups: [cyan]{len(hard_negatives):,}[/cyan]")

    # Setup
    augmentor = ScanAugmentation()
    dataset = TripletCardDataset(manifest, hard_negatives, augmentor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=triplet_collate,
        drop_last=True,
        persistent_workers=True,
    )

    model, proj_head = setup_model(device)

    # Optimizer — only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params += list(proj_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # Cosine annealing scheduler
    total_steps = epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    # Triplet loss
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    os.makedirs(output_dir, exist_ok=True)

    console.print(f"\n[bold]Training config:[/bold]")
    console.print(f"  Epochs:     [cyan]{epochs}[/cyan]")
    console.print(f"  Batch size: [cyan]{batch_size}[/cyan]")
    console.print(f"  LR:         [cyan]{lr}[/cyan]")
    console.print(f"  Margin:     [cyan]{margin}[/cyan]")
    console.print(f"  Steps/epoch: [cyan]{len(dataloader)}[/cyan]")
    console.print(f"  Total steps: [cyan]{total_steps}[/cyan]")
    console.print(f"  Output:     [cyan]{output_dir}[/cyan]\n")

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        proj_head.train()
        epoch_loss = 0.0
        epoch_steps = 0

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Epoch {epoch}/{epochs}", total=len(dataloader))

            for batch in dataloader:
                if batch is None:
                    progress.advance(task)
                    continue

                anchors, positives, negatives = [b.to(device) for b in batch]

                # Forward pass through backbone
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    a_emb = model(anchors)
                    p_emb = model(positives)
                    n_emb = model(negatives)

                    # Project for loss computation
                    a_proj = proj_head(a_emb)
                    p_proj = proj_head(p_emb)
                    n_proj = proj_head(n_emb)

                    loss = triplet_loss_fn(a_proj, p_proj, n_proj)

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                epoch_steps += 1

                progress.update(task, advance=1,
                                description=f"Epoch {epoch}/{epochs} — loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        console.print(f"  Epoch {epoch}: avg_loss={avg_loss:.4f}, lr={current_lr:.2e}, elapsed={elapsed:.0f}s")

        # Save checkpoint every 5 epochs and on improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(output_dir, "dinov2_finetuned_best.pt")
            torch.save(model.state_dict(), checkpoint_path)
            console.print(f"  [green]New best! Saved → {checkpoint_path}[/green]")

        if epoch % 5 == 0 or epoch == epochs:
            checkpoint_path = os.path.join(output_dir, f"dinov2_finetuned_epoch{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)

    # Save final backbone (this is what you deploy)
    final_path = os.path.join(output_dir, "dinov2_finetuned_backbone.pt")
    torch.save(model.state_dict(), final_path)

    total_time = time.time() - start_time
    console.print(f"\n[green]Training complete![/green]")
    console.print(f"  Total time:  [cyan]{total_time / 60:.1f} minutes[/cyan]")
    console.print(f"  Best loss:   [cyan]{best_loss:.4f}[/cyan]")
    console.print(f"  Final model: [green]{final_path}[/green]")
    console.print(f"\n[green]Next: python training/03_reembed_collection.py --checkpoint {final_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 for card matching")
    parser.add_argument("--manifest", type=str, default="./training_data/manifest.json",
                        help="Path to manifest.json from Step 1")
    parser.add_argument("--hard-negatives", type=str, default="./training_data/hard_negatives.json",
                        help="Path to hard_negatives.json from Step 1")
    parser.add_argument("--output", type=str, default="./checkpoints",
                        help="Output directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Triplet batch size — 16 safe on 16GB VRAM (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--margin", type=float, default=0.3,
                        help="Triplet loss margin (default: 0.3)")

    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        console.print(f"[red]Manifest not found: {args.manifest}[/red]")
        console.print("[yellow]Run Step 1 first: python training/01_export_training_data.py[/yellow]")
        sys.exit(1)
    if not os.path.exists(args.hard_negatives):
        console.print(f"[red]Hard negatives not found: {args.hard_negatives}[/red]")
        sys.exit(1)

    train(args.manifest, args.hard_negatives, args.output,
          args.epochs, args.batch_size, args.lr, args.margin)


if __name__ == "__main__":
    main()
