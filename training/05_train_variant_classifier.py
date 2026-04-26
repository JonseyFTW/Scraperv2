#!/usr/bin/env python3
"""
Step 5 (Part B): Train the foil-pattern variant classifier.

Head-on-frozen-DINOv2 architecture. The backbone (DINOv2 ViT-L/14-reg, optionally
with the fine-tuned weights from Step 2) computes 1024-dim features; a small
classification head learns to map those features to variant classes.

Architectures:
    V1 — linear probe   (--arch linear)   Linear(1024, C)
    V2 — small MLP      (--arch mlp)      Linear(1024,256) -> GELU -> Dropout -> Linear(256,C)

Key training optimizations (all opt-in via flags, all on by default where safe):
    - Feature caching (--cache / --no-cache): precompute DINOv2 features once
      to disk so training is just a Linear head on cached tensors. ~300x
      per-epoch speedup.
    - Set-stratified batching (--sampler set_stratified, default): each batch
      draws K sets and N variants per set — puts the fine-grained foil-vs-base
      gradient signal where it belongs. Alternative: class_balanced / random.
    - Warm-restart cosine scheduler (--scheduler cosine_warm_restarts).
    - Hierarchical masked loss (--hierarchical): training loss is restricted
      per-sample to the classes belonging to that sample's set. At inference,
      the client supplies the per-set candidate shortlist (the existing
      ``candidate_labels`` RunPod handler param).

The produced checkpoint embeds the preprocess spec and backbone identifier so
the RunPod handler can refuse mismatched loads.

Run from repo root:
    python training/05_train_variant_classifier.py
    python training/05_train_variant_classifier.py --arch mlp --epochs 30 --batch 4096
    python training/05_train_variant_classifier.py --hierarchical --sampler set_stratified \\
        --scheduler cosine_warm_restarts --finetuned-backbone ./checkpoints/dinov2_finetuned_backbone.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also expose the training/ directory so ``variant_classifier_lib`` imports work
# regardless of whether training/ is treated as a package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from embeddings_dinov2 import PREPROCESS_SPEC, build_preprocess

# Shared library (same directory).
from variant_classifier_lib import (  # noqa: E402
    RARE_CLASS,
    sha256_of_file,
    load_jsonl,
    derive_set_labels,
    ImageDataset,
    collate_image_skip_none,
    CachedFeatureDataset,
    collate_features,
    cache_features_if_needed,
    load_cached_meta,
    SetStratifiedBatchSampler,
    build_class_balanced_sampler,
    build_head,
    setup_backbone,
    extract_features,
    build_set_class_mask,
    masked_cross_entropy,
    evaluate_head,
    write_eval_outputs,
    color_hist_dim,
    COLOR_HIST_CHOICES,
    build_projection_head,
    supcon_within_set_loss,
    head_forward_with_trunk,
    FOIL_TONES,
    build_class_to_foil_tone,
)

console = Console()


# ---------------------------------------------------------------------------
# Feature-flag stamping
# ---------------------------------------------------------------------------

def _feature_flags_from_args(args) -> dict:
    """Snapshot of which experimental recs are enabled for this run.

    Stamped into the checkpoint and into runs/<run_name>/summary.json so the
    07_compare_runs.py harness can label the diff columns with the active rec.
    """
    return {
        "color_hist":      getattr(args, "color_hist",      "none"),
        "supcon_lambda":   float(getattr(args, "supcon_lambda",   0.0)),
        "foil_aux_lambda": float(getattr(args, "foil_aux_lambda", 0.0)),
        "edge_channel":    bool(getattr(args, "edge_channel",    False)),
    }


def _default_run_name(args) -> str:
    """Build a stable, descriptive run name from the active feature flags.

    Format:   <arch>[__color=lab32][__supcon=0.1][__foil=0.5][__edge]__YYYYMMDDHHMMSS
    """
    parts = [args.arch]
    if getattr(args, "color_hist", "none") not in (None, "none"):
        parts.append(f"color={args.color_hist}")
    if float(getattr(args, "supcon_lambda", 0.0)) > 0:
        parts.append(f"supcon={args.supcon_lambda}")
    if float(getattr(args, "foil_aux_lambda", 0.0)) > 0:
        parts.append(f"foil={args.foil_aux_lambda}")
    if getattr(args, "edge_channel", False):
        parts.append("edge")
    parts.append(datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    return "__".join(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_manifests(data_dir: Path):
    label_map_path = data_dir / "label_map.json"
    train_path = data_dir / "variant_manifest_train.jsonl"
    val_path   = data_dir / "variant_manifest_val.jsonl"
    for p in (label_map_path, train_path, val_path):
        if not p.exists():
            console.print(f"[red]Missing {p} — run 04_export_variant_training_data.py first[/red]")
            sys.exit(1)

    label_map = json.loads(label_map_path.read_text())
    labels = label_map["labels"]
    train_records = load_jsonl(str(train_path))
    val_records   = load_jsonl(str(val_path))

    # set_labels was added by the refactor — older manifests didn't emit it,
    # so derive from the records themselves in that case.
    set_labels = label_map.get("set_labels")
    if not set_labels:
        combined = train_records + val_records
        set_labels, _ = derive_set_labels(combined)
    set_to_idx = {s: i for i, s in enumerate(set_labels)}

    # Backfill set_idx onto manifest records that pre-date the refactor, so
    # everything downstream can assume it exists.
    for rec in train_records:
        if "set_idx" not in rec:
            rec["set_idx"] = set_to_idx.get((rec.get("set_slug") or "").strip().lower(), 0)
    for rec in val_records:
        if "set_idx" not in rec:
            rec["set_idx"] = set_to_idx.get((rec.get("set_slug") or "").strip().lower(), 0)

    return labels, set_labels, set_to_idx, train_records, val_records


# ---------------------------------------------------------------------------
# Loader construction
# ---------------------------------------------------------------------------

def _build_loaders_cached(
    cache_dir: Path,
    train_features: Path,
    val_features: Path,
    args,
    train_color: Path | None = None,
    val_color: Path | None = None,
):
    """Construct DataLoaders over the fp16 feature cache."""
    import torch
    from torch.utils.data import DataLoader, RandomSampler

    train_labels, train_set_idxs = load_cached_meta(cache_dir / "train_meta.npz")
    val_labels,   val_set_idxs   = load_cached_meta(cache_dir / "val_meta.npz")

    train_ds = CachedFeatureDataset(
        str(train_features), train_labels, train_set_idxs,
        color_path=str(train_color) if train_color else None,
    )
    val_ds   = CachedFeatureDataset(
        str(val_features),   val_labels,   val_set_idxs,
        color_path=str(val_color) if val_color else None,
    )

    train_loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_features,
        persistent_workers=args.workers > 0,
    )

    if args.sampler == "set_stratified":
        # Interpret --batch as the effective batch size; decompose into k*n.
        k = max(1, args.sets_per_batch)
        n = max(1, args.batch // k)
        batch_sampler = SetStratifiedBatchSampler(
            train_set_idxs,
            k_sets_per_batch=k,
            n_per_set=n,
            num_batches=max(1, len(train_ds) // (k * n)),
            seed=args.seed,
        )
        console.print(
            f"[cyan]Sampler:[/cyan] set_stratified  "
            f"(k_sets={k}, n_per_set={n}, batch={batch_sampler.batch_size}, "
            f"batches/epoch={len(batch_sampler):,})"
        )
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **train_loader_kwargs)
    elif args.sampler == "class_balanced":
        sampler = build_class_balanced_sampler(train_labels)
        console.print(f"[cyan]Sampler:[/cyan] class_balanced (WeightedRandomSampler)")
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, sampler=sampler, drop_last=False, **train_loader_kwargs
        )
    else:
        console.print(f"[cyan]Sampler:[/cyan] random")
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True, drop_last=False, **train_loader_kwargs
        )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=collate_features, persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader


def _build_loaders_image(train_records, val_records, set_to_idx, args):
    """Fallback path: load images on the fly (slow, ~one DINOv2 pass per epoch)."""
    from torch.utils.data import DataLoader

    transform = build_preprocess()
    train_ds = ImageDataset(
        train_records, transform, set_to_idx,
        color_hist_kind=args.color_hist, color_hist_bins=args.color_hist_bins,
    )
    val_ds   = ImageDataset(
        val_records,   transform, set_to_idx,
        color_hist_kind=args.color_hist, color_hist_bins=args.color_hist_bins,
    )

    if args.sampler == "set_stratified":
        set_arr = np.array([r["set_idx"] for r in train_records], dtype=np.int64)
        k = max(1, args.sets_per_batch)
        n = max(1, args.batch // k)
        batch_sampler = SetStratifiedBatchSampler(
            set_arr, k_sets_per_batch=k, n_per_set=n,
            num_batches=max(1, len(train_ds) // (k * n)), seed=args.seed,
        )
        console.print(
            f"[cyan]Sampler:[/cyan] set_stratified "
            f"(k_sets={k}, n_per_set={n}, batch={batch_sampler.batch_size})"
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=batch_sampler,
            num_workers=args.workers, pin_memory=True,
            collate_fn=collate_image_skip_none,
            persistent_workers=args.workers > 0,
        )
    elif args.sampler == "class_balanced":
        label_arr = np.array([r["label_idx"] for r in train_records], dtype=np.int64)
        sampler = build_class_balanced_sampler(label_arr)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, sampler=sampler,
            num_workers=args.workers, pin_memory=True,
            collate_fn=collate_image_skip_none,
            persistent_workers=args.workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True,
            num_workers=args.workers, pin_memory=True,
            collate_fn=collate_image_skip_none, drop_last=False,
            persistent_workers=args.workers > 0,
        )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=collate_image_skip_none, persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_scheduler(optim, args):
    import torch
    if args.scheduler == "cosine_warm_restarts":
        console.print(
            f"[cyan]Scheduler:[/cyan] CosineAnnealingWarmRestarts "
            f"(T_0={args.scheduler_t0}, T_mult={args.scheduler_tmult})"
        )
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=args.scheduler_t0, T_mult=args.scheduler_tmult
        )
    console.print(f"[cyan]Scheduler:[/cyan] CosineAnnealingLR (T_max={args.epochs})")
    return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)


def train(args):
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[green]GPU: {gpu} ({vram:.1f}GB VRAM)[/green]")
    else:
        console.print("[yellow]WARNING: CPU training will be very slow[/yellow]")

    data_dir = Path(args.data_dir)
    labels, set_labels, set_to_idx, train_records, val_records = _load_manifests(data_dir)
    num_classes = len(labels)
    num_sets = len(set_labels)
    console.print(f"  Labels: [cyan]{num_classes}[/cyan]   Sets: [cyan]{num_sets}[/cyan]")
    console.print(f"  Train: [cyan]{len(train_records):,}[/cyan]   Val: [cyan]{len(val_records):,}[/cyan]")

    # ── Feature cache (Tier 1.1) ─────────────────────────────────────────
    use_cache = args.cache and device == "cuda"
    if args.cache and device != "cuda":
        console.print("[yellow]Feature caching requires CUDA — falling back to image dataset[/yellow]")

    # ── Color-hist side channel sizing (rec 2) ───────────────────────────
    hist_dim = color_hist_dim(args.color_hist, args.color_hist_bins)
    if hist_dim:
        console.print(
            f"[cyan]Color-hist side channel:[/cyan] kind={args.color_hist} "
            f"bins={args.color_hist_bins} dim={hist_dim}"
        )

    backbone = None
    if use_cache:
        _, train_feat, val_feat, train_color, val_color = cache_features_if_needed(
            data_dir=data_dir,
            records_train=train_records,
            records_val=val_records,
            set_to_idx=set_to_idx,
            finetuned_backbone=args.finetuned_backbone,
            device=device,
            batch_size=args.cache_batch,
            num_workers=args.workers,
            console=console,
            color_hist_kind=args.color_hist,
            color_hist_bins=args.color_hist_bins,
        )
        cache_dir = train_feat.parent
        train_loader, val_loader = _build_loaders_cached(
            cache_dir, train_feat, val_feat, args,
            train_color=train_color, val_color=val_color,
        )
    else:
        backbone = setup_backbone(device, args.finetuned_backbone, console=console)
        train_loader, val_loader = _build_loaders_image(
            train_records, val_records, set_to_idx, args
        )

    # ── Hierarchical mask (Tier 3.7) ─────────────────────────────────────
    class_set_mask = None
    if args.hierarchical:
        class_set_mask = build_set_class_mask(labels, set_labels)
        console.print(
            f"[cyan]Hierarchical mode:[/cyan] per-sample masked softmax "
            f"(mask shape={tuple(class_set_mask.shape)})"
        )

    # ── Head + optimizer + scheduler ─────────────────────────────────────
    head_input_dim = 1024 + hist_dim
    head = build_head(
        args.arch, head_input_dim, num_classes,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)

    # Rec 3: SupCon projection. Hung off the head's trunk output (the pre-
    # classifier hidden representation for --arch mlp; the input features for
    # --arch linear, where it does nothing useful since features are frozen).
    projection = None
    if args.supcon_lambda > 0:
        if args.arch == "linear":
            console.print(
                "[yellow]--supcon-lambda > 0 with --arch linear has no effect on the "
                "classifier (no shared trunk above the frozen DINOv2 features). "
                "Disabling SupCon. Use --arch mlp to enable.[/yellow]"
            )
            args.supcon_lambda = 0.0
        else:
            trunk_dim = args.hidden_dim
            projection = build_projection_head(
                input_dim=trunk_dim,
                proj_dim=args.supcon_proj_dim,
            ).to(device)
            console.print(
                f"[cyan]SupCon (rec 3):[/cyan] lambda={args.supcon_lambda} "
                f"temp={args.supcon_temperature} proj_dim={args.supcon_proj_dim} "
                f"trunk_dim={trunk_dim}"
            )

    # Rec 4: foil-tone auxiliary head. Trains the trunk to encode foil tone
    # explicitly (gold/orange/blue/red/holo/shimmer/...). Same trunk-sharing
    # constraint as SupCon — only useful with --arch mlp.
    foil_aux = None
    class_to_foil_tone_t = None
    if args.foil_aux_lambda > 0:
        if args.arch == "linear":
            console.print(
                "[yellow]--foil-aux-lambda > 0 with --arch linear has no effect on the "
                "classifier (no shared trunk above frozen DINOv2 features). "
                "Disabling foil aux. Use --arch mlp to enable.[/yellow]"
            )
            args.foil_aux_lambda = 0.0
        else:
            cls_tone_np = build_class_to_foil_tone(labels)
            class_to_foil_tone_t = torch.from_numpy(cls_tone_np).to(device)
            num_tones = len(FOIL_TONES)
            foil_aux = nn.Linear(args.hidden_dim, num_tones).to(device)
            tone_counts = np.bincount(cls_tone_np, minlength=num_tones)
            console.print(
                f"[cyan]Foil aux head (rec 4):[/cyan] lambda={args.foil_aux_lambda} "
                f"num_tones={num_tones}  none-bucket={tone_counts[0]}/{len(labels)} classes"
            )

    params = list(head.parameters())
    if projection is not None:
        params += list(projection.parameters())
    if foil_aux is not None:
        params += list(foil_aux.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = _build_scheduler(optim, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    console.print(
        f"[cyan]Regularization:[/cyan] weight_decay={args.weight_decay:.0e}  "
        f"label_smoothing={args.label_smoothing}  dropout={args.dropout}  "
        f"feat_dropout={args.feat_dropout}  hidden_dim={args.hidden_dim}"
    )

    best_top1 = -1.0
    best_top3 = -1.0
    epochs_without_improve = 0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path      = out_dir / f"variant_classifier_{args.arch}.pt"
    last_ckpt_path = out_dir / f"variant_classifier_{args.arch}_last.pt"

    start_epoch = 1
    if args.resume:
        resume_path = args.resume if args.resume != "auto" else str(last_ckpt_path)
        if os.path.exists(resume_path):
            console.print(f"[cyan]Resuming from {resume_path}[/cyan]")
            state = torch.load(resume_path, map_location=device, weights_only=False)
            head.load_state_dict(state["head_state_dict"])
            if projection is not None and state.get("projection_state_dict"):
                try:
                    projection.load_state_dict(state["projection_state_dict"])
                except Exception as e:
                    console.print(f"[yellow]Projection resume skipped ({e})[/yellow]")
            if foil_aux is not None and state.get("foil_aux_state_dict"):
                try:
                    foil_aux.load_state_dict(state["foil_aux_state_dict"])
                except Exception as e:
                    console.print(f"[yellow]Foil aux resume skipped ({e})[/yellow]")
            if "optimizer_state_dict" in state:
                optim.load_state_dict(state["optimizer_state_dict"])
            if "scheduler_state_dict" in state:
                try:
                    sched.load_state_dict(state["scheduler_state_dict"])
                except Exception as e:
                    console.print(f"[yellow]Scheduler resume skipped ({e})[/yellow]")
            start_epoch = int(state.get("epoch", 0)) + 1
            best_top1 = float(state.get("best_top1", -1.0))
            best_top3 = float(state.get("best_top3", -1.0))
            console.print(
                f"  Resumed at epoch {start_epoch}/{args.epochs}  "
                f"best val@1 so far: {best_top1:.3f}"
            )
            if start_epoch > args.epochs:
                console.print("[yellow]All epochs already completed — nothing to do.[/yellow]")
                return
        else:
            console.print(
                f"[yellow]--resume specified but {resume_path} not found; starting fresh.[/yellow]"
            )

    def _forward(batch, training: bool, want_trunk: bool = False):
        """Unpack a batch and return (logits, labels, set_idxs[, trunk]).

        ``feat_dropout`` applies stochastic feature dropout before the head —
        cheap feature-space regularization when image augmentation isn't an
        option (cached features are a single fixed view per image).

        When ``want_trunk`` is True, the head's pre-classifier representation
        is also returned so auxiliary losses (SupCon) can attach to it.
        """
        import torch.nn.functional as F
        if use_cache:
            # Cache row already includes the color hist (if enabled), since
            # CachedFeatureDataset concatenates at __getitem__.
            feats, labels_t, set_idxs_t = batch
            feats = feats.to(device, non_blocking=True)
        else:
            imgs, color, labels_t, set_idxs_t = batch
            imgs = imgs.to(device, non_blocking=True)
            feats = extract_features(backbone, imgs, device)
            if color is not None:
                feats = torch.cat([feats, color.to(device, non_blocking=True)], dim=-1)
        if training and args.feat_dropout > 0.0:
            feats = F.dropout(feats, p=args.feat_dropout, training=True)
        labels_t   = labels_t.to(device, non_blocking=True)
        set_idxs_t = set_idxs_t.to(device, non_blocking=True)
        if want_trunk:
            logits, trunk = head_forward_with_trunk(head, feats)
            return logits, labels_t, set_idxs_t, trunk
        return head(feats), labels_t, set_idxs_t

    for epoch in range(start_epoch, args.epochs + 1):
        head.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        n_correct = 0

        with Progress(
            TextColumn(f"Epoch {epoch}/{args.epochs}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("train", total=len(train_loader), loss=0.0)
            running_supcon_loss = 0.0
            running_foil_loss = 0.0
            for batch in train_loader:
                if batch is None:
                    progress.advance(task); continue

                want_trunk = (args.supcon_lambda > 0) or (args.foil_aux_lambda > 0)
                if want_trunk:
                    logits, labels_t, set_idxs_t, trunk = _forward(
                        batch, training=True, want_trunk=True
                    )
                else:
                    logits, labels_t, set_idxs_t = _forward(batch, training=True)
                    trunk = None

                if args.hierarchical:
                    loss = masked_cross_entropy(
                        logits, labels_t, set_idxs_t, class_set_mask,
                        label_smoothing=args.label_smoothing,
                    )
                    # For top-1 accuracy we also mask at training time so the
                    # reported metric matches the loss we're optimizing.
                    pred = logits.masked_fill(
                        ~class_set_mask.to(logits.device)[set_idxs_t], float("-inf")
                    ).argmax(-1)
                else:
                    loss = criterion(logits, labels_t)
                    pred = logits.argmax(-1)

                # Rec 3: same-set SupCon. Anchors = batch elements; positives =
                # same class; candidates restricted to same-set siblings.
                if projection is not None and trunk is not None:
                    import torch.nn.functional as F_
                    z = projection(trunk)
                    z = F_.normalize(z, dim=-1)
                    supcon = supcon_within_set_loss(
                        z, labels_t, set_idxs_t, temperature=args.supcon_temperature
                    )
                    loss = loss + args.supcon_lambda * supcon
                    running_supcon_loss += float(supcon.detach()) * labels_t.size(0)

                # Rec 4: foil-tone aux CE. Forces the trunk to encode foil
                # tone explicitly so gold/orange/blue/red/holo/shimmer don't
                # collapse into the same neighborhood.
                if foil_aux is not None and trunk is not None:
                    import torch.nn.functional as F_
                    tone_targets = class_to_foil_tone_t[labels_t]
                    tone_logits = foil_aux(trunk)
                    aux_loss = F_.cross_entropy(tone_logits, tone_targets)
                    loss = loss + args.foil_aux_lambda * aux_loss
                    running_foil_loss += float(aux_loss.detach()) * labels_t.size(0)

                optim.zero_grad()
                loss.backward()
                optim.step()

                bs = labels_t.size(0)
                running_loss += loss.item() * bs
                n_seen += bs
                n_correct += (pred == labels_t).sum().item()
                progress.update(task, advance=1, loss=running_loss / max(n_seen, 1))

        # Step once per epoch. Warm-restart variant also accepts a float step,
        # but per-epoch is the common convention and matches the legacy path.
        sched.step()
        train_loss = running_loss / max(n_seen, 1)
        train_top1 = n_correct / max(n_seen, 1)
        train_supcon = running_supcon_loss / max(n_seen, 1) if projection is not None else None
        train_foil   = running_foil_loss   / max(n_seen, 1) if foil_aux   is not None else None

        # ── Eval ─────────────────────────────────────────────────────────
        head.eval()
        v_seen = 0
        v_top1 = 0
        v_top3 = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                logits, labels_t, set_idxs_t = _forward(batch, training=False)
                if args.hierarchical:
                    logits = logits.masked_fill(
                        ~class_set_mask.to(logits.device)[set_idxs_t], float("-inf")
                    )
                k = min(3, num_classes)
                topk = logits.topk(k, dim=-1).indices
                v_seen += labels_t.size(0)
                v_top1 += (topk[:, 0] == labels_t).sum().item()
                v_top3 += (topk == labels_t.unsqueeze(-1)).any(-1).sum().item()

        val_top1 = v_top1 / max(v_seen, 1)
        val_top3 = v_top3 / max(v_seen, 1)
        dt = time.time() - t0
        supcon_msg = f"  supcon {train_supcon:.4f}" if train_supcon is not None else ""
        foil_msg   = f"  foil_ce {train_foil:.4f}"   if train_foil   is not None else ""
        console.print(
            f"  epoch {epoch:>2d}  loss {train_loss:.4f}{supcon_msg}{foil_msg}  "
            f"train@1 {train_top1:.3f}  val@1 {val_top1:.3f}  val@3 {val_top3:.3f}  ({dt:.0f}s)"
        )

        # ── Checkpointing ────────────────────────────────────────────────
        base_payload = {
            "head_state_dict":        head.state_dict(),
            "label_map":              labels,
            "set_labels":             set_labels,
            "backbone":               "dinov2_vitl14_reg",
            "finetuned_backbone":     args.finetuned_backbone,
            "finetuned_backbone_sha": sha256_of_file(args.finetuned_backbone),
            "arch":                   args.arch,
            "hidden_dim":             args.hidden_dim,
            "dropout":                args.dropout,
            "feat_dropout":           args.feat_dropout,
            "weight_decay":           args.weight_decay,
            "label_smoothing":        args.label_smoothing,
            "input_dim":              head_input_dim,
            "color_hist_kind":        args.color_hist,
            "color_hist_bins":        int(args.color_hist_bins),
            "color_hist_dim":         hist_dim,
            "supcon_lambda":          float(args.supcon_lambda),
            "supcon_temperature":     float(args.supcon_temperature),
            "supcon_proj_dim":        int(args.supcon_proj_dim),
            "projection_state_dict":  (projection.state_dict() if projection is not None else None),
            "foil_aux_lambda":        float(args.foil_aux_lambda),
            "foil_tones":             list(FOIL_TONES) if foil_aux is not None else None,
            "foil_aux_state_dict":    (foil_aux.state_dict() if foil_aux is not None else None),
            "preprocess":             PREPROCESS_SPEC,
            "trained_at":             datetime.now(timezone.utc).isoformat(),
            "train_samples":          len(train_records),
            "val_samples":            len(val_records),
            "hierarchical":           args.hierarchical,
            "sampler":                args.sampler,
            "scheduler":              args.scheduler,
            "epoch":                  epoch,
            "run_name":               args.run_name,
            "feature_flags":          _feature_flags_from_args(args),
        }

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_top3 = val_top3
            epochs_without_improve = 0
            torch.save(
                {**base_payload, "val_top1": best_top1, "val_top3": best_top3},
                ckpt_path,
            )
            console.print(f"    [green]saved best -> {ckpt_path} (top1={best_top1:.3f})[/green]")
        else:
            epochs_without_improve += 1

        torch.save(
            {
                **base_payload,
                "optimizer_state_dict": optim.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "val_top1":             val_top1,
                "val_top3":             val_top3,
                "best_top1":            best_top1,
                "best_top3":            best_top3,
            },
            last_ckpt_path,
        )

        # Early stopping: bail once val@1 has plateaued for --patience epochs.
        # Safe for long runs so they don't waste compute past convergence.
        if args.patience > 0 and epochs_without_improve >= args.patience:
            console.print(
                f"[yellow]Early stopping at epoch {epoch}: "
                f"no val@1 improvement for {args.patience} epochs.[/yellow]"
            )
            break

    console.print(f"\n[bold green]Done.[/bold green] best val@1={best_top1:.3f}  val@3={best_top3:.3f}")
    console.print(f"Checkpoint: [cyan]{ckpt_path}[/cyan]")

    # ── Auto-eval (feature comparison harness) ───────────────────────────
    # Re-load best head and run the full eval on the cached val loader so
    # every training run drops a comparable summary.json + top_confusions.csv
    # under runs/<run_name>/. This is what 07_compare_runs.py reads to diff
    # rec-by-rec impact (top1/top3/macro_f1, plus confusion-pair deltas).
    if args.auto_eval:
        run_name = args.run_name
        run_dir = Path(args.eval_output_dir) / run_name
        console.print(f"\n[bold]Auto-eval[/bold] -> [cyan]{run_dir}[/cyan]")

        # Reload the BEST checkpoint (val_top1 leader) for the report so the
        # numbers match what gets shipped, not the last epoch.
        best_state = torch.load(ckpt_path, map_location=device, weights_only=False)
        head.load_state_dict(best_state["head_state_dict"])

        result = evaluate_head(
            head=head,
            val_loader=val_loader,
            labels=labels,
            class_set_mask=class_set_mask,
            device=device,
            backbone=backbone,
            use_cache=use_cache,
            top_confusions=args.eval_top_confusions,
        )

        summary_extra = {
            "checkpoint":     str(ckpt_path),
            "n_labels":       num_classes,
            "n_sets":         num_sets,
            "arch":           args.arch,
            "hierarchical":   args.hierarchical,
            "trained_at":     base_payload["trained_at"],
            "backbone":       "dinov2_vitl14_reg",
            "finetuned_sha":  sha256_of_file(args.finetuned_backbone),
            "sampler":        args.sampler,
            "scheduler":      args.scheduler,
            "run_name":       run_name,
            "feature_flags":  _feature_flags_from_args(args),
            "epochs_trained": epoch,
        }
        per_class_path, confusions_path, summary_path = write_eval_outputs(
            run_dir, result, summary_extra=summary_extra
        )

        console.print(
            f"  samples={result['n_val']:,}  "
            f"top1={result['top1']:.4f}  top3={result['top3']:.4f}  "
            f"macro_f1={result['macro_f1']:.4f}"
        )
        console.print(f"  Wrote [green]{summary_path}[/green]")
        console.print(
            f"  Diff vs other runs: "
            f"[cyan]python training/07_compare_runs.py {args.eval_output_dir}[/cyan]"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./training_data/variant_classifier")
    ap.add_argument("--output-dir", default="./checkpoints")
    ap.add_argument("--arch", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--finetuned-backbone", default=None,
                    help="Optional path to a fine-tuned DINOv2 backbone .pt from step 2")

    ap.add_argument("--epochs", type=int, default=30,
                    help="With feature caching, 30-50 epochs is trivial.")
    ap.add_argument("--batch", type=int, default=4096,
                    help="Head-training batch size. With cached features the head "
                         "is just a Linear, so 4096-8192 fits comfortably on a 4070.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    # Regularization
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="AdamW weight decay. Bump to 5e-4 or 1e-3 when the head "
                         "is overfitting (train@1 >> val@1).")
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="Cross-entropy label smoothing. In hierarchical mode the "
                         "smoothing is correctly distributed over the per-sample "
                         "allowed classes only.")
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout inside the MLP head. Ignored for --arch linear.")
    ap.add_argument("--hidden-dim", type=int, default=256,
                    help="Hidden-layer width of the MLP head. Ignored for --arch linear.")
    ap.add_argument("--feat-dropout", type=float, default=0.0,
                    help="Stochastic dropout on cached DINOv2 features before the head. "
                         "Cheap regularizer for the fixed-features regime where image "
                         "augmentation isn't available.")
    ap.add_argument("--patience", type=int, default=0,
                    help="Early-stop after N epochs without val@1 improvement. "
                         "0 (default) disables early stopping.")

    # Feature cache
    ap.add_argument("--cache", dest="cache", action="store_true", default=True,
                    help="Precompute DINOv2 features once and train the head on "
                         "cached fp16 tensors (default).")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    help="Disable feature cache; run DINOv2 every step (slow).")
    ap.add_argument("--cache-batch", type=int, default=64,
                    help="DINOv2 batch size used while building the cache.")

    # Sampler
    ap.add_argument("--sampler", choices=["random", "set_stratified", "class_balanced"],
                    default="set_stratified",
                    help="Batch composition. 'set_stratified' (default) draws K sets "
                         "and N variants per set to focus gradients on same-set discrimination.")
    ap.add_argument("--sets-per-batch", type=int, default=8,
                    help="K in the set-stratified sampler. N is inferred from --batch.")

    # Scheduler
    ap.add_argument("--scheduler", choices=["cosine", "cosine_warm_restarts"],
                    default="cosine_warm_restarts",
                    help="LR schedule. Warm restarts prevent the LR from collapsing "
                         "to zero once extended training makes it useful.")
    ap.add_argument("--scheduler-t0", type=int, default=5,
                    help="Initial cycle length (epochs) for cosine_warm_restarts.")
    ap.add_argument("--scheduler-tmult", type=int, default=2,
                    help="Cycle multiplier for cosine_warm_restarts.")

    # Hierarchical
    ap.add_argument("--hierarchical", action="store_true",
                    help="Train variant head with per-sample masked softmax — loss "
                         "restricted to the classes belonging to that sample's set. "
                         "Matches the inference flow: search picks set, head ranks variants.")

    # Rec 2: color-histogram side channel
    ap.add_argument("--color-hist", choices=list(COLOR_HIST_CHOICES), default="none",
                    help="Concatenate a 3*bins-dim color histogram onto each DINOv2 "
                         "feature so the head sees explicit color signal. DINOv2 is "
                         "color-invariant by design, which is fatal for parallels "
                         "where color is the only label-distinguishing feature "
                         "(Topps Now purple/blue, OPC red/blue border, mosaic "
                         "finishes, etc.). Recommended: lab32. Cache is keyed on "
                         "(kind, bins) so changing this rebuilds automatically.")
    ap.add_argument("--color-hist-bins", type=int, default=32,
                    help="Per-channel bin count for --color-hist. Total dim = 3*bins. "
                         "32 -> 96-dim side channel.")

    # Rec 3: pairwise contrastive (SupCon) within-set
    ap.add_argument("--supcon-lambda", type=float, default=0.0,
                    help="Weight on the supervised-contrastive auxiliary loss "
                         "(rec 3). Loss is restricted to same-set candidates so the "
                         "gradient pulls sibling parallels apart in the projection "
                         "space. 0 disables. Try 0.1–0.5 with --arch mlp; useless "
                         "with --arch linear (no shared trunk).")
    ap.add_argument("--supcon-temperature", type=float, default=0.1,
                    help="SupCon temperature τ. Lower = harder.")
    ap.add_argument("--supcon-proj-dim", type=int, default=128,
                    help="Output dim of the SupCon projection head.")

    # Rec 4: foil-tone auxiliary head
    ap.add_argument("--foil-aux-lambda", type=float, default=0.0,
                    help="Weight on the foil-tone auxiliary CE loss (rec 4). The "
                         "tone for each class is parsed from its variant label "
                         "(gold/orange/blue/red/holo/shimmer/...). Forces the trunk "
                         "to encode foil tone explicitly so it doesn't get lost to "
                         "one-hot collapse. 0 disables. Try 0.1–0.3 with --arch mlp.")

    ap.add_argument("--resume", default=None, nargs="?", const="auto",
                    help="Resume from checkpoint. Bare --resume uses variant_classifier_{arch}_last.pt; "
                         "pass an explicit path to use a different file.")

    # Run identification + auto-eval harness
    ap.add_argument("--run-name", default=None,
                    help="Name for this training run; outputs land at "
                         "<eval-output-dir>/<run-name>/. Defaults to a flag-derived "
                         "name like 'mlp__color=lab32__supcon=0.1__YYYYMMDDHHMMSS'.")
    ap.add_argument("--auto-eval", dest="auto_eval", action="store_true", default=True,
                    help="After training, run a full val-set eval and write "
                         "summary.json + top_confusions.csv + per_class_metrics.csv "
                         "for 07_compare_runs.py to diff against other runs (default).")
    ap.add_argument("--no-auto-eval", dest="auto_eval", action="store_false",
                    help="Skip the post-training eval pass.")
    ap.add_argument("--eval-output-dir", default="./eval/runs",
                    help="Parent directory for per-run eval outputs.")
    ap.add_argument("--eval-top-confusions", type=int, default=200,
                    help="How many (true,pred) confusion pairs to write. Bumped "
                         "from step-6's 50 so the comparison harness has long enough "
                         "tails to track regression on rare hard pairs.")

    args = ap.parse_args()
    if not args.run_name:
        args.run_name = _default_run_name(args)

    train(args)


if __name__ == "__main__":
    main()
