#!/usr/bin/env python3
"""
Variant-classifier shared library.

This module is imported by ``05_train_variant_classifier.py`` and
``06_eval_variant_classifier.py``. Keeping the heavy-lifting here (instead of
in files whose names start with a digit) avoids the importlib hack that 06
used to need, and gives us a single source of truth for:

    - dataset wrappers (image-on-the-fly and cached-feature)
    - feature cache builder (one-shot precompute of DINOv2 features)
    - batch samplers (random / set-stratified / class-balanced)
    - head architectures (linear / mlp)
    - backbone loader + per-batch feature extractor
    - hierarchical mask builder (per-set allowed classes)

The preprocess spec and backbone identifier are mirrored into every checkpoint
so the RunPod handler can refuse mismatched loads.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

# Repo root on path so we can import ``embeddings_dinov2`` when callers run us
# from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from embeddings_dinov2 import PREPROCESS_SPEC, build_preprocess  # noqa: E402


RARE_CLASS = "__rare__"


# ---------------------------------------------------------------------------
# File + manifest helpers
# ---------------------------------------------------------------------------

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


def derive_set_labels(records: list[dict]) -> tuple[list[str], dict[str, int]]:
    """Build the canonical sorted set_label list from manifest records.

    Used when the manifest was produced by an older version of step 4 that
    didn't stash ``set_idx`` / ``set_labels``.
    """
    sets = sorted({(r.get("set_slug") or "").strip().lower() for r in records})
    return sets, {s: i for i, s in enumerate(sets)}


# ---------------------------------------------------------------------------
# Image-on-the-fly dataset (legacy path — used when --no-cache or when
# backbone/preprocess changes and the cache is being rebuilt).
# ---------------------------------------------------------------------------

class ImageDataset:
    """Returns (image_tensor, label_idx, set_idx) tuples."""

    def __init__(self, records: list[dict], transform, set_to_idx: dict[str, int]):
        self.records = records
        self.transform = transform
        self.set_to_idx = set_to_idx

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image
        rec = self.records[idx]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            return None
        set_slug = (rec.get("set_slug") or "").strip().lower()
        set_idx = self.set_to_idx.get(set_slug, 0)
        return self.transform(img), int(rec["label_idx"]), int(set_idx)


def collate_image_skip_none(batch):
    import torch
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs     = torch.stack([b[0] for b in batch])
    labels   = torch.tensor([b[1] for b in batch], dtype=torch.long)
    set_idxs = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, labels, set_idxs


# ---------------------------------------------------------------------------
# Cached-feature dataset (Tier 1.1: feature caching)
# ---------------------------------------------------------------------------

class CachedFeatureDataset:
    """Memory-mapped reader over a precomputed DINOv2 feature tensor.

    The backbone is frozen and the preprocess is deterministic, so there is no
    reason to re-run DINOv2 every epoch. We compute features once to disk and
    the "model" becomes a tiny Linear head on cached 1024-dim vectors — which
    turns each training epoch from hours into seconds.
    """

    def __init__(self, features_path: str, labels: np.ndarray, set_idxs: np.ndarray):
        # mmap_mode="r" keeps the 4-8GB tensor out of RAM — the OS pages it in
        # as the sampler touches random indices.
        self.features = np.load(features_path, mmap_mode="r")
        assert self.features.shape[0] == len(labels) == len(set_idxs), (
            f"cache length mismatch: features={self.features.shape[0]} "
            f"labels={len(labels)} set_idxs={len(set_idxs)}"
        )
        self.labels = labels
        self.set_idxs = set_idxs

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Copy the row — the memmap is fp16 on disk; cast happens in collate.
        feat = np.array(self.features[idx], copy=True)
        return feat, int(self.labels[idx]), int(self.set_idxs[idx])


def collate_features(batch):
    import torch
    if not batch:
        return None
    feats = np.stack([b[0] for b in batch], axis=0)
    labels = np.array([b[1] for b in batch], dtype=np.int64)
    set_idxs = np.array([b[2] for b in batch], dtype=np.int64)
    # fp16 on disk -> fp32 on GPU for stable head training.
    return (
        torch.from_numpy(feats).float(),
        torch.from_numpy(labels),
        torch.from_numpy(set_idxs),
    )


# ---------------------------------------------------------------------------
# Feature cache builder
# ---------------------------------------------------------------------------

def _cache_key(finetuned_backbone: str | None) -> str:
    if not finetuned_backbone:
        return "base"
    sha = sha256_of_file(finetuned_backbone)
    return (sha or "unknown")[:16]


def _expected_cache_info(
    records_train: list[dict],
    records_val: list[dict],
    finetuned_backbone: str | None,
) -> dict:
    return {
        "preprocess_spec":     PREPROCESS_SPEC,
        "finetuned_backbone":  finetuned_backbone,
        "finetuned_sha":       sha256_of_file(finetuned_backbone),
        "n_train":             len(records_train),
        "n_val":               len(records_val),
    }


def cache_features_if_needed(
    data_dir: Path,
    records_train: list[dict],
    records_val: list[dict],
    set_to_idx: dict[str, int],
    finetuned_backbone: str | None,
    device: str,
    batch_size: int = 64,
    num_workers: int = 4,
    console=None,
) -> tuple[Path, Path, Path]:
    """Ensure a fp16 feature cache exists on disk; build it if not.

    Returns (cache_dir, train_features_path, val_features_path).

    Layout:
        <data_dir>/features_cache/<backbone_key>/
            train.npy            (N_train, 1024) fp16
            val.npy              (N_val,   1024) fp16
            train_meta.npz       (labels: int64[N_train], set_idxs: int64[N_train])
            val_meta.npz
            cache_info.json      (preprocess + backbone SHAs for invalidation)
    """
    import torch
    from PIL import Image

    key = _cache_key(finetuned_backbone)
    cache_dir = data_dir / "features_cache" / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_feat = cache_dir / "train.npy"
    val_feat   = cache_dir / "val.npy"
    train_meta = cache_dir / "train_meta.npz"
    val_meta   = cache_dir / "val_meta.npz"
    info_path  = cache_dir / "cache_info.json"

    expected = _expected_cache_info(records_train, records_val, finetuned_backbone)

    if info_path.exists() and train_feat.exists() and val_feat.exists() \
            and train_meta.exists() and val_meta.exists():
        try:
            stored = json.loads(info_path.read_text())
        except Exception:
            stored = {}
        if stored == expected:
            if console:
                console.print(f"[green]Feature cache hit:[/green] {cache_dir}")
            return cache_dir, train_feat, val_feat
        if console:
            console.print(f"[yellow]Feature cache stale — rebuilding {cache_dir}[/yellow]")

    # Cache miss — load backbone once and stream-write features.
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

    if console:
        console.print(f"[cyan]Building feature cache → {cache_dir}[/cyan]")

    backbone = setup_backbone(device, finetuned_backbone, console=console)
    transform = build_preprocess()

    def _write_split(records, feat_path: Path, meta_path: Path, split_name: str):
        n = len(records)
        # memmap in "w+" mode to avoid holding 2M x 1024 fp16 = 4GB in memory.
        memmap = np.lib.format.open_memmap(
            str(feat_path), mode="w+", dtype=np.float16, shape=(n, 1024)
        )
        labels   = np.zeros(n, dtype=np.int64)
        set_idxs = np.zeros(n, dtype=np.int64)

        valid_mask = np.zeros(n, dtype=bool)
        bad_count = 0

        progress_ctx = Progress(
            TextColumn(f"Caching {split_name}"), BarColumn(),
            TextColumn("{task.completed}/{task.total}"), TimeRemainingColumn(),
        ) if console else None
        task = None
        if progress_ctx:
            progress_ctx.__enter__()
            task = progress_ctx.add_task("cache", total=n)

        try:
            i = 0
            while i < n:
                chunk_records = records[i:i + batch_size]
                tensors = []
                idx_in_chunk = []
                for j, rec in enumerate(chunk_records):
                    try:
                        img = Image.open(rec["image_path"]).convert("RGB")
                        tensors.append(transform(img))
                        idx_in_chunk.append(j)
                    except Exception:
                        bad_count += 1

                if tensors:
                    batch = torch.stack(tensors).to(device, non_blocking=True)
                    if device == "cuda":
                        batch = batch.half()
                    with torch.no_grad():
                        feats = backbone(batch)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                    feats_np = feats.detach().cpu().to(torch.float16).numpy()
                    for j_local, rec_local_idx in enumerate(idx_in_chunk):
                        out_idx = i + rec_local_idx
                        memmap[out_idx] = feats_np[j_local]
                        labels[out_idx]   = int(chunk_records[rec_local_idx]["label_idx"])
                        set_slug = (chunk_records[rec_local_idx].get("set_slug") or "").strip().lower()
                        set_idxs[out_idx] = int(set_to_idx.get(set_slug, 0))
                        valid_mask[out_idx] = True

                i += len(chunk_records)
                if progress_ctx:
                    progress_ctx.update(task, advance=len(chunk_records))
        finally:
            if progress_ctx:
                progress_ctx.__exit__(None, None, None)

        # Drop rows whose images failed to load. We compact by copying valid
        # rows to a contiguous prefix and truncating the memmap.
        if bad_count:
            if console:
                console.print(f"  [yellow]{split_name}: skipped {bad_count} unreadable images[/yellow]")
            keep = np.where(valid_mask)[0]
            if len(keep) != n:
                # Write a compacted copy then atomically swap.
                tmp = feat_path.with_suffix(".compact.npy")
                compacted = np.lib.format.open_memmap(
                    str(tmp), mode="w+", dtype=np.float16, shape=(len(keep), 1024)
                )
                for out_i, src_i in enumerate(keep):
                    compacted[out_i] = memmap[src_i]
                del compacted, memmap
                os.replace(tmp, feat_path)
                labels   = labels[keep]
                set_idxs = set_idxs[keep]

        memmap = None  # flush
        np.savez(meta_path, labels=labels, set_idxs=set_idxs)

    _write_split(records_train, train_feat, train_meta, "train")
    _write_split(records_val,   val_feat,   val_meta,   "val")

    info_path.write_text(json.dumps(expected, indent=2))
    if console:
        console.print(f"[green]Feature cache ready:[/green] {cache_dir}")
    return cache_dir, train_feat, val_feat


def load_cached_meta(meta_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(meta_path)
    return data["labels"], data["set_idxs"]


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class SetStratifiedBatchSampler:
    """Batch sampler that draws K sets per batch and N samples per set.

    With 25K variant classes spread across thousands of sets, a random batch
    of 128 almost never contains two variants from the same set — so the loss
    is dominated by trivial cross-set discrimination. Stratifying by set puts
    the fine-grained foil-vs-base gradient signal directly in each batch.
    """

    def __init__(
        self,
        set_idxs: np.ndarray,
        k_sets_per_batch: int,
        n_per_set: int,
        num_batches: int | None = None,
        seed: int = 42,
    ):
        self.by_set: dict[int, list[int]] = {}
        for idx, s in enumerate(set_idxs):
            self.by_set.setdefault(int(s), []).append(idx)
        self.set_keys = np.array(sorted(self.by_set.keys()), dtype=np.int64)
        self.k = max(1, int(k_sets_per_batch))
        self.n = max(1, int(n_per_set))
        self._n_batches = (
            num_batches
            if num_batches is not None
            else max(1, len(set_idxs) // (self.k * self.n))
        )
        self._rng = np.random.default_rng(seed)

    @property
    def batch_size(self) -> int:
        return self.k * self.n

    def __len__(self):
        return self._n_batches

    def __iter__(self) -> Iterator[list[int]]:
        k = min(self.k, len(self.set_keys))
        for _ in range(self._n_batches):
            chosen = self._rng.choice(self.set_keys, size=k, replace=False)
            batch: list[int] = []
            for s in chosen:
                pool = self.by_set[int(s)]
                replace = len(pool) < self.n
                picks = self._rng.choice(pool, size=self.n, replace=replace)
                batch.extend(int(i) for i in picks)
            yield batch


def build_class_balanced_sampler(labels: np.ndarray, num_samples: int | None = None):
    """WeightedRandomSampler weighting each sample by 1 / class_count.

    Complements the set-stratified batch sampler — use one or the other. Classes
    range from 10 to 780 samples, and without balancing the rare classes are
    drowned out by Panini-Prizm-Base etc.
    """
    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = np.asarray(labels, dtype=np.int64)
    bincount = np.bincount(labels)
    bincount[bincount == 0] = 1  # avoid div-by-zero for absent classes
    per_class_weight = 1.0 / bincount
    sample_weights = per_class_weight[labels]
    total = int(num_samples or len(labels))
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=total,
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Heads and backbones
# ---------------------------------------------------------------------------

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


def setup_backbone(device, finetuned_path: str | None, console=None):
    import torch
    if console:
        console.print("[cyan]Loading DINOv2-ViT-L/14-reg (frozen)...[/cyan]")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    if finetuned_path:
        if console:
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


# ---------------------------------------------------------------------------
# Hierarchical set→class mask (Tier 3.7)
# ---------------------------------------------------------------------------

def build_set_class_mask(labels: list[str], set_labels: list[str]):
    """Return a (num_sets, num_classes) bool mask: True iff class c is allowed for set s.

    The `__rare__` class is allowed for every set so rare-bucket samples aren't
    starved of predictive capacity. Sets that have only rare variants end up
    with only `__rare__` allowed, which is what we want.
    """
    import torch

    num_sets = len(set_labels)
    num_classes = len(labels)
    mask = torch.zeros((num_sets, num_classes), dtype=torch.bool)

    set_to_idx = {s: i for i, s in enumerate(set_labels)}
    for cls_idx, key in enumerate(labels):
        if key == RARE_CLASS:
            mask[:, cls_idx] = True
            continue
        # Class keys are "<set_slug>::<variant>".
        set_slug = key.split("::", 1)[0]
        s = set_to_idx.get(set_slug)
        if s is not None:
            mask[s, cls_idx] = True

    return mask


def masked_cross_entropy(
    logits,
    targets,
    sample_set_idx,
    class_set_mask,
    label_smoothing: float = 0.0,
):
    """Cross-entropy loss where logits outside the sample's set are suppressed.

    This focuses gradients on "which variant within this set" instead of the
    easy "which set entirely" problem, which is the whole point of the
    hierarchical restructuring.
    """
    import torch
    import torch.nn.functional as F

    # (B, C) gather the per-sample allowed-class mask.
    sample_mask = class_set_mask.to(logits.device)[sample_set_idx]
    # Masked softmax via -inf on disallowed classes. The gold class is always
    # allowed by construction (see build_set_class_mask).
    masked = logits.masked_fill(~sample_mask, float("-inf"))
    return F.cross_entropy(masked, targets, label_smoothing=label_smoothing)
