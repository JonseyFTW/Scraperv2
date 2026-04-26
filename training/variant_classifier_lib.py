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
# Color-histogram side channel (rec 2)
# ---------------------------------------------------------------------------
# DINOv2 is largely color-invariant by design — its self-supervised pretraining
# pushes representations to ignore color shifts. For ~30-40% of our top
# confusions, color IS the only thing that separates classes (Topps Now
# purple↔blue, OPC red/blue border, mosaic finishes, etc.). Concatenating a
# tiny per-image color histogram onto the DINOv2 feature gives the head the
# explicit color signal it needs without changing the backbone.
#
# The histogram is computed once per image during cache-building, persisted as
# fp16 next to the DINOv2 features, and concatenated into the feature vector
# at training and eval time. Cache invalidation is keyed on (kind, bins) so
# changing the setting forces a rebuild.

COLOR_HIST_CHOICES = ("none", "lab32", "hsv32", "rgb32")


# ---------------------------------------------------------------------------
# Edge / die-cut profile (rec 5)
# ---------------------------------------------------------------------------
# Die-cut variants (zebra prizm die cut vs base, dragon scale die cut vs base)
# differ from base cards only at the silhouette: die-cuts have irregular
# notches at the card boundary while bases have straight edges. DINOv2's
# global features mostly ignore the silhouette since the same artwork sits in
# the middle of both. A 16-dim edge profile captures the boundary signal
# directly without standing up a full segmentation model.
EDGE_PROFILE_DIM = 16  # 4 borders (top/right/bottom/left) x 4 stats (mean/std/max/p90)


def edge_profile_dim(enabled: bool) -> int:
    return EDGE_PROFILE_DIM if enabled else 0


def compute_edge_profile(pil_image, target_size: int = 64, border_thickness: int = 4) -> np.ndarray:
    """16-dim per-image edge-profile vector.

    Pipeline:
        1. Resize the image to ``target_size × target_size`` grayscale
           (deterministic so stats are comparable across cards of different
           original resolutions).
        2. 3-pixel central-difference gradient (cheap Sobel approximation).
        3. For each of the 4 image borders (top / right / bottom / left), take
           the ``border_thickness``-pixel-wide strip and compute
           (mean, std, max, p90) of the gradient magnitude.

    Output is a (16,) float32 vector. A regular full-bleed card produces 4
    similar high-mean / low-std edges; a die-cut produces high-std edges with
    larger max because the gradient is dominated by the notches.
    """
    from PIL import Image
    if pil_image.mode != "L":
        gray = pil_image.convert("L")
    else:
        gray = pil_image
    gray = gray.resize((target_size, target_size), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32) / 255.0

    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
    grad = np.sqrt(gx * gx + gy * gy)

    bt = border_thickness
    strips = {
        "top":    grad[:bt, :].ravel(),
        "right":  grad[:, -bt:].ravel(),
        "bottom": grad[-bt:, :].ravel(),
        "left":   grad[:, :bt].ravel(),
    }
    out: list[float] = []
    for k in ("top", "right", "bottom", "left"):
        s = strips[k]
        out.append(float(s.mean()))
        out.append(float(s.std()))
        out.append(float(s.max()))
        out.append(float(np.percentile(s, 90)))
    return np.asarray(out, dtype=np.float32)


def color_hist_dim(kind: str | None, bins: int) -> int:
    """Total flattened histogram dimension. 0 when disabled."""
    if not kind or kind == "none":
        return 0
    return 3 * int(bins)


def compute_color_histogram(
    pil_image,
    kind: str,
    bins: int,
    center_crop_pct: float = 0.7,
) -> np.ndarray:
    """L2-normalized concatenated 3-channel histogram for a PIL image.

    The crop trims the outer border of the image so background pixels (the
    matte beyond the card edge) don't dominate the histogram — sports-card
    crops generally have the card filling roughly the center 70%.

    kind:
        ``lab32``  perceptual color space; best for the parallel-color failure
                   modes since L/a/b separate luminance from chroma.
        ``hsv32``  H ranges over hue circle; useful when foil tone is the
                   discriminator (orange vs gold).
        ``rgb32``  raw RGB; baseline.
        ``none``   disabled — returns a zero-length array.
    """
    if not kind or kind == "none":
        return np.zeros(0, dtype=np.float32)

    W, H = pil_image.size
    if center_crop_pct < 1.0 and W > 4 and H > 4:
        cw, ch = int(W * center_crop_pct), int(H * center_crop_pct)
        x0, y0 = (W - cw) // 2, (H - ch) // 2
        pil_image = pil_image.crop((x0, y0, x0 + cw, y0 + ch))

    if kind == "lab32":
        arr = np.asarray(pil_image.convert("LAB"), dtype=np.uint8)
    elif kind == "hsv32":
        arr = np.asarray(pil_image.convert("HSV"), dtype=np.uint8)
    elif kind == "rgb32":
        arr = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    else:
        raise ValueError(f"Unknown color-hist kind: {kind!r}")

    hists = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 256))
        hists.append(h.astype(np.float32))
    out = np.concatenate(hists)
    norm = float(np.linalg.norm(out))
    if norm > 0:
        out /= norm
    return out


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
    """Returns ``(image_tensor, side_channels, label_idx, set_idx)`` tuples.

    ``side_channels`` is a numpy float32 array combining color hist (if
    enabled) and edge profile (if enabled), in that fixed order. It is
    ``None`` when neither is enabled. Step 6's eval path mirrors whatever
    the checkpoint was trained with.
    """

    def __init__(
        self,
        records: list[dict],
        transform,
        set_to_idx: dict[str, int],
        color_hist_kind: str = "none",
        color_hist_bins: int = 32,
        edge_channel: bool = False,
    ):
        self.records = records
        self.transform = transform
        self.set_to_idx = set_to_idx
        self.color_hist_kind = color_hist_kind or "none"
        self.color_hist_bins = int(color_hist_bins)
        self.color_enabled = self.color_hist_kind != "none"
        self.edge_enabled = bool(edge_channel)

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
        side_parts: list[np.ndarray] = []
        if self.color_enabled:
            side_parts.append(
                compute_color_histogram(img, self.color_hist_kind, self.color_hist_bins)
            )
        if self.edge_enabled:
            side_parts.append(compute_edge_profile(img))
        side = np.concatenate(side_parts) if side_parts else None
        return self.transform(img), side, int(rec["label_idx"]), int(set_idx)


def collate_image_skip_none(batch):
    import torch
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs     = torch.stack([b[0] for b in batch])
    has_side = batch[0][1] is not None
    if has_side:
        side = torch.from_numpy(np.stack([b[1] for b in batch], axis=0)).float()
    else:
        side = None
    labels   = torch.tensor([b[2] for b in batch], dtype=torch.long)
    set_idxs = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return imgs, side, labels, set_idxs


# ---------------------------------------------------------------------------
# Cached-feature dataset (Tier 1.1: feature caching)
# ---------------------------------------------------------------------------

class CachedFeatureDataset:
    """Memory-mapped reader over a precomputed DINOv2 feature tensor.

    The backbone is frozen and the preprocess is deterministic, so there is no
    reason to re-run DINOv2 every epoch. We compute features once to disk and
    the "model" becomes a tiny Linear head on cached 1024-dim vectors — which
    turns each training epoch from hours into seconds.

    When a color-histogram cache is also present (rec 2), it is concatenated
    onto each feature vector at lookup time so the downstream head sees a
    single ``(1024 + 3*bins)``-dim input.
    """

    def __init__(
        self,
        features_path: str,
        labels: np.ndarray,
        set_idxs: np.ndarray,
        color_path: str | None = None,
        edge_path: str | None = None,
    ):
        # mmap_mode="r" keeps the 4-8GB tensor out of RAM — the OS pages it in
        # as the sampler touches random indices.
        self.features = np.load(features_path, mmap_mode="r")
        assert self.features.shape[0] == len(labels) == len(set_idxs), (
            f"cache length mismatch: features={self.features.shape[0]} "
            f"labels={len(labels)} set_idxs={len(set_idxs)}"
        )
        self.color = np.load(color_path, mmap_mode="r") if color_path else None
        if self.color is not None:
            assert self.color.shape[0] == self.features.shape[0], (
                f"color cache length {self.color.shape[0]} does not match "
                f"feature cache length {self.features.shape[0]}"
            )
        self.edge = np.load(edge_path, mmap_mode="r") if edge_path else None
        if self.edge is not None:
            assert self.edge.shape[0] == self.features.shape[0], (
                f"edge cache length {self.edge.shape[0]} does not match "
                f"feature cache length {self.features.shape[0]}"
            )
        self.labels = labels
        self.set_idxs = set_idxs

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Copy the rows — the memmap is fp16 on disk; cast happens in collate.
        feat = np.array(self.features[idx], copy=True)
        if self.color is not None:
            color = np.array(self.color[idx], copy=True)
            feat = np.concatenate([feat, color], axis=0)
        if self.edge is not None:
            edge = np.array(self.edge[idx], copy=True)
            feat = np.concatenate([feat, edge], axis=0)
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
    color_hist_kind: str = "none",
    color_hist_bins: int = 0,
    edge_channel: bool = False,
) -> dict:
    return {
        "preprocess_spec":     PREPROCESS_SPEC,
        "finetuned_backbone":  finetuned_backbone,
        "finetuned_sha":       sha256_of_file(finetuned_backbone),
        "n_train":             len(records_train),
        "n_val":               len(records_val),
        "color_hist_kind":     color_hist_kind or "none",
        "color_hist_bins":     int(color_hist_bins) if color_hist_kind and color_hist_kind != "none" else 0,
        "edge_channel":        bool(edge_channel),
        "edge_profile_dim":    EDGE_PROFILE_DIM if edge_channel else 0,
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
    color_hist_kind: str = "none",
    color_hist_bins: int = 32,
    edge_channel: bool = False,
) -> tuple[Path, Path, Path, Path | None, Path | None, Path | None, Path | None]:
    """Ensure a fp16 feature cache exists on disk; build it if not.

    Returns ``(cache_dir, train_features_path, val_features_path,
    train_color_path, val_color_path)`` — the color paths are ``None`` when
    ``color_hist_kind == "none"``.

    Layout:
        <data_dir>/features_cache/<backbone_key>__<color_key>/
            train.npy            (N_train, 1024) fp16
            val.npy              (N_val,   1024) fp16
            train_color.npy      (N_train, 3*bins) fp16   [if color enabled]
            val_color.npy        (N_val,   3*bins) fp16   [if color enabled]
            train_meta.npz       (labels: int64[N_train], set_idxs: int64[N_train])
            val_meta.npz
            cache_info.json      (preprocess + backbone SHAs + color cfg for invalidation)
    """
    import torch
    from PIL import Image

    base_key = _cache_key(finetuned_backbone)
    color_key = "color-none" if (not color_hist_kind or color_hist_kind == "none") else f"color-{color_hist_kind}-{color_hist_bins}"
    edge_key = "edge-on" if edge_channel else "edge-off"
    key = f"{base_key}__{color_key}__{edge_key}"
    hist_dim = color_hist_dim(color_hist_kind, color_hist_bins)
    edge_dim = edge_profile_dim(edge_channel)
    cache_dir = data_dir / "features_cache" / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_feat = cache_dir / "train.npy"
    val_feat   = cache_dir / "val.npy"
    train_color = cache_dir / "train_color.npy" if hist_dim else None
    val_color   = cache_dir / "val_color.npy"   if hist_dim else None
    train_edge  = cache_dir / "train_edge.npy"  if edge_dim else None
    val_edge    = cache_dir / "val_edge.npy"    if edge_dim else None
    train_meta = cache_dir / "train_meta.npz"
    val_meta   = cache_dir / "val_meta.npz"
    info_path  = cache_dir / "cache_info.json"

    expected = _expected_cache_info(
        records_train, records_val, finetuned_backbone,
        color_hist_kind=color_hist_kind, color_hist_bins=color_hist_bins,
        edge_channel=edge_channel,
    )

    required_files = [train_feat, val_feat, train_meta, val_meta]
    if hist_dim:
        required_files += [train_color, val_color]
    if edge_dim:
        required_files += [train_edge, val_edge]
    if info_path.exists() and all(p.exists() for p in required_files):
        try:
            stored = json.loads(info_path.read_text())
        except Exception:
            stored = {}
        if stored == expected:
            if console:
                console.print(f"[green]Feature cache hit:[/green] {cache_dir}")
            return cache_dir, train_feat, val_feat, train_color, val_color, train_edge, val_edge
        if console:
            console.print(f"[yellow]Feature cache stale — rebuilding {cache_dir}[/yellow]")

    # Cache miss — load backbone once and stream-write features.
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

    if console:
        console.print(f"[cyan]Building feature cache → {cache_dir}[/cyan]")

    backbone = setup_backbone(device, finetuned_backbone, console=console)
    # Note (rec 1): build_preprocess() is the deterministic DINOv2 preprocess —
    # resize, center-crop, ImageNet normalize. There is intentionally no
    # ColorJitter / hue-shift / saturation augmentation here, because color is
    # the only label-distinguishing feature for the parallel-color confusions
    # (Topps Now purple↔blue, OPC red/blue border, etc.). If you ever add
    # augmentation, keep it shape/crop/blur only — never color.
    transform = build_preprocess()

    def _write_split(records, feat_path: Path, color_path: Path | None,
                     edge_path: Path | None, meta_path: Path, split_name: str):
        n = len(records)
        # memmap in "w+" mode to avoid holding 2M x 1024 fp16 = 4GB in memory.
        memmap = np.lib.format.open_memmap(
            str(feat_path), mode="w+", dtype=np.float16, shape=(n, 1024)
        )
        color_mm = None
        if color_path is not None and hist_dim:
            color_mm = np.lib.format.open_memmap(
                str(color_path), mode="w+", dtype=np.float16, shape=(n, hist_dim)
            )
        edge_mm = None
        if edge_path is not None and edge_dim:
            edge_mm = np.lib.format.open_memmap(
                str(edge_path), mode="w+", dtype=np.float16, shape=(n, edge_dim)
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
                color_vecs: list[np.ndarray] = []
                edge_vecs:  list[np.ndarray] = []
                idx_in_chunk = []
                for j, rec in enumerate(chunk_records):
                    try:
                        img = Image.open(rec["image_path"]).convert("RGB")
                        tensors.append(transform(img))
                        if color_mm is not None:
                            color_vecs.append(
                                compute_color_histogram(img, color_hist_kind, color_hist_bins)
                            )
                        if edge_mm is not None:
                            edge_vecs.append(compute_edge_profile(img))
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
                        if color_mm is not None:
                            color_mm[out_idx] = color_vecs[j_local].astype(np.float16)
                        if edge_mm is not None:
                            edge_mm[out_idx] = edge_vecs[j_local].astype(np.float16)
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
                # Write a compacted copy then atomically swap. Repeat for the
                # color memmap if present so the two stay row-aligned.
                tmp = feat_path.with_suffix(".compact.npy")
                compacted = np.lib.format.open_memmap(
                    str(tmp), mode="w+", dtype=np.float16, shape=(len(keep), 1024)
                )
                for out_i, src_i in enumerate(keep):
                    compacted[out_i] = memmap[src_i]
                del compacted, memmap
                os.replace(tmp, feat_path)
                memmap = None

                if color_mm is not None and color_path is not None:
                    tmp_c = color_path.with_suffix(".compact.npy")
                    compacted_c = np.lib.format.open_memmap(
                        str(tmp_c), mode="w+", dtype=np.float16, shape=(len(keep), hist_dim)
                    )
                    for out_i, src_i in enumerate(keep):
                        compacted_c[out_i] = color_mm[src_i]
                    del compacted_c, color_mm
                    os.replace(tmp_c, color_path)
                    color_mm = None

                if edge_mm is not None and edge_path is not None:
                    tmp_e = edge_path.with_suffix(".compact.npy")
                    compacted_e = np.lib.format.open_memmap(
                        str(tmp_e), mode="w+", dtype=np.float16, shape=(len(keep), edge_dim)
                    )
                    for out_i, src_i in enumerate(keep):
                        compacted_e[out_i] = edge_mm[src_i]
                    del compacted_e, edge_mm
                    os.replace(tmp_e, edge_path)
                    edge_mm = None

                labels   = labels[keep]
                set_idxs = set_idxs[keep]

        memmap = None  # flush
        color_mm = None
        edge_mm = None
        np.savez(meta_path, labels=labels, set_idxs=set_idxs)

    _write_split(records_train, train_feat, train_color, train_edge, train_meta, "train")
    _write_split(records_val,   val_feat,   val_color,   val_edge,   val_meta,   "val")

    info_path.write_text(json.dumps(expected, indent=2))
    if console:
        console.print(f"[green]Feature cache ready:[/green] {cache_dir}")
    return cache_dir, train_feat, val_feat, train_color, val_color, train_edge, val_edge


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

def build_head(arch: str, input_dim: int, num_classes: int,
               hidden_dim: int = 256, dropout: float = 0.1):
    """Build the classification head.

    ``linear`` ignores ``hidden_dim`` / ``dropout``. ``mlp`` is a single hidden
    layer with GELU + Dropout; bump ``dropout`` for stronger regularization when
    the head is overfitting.
    """
    import torch.nn as nn
    if arch == "linear":
        return nn.Linear(input_dim, num_classes)
    if arch == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    raise ValueError(f"Unknown arch: {arch}")


def head_forward_with_trunk(head, x):
    """Forward through ``head`` and return ``(logits, trunk_features)``.

    The "trunk" is the input to the head's final ``Linear`` layer. Auxiliary
    heads (SupCon projection — rec 3, foil aux — rec 4) hang off this trunk so
    their gradients flow through the same shared layers as the classifier.

    For ``nn.Linear`` (the --arch linear case) there is no trunk — gradients
    from auxiliary losses can't influence the classifier. Caller is expected
    to short-circuit auxiliary losses in that case.
    """
    import torch.nn as nn
    if isinstance(head, nn.Linear):
        return head(x), x
    # nn.Sequential MLP: pass through every layer except the final Linear, then
    # apply the final layer separately so we can return the pre-classifier
    # representation as the trunk output.
    layers = list(head)
    trunk = x
    for layer in layers[:-1]:
        trunk = layer(trunk)
    logits = layers[-1](trunk)
    return logits, trunk


# ---------------------------------------------------------------------------
# Pairwise contrastive (rec 3): SupCon with same-set candidate masking
# ---------------------------------------------------------------------------
# DINOv2 + CE alone collapses sibling parallels (e.g. Topps Now purple vs blue)
# into the same neighborhood because the visual difference is tiny and the
# softmax has no incentive to push them further apart than "different argmax".
# A supervised contrastive loss restricted to within-set pairs explicitly pulls
# same-class samples together and pushes same-set siblings apart — exactly the
# fine-grained gradient signal these confusions need.
#
# We piggyback on the existing set_stratified sampler: each batch already
# contains K sets x N variants per set, so within-set candidates are abundant.

class ProjectionHead:
    """Two-layer MLP projection used for SupCon. Output is L2-normalized.

    Defined as a factory function rather than a torch.nn.Module subclass to
    avoid importing torch at module import time (other lib functions defer
    that). Use ``build_projection_head`` below.
    """
    pass


def build_projection_head(input_dim: int, proj_dim: int = 128,
                          hidden_dim: int | None = None, dropout: float = 0.0):
    import torch.nn as nn
    hd = hidden_dim or input_dim
    layers = [nn.Linear(input_dim, hd), nn.GELU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hd, proj_dim))
    return nn.Sequential(*layers)


def supcon_within_set_loss(z, labels, set_idxs, temperature: float = 0.1):
    """Supervised contrastive loss with same-set candidate masking.

    For each anchor i:
        candidates = j s.t. set_idxs[j] == set_idxs[i] and j != i
        positives  = j s.t. labels[j]   == labels[i]   and j is a candidate
        loss_i     = -(1/|P|) Σ_p log( exp(sim_ip/τ) / Σ_c exp(sim_ic/τ) )

    Anchors with zero positives in their batch are skipped (they happen when a
    rare class lands in a batch with no siblings).

    ``z`` MUST already be L2-normalized for the dot-product to be a cosine
    similarity. ``ProjectionHead`` handles that for the SupCon pathway.

    Returns 0 (with grad enabled) when no anchor has a positive — keeps the
    optimizer happy on degenerate batches.
    """
    import torch

    B = z.size(0)
    if B < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype, requires_grad=True)

    sim = (z @ z.t()) / float(temperature)

    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    same_set  = set_idxs.unsqueeze(0) == set_idxs.unsqueeze(1)
    cand_mask = same_set & ~self_mask
    same_cls  = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask  = same_cls & cand_mask

    has_pos = pos_mask.any(dim=-1)
    if not has_pos.any():
        return torch.zeros((), device=z.device, dtype=z.dtype, requires_grad=True)

    # logsumexp over candidates only — masked positions contribute nothing.
    sim_for_lse = sim.masked_fill(~cand_mask, float("-inf"))
    log_denom = torch.logsumexp(sim_for_lse, dim=-1)  # (B,)

    # log P(j | i) = sim[i,j] - log_denom[i] for j in candidate set.
    log_prob = sim - log_denom.unsqueeze(-1)

    pos_count = pos_mask.float().sum(dim=-1).clamp(min=1.0)
    per_anchor_loss = -(log_prob * pos_mask.float()).sum(dim=-1) / pos_count
    return per_anchor_loss[has_pos].mean()


# ---------------------------------------------------------------------------
# Foil-tone auxiliary head (rec 4)
# ---------------------------------------------------------------------------
# Foil and refractor tones (gold/orange/blue/red/holo/shimmer/etc.) are
# derivable from the variant label string. We hang a tiny CE aux head off the
# shared MLP trunk that predicts the foil tone, and mix it into the loss with
# --foil-aux-lambda. This forces the trunk to encode foil tone explicitly,
# which is a much weaker signal than the per-class label and so survives in
# the feature space rather than being lost to one-hot collapse.

# Closed canonical taxonomy. Order matters only for output stability — at
# train time we index class -> tone via this list, and the same list is
# stamped into the checkpoint so the aux head's output column meanings are
# reproducible.
FOIL_TONES = [
    "none",        # no foil/refractor keyword detected (base, border-color-only, mini, etc.)
    "gold",
    "rainbow",
    "silver",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "aqua",
    "black",
    "white",
    "negative",
    "holo",        # "holofoil" / "hologram" / "holo blue" — generic holo
    "mosaic",
    "wave",
    "shimmer",
    "refractor",   # "refractor" / "prizm" with no color qualifier
    "ice",
    "die_cut",
]


# Color tokens are checked before texture tokens so "blue refractor" -> "blue"
# rather than "refractor". The first match wins.
import re as _re
_FOIL_KEYWORD_RULES = [
    (_re.compile(r"\bgold\b"),                  "gold"),
    (_re.compile(r"\brainbow\b"),               "rainbow"),
    (_re.compile(r"\bsilver\b"),                "silver"),
    (_re.compile(r"\bred\b"),                   "red"),
    (_re.compile(r"\borange\b"),                "orange"),
    (_re.compile(r"\b(yellow|canary)\b"),       "yellow"),
    (_re.compile(r"\bgreen\b"),                 "green"),
    (_re.compile(r"\bblue\b"),                  "blue"),
    (_re.compile(r"\b(purple|amethyst)\b"),     "purple"),
    (_re.compile(r"\b(pink|fuchsia)\b"),        "pink"),
    (_re.compile(r"\b(aqua|teal)\b"),           "aqua"),
    (_re.compile(r"\bblack\b"),                 "black"),
    (_re.compile(r"\bwhite\b"),                 "white"),
    (_re.compile(r"\bnegative\b"),              "negative"),
    (_re.compile(r"\b(holo|hologram|holofoil)\b"), "holo"),
    (_re.compile(r"\bmosaic\b"),                "mosaic"),
    (_re.compile(r"\bwave\b"),                  "wave"),
    (_re.compile(r"\bshimmer\b"),               "shimmer"),
    (_re.compile(r"\b(refractor|prizm)\b"),     "refractor"),
    (_re.compile(r"\bice\b"),                   "ice"),
    (_re.compile(r"\b(die cut|die-cut)\b"),     "die_cut"),
]


def parse_foil_tone(variant_label: str) -> str:
    """Map a free-text variant label to one of FOIL_TONES."""
    if not variant_label:
        return "none"
    s = variant_label.lower()
    for pat, tone in _FOIL_KEYWORD_RULES:
        if pat.search(s):
            return tone
    return "none"


def build_class_to_foil_tone(labels: list[str]) -> np.ndarray:
    """For each class index, return the corresponding foil-tone index.

    ``labels`` are full class keys (``<set>::<variant>``). ``__rare__`` and
    classes without a foil-tone keyword are mapped to ``none``.
    """
    tone_to_idx = {t: i for i, t in enumerate(FOIL_TONES)}
    out = np.zeros(len(labels), dtype=np.int64)
    for i, label in enumerate(labels):
        if label == RARE_CLASS:
            out[i] = tone_to_idx["none"]
            continue
        variant = label.split("::", 1)[1] if "::" in label else label
        out[i] = tone_to_idx[parse_foil_tone(variant)]
    return out


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

    Gradients focus on "which variant within this set" instead of the easy
    "which set entirely" problem, which is the point of the hierarchical
    restructuring.

    Label smoothing is distributed uniformly over the *allowed* classes for
    each sample rather than over all C classes, which is the correct per-sample
    smoothing under a masked softmax and avoids the ``loss=inf`` NaN that
    ``F.cross_entropy(masked, targets, label_smoothing=α)`` produces when α
    tries to put mass on -inf logits.
    """
    import torch
    import torch.nn.functional as F

    # (B, C) gather the per-sample allowed-class mask.
    sample_mask = class_set_mask.to(logits.device)[sample_set_idx]
    masked_logits = logits.masked_fill(~sample_mask, float("-inf"))

    if label_smoothing <= 0.0:
        return F.cross_entropy(masked_logits, targets)

    # Manually compute smoothed CE while respecting the mask.
    log_probs = F.log_softmax(masked_logits, dim=-1)
    # log_probs at masked positions is -inf. Zero them out in the final sum so
    # 0 * -inf never appears — the smoothed target has zero mass there anyway.
    log_probs = torch.where(sample_mask, log_probs, torch.zeros_like(log_probs))

    # Smoothed target: (1-α) on gold + α uniformly over the K allowed classes.
    # Equivalent closed form: α/K on each allowed class, plus (1-α) extra on gold.
    K = sample_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
    uniform_over_allowed = sample_mask.float() / K
    one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
    smoothed_target = (1.0 - label_smoothing) * one_hot + label_smoothing * uniform_over_allowed

    loss = -(smoothed_target * log_probs).sum(dim=-1)
    return loss.mean()


# ---------------------------------------------------------------------------
# Evaluation core (shared by step 5 auto-eval and step 6 standalone eval)
# ---------------------------------------------------------------------------

def evaluate_head(
    head,
    val_loader,
    labels: list[str],
    class_set_mask=None,
    device: str = "cuda",
    backbone=None,
    use_cache: bool = True,
    top_confusions: int = 50,
):
    """Run a full pass over ``val_loader`` and compute metrics + confusions.

    ``val_loader`` may yield either cached features (use_cache=True) or images
    that need DINOv2 forwarding (use_cache=False, ``backbone`` required).

    Returns a dict with:
        top1, top3, macro_f1, n_val,
        per_class:  list[(label, support, prec, rec, f1)]
        confusions: list[(true_label, pred_label, count)]  truncated to top_confusions
    """
    import torch
    from collections import Counter

    head.eval()
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    confusions: Counter = Counter()
    total = 0
    top1_correct = 0
    top3_correct = 0

    if class_set_mask is not None:
        class_set_mask = class_set_mask.to(device)

    num_classes = len(labels)

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            if use_cache:
                # Cache path: feature row already includes color hist (if any)
                # because CachedFeatureDataset concatenates at lookup time.
                feats, gold, set_idxs = batch
                feats = feats.to(device, non_blocking=True)
            else:
                imgs, side, gold, set_idxs = batch
                imgs = imgs.to(device, non_blocking=True)
                feats = extract_features(backbone, imgs, device)
                if side is not None:
                    feats = torch.cat([feats, side.to(device, non_blocking=True)], dim=-1)
            gold = gold.to(device, non_blocking=True)
            set_idxs = set_idxs.to(device, non_blocking=True)

            logits = head(feats)
            if class_set_mask is not None:
                logits = logits.masked_fill(~class_set_mask[set_idxs], float("-inf"))
            k = min(3, num_classes)
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

    per_class = []
    f1s = []
    for idx, label in enumerate(labels):
        support = tp[idx] + fn[idx]
        prec = tp[idx] / (tp[idx] + fp[idx]) if (tp[idx] + fp[idx]) else 0.0
        rec  = tp[idx] / support if support else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        if support:
            f1s.append(f1)
        per_class.append((label, support, prec, rec, f1))
    macro_f1 = sum(f1s) / max(len(f1s), 1)

    confusion_rows = [
        (labels[g], labels[p], c)
        for (g, p), c in confusions.most_common(top_confusions)
    ]

    return {
        "top1":       top1,
        "top3":       top3,
        "macro_f1":   macro_f1,
        "n_val":      total,
        "per_class":  per_class,
        "confusions": confusion_rows,
    }


def write_eval_outputs(out_dir, eval_result: dict, summary_extra: dict | None = None):
    """Write per_class_metrics.csv, top_confusions.csv, and summary.json.

    ``summary_extra`` is merged into summary.json so callers can stamp the run
    with checkpoint path, training flags, run name, etc.
    """
    import csv
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_class_path = out_dir / "per_class_metrics.csv"
    with open(per_class_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "support", "precision", "recall", "f1"])
        for label, support, prec, rec, f1 in eval_result["per_class"]:
            w.writerow([label, support, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

    confusions_path = out_dir / "top_confusions.csv"
    with open(confusions_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_label", "pred_label", "count"])
        for true_label, pred_label, count in eval_result["confusions"]:
            w.writerow([true_label, pred_label, count])

    summary = {
        "n_val":    eval_result["n_val"],
        "top1":     eval_result["top1"],
        "top3":     eval_result["top3"],
        "macro_f1": eval_result["macro_f1"],
    }
    if summary_extra:
        summary.update(summary_extra)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return per_class_path, confusions_path, out_dir / "summary.json"
