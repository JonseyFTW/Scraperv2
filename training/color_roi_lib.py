#!/usr/bin/env python3
"""
Per-set color-ROI classifier (rec 6).

For sets where a color swap is the *only* difference between variants
(Topps Heritage red/black border, OPC red/blue border, Topps Day mother's/
father's day ribbon, Donruss press proof red/blue, Topps 2024 aqua/yellow,
etc.), the variant MLP fundamentally cannot win against DINOv2's color-
invariant representation: the foreground artwork is identical, only a thin
border or ribbon changes. Even a color-histogram side channel (rec 2) gets
washed out because the histogram is averaged over the whole card.

This module solves that subset with a deterministic, NN-free pipeline:

    1. Per-set ROI registry  — small (x0,y0,x1,y1) box where the discriminating
       color lives (the top border strip, a corner ribbon, a logo bar, ...).
    2. Reference table       — mean LAB across training samples per (set,
       variant), built once by training/08_train_color_roi.py.
    3. Classify              — at inference, crop the ROI of the query image,
       compute its mean LAB, return the variant whose reference LAB is closest.

Designed to be cheap enough to run inside the RunPod handler as an optional
step *after* the variant MLP. The handler picks the ROI classifier's answer
when (a) the set is in the registry, (b) the candidate_labels list is one of
the registered swap pairs, and (c) the LAB distance to the winner is small
enough to be confident.

Editing the registry: each entry is keyed on the set_slug as it appears in the
training manifest (lower-case, hyphenated, no leading "year-"). Both
``candidates`` and ``roi`` are optional — if a set has no manually-tuned ROI,
the trainer falls back to the center 70% crop, which works fine for cards
where the entire image is roughly the discriminator.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


# ROI is (x0, y0, x1, y1) normalized to 0..1 of the original image.
# Hand-tuned for the highest-error color-swap-only sets surfaced in the
# original confusion analysis. Add new sets as you find them — the trainer
# will automatically include any set added here.
COLOR_SWAP_REGISTRY: dict[str, dict] = {
    # Topps Heritage red/black border — the top inch of the card is the border.
    "baseball-cards-2023-topps-heritage": {
        "candidates": ["red border", "black border"],
        "roi": (0.0, 0.0, 1.0, 0.06),
    },
    "baseball-cards-2021-topps-heritage": {
        "candidates": ["red", "black border"],
        "roi": (0.0, 0.0, 1.0, 0.06),
    },
    "baseball-cards-2024-topps-heritage": {
        "candidates": ["black", "white"],
        "roi": (0.0, 0.0, 1.0, 0.06),
    },
    # O-Pee-Chee 2021/2022 red/blue border — left vertical stripe.
    "hockey-cards-2021-o-pee-chee": {
        "candidates": ["red border", "blue border"],
        "roi": (0.0, 0.0, 0.06, 1.0),
    },
    "hockey-cards-2022-o-pee-chee": {
        "candidates": ["red border", "blue border"],
        "roi": (0.0, 0.0, 0.06, 1.0),
    },
    # Topps Day mother's day pink / father's day blue — corner ribbon, lower-right.
    "baseball-cards-2024-topps-day": {
        "candidates": ["mother's day pink", "father's day blue"],
        "roi": (0.55, 0.85, 1.0, 1.0),
    },
    # Donruss press proof red/blue — top border strip.
    "football-cards-2022-panini-donruss": {
        "candidates": ["press proof red", "press proof blue"],
        "roi": (0.0, 0.0, 1.0, 0.06),
    },
    # Topps 2024 aqua/yellow — base-card foil tone shows mostly along edges.
    "baseball-cards-2024-topps": {
        "candidates": ["aqua", "yellow"],
        "roi": (0.0, 0.0, 0.05, 1.0),
    },
    # Topps Now color foils — full-bleed foil card, the whole image is colored.
    "baseball-cards-2025-topps-now": {
        "candidates": ["orange foil", "gold foil", "red foil", "black foil"],
        "roi": (0.05, 0.05, 0.95, 0.95),
    },
    "baseball-cards-2022-topps-now": {
        "candidates": ["purple", "blue", "red"],
        "roi": (0.05, 0.05, 0.95, 0.95),
    },
    "baseball-cards-2023-topps-now": {
        "candidates": ["purple", "blue", "red"],
        "roi": (0.05, 0.05, 0.95, 0.95),
    },
    "baseball-cards-2021-topps-now": {
        "candidates": ["purple", "blue"],
        "roi": (0.05, 0.05, 0.95, 0.95),
    },
}


def is_color_swap_set(set_slug: str | None) -> bool:
    return bool(set_slug) and set_slug in COLOR_SWAP_REGISTRY


def get_roi(set_slug: str) -> tuple[float, float, float, float]:
    """Return the ROI tuple for ``set_slug``, or the default center-70% box."""
    cfg = COLOR_SWAP_REGISTRY.get(set_slug, {})
    return cfg.get("roi", (0.15, 0.15, 0.85, 0.85))


def get_candidates(set_slug: str) -> list[str]:
    """Return the registered candidate variant labels for the set, or ``[]``."""
    return list(COLOR_SWAP_REGISTRY.get(set_slug, {}).get("candidates", []))


def extract_roi_lab(pil_image, set_slug: str) -> np.ndarray | None:
    """Mean LAB color over the per-set ROI. Returns None on degenerate crop."""
    roi = get_roi(set_slug)
    W, H = pil_image.size
    x0 = max(0, int(roi[0] * W))
    y0 = max(0, int(roi[1] * H))
    x1 = min(W, int(roi[2] * W))
    y1 = min(H, int(roi[3] * H))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = pil_image.crop((x0, y0, x1, y1)).convert("LAB")
    arr = np.asarray(crop, dtype=np.float32)
    return arr.reshape(-1, 3).mean(axis=0)


def classify_color_roi(
    pil_image,
    set_slug: str,
    references: dict,
    candidates: Iterable[str] | None = None,
) -> tuple[str, float, dict[str, float]] | None:
    """Pick the variant whose reference LAB is closest to the query crop.

    references: ``{set_slug: {variant_label: [L, a, b], ...}}``
    candidates: optional shortlist (defaults to registry candidates if any).

    Returns ``(best_variant, best_distance, all_distances)`` or None when the
    set isn't registered / has no references / the crop is degenerate.
    """
    set_refs = references.get(set_slug)
    if not set_refs:
        return None
    lab = extract_roi_lab(pil_image, set_slug)
    if lab is None:
        return None
    if candidates is None:
        candidates = list(set_refs.keys())

    distances: dict[str, float] = {}
    for variant in candidates:
        ref = set_refs.get(variant)
        if ref is None:
            continue
        d = float(np.linalg.norm(lab - np.asarray(ref, dtype=np.float32)))
        distances[variant] = d

    if not distances:
        return None
    best = min(distances, key=distances.get)
    return best, distances[best], distances


def classify_with_confidence(
    pil_image,
    set_slug: str,
    references: dict,
    candidates: Iterable[str] | None = None,
    margin: float = 4.0,
) -> tuple[str, float, dict[str, float]] | None:
    """Same as ``classify_color_roi`` but only returns a result when the
    second-best distance is at least ``margin`` LAB units away — i.e. when
    the ROI classifier is actually confident.

    LAB distance margin reference (rough):
        margin <  2  : indistinguishable to the human eye
        margin ~  4  : just-noticeable difference
        margin >  8  : clearly different colors

    Tighter margin = the ROI classifier defers to the MLP more often.
    """
    result = classify_color_roi(pil_image, set_slug, references, candidates)
    if result is None:
        return None
    best, best_d, distances = result
    if len(distances) < 2:
        return result
    sorted_d = sorted(distances.values())
    second_d = sorted_d[1]
    if (second_d - best_d) < margin:
        return None  # not confident; let the MLP decide
    return best, best_d, distances
