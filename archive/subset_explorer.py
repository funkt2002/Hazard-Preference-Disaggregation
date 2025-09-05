#!/usr/bin/env python3
"""
Explore contiguous ~20-parcel subsets and rank them by "mixed" weights.

Workflow:
- Load parcels
- Build contiguity graph (Queen adjacency via spatial index with touches predicate)
- Sample seed parcels and grow connected clusters to target size range
- Score each cluster by weight entropy (higher = more mixed) and dominance (lower = better)
- Show top K clusters with maps (plt.show) and print their metrics

Notes:
- Uses SA to optimize average rank for each cluster; if Gurobi is available, you can optionally switch to linear-UTA weights.
- No figures are saved; everything is shown in the integrated terminal per user preference.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box


# Paths/config
BASE_PATH = Path("/Users/theofunk/Desktop/NARSC paper")
DATA_DIR = BASE_PATH / "data"

# Criteria (include optional WUI if present; synthesize wui_s if only raw exists)
CRITERIA_COLS_BASE = ["qtrmi_s", "hvhsz_s", "agfb_s", "hbrn_s", "slope_s"]
EXTRA_OPTIONAL = ["wui_s"]

# Cluster search params
TARGET_SIZE = 20
SIZE_MIN = 18
SIZE_MAX = 22
NUM_SEEDS = 200
TOP_K = 10
RANDOM_SEED = 42

# Fast mode: skip SA/weight scoring, just find and map N clusters quickly
FAST_MAP_ONLY = True
MAP_COUNT = 10

# WUI threshold constraint (clusters built only from parcels with wui_s > threshold)
WUI_ATTR_THRESHOLD = 0.1

# Optional: write found clusters to shapefiles (off by default)
SAVE_SUBSETS = False
SAVE_DIR = DATA_DIR

# Bias a portion of seeds toward the spatial center to get a few central clusters
CENTER_SEED_FRACTION = 0.3  # 30% of seeds from parcels closest to global centroid


def ensure_features(gdf: gpd.GeoDataFrame, criteria_cols: List[str]) -> gpd.GeoDataFrame:
    if "hvhsz_s" not in gdf.columns and "hvhsz" in gdf.columns:
        vals = pd.to_numeric(gdf["hvhsz"], errors="coerce").fillna(0.0)
        vmin, vmax = float(vals.min()), float(vals.max())
        gdf["hvhsz_s"] = (vals - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    if "wui_s" not in gdf.columns:
        if "wui" in gdf.columns:
            vals = pd.to_numeric(gdf["wui"], errors="coerce").fillna(0.0)
            vmin, vmax = float(vals.min()), float(vals.max())
            gdf["wui_s"] = (vals - vmin) / (vmax - vmin) if vmax > vmin else 0.0
        else:
            gdf["wui_s"] = 0.0
    for c in criteria_cols:
        if c not in gdf.columns:
            gdf[c] = 0.0
    return gdf


def build_adjacency(gdf: gpd.GeoDataFrame) -> List[List[int]]:
    geoms = gdf.geometry.values
    sindex = gdf.sindex
    neighbors: List[List[int]] = [[] for _ in range(len(gdf))]
    for i, geom in enumerate(geoms):
        try:
            cand = list(sindex.query(geom, predicate="touches"))  # fast path
        except Exception:
            # Fallback: intersects then filter by touches
            cand = list(sindex.query(geom))
            cand = [j for j in cand if i != j and geoms[j].touches(geom)]
        neighbors[i] = [j for j in cand if j != i]
    return neighbors


def _choose_wui_values(parcels: gpd.GeoDataFrame) -> np.ndarray:
    """Return a 0-1 normalized WUI vector from available columns.
    Preference: 'hwui_s' → 'wui_s' → 'hlfmi_wui' (scaled 0-1 if 0-100).
    """
    for col in ("hwui_s", "wui_s", "hlfmi_wui"):
        if col in parcels.columns:
            s = pd.to_numeric(parcels[col], errors="coerce").fillna(0.0)
            arr = s.to_numpy()
            if col == "hlfmi_wui" and np.nanmax(arr) > 1.0:
                arr = arr / 100.0
            # Clip to [0,1]
            arr = np.clip(arr, 0.0, 1.0)
            return arr
    # Fallback: all zeros if no column found
    return np.zeros(len(parcels), dtype=float)


def _make_allowed_set(parcels: gpd.GeoDataFrame, threshold: float) -> Set[int]:
    vals = _choose_wui_values(parcels)
    return {i for i, v in enumerate(vals) if v > threshold}


def grow_cluster(seed: int, neighbors: List[List[int]], size_min: int, size_max: int, allowed: Optional[Set[int]] = None) -> List[int] | None:
    # BFS growth to get a connected set within size range; stop when exceeding max
    if allowed is not None and seed not in allowed:
        return None
    visited = set([seed])
    queue = [seed]
    while queue and len(visited) < size_max:
        u = queue.pop(0)
        for v in neighbors[u]:
            if v not in visited and (allowed is None or v in allowed):
                visited.add(v)
                if len(visited) >= size_max:
                    break
                queue.append(v)
    if size_min <= len(visited) <= size_max:
        return list(visited)
    # If overshot, trim to closest to target by BFS order
    if len(visited) > size_max:
        return list(sorted(list(visited)))[:size_max]
    return None


def average_rank(weights: np.ndarray, X: np.ndarray, selected_idx: List[int]) -> float:
    if len(selected_idx) == 0:
        return 1e9
    w = np.array(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        return 1e9
    w /= s
    scores = X @ w
    order = np.argsort(scores)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order)) + 1
    return float(np.mean(ranks[selected_idx]))


def run_sa(X: np.ndarray, selected_idx: List[int], w0: np.ndarray, max_iters: int = 150) -> Tuple[np.ndarray, float]:
    current_w = w0.copy()
    current_obj = average_rank(current_w, X, selected_idx)
    best_w = current_w.copy()
    best_obj = current_obj
    initial_temp = max(0.001, best_obj * 0.1)
    cooling = 0.95
    min_temp = 1e-3
    for it in range(max_iters):
        temp = max(min_temp, initial_temp * (cooling ** it))
        step = 0.05 * (temp / initial_temp) + 0.01
        neighbor = np.maximum(0.001, current_w + np.random.normal(0, step, size=current_w.shape))
        obj = average_rank(neighbor, X, selected_idx)
        if obj < current_obj or (temp > 0 and np.random.rand() < np.exp((current_obj - obj) / temp)):
            current_w, current_obj = neighbor, obj
            if obj < best_obj:
                best_w, best_obj = neighbor.copy(), obj
    return best_w, best_obj


def multi_start_sa(X: np.ndarray, selected_idx: List[int], num_starts: int = 30) -> Tuple[np.ndarray, float]:
    n_features = X.shape[1]
    selected_means = X[selected_idx].mean(axis=0)
    global_means = X.mean(axis=0)
    advantage = np.clip(selected_means - global_means, a_min=0.0, a_max=None)
    analytical = advantage / advantage.sum() if advantage.sum() > 0 else np.ones(n_features) / n_features
    starts = [analytical, np.ones(n_features) / n_features]
    while len(starts) < num_starts:
        rnd = np.random.rand(n_features)
        rnd = np.maximum(0.01, rnd)
        rnd /= rnd.sum()
        starts.append(rnd)
    best_w, best_obj = None, float('inf')
    for w0 in starts:
        w, obj = run_sa(X, selected_idx, w0)
        if obj < best_obj:
            best_w, best_obj = w, obj
    assert best_w is not None
    return best_w / best_w.sum(), best_obj


def entropy_score(weights: np.ndarray, eps: float = 1e-9) -> float:
    w = weights / max(weights.sum(), eps)
    m = len(w)
    return float(-(w * np.log(w + eps)).sum() / np.log(m))  # normalized to [0,1]


@dataclass
class Candidate:
    indices: List[int]
    label: str
    entropy: float
    dominance: float
    avg_rank: float
    weights: np.ndarray


def plot_cluster(parcels: gpd.GeoDataFrame, idx: List[int], label: str) -> None:
    subs = parcels.iloc[idx]
    bounds = parcels.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.05
    pad_y = (bounds[3] - bounds[1]) * 0.05
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    parcels.boundary.plot(ax=ax, color='lightgray', linewidth=0.2)
    subs.plot(ax=ax, color='gold', edgecolor='black', linewidth=0.6, alpha=0.8)
    # Star markers at parcel centroids for this subset
    try:
        cent = subs.geometry.centroid
        ax.scatter(cent.x.to_numpy(), cent.y.to_numpy(), marker='*', s=60, color='red', edgecolors='black', linewidths=0.5, alpha=0.9, zorder=3)
    except Exception:
        pass
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax.set_title(f"Subset {label} — size={len(idx)}")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def main():
    print("SUBSET EXPLORER — contiguous ~20-parcel clusters")
    print("=" * 80)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    parcels = gpd.read_file(DATA_DIR / "parcels.shp")
    parcels["parcel_id"] = parcels["parcel_id"].astype(str)

    # Determine available criteria columns
    criteria_cols = CRITERIA_COLS_BASE + [c for c in EXTRA_OPTIONAL if c in parcels.columns or c == "wui_s"]
    parcels = ensure_features(parcels, criteria_cols)
    X = parcels[criteria_cols].astype(float).fillna(0.0).values

    print(f"Parcels: {len(parcels):,}; Using criteria: {criteria_cols}")
    print("Building contiguity...")
    neighbors = build_adjacency(parcels)
    print("Contiguity ready.")

    # Restrict to parcels with wui_s > threshold
    allowed_set = _make_allowed_set(parcels, WUI_ATTR_THRESHOLD)
    if not allowed_set:
        print(f"No parcels satisfy wui_s > {WUI_ATTR_THRESHOLD}. Adjust threshold or data.")
        return

    # Sample seeds only from allowed set and grow clusters within allowed
    allowed_list = sorted(list(allowed_set))
    # Bias a fraction toward parcels closest to the spatial centroid
    centroids = parcels.geometry.centroid
    cx, cy = centroids.x.to_numpy(), centroids.y.to_numpy()
    gx, gy = float(np.median(cx)), float(np.median(cy))
    dists = np.sqrt((cx - gx) ** 2 + (cy - gy) ** 2)
    allowed_d = dists[allowed_list]
    order = np.argsort(allowed_d)
    k_center = int(max(1, min(len(allowed_list), NUM_SEEDS * CENTER_SEED_FRACTION)))
    center_pool = [allowed_list[i] for i in order[:k_center]]
    rand_pool = list(set(allowed_list) - set(center_pool))
    num_center = min(k_center, int(NUM_SEEDS * CENTER_SEED_FRACTION))
    num_rand = max(0, min(len(rand_pool), NUM_SEEDS - num_center))
    seeds = random.sample(center_pool, min(num_center, len(center_pool))) + (
        random.sample(rand_pool, num_rand) if num_rand > 0 else []
    )
    labels = [chr(ord('A') + i) for i in range(26)]
    label_idx = 0

    if FAST_MAP_ONLY:
        found: List[Tuple[str, List[int]]] = []
        signatures: set[Tuple[int, ...]] = set()
        for si, seed in enumerate(seeds):
            cluster = grow_cluster(seed, neighbors, SIZE_MIN, SIZE_MAX, allowed=allowed_set)
            if not cluster or not (SIZE_MIN <= len(cluster) <= SIZE_MAX):
                continue
            signature = tuple(sorted(cluster))
            if signature in signatures:
                continue
            signatures.add(signature)
            label = labels[label_idx] if label_idx < len(labels) else f"Z{si}"
            label_idx += 1
            found.append((label, cluster))
            if len(found) >= MAP_COUNT:
                break
        if not found:
            print("No clusters found. Try raising NUM_SEEDS or widening SIZE_MIN/MAX.")
            return
        print(f"\nShowing {len(found)} clusters:")
        for label, idxs in found:
            print(f"  {label}: size={len(idxs)}")
            if SAVE_SUBSETS:
                try:
                    out = parcels.iloc[idxs].copy()
                    out['subset'] = label
                    out.to_file(SAVE_DIR / f"subset_{label}.shp")
                except Exception as e:
                    print(f"  (warn) failed to save subset {label}: {e}")
        for label, idxs in found:
            plot_cluster(parcels, idxs, label)
        return
    else:
        candidates: List[Candidate] = []
        for si, seed in enumerate(seeds):
            cluster = grow_cluster(seed, neighbors, SIZE_MIN, SIZE_MAX)
            if not cluster or not (SIZE_MIN <= len(cluster) <= SIZE_MAX):
                continue
            signature = tuple(sorted(cluster))
            if any(set(signature) == set(c.indices) for c in candidates):
                continue
            selected_idx = cluster
            w_sa, avg_rank_obj = multi_start_sa(X, selected_idx, num_starts=15)
            ent = entropy_score(w_sa)
            dom = float(np.max(w_sa / max(w_sa.sum(), 1e-9)))
            label = labels[label_idx] if label_idx < len(labels) else f"Z{si}"
            label_idx += 1
            candidates.append(Candidate(indices=cluster, label=label, entropy=ent, dominance=dom, avg_rank=avg_rank_obj, weights=w_sa))

        if not candidates:
            print("No candidate clusters found. Try increasing NUM_SEEDS or widening SIZE_MIN/MAX.")
            return

        def score(c: Candidate) -> float:
            return 0.7 * c.entropy - 0.2 * (c.dominance - 1.0 / len(c.weights)) - 0.1 * (c.avg_rank / max(len(parcels), 1))

        candidates.sort(key=score, reverse=True)
        top = candidates[:TOP_K]

        print("\nTop candidates:")
        for c in top:
            w = c.weights / c.weights.sum()
            top2_idx = np.argsort(w)[::-1][:2]
            crit_top2 = [(criteria_cols[i], float(w[i])) for i in top2_idx]
            print(f"  {c.label}: size={len(c.indices)} | entropy={c.entropy:.3f} | dominance={c.dominance:.2f} | avg_rank={c.avg_rank:.1f} | top2={crit_top2}")

        for c in top:
            plot_cluster(parcels, c.indices, c.label)


if __name__ == "__main__":
    main()


