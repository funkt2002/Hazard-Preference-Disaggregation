#!/usr/bin/env python3
"""
Simplified Subset-Based Simulated Annealing (SA) for Rank Minimization

- Loads `parcels.shp` and prefers `subset_1.shp` (fallback to `parcels_subset.shp`) from the local data directory
- Runs a compact multi-start SA to minimize the average rank of the subset
- Displays simple Matplotlib visuals in the integrated terminal (not saved)
"""

import time
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths and configuration
# -----------------------------------------------------------------------------
BASE_PATH = Path("/Users/theofunk/Desktop/NARSC paper")
DATA_DIR = BASE_PATH / "data"
CRITERIA_COLS = ["qtrmi_s", "hvhsz_s", "agfb_s", "hbrn_s", "slope_s"]
MAX_PARCELS: int | None = None  # set to an int to limit for quick tests


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_data() -> Tuple[np.ndarray, List[int], List[str]]:
    parcels = gpd.read_file(DATA_DIR / "parcels.shp")
    # Prefer new naming; fallback to legacy
    subset_path = DATA_DIR / "subset_1.shp"
    if not subset_path.exists():
        subset_path = DATA_DIR / "parcels_subset.shp"
    subset = gpd.read_file(subset_path)

    if isinstance(MAX_PARCELS, int) and MAX_PARCELS > 0:
        parcels = parcels.head(MAX_PARCELS).copy()

    parcels["parcel_id"] = parcels["parcel_id"].astype(str)
    subset_ids = set(subset["parcel_id"].astype(str))

    # Minimal feature hygiene for hvhsz_s
    if "hvhsz_s" not in parcels.columns and "hvhsz" in parcels.columns:
        vals = pd.to_numeric(parcels["hvhsz"], errors="coerce").fillna(0.0)
        vmin, vmax = float(vals.min()), float(vals.max())
        parcels["hvhsz_s"] = (vals - vmin) / (vmax - vmin) if vmax > vmin else 0.0

    # Ensure all criteria exist
    for col in CRITERIA_COLS:
        if col not in parcels.columns:
            parcels[col] = 0.0

    X = parcels[CRITERIA_COLS].astype(float).fillna(0.0).values
    parcel_ids = parcels["parcel_id"].tolist()
    selected_idx = [i for i, pid in enumerate(parcel_ids) if pid in subset_ids]

    if len(selected_idx) == 0:
        raise ValueError("No selected parcels found in subset (check parcel_id alignment)")

    return X, selected_idx, CRITERIA_COLS


# -----------------------------------------------------------------------------
# SA objective and optimization
# -----------------------------------------------------------------------------
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
    ranks[order] = np.arange(len(order)) + 1  # rank 1 is best
    return float(np.mean(ranks[selected_idx]))


def run_sa(X: np.ndarray, selected_idx: List[int], w0: np.ndarray, max_iters: int = 150) -> Tuple[np.ndarray, float, List[float]]:
    current_w = w0.copy()
    current_obj = average_rank(current_w, X, selected_idx)
    best_w = current_w.copy()
    best_obj = current_obj
    history: List[float] = []

    initial_temp = max(0.001, best_obj * 0.1)
    cooling = 0.95
    min_temp = 1e-3

    for it in range(max_iters):
        temp = max(min_temp, initial_temp * (cooling ** it))
        step = 0.05 * (temp / initial_temp) + 0.01
        neighbor = np.maximum(0.001, current_w + np.random.normal(0, step, size=current_w.shape))
        neigh_obj = average_rank(neighbor, X, selected_idx)

        if neigh_obj < current_obj:
            accept = True
        else:
            accept_prob = np.exp((current_obj - neigh_obj) / temp) if temp > 0 else 0.0
            accept = np.random.rand() < accept_prob

        if accept:
            current_w, current_obj = neighbor, neigh_obj
            if neigh_obj < best_obj:
                best_w, best_obj = neighbor.copy(), neigh_obj

        history.append(best_obj)

    return best_w, best_obj, history


def multi_start_sa(X: np.ndarray, selected_idx: List[int], num_starts: int = 50) -> Tuple[np.ndarray, float, List[List[float]]]:
    n_features = X.shape[1]

    # Analytical start: emphasize criteria where selected mean > global mean
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

    best_w, best_obj = None, float("inf")
    all_histories: List[List[float]] = []
    for i, w0 in enumerate(starts):
        w, obj, hist = run_sa(X, selected_idx, w0)
        all_histories.append(hist)
        if obj < best_obj:
            best_w, best_obj = w, obj
        if i < 5 or (i + 1) % 10 == 0:
            print(f"  SA run {i+1}/{num_starts}: best_obj={obj:.1f}")

    assert best_w is not None
    return best_w, best_obj, all_histories


# -----------------------------------------------------------------------------
# Visuals (Matplotlib)
# -----------------------------------------------------------------------------
def plot_weights(weights: np.ndarray, factors: List[str]) -> None:
    w = weights / weights.sum()
    plt.figure(figsize=(8, 4))
    plt.bar(factors, w, color="skyblue")
    plt.ylim(0, 1)
    plt.ylabel("Weight")
    plt.title("Optimized Weights (normalized)")
    plt.xticks(rotation=30)
    plt.tight_layout()


def plot_convergence(histories: List[List[float]]) -> None:
    plt.figure(figsize=(8, 4))
    for i, hist in enumerate(histories[:10]):  # show up to 10 runs
        plt.plot(hist, alpha=0.7, label=f"run {i+1}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Average Rank")
    plt.title("SA Convergence (best-so-far)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("SUBSET-BASED SA (SIMPLIFIED)")
    print("=" * 60)

    X, selected_idx, factors = load_data()
    print(f"Loaded {X.shape[0]} parcels, {X.shape[1]} features")
    print(f"Subset size: {len(selected_idx)}")

    start = time.time()
    best_w, best_obj, histories = multi_start_sa(X, selected_idx, num_starts=50)
    elapsed = time.time() - start

    print("\n--- Results ---")
    print("Optimized weights (normalized):")
    w = best_w / best_w.sum()
    for j, name in enumerate(factors):
        print(f"  {name:10s}: {w[j]:.3f}")
    print(f"Best average rank: {best_obj:.1f}")
    print(f"Runtime: {elapsed:.2f}s")

    # Visuals
    plot_weights(best_w, factors)
    plot_convergence(histories)
    plt.show()


if __name__ == "__main__":
    main()