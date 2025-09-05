#!/usr/bin/env python3
"""
Compare Linear UTA (from UTASTAR outputs) vs Simulated Annealing (SA)
for each subset shapefile in data/ (prefer subset_*.shp; fallback to parcels_subset*.shp).

Outputs:
- Console table with per-subset ranking metrics for Linear vs SA
- CSV extras/outputs/linear_vs_sa_summary.csv
- CSV extras/outputs/linear_vs_sa_weights.csv (per-subset weights per approach)

Notes:
- Linear UTA weights and rankings are read from the JSON produced by UTASTAR:
  extras/outputs/uta_star_comparison_<subset>.json
- SA is recomputed quickly here using the same criteria columns.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import json

import numpy as np
import pandas as pd
import geopandas as gpd


# Paths/config
BASE_PATH = Path("/Users/theofunk/Desktop/NARSC paper")
DATA_DIR = BASE_PATH / "data"
OUTPUTS_DIR = BASE_PATH / "extras" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

CRITERIA_COLS = ["qtrmi_s", "hvhsz_s", "agfb_s", "hbrn_s", "slope_s"]
HINGE_DELTA = 1e-3
HINGE_MAX_NON_SUBSET = 20000


def ensure_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "hvhsz_s" not in gdf.columns and "hvhsz" in gdf.columns:
        vals = pd.to_numeric(gdf["hvhsz"], errors="coerce").fillna(0.0)
        vmin, vmax = float(vals.min()), float(vals.max())
        gdf["hvhsz_s"] = (vals - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    for c in CRITERIA_COLS:
        if c not in gdf.columns:
            gdf[c] = 0.0
    return gdf


def evaluate_ranking(scores: np.ndarray, subset_idx: List[int]) -> Dict:
    ranks = 1 + np.argsort(np.argsort(-scores))
    subset_ranks = ranks[subset_idx]
    return {
        'average_rank': float(np.mean(subset_ranks)),
        'median_rank': float(np.median(subset_ranks)),
        'best_rank': int(np.min(subset_ranks)),
        'worst_rank': int(np.max(subset_ranks)),
        'top_500': int(np.sum(subset_ranks <= 500)),
        'top_1000': int(np.sum(subset_ranks <= 1000))
    }


def compute_hinge_metrics(scores: np.ndarray, subset_idx: List[int], non_subset_idx: List[int], delta: float = HINGE_DELTA) -> Dict[str, float]:
    """
    Compute pairwise hinge-loss metrics between subset items (p) and non-subset items (q):
      loss(p,q) = max(0, delta - (scores[p] - scores[q]))
    Returns: total_hinge, violations, avg_hinge, num_pairs
    """
    if not subset_idx or not non_subset_idx:
        return {'total_hinge': 0.0, 'violations': 0, 'avg_hinge': 0.0, 'num_pairs': 0}
    total = 0.0
    viols = 0
    for p in subset_idx:
        sp = scores[p]
        # vectorized over q for speed
        margins = sp - scores[non_subset_idx]
        losses = np.maximum(0.0, delta - margins)
        total += float(np.sum(losses))
        viols += int(np.count_nonzero(losses > 1e-12))
    num_pairs = len(subset_idx) * len(non_subset_idx)
    avg = total / num_pairs if num_pairs > 0 else 0.0
    return {'total_hinge': total, 'violations': viols, 'avg_hinge': avg, 'num_pairs': num_pairs}


def compute_linear_uta_weights(X: np.ndarray, subset_idx: List[int], non_subset_idx: List[int], delta: float = HINGE_DELTA) -> Tuple[np.ndarray | None, float | None, int | None]:
    """
    Fallback: solve UTASTAR-style linear weights by minimizing pairwise hinge slacks
    using gurobipy, mirroring the formulation in UTASTAR.py.
    Returns (weights, total_error, violations) or (None, None, None) if unavailable.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None, None, None

    n, m = X.shape
    model = gp.Model("UTA-Weights-Fallback")
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = 60

    w = {i: model.addVar(lb=0, ub=1, name=f"w_{i}") for i in range(m)}
    sigma_plus = {}
    sigma_minus = {}
    for p_idx in subset_idx:
        for q_idx in non_subset_idx:
            sigma_plus[p_idx, q_idx] = model.addVar(lb=0, name=f"sp_{p_idx}_{q_idx}")
            sigma_minus[p_idx, q_idx] = model.addVar(lb=0, name=f"sm_{p_idx}_{q_idx}")
    model.update()

    model.addConstr(gp.quicksum(w[i] for i in range(m)) == 1)
    for p_idx in subset_idx:
        for q_idx in non_subset_idx:
            score_p = gp.quicksum(X[p_idx, i] * w[i] for i in range(m))
            score_q = gp.quicksum(X[q_idx, i] * w[i] for i in range(m))
            model.addConstr(score_p - score_q >= delta - sigma_plus[p_idx, q_idx] + sigma_minus[p_idx, q_idx])

    obj = gp.quicksum(sigma_plus[key] + sigma_minus[key] for key in sigma_plus)
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        return None, None, None

    weights = np.array([w[i].X for i in range(m)], dtype=float)
    violations = sum(1 for key in sigma_plus if sigma_plus[key].X > 1e-6 or sigma_minus[key].X > 1e-6)
    total_error = float(model.ObjVal)
    return weights, total_error, violations


# SA optimization (compact)
def average_rank(weights: np.ndarray, X: np.ndarray, selected_idx: List[int]) -> float:
    if len(selected_idx) == 0:
        return 1e9
    w = np.array(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        return 1e9
    w /= s
    scores = X @ w
    ranks = 1 + np.argsort(np.argsort(-scores))
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


def multi_start_sa(X: np.ndarray, selected_idx: List[int], num_starts: int = 50) -> Tuple[np.ndarray, float]:
    n_features = X.shape[1]
    # Analytical start
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


def main():
    print("LINEAR UTA vs SA COMPARISON (by subset)")
    print("=" * 80)

    # Load parcels
    parcels = gpd.read_file(DATA_DIR / 'parcels.shp')
    parcels = ensure_features(parcels)
    parcels['parcel_id'] = parcels['parcel_id'].astype(str)
    X = parcels[CRITERIA_COLS].astype(float).values

    # Prefer new naming
    subset_files = sorted([p for p in DATA_DIR.glob('subset_*.shp')])
    if not subset_files:
        subset_files = sorted([p for p in DATA_DIR.glob('parcels_subset*.shp')])

    summary_rows = []
    weight_rows = []

    for subset_path in subset_files:
        subset_label = subset_path.stem
        print(f"\nSubset: {subset_label}")
        subset = gpd.read_file(subset_path)
        subset_ids = set(subset['parcel_id'].astype(str))
        parcel_ids = parcels['parcel_id'].tolist()
        subset_idx = [i for i, pid in enumerate(parcel_ids) if pid in subset_ids]
        if not subset_idx:
            print("  (no matching parcel_ids; skipping)")
            continue

        # Build non-subset index and optionally sample to cap pairs (match UTASTAR scale)
        non_subset_idx = [i for i, pid in enumerate(parcel_ids) if pid not in subset_ids]
        if len(non_subset_idx) > HINGE_MAX_NON_SUBSET:
            step = len(non_subset_idx) / HINGE_MAX_NON_SUBSET
            non_subset_idx = [non_subset_idx[int(i * step)] for i in range(HINGE_MAX_NON_SUBSET)]

        # Linear UTA from UTASTAR JSON (preferred), else fallback to direct solve with gurobipy
        json_path = OUTPUTS_DIR / f"uta_star_comparison_{subset_label}.json"
        linear_weights = None
        linear_rank = None
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                wvals = data.get('weights', {}).get('values')
                if wvals is not None:
                    linear_weights = np.array(wvals, dtype=float)
                linear_rank = data.get('weight_ranking')
            except Exception:
                pass

        if linear_weights is None:
            # Fallback: compute UTASTAR linear weights directly
            print("  JSON missing — solving UTASTAR linear weights directly (fallback)...")
            lw, lw_err, lw_viol = compute_linear_uta_weights(X, subset_idx, non_subset_idx, delta=HINGE_DELTA)
            if lw is None:
                print("  Fallback failed (likely missing gurobipy); skipping subset.")
                continue
            linear_weights = lw

        # SA optimization
        sa_weights, sa_obj = multi_start_sa(X, subset_idx, num_starts=50)

        # Rankings
        scores_linear = X @ (linear_weights / np.sum(linear_weights))
        rank_linear = evaluate_ranking(scores_linear, subset_idx)
        scores_sa = X @ sa_weights
        rank_sa = evaluate_ranking(scores_sa, subset_idx)

        print(f"  Linear avg rank: {rank_linear['average_rank']:.1f} | SA avg rank: {rank_sa['average_rank']:.1f}")

        summary_rows.append({
            'subset': subset_label,
            'linear_avg': rank_linear['average_rank'],
            'linear_min': rank_linear['best_rank'],
            'linear_max': rank_linear['worst_rank'],
            'linear_top500': rank_linear['top_500'],
            'sa_avg': rank_sa['average_rank'],
            'sa_min': rank_sa['best_rank'],
            'sa_max': rank_sa['worst_rank'],
            'sa_top500': rank_sa['top_500'],
        })

        # Per-criterion weights (normalized)
        lw = (linear_weights / np.sum(linear_weights)).tolist()
        sw = sa_weights.tolist()
        weight_rows.append({
            'subset': subset_label,
            'approach': 'linear',
            **{CRITERIA_COLS[i]: lw[i] for i in range(len(CRITERIA_COLS))}
        })
        weight_rows.append({
            'subset': subset_label,
            'approach': 'sa',
            **{CRITERIA_COLS[i]: sw[i] for i in range(len(CRITERIA_COLS))}
        })

        # Pairwise hinge-loss metrics (script evaluation set)
        hinge_linear = compute_hinge_metrics(scores_linear, subset_idx, non_subset_idx, delta=HINGE_DELTA)
        hinge_sa = compute_hinge_metrics(scores_sa, subset_idx, non_subset_idx, delta=HINGE_DELTA)

        print("  Hinge metrics (script pairs, delta=%.4g):" % HINGE_DELTA)
        print("    Linear (UTASTAR weights): violations=%d | total_hinge=%.4f | avg_hinge=%.6f | pairs=%d" %
              (hinge_linear['violations'], hinge_linear['total_hinge'], hinge_linear['avg_hinge'], hinge_linear['num_pairs']))
        print("    SA:                      violations=%d | total_hinge=%.4f | avg_hinge=%.6f | pairs=%d" %
              (hinge_sa['violations'], hinge_sa['total_hinge'], hinge_sa['avg_hinge'], hinge_sa['num_pairs']))

        # If JSON includes UTASTAR-reported totals, echo them for reference; else echo fallback solver totals
        try:
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                wmeta = data.get('weights', {})
                json_err = float(wmeta.get('total_error', 0.0))
                json_viol = int(wmeta.get('violations', 0))
                print("    UTASTAR JSON (weights):  violations=%d | total_error=%.4f (solver objective)" % (json_viol, json_err))
            else:
                # If we solved weights on the fly, also report those totals if we captured them
                if 'lw' in locals() and lw is not None and 'lw_err' in locals() and lw_err is not None:
                    print("    Fallback solve (weights): violations=%s | total_error=%.4f" % (str(lw_viol), float(lw_err)))
        except Exception:
            pass

    # Print and save tables
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
            print("\nSummary (rank metrics):")
            print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        out_csv = OUTPUTS_DIR / 'linear_vs_sa_summary.csv'
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

    if weight_rows:
        dfw = pd.DataFrame(weight_rows)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
            print("\nWeights (per subset):")
            print(dfw.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        out_wcsv = OUTPUTS_DIR / 'linear_vs_sa_weights.csv'
        dfw.to_csv(out_wcsv, index=False)
        print(f"Saved: {out_wcsv}")


if __name__ == "__main__":
    main()

