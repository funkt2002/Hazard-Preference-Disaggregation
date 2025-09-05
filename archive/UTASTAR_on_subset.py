#!/usr/bin/env python3
"""
Run UTA-STAR methods (piecewise utilities and linear weights) with identical
parameters on a small set of contiguous ~20-parcel subsets discovered on-the-fly.

Outputs (printed to terminal only):
- Table: Average Rank per subset (Piecewise)
- Table: Average Rank per subset (Linear Weights)
- Table: Comparison (Piecewise vs Linear averages)

Notes:
- No files are saved. Figures are not produced in this script.
- Requires gurobipy.
"""

from __future__ import annotations

import random
from pathlib import Path
import contextlib
import io
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    raise SystemExit("Error: gurobipy is required for UTASTAR_on_subset.py")


# Paths/config
BASE_PATH = Path("/Users/theofunk/Desktop/NARSC paper")
DATA_DIR = BASE_PATH / "data"

# Criteria (include optional WUI if present; prefer 'hwui_s')
CRITERIA_COLS_BASE = ["qtrmi_s", "hvhsz_s", "agfb_s", "hbrn_s", "slope_s"]
EXTRA_OPTIONAL = ["hwui_s"]

# Cluster finder params (match subset_explorer defaults)
SIZE_MIN = 18
SIZE_MAX = 22
NUM_SEEDS = 200
NUM_SUBSETS = 10
RANDOM_SEED = 42

# UTA-STAR params (match UTASTAR.py defaults)
ALPHA = 3
DELTA = 1e-3
TIME_LIMIT_SEC = 60
NON_SUBSET_SAMPLE_SIZE = 10000
WUI_ATTR_THRESHOLD = 0.2  # restrict analysis to parcels with hwui_s > threshold


def ensure_features(gdf: gpd.GeoDataFrame, criteria_cols: List[str]) -> gpd.GeoDataFrame:
    if "hvhsz_s" not in gdf.columns and "hvhsz" in gdf.columns:
        vals = pd.to_numeric(gdf["hvhsz"], errors="coerce").fillna(0.0)
        vmin, vmax = float(vals.min()), float(vals.max())
        gdf["hvhsz_s"] = (vals - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    if "hwui_s" not in gdf.columns:
        if "hwui" in gdf.columns:
            vals = pd.to_numeric(gdf["hwui"], errors="coerce").fillna(0.0)
            vmin, vmax = float(vals.min()), float(vals.max())
            gdf["hwui_s"] = (vals - vmin) / (vmax - vmin) if vmax > vmin else 0.0
        elif "hlfmi_wui" in gdf.columns:
            vals = pd.to_numeric(gdf["hlfmi_wui"], errors="coerce").fillna(0.0)
            vmax = float(vals.max()) if len(vals) else 0.0
            arr = (vals / 100.0) if vmax > 1.0 else vals
            gdf["hwui_s"] = np.clip(arr, 0.0, 1.0)
        else:
            gdf["hwui_s"] = 0.0
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
            cand = list(sindex.query(geom, predicate="touches"))
        except Exception:
            cand = list(sindex.query(geom))
            cand = [j for j in cand if i != j and geoms[j].touches(geom)]
        neighbors[i] = [j for j in cand if j != i]
    return neighbors


def grow_cluster(seed: int, neighbors: List[List[int]], size_min: int, size_max: int, allowed: Optional[Set[int]] = None) -> List[int] | None:
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
    if len(visited) > size_max:
        return list(sorted(list(visited)))[:size_max]
    return None


def precompute_interpolation(X: np.ndarray, alpha: int = ALPHA) -> Tuple[np.ndarray, np.ndarray]:
    n, m = X.shape
    breakpoints = np.linspace(0, 1, alpha + 1)
    W = np.zeros((n, m, alpha + 1))
    for a in range(n):
        for i in range(m):
            x_val = X[a, i]
            for k in range(alpha):
                if breakpoints[k] <= x_val <= breakpoints[k + 1]:
                    theta = (x_val - breakpoints[k]) / (breakpoints[k + 1] - breakpoints[k]) if (breakpoints[k + 1] - breakpoints[k]) > 1e-10 else 0.0
                    W[a, i, k] = 1 - theta
                    W[a, i, k + 1] = theta
                    break
    return W, breakpoints


def build_silent_env() -> gp.Env:
    # Suppress gurobi parameter echo by redirecting stdout during env setup
    with contextlib.redirect_stdout(io.StringIO()):
        env = gp.Env()
        env.setParam('LogToConsole', 0)
        env.setParam('OutputFlag', 0)
        env.setParam('LogFile', "")
        env.start()
    return env


def solve_uta_star_utilities(X: np.ndarray, subset_idx: List[int], non_subset_idx: List[int], env: gp.Env) -> Optional[Dict]:
    n, m = X.shape
    W, breakpoints = precompute_interpolation(X, ALPHA)
    model = gp.Model("UTA-STAR-Utilities", env=env)
    model.Params.TimeLimit = TIME_LIMIT_SEC
    u = {(i, k): model.addVar(lb=0, ub=1, name=f"u_{i}_{k}") for i in range(m) for k in range(ALPHA + 1)}
    sp = {(p, q): model.addVar(lb=0, name=f"sp_{p}_{q}") for p in subset_idx for q in non_subset_idx}
    sm = {(p, q): model.addVar(lb=0, name=f"sm_{p}_{q}") for p in subset_idx for q in non_subset_idx}
    model.update()
    for i in range(m):
        model.addConstr(u[(i, 0)] == 0)
    model.addConstr(gp.quicksum(u[(i, ALPHA)] for i in range(m)) == 1)
    for i in range(m):
        for k in range(ALPHA):
            model.addConstr(u[(i, k + 1)] >= u[(i, k)])
    for p in subset_idx:
        for q in non_subset_idx:
            Up = gp.quicksum(W[p, i, k] * u[(i, k)] for i in range(m) for k in range(ALPHA + 1))
            Uq = gp.quicksum(W[q, i, k] * u[(i, k)] for i in range(m) for k in range(ALPHA + 1))
            model.addConstr(Up - Uq >= DELTA - sp[(p, q)] + sm[(p, q)])
    obj = gp.quicksum(sp[key] + sm[key] for key in sp)
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return None
    utilities = {str(i): [u[(i, k)].X for k in range(ALPHA + 1)] for i in range(m)}
    violations = sum(1 for key in sp if sp[key].X > 1e-6 or sm[key].X > 1e-6)
    return {
        "utilities": utilities,
        "total_error": float(model.ObjVal),
        "violations": violations,
        "breakpoints": np.linspace(0, 1, ALPHA + 1).tolist(),
        "alpha": ALPHA,
    }


def solve_uta_star_weights(X: np.ndarray, subset_idx: List[int], non_subset_idx: List[int], env: gp.Env) -> Optional[Dict]:
    n, m = X.shape
    model = gp.Model("UTA-STAR-Weights", env=env)
    model.Params.TimeLimit = TIME_LIMIT_SEC
    w = {i: model.addVar(lb=0, ub=1, name=f"w_{i}") for i in range(m)}
    sp = {(p, q): model.addVar(lb=0, name=f"sp_{p}_{q}") for p in subset_idx for q in non_subset_idx}
    sm = {(p, q): model.addVar(lb=0, name=f"sm_{p}_{q}") for p in subset_idx for q in non_subset_idx}
    model.update()
    model.addConstr(gp.quicksum(w[i] for i in range(m)) == 1)
    for p in subset_idx:
        for q in non_subset_idx:
            score_p = gp.quicksum(X[p, i] * w[i] for i in range(m))
            score_q = gp.quicksum(X[q, i] * w[i] for i in range(m))
            model.addConstr(score_p - score_q >= DELTA - sp[(p, q)] + sm[(p, q)])
    obj = gp.quicksum(sp[key] + sm[key] for key in sp)
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return None
    weights = np.array([w[i].X for i in range(m)], dtype=float)
    violations = sum(1 for key in sp if sp[key].X > 1e-6 or sm[key].X > 1e-6)
    return {
        "weights": weights,
        "total_error": float(model.ObjVal),
        "violations": violations,
    }


def compute_utility_scores(X: np.ndarray, utilities: Dict[str, List[float]], breakpoints: List[float], alpha: int) -> np.ndarray:
    n, m = X.shape
    scores = np.zeros(n)
    bps = np.array(breakpoints, dtype=float)
    for a in range(n):
        s = 0.0
        for i in range(m):
            x_val = X[a, i]
            for k in range(alpha):
                if bps[k] <= x_val <= bps[k + 1]:
                    theta = (x_val - bps[k]) / (bps[k + 1] - bps[k]) if (bps[k + 1] - bps[k]) > 1e-10 else 0.0
                    u_vals = utilities[str(i)]
                    s += (1 - theta) * u_vals[k] + theta * u_vals[k + 1]
                    break
        scores[a] = s
    return scores


def evaluate_ranking(scores: np.ndarray, subset_idx: List[int]) -> Dict:
    ranks = 1 + np.argsort(np.argsort(-scores))
    subset_ranks = ranks[subset_idx]
    return {
        "average_rank": float(np.mean(subset_ranks)),
        "median_rank": float(np.median(subset_ranks)),
        "best_rank": int(np.min(subset_ranks)),
        "worst_rank": int(np.max(subset_ranks)),
        "top_500": int(np.sum(subset_ranks <= 500)),
        "top_1000": int(np.sum(subset_ranks <= 1000)),
    }


def derived_weights_from_utilities(util: Dict[str, List[float]], criteria_cols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, c in enumerate(criteria_cols):
        vals = util.get(str(i), [0.0])
        out[c] = float(vals[-1]) if len(vals) > 0 else 0.0
    return out


def format_weights_text(weights_map: Dict[str, float], max_cols: int = 5) -> str:
    items = sorted(weights_map.items())
    lines: List[str] = []
    line = []
    for k, v in items:
        line.append(f"{k}:{v:.2f}")
        if len(line) >= max_cols:
            lines.append("  ".join(line))
            line = []
    if line:
        lines.append("  ".join(line))
    return "\n".join(lines)


def format_utilities_table(utilities: Dict[str, List[float]], criteria_cols: List[str], breakpoints: List[float]) -> str:
    """Create a compact text table of utility values per criterion at each breakpoint.
    Rows: criteria; Columns: BP0..BPk. Values formatted to 3 decimals.
    """
    # Determine number of breakpoints from first entry
    if not utilities:
        return "(no utilities)"
    num_bps = len(next(iter(utilities.values())))
    header = "    " + "  ".join([f"BP{i}" for i in range(num_bps)])
    lines: List[str] = [header]
    for i, name in enumerate(criteria_cols):
        vals = utilities.get(str(i))
        if vals is None:
            continue
        row = f"{name:8s} " + "  ".join(f"{v:0.3f}" for v in vals)
        lines.append(row)
    return "\n".join(lines)


def plot_score_map(parcels: gpd.GeoDataFrame, scores: np.ndarray, subset_idx: List[int], label: str, title: str, annotation: str, ranking: Dict | None = None, star_size: int = 90) -> None:
    gdf = parcels.copy()
    gdf['score'] = scores
    ranks = 1 + np.argsort(np.argsort(-scores))
    top500_mask = ranks <= 500
    bounds = gdf.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.05
    pad_y = (bounds[3] - bounds[1]) * 0.05
    # Wider aspect to emphasize map area
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    vmin, vmax = float(gdf['score'].min()), float(gdf['score'].max())
    gdf.plot(
        column='score', cmap='Reds', vmin=vmin, vmax=vmax, ax=ax,
        linewidth=0.0, edgecolor=None, legend=True,
        legend_kwds={'shrink': 0.6, 'fraction': 0.03, 'pad': 0.01, 'orientation': 'horizontal'}
    )
    # Blue outline top 500
    gdf[top500_mask].boundary.plot(ax=ax, color='blue', linewidth=0.6)
    # Gold stars on subset
    subs = gdf.iloc[subset_idx]
    try:
        cent = subs.geometry.centroid
        ax.scatter(cent.x.to_numpy(), cent.y.to_numpy(), marker='*', s=star_size, color='gold', edgecolors='black', linewidths=0.4, alpha=0.9, zorder=3)
    except Exception:
        pass
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax.set_title(f"{title} — {label}")
    ax.set_axis_off()
    # Annotation text box (append ranking summary if provided)
    ann_text = annotation
    if ranking is not None:
        try:
            ann_text += "\n" + f"Avg:{ranking.get('average_rank', float('nan')):.1f}  Min:{ranking.get('best_rank','?')}  Max:{ranking.get('worst_rank','?')}  Top500:{ranking.get('top_500','?')}"
        except Exception:
            pass
    ax.text(0.01, 0.01, ann_text, transform=ax.transAxes, fontsize=8, color='black',
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.08)
    plt.tight_layout()
    plt.show()


def main():
    print("UTASTAR ON SUBSETS — piecewise vs linear (average rank)")
    print("=" * 80)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    parcels = gpd.read_file(DATA_DIR / "parcels.shp")
    parcels["parcel_id"] = parcels["parcel_id"].astype(str)
    criteria_cols = CRITERIA_COLS_BASE + [c for c in EXTRA_OPTIONAL if c in parcels.columns or c == "hwui_s"]
    parcels = ensure_features(parcels, criteria_cols)
    X = parcels[criteria_cols].astype(float).fillna(0.0).values

    # Build contiguity and harvest first NUM_SUBSETS clusters
    neighbors = build_adjacency(parcels)
    # Build allowed set for WUI membership (subsets must be within WUI, but scoring uses all parcels)
    if "hwui_s" in parcels.columns:
        wui_vals = pd.to_numeric(parcels["hwui_s"], errors="coerce").fillna(0.0).to_numpy()
        allowed_set: Set[int] = {i for i, v in enumerate(wui_vals) if v > WUI_ATTR_THRESHOLD}
        if not allowed_set:
            print(f"No parcels satisfy hwui_s > {WUI_ATTR_THRESHOLD}")
            return
        seed_pool = sorted(list(allowed_set))
    else:
        print("Warning: hwui_s not found; proceeding without WUI restriction for subsets")
        allowed_set = None
        seed_pool = list(range(len(parcels)))
    seeds = random.sample(seed_pool, min(NUM_SEEDS, len(seed_pool)))
    labels = [chr(ord('A') + i) for i in range(26)]
    found: List[Tuple[str, List[int]]] = []
    seen: set[Tuple[int, ...]] = set()
    li = 0
    for si, seed in enumerate(seeds):
        cluster = grow_cluster(seed, neighbors, SIZE_MIN, SIZE_MAX, allowed=allowed_set)
        if not cluster or not (SIZE_MIN <= len(cluster) <= SIZE_MAX):
            continue
        sig = tuple(sorted(cluster))
        if sig in seen:
            continue
        seen.add(sig)
        label = labels[li] if li < len(labels) else f"Z{si}"
        li += 1
        found.append((label, cluster))
        if len(found) >= NUM_SUBSETS:
            break

    if not found:
        print("No subsets discovered. Increase NUM_SEEDS or widen size bounds.")
        return

    # Silent Gurobi environment
    env = build_silent_env()

    # Run UTASTAR on each subset
    results_piece: List[Dict] = []
    results_linear: List[Dict] = []

    for label, idx in found:
        subset_idx = idx
        universe = np.arange(len(parcels))
        non_subset_idx = [int(i) for i in universe if int(i) not in subset_idx]
        # Sample non-subset to cap size
        if len(non_subset_idx) > NON_SUBSET_SAMPLE_SIZE:
            step = len(non_subset_idx) / NON_SUBSET_SAMPLE_SIZE
            non_subset_idx = [non_subset_idx[int(i * step)] for i in range(NON_SUBSET_SAMPLE_SIZE)]

        util = solve_uta_star_utilities(X, subset_idx, non_subset_idx, env)
        wres = solve_uta_star_weights(X, subset_idx, non_subset_idx, env)

        if util is not None:
            scores_u = compute_utility_scores(X, util["utilities"], util["breakpoints"], ALPHA)
            r_u = evaluate_ranking(scores_u, subset_idx)
            results_piece.append({"subset": label, "avg": r_u["average_rank"], "min": r_u["best_rank"], "max": r_u["worst_rank"]})
            # Map: piecewise
            dw = derived_weights_from_utilities(util["utilities"], criteria_cols)
            util_table = format_utilities_table(util["utilities"], criteria_cols, util["breakpoints"])  # small table
            annot = "Piecewise (u@1):\n" + format_weights_text(dw) + "\n\n" + util_table
            plot_score_map(parcels, scores_u, subset_idx, label, "Piecewise Utility Score", annot, ranking=r_u, star_size=110)
        else:
            results_piece.append({"subset": label, "avg": np.nan, "min": np.nan, "max": np.nan})

        if wres is not None:
            scores_w = X @ wres["weights"]
            r_w = evaluate_ranking(scores_w, subset_idx)
            results_linear.append({"subset": label, "avg": r_w["average_rank"], "min": r_w["best_rank"], "max": r_w["worst_rank"]})
            # Map: linear
            wmap = {criteria_cols[i]: float(wres["weights"][i]) for i in range(len(criteria_cols))}
            annot = "Linear Weights:\n" + format_weights_text(wmap)
            plot_score_map(parcels, scores_w, subset_idx, label, "Linear Weighted Score", annot, ranking=r_w, star_size=110)
        else:
            results_linear.append({"subset": label, "avg": np.nan, "min": np.nan, "max": np.nan})

    # Build tables
    df_piece = pd.DataFrame(results_piece).sort_values("subset").reset_index(drop=True)
    df_linear = pd.DataFrame(results_linear).sort_values("subset").reset_index(drop=True)
    comp = pd.merge(df_piece, df_linear, on="subset", suffixes=("_piecewise", "_linear"))
    comp["improvement_pct"] = (comp["avg_linear"] - comp["avg_piecewise"]) / comp["avg_linear"] * 100.0

    # Print tables
    print("\nTest — Piecewise UTA-STAR (Average Rank of Subset)")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 160):
        print(df_piece[["subset", "avg"]].to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\nTest — Linear Weights (UTA-STAR) (Average Rank of Subset)")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 160):
        print(df_linear[["subset", "avg"]].to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\nComparison — Piecewise vs Linear (Average Rank)")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 160):
        print(comp[["subset", "avg_piecewise", "avg_linear", "improvement_pct"]].to_string(index=False, float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()


