#!/usr/bin/env python3
"""
Top-K UTA-STAR for Mission Ridge (Mission Canyon/Montrose) on WUI parcels

Formulation: threshold-based top-K membership with piecewise additive marginals.
Guarantees: All selected Mission Ridge parcels are members of the top K (K=500)
for at least one additive, monotone utility function consistent with the data.

Variables
- u[i,k]: utility at breakpoint k for criterion i (monotone, sum of maxima == 1, u[i,0]=0)
- y[j] in {0,1}: top-K membership indicator for parcel j
- tau in [0,1]: top-K threshold utility
- eps >= 0: safety margin between members and non-members

Linking constraints for each parcel j:
  U_j >= tau + eps - M*(1 - y[j])
  U_j <= tau - eps + M*y[j]
with M = 1 since utilities are normalized to [0,1]. Sum_j y[j] <= K and y[j]=1 for selected j.

Displays utility plots and a choropleth score map in the terminal (no files saved).
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Configure Gurobi license (same path convention as other scripts)
os.environ["GRB_LICENSE_FILE"] = "/Users/theofunk/Desktop/Narsc Paper/gurobi.lic"

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as e:
    raise SystemExit("Error: Gurobi (gurobipy) is required. Install and ensure a valid license.")


# -----------------------------------------------------------------------------
# Paths and configuration
# -----------------------------------------------------------------------------

BASE_PATH = Path("/Users/theofunk/Desktop/Narsc Paper")
DATA_DIR = BASE_PATH / "data" / "demo_data"

# Dataset: WUI parcels (quantile normalized)
WUI_DATASET_PATH = DATA_DIR / "dataset" / "fire_risk_local_quantile_wui_parcels" / "fire_risk_local_quantile_wui_parcels.shp"

# Mission Ridge subset (Mission Canyon/Montrose selection)
MISSION_SELECTION_PATH = DATA_DIR / "selections" / "mission_canyon_montrose_quantile" / "fire_risk_local_quantile_spatial_parcels.shp"

# Criteria consistent with your demo scripts (quantile/WUI)
CRITERIA_COLS = ['agfb_s', 'hfbfm_s', 'qtrmi_s', 'slope_s', 'hvhsz_s', 'travel_s']

# UTA piecewise segmentation
ALPHA = 4  # segments → breakpoints = ALPHA + 1

# Top-K parameter
TOP_K = 500


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_breakpoints_for_alpha(alpha: int) -> np.ndarray:
    if alpha == 1:
        return np.array([0.0, 1.0])
    elif alpha == 2:
        return np.array([0.0, 0.5, 1.0])
    elif alpha == 3:
        return np.array([0.0, 1.0/3.0, 2.0/3.0, 1.0])
    elif alpha == 4:
        return np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        return np.linspace(0, 1, alpha + 1)


def precompute_interpolation(X: np.ndarray, alpha: int) -> Tuple[np.ndarray, np.ndarray]:
    n, m = X.shape
    breakpoints = get_breakpoints_for_alpha(alpha)
    W = np.zeros((n, m, alpha + 1))
    for a in range(n):
        for i in range(m):
            x_val = X[a, i]
            for k in range(alpha):
                if breakpoints[k] <= x_val <= breakpoints[k+1]:
                    dx = breakpoints[k+1] - breakpoints[k]
                    theta = (x_val - breakpoints[k]) / dx if dx > 1e-10 else 0.0
                    W[a, i, k] = 1 - theta
                    W[a, i, k+1] = theta
                    break
    return W, breakpoints


def load_wui_and_selection() -> Tuple[gpd.GeoDataFrame, List[int], List[str]]:
    print("Loading WUI dataset and Mission Ridge selection...")
    parcels = gpd.read_file(WUI_DATASET_PATH)
    print(f"  WUI parcels loaded: {len(parcels):,}")

    # Ensure required criteria columns exist
    missing_cols = [c for c in CRITERIA_COLS if c not in parcels.columns]
    if missing_cols:
        print(f"  Warning: missing columns {missing_cols}; filling with 0.0")
        for c in missing_cols:
            parcels[c] = 0.0

    # Normalize criteria 0..1 on this dataset
    X = parcels[CRITERIA_COLS].fillna(0).values.copy()
    for i, col in enumerate(CRITERIA_COLS):
        v = X[:, i]
        vmin, vmax = float(np.min(v)), float(np.max(v))
        if vmax > vmin:
            X[:, i] = (v - vmin) / (vmax - vmin)
            print(f"    {col}: [{vmin:.3f}, {vmax:.3f}] -> [0,1]")
        else:
            print(f"    {col}: constant; left as-is")

    # Map Mission selection to parcel indices via parcel_id
    try:
        sel = gpd.read_file(MISSION_SELECTION_PATH)
        sel_ids = set(sel['parcel_id'].astype(str))
    except Exception as e:
        raise SystemExit(f"Error loading Mission Ridge selection: {e}")

    parcel_ids = parcels['parcel_id'].astype(str).tolist()
    selected_idx = [i for i, pid in enumerate(parcel_ids) if pid in sel_ids]
    print(f"  Mission Ridge parcels found in WUI dataset: {len(selected_idx)}")
    if len(selected_idx) == 0:
        print("  Warning: No Mission Ridge parcels found in this dataset.")

    return parcels, selected_idx, parcel_ids


def solve_topk_uta(X: np.ndarray, selected_idx: List[int], alpha: int = ALPHA, K: int = TOP_K) -> Optional[Dict]:
    """
    Solve threshold-based top-K UTA with piecewise utilities ensuring selected_idx ⊆ top-K.
    Maximizes eps (safety margin).
    Returns utilities, breakpoints, y membership, tau, eps if feasible.
    """
    n, m = X.shape
    W, breakpoints = precompute_interpolation(X, alpha)

    model = gp.Model("TopK-UTA")
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = 300  # seconds

    # Utility variables at breakpoints
    u = {}
    for i in range(m):
        for k in range(alpha + 1):
            u[i, k] = model.addVar(lb=0.0, ub=1.0, name=f"u_{i}_{k}")

    # Membership binaries
    y = {j: model.addVar(vtype=GRB.BINARY, name=f"y_{j}") for j in range(n)}

    # Threshold and safety margin
    tau = model.addVar(lb=0.0, ub=1.0, name="tau")
    eps = model.addVar(lb=0.0, ub=1.0, name="eps")

    model.update()

    # Constraints: normalization and monotonicity
    for i in range(m):
        model.addConstr(u[i, 0] == 0.0)
        for k in range(alpha):
            model.addConstr(u[i, k+1] >= u[i, k])
    model.addConstr(gp.quicksum(u[i, alpha] for i in range(m)) == 1.0)

    # Precompute parcel utility linear expressions U_j
    U_expr: Dict[int, gp.LinExpr] = {}
    for j in range(n):
        U_expr[j] = gp.quicksum(W[j, i, k] * u[i, k] for i in range(m) for k in range(alpha + 1))

    # Top-K linking constraints with M = 1 (utilities are in [0,1])
    M = 1.0
    for j in range(n):
        # If y_j=1 → U_j >= tau + eps
        model.addConstr(U_expr[j] >= tau + eps - M * (1 - y[j]))
        # If y_j=0 → U_j <= tau - eps
        model.addConstr(U_expr[j] <= tau - eps + M * y[j])

    # Cardinality: at most K members
    model.addConstr(gp.quicksum(y[j] for j in range(n)) <= K)

    # Force membership for all selected parcels
    for j in selected_idx:
        model.addConstr(y[j] == 1)

    # Objective: maximize eps (robust separation between in/out sets)
    model.setObjective(eps, GRB.MAXIMIZE)

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        print("Optimization did not find a feasible solution.")
        return None

    # Extract solution
    utilities = {str(i): [u[i, k].X for k in range(alpha + 1)] for i in range(m)}
    y_vals = np.array([int(round(y[j].X)) for j in range(n)], dtype=int)
    tau_val = float(tau.X)
    eps_val = float(eps.X)

    return {
        'utilities': utilities,
        'breakpoints': breakpoints.tolist(),
        'alpha': alpha,
        'y': y_vals,
        'tau': tau_val,
        'eps': eps_val,
    }


    


def compute_scores(X: np.ndarray, utilities: Dict[str, List[float]], breakpoints: np.ndarray, alpha: int) -> np.ndarray:
    n, m = X.shape
    scores = np.zeros(n)
    for a in range(n):
        total = 0.0
        for i in range(m):
            x_val = X[a, i]
            for k in range(alpha):
                if breakpoints[k] <= x_val <= breakpoints[k+1]:
                    dx = breakpoints[k+1] - breakpoints[k]
                    theta = (x_val - breakpoints[k]) / dx if dx > 1e-10 else 0.0
                    u0 = utilities[str(i)][k]
                    u1 = utilities[str(i)][k+1]
                    total += (1 - theta) * u0 + theta * u1
                    break
        scores[a] = total
    return scores


def plot_utility_functions_terminal(utility_result: Dict) -> None:
    if not utility_result:
        return
    bps = utility_result['breakpoints']
    alpha = int(utility_result['alpha'])
    num_criteria = len(CRITERIA_COLS)
    cols = min(3, num_criteria)
    rows = int(np.ceil(num_criteria / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()
    for i, col in enumerate(CRITERIA_COLS):
        ax = axes[i]
        utils = utility_result['utilities'][str(i)]
        for x in bps:
            ax.axvline(x, color='lightgray', linestyle='--', linewidth=1)
        ax.plot(bps, utils, marker='o', color='cyan', label='utility')
        ax.plot([bps[0], bps[-1]], [utils[0], utils[-1]], color='gold', linestyle='--', label='linear ref')
        ax.set_title(col)
        ax.set_xlabel("criterion (0..1)")
        ax.set_ylabel("utility")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.4)
        ax.legend()
    for j in range(num_criteria, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show(block=True)


def generate_score_map(parcels: gpd.GeoDataFrame, scores: np.ndarray, selected_idx: List[int], parcel_ids: List[str]) -> None:
    try:
        fig, ax = plt.subplots(figsize=(16, 12))
        score_dict = {pid: sc for pid, sc in zip(parcel_ids, scores)}
        gdf = parcels.copy()
        gdf['score'] = gdf['parcel_id'].astype(str).map(score_dict).fillna(0)
        gdf['rank'] = gdf['score'].rank(method='min', ascending=False).astype(int)
        top500_mask = gdf['rank'] <= 500
        subset_ids = set(parcel_ids[i] for i in selected_idx)
        gdf['is_subset'] = gdf['parcel_id'].astype(str).isin(subset_ids)

        cmap = LinearSegmentedColormap.from_list('score_cmap', ['white', '#ff6666', '#cc0000', '#660000'])
        norm = Normalize(vmin=0.2, vmax=0.8)
        gdf.plot(column='score', ax=ax, cmap=cmap, norm=norm, edgecolor='none', linewidth=0)
        if top500_mask.any():
            gdf[top500_mask].plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.3, alpha=0.9)

        # Mark subset centroid with a star
        try:
            sub_geoms = gdf.loc[gdf['is_subset'], 'geometry']
            if len(sub_geoms) > 0:
                centroid = sub_geoms.unary_union.centroid
                ax.scatter([centroid.x], [centroid.y], marker='*', s=120, color='yellow', edgecolors='black', linewidths=0.7, zorder=10)
        except Exception:
            pass

        ax.set_axis_off()
        ax.set_frame_on(False)
        bounds = gdf.total_bounds
        margin = 0.02 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        ax.margins(0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"Error generating map: {e}")


def main() -> None:
    print("="*80)
    print("TOP-K UTA-STAR: Mission Ridge on WUI Parcels (K=500)")
    print("="*80)

    # Load data
    parcels, selected_idx, parcel_ids = load_wui_and_selection()
    X = parcels[CRITERIA_COLS].fillna(0).values
    # Re-normalize to 0..1 just to ensure consistency if values changed
    for i in range(X.shape[1]):
        vmin, vmax = float(np.min(X[:, i])), float(np.max(X[:, i]))
        if vmax > vmin:
            X[:, i] = (X[:, i] - vmin) / (vmax - vmin)

    # Solve top-K UTA (piecewise)
    result = solve_topk_uta(X, selected_idx, alpha=ALPHA, K=TOP_K)
    if not result:
        return

    # Report
    y = result['y']
    tau = result['tau']
    eps = result['eps']
    print(f"\nSolution: eps={eps:.4f}, tau={tau:.4f}, |topK|={int(y.sum())}")
    in_top_selected = sum(y[j] == 1 for j in selected_idx)
    print(f"  Selected in top-K: {in_top_selected}/{len(selected_idx)}")

    # Compute scores and ranking to display
    scores = compute_scores(X, result['utilities'], np.array(result['breakpoints']), result['alpha'])
    ranks = 1 + np.argsort(np.argsort(-scores))
    subset_ranks = ranks[selected_idx] if len(selected_idx) else np.array([])
    if len(subset_ranks):
        print(f"  Avg rank (selected): {float(np.mean(subset_ranks)):.1f}")
        print(f"  Min/Max rank (selected): {int(np.min(subset_ranks))}/{int(np.max(subset_ranks))}")
        print(f"  Selected in top 500 by rank: {int(np.sum(subset_ranks <= 500))}/{len(selected_idx)}")

    # (Reverted) No range analysis or linear-weights variant in this simpler version

    # Show utility plots
    print("\nRendering utility functions...")
    plot_utility_functions_terminal(result)

    # Show score map
    print("Generating score map...")
    generate_score_map(parcels, scores, selected_idx, parcel_ids)


if __name__ == "__main__":
    main()


