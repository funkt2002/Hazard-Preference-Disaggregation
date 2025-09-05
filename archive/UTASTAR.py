#!/usr/bin/env python3
"""
UTA-STAR implementation with two approaches:
1. Piecewise linear utility functions (standard UTA-STAR)
2. Linear weights using same preference framework
Compares ranking performance of both approaches.
"""

import os
import sys
import time as time_module
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import matplotlib.pyplot as plt

try:
    import gurobipy as gp
    from gurobipy import GRB
    print("Using Gurobi solver for optimization")
except ImportError:
    print("Error: Gurobi (gurobipy) is required for this script. Please install and ensure a valid license.")
    sys.exit(1)

# Path configuration
BASE_PATH = Path("/Users/theofunk/Desktop/NARSC paper")
DATA_DIR = BASE_PATH / "data"
OUTPUTS_DIR = BASE_PATH / "extras" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Criteria columns (include optional WUI if present; prefer 'hwui_s')
CRITERIA_COLS = ['qtrmi_s', 'hvhsz_s', 'agfb_s', 'hbrn_s', 'slope_s', 'hwui_s']
# Candidate numbers of segments (alpha) for piecewise utilities
# Note: alpha = segments; interior breakpoints = alpha - 1
# "1 breakpoint" corresponds to alpha=2; up to 5 breakpoints → alpha=6
ALPHA_LIST: list[int] = [2, 3, 4, 5, 6]

# -----------------------------------------------------------------------------
# SECTION: Data Loading
# -----------------------------------------------------------------------------

def load_data() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load parcels and subset data."""
    print("Loading data...")
    parcels = gpd.read_file(DATA_DIR / "parcels.shp")
    # Prefer new naming for a single default subset; fallback to legacy
    subset_path = None
    try:
        # Use subset_1.shp if present as the default target subset
        cand = DATA_DIR / "subset_1.shp"
        subset_path = cand if cand.exists() else (DATA_DIR / "parcels_subset.shp")
    except Exception:
        subset_path = DATA_DIR / "parcels_subset.shp"
    subset = gpd.read_file(subset_path)
    
    print(f"  Full dataset: {len(parcels):,} parcels")
    print(f"  Target subset: {len(subset):,} parcels")
    
    return parcels, subset

def prepare_data(parcels: gpd.GeoDataFrame, subset: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """
    Prepare data for UTA-STAR analysis.
    Returns: X matrix, normalized X, parcel IDs, subset indices, non-subset indices
    """
    # Extract subset IDs
    subset_ids = set(subset['parcel_id'].astype(str))
    
    # Use all parcels
    working_parcels = parcels.copy()
    print(f"  Using all {len(working_parcels):,} parcels")

    # Ensure hvhsz_s exists; if missing but raw hvhsz exists, create min-max normalized hvhsz_s
    if 'hvhsz_s' not in working_parcels.columns and 'hvhsz' in working_parcels.columns:
        vals = working_parcels['hvhsz'].astype(float)
        vmin = vals.min()
        vmax = vals.max()
        if pd.notna(vmin) and pd.notna(vmax) and vmax > vmin:
            working_parcels['hvhsz_s'] = (vals - vmin) / (vmax - vmin)
        else:
            # Fallback to zeros if normalization is not possible
            working_parcels['hvhsz_s'] = 0.0
    # Ensure hwui_s exists; if missing, derive from 'hwui' or scaled 'hlfmi_wui'
    if 'hwui_s' not in working_parcels.columns:
        try:
            if 'hwui' in working_parcels.columns:
                vals = pd.to_numeric(working_parcels['hwui'], errors='coerce').fillna(0.0).astype(float)
                vmin = float(vals.min())
                vmax = float(vals.max())
                working_parcels['hwui_s'] = (vals - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            elif 'hlfmi_wui' in working_parcels.columns:
                vals = pd.to_numeric(working_parcels['hlfmi_wui'], errors='coerce').fillna(0.0).astype(float)
                # If values look like 0..100, scale to 0..1
                vmax = float(vals.max()) if len(vals) else 0.0
                vals = vals / 100.0 if vmax > 1.0 else vals
                vals = np.clip(vals, 0.0, 1.0)
                working_parcels['hwui_s'] = vals
            else:
                working_parcels['hwui_s'] = 0.0
        except Exception:
            working_parcels['hwui_s'] = 0.0
    
    # Guarantee all criteria columns exist
    for c in CRITERIA_COLS:
        if c not in working_parcels.columns:
            working_parcels[c] = 0.0
    # Guarantee all criteria columns exist
    for c in CRITERIA_COLS:
        if c not in working_parcels.columns:
            working_parcels[c] = 0.0
    # Extract data matrix
    X = working_parcels[CRITERIA_COLS].values
    parcel_ids = working_parcels['parcel_id'].astype(str).tolist()
    
    # Get indices
    subset_idx = [i for i, pid in enumerate(parcel_ids) if pid in subset_ids]
    non_subset_mask = np.ones(len(parcel_ids), dtype=bool)
    non_subset_mask[subset_idx] = False
    non_subset_idx = np.where(non_subset_mask)[0]
    
    # Sample non-subset for efficiency
    sample_size = 10000
    if len(non_subset_idx) > sample_size:
        print(f"  Sampling {sample_size:,} non-subset parcels for comparison")
        np.random.seed(42)
        step = len(non_subset_idx) / sample_size
        sampled_indices = [non_subset_idx[int(i * step)] for i in range(sample_size)]
        non_subset_idx = sampled_indices
    
    print(f"  Subset: {len(subset_idx)} parcels")
    print(f"  Non-subset (for comparison): {len(non_subset_idx):,} parcels")
    
    return X, X, parcel_ids, subset_idx, non_subset_idx

# -----------------------------------------------------------------------------
# SECTION: UTA-STAR (Piecewise Utilities) – helpers
# -----------------------------------------------------------------------------
def precompute_interpolation(X: np.ndarray, alpha: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute interpolation weights for piecewise linear utilities.
    Returns: W matrix and breakpoints
    """
    n, m = X.shape
    breakpoints = np.linspace(0, 1, alpha + 1)
    W = np.zeros((n, m, alpha + 1))
    
    for a in range(n):
        for i in range(m):
            x_val = X[a, i]
            for k in range(alpha):
                if breakpoints[k] <= x_val <= breakpoints[k+1]:
                    if breakpoints[k+1] - breakpoints[k] > 1e-10:
                        theta = (x_val - breakpoints[k]) / (breakpoints[k+1] - breakpoints[k])
                    else:
                        theta = 0.0
                    W[a, i, k] = 1 - theta
                    W[a, i, k+1] = theta
                    break
    
    return W, breakpoints

def solve_uta_star_utilities(X: np.ndarray, subset_idx: List[int], non_subset_idx: List[int], 
                            alpha: int = 3, delta: float = 1e-3, print_header: bool = True) -> Optional[Dict]:
    """
    Solve UTA-STAR with piecewise linear utilities.
    Returns utilities dictionary with results.
    """
    if print_header:
        print("\n" + "="*60)
        print("APPROACH 1: UTA-STAR WITH PIECEWISE LINEAR UTILITIES")
        print("="*60)
    
    n, m = X.shape
    W, breakpoints = precompute_interpolation(X, alpha)
    
    model = gp.Model("UTA-STAR-Utilities")
    # Use attribute-style params to avoid console messages
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = 60
    
    # Variables: u[i,k] for utility values at breakpoints
    u = {}
    for i in range(m):
        for k in range(alpha + 1):
            u[i,k] = model.addVar(lb=0, ub=1, name=f"u_{i}_{k}")
    
    # Error variables
    sigma_plus = {}
    sigma_minus = {}
    for p_idx in subset_idx:
        for q_idx in non_subset_idx:
            sigma_plus[p_idx,q_idx] = model.addVar(lb=0, name=f"sp_{p_idx}_{q_idx}")
            sigma_minus[p_idx,q_idx] = model.addVar(lb=0, name=f"sm_{p_idx}_{q_idx}")
    
    model.update()
    
    # Constraints
    # Normalization: u[i,0] = 0
    for i in range(m):
        model.addConstr(u[i,0] == 0)
    
    # Normalization: sum u[i,alpha] = 1
    model.addConstr(gp.quicksum(u[i,alpha] for i in range(m)) == 1)
    
    # Monotonicity
    for i in range(m):
        for k in range(alpha):
            model.addConstr(u[i,k+1] >= u[i,k])
    
    # Preference constraints
    for p_idx in subset_idx:
        for q_idx in non_subset_idx:
            U_p = gp.quicksum(W[p_idx,i,k] * u[i,k] for i in range(m) for k in range(alpha + 1))
            U_q = gp.quicksum(W[q_idx,i,k] * u[i,k] for i in range(m) for k in range(alpha + 1))
            model.addConstr(U_p - U_q >= delta - sigma_plus[p_idx,q_idx] + sigma_minus[p_idx,q_idx])
    
    # Objective: minimize total error
    obj = gp.quicksum(sigma_plus[key] + sigma_minus[key] for key in sigma_plus)
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Solve
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        # Extract utilities
        utilities = {}
        for i in range(m):
            utilities[str(i)] = [u[i,k].X for k in range(alpha + 1)]
        
        # Count violations
        violations = sum(1 for key in sigma_plus if sigma_plus[key].X > 1e-6 or sigma_minus[key].X > 1e-6)
        total_error = model.ObjVal
        
        return {
            'utilities': utilities,
            'total_error': total_error,
            'violations': violations,
            'breakpoints': breakpoints.tolist(),
            'alpha': alpha
        }
    
    return None

# -----------------------------------------------------------------------------
# SECTION: UTA-STAR (Linear Weights)
# -----------------------------------------------------------------------------
def solve_uta_star_weights(X: np.ndarray, subset_idx: List[int], non_subset_idx: List[int], 
                          delta: float = 1e-3) -> Optional[Dict]:
    """
    Solve for LINEAR WEIGHTS using UTA-STAR preference framework.
    Same constraints but with linear weights instead of utilities.
    """
    print("\n" + "="*60)
    print("APPROACH 2: LINEAR WEIGHTS WITH UTA-STAR PREFERENCES")
    print("="*60)
    
    n, m = X.shape
    
    model = gp.Model("UTA-STAR-Weights")
    # Use attribute-style params to avoid console messages
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = 60
    
    # Variables: weights for each criterion
    w = {}
    for i in range(m):
        w[i] = model.addVar(lb=0, ub=1, name=f"w_{i}")
    
    # Error variables
    sigma_plus = {}
    sigma_minus = {}
    for p_idx in subset_idx:
        for q_idx in non_subset_idx:
            sigma_plus[p_idx,q_idx] = model.addVar(lb=0, name=f"sp_{p_idx}_{q_idx}")
            sigma_minus[p_idx,q_idx] = model.addVar(lb=0, name=f"sm_{p_idx}_{q_idx}")
    
    model.update()
    
    # Constraint: weights sum to 1
    model.addConstr(gp.quicksum(w[i] for i in range(m)) == 1)
    
    # Preference constraints with linear scoring
    for p_idx in subset_idx:
        for q_idx in non_subset_idx:
            score_p = gp.quicksum(X[p_idx,i] * w[i] for i in range(m))
            score_q = gp.quicksum(X[q_idx,i] * w[i] for i in range(m))
            model.addConstr(score_p - score_q >= delta - sigma_plus[p_idx,q_idx] + sigma_minus[p_idx,q_idx])
    
    # Objective: minimize total error
    obj = gp.quicksum(sigma_plus[key] + sigma_minus[key] for key in sigma_plus)
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Solve
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        # Extract weights
        weights = np.array([w[i].X for i in range(m)])
        
        # Count violations
        violations = sum(1 for key in sigma_plus if sigma_plus[key].X > 1e-6 or sigma_minus[key].X > 1e-6)
        total_error = model.ObjVal
        
        return {
            'weights': weights,
            'total_error': total_error,
            'violations': violations
        }
    
    return None

def compute_utility_scores(X: np.ndarray, utilities: Dict, breakpoints: np.ndarray, alpha: int) -> np.ndarray:
    """Compute scores using piecewise linear utilities."""
    n, m = X.shape
    scores = np.zeros(n)
    
    for a in range(n):
        total_utility = 0.0
        for i in range(m):
            x_val = X[a, i]
            # Find which segment x_val falls into
            for k in range(alpha):
                if breakpoints[k] <= x_val <= breakpoints[k+1]:
                    # Linear interpolation within segment
                    if breakpoints[k+1] - breakpoints[k] > 1e-10:
                        theta = (x_val - breakpoints[k]) / (breakpoints[k+1] - breakpoints[k])
                    else:
                        theta = 0.0
                    u_val = (1 - theta) * utilities[str(i)][k] + theta * utilities[str(i)][k+1]
                    total_utility += u_val
                    break
        scores[a] = total_utility
    
    return scores

def evaluate_ranking(scores: np.ndarray, subset_idx: List[int]) -> Dict:
    """Evaluate ranking performance."""
    # Calculate ranks (rank 1 = highest score)
    ranks = 1 + np.argsort(np.argsort(-scores))
    
    # Get subset ranks
    subset_ranks = ranks[subset_idx]
    
    return {
        'average_rank': float(np.mean(subset_ranks)),
        'median_rank': float(np.median(subset_ranks)),
        'best_rank': int(np.min(subset_ranks)),
        'worst_rank': int(np.max(subset_ranks)),
        'top_500': int(np.sum(subset_ranks <= 500)),
        'top_1000': int(np.sum(subset_ranks <= 1000)),
        'top_500_rate': float(np.sum(subset_ranks <= 500) / len(subset_idx) * 100),
        'top_1000_rate': float(np.sum(subset_ranks <= 1000) / len(subset_idx) * 100)
    }

def plot_utility_functions_terminal(utility_result: Dict, subset_label: str | None = None) -> None:
    """Render all piecewise utility functions for a subset in one figure with subplots."""
    if not utility_result:
        return

    breakpoints: List[float] = utility_result['breakpoints']
    alpha: int = int(utility_result['alpha'])

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
        # Vertical guides at breakpoints
        for x in breakpoints:
            ax.axvline(x, color='lightgray', linestyle='--', linewidth=1)
        # Utility curve with markers at breakpoints
        ax.plot(breakpoints, utils, marker='o', color='cyan', label='utility')
        # Annotate each breakpoint value
        for k, (x, u) in enumerate(zip(breakpoints, utils)):
            ax.annotate(f"BP{k}: x={x:.2f}\nu={u:.2f}",
                        xy=(x, u), xytext=(4, 6), textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.8))
        # Straight linear reference from first to last point
        ax.plot([breakpoints[0], breakpoints[-1]], [utils[0], utils[-1]], color='gold', linestyle='--', label='linear ref')
        ax.set_title(f"{col}")
        ax.set_xlabel("criterion value (normalized)")
        ax.set_ylabel("utility")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.4)
        ax.legend()

    # Hide any unused subplots
    for j in range(num_criteria, len(axes)):
        axes[j].set_visible(False)

    suptitle = "Utility functions per criterion"
    if subset_label:
        suptitle += f" – {subset_label}"
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def format_piecewise_equations(utility_result: Dict, criteria: List[str]) -> str:
    """
    Return human-readable piecewise linear equations for each criterion.
    For each interval [bp_k, bp_{k+1}], show slope-intercept form.
    """
    try:
        breakpoints: List[float] = list(utility_result['breakpoints'])
        utils_by_idx: Dict[str, List[float]] = utility_result['utilities']
        alpha = int(utility_result.get('alpha', len(breakpoints) - 1))
    except Exception:
        return "(unable to format piecewise equations)"

    lines: List[str] = []
    for i, name in enumerate(criteria):
        vals = utils_by_idx.get(str(i))
        if not vals or len(vals) != len(breakpoints):
            continue
        lines.append(f"{name}(x) =")
        for k in range(alpha):
            x0, x1 = float(breakpoints[k]), float(breakpoints[k+1])
            u0, u1 = float(vals[k]), float(vals[k+1])
            dx = (x1 - x0) if (x1 - x0) != 0 else 1.0
            m = (u1 - u0) / dx
            b = u0 - m * x0
            lines.append(f"  if {x0:.2f} <= x <= {x1:.2f}:  u(x) = {m:.4f} * x + {b:.4f}")
        lines.append("")
    return "\n".join(lines)

def main():
    """Main execution function."""
    print("="*80)
    print("UTA-STAR COMPARISON: UTILITIES vs LINEAR WEIGHTS")
    print("="*80)
    
    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    parcels, _ = load_data()

    # Iterate over all subset shapefiles matching pattern
    # Accept both old and new naming; prioritize new subset_*.shp if present
    subset_files = sorted([p for p in DATA_DIR.glob('subset_*.shp')])
    if not subset_files:
        subset_files = sorted([p for p in DATA_DIR.glob('parcels_subset*.shp')])
    # Prepare a single GeoPackage to hold per-subset map layers
    map_output_file = OUTPUTS_DIR / "uta_maps.gpkg"
    try:
        if map_output_file.exists():
            map_output_file.unlink()
    except Exception:
        pass
    final_rows = []
    combined_subset_ids: set[str] = set()

    for subset_path in subset_files:
        subset = gpd.read_file(subset_path)
        subset_label = subset_path.stem
        if 'parcel_id' in subset.columns:
            combined_subset_ids.update(subset['parcel_id'].astype(str).tolist())

        print("\n" + "#"*80)
        print(f"SUBSET: {subset_label}")
        print("#"*80)

        # Prepare data
        X, X_norm, parcel_ids, subset_idx, non_subset_idx = prepare_data(parcels, subset)

        # Optional: Explore nearby contiguous clusters around this subset to find ~20 parcel clusters with mixed weights
        # We use a simple contiguity-based cluster search when the current subset size is not ~20.
        # This mode can be extended; currently we just proceed with the provided subset.

        # ------------------------------------------------------------------
        # UTA-STAR (Piecewise Utilities) — multiple alphas
        # ------------------------------------------------------------------
        utility_results_by_alpha: dict[int, dict] = {}
        for idx_alpha, alph in enumerate(ALPHA_LIST):
            start_time = time_module.time()
            ures = solve_uta_star_utilities(X, subset_idx, non_subset_idx, alpha=alph, print_header=(idx_alpha == 0))
            utime = time_module.time() - start_time
            if ures:
                utility_results_by_alpha[alph] = {**ures, 'solve_time': utime}
                print(f"\n✓ Piecewise utilities (alpha={alph}) in {utime:.2f}s  | error={ures['total_error']:.4f} | viol={ures['violations']}")
                # Rankings for this alpha
                scores_u_this = compute_utility_scores(X, ures['utilities'], np.array(ures['breakpoints']), ures['alpha'])
                r_u_this = evaluate_ranking(scores_u_this, subset_idx)
                print(f"  Ranking (alpha={alph}): avg={r_u_this['average_rank']:.1f} | min={r_u_this['best_rank']} | max={r_u_this['worst_rank']} | top500={r_u_this['top_500']}")
                # Plot utility functions for this alpha with breakpoint annotations
                print("  Rendering utility functions (alpha={})...".format(alph))
                plot_utility_functions_terminal(ures, subset_label=f"{subset_label} (alpha={alph})")
                plt.close('all')
            else:
                print(f"\n✗ Piecewise utilities (alpha={alph}) failed")
        
        # Evaluate and display the best alpha (by average rank) and a summary table
        best_alpha = None
        best_avg = float('inf')
        utility_ranking = None
        utility_scores = None
        derived_weights = None
        if utility_results_by_alpha:
            rows = []
            for alph, ures in utility_results_by_alpha.items():
                scores_u = compute_utility_scores(X, ures['utilities'], np.array(ures['breakpoints']), ures['alpha'])
                r_u = evaluate_ranking(scores_u, subset_idx)
                rows.append({'alpha': alph, 'avg': r_u['average_rank'], 'min': r_u['best_rank'], 'max': r_u['worst_rank'], 'time': ures.get('solve_time', 0.0)})
                if r_u['average_rank'] < best_avg:
                    best_avg = r_u['average_rank']
                    best_alpha = alph
                    utility_ranking = r_u
                    utility_scores = scores_u
                    derived_weights = {col: float(ures['utilities'][str(i)][-1]) for i, col in enumerate(CRITERIA_COLS)}
            print("\nPiecewise alpha sweep (by average rank):")
            df_alpha = pd.DataFrame(rows).sort_values('alpha')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
                print(df_alpha.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            print(f"\nBest alpha: {best_alpha} (avg rank {best_avg:.1f})")
        else:
            print("\n✗ Utility optimization failed for all alphas")
            utility_ranking = None
        
        # ------------------------------------------------------------------
        # ORDERED RUNS
        # 1) UTA-STAR (Linear Weights)
        # ------------------------------------------------------------------
        start_time = time_module.time()
        weight_result = solve_uta_star_weights(X, subset_idx, non_subset_idx)
        weight_time = time_module.time() - start_time
        
        if weight_result:
            print(f"\n✓ Optimization successful in {weight_time:.2f}s")
            print(f"  Total error: {weight_result['total_error']:.4f}")
            print(f"  Violations: {weight_result['violations']:,}")
            
            # Display weights
            print("\nOptimized Linear Weights:")
            for i, col in enumerate(CRITERIA_COLS):
                print(f"  {col:10s}: {weight_result['weights'][i]:.1%}")
            
            # Compute scores and evaluate
            weight_scores = X @ weight_result['weights']
            weight_ranking = evaluate_ranking(weight_scores, subset_idx)
            
            print(f"\nRanking Performance:")
            print(f"  Average rank: {weight_ranking['average_rank']:.1f}")
            print(f"  Median rank:  {weight_ranking['median_rank']:.1f}")
            print(f"  Best rank:    {weight_ranking['best_rank']}")
            print(f"  Worst rank:   {weight_ranking['worst_rank']}")
            print(f"  Top 500:      {weight_ranking['top_500']}/{len(subset_idx)} ({weight_ranking['top_500_rate']:.1f}%)")
            print(f"  Top 1000:     {weight_ranking['top_1000']}/{len(subset_idx)} ({weight_ranking['top_1000_rate']:.1f}%)")
        else:
            print("\n✗ Weight optimization failed")
            weight_ranking = None
        
        # ------------------------------------------------------------------
        # Comparison and Reporting (per subset)
        # ------------------------------------------------------------------
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        if utility_ranking and weight_ranking:
            print(f"\nAverage Rank Performance:")
            print(f"  Piecewise Utilities: {utility_ranking['average_rank']:8.1f}")
            print(f"  Linear Weights:      {weight_ranking['average_rank']:8.1f}")
            
            improvement = (weight_ranking['average_rank'] - utility_ranking['average_rank']) / weight_ranking['average_rank'] * 100
            
            if improvement > 0:
                print(f"\n✓ Utilities perform {improvement:.1f}% better than linear weights")
            else:
                print(f"\n✗ Linear weights perform {-improvement:.1f}% better than utilities")
            
            print(f"\nTop 500 Performance:")
            print(f"  Piecewise Utilities: {utility_ranking['top_500_rate']:5.1f}%")
            print(f"  Linear Weights:      {weight_ranking['top_500_rate']:5.1f}%")
            
            print(f"\nSolve Time:")
            if 'solve_time' in (utility_results_by_alpha.get(best_alpha) or {}):
                print(f"  Piecewise Utilities: {utility_results_by_alpha[best_alpha]['solve_time']:5.2f}s (alpha={best_alpha})")
            else:
                print("  Piecewise Utilities: n/a")
            print(f"  Linear Weights:      {weight_time:5.2f}s")
            
            # Build friendlier utility mapping by criterion name (from best alpha)
            ures_best = utility_results_by_alpha.get(best_alpha)
            utilities_by_criterion = {col: ures_best['utilities'][str(i)] for i, col in enumerate(CRITERIA_COLS)} if ures_best else {}
            weights_by_criterion = {
                col: float(weight_result['weights'][i]) for i, col in enumerate(CRITERIA_COLS)
            }
            # Dense utility curves (for convenience; best alpha)
            x_dense = np.linspace(0, 1, 101).tolist()
            curves_by_criterion = {}
            for i, col in enumerate(CRITERIA_COLS):
                y_dense = np.interp(np.linspace(0,1,101), np.array(ures_best['breakpoints']), np.array(ures_best['utilities'][str(i)])).tolist()
                curves_by_criterion[col] = y_dense
            # Derived weights from utilities (best alpha)
            derived_weights_from_utilities = {col: float(ures_best['utilities'][str(i)][-1]) for i, col in enumerate(CRITERIA_COLS)} if ures_best else {}
            # Utility values at each breakpoint per criterion (for summary CSV; best alpha)
            utility_values_by_breakpoint = {}
            for i, col in enumerate(CRITERIA_COLS):
                utils_list = ures_best['utilities'][str(i)] if ures_best else []
                for k, uval in enumerate(utils_list):
                    utility_values_by_breakpoint[f'utilbp_{col}_{k}'] = float(uval)
            
            # Save results (includes utility functions)
            results = {
                'utilities': utility_result,  # includes raw utilities indexed by criterion number
                'utilities_by_criterion': utilities_by_criterion,
                'utility_curves_dense': {
                    'x': x_dense,
                    'by_criterion': curves_by_criterion
                },
                'derived_weights_from_utilities': derived_weights_from_utilities,
                'utility_ranking': utility_ranking,
                'weights': {
                    'values': weight_result['weights'].tolist(),
                    'by_criterion': weights_by_criterion,
                    'total_error': weight_result['total_error'],
                    'violations': weight_result['violations']
                },
                'weight_ranking': weight_ranking,
                'comparison': {
                    'utility_avg_rank': utility_ranking['average_rank'],
                    'weight_avg_rank': weight_ranking['average_rank'],
                    'improvement_pct': improvement
                }
            }
            
            output_file = OUTPUTS_DIR / f"uta_star_comparison_{subset_label}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {output_file}")

            # Create per-parcel mapping outputs (scores and ranks for both approaches)
            ranks_u = 1 + np.argsort(np.argsort(-utility_scores))
            ranks_w = 1 + np.argsort(np.argsort(-weight_scores))
            in_subset_mask = np.zeros(len(parcel_ids), dtype=bool)
            in_subset_mask[subset_idx] = True

            gdf_map = parcels.copy()
            gdf_map['score_u'] = utility_scores
            gdf_map['rank_u'] = ranks_u.astype(int)
            gdf_map['score_w'] = weight_scores
            gdf_map['rank_w'] = ranks_w.astype(int)
            gdf_map['in_subset'] = in_subset_mask
            gdf_map['subset'] = subset_label

            # Write this subset as a separate layer in a single GeoPackage
            try:
                gdf_map.to_file(map_output_file, layer=subset_label, driver='GPKG')
                print(f"Map layer written: {map_output_file} (layer={subset_label})")
            except Exception as e:
                print(f"Warning: failed to write map layer for {subset_label}: {e}")
            
            # Collect two rows (piecewise and linear) for final table
            row_piecewise = {
                'subset': subset_label,
                'approach': 'piecewise',
                'avg': utility_ranking['average_rank'],
                'min': utility_ranking['best_rank'],
                'max': utility_ranking['worst_rank'],
                'top500': utility_ranking['top_500'],
            }
            for col in CRITERIA_COLS:
                row_piecewise[col] = derived_weights_from_utilities[col]
            # add utility breakpoints
            row_piecewise.update(utility_values_by_breakpoint)
            final_rows.append(row_piecewise)

            row_linear = {
                'subset': subset_label,
                'approach': 'linear',
                'avg': weight_ranking['average_rank'],
                'min': weight_ranking['best_rank'],
                'max': weight_ranking['worst_rank'],
                'top500': weight_ranking['top_500'],
            }
            for col in CRITERIA_COLS:
                row_linear[col] = weights_by_criterion[col]
            final_rows.append(row_linear)

            # Summary table will be printed at the end for all subsets
        
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        
        if utility_result and weight_result:
            
            # Check if utilities are mostly linear or have strong non-linearities
            max_nonlinearity = 0
            for i in range(len(CRITERIA_COLS)):
                utils = utility_result['utilities'][str(i)]
                # Check deviation from linearity
                linear_approx = np.linspace(utils[0], utils[-1], len(utils))
                deviation = np.max(np.abs(np.array(utils) - linear_approx))
                max_nonlinearity = max(max_nonlinearity, deviation)
            
            if max_nonlinearity > 0.2:
                print("\n✓ Strong non-linearities detected - utilities significantly better than weights")
            else:
                print("\n◐ Moderate non-linearities - utilities slightly better than weights")

    # Final summary table across all subsets (two rows per subset)
    if final_rows:
        print("\n" + "="*80)
        print("FINAL SUMMARY TABLE (ALL SUBSETS)")
        print("="*80)
        # dynamic columns include utility breakpoint columns prefixed with 'utilbp_'
        df_cols = ['subset', 'approach'] + CRITERIA_COLS + ['avg', 'min', 'max', 'top500']
        df = pd.DataFrame(final_rows)
        # Place breakpoint cols at the end
        ordered_cols = [c for c in df_cols if c in df.columns] + [c for c in df.columns if c.startswith('utilbp_')]
        df = df[ordered_cols]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 240):
            print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        # Write CSV
        csv_path = OUTPUTS_DIR / 'uta_final_summary.csv'
        try:
            df.to_csv(csv_path, index=False)
            print(f"\nSummary CSV saved to: {csv_path}")
        except Exception as e:
            print(f"Warning: failed to write summary CSV: {e}")

        # Also write simplified, friendly overview (<= 10 columns)
        overview_rows = []
        for r in final_rows:
            subset_name = r.get('subset')
            approach = r.get('approach')
            avg_rank = r.get('avg')
            min_rank = r.get('min')
            max_rank = r.get('max')
            top500 = r.get('top500')
            # Determine dominant and second-most criteria by weight
            weight_map = {c: float(r.get(c, 0.0)) for c in CRITERIA_COLS}
            sorted_items = sorted(weight_map.items(), key=lambda x: x[1], reverse=True)
            (dom_c, dom_w) = sorted_items[0]
            (sec_c, sec_w) = sorted_items[1] if len(sorted_items) > 1 else (None, 0.0)
            overview_rows.append({
                'Subset': subset_name,
                'Approach': 'Piecewise' if approach == 'piecewise' else 'Linear',
                'AvgRank': avg_rank,
                'MinRank': min_rank,
                'MaxRank': max_rank,
                'Top500': top500,
                'DominantCriterion': dom_c,
                'DominantWeight': dom_w,
                'SecondCriterion': sec_c,
                'SecondWeight': sec_w,
            })

        df_overview = pd.DataFrame(overview_rows, columns=[
            'Subset','Approach','AvgRank','MinRank','MaxRank','Top500','DominantCriterion','DominantWeight','SecondCriterion','SecondWeight'
        ])
        overview_csv = OUTPUTS_DIR / 'uta_summary_overview.csv'
        try:
            df_overview.to_csv(overview_csv, index=False)
            print(f"Friendly overview CSV saved to: {overview_csv}")
        except Exception as e:
            print(f"Warning: failed to write overview CSV: {e}")

        # Write piecewise utility-at-breakpoints by criterion (narrow)
        piecewise_rows = []
        # Detect number of breakpoints from utilbp_ keys
        utilbp_keys = [k for k in (final_rows[0].keys()) if isinstance(k, str) and k.startswith('utilbp_')]
        # Fallback to 4 breakpoints (alpha=3) if detection fails
        max_k = 3
        if utilbp_keys:
            try:
                max_k = max(int(k.split('_')[-1]) for k in utilbp_keys)
            except Exception:
                max_k = 3

        for r in final_rows:
            if r.get('approach') != 'piecewise':
                continue
            subset_name = r.get('subset')
            for c in CRITERIA_COLS:
                row_narrow = {'Subset': subset_name, 'Criterion': c}
                for k in range(max_k + 1):
                    row_narrow[f'BP{k}'] = float(r.get(f'utilbp_{c}_{k}', 0.0))
                piecewise_rows.append(row_narrow)

        df_utils = pd.DataFrame(piecewise_rows, columns=['Subset','Criterion'] + [f'BP{k}' for k in range(max_k + 1)])
        utils_csv = OUTPUTS_DIR / 'uta_piecewise_utilities.csv'
        try:
            df_utils.to_csv(utils_csv, index=False)
            print(f"Piecewise utilities CSV saved to: {utils_csv}")
        except Exception as e:
            print(f"Warning: failed to write piecewise utilities CSV: {e}")

    # ------------------------------------------------------------------
    # Combined-subset run (all subset parcel_ids together as preferences)
    # ------------------------------------------------------------------
    if combined_subset_ids:
        print("\n" + "#"*80)
        print("SUBSET: combined_all_subsets")
        print("#"*80)

        # Build a subset GeoDataFrame on-the-fly
        combined_subset = gpd.GeoDataFrame({'parcel_id': list(combined_subset_ids)}, geometry=None, crs=None)

        # Prepare data
        X, X_norm, parcel_ids, subset_idx, non_subset_idx = prepare_data(parcels, combined_subset)

        # Piecewise
        start_time = time_module.time()
        utility_result = solve_uta_star_utilities(X, subset_idx, non_subset_idx, alpha=3)
        utility_time = time_module.time() - start_time

        # Linear
        start_time = time_module.time()
        weight_result = solve_uta_star_weights(X, subset_idx, non_subset_idx)
        weight_time = time_module.time() - start_time

        if utility_result and weight_result:
            # Scores and rankings
            utility_scores = compute_utility_scores(X, utility_result['utilities'], 
                                                   np.array(utility_result['breakpoints']), 
                                                   utility_result['alpha'])
            weight_scores = X @ weight_result['weights']
            utility_ranking = evaluate_ranking(utility_scores, subset_idx)
            weight_ranking = evaluate_ranking(weight_scores, subset_idx)

            print(f"\nCombined Run Ranking Performance:")
            print(f"  Piecewise avg: {utility_ranking['average_rank']:.1f} | min: {utility_ranking['best_rank']} | max: {utility_ranking['worst_rank']} | top500: {utility_ranking['top_500']}")
            print(f"  Linear    avg: {weight_ranking['average_rank']:.1f} | min: {weight_ranking['best_rank']} | max: {weight_ranking['worst_rank']} | top500: {weight_ranking['top_500']}")

            # Append to final summary
            derived_weights_from_utilities = {col: float(utility_result['utilities'][str(i)][-1]) for i, col in enumerate(CRITERIA_COLS)}
            weights_by_criterion = {col: float(weight_result['weights'][i]) for i, col in enumerate(CRITERIA_COLS)}

            row_piecewise = {'subset': 'combined_all_subsets', 'approach': 'piecewise', 'avg': utility_ranking['average_rank'], 'min': utility_ranking['best_rank'], 'max': utility_ranking['worst_rank'], 'top500': utility_ranking['top_500']}
            for col in CRITERIA_COLS:
                row_piecewise[col] = derived_weights_from_utilities[col]
            final_rows.append(row_piecewise)

            row_linear = {'subset': 'combined_all_subsets', 'approach': 'linear', 'avg': weight_ranking['average_rank'], 'min': weight_ranking['best_rank'], 'max': weight_ranking['worst_rank'], 'top500': weight_ranking['top_500']}
            for col in CRITERIA_COLS:
                row_linear[col] = weights_by_criterion[col]
            final_rows.append(row_linear)

            # Rebuild final tables and CSVs including combined
            df = pd.DataFrame(final_rows)
            ordered_cols = ['subset', 'approach'] + CRITERIA_COLS + ['avg', 'min', 'max', 'top500'] + [c for c in df.columns if c.startswith('utilbp_')]
            df = df[[c for c in ordered_cols if c in df.columns]]
            print("\n" + "="*80)
            print("FINAL SUMMARY TABLE (ALL SUBSETS, INCLUDING COMBINED)")
            print("="*80)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 240):
                print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            csv_path = OUTPUTS_DIR / 'uta_final_summary.csv'
            try:
                df.to_csv(csv_path, index=False)
                print(f"Updated summary CSV saved to: {csv_path}")
            except Exception as e:
                print(f"Warning: failed to write summary CSV: {e}")

            # Friendly overview CSV
            overview_rows = []
            for r in final_rows:
                subset_name = r.get('subset')
                approach = r.get('approach')
                avg_rank = r.get('avg')
                min_rank = r.get('min')
                max_rank = r.get('max')
                top500 = r.get('top500')
                weight_map = {c: float(r.get(c, 0.0)) for c in CRITERIA_COLS}
                sorted_items = sorted(weight_map.items(), key=lambda x: x[1], reverse=True)
                (dom_c, dom_w) = sorted_items[0]
                (sec_c, sec_w) = sorted_items[1] if len(sorted_items) > 1 else (None, 0.0)
                overview_rows.append({
                    'Subset': subset_name,
                    'Approach': 'Piecewise' if approach == 'piecewise' else 'Linear',
                    'AvgRank': avg_rank,
                    'MinRank': min_rank,
                    'MaxRank': max_rank,
                    'Top500': top500,
                    'DominantCriterion': dom_c,
                    'DominantWeight': dom_w,
                    'SecondCriterion': sec_c,
                    'SecondWeight': sec_w,
                })
            df_overview = pd.DataFrame(overview_rows, columns=['Subset','Approach','AvgRank','MinRank','MaxRank','Top500','DominantCriterion','DominantWeight','SecondCriterion','SecondWeight'])
            overview_csv = OUTPUTS_DIR / 'uta_summary_overview.csv'
            try:
                df_overview.to_csv(overview_csv, index=False)
                print(f"Updated friendly overview CSV saved to: {overview_csv}")
            except Exception as e:
                print(f"Warning: failed to write overview CSV: {e}")

            # Piecewise utilities CSV (long form)
            piecewise_rows = []
            utilbp_keys = [k for k in (final_rows[0].keys()) if isinstance(k, str) and k.startswith('utilbp_')]
            max_k = 3
            if utilbp_keys:
                try:
                    max_k = max(int(k.split('_')[-1]) for k in utilbp_keys)
                except Exception:
                    max_k = 3
            for r in final_rows:
                if r.get('approach') != 'piecewise':
                    continue
                subset_name = r.get('subset')
                for c in CRITERIA_COLS:
                    row_narrow = {'Subset': subset_name, 'Criterion': c}
                    for k in range(max_k + 1):
                        row_narrow[f'BP{k}'] = float(r.get(f'utilbp_{c}_{k}', 0.0))
                    piecewise_rows.append(row_narrow)
            df_utils = pd.DataFrame(piecewise_rows, columns=['Subset','Criterion'] + [f'BP{k}' for k in range(max_k + 1)])
            utils_csv = OUTPUTS_DIR / 'uta_piecewise_utilities.csv'
            try:
                df_utils.to_csv(utils_csv, index=False)
                print(f"Updated piecewise utilities CSV saved to: {utils_csv}")
            except Exception as e:
                print(f"Warning: failed to write piecewise utilities CSV: {e}")

if __name__ == "__main__":
    main()