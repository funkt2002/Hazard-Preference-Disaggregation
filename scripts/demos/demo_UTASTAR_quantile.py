#!/usr/bin/env python3
"""
Demo UTA-STAR implementation - WUI QUANTILE NORMALIZATION ONLY.
Uses combined demo selections as preferred parcels and tests on WUI quantile dataset.

Focuses only on WUI parcels with quantile normalization:
- quantile_wui (26,396 WUI parcels with quantile normalization)
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
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, Normalize
import matplotlib.patches as mpatches
import io
import base64
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from shapely.geometry import box

# Simple Gurobi license configuration - hardcoded path
os.environ["GRB_LICENSE_FILE"] = "/Users/theofunk/Desktop/Narsc Paper/gurobi.lic"

try:
    import gurobipy as gp
    from gurobipy import GRB
    print("Using Gurobi solver for optimization")
except ImportError:
    print("Error: Gurobi (gurobipy) is required for this script. Please install and ensure a valid license.")
    sys.exit(1)

# Path configuration - updated for new structure
BASE_PATH = Path("/Users/theofunk/Desktop/Narsc Paper")
DATA_DIR = BASE_PATH / "data"
OUTPUTS_DIR = BASE_PATH / "outputs" / "demo_utastar_quantile"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Criteria columns for demo analysis
CRITERIA_COLS = ['agfb_s', 'hfbfm_s', 'qtrmi_s', 'slope_s', 'hvhsz_s', 'travel_s']

# Alpha values for piecewise utilities (extended for convergence analysis)
ALPHA_LIST: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Reference set configuration (use sampling like original UTASTAR)
USE_REFERENCE_SAMPLE: bool = True
REFERENCE_SAMPLE_SIZE: int = 5000
REFERENCE_SAMPLE_SEED: int = 42

# Dataset configurations - WUI QUANTILE ONLY
DATASET_CONFIGS = [
    {
        'name': 'quantile_wui', 
        'path': 'fire_risk_local_quantile_wui_parcels.shp',
        'description': 'WUI parcels with quantile normalization'
    }
]

# Selection paths (OPTIMAL COMBINATION: San Roque + Painted Cave + Above E Mtn Drive)
SELECTION_PATHS = [
    'subsets/painted_cave_quantile.shp',
    'subsets/mission_canyon_montrose_quantile.shp',
    'subsets/above_E_mtn_drive.shp'
]

# Friendly display names for plotting
FRIENDLY_LABELS: Dict[str, str] = {
    'agfb_s': 'Agriculture and Fuel Break % within a Half Mile',
    'hfbfm_s': 'Fire Behavior Fuel Model (normalized)',
    'qtrmi_s': 'Count of Structures within a Quarter Mile',
    'slope_s': 'Slope (%)',
    'hvhsz_s': 'Very High Hazard Severity Zones % within a Half Mile',
    'travel_s': 'Travel Time (normalized)'
}

def friendly_label(column_name: str) -> str:
    """Return a human-friendly display label for a criterion column name."""
    return FRIENDLY_LABELS.get(column_name, column_name)

def create_convergence_plot(utility_results_by_alpha: Dict, dataset_name: str) -> Optional[Path]:
    """
    Create a clean academic-style convergence plot showing how optimization error
    decreases as alpha (model complexity) increases.
    
    Args:
        utility_results_by_alpha: Dictionary mapping alpha -> optimization results
        dataset_name: Name of dataset for plot title and filename
    
    Returns:
        Path to saved plot or None if failed
    """
    try:
        # Extract data for plotting
        alphas = sorted(utility_results_by_alpha.keys())
        errors = [utility_results_by_alpha[alpha]['total_error'] for alpha in alphas]
        violations = [utility_results_by_alpha[alpha]['violations'] for alpha in alphas]
        
        # Create figure with academic styling
        plt.style.use('default')  # Clean default style
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
        # Plot 1: Total Error vs Alpha
        ax1.plot(alphas, errors, 'o-', color='#2E86AB', linewidth=2, markersize=6, 
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2E86AB')
        ax1.set_ylabel('Total Optimization Error', fontsize=11, fontweight='medium')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(bottom=0)
        
        # Add error values as text labels
        for alpha, error in zip(alphas, errors):
            ax1.annotate(f'{error:.3f}', (alpha, error), 
                        textcoords="offset points", xytext=(0,8), ha='center',
                        fontsize=9, color='#2E86AB')
        
        # Plot 2: Violations vs Alpha  
        ax2.plot(alphas, violations, 's-', color='#A23B72', linewidth=2, markersize=6,
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#A23B72')
        ax2.set_xlabel('Alpha (Number of Segments)', fontsize=11, fontweight='medium')
        ax2.set_ylabel('Preference Violations', fontsize=11, fontweight='medium')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(bottom=0)
        
        # Add violation values as text labels
        for alpha, viol in zip(alphas, violations):
            ax2.annotate(f'{viol}', (alpha, viol), 
                        textcoords="offset points", xytext=(0,8), ha='center',
                        fontsize=9, color='#A23B72')
        
        # Set x-axis ticks and labels
        ax2.set_xticks(alphas)
        ax2.set_xticklabels([str(a) for a in alphas])
        
        # Overall styling
        fig.suptitle('UTA-STAR Model Complexity vs. Optimization Performance', 
                    fontsize=13, fontweight='bold', y=0.95)
        
        # Add subtitle with dataset info
        ax1.text(0.5, 1.05, f'Dataset: {dataset_name}', transform=ax1.transAxes, 
                ha='center', va='bottom', fontsize=10, style='italic')
        
        # Tight layout with space for suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Save plot
        plot_filename = f"convergence_analysis_{dataset_name}.png"
        plot_path = OUTPUTS_DIR / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"  Convergence plot saved: {plot_path}")
        
        # Display plot
        plt.show()
        plt.close(fig)
        
        return plot_path
        
    except Exception as e:
        print(f"  Warning: Could not create convergence plot: {e}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------------------------------------------------------
# SECTION: Data Loading Functions
# -----------------------------------------------------------------------------

def load_demo_selections() -> set[str]:
    """
    Load all demo area selections and return combined set of parcel IDs.
    Returns the union of all parcel_ids from the selection areas.
    """
    print("Loading demo area selections...")
    all_preferred_ids = set()
    
    for sel_path in SELECTION_PATHS:
        full_path = DATA_DIR / sel_path
        try:
            sel_gdf = gpd.read_file(full_path)
            sel_ids = set(sel_gdf['parcel_id'].astype(str))
            area_name = sel_path.split('/')[1]  # Extract area name
            print(f"  {area_name}: {len(sel_ids)} parcels")
            all_preferred_ids.update(sel_ids)
        except Exception as e:
            print(f"  Warning: Could not load {sel_path}: {e}")
    
    print(f"  Total combined preferred parcels: {len(all_preferred_ids)}")
    return all_preferred_ids

def load_dataset(config: Dict) -> gpd.GeoDataFrame:
    """Load a specific dataset configuration."""
    dataset_path = DATA_DIR / config['path']
    print(f"Loading dataset: {config['description']}")
    print(f"  Path: {dataset_path}")
    
    try:
        gdf = gpd.read_file(dataset_path)
        print(f"  Loaded: {len(gdf):,} parcels")
        return gdf
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        raise

def prepare_demo_data(gdf: gpd.GeoDataFrame, preferred_ids: set[str]) -> Tuple[np.ndarray, List[str], List[int], List[int]]:
    """
    Prepare data for UTA-STAR analysis using demo preferred parcels.
    Returns: X matrix, parcel IDs, preferred indices, non-preferred indices
    """
    print(f"Preparing data...")
    
    # Ensure all required criteria columns exist
    missing_cols = [col for col in CRITERIA_COLS if col not in gdf.columns]
    if missing_cols:
        print(f"  Warning: Missing columns {missing_cols}")
        for col in missing_cols:
            gdf[col] = 0.0
    
    # Extract data matrix
    X = gdf[CRITERIA_COLS].values
    parcel_ids = gdf['parcel_id'].astype(str).tolist()
    
    # Renormalize criteria values if this is a WUI dataset
    if 'wui' in str(gdf.columns).lower() or len(gdf) < 50000:  # WUI datasets are smaller
        print(f"  Detected WUI dataset - renormalizing criteria values to 0-1 range")
        for i, col in enumerate(CRITERIA_COLS):
            col_values = X[:, i]
            col_min = col_values.min()
            col_max = col_values.max()
            if col_max > col_min:  # Avoid division by zero
                X[:, i] = (col_values - col_min) / (col_max - col_min)
                print(f"    {col}: [{col_min:.3f}, {col_max:.3f}] -> [0.000, 1.000]")
            else:
                print(f"    {col}: constant values, keeping as-is")
    else:
        print(f"  Using original normalization (all parcels dataset)")
    
    # Get indices for preferred parcels
    preferred_idx = [i for i, pid in enumerate(parcel_ids) if pid in preferred_ids]
    
    # Verify we found all preferred parcels
    found_preferred = len(preferred_idx)
    expected_preferred = len(preferred_ids)
    
    print(f"  Preferred parcels: {found_preferred}/{expected_preferred} found in dataset")
    
    if found_preferred < expected_preferred:
        missing = expected_preferred - found_preferred
        print(f"  Warning: {missing} preferred parcels not found in this dataset")
    
    # Create reference set: use stratified spatial sample if enabled
    if USE_REFERENCE_SAMPLE:
        sampled_labels = stratified_spatial_sample(gdf, list(preferred_ids), target_size=REFERENCE_SAMPLE_SIZE, seed=REFERENCE_SAMPLE_SEED)
        # Map sampled labels back to positional indices
        sampled_label_set = set(sampled_labels)
        non_preferred_idx = [i for i, (pid, idx_label) in enumerate(zip(parcel_ids, gdf.index)) 
                           if (pid not in preferred_ids and idx_label in sampled_label_set)]
        print(f"  Non-preferred reference (sampled): {len(non_preferred_idx):,} parcels (target={REFERENCE_SAMPLE_SIZE:,})")
    else:
        # Use ALL non-preferred parcels
        non_preferred_idx = [i for i, pid in enumerate(parcel_ids) if pid not in preferred_ids]
        print(f"  Non-preferred reference (all): {len(non_preferred_idx):,} parcels")
    
    return X, parcel_ids, preferred_idx, non_preferred_idx

def stratified_spatial_sample(parcels_gdf: gpd.GeoDataFrame, preferred_ids: List[str], target_size: int = 5000, seed: int = 42) -> List[int]:
    """
    Create stratified spatial sample of parcels outside the preferred set.
    Uses grid-based stratification for geographic representation.
    Returns a list of GeoDataFrame index labels for the sampled reference parcels.
    """
    np.random.seed(seed)

    preferred_id_set = set(map(str, preferred_ids))
    candidates = parcels_gdf[~parcels_gdf['parcel_id'].astype(str).isin(preferred_id_set)].copy()

    if len(candidates) <= target_size:
        return candidates.index.tolist()

    bounds = candidates.total_bounds
    n_grid = min(int(np.sqrt(target_size / 10)), 15)  # ~10 samples per cell, cap at 15x15
    n_grid = max(1, n_grid)

    x_bins = np.linspace(bounds[0], bounds[2], n_grid + 1)
    y_bins = np.linspace(bounds[1], bounds[3], n_grid + 1)

    centroids = candidates.geometry.centroid
    x_indices = np.digitize(centroids.x, x_bins) - 1
    y_indices = np.digitize(centroids.y, y_bins) - 1
    cell_ids = x_indices * n_grid + y_indices

    sampled_indices: List[int] = []
    unique_cells, cell_counts = np.unique(cell_ids, return_counts=True)
    for cell_id, cell_count in zip(unique_cells, cell_counts):
        cell_target = max(1, int(target_size * cell_count / len(candidates)))
        cell_mask = cell_ids == cell_id
        cell_candidates = candidates.index[cell_mask]
        if len(cell_candidates) <= cell_target:
            sampled_indices.extend(cell_candidates.tolist())
        else:
            sample = np.random.choice(cell_candidates, cell_target, replace=False)
            sampled_indices.extend(sample.tolist())

    if len(sampled_indices) > target_size:
        sampled_indices = np.random.choice(sampled_indices, target_size, replace=False).tolist()

    print(f"  Stratified spatial sample (reference): {len(sampled_indices):,} parcels")
    return sampled_indices

def identify_parcel_subsets(parcel_ids: List[str]) -> Dict[str, str]:
    """
    Identify which subset each parcel belongs to based on the demo selections.
    Returns dict mapping parcel_id to subset name (or None if not in any subset).
    """
    parcel_to_subset = {}
    
    # Map each selection to its subset name
    subset_mappings = [
        ('painted_cave_quantile', 'selections/painted_cave_quantile/fire_risk_local_quantile_spatial_parcels.shp'),
        ('san_roque_to_mission_canyon', 'selections/san_roque_to_misssion_canyon/san_roque_reduced_200.shp'),
        ('above_E_mtn_drive', 'selections/above_E_mtn_drive/fire_risk_local_raw_minmax_spatial_parcels.shp')
    ]
    
    for subset_name, sel_path in subset_mappings:
        try:
            full_path = DATA_DIR / sel_path
            sel_gdf = gpd.read_file(full_path)
            sel_ids = set(sel_gdf['parcel_id'].astype(str))
            
            for pid in sel_ids:
                parcel_to_subset[pid] = subset_name
                
        except Exception as e:
            print(f"  Warning: Could not load {sel_path} for subset identification: {e}")
    
    return parcel_to_subset

def save_utility_scores_shapefile(gdf: gpd.GeoDataFrame, scores: np.ndarray, parcel_ids: List[str],
                                dataset_name: str, alpha: int, approach: str = "piecewise") -> Optional[Path]:
    """
    Save utility scores for all parcels as a shapefile with subset identification.
    
    Args:
        gdf: Original geodataframe with geometry
        scores: Computed utility scores for all parcels
        parcel_ids: List of parcel IDs matching scores order
        dataset_name: Name of the dataset (e.g., 'quantile_all')
        alpha: Alpha value used (for piecewise approach)
        approach: 'piecewise' or 'linear'
    
    Returns:
        Path to saved shapefile or None if failed
    """
    try:
        # Create output geodataframe
        output_gdf = gdf.copy()
        
        # Map scores to parcels
        score_dict = {pid: score for pid, score in zip(parcel_ids, scores)}
        output_gdf['utility_score'] = output_gdf['parcel_id'].astype(str).map(score_dict).fillna(0)
        
        # Calculate ranks (1 = highest score)
        output_gdf['utility_rank'] = output_gdf['utility_score'].rank(method='min', ascending=False).astype(int)
        
        # Add metadata columns
        output_gdf['dataset'] = dataset_name
        output_gdf['approach'] = approach
        output_gdf['alpha'] = alpha if approach == 'piecewise' else 1
        
        # Identify subset membership
        parcel_to_subset = identify_parcel_subsets(parcel_ids)
        output_gdf['subset'] = output_gdf['parcel_id'].astype(str).map(parcel_to_subset)
        output_gdf['subset'] = output_gdf['subset'].fillna('none')  # Fill non-subset parcels with 'none'
        
        # Create organized filename
        if approach == 'piecewise':
            filename = f"{dataset_name}_piecewise_alpha{alpha}_utilities.shp"
        else:
            filename = f"{dataset_name}_linear_weights_utilities.shp"
        
        output_path = SHAPEFILE_OUTPUT_DIR / filename
        
        # Save shapefile
        output_gdf.to_file(output_path)
        
        print(f"    Utility scores shapefile saved: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"    Warning: Could not save utility scores shapefile: {e}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------------------------------------------------------
# SECTION: UTA-STAR Implementation
# -----------------------------------------------------------------------------

def get_breakpoints_for_alpha(alpha: int) -> np.ndarray:
    """Generate specific breakpoint patterns for each alpha value."""
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

def precompute_interpolation(X: np.ndarray, alpha: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute interpolation weights for piecewise linear utilities."""
    n, m = X.shape
    breakpoints = get_breakpoints_for_alpha(alpha)
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

def solve_uta_star_utilities(X: np.ndarray, preferred_idx: List[int], non_preferred_idx: List[int], 
                            alpha: int = 3, delta: float = 1e-3, print_header: bool = True) -> Optional[Dict]:
    """Solve UTA-STAR with piecewise linear utilities."""
    if print_header:
        print("\n" + "="*60)
        print("APPROACH 1: UTA-STAR WITH PIECEWISE LINEAR UTILITIES")
        print("="*60)
    
    n, m = X.shape
    W, breakpoints = precompute_interpolation(X, alpha)
    
    model = gp.Model("UTA-STAR-Utilities")
    model.Params.LogToConsole = 1  # Show Gurobi's internal progress
    model.Params.TimeLimit = 300  # 5 minutes
    
    # Variables: u[i,k] for utility values at breakpoints
    u = {}
    for i in range(m):
        for k in range(alpha + 1):
            u[i,k] = model.addVar(lb=0, ub=1, name=f"u_{i}_{k}")
    
    # Error variables
    sigma_plus = {}
    sigma_minus = {}
    for p_idx in preferred_idx:
        for q_idx in non_preferred_idx:
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
    for p_idx in preferred_idx:
        for q_idx in non_preferred_idx:
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

def solve_uta_star_weights(X: np.ndarray, preferred_idx: List[int], non_preferred_idx: List[int], 
                          delta: float = 1e-3) -> Optional[Dict]:
    """Solve for LINEAR WEIGHTS using UTA-STAR preference framework."""
    print("\n" + "="*60)
    print("APPROACH 2: LINEAR WEIGHTS WITH UTA-STAR PREFERENCES")
    print("="*60)
    
    n, m = X.shape
    
    model = gp.Model("UTA-STAR-Weights")
    model.Params.LogToConsole = 1  # Show Gurobi's internal progress
    model.Params.TimeLimit = 300  # 5 minutes
    
    # Variables: weights for each criterion
    w = {}
    for i in range(m):
        w[i] = model.addVar(lb=0, ub=1, name=f"w_{i}")
    
    # Error variables
    sigma_plus = {}
    sigma_minus = {}
    for p_idx in preferred_idx:
        for q_idx in non_preferred_idx:
            sigma_plus[p_idx,q_idx] = model.addVar(lb=0, name=f"sp_{p_idx}_{q_idx}")
            sigma_minus[p_idx,q_idx] = model.addVar(lb=0, name=f"sm_{p_idx}_{q_idx}")
    
    model.update()
    
    # Constraint: weights sum to 1
    model.addConstr(gp.quicksum(w[i] for i in range(m)) == 1)
    
    # Preference constraints with linear scoring
    for p_idx in preferred_idx:
        for q_idx in non_preferred_idx:
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

def evaluate_ranking(scores: np.ndarray, preferred_idx: List[int]) -> Dict:
    """Evaluate ranking performance."""
    if len(preferred_idx) == 0:
        return {
            'average_rank': float('nan'),
            'median_rank': float('nan'),
            'best_rank': -1,
            'worst_rank': -1,
            'top_500': 0,
            'top_1000': 0,
            'top_500_rate': 0.0,
            'top_1000_rate': 0.0,
        }
    
    # Calculate ranks (rank 1 = highest score)
    ranks = 1 + np.argsort(np.argsort(-scores))
    
    # Get preferred ranks
    preferred_ranks = ranks[preferred_idx]
    
    return {
        'average_rank': float(np.mean(preferred_ranks)),
        'median_rank': float(np.median(preferred_ranks)),
        'best_rank': int(np.min(preferred_ranks)),
        'worst_rank': int(np.max(preferred_ranks)),
        'top_500': int(np.sum(preferred_ranks <= 500)),
        'top_1000': int(np.sum(preferred_ranks <= 1000)),
        'top_500_rate': float(np.sum(preferred_ranks <= 500) / len(preferred_idx) * 100),
        'top_1000_rate': float(np.sum(preferred_ranks <= 1000) / len(preferred_idx) * 100)
    }

# -----------------------------------------------------------------------------
# SECTION: Visualization Functions
# -----------------------------------------------------------------------------

def plot_utility_functions_terminal(utility_result: Dict, dataset_name: str = "") -> None:
    """Render all piecewise utility functions in one figure with subplots."""
    if not utility_result:
        return

    breakpoints: List[float] = utility_result['breakpoints']
    alpha: int = int(utility_result['alpha'])

    num_criteria = len(CRITERIA_COLS)
    cols = min(3, num_criteria)
    rows = int(np.ceil(num_criteria / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), dpi=1200)
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
        ax.set_title(f"{friendly_label(col)}")
        ax.set_xlabel("criterion value (normalized)")
        ax.set_ylabel("utility")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.4)
        ax.legend()

    # Hide any unused subplots
    for j in range(num_criteria, len(axes)):
        axes[j].set_visible(False)

    suptitle = "Demo UTA-STAR Quantile: Utility functions per criterion"
    if dataset_name:
        suptitle += f" – {dataset_name}"
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=True)

def generate_score_map(parcels: gpd.GeoDataFrame, scores: np.ndarray, preferred_idx: List[int],
                       parcel_ids: List[str], title: str = "Score Map", subtitle: str = "",
                       output_name: str = "score_map", ranks: Dict = None, alpha_value: int = None) -> Optional[Path]:
    """Generate a choropleth map showing scores for each parcel."""
    try:
        # Create figure with proper aspect ratio and high DPI
        fig, ax = plt.subplots(figsize=(16, 12), dpi=1200)
        
        # Match parcel IDs to scores
        score_dict = {pid: score for pid, score in zip(parcel_ids, scores)}
        parcels = parcels.copy()
        parcels['score'] = parcels['parcel_id'].astype(str).map(score_dict).fillna(0)
        
        # Calculate ranks (1 = highest score)
        parcels['rank'] = parcels['score'].rank(method='min', ascending=False).astype(int)
        
        # Identify top 500 parcels
        top500_mask = parcels['rank'] <= 500
        
        # Identify preferred parcels
        preferred_parcel_ids = set(parcel_ids[i] for i in preferred_idx)
        parcels['is_preferred'] = parcels['parcel_id'].astype(str).isin(preferred_parcel_ids)
        
        # Create colormap (white to red)
        cmap = LinearSegmentedColormap.from_list('score_cmap', ['white', '#ff6666', '#cc0000', '#660000'])
        
        # Normalize scores for coloring
        vmin, vmax = 0.2, 0.8
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Plot all parcels with score coloring
        parcels.plot(column='score', ax=ax, cmap=cmap, norm=norm, 
                    edgecolor='none', linewidth=0)
        
        # Overlay top 500 parcels with blue outline
        if top500_mask.any():
            parcels[top500_mask].plot(ax=ax, facecolor='none', 
                                      edgecolor='blue', linewidth=0.3, alpha=0.9)

        # Mark each selection area with its own star
        try:
            selection_areas = ['painted_cave_quantile', 'san_roque_to_mission_canyon', 'above_E_mtn_drive']
            star_colors = ['yellow', 'orange', 'red']  # Different color per area
            
            for area_name, color in zip(selection_areas, star_colors):
                area_parcels = parcels.loc[parcels['subset'] == area_name, 'geometry']
                if len(area_parcels) > 0:
                    area_centroid = area_parcels.unary_union.centroid
                    ax.scatter([area_centroid.x], [area_centroid.y], marker='*', s=150,
                              color=color, edgecolors='black', linewidths=0.8, zorder=10)
        except Exception:
            pass

        # Clean up axes
        ax.set_axis_off()
        ax.set_frame_on(False)
        
        # Set axis limits to focus tightly on parcels data
        bounds = parcels.total_bounds
        margin = 0.02 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        ax.margins(0)
        
        # Remove all subplot padding/margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Display the map
        plt.show()
        plt.close(fig)
        
        return None
        
    except Exception as e:
        print(f"Error generating map: {e}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------------------------------------------------------
# SECTION: Main Execution
# -----------------------------------------------------------------------------

def main():
    """Main execution function for demo UTA-STAR analysis - QUANTILE ONLY."""
    print("="*80)
    print("DEMO UTA-STAR: WUI QUANTILE NORMALIZATION ONLY")
    print("="*80)
    
    # Load combined demo selections (preferred parcels)
    preferred_ids = load_demo_selections()
    if not preferred_ids:
        print("Error: No preferred parcels found in demo selections")
        return
    
    # Results storage
    all_results = []
    
    # Test each quantile dataset configuration
    for config in DATASET_CONFIGS:
        print("\n" + "#"*80)
        print(f"DATASET: {config['name']} ({config['description']})")
        print("#"*80)
        
        try:
            # Load dataset
            gdf = load_dataset(config)
            
            # Prepare data
            X, parcel_ids, preferred_idx, non_preferred_idx = prepare_demo_data(gdf, preferred_ids)
            
            if len(preferred_idx) == 0:
                print("⚠️  No preferred parcels found in this dataset, skipping...")
                continue
            
            # Test all alpha values for piecewise utilities
            utility_results_by_alpha = {}
            print(f"\nTesting alpha values {ALPHA_LIST} for piecewise utilities...")
            
            for idx_alpha, alpha in enumerate(ALPHA_LIST):
                start_time = time_module.time()
                utility_result = solve_uta_star_utilities(
                    X, preferred_idx, non_preferred_idx, alpha=alpha, 
                    print_header=(idx_alpha == 0)
                )
                solve_time = time_module.time() - start_time
                
                if utility_result:
                    utility_results_by_alpha[alpha] = {**utility_result, 'solve_time': solve_time}
                    print(f"\n✓ Alpha {alpha} solved in {solve_time:.2f}s | error={utility_result['total_error']:.4f} | violations={utility_result['violations']}")
                    
                    # Evaluate ranking performance for this alpha
                    scores_u = compute_utility_scores(X, utility_result['utilities'], 
                                                    np.array(utility_result['breakpoints']), alpha)
                    ranking_u = evaluate_ranking(scores_u, preferred_idx)
                    
                    bp_pattern = get_breakpoints_for_alpha(alpha)
                    bp_str = ', '.join([f"{bp:.2f}" for bp in bp_pattern])
                    print(f"  Ranking: avg={ranking_u['average_rank']:.1f} | min={ranking_u['best_rank']} | max={ranking_u['worst_rank']} | top500={ranking_u['top_500']} | breakpoints=[{bp_str}]")
                    
                    # Show utility functions for this alpha
                    print(f"  Rendering utility functions (alpha={alpha})...")
                    plot_utility_functions_terminal(utility_result, f"{config['name']} (alpha={alpha})")
                    
                    # Generate score map for this alpha
                    try:
                        print(f"  Generating score map (alpha={alpha})...")
                        generate_score_map(
                            gdf, scores_u, preferred_idx, parcel_ids,
                            title=f"Demo UTA-STAR Quantile Scores — {config['name']}",
                            subtitle=f"alpha={alpha} (segments={alpha})",
                            output_name=f"demo_quantile_map_{config['name']}_alpha{alpha}",
                            ranks=ranking_u,
                            alpha_value=alpha
                        )
                    except Exception as e:
                        print(f"    Warning: Could not generate map: {e}")
                    
                    # Save utility scores shapefile for this alpha
                    try:
                        print(f"  Saving utility scores shapefile (alpha={alpha})...")
                        save_utility_scores_shapefile(
                            gdf, scores_u, parcel_ids, config['name'], alpha, "piecewise"
                        )
                    except Exception as e:
                        print(f"    Warning: Could not save utility scores shapefile: {e}")
                    
                    plt.close('all')
                else:
                    print(f"\n✗ Alpha {alpha} optimization failed")
            
            # Find best alpha by average rank
            best_alpha = None
            best_avg_rank = float('inf')
            best_utility_ranking = None
            best_utility_scores = None
            
            if utility_results_by_alpha:
                alpha_summary = []
                for alpha, result in utility_results_by_alpha.items():
                    scores_u = compute_utility_scores(X, result['utilities'], 
                                                    np.array(result['breakpoints']), alpha)
                    ranking_u = evaluate_ranking(scores_u, preferred_idx)
                    
                    alpha_summary.append({
                        'alpha': alpha,
                        'avg_rank': ranking_u['average_rank'],
                        'min_rank': ranking_u['best_rank'],
                        'max_rank': ranking_u['worst_rank'],
                        'solve_time': result.get('solve_time', 0.0),
                        'error': result.get('total_error', 0.0),
                        'violations': result.get('violations', 0)
                    })
                    
                    if ranking_u['average_rank'] < best_avg_rank:
                        best_avg_rank = ranking_u['average_rank']
                        best_alpha = alpha
                        best_utility_ranking = ranking_u
                        best_utility_scores = scores_u
                
                # Display alpha comparison
                print("\nPiecewise utilities alpha comparison:")
                df_alpha = pd.DataFrame(alpha_summary).sort_values('alpha')
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
                    print(df_alpha.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
                print(f"\nBest alpha: {best_alpha} (avg rank {best_avg_rank:.1f})")
                
                # Generate convergence plot
                print(f"\nGenerating convergence analysis plot...")
                try:
                    create_convergence_plot(utility_results_by_alpha, config['name'])
                except Exception as e:
                    print(f"  Warning: Could not create convergence plot: {e}")
            
            # Test linear weights approach
            print(f"\nTesting linear weights approach...")
            start_time = time_module.time()
            weight_result = solve_uta_star_weights(X, preferred_idx, non_preferred_idx)
            weight_time = time_module.time() - start_time
            
            if weight_result:
                print(f"\n✓ Linear weights solved in {weight_time:.2f}s")
                print(f"  Total error: {weight_result['total_error']:.4f}")
                print(f"  Violations: {weight_result['violations']:,}")
                
                # Display weights
                print("\nOptimized Linear Weights:")
                for i, col in enumerate(CRITERIA_COLS):
                    print(f"  {col:10s}: {weight_result['weights'][i]:.1%}")
                
                # Evaluate ranking
                weight_scores = X @ weight_result['weights']
                weight_ranking = evaluate_ranking(weight_scores, preferred_idx)
                
                print(f"\nLinear Weights Ranking Performance:")
                print(f"  Average rank: {weight_ranking['average_rank']:.1f}")
                print(f"  Median rank:  {weight_ranking['median_rank']:.1f}")
                print(f"  Best rank:    {weight_ranking['best_rank']}")
                print(f"  Worst rank:   {weight_ranking['worst_rank']}")
                print(f"  Top 500:      {weight_ranking['top_500']}/{len(preferred_idx)} ({weight_ranking['top_500_rate']:.1f}%)")
                print(f"  Top 1000:     {weight_ranking['top_1000']}/{len(preferred_idx)} ({weight_ranking['top_1000_rate']:.1f}%)")
                
                # Generate final comparison map for linear weights
                try:
                    print(f"  Generating linear weights score map...")
                    generate_score_map(
                        gdf, weight_scores, preferred_idx, parcel_ids,
                        title=f"Demo Linear Weights Quantile Scores — {config['name']}",
                        subtitle="Linear weighted sum (alpha=1)",
                        output_name=f"demo_quantile_map_linear_{config['name']}",
                        ranks=weight_ranking,
                        alpha_value=1
                    )
                except Exception as e:
                    print(f"    Warning: Could not generate linear weights map: {e}")
                
                # Save linear weights utility scores shapefile
                try:
                    print(f"  Saving linear weights utility scores shapefile...")
                    save_utility_scores_shapefile(
                        gdf, weight_scores, parcel_ids, config['name'], 1, "linear"
                    )
                except Exception as e:
                    print(f"    Warning: Could not save linear weights shapefile: {e}")
            else:
                print(f"\n✗ Linear weights optimization failed")
                weight_ranking = None
                weight_scores = None
            
            # Compare approaches
            if best_utility_ranking and weight_ranking:
                print("\n" + "="*60)
                print("APPROACH COMPARISON")
                print("="*60)
                
                print(f"\nAverage Rank Performance:")
                print(f"  Piecewise Utilities (α={best_alpha}): {best_utility_ranking['average_rank']:8.1f}")
                print(f"  Linear Weights:                       {weight_ranking['average_rank']:8.1f}")
                
                improvement = (weight_ranking['average_rank'] - best_utility_ranking['average_rank']) / weight_ranking['average_rank'] * 100
                
                if improvement > 0:
                    print(f"\n✓ Utilities perform {improvement:.1f}% better than linear weights")
                else:
                    print(f"\n✗ Linear weights perform {-improvement:.1f}% better than utilities")
                
                print(f"\nTop 500 Performance:")
                print(f"  Piecewise Utilities: {best_utility_ranking['top_500_rate']:5.1f}%")
                print(f"  Linear Weights:      {weight_ranking['top_500_rate']:5.1f}%")
            
            # Store results for final summary
            if best_utility_ranking:
                all_results.append({
                    'dataset': config['name'],
                    'approach': 'piecewise',
                    'alpha': best_alpha,
                    'avg_rank': best_utility_ranking['average_rank'],
                    'min_rank': best_utility_ranking['best_rank'],
                    'max_rank': best_utility_ranking['worst_rank'],
                    'top_500': best_utility_ranking['top_500'],
                    'top_500_rate': best_utility_ranking['top_500_rate'],
                    'violations': utility_results_by_alpha.get(best_alpha, {}).get('violations', None)
                })
            
            if weight_ranking:
                all_results.append({
                    'dataset': config['name'],
                    'approach': 'linear',
                    'alpha': 1,
                    'avg_rank': weight_ranking['average_rank'],
                    'min_rank': weight_ranking['best_rank'],
                    'max_rank': weight_ranking['worst_rank'],
                    'top_500': weight_ranking['top_500'],
                    'top_500_rate': weight_ranking['top_500_rate'],
                    'violations': weight_result.get('violations', None)
                })
            
        except Exception as e:
            print(f"\n✗ Error processing dataset {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary across all datasets
    if all_results:
        print("\n" + "="*80)
        print("FINAL SUMMARY: QUANTILE DATASETS ONLY")
        print("="*80)
        
        df_results = pd.DataFrame(all_results)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
            print(df_results.to_string(index=False, float_format=lambda x: f"{x:.1f}" if isinstance(x, float) else x))
        
        # Save summary CSV
        csv_path = OUTPUTS_DIR / 'demo_utastar_quantile_summary.csv'
        try:
            df_results.to_csv(csv_path, index=False)
            print(f"\nResults summary saved to: {csv_path}")
        except Exception as e:
            print(f"Warning: Could not save summary CSV: {e}")
    
    print(f"\nDemo UTA-STAR Quantile analysis complete!")
    print(f"Preferred parcels analyzed: {len(preferred_ids)}")
    print(f"Quantile datasets tested: {len(DATASET_CONFIGS)}")

if __name__ == "__main__":
    main()