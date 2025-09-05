
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

# Path configuration
BASE_PATH = Path("/Users/theofunk/Desktop/Narsc Paper")
DATA_DIR = BASE_PATH / "data"
SUBSETS_DIR = DATA_DIR / "subsets"
OUTPUTS_DIR = BASE_PATH / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SUBSET_1_OUTPUT_DIR = OUTPUTS_DIR / "subset_1"
SUBSET_1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Criteria columns (include optional WUI if present; prefer 'hwui_s')
CRITERIA_COLS = ['qtrmi_s', 'hvhsz_s', 'agfb_s', 'hbrn_s', 'slope_s', 'hwui_s', 'fbfm_s']
# Candidate numbers of segments (alpha) for piecewise utilities
# Note: alpha = segments; interior breakpoints = alpha - 1
# Order: Linear (1), 1 middle bp (2), 2 inner bps (3), 3 inner bps (4)
ALPHA_LIST: list[int] = [1, 2, 3, 4]

# Reference set configuration
# Toggle to use a spatially stratified sample for the non-subset reference set.
# If False, uses ALL non-subset parcels.
USE_REFERENCE_SAMPLE: bool = True
REFERENCE_SAMPLE_SIZE: int = 10000
REFERENCE_SAMPLE_SEED: int = 42

# Friendly display names for plotting and human-readable outputs
FRIENDLY_LABELS: Dict[str, str] = {
    # Structures within quarter mile
    'qtrmi_s': 'Count of Structures within a Quarter Mile',
    'qmtrmi_s': 'Count of Structures within a Quarter Mile',  # alias just in case
    # Very High Hazard Severity Zones within half mile
    'hvhsz_s': 'Very High Hazard Severity Zones % within a Half Mile',
    'hvsz_s': 'Very High Hazard Severity Zones % within a Half Mile',      # alias
    # Agriculture and Fuel Break within half mile
    'agfb_s': 'Agriculture and Fuel Break % within a Half Mile',
    # Burn scar within half mile
    'hbrn_s': 'Burn Scar % within a Half Mile',
    # WUI coverage within half mile
    'hwui_s': 'WUI Coverage % within a Half Mile',
    # Slope (unit unspecified; using percent as a friendly default)
    'slope_s': 'Slope (%)',
    # Fire Behavior Fuel Model (normalized)
    'fbfm_s': 'Fire Behavior Fuel Model (normalized)'
}

def friendly_label(column_name: str) -> str:
    """Return a human-friendly display label for a criterion column name."""
    return FRIENDLY_LABELS.get(column_name, column_name)

# -----------------------------------------------------------------------------
# SECTION: Data Loading
# -----------------------------------------------------------------------------

def load_all_subsets() -> Tuple[gpd.GeoDataFrame, List[Tuple[str, gpd.GeoDataFrame]]]:
    """Load parcels and all subset data from subsets folder."""
    print("Loading data...")
    parcels = gpd.read_file(DATA_DIR / "parcels.shp")
    
    # Find all subset shapefiles
    subset_files = []
    if SUBSETS_DIR.exists():
        subset_files = list(SUBSETS_DIR.glob("*.shp"))
    
    # Also check for legacy files in main data directory
    legacy_files = [
        DATA_DIR / "parcels_subset.shp",
        DATA_DIR / "subset_1.shp"
    ]
    for f in legacy_files:
        if f.exists():
            subset_files.append(f)
    
    if not subset_files:
        raise FileNotFoundError("No subset files found in subsets folder or main data directory")
    
    # Load all subsets
    subsets = []
    for subset_file in sorted(subset_files):
        subset_name = subset_file.stem  # filename without extension
        subset_gdf = gpd.read_file(subset_file)
        subsets.append((subset_name, subset_gdf))
        print(f"  Loaded subset '{subset_name}': {len(subset_gdf):,} parcels")
    
    print(f"  Full dataset: {len(parcels):,} parcels")
    print(f"  Total subsets found: {len(subsets)}")
    
    return parcels, subsets

def load_data() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load parcels and subset data (legacy single subset function)."""
    parcels, subsets = load_all_subsets()
    if subsets:
        # Return first subset for backward compatibility
        return parcels, subsets[0][1]
    else:
        raise FileNotFoundError("No subsets found")

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

    # Ensure fbfm_s exists; detect likely fbfm column and min-max normalize if numeric
    if 'fbfm_s' not in working_parcels.columns:
        try:
            # Find a column that starts with 'fbfm' (case-insensitive) or equals 'fbfm'
            fbfm_cols = [c for c in working_parcels.columns if str(c).lower().startswith('fbfm')]
            if not fbfm_cols and 'fbfm' in working_parcels.columns:
                fbfm_cols = ['fbfm']
            if fbfm_cols:
                col = fbfm_cols[0]
                # Try numeric; if categorical/string, map to codes stably
                series = pd.to_numeric(working_parcels[col], errors='coerce')
                if series.notna().sum() > 0:
                    vals = series.fillna(series.min())
                    vmin = float(vals.min())
                    vmax = float(vals.max())
                    working_parcels['fbfm_s'] = (vals - vmin) / (vmax - vmin) if vmax > vmin else 0.0
                else:
                    # Categorical fallback: factorize then scale
                    codes, _ = pd.factorize(working_parcels[col].astype(str).str.strip().str.lower())
                    if len(codes) > 0:
                        vmin = float(np.min(codes))
                        vmax = float(np.max(codes))
                        working_parcels['fbfm_s'] = (codes - vmin) / (vmax - vmin) if vmax > vmin else 0.0
                    else:
                        working_parcels['fbfm_s'] = 0.0
            else:
                working_parcels['fbfm_s'] = 0.0
        except Exception:
            working_parcels['fbfm_s'] = 0.0
    
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
    # Choose reference set: full non-subset or spatially stratified 10k sample
    if USE_REFERENCE_SAMPLE:
        sampled_labels = stratified_spatial_sample(working_parcels, list(subset_ids), target_size=REFERENCE_SAMPLE_SIZE, seed=REFERENCE_SAMPLE_SEED)
        # sampled_labels are GeoDataFrame index labels; map them back to positional indices in working_parcels
        sampled_label_set = set(sampled_labels)
        non_subset_idx = [i for i, (pid, idx_label) in enumerate(zip(parcel_ids, working_parcels.index)) if (pid not in subset_ids and idx_label in sampled_label_set)]
        print(f"  Subset: {len(subset_idx)} parcels")
        print(f"  Non-subset reference (sampled): {len(non_subset_idx):,} parcels (target={REFERENCE_SAMPLE_SIZE:,})")
    else:
        # Use ALL parcels outside the subset as the reference set
        non_subset_idx = [i for i, pid in enumerate(parcel_ids) if pid not in subset_ids]
        print(f"  Subset: {len(subset_idx)} parcels")
        print(f"  Non-subset reference (all): {len(non_subset_idx):,} parcels")
    
    return X, X, parcel_ids, subset_idx, non_subset_idx

# -----------------------------------------------------------------------------
# SECTION: UTA-STAR (Piecewise Utilities) – helpers
# -----------------------------------------------------------------------------

def get_breakpoints_for_alpha(alpha: int) -> np.ndarray:
    """
    Generate specific breakpoint patterns for each alpha value:
    alpha=1: Linear - 1 segment (0, 1) with 2 breakpoints only at start and end
    alpha=2: One middle bp (0, 0.5, 1) 
    alpha=3: Two inner bps (0, 0.33, 0.67, 1)
    alpha=4: Three inner bps (0, 0.25, 0.5, 0.75, 1)
    Note: alpha represents segments, so we need alpha+1 breakpoints
    """
    if alpha == 1:
        # Linear: 1 segment, 2 breakpoints at 0 and 1 only
        return np.array([0.0, 1.0])
    elif alpha == 2:
        # 2 segments: add breakpoint at middle (0.5)
        return np.array([0.0, 0.5, 1.0])
    elif alpha == 3:
        # 3 segments: add breakpoints at 1/3 and 2/3
        return np.array([0.0, 1.0/3.0, 2.0/3.0, 1.0])
    elif alpha == 4:
        # 4 segments: breakpoints at quarters
        return np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        # Fallback to uniform spacing for other values
        return np.linspace(0, 1, alpha + 1)

def generate_score_map(parcels: gpd.GeoDataFrame, scores: np.ndarray, subset_idx: List[int],
                       parcel_ids: List[str], title: str = "Score Map", subtitle: str = "",
                       output_name: str = "score_map", ranks: Dict = None, alpha_value: int = None) -> Optional[Path]:
    """
    Generate a choropleth map showing scores for each parcel.
    - Colors parcels by score (red=high, white=low)
    - Outlines top 500 scoring parcels in blue
    - Marks subset centroid with a star
    - Shows ranking statistics in legend
    - Notes the alpha value used
    """
    try:
        # Create figure with proper aspect ratio - larger to focus on parcels
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Match parcel IDs to scores
        score_dict = {pid: score for pid, score in zip(parcel_ids, scores)}
        parcels = parcels.copy()
        parcels['score'] = parcels['parcel_id'].astype(str).map(score_dict).fillna(0)
        
        # Calculate ranks (1 = highest score)
        parcels['rank'] = parcels['score'].rank(method='min', ascending=False).astype(int)
        
        # Identify top 500 parcels
        top500_mask = parcels['rank'] <= 500
        
        # Identify subset parcels
        subset_parcel_ids = set(parcel_ids[i] for i in subset_idx)
        parcels['is_subset'] = parcels['parcel_id'].astype(str).isin(subset_parcel_ids)
        
        # Create colormap (white to red, matching your image)
        cmap = LinearSegmentedColormap.from_list('score_cmap', ['white', '#ff6666', '#cc0000', '#660000'])
        
        # Normalize scores for coloring (0.2 to 0.8 range to match your image)
        vmin, vmax = 0.2, 0.8
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Plot all parcels with score coloring
        parcels.plot(column='score', ax=ax, cmap=cmap, norm=norm, 
                    edgecolor='none', linewidth=0)
        
        # Overlay top 500 parcels with a fine blue outline
        if top500_mask.any():
            parcels[top500_mask].plot(ax=ax, facecolor='none', 
                                      edgecolor='blue', linewidth=0.3, alpha=0.9)

        # Mark subset centroid with a small star for reference
        try:
            subset_geoms = parcels.loc[parcels['is_subset'], 'geometry']
            if len(subset_geoms) > 0:
                centroid = subset_geoms.unary_union.centroid
                ax.scatter([centroid.x], [centroid.y], marker='*', s=120,
                          color='yellow', edgecolors='black', linewidths=0.7, zorder=10)
        except Exception:
            pass

        # Remove colorbar/legend to focus on parcels
        
        # No title; plot parcels only
        
        # Remove statistics box to focus on parcels only
        
        # Clean up axes
        ax.set_axis_off()
        ax.set_frame_on(False)
        
        # Set axis limits to focus tightly on parcels data
        bounds = parcels.total_bounds
        margin = 0.02 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])  # 2% margin
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        ax.margins(0)
        
        # Remove all subplot padding/margins so only parcels are visible
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

def stratified_spatial_sample(parcels_gdf: gpd.GeoDataFrame, subset_ids: List[str], target_size: int = 10000, seed: int = 42) -> List[int]:
    """
    Create stratified spatial sample of parcels outside the subset.
    Uses grid-based stratification for geographic representation (mirrors UTASTAR_clean).
    Returns a list of GeoDataFrame index labels for the sampled reference parcels.
    """
    np.random.seed(seed)

    subset_id_set = set(map(str, subset_ids))
    candidates = parcels_gdf[~parcels_gdf['parcel_id'].astype(str).isin(subset_id_set)].copy()

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

def precompute_interpolation(X: np.ndarray, alpha: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute interpolation weights for piecewise linear utilities.
    Returns: W matrix and breakpoints
    """
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
    # Handle empty subset case
    if len(subset_idx) == 0:
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

    suptitle = "Utility functions per criterion"
    if subset_label:
        suptitle += f" – {subset_label}"
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=True)

    pass

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
        nice_name = friendly_label(name)
        lines.append(f"{nice_name}(x) =")
        for k in range(alpha):
            x0, x1 = float(breakpoints[k]), float(breakpoints[k+1])
            u0, u1 = float(vals[k]), float(vals[k+1])
            dx = (x1 - x0) if (x1 - x0) != 0 else 1.0
            m = (u1 - u0) / dx
            b = u0 - m * x0
            lines.append(f"  if {x0:.2f} <= x <= {x1:.2f}:  u(x) = {m:.4f} * x + {b:.4f}")
        lines.append("")
    return "\n".join(lines)

def save_risk_scores_shapefile(parcels: gpd.GeoDataFrame, scores: np.ndarray, 
                               parcel_ids: List[str], alpha: int, subset_name: str,
                               utility_result: Dict, output_dir: Path) -> Optional[Path]:
    """
    Save risk scores for all parcels as a shapefile for a specific alpha value.
    Returns path to saved shapefile or None if failed.
    """
    try:
        # Create a copy of the parcels GeoDataFrame
        output_gdf = parcels.copy()
        
        # Map scores to parcels
        score_dict = {pid: score for pid, score in zip(parcel_ids, scores)}
        output_gdf['risk_score'] = output_gdf['parcel_id'].astype(str).map(score_dict).fillna(0)
        
        # Calculate ranks (1 = highest score/risk)
        output_gdf['risk_rank'] = output_gdf['risk_score'].rank(method='min', ascending=False).astype(int)
        
        # Add alpha information
        output_gdf['alpha'] = alpha
        output_gdf['subset'] = subset_name
        
        # Add utility function endpoint values as additional attributes
        if utility_result and 'utilities' in utility_result:
            for i, col in enumerate(CRITERIA_COLS):
                if str(i) in utility_result['utilities']:
                    utils = utility_result['utilities'][str(i)]
                    if utils:
                        # Add the final utility value (weight) for this criterion
                        output_gdf[f'{col}_weight'] = utils[-1]
        
        # Create output filename
        output_file = output_dir / f"{subset_name}_risk_scores_alpha{alpha}.shp"
        
        # Save shapefile
        output_gdf.to_file(output_file)
        
        print(f"    Risk scores shapefile saved: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"    Warning: Could not save risk scores shapefile for alpha {alpha}: {e}")
        return None

# -----------------------------------------------------------------------------
# SECTION: Report rendering helpers (HTML)
# -----------------------------------------------------------------------------

def _fig_to_data_uri(fig: Figure) -> str:
    """Encode a Matplotlib figure as a data URI (PNG) for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)
    return f"data:image/png;base64,{b64}"

def make_utility_plot_image(utility_result: Dict, subset_label: str | None = None) -> str:
    """Render the utility functions plot to a base64 data URI (does not show)."""
    if not utility_result:
        return ""
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
        for x in breakpoints:
            ax.axvline(x, color='lightgray', linestyle='--', linewidth=1)
        ax.plot(breakpoints, utils, marker='o', color='cyan', label='utility')
        for k, (x, u) in enumerate(zip(breakpoints, utils)):
            ax.annotate(f"BP{k}: x={x:.2f}\nu={u:.2f}",
                        xy=(x, u), xytext=(4, 6), textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.8))
        ax.plot([breakpoints[0], breakpoints[-1]], [utils[0], utils[-1]], color='gold', linestyle='--', label='linear ref')
        ax.set_title(f"{friendly_label(col)}")
        ax.set_xlabel("criterion value (normalized)")
        ax.set_ylabel("utility")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.4)
        ax.legend()
    for j in range(num_criteria, len(axes)):
        axes[j].set_visible(False)
    suptitle = "Utility functions per criterion"
    if subset_label:
        suptitle += f" – {subset_label}"
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    return _fig_to_data_uri(fig)

def make_weights_bar_image(weights_by_criterion: Dict[str, float], title: str) -> str:
    """Render a simple bar chart of criterion weights to a base64 data URI."""
    criteria = list(weights_by_criterion.keys())
    values = [float(weights_by_criterion[c]) for c in criteria]
    labels = [friendly_label(c) for c in criteria]
    x = np.arange(len(criteria))
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(criteria)), 4))
    ax.bar(x, values, color='steelblue')
    ax.set_ylim(0, 1)
    ax.set_ylabel('weight')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    for i, v in enumerate(values):
        ax.text(x[i], v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    return _fig_to_data_uri(fig)

def build_html_report(report_sections: List[Dict], final_table_html: str, overview_table_html: str, output_path: Path) -> Path:
    """Assemble a single self-contained HTML report and write to output_path."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    css = """
    <style>
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #222; }
    h1, h2, h3 { margin: 0.6em 0 0.4em; }
    .muted { color: #666; }
    .section { margin-bottom: 36px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; }
    img { max-width: 100%; height: auto; border: 1px solid #eee; }
    table { border-collapse: collapse; width: 100%; }
    table, th, td { border: 1px solid #ddd; }
    th, td { padding: 6px 8px; }
    pre { background: #f7f7f7; padding: 12px; border-radius: 6px; overflow-x: auto; }
    .kpi { display: inline-block; margin-right: 16px; padding: 6px 10px; background: #f1f5ff; border: 1px solid #d5e0ff; border-radius: 6px; }
    </style>
    """
    header = f"""
    <h1>UTA-STAR Report</h1>
    <div class='muted'>Generated {ts}</div>
    <div class='section card'>
      <h2>Overview</h2>
      <p>This report compares two approaches under the same preference framework: (1) piecewise linear utilities (standard UTA-STAR) and (2) a linear weighted-sum baseline. It summarizes performance, utilities, and weights per subset, with plots and equations.</p>
      <h3>How this differs from a pure weighted-sum</h3>
      <p>UTA-STAR estimates per-criterion utility functions that can be non-linear, allowing different marginal returns across the criterion range. The linear baseline constrains utilities to u_i(x)=w_i·x, i.e., straight lines through the origin with slope w_i. When utilities are strongly non-linear, UTA-STAR can fit preferences better.</p>
      <h3>Benefits</h3>
      <ul>
        <li>Captures non-linear preferences per criterion via breakpoints.</li>
        <li>Respects monotonicity and normalizes total utility.</li>
        <li>Comparable linear model provides interpretability and a speed baseline.</li>
      </ul>
    </div>
    """
    body_parts = [css, header]
    for sec in report_sections:
        body_parts.append("""
        <div class='section card'>
          <h2>Subset: {subset}</h2>
          <div class='muted'>Best alpha: {alpha} | Breakpoints: {bp}</div>
          <div>
            <span class='kpi'>AvgRank (Utilities): {avg_u:.2f}</span>
            <span class='kpi'>AvgRank (Linear): {avg_w:.2f}</span>
            <span class='kpi'>Top500 (U): {top500_u:.1f}%</span>
            <span class='kpi'>Top500 (W): {top500_w:.1f}%</span>
          </div>
          <div class='grid'>
            <div>
              <h3>Piecewise utility functions</h3>
              <img src='{util_img}' alt='utility plot'/>
            </div>
            <div>
              <h3>Linear weights</h3>
              <img src='{w_img}' alt='weights bar'/>
            </div>
          </div>
          <h3>Piecewise equations</h3>
          <pre>{equations}</pre>
        </div>
        """.format(
            subset=sec.get('subset'), alpha=sec.get('alpha'), bp=sec.get('bp_str', ''),
            avg_u=sec.get('avg_rank_u', float('nan')), avg_w=sec.get('avg_rank_w', float('nan')),
            top500_u=sec.get('top500_u', float('nan')), top500_w=sec.get('top500_w', float('nan')),
            util_img=sec.get('utility_img', ''), w_img=sec.get('weights_img', ''),
            equations=sec.get('equations', '')
        ))
    body_parts.append("""
    <div class='section card'>
      <h2>Final Summary Table (All Subsets)</h2>
      {table_html}
    </div>
    <div class='section card'>
      <h2>Friendly Overview</h2>
      {overview_html}
    </div>
    """.format(table_html=final_table_html, overview_html=overview_table_html))
    html = """
    <!DOCTYPE html>
    <html><head><meta charset='utf-8'><title>UTA-STAR Report</title></head>
    <body>{body}</body></html>
    """.format(body='\n'.join(body_parts))
    output_path.write_text(html, encoding='utf-8')
    return output_path

def main():
    """Main execution function."""
    print("="*80)
    print("UTA-STAR COMPARISON: UTILITIES vs LINEAR WEIGHTS")
    print("="*80)
    
    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    parcels, all_subsets = load_all_subsets()
    # Map export disabled
    final_rows = []
    combined_subset_ids: set[str] = set()

    for subset_label, subset in all_subsets:
        if 'parcel_id' in subset.columns:
            combined_subset_ids.update(subset['parcel_id'].astype(str).tolist())

        print("\n" + "#"*80)
        print(f"SUBSET: {subset_label}")
        print("#"*80)
        
        # Skip empty subsets
        if len(subset) == 0:
            print(f"⚠️  Skipping empty subset '{subset_label}' (0 parcels)")
            continue

        # Prepare data
        X, X_norm, parcel_ids, subset_idx, non_subset_idx = prepare_data(parcels, subset)

        # Optional: Explore nearby contiguous clusters around this subset to find ~20 parcel clusters with mixed weights
        # We use a simple contiguity-based cluster search when the current subset size is not ~20.
        # This mode can be extended; currently we just proceed with the provided subset.

        # ------------------------------------------------------------------
        # UTA-STAR (Piecewise Utilities) — multiple alphas
        # ------------------------------------------------------------------
        utility_results_by_alpha: dict[int, dict] = {}
        per_alpha_util_rows: list[dict] = []
        per_alpha_rank_rows: list[dict] = []
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
                bp_pattern = get_breakpoints_for_alpha(alph)
                bp_str = ', '.join([f"{bp:.2f}" for bp in bp_pattern])
                print(f"  Ranking (alpha={alph}): avg={r_u_this['average_rank']:.1f} | min={r_u_this['best_rank']} | max={r_u_this['worst_rank']} | top500={r_u_this['top_500']} | breakpoints=[{bp_str}]")
                # Plot utility functions for this alpha with breakpoint annotations
                print("  Rendering utility functions (alpha={})...".format(alph))
                plot_utility_functions_terminal(ures, subset_label=f"{subset_label} (alpha={alph})")
                
                # Generate and show choropleth map for this alpha immediately after utility plot
                try:
                    label_norm = str(subset_label).strip().lower()
                    # Generate map for all subsets, not just subset_1
                    if True:  # Enable for all subsets
                        print(f"  Generating score map (alpha={alph})...")
                        map_path = generate_score_map(
                            parcels, scores_u_this, subset_idx, parcel_ids,
                            title=f"Piecewise Utility Score — {subset_label}",
                            subtitle=f"alpha={alph} (segments={alph})",
                            output_name=f"map_piecewise_{subset_label}_alpha{alph}",
                            ranks=r_u_this,
                            alpha_value=alph
                        )
                        if map_path:
                            print(f"    Map saved: {map_path}")
                except Exception as e:
                    print(f"    Warning: Could not generate map: {e}")
                
                # Save risk scores shapefile for subset_1 only
                try:
                    label_norm = str(subset_label).strip().lower()
                    is_subset_one = (label_norm == 'subset_1') or label_norm.endswith('_1') or label_norm == 'parcels_subset'
                    if is_subset_one:
                        print(f"  Saving risk scores shapefile (alpha={alph})...")
                        shapefile_path = save_risk_scores_shapefile(
                            parcels, scores_u_this, parcel_ids, alph, subset_label, 
                            ures, SUBSET_1_OUTPUT_DIR
                        )
                except Exception as e:
                    print(f"    Warning: Could not save risk scores shapefile: {e}")
                
                # Close plots after both utility and map are shown
                plt.close('all')
                # Collect rows for per-alpha utilities table (one row per criterion)
                try:
                    bps = list(ures['breakpoints'])
                    for i, col in enumerate(CRITERIA_COLS):
                        utils = list(ures['utilities'][str(i)])
                        row = {'alpha': alph, 'criterion': col}
                        for k, (x, uval) in enumerate(zip(bps, utils)):
                            row[f'BP{k}'] = float(uval)
                        per_alpha_util_rows.append(row)
                except Exception:
                    pass
                # Collect ranking stats per alpha
                per_alpha_rank_rows.append({
                    'alpha': alph,
                    'avg': r_u_this['average_rank'],
                    'median': r_u_this['median_rank'],
                    'best': r_u_this['best_rank'],
                    'worst': r_u_this['worst_rank'],
                    'top500': r_u_this['top_500'],
                    'top1000': r_u_this['top_1000'],
                })
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
                bp_pattern = get_breakpoints_for_alpha(alph)
                bp_str = ', '.join([f"{bp:.2f}" for bp in bp_pattern])
                rows.append({'alpha': alph, 'avg': r_u['average_rank'], 'min': r_u['best_rank'], 'max': r_u['worst_rank'], 'time': ures.get('solve_time', 0.0), 'error': ures.get('total_error', 0.0), 'breakpoints': f'[{bp_str}]'})
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
            
            # Show breakpoint patterns for each alpha
            print("\nBreakpoint patterns used:")
            for alph in sorted(utility_results_by_alpha.keys()):
                bp_pattern = get_breakpoints_for_alpha(alph)
                bp_str = ', '.join([f"{bp:.2f}" for bp in bp_pattern])
                print(f"  alpha {alph}: [{bp_str}]")
        else:
            print("\n✗ Utility optimization failed for all alphas")
            utility_ranking = None
        # Print requested final tables for subset 1: utilities per alpha and ranking stats per alpha
        try:
            label_norm = str(subset_label).strip().lower()
            is_subset_one = (label_norm == 'subset_1') or label_norm.endswith('_1') or label_norm == 'parcels_subset'
        except Exception:
            is_subset_one = False
        if is_subset_one and (per_alpha_util_rows or per_alpha_rank_rows):
            if per_alpha_util_rows:
                print("\n" + "-"*80)
                print("UTILITY FUNCTIONS BY ALPHA (Subset 1)")
                print("-"*80)
                df_utils_alpha = pd.DataFrame(per_alpha_util_rows)
                # Order columns: alpha, criterion, BP0..BPn
                bp_cols = sorted([c for c in df_utils_alpha.columns if c.startswith('BP')], key=lambda x: int(x[2:]))
                df_utils_alpha = df_utils_alpha[['alpha', 'criterion'] + bp_cols].sort_values(['alpha','criterion'])
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
                    print(df_utils_alpha.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            if per_alpha_rank_rows:
                print("\n" + "-"*80)
                print("RANKING STATS BY ALPHA (Subset 1)")
                print("-"*80)
                df_rank_alpha = pd.DataFrame(per_alpha_rank_rows).sort_values('alpha')
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
                    print(df_rank_alpha.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        
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
                'utilities': ures_best,  # includes raw utilities indexed by criterion number
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

            # Generate final comparison maps for best alpha
            if best_alpha and utility_scores is not None and weight_scores is not None:
                print(f"\nGenerating final comparison maps...")
                
                # Piecewise utility map
                map_path_u = generate_score_map(
                    parcels, utility_scores, subset_idx, parcel_ids,
                    title=f"Piecewise Utility Score — {subset_label}",
                    subtitle=f"Best alpha={best_alpha}",
                    output_name=f"map_final_piecewise_{subset_label}",
                    ranks=utility_ranking,
                    alpha_value=best_alpha
                )
                if map_path_u:
                    print(f"  Piecewise map saved: {map_path_u}")
                
                # Linear weights map  
                map_path_w = generate_score_map(
                    parcels, weight_scores, subset_idx, parcel_ids,
                    title=f"Linear Weighted Score — {subset_label}",
                    subtitle="Linear weights (alpha=1)",
                    output_name=f"map_final_linear_{subset_label}",
                    ranks=weight_ranking,
                    alpha_value=1
                )
                if map_path_w:
                    print(f"  Linear map saved: {map_path_w}")
            
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
        
        if ures_best and weight_result:
            
            # Check if utilities are mostly linear or have strong non-linearities
            max_nonlinearity = 0
            for i in range(len(CRITERIA_COLS)):
                utils = ures_best['utilities'][str(i)]
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

        # Build HTML report sections while we have detailed context
        report_sections: List[Dict] = []
        # We will regenerate per-subset visuals using stored JSON files if available; otherwise, skip images.
        try:
            for fn in sorted(OUTPUTS_DIR.glob('uta_star_comparison_*.json')):
                try:
                    subset_name = fn.stem.replace('uta_star_comparison_', '')
                    with open(fn, 'r') as f:
                        js = json.load(f)
                    ures = js.get('utilities') or {}
                    util_img = make_utility_plot_image(ures, subset_label=subset_name) if ures else ''
                    weights_by_criterion = js.get('weights', {}).get('by_criterion', {})
                    w_img = make_weights_bar_image(weights_by_criterion, f"Linear weights – {subset_name}") if weights_by_criterion else ''
                    eq = format_piecewise_equations(ures, CRITERIA_COLS) if ures else ''
                    avg_u = float(js.get('utility_ranking', {}).get('average_rank', float('nan')))
                    avg_w = float(js.get('weight_ranking', {}).get('average_rank', float('nan')))
                    top500_u = float(js.get('utility_ranking', {}).get('top_500_rate', float('nan')))
                    top500_w = float(js.get('weight_ranking', {}).get('top_500_rate', float('nan')))
                    bp = ures.get('breakpoints') if isinstance(ures, dict) else None
                    if bp:
                        if isinstance(bp, list):
                            bp_str = ', '.join([f"{float(val):.2f}" for val in bp])
                        else:
                            bp_str = f"{float(bp):.2f}"
                    else:
                        bp_str = ''
                    report_sections.append({
                        'subset': subset_name,
                        'alpha': int((ures or {}).get('alpha', 0)),
                        'bp_str': f"[{bp_str}]" if bp_str else '',
                        'utility_img': util_img,
                        'weights_img': w_img,
                        'equations': eq,
                        'avg_rank_u': avg_u,
                        'avg_rank_w': avg_w,
                        'top500_u': top500_u,
                        'top500_w': top500_w,
                    })
                except Exception:
                    continue
        except Exception:
            report_sections = []

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

        # Build HTML report after we have both tables
        try:
            final_table_html = df.to_html(index=False)
            overview_table_html = df_overview.to_html(index=False)
        except Exception:
            final_table_html = df.to_html(index=False)
            overview_table_html = ""
        report_path = OUTPUTS_DIR / 'uta_star_report.html'
        try:
            build_html_report(report_sections, final_table_html, overview_table_html, report_path)
            print(f"HTML report written to: {report_path}")
        except Exception as e:
            print(f"Warning: failed to write HTML report: {e}")

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
            derived_weights_from_utilities = {col: float(ures_best['utilities'][str(i)][-1]) for i, col in enumerate(CRITERIA_COLS)} if ures_best else {}
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