#!/usr/bin/env python3
"""
Create diverse subset selections by sampling from high-risk areas with variety.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# Path configuration
BASE_PATH = Path("/Users/theofunk/Desktop/Narsc Paper")
DATA_DIR = BASE_PATH / "data"
SUBSETS_DIR = DATA_DIR / "subsets"
MAPS_DIR = BASE_PATH / "extras" / "subset_maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)

# Criteria columns for diversity
CRITERIA_COLS = ['qtrmi_s', 'hvhsz_s', 'hagfb_s', 'hbrn_s', 'slope_s', 'hwui_s']

def create_subset_map(parcels, subset, subset_name, config):
    """Create a map showing the spatial distribution of a subset."""
    print(f"    Creating map for {subset_name}...")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Map 1: Subset location in context
    # Show all parcels in light gray
    parcels_sample = parcels.sample(n=min(5000, len(parcels)))  # Sample for performance
    parcels_sample.plot(ax=ax1, color='lightgray', alpha=0.3, markersize=0.1)
    
    # Highlight the subset in red
    subset.plot(ax=ax1, color='red', alpha=0.8, markersize=2)
    
    # Add title and labels
    ax1.set_title(f'{subset_name} - Spatial Distribution\n'
                 f'Strategy: {config["strategy"]}, Size: {len(subset)} parcels')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_aspect('equal')
    
    # Map 2: Dominant criterion visualization
    if config['primary']:
        primary_col = config['primary']
        
        # Create color map for the primary criterion
        vmin, vmax = parcels[primary_col].quantile([0.1, 0.9])
        
        # Show all parcels colored by primary criterion
        parcels_sample.plot(ax=ax2, column=primary_col, cmap='YlOrRd', 
                           alpha=0.6, markersize=0.5, vmin=vmin, vmax=vmax)
        
        # Highlight subset with black borders
        subset.plot(ax=ax2, facecolor='none', edgecolor='black', 
                   linewidth=2, alpha=1.0)
        
        ax2.set_title(f'{subset_name} - {primary_col.upper()} Values\n'
                     f'Selected parcels outlined in black')
    else:
        # For diverse/moderate strategies, show multiple criteria
        # Create a composite risk score
        risk_cols = [f'{col}_norm' for col in CRITERIA_COLS]
        if all(col in parcels.columns for col in risk_cols):
            composite_risk = parcels[risk_cols].mean(axis=1)
            vmin, vmax = composite_risk.quantile([0.1, 0.9])
            
            parcels_sample.plot(ax=ax2, column=composite_risk.loc[parcels_sample.index], 
                               cmap='YlOrRd', alpha=0.6, markersize=0.5, vmin=vmin, vmax=vmax)
            subset.plot(ax=ax2, facecolor='none', edgecolor='black', 
                       linewidth=2, alpha=1.0)
            
            ax2.set_title(f'{subset_name} - Composite Risk Score\n'
                         f'Selected parcels outlined in black')
        else:
            # Fallback: just show locations
            parcels_sample.plot(ax=ax2, color='lightgray', alpha=0.3, markersize=0.1)
            subset.plot(ax=ax2, color='red', alpha=0.8, markersize=2)
            ax2.set_title(f'{subset_name} - Selected Locations')
    
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_aspect('equal')
    
    # Remove axis ticks for cleaner look
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save map
    map_path = MAPS_DIR / f"{subset_name}_map.png"
    plt.savefig(map_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved map: {map_path}")
    
    # Also calculate and report contiguity statistics
    centroids = subset.geometry.centroid
    if len(centroids) > 1:
        distances = []
        for i, cent in enumerate(centroids):
            other_distances = [cent.distance(other) for j, other in enumerate(centroids) if j != i]
            if other_distances:
                distances.append(min(other_distances))
        
        if distances:
            print(f"    Contiguity stats: min={min(distances):.0f}, max={max(distances):.0f}, "
                  f"mean={np.mean(distances):.0f} units")

def create_diverse_subsets():
    """Create 9 diverse subsets with different characteristics."""
    print("Loading parcels data...")
    parcels = gpd.read_file(DATA_DIR / "parcels.shp")
    print(f"Total parcels: {len(parcels):,}")
    
    # Filter to parcels with valid criteria data
    valid_parcels = parcels.dropna(subset=CRITERIA_COLS)
    print(f"Parcels with valid criteria: {len(valid_parcels):,}")
    
    # Calculate composite risk score (weighted sum)
    for col in CRITERIA_COLS:
        valid_parcels[f'{col}_norm'] = (valid_parcels[col] - valid_parcels[col].min()) / (valid_parcels[col].max() - valid_parcels[col].min())
    
    # Create different subset strategies
    subset_configs = [
        # Subset 1: High fire burn risk focus
        {'name': 'subset_1', 'size': 17, 'primary': 'hbrn_s', 'secondary': 'slope_s', 'strategy': 'high_primary'},
        
        # Subset 2: High vegetation/fuel focus  
        {'name': 'subset_2', 'size': 24, 'primary': 'hagfb_s', 'secondary': 'hvhsz_s', 'strategy': 'high_primary'},
        
        # Subset 3: Mixed high hazard
        {'name': 'subset_3', 'size': 24, 'primary': 'hvhsz_s', 'secondary': 'qtrmi_s', 'strategy': 'balanced'},
        
        # Subset 4: Steep slopes with WUI
        {'name': 'subset_4', 'size': 43, 'primary': 'slope_s', 'secondary': 'hwui_s', 'strategy': 'high_both'},
        
        # Subset 5: Distance-based risk (qtrmi)
        {'name': 'subset_5', 'size': 39, 'primary': 'qtrmi_s', 'secondary': 'hagfb_s', 'strategy': 'high_primary'},
        
        # Subset 6: Diverse mix - all criteria
        {'name': 'subset_6', 'size': 24, 'primary': None, 'secondary': None, 'strategy': 'diverse'},
        
        # Subset 7: WUI-focused areas
        {'name': 'subset_7', 'size': 17, 'primary': 'hwui_s', 'secondary': 'hbrn_s', 'strategy': 'high_primary'},
        
        # Subset 8: Extreme single factor (for contrast)
        {'name': 'subset_8', 'size': 21, 'primary': 'hagfb_s', 'secondary': None, 'strategy': 'extreme'},
        
        # Subset 9: Balanced moderate risk
        {'name': 'subset_9', 'size': 38, 'primary': None, 'secondary': None, 'strategy': 'moderate'}
    ]
    
    # Backup existing subsets
    backup_dir = DATA_DIR / "subsets_backup"
    backup_dir.mkdir(exist_ok=True)
    
    print(f"\nBacking up existing subsets to {backup_dir}...")
    for file in SUBSETS_DIR.glob("*.*"):
        if file.is_file():
            backup_path = backup_dir / file.name
            if backup_path.exists():
                backup_path.unlink()
            file.rename(backup_path)
    
    print("\nCreating new diverse subsets...")
    print("="*60)
    
    for config in subset_configs:
        print(f"\nCreating {config['name']} (size={config['size']}, strategy={config['strategy']})")
        
        # Apply selection strategy
        if config['strategy'] == 'high_primary':
            # High values in primary criterion, moderate in secondary
            mask = valid_parcels[f"{config['primary']}_norm"] > 0.7
            if config['secondary']:
                mask &= valid_parcels[f"{config['secondary']}_norm"] > 0.3
            candidates = valid_parcels[mask]
            
        elif config['strategy'] == 'high_both':
            # High values in both criteria
            mask = (valid_parcels[f"{config['primary']}_norm"] > 0.6) & \
                   (valid_parcels[f"{config['secondary']}_norm"] > 0.6)
            candidates = valid_parcels[mask]
            
        elif config['strategy'] == 'balanced':
            # Balanced high values across primary and secondary
            score = valid_parcels[f"{config['primary']}_norm"] * 0.5 + \
                   valid_parcels[f"{config['secondary']}_norm"] * 0.5
            candidates = valid_parcels[score > 0.6]
            
        elif config['strategy'] == 'extreme':
            # Extreme values in one criterion only
            mask = valid_parcels[f"{config['primary']}_norm"] > 0.9
            candidates = valid_parcels[mask]
            
        elif config['strategy'] == 'diverse':
            # Sample with diversity - different combinations
            # Calculate variance across criteria
            criteria_values = valid_parcels[[f'{col}_norm' for col in CRITERIA_COLS]]
            variance = criteria_values.var(axis=1)
            # Select high variance parcels (diverse risk factors)
            candidates = valid_parcels[variance > variance.quantile(0.7)]
            
        elif config['strategy'] == 'moderate':
            # Moderate risk across multiple factors
            criteria_values = valid_parcels[[f'{col}_norm' for col in CRITERIA_COLS]]
            mean_risk = criteria_values.mean(axis=1)
            # Select moderate risk (40-70 percentile)
            mask = (mean_risk > mean_risk.quantile(0.4)) & (mean_risk < mean_risk.quantile(0.7))
            candidates = valid_parcels[mask]
        
        else:
            candidates = valid_parcels
        
        print(f"  Candidates after filtering: {len(candidates)}")
        
        # If not enough candidates, relax criteria
        if len(candidates) < config['size'] * 2:
            print(f"  Relaxing criteria to get more candidates...")
            candidates = valid_parcels.sample(n=min(config['size'] * 5, len(valid_parcels)))
        
        # Select spatially contiguous parcels
        if len(candidates) >= config['size']:
            # Start with a random seed parcel
            seed_idx = np.random.choice(len(candidates))
            seed_parcel = candidates.iloc[[seed_idx]]
            selected = seed_parcel
            
            # Build KDTree for efficient spatial queries
            coords = np.column_stack([
                candidates.geometry.centroid.x.values,
                candidates.geometry.centroid.y.values
            ])
            tree = cKDTree(coords)
            
            # Grow selection spatially
            while len(selected) < config['size'] and len(selected) < len(candidates):
                # Find nearest unselected parcels
                selected_coords = np.column_stack([
                    selected.geometry.centroid.x.values,
                    selected.geometry.centroid.y.values
                ])
                
                # Query nearest neighbors
                distances, indices = tree.query(selected_coords, k=min(10, len(candidates)))
                
                # Flatten and get unique indices
                neighbor_indices = np.unique(indices.flatten())
                
                # Remove already selected
                selected_mask = candidates.index.isin(selected.index)
                available_indices = [i for i in neighbor_indices 
                                   if not selected_mask[i]]
                
                if available_indices:
                    # Select next parcel (prefer diverse criteria values)
                    next_candidates = candidates.iloc[available_indices[:min(5, len(available_indices))]]
                    
                    # Choose based on diversity from current selection
                    if len(selected) > 1 and config['strategy'] == 'diverse':
                        current_mean = selected[CRITERIA_COLS].mean()
                        diversity = ((next_candidates[CRITERIA_COLS] - current_mean) ** 2).sum(axis=1)
                        next_idx = diversity.idxmax()
                    else:
                        next_idx = next_candidates.index[0]
                    
                    selected = pd.concat([selected, candidates.loc[[next_idx]]])
                else:
                    break
            
            # Trim to exact size
            selected = selected.iloc[:config['size']]
            
            # Save subset
            output_path = SUBSETS_DIR / f"{config['name']}.shp"
            selected.to_file(output_path)
            
            # Create map
            create_subset_map(valid_parcels, selected, config['name'], config)
            
            # Report statistics
            print(f"  ✓ Created {config['name']}: {len(selected)} parcels")
            
            # Show criteria distribution
            means = selected[CRITERIA_COLS].mean()
            stds = selected[CRITERIA_COLS].std()
            print(f"    Criteria means: {dict(means.round(3))}")
            print(f"    Criteria stds:  {dict(stds.round(3))}")
            
            # Show top 2 criteria
            top_2 = means.nlargest(2)
            print(f"    Dominant: {top_2.index[0]} ({top_2.iloc[0]:.3f}), "
                  f"Secondary: {top_2.index[1]} ({top_2.iloc[1]:.3f})")
        else:
            print(f"  ✗ Not enough candidates for {config['name']}")
    
    print("\n" + "="*60)
    print("✓ New diverse subsets created successfully!")
    print(f"✓ Saved to: {SUBSETS_DIR}")
    print("\nYou can now run UTASTAR.py to analyze the new subsets.")

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    create_diverse_subsets()