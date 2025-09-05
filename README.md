# NARSC Paper - Hazard Preference Disaggregation Research

## Project Structure

```
├── scripts/                    # All Python code
│   ├── core/                   # Core UTASTAR implementations
│   │   ├── UTASTAR.py         # Main UTA-STAR algorithm
│   │   ├── rank_min_MIP.py    # MIP-based ranking
│   │   └── rank_min_SA.py     # Simulated Annealing ranking
│   ├── demos/                  # Demo scripts
│   │   ├── demo_UTASTAR_quantile.py
│   │   ├── demo_UTASTAR_raw_minmax.py
│   │   └── demo_UTASTAR_single_subset.py
│   ├── validation/             # Validation and testing
│   │   ├── cross_area_validation.py
│   │   └── demo_validation_approach.py
│   └── utilities/              # Utility scripts
│       ├── create_diverse_subsets.py
│       ├── make_subset_user.py
│       └── topk_utastar_mission_wui.py
├── data/                       # Essential data only (gitignored - local only)
│   ├── parcels.shp            # Original parcels file (62,416 parcels)
│   ├── fire_risk_local_quantile_wui_parcels.shp     # Quantile WUI dataset (26,396 parcels)
│   ├── fire_risk_local_quantile_all_parcels.shp    # Quantile all parcels (62,416 parcels)
│   ├── fire_risk_local_raw_minmax_all_parcels.shp  # Raw min-max all parcels (62,416 parcels)
│   ├── parcels_subset.shp     # Core script subset (17 parcels)
│   └── subsets/               # Key subset selections
│       ├── painted_cave_quantile.shp         # 83 parcels
│       ├── mission_canyon_montrose_quantile.shp # 150 parcels
│       ├── above_E_mtn_drive.shp             # 38 parcels
│       ├── san_antonio_creek.shp             # Contains WUI parcels
│       └── subset_[1-9].shp                  # Numbered subsets
├── outputs/                    # Generated outputs (gitignored)
├── archive/                    # Archived old data (gitignored)
└── gurobi.lic                 # Gurobi license file
```

## Setup Instructions

### Data Setup
1. **Data files are local only** - Each machine needs its own data directory
2. **Essential data files** should be copied to `data/` directory:
   - Main parcels file: `parcels.shp` (original source data)
   - Scored versions: `fire_risk_local_*_parcels.shp` files
   - Subsets: Placed in `data/subsets/` directory

### Prerequisites
- Python 3.x with geopandas, pandas, numpy, matplotlib
- Gurobi optimization library with valid license
- Shapefile data properly derived from `parcels.shp`

### Running Scripts
All scripts should be run from the project root directory:
```bash
python3 scripts/demos/demo_UTASTAR_quantile.py
python3 scripts/core/UTASTAR.py
```

## Git Usage
- Only code files are tracked in Git
- Data files, outputs, and archives are gitignored
- Safe to clone on multiple machines - just add local data directory

## Data Organization
- **Clean data/ directory**: Contains only essential files used by scripts (7.1GB total)
- **Archived unused data**: 8.4GB of reference data safely stored in `archive/`
- **All shapefiles verified**: Properly derived from original `parcels.shp` with valid geometry
- **Three main datasets**: Raw min-max (62K parcels), Quantile all (62K parcels), Quantile WUI (26K parcels)
- **Four key subsets**: Essential area selections for preference learning