# MAIN OUTPUTS - Hazard Preference Disaggregation

This folder contains the essential outputs using parcels.shp geometry with computed risk data.

## Structure:

### 1_unlearned_parcels/
- **raw_minmax_wui_parcels**: Raw min-max normalized scoring for WUI parcels (~26K features)  
- **quantile_all_parcels**: Quantile normalized scoring (if available)
- These use unlearned/baseline scoring methods

### 2_subsets_parcels/
- **above_E_mtn_drive**: Small subset (~38 parcels)
- **san_antonio_creek**: Small subset (~76 parcels)
- **painted_cave_quantile**: Quantile-scored subset (~83 parcels) 
- **mission_canyon_montrose_quantile**: Quantile-scored subset (~150 parcels)
- Geographic subsets used for preference learning

### 3_utastar_wui_learned/
- **[subset]_alpha_[N]_risk_scores**: UTASTAR results using learned utilities
- Each alpha value represents different preference learning parameters
- Applied to all parcels using weights learned from WUI subsets
- Contains risk_score and risk_rank fields with learned preferences

## Key Fields:
- `risk_score`: Computed risk score using learned utilities
- `risk_rank`: Ranking based on risk scores  
- `score`: Original scoring (in subsets)
- `rank`: Original ranking (in subsets)
- `alpha`: Alpha parameter value used in learning

## Notes:
- All files use EPSG:3857 (Web Mercator) coordinate system
- Geometry from original parcels.shp for maximum compatibility
- Ready for analysis and visualization in ArcGIS/QGIS
