#!/usr/bin/env python3
"""
Simplified Exact Rank Minimization using MIP
"""

import os
import numpy as np
import geopandas as gpd
import gurobipy as gp
from gurobipy import GRB

# --- Config ---
CRITERIA_COLS = ['qtrmi_s', 'hvhsz_s', 'agfb_s', 'hbrn_s', 'slope_s']
DATA_DIR = "/Users/theofunk/Desktop/NARSC paper/data"
MAX_PARCELS = None  # use full dataset by default

def normalize_columns(df, cols):
    """Min-max normalize selected columns in-place"""
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
            continue
        vals = df[col].astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            df[col] = (vals - vmin) / (vmax - vmin)
        else:
            df[col] = 0.0

def load_and_prepare_data():
    """Load shapefiles and return normalized matrix + selected subset indices"""
    parcels = gpd.read_file(os.path.join(DATA_DIR, "parcels.shp"))
    # Prefer new naming; fallback to legacy
    subset_path = os.path.join(DATA_DIR, "subset_1.shp")
    if not os.path.exists(subset_path):
        subset_path = os.path.join(DATA_DIR, "parcels_subset.shp")
    subset = gpd.read_file(subset_path)
    
    if isinstance(MAX_PARCELS, int) and MAX_PARCELS > 0:
        parcels = parcels.head(MAX_PARCELS).copy()
    subset_ids = set(subset['parcel_id'].astype(str))
    parcels['parcel_id'] = parcels['parcel_id'].astype(str)

    normalize_columns(parcels, CRITERIA_COLS)

    X = parcels[CRITERIA_COLS].values
    parcel_ids = parcels['parcel_id'].tolist()
    selected_idx = [i for i, pid in enumerate(parcel_ids) if pid in subset_ids]

    if len(selected_idx) == 0:
        raise ValueError("No selected parcels found in subset!")

    return X, selected_idx, parcel_ids

def solve_mip_exact_rank(X, selected_idx, epsilon=1e-6):
    n, m = X.shape
    model = gp.Model("ExactRankMinimization")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 60)
    model.setParam('MIPGap', 1e-6)

    # Variables: weights w_j
    w = model.addVars(m, lb=0.0, ub=1.0, name="w")

    # Binary y_qi = 1 if q outranks i
    y = {}
    for i in selected_idx:
        for q in range(n):
            if q != i:
                y[q, i] = model.addVar(vtype=GRB.BINARY, name=f"y_{q}_{i}")

    # Weights must sum to 1
    model.addConstr(gp.quicksum(w[j] for j in range(m)) == 1)

    # Ranking constraints
    for i in selected_idx:
        for q in range(n):
            if q == i:
                continue
            S_q = gp.quicksum(w[j] * X[q, j] for j in range(m))
            S_i = gp.quicksum(w[j] * X[i, j] for j in range(m))
            model.addConstr(S_q - S_i - epsilon <= 1.0 * y[q, i])

    # Objective: minimize total outrankers
    model.setObjective(gp.quicksum(y[q, i] for (q, i) in y), GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        weights = np.array([w[j].X for j in range(m)])
        return weights, model.objVal
    else:
        raise RuntimeError("Optimization did not succeed")

def evaluate_ranks(X, weights, selected_idx, epsilon=1e-6):
    scores = X @ weights
    ranks = []
    for i in selected_idx:
        outranked_by = np.sum(scores >= scores[i] + epsilon)
        ranks.append(1 + outranked_by)
    return ranks

def main():
    print("=== Exact Rank Minimization (Simplified) ===")
    X, selected_idx, parcel_ids = load_and_prepare_data()

    print(f"Loaded {X.shape[0]} parcels, {X.shape[1]} features")
    print(f"Selected subset size: {len(selected_idx)}")

    weights, total_outrankers = solve_mip_exact_rank(X, selected_idx)
    ranks = evaluate_ranks(X, weights, selected_idx)

    print("\n--- Results ---")
    print("Optimal Weights:")
    for j, col in enumerate(CRITERIA_COLS):
        print(f"  {col:10s}: {weights[j]:.4f}")
    print(f"\nTotal outrankers: {int(total_outrankers)}")
    print(f"Average rank: {np.mean(ranks):.2f}")
    print("Individual ranks:", ranks)

if __name__ == "__main__":
    main()