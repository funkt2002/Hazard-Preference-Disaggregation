#!/usr/bin/env python3
"""
Create a subset shapefile from a user-provided list of parcel IDs.

Reads data/parcels.shp and writes data/subset_user.shp with only
the requested parcel geometries. Any IDs not found are reported.
"""

from pathlib import Path
import geopandas as gpd

BASE_PATH = Path("/Users/theofunk/Desktop/NARSC paper")
DATA_DIR = BASE_PATH / "data"

# Paste the list below (only the first token per line is used as parcel_id)
RAW_LIST = """
p_58848	18.609	0.814	22.527	0.228	63.916	0.639	33.000	0.094	26.410	0.592	0.655
p_58845.0	17.745	0.177	23.568	0.239	71.590	0.716	45.000	0.130	17.290	0.387	0.592
p_57930	18.285	0.817	20.981	0.213	55.080	0.551	21.000	0.059	34.420	0.772	0.587
p_57897.0	17.685	0.177	23.774	0.241	70.616	0.706	42.000	0.121	10.920	0.244	0.583
p_57909.0	17.987	0.180	23.191	0.235	70.082	0.701	44.000	0.127	8.000	0.179	0.580
p_57968.0	17.951	0.180	23.185	0.235	69.687	0.697	42.000	0.121	8.880	0.199	0.576
p_57890.0	17.755	0.178	23.314	0.236	68.150	0.682	40.000	0.115	8.640	0.193	0.564
p_58846.0	18.340	0.183	22.710	0.230	67.174	0.672	42.000	0.121	10.160	0.227	0.558
p_57903.0	18.308	0.183	22.766	0.231	66.072	0.661	39.000	0.112	11.260	0.252	0.549
p_57970.0	18.120	0.181	22.653	0.230	66.121	0.661	39.000	0.112	10.820	0.242	0.549
p_57946.0	17.391	0.174	23.445	0.238	65.622	0.656	35.000	0.101	14.320	0.321	0.544
p_57895.0	18.306	0.183	22.767	0.231	64.596	0.646	37.000	0.107	8.590	0.192	0.537
p_57902.0	17.584	0.176	23.076	0.234	63.292	0.633	34.000	0.098	11.650	0.261	0.526
p_57925.0	18.271	0.183	22.150	0.225	62.435	0.624	31.000	0.089	24.980	0.560	0.521
p_57894.0	17.881	0.179	22.585	0.229	61.146	0.611	32.000	0.092	10.650	0.238	0.510
p_57932.0	18.181	0.182	22.028	0.223	60.494	0.605	28.000	0.081	23.390	0.524	0.506
p_57937.0
"""


def parse_ids(raw: str) -> list[str]:
    ids: list[str] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Take only the first token on the line
        tok = line.split()[0]
        # Normalize trailing .0 if present
        if tok.endswith(".0"):
            tok = tok[:-2]
        ids.append(tok)
    # Deduplicate preserving order
    seen = set()
    out = []
    for pid in ids:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def main():
    parcels_path = DATA_DIR / "parcels.shp"
    out_path = DATA_DIR / "subset_user.shp"

    print(f"Reading: {parcels_path}")
    gdf = gpd.read_file(parcels_path)
    gdf["parcel_id"] = gdf["parcel_id"].astype(str)

    ids = parse_ids(RAW_LIST)
    print(f"Requested {len(ids)} parcel_ids")

    sub = gdf[gdf["parcel_id"].isin(ids)].copy()
    found = set(sub["parcel_id"].tolist())
    missing = [pid for pid in ids if pid not in found]
    print(f"Found {len(sub)} / {len(ids)}")
    if missing:
        print("Missing IDs:")
        for pid in missing:
            print(f"  - {pid}")

    if len(sub) == 0:
        print("No matching parcels; not writing output.")
        return

    print(f"Writing: {out_path}")
    sub.to_file(out_path)
    print("Done.")


if __name__ == "__main__":
    main()

