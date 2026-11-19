"""ThermalEye Pipeline Experimental Validation Module.

Conducts empirical sensitivity analyses and statistical validation on PostGIS data:
  1. ST-DBSCAN parameter sensitivity sweep across spatial (1-5 km) and temporal (12-72 hrs) regimes.
  2. Priority scoring validation, including OSM infrastructure ablation, Spearman rank correlation,
     and Precision@K for persistent high-consequence thermal events.
All metrics are dynamically evaluated against live PostGIS database records.
"""

from __future__ import annotations
import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import numpy as np
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
from scipy import stats
from scipy.sparse import csgraph
from sklearn.neighbors import NearestNeighbors

from classify_landcover import get_database_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("thermaleye.ml_validation")

ARTIFACTS_DIR = WORKSPACE_DIR / "artifacts"


def run_st_dbscan_sweep(engine, sample_limit: int = 3000) -> Dict[str, Any]:
    """Execute Spatio-Temporal DBSCAN sensitivity analysis across epsilon ranges."""
    logger.info("Executing ST-DBSCAN parameter sensitivity sweep on FIRMS detections...")

    # Load detections with timestamps and geometries
    query = f"""
    SELECT
        id::text AS detection_id,
        ST_Y(geom) AS latitude,
        ST_X(geom) AS longitude,
        acq_datetime,
        frp
    FROM raw_firms_detections
    ORDER BY acq_datetime DESC
    LIMIT {sample_limit}
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        logger.warning("No FIRMS detections found in database.")
        return {"total_samples": 0, "sweep_results": []}

    # Convert to GeoDataFrame and project to meters (EPSG:7755 - India)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=7755)

    coords_xy = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    timestamps_sec = pd.to_datetime(gdf["acq_datetime"]).astype("int64").values // 10**9

    eps_km_list = [1.0, 2.0, 3.0, 5.0]
    eps_hours_list = [12, 24, 48, 72]

    # Pre-index spatial nearest neighbors with maximum search radius (5000m)
    max_radius_m = max(eps_km_list) * 1000.0
    nn = NearestNeighbors(radius=max_radius_m, metric="euclidean")
    nn.fit(coords_xy)
    spatial_adj = nn.radius_neighbors_graph(coords_xy, mode="distance").tocoo()

    sweep_records: List[Dict[str, Any]] = []

    for eps_km in eps_km_list:
        spatial_eps_m = eps_km * 1000.0
        for eps_hours in eps_hours_list:
            temporal_eps_sec = eps_hours * 3600.0

            # Filter spatial neighbors within current spatial_eps_m
            row_idx = spatial_adj.row
            col_idx = spatial_adj.col
            dist_vals = spatial_adj.data

            valid_spatial_mask = (dist_vals <= spatial_eps_m) & (row_idx < col_idx)
            src_nodes = row_idx[valid_spatial_mask]
            dst_nodes = col_idx[valid_spatial_mask]

            # Filter edges by temporal difference
            time_diffs = np.abs(timestamps_sec[src_nodes] - timestamps_sec[dst_nodes])
            valid_temporal_mask = time_diffs <= temporal_eps_sec

            final_src = src_nodes[valid_temporal_mask]
            final_dst = dst_nodes[valid_temporal_mask]

            # Construct symmetric adjacency graph
            n_points = len(gdf)
            adj_data = np.ones(len(final_src) * 2, dtype=np.int32)
            adj_rows = np.concatenate([final_src, final_dst])
            adj_cols = np.concatenate([final_dst, final_src])

            from scipy.sparse import csr_matrix
            st_graph = csr_matrix((adj_data, (adj_rows, adj_cols)), shape=(n_points, n_points))

            # Connected components / clusters
            n_clusters, labels = csgraph.connected_components(st_graph, directed=False)

            # Analyze cluster properties
            cluster_sizes = pd.Series(labels).value_counts().values
            singleton_clusters = int(np.sum(cluster_sizes == 1))
            multi_point_clusters = int(np.sum(cluster_sizes > 1))
            multi_point_sizes = cluster_sizes[cluster_sizes > 1]

            mean_cluster_size = round(float(np.mean(multi_point_sizes)), 2) if len(multi_point_sizes) > 0 else 1.0
            max_cluster_size = int(np.max(cluster_sizes)) if len(cluster_sizes) > 0 else 1
            noise_ratio = round(singleton_clusters / n_points, 4)

            # Calculate mean cluster temporal duration
            durations: List[float] = []
            for lab in np.unique(labels):
                pts_t = timestamps_sec[labels == lab]
                if len(pts_t) > 1:
                    durations.append(float((pts_t.max() - pts_t.min()) / 3600.0))
            mean_duration_hrs = round(float(np.mean(durations)), 2) if durations else 0.0

            sweep_records.append({
                "eps_km": eps_km,
                "eps_hours": eps_hours,
                "total_clusters": int(n_clusters),
                "multi_point_clusters": multi_point_clusters,
                "noise_ratio": noise_ratio,
                "mean_points_per_cluster": mean_cluster_size,
                "max_points_in_cluster": max_cluster_size,
                "mean_cluster_duration_hours": mean_duration_hrs,
            })

    return {
        "sample_points_evaluated": len(gdf),
        "parameter_grid_size": len(sweep_records),
        "sweep": sweep_records,
    }


def compute_priority_scoring_validation(engine) -> Dict[str, Any]:
    """Empirically validate priority scoring against persistence, intensity, and OSM ablation."""
    logger.info("Computing Priority Scoring ablation, correlation, and Precision@K...")

    query = """
    SELECT
        id::text AS event_id,
        priority_score,
        priority_reasons,
        max_frp,
        detection_count,
        COALESCE(EXTRACT(EPOCH FROM (last_seen - first_seen)) / 3600.0, 0.0) AS persistence_hours,
        nearest_osm_name,
        nearest_osm_type,
        nearest_osm_distance_km
    FROM candidate_events
    WHERE priority_score IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        logger.warning("No scored candidate events found in database.")
        return {"events_evaluated": 0}

    # Extract or ablate OSM contribution from priority score
    # Heuristic scoring assigns up to 30 pts for proximity to OSM industrial infrastructure
    ablated_scores: List[float] = []
    osm_bonuses: List[float] = []

    for _, row in df.iterrows():
        reasons = row["priority_reasons"] or {}
        dist_km = row["nearest_osm_distance_km"]
        if isinstance(reasons, dict) and "infrastructure_proximity_score" in reasons:
            infra_pts = float(reasons["infrastructure_proximity_score"])
        elif pd.notna(dist_km) and dist_km < 5.0:
            infra_pts = max(0.0, 30.0 * (1.0 - (dist_km / 5.0)))
        else:
            infra_pts = 0.0

        full_score = float(row["priority_score"])
        ablated_score = max(0.0, full_score - infra_pts)
        ablated_scores.append(ablated_score)
        osm_bonuses.append(infra_pts)

    df["ablated_score"] = ablated_scores
    df["osm_infra_bonus"] = osm_bonuses

    # 1. Spearman Rank Correlations
    rho_score_frp, p_score_frp = stats.spearmanr(df["priority_score"], df["max_frp"])
    rho_score_persist, p_score_persist = stats.spearmanr(df["priority_score"], df["persistence_hours"])
    rho_score_count, p_score_count = stats.spearmanr(df["priority_score"], df["detection_count"])

    rho_ablated_frp, _ = stats.spearmanr(df["ablated_score"], df["max_frp"])
    rho_ablated_persist, _ = stats.spearmanr(df["ablated_score"], df["persistence_hours"])

    # 2. Precision@K for persistent high-consequence thermal events
    # Ground truth criterion: persistent detection_count >= 2 and max_frp >= 20.0 MW
    ground_truth_target = (df["detection_count"] >= 2) & (df["max_frp"] >= 20.0)
    df["is_high_consequence"] = ground_truth_target

    total_positive_targets = int(ground_truth_target.sum())
    precision_at_k: Dict[str, Any] = {}

    df_full_sorted = df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
    df_ablated_sorted = df.sort_values(by="ablated_score", ascending=False).reset_index(drop=True)

    for k in [10, 25, 50, 100]:
        k_eval = min(k, len(df))
        prec_full = float(df_full_sorted.iloc[:k_eval]["is_high_consequence"].mean())
        prec_ablated = float(df_ablated_sorted.iloc[:k_eval]["is_high_consequence"].mean())

        precision_at_k[f"top_{k}"] = {
            "k": k_eval,
            "full_pipeline_precision": round(prec_full, 4),
            "osm_ablated_precision": round(prec_ablated, 4),
            "osm_precision_lift": round(prec_full - prec_ablated, 4),
        }

    return {
        "events_evaluated": len(df),
        "total_high_consequence_events": total_positive_targets,
        "spearman_rank_correlation": {
            "priority_vs_max_frp": {
                "rho": round(float(rho_score_frp), 4),
                "p_value": float(f"{p_score_frp:.4e}"),
            },
            "priority_vs_persistence_hours": {
                "rho": round(float(rho_score_persist), 4),
                "p_value": float(f"{p_score_persist:.4e}"),
            },
            "priority_vs_detection_count": {
                "rho": round(float(rho_score_count), 4),
                "p_value": float(f"{p_score_count:.4e}"),
            },
            "ablated_vs_max_frp": {
                "rho": round(float(rho_ablated_frp), 4),
            },
            "ablated_vs_persistence_hours": {
                "rho": round(float(rho_ablated_persist), 4),
            },
        },
        "precision_at_k": precision_at_k,
    }


def main() -> None:
    """Run full experimental validation and export artifacts/experimental_validation.json."""
    engine = create_engine(get_database_url(), echo=False)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    st_sweep = run_st_dbscan_sweep(engine)
    scoring_val = compute_priority_scoring_validation(engine)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "st_dbscan_sensitivity_analysis": st_sweep,
        "priority_scoring_validation": scoring_val,
    }

    out_file = ARTIFACTS_DIR / "experimental_validation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Successfully exported experimental validation artifact -> %s", out_file)
    print(f"Validation Artifact Exported -> {out_file}")


if __name__ == "__main__":
    main()
