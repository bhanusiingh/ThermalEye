import os
import uuid
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY as PG_ARRAY
from sklearn.cluster import DBSCAN
from dotenv import load_dotenv

# Walk up to find .env if not loaded
_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_script_dir, ".env")
load_dotenv(dotenv_path=_env_path)


def get_database_url() -> str:
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://thermaleye:thermaleye_secret@localhost:5432/thermaleye",
    )
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # Prioritize psycopg2 driver prefix for GeoPandas compatibility
    if "+" not in url.split("://")[0]:
        try:
            import psycopg2  # psycopg v2
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        except ImportError:
            try:
                import psycopg  # psycopg v3 fallback
                url = url.replace("postgresql://", "postgresql+psycopg://", 1)
            except ImportError:
                pass
    return url


DB_URL = get_database_url()


def run_clustering():
    """
    Cluster unassigned thermal detections using DBSCAN (2 km radius),
    insert candidate_events, and link detections via candidate_event_id.

    Guarantees transactional consistency: event creation and detection
    linking happen within a single database transaction.
    """
    engine = create_engine(DB_URL)

    # 1. Load unclustered raw detections
    query = "SELECT * FROM raw_firms_detections WHERE candidate_event_id IS NULL"
    gdf = gpd.read_postgis(query, engine, geom_col="geom")

    if gdf.empty:
        print("No new detections to cluster.")
        return 0, 0

    # 2. Reproject to India's local CRS (EPSG:7755) for accurate distance in meters
    gdf = gdf.to_crs(epsg=7755)

    # 3. Spatial Clustering (DBSCAN) - 2km radius
    coords = gdf.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
    db = DBSCAN(eps=2000, min_samples=1).fit(coords)
    gdf["cluster_id"] = db.labels_

    # 4. Process Clusters and Insert into candidate_events
    gdf = gdf.to_crs(epsg=4326)  # Reproject back to WGS84

    insert_stmt = text(
        """
        INSERT INTO candidate_events 
        (centroid, footprint, first_seen, last_seen, max_frp, detection_count)
        VALUES 
        (ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 
         ST_GeomFromText(:footprint, 4326),
         :first_seen, :last_seen, :max_frp, :detection_count)
        RETURNING id
        """
    )

    update_stmt = text(
        """
        UPDATE raw_firms_detections 
        SET candidate_event_id = :event_id 
        WHERE id = ANY(CAST(:point_ids AS uuid[]))
        """
    ).bindparams(
        bindparam("event_id", type_=PG_UUID(as_uuid=True)),
        bindparam("point_ids", type_=PG_ARRAY(PG_UUID(as_uuid=True))),
    )

    total_events = 0
    total_linked = 0

    # Entire clustering run is wrapped in a single transaction for atomicity
    with engine.begin() as conn:
        for cluster_id, group in gdf.groupby("cluster_id"):
            centroid = group.geometry.unary_union.centroid
            first_seen = group["acq_datetime"].min()
            last_seen = group["acq_datetime"].max()
            raw_max_frp = group["frp"].max()
            max_frp = float(raw_max_frp) if pd.notna(raw_max_frp) else None
            detection_count = len(group)

            footprint_wkt = None
            if len(group) >= 3:
                hull = group.geometry.unary_union.convex_hull
                if hull.geom_type == "Polygon":
                    footprint_wkt = hull.wkt

            # Insert candidate event
            result = conn.execute(
                insert_stmt,
                {
                    "lon": float(centroid.x),
                    "lat": float(centroid.y),
                    "footprint": footprint_wkt,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "max_frp": max_frp,
                    "detection_count": int(detection_count),
                },
            )
            event_id = result.fetchone()[0]
            event_uuid = (
                event_id
                if isinstance(event_id, uuid.UUID)
                else uuid.UUID(str(event_id))
            )

            # Ensure all point_ids are Python uuid.UUID objects
            point_ids = [
                pid if isinstance(pid, uuid.UUID) else uuid.UUID(str(pid))
                for pid in group["id"].tolist()
            ]

            # Update raw detections with candidate_event_id FK
            update_res = conn.execute(
                update_stmt,
                {"event_id": event_uuid, "point_ids": point_ids},
            )
            total_events += 1
            total_linked += update_res.rowcount

    print(
        f"Successfully clustered {total_linked} points into {total_events} candidate events."
    )
    return total_events, total_linked


if __name__ == "__main__":
    run_clustering()
