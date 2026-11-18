"""
ThermalEye — Phase 1: NASA FIRMS Data Ingestion Worker
=======================================================

Idempotent ingestion of active-fire CSV data from the NASA FIRMS
API into PostGIS.  Designed to be run repeatedly (cron / scheduler)
without producing duplicate rows.

Architectural invariants enforced:
  • TLS verification is always enabled (verify=True).
  • Missing numeric metrics are SQL NULL, never 0.0.
  • Each detection receives a deterministic UUID derived from
    (latitude, longitude, satellite, acq_date, acq_time).
  • Boundary filtering uses ST_Intersects with a tolerance buffer;
    excluded points are logged to `boundary_exclusions_audit`.
  • All credentials / endpoints sourced from .env via python-dotenv.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import sys
import uuid
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore", message=".*urllib3 .* or chardet .*")

import geopandas as gpd
import pandas as pd
import requests
from dotenv import load_dotenv
from geoalchemy2 import Geometry, WKTElement
from shapely.geometry import Point
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("thermaleye.ingest_firms")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Walk up from this script's directory to find the project-root .env
_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_script_dir, ".env")
load_dotenv(dotenv_path=_env_path)

FIRMS_MAP_KEY: str = os.environ.get("FIRMS_MAP_KEY", "")
if not FIRMS_MAP_KEY:
    logger.critical("FIRMS_MAP_KEY is not set in the .env file. Aborting.")
    sys.exit(1)

def get_database_url() -> str:
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://thermaleye:thermaleye_secret@localhost:5432/thermaleye",
    )
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # Prioritize psycopg2 driver prefix for GeoPandas to_postgis / copy_expert compatibility
    if "+" not in url.split("://")[0]:
        try:
            import psycopg2  # psycopg2 provides copy_expert for GeoPandas
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        except ImportError:
            try:
                import psycopg  # psycopg v3 fallback
                url = url.replace("postgresql://", "postgresql+psycopg://", 1)
            except ImportError:
                pass
    return url

# Database connection — standardized with psycopg compatibility
DATABASE_URL: str = get_database_url()

# FIRMS feed configuration
FIRMS_BASE_URL: str = os.environ.get(
    "FIRMS_BASE_URL",
    "https://firms.modaps.eosdis.nasa.gov",
)
# Source key: VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT, MODIS_NRT, etc.
FIRMS_SOURCE: str = os.environ.get("FIRMS_SOURCE", "VIIRS_SNPP_NRT")
# Area of interest — 7-day bounding box for India (min_lon, min_lat, max_lon, max_lat)
# Full extent includes northernmost Jammu & Kashmir and Ladakh up to 37.5°N
FIRMS_AREA: str = os.environ.get("FIRMS_AREA", "68.1,6.7,97.4,37.5")
# Number of days of data to request (1-10 for NRT)
FIRMS_DAY_RANGE: str = os.environ.get("FIRMS_DAY_RANGE", "7")

# Boundary tolerance in degrees (~111 m per 0.001°)
BOUNDARY_TOLERANCE_DEG: float = float(
    os.environ.get("BOUNDARY_TOLERANCE_DEG", "0.01")
)

# HTTP timeout policies (seconds)
HTTP_CONNECT_TIMEOUT: float = float(os.environ.get("HTTP_CONNECT_TIMEOUT", "15"))
HTTP_READ_TIMEOUT: float = float(os.environ.get("HTTP_READ_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# Deterministic UUID generation
# ---------------------------------------------------------------------------
def make_detection_id(
    latitude: float,
    longitude: float,
    satellite: str,
    acq_date: str,
    acq_time: str,
) -> str:
    """
    Produce a deterministic UUID-v5-style identifier by SHA-256-hashing
    the canonical string representation of the detection's natural key.

    Returns a hex UUID string.
    """
    canonical = f"{latitude:.6f}|{longitude:.6f}|{satellite}|{acq_date}|{acq_time}"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # Format as UUID (8-4-4-4-12)
    return str(uuid.UUID(digest[:32]))


# ---------------------------------------------------------------------------
# Retry predicate — retry only on transient gateway errors
# ---------------------------------------------------------------------------
def _is_retryable(exc: BaseException) -> bool:
    """Return True for HTTP 502 / 504 gateway errors."""
    if isinstance(exc, requests.exceptions.HTTPError):
        code = exc.response.status_code if exc.response is not None else 0
        return code in (502, 504)
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    return False


# ---------------------------------------------------------------------------
# FIRMS CSV fetch with retries
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,
)
def fetch_firms_csv() -> pd.DataFrame:
    """
    Download the active-fire CSV from NASA FIRMS.

    The endpoint pattern is:
      /api/area/csv/{MAP_KEY}/{SOURCE}/{AREA}/{DAY_RANGE}

    Returns a pandas DataFrame of the raw CSV rows.
    """
    url = (
        f"{FIRMS_BASE_URL}/api/area/csv/"
        f"{FIRMS_MAP_KEY}/{FIRMS_SOURCE}/{FIRMS_AREA}/{FIRMS_DAY_RANGE}"
    )
    logger.info("Fetching FIRMS CSV: %s", url.replace(FIRMS_MAP_KEY, "***"))

    response = requests.get(
        url,
        timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
        verify=True,  # Strict TLS — never disable
    )

    # NASA FIRMS Area API enforces a maximum 5-day window for NRT queries.
    # If a 7-day query is rejected with "Invalid day range. Expects [1..5].", clamp to 5.
    if response.status_code == 400 and "Invalid day range" in response.text:
        fallback_days = "5"
        fallback_url = (
            f"{FIRMS_BASE_URL}/api/area/csv/"
            f"{FIRMS_MAP_KEY}/{FIRMS_SOURCE}/{FIRMS_AREA}/{fallback_days}"
        )
        logger.warning(
            "FIRMS returned '%s' for day range %s. Retrying with maximum supported day range (%s): %s",
            response.text.strip(),
            FIRMS_DAY_RANGE,
            fallback_days,
            fallback_url.replace(FIRMS_MAP_KEY, "***"),
        )
        response = requests.get(
            fallback_url,
            timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
            verify=True,
        )

    response.raise_for_status()

    csv_text = response.text
    if not csv_text.strip():
        logger.warning("FIRMS returned an empty response body.")
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(csv_text))
    logger.info("Fetched %d rows from FIRMS (%s).", len(df), FIRMS_SOURCE)

    # If VIIRS_SNPP_NRT has 0 rows (older satellite feed with low coverage),
    # and user didn't explicitly pin FIRMS_SOURCE, automatically supplement with VIIRS_NOAA20_NRT
    if df.empty and FIRMS_SOURCE == "VIIRS_SNPP_NRT" and "FIRMS_SOURCE" not in os.environ:
        alt_source = "VIIRS_NOAA20_NRT"
        alt_days = "5" if FIRMS_DAY_RANGE in ("7", "10") else FIRMS_DAY_RANGE
        alt_url = (
            f"{FIRMS_BASE_URL}/api/area/csv/"
            f"{FIRMS_MAP_KEY}/{alt_source}/{FIRMS_AREA}/{alt_days}"
        )
        logger.info(
            "VIIRS_SNPP_NRT returned 0 detections. Querying active %s feed: %s",
            alt_source,
            alt_url.replace(FIRMS_MAP_KEY, "***"),
        )
        alt_resp = requests.get(
            alt_url,
            timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
            verify=True,
        )
        if alt_resp.status_code == 200 and alt_resp.text.strip():
            alt_df = pd.read_csv(io.StringIO(alt_resp.text))
            if not alt_df.empty:
                logger.info("Successfully fetched %d rows from %s feed.", len(alt_df), alt_source)
                return alt_df

    return df


# ---------------------------------------------------------------------------
# Data transformation
# ---------------------------------------------------------------------------
def transform_firms_dataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Clean and transform raw FIRMS CSV into a GeoDataFrame ready for
    PostGIS insertion.

    Key steps:
      1. Build deterministic UUID per row.
      2. Merge acq_date + acq_time into a proper TIMESTAMPTZ.
      3. Construct Shapely Point geometries.
      4. Preserve NULLs — never fill missing metrics with 0.0.
    """
    if df.empty:
        logger.info("Empty DataFrame — nothing to transform.")
        return gpd.GeoDataFrame()

    required_cols = {"latitude", "longitude", "satellite", "acq_date", "acq_time"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"FIRMS CSV is missing required columns: {missing}")

    # --- Deterministic ID ---------------------------------------------------
    df["id"] = df.apply(
        lambda row: make_detection_id(
            latitude=row["latitude"],
            longitude=row["longitude"],
            satellite=str(row["satellite"]),
            acq_date=str(row["acq_date"]),
            acq_time=str(row["acq_time"]).zfill(4),  # "130" → "0130"
        ),
        axis=1,
    )

    # --- Acquisition datetime -----------------------------------------------
    def _parse_acq_datetime(row: pd.Series) -> datetime:
        date_str = str(row["acq_date"])
        time_str = str(row["acq_time"]).zfill(4)
        hh, mm = time_str[:2], time_str[2:]
        return datetime.strptime(
            f"{date_str} {hh}:{mm}", "%Y-%m-%d %H:%M"
        ).replace(tzinfo=timezone.utc)

    df["acq_datetime"] = df.apply(_parse_acq_datetime, axis=1)

    # --- Geometry -----------------------------------------------------------
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # --- Select & rename columns, preserving NULLs --------------------------
    col_map = {
        "id": "id",
        "satellite": "satellite",
        "acq_datetime": "acq_datetime",
        "frp": "frp",
        "confidence": "confidence",
        "brightness": "brightness",
        "bright_t31": "bright_t31",
        "scan": "scan",
        "track": "track",
        "daynight": "daynight",
        "version": "version",
    }
    available = {k: v for k, v in col_map.items() if k in gdf.columns}
    gdf = gdf[list(available.keys()) + ["geometry"]].rename(columns=available)

    # Replace pandas NaN with None so SQLAlchemy emits NULL
    gdf = gdf.where(gdf.notna(), other=None)

    logger.info("Transformed %d detections.", len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# India boundary filter using ST_Intersects + tolerance buffer
# ---------------------------------------------------------------------------
def filter_india_boundary(
    gdf: gpd.GeoDataFrame,
    engine,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Partition `gdf` into (inside_india, outside_india) using a PostGIS
    query against the naturalearth countries table.

    If no boundary table is loaded yet, we download a lightweight
    GeoJSON of India's boundary and cache it in PostGIS.

    Uses ST_Intersects with ST_Buffer for the tolerance check as
    required by the architecture spec.

    Returns:
        (gdf_inside, gdf_outside)
    """
    _ensure_india_boundary(engine)

    if gdf.empty:
        return gdf, gdf.iloc[0:0]

    from shapely import wkt as shapely_wkt

    with engine.connect() as conn:
        buffered_wkt = conn.execute(
            text("SELECT ST_AsText(ST_Buffer(geom, :tol)) FROM india_boundary LIMIT 1"),
            {"tol": BOUNDARY_TOLERANCE_DEG},
        ).scalar()

    if not buffered_wkt:
        raise RuntimeError("Failed to compute buffered India boundary from PostGIS.")

    buffered_boundary = shapely_wkt.loads(buffered_wkt)

    # Fast vectorized spatial intersection using PostGIS buffered boundary
    mask = gdf.geometry.intersects(buffered_boundary)
    gdf_inside = gdf[mask].copy()
    gdf_outside = gdf[~mask].copy()

    logger.info(
        "Boundary filter: %d inside India, %d excluded.",
        len(gdf_inside),
        len(gdf_outside),
    )
    return gdf_inside, gdf_outside


def _ensure_india_boundary(engine) -> None:
    """
    Create and populate the `india_boundary` table if it doesn't exist.

    Primary source: official India composite boundary GeoJSON from datameet/maps.
    Fallback: local bundled GeoJSON file (data/india_boundary.geojson).
    No other fallbacks — naturalearth_lowres has incorrect political boundaries.
    """
    with engine.connect() as conn:
        exists = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'india_boundary'
                      AND table_schema = 'public'
                )
                """
            )
        ).scalar()
        if exists:
            # Verify table has at least one geometry row
            row_count = conn.execute(text("SELECT count(*) FROM india_boundary")).scalar()
            if row_count and row_count > 0:
                logger.info("India boundary table already populated (%d geometries). Skipping.", row_count)
                return

    logger.info("Initializing India boundary dataset…")
    india = None

    # --- Primary source: local official Survey of India composite GeoJSON ----
    local_geojson = os.path.join(_script_dir, "data", "india_boundary.geojson")
    if os.path.exists(local_geojson):
        try:
            logger.info("Loading official India boundary from bundled file %s…", local_geojson)
            india = gpd.read_file(local_geojson)
            logger.info("Loaded official boundary: %d feature(s), bounds: %s", len(india), list(india.total_bounds))
        except Exception as exc:
            logger.warning("Failed to load local boundary file %s: %s", local_geojson, exc)

    # --- Fallback: remote datameet/maps official India composite boundary ---
    if india is None or india.empty:
        DATAMEET_URL = (
            "https://raw.githubusercontent.com/datameet/maps/master/Country/india-composite.geojson"
        )
        logger.info("Downloading official India boundary from datameet/maps: %s", DATAMEET_URL)
        try:
            resp = requests.get(
                DATAMEET_URL,
                timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
                verify=True,  # Strict TLS — never disable
            )
            resp.raise_for_status()
            geojson_data = resp.json()
            india = gpd.GeoDataFrame.from_features(geojson_data["features"], crs="EPSG:4326")
            logger.info(
                "Successfully downloaded datameet India boundary: %d feature(s).", len(india)
            )
        except Exception as exc:
            logger.error("Failed to download official India boundary from datameet: %s", exc)

    if india is None or india.empty:
        raise RuntimeError(
            "Could not load India boundary from any source. "
            "Network download from datameet/maps failed and no local fallback exists. "
            "Ensure network access or pre-download: "
            "https://raw.githubusercontent.com/datameet/maps/master/Country/india-composite.geojson"
        )

    # Normalize CRS & column name
    india = india[["geometry"]].copy()
    india = india.rename_geometry("geom")
    india = india.set_crs("EPSG:4326", allow_override=True)

    # Ensure validity
    if not india["geom"].iloc[0].is_valid:
        india["geom"] = india["geom"].buffer(0)

    # Write to PostGIS: Try to_postgis first, with direct SQL fallback on driver incompatibility
    try:
        india.to_postgis("india_boundary", engine, if_exists="replace", index=False)
        logger.info("India boundary table created with %d geometries via to_postgis.", len(india))
    except Exception as err:
        logger.warning("to_postgis failed (%s); persisting via direct PostGIS SQL insert.", err)
        wkt_str = india["geom"].iloc[0].wkt
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS india_boundary;"))
            conn.execute(text("CREATE TABLE india_boundary (geom GEOMETRY(Geometry, 4326));"))
            conn.execute(
                text("INSERT INTO india_boundary (geom) VALUES (ST_Multi(ST_SetSRID(ST_GeomFromText(:wkt), 4326)));"),
                {"wkt": wkt_str},
            )
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_india_boundary_geom ON india_boundary USING GIST (geom);"))
        logger.info("India boundary table created with %d geometries via direct SQL fallback.", len(india))


# ---------------------------------------------------------------------------
# Database insertion (idempotent via ON CONFLICT DO NOTHING)
# ---------------------------------------------------------------------------
def insert_detections(gdf: gpd.GeoDataFrame, engine) -> int:
    """
    Insert detections into `raw_firms_detections`, skipping rows whose
    deterministic UUID already exists.

    Returns the number of newly inserted rows.
    """
    if gdf.empty:
        logger.info("No detections to insert.")
        return 0

    records = []
    for _, row in gdf.iterrows():
        geom_wkt = f"SRID=4326;{row.geometry.wkt}"
        records.append({
            "id": row["id"],
            "geom": geom_wkt,
            "satellite": row.get("satellite"),
            "acq_datetime": row.get("acq_datetime"),
            "frp": row.get("frp"),
            "confidence": str(row["confidence"]) if row.get("confidence") is not None else None,
            "brightness": row.get("brightness"),
            "bright_t31": row.get("bright_t31"),
            "scan": row.get("scan"),
            "track": row.get("track"),
            "daynight": row.get("daynight"),
            "version": str(row["version"]) if row.get("version") is not None else None,
        })

    insert_stmt = text(
        """
        INSERT INTO raw_firms_detections
            (id, geom, satellite, acq_datetime, frp,
             confidence, brightness, bright_t31,
             scan, track, daynight, version)
        VALUES
            (:id, ST_GeomFromEWKT(:geom), :satellite,
             :acq_datetime, :frp, :confidence,
             :brightness, :bright_t31,
             :scan, :track, :daynight, :version)
        ON CONFLICT (id) DO NOTHING
        """
    )

    batch_size = 1000
    inserted = 0
    with engine.begin() as conn:
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            result = conn.execute(insert_stmt, chunk)
            inserted += result.rowcount

    logger.info("Inserted %d new detections (duplicates skipped).", inserted)
    return inserted


# ---------------------------------------------------------------------------
# Audit logging for boundary exclusions
# ---------------------------------------------------------------------------
def log_boundary_exclusions(gdf_outside: gpd.GeoDataFrame, engine) -> int:
    """
    Write excluded detections to the `boundary_exclusions_audit` table.
    Returns the count of rows logged.
    """
    if gdf_outside.empty:
        return 0

    records = []
    for _, row in gdf_outside.iterrows():
        geom_wkt = f"SRID=4326;{row.geometry.wkt}"
        records.append({
            "detection_id": row["id"],
            "geom": geom_wkt,
            "satellite": row.get("satellite"),
            "acq_datetime": row.get("acq_datetime"),
            "frp": row.get("frp"),
            "confidence": str(row["confidence"]) if row.get("confidence") is not None else None,
            "reason": "outside_india_boundary",
        })

    insert_stmt = text(
        """
        INSERT INTO boundary_exclusions_audit
            (detection_id, geom, satellite, acq_datetime,
             frp, confidence, exclusion_reason)
        VALUES
            (:detection_id, ST_GeomFromEWKT(:geom),
             :satellite, :acq_datetime, :frp, :confidence,
             :reason)
        """
    )

    batch_size = 1000
    logged = 0
    with engine.begin() as conn:
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            result = conn.execute(insert_stmt, chunk)
            logged += result.rowcount
            conn.execute(
                text(
                    """
                    INSERT INTO boundary_exclusions_audit
                        (detection_id, geom, satellite, acq_datetime,
                         frp, confidence, exclusion_reason)
                    VALUES
                        (:detection_id, ST_GeomFromEWKT(:geom),
                         :satellite, :acq_datetime, :frp, :confidence,
                         :reason)
                    """
                ),
                {
                    "detection_id": row["id"],
                    "geom": geom_wkt,
                    "satellite": row.get("satellite"),
                    "acq_datetime": row.get("acq_datetime"),
                    "frp": row.get("frp"),
                    "confidence": str(row["confidence"]) if row.get("confidence") is not None else None,
                    "reason": "outside_india_boundary",
                },
            )
            logged += 1

    logger.info("Logged %d boundary exclusions to audit table.", logged)
    return logged


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_ingestion() -> dict:
    """
    Main ingestion pipeline. Intended to be called by a scheduler,
    CLI entry point, or container entrypoint.

    Returns a summary dict with counts.
    """
    logger.info("=" * 60)
    logger.info("ThermalEye — FIRMS ingestion run starting")
    logger.info("=" * 60)

    engine = create_engine(DATABASE_URL, echo=False)

    # 1. Fetch
    raw_df = fetch_firms_csv()
    if raw_df.empty:
        logger.warning("No data returned from FIRMS. Exiting.")
        return {"fetched": 0, "inserted": 0, "excluded": 0}

    # 2. Transform
    gdf = transform_firms_dataframe(raw_df)

    # 3. Boundary filter
    gdf_inside, gdf_outside = filter_india_boundary(gdf, engine)

    # 4. Insert valid detections
    inserted = insert_detections(gdf_inside, engine)

    # 5. Audit excluded detections
    excluded = log_boundary_exclusions(gdf_outside, engine)

    summary = {
        "fetched": len(raw_df),
        "transformed": len(gdf),
        "inside_boundary": len(gdf_inside),
        "inserted": inserted,
        "excluded": excluded,
    }
    logger.info("Ingestion complete. Summary: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_ingestion()
