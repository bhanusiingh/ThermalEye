"""
ThermalEye — Phase 3+4: FastAPI Backend
=========================================

Asynchronous REST API serving GeoJSON payloads from PostGIS for
the Deck.gl intelligence dashboard.

Architectural invariants:
  • All responses are valid GeoJSON FeatureCollections.
  • Raw database errors are never exposed to the client.
  • priority_reasons JSONB and OSM disclaimer are always included.
  • ML disclaimer is always included when land-cover data exists.
  • Sentinel-2 imagery served via /static/imagery/ mount.
  • NULL fields are surfaced as JSON null, never as 0 or "".
"""

from __future__ import annotations

import gc
import io
import json
import logging
import os
import re
import sys
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

import asyncpg
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ExifTags, UnidentifiedImageError
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.prepared import prep

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("thermaleye.api")

# ---------------------------------------------------------------------------
# Environment & Database Configuration
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_script_dir, ".env"))


def resolve_database_url() -> str:
    """Resolve database URL supporting direct cloud DSNs (Neon Postgres) with SSL parameters,

    falling back to granular DB_* environment variables for local Docker Compose.
    """
    raw_url = os.environ.get("DATABASE_URL", "").strip()
    if raw_url:
        # Preserve query parameters (e.g. sslmode=require, channel_binding=require for Neon Postgres)
        if raw_url.startswith("postgres://"):
            raw_url = raw_url.replace("postgres://", "postgresql://", 1)
        return raw_url

    # Fallback to local Docker Compose environment variables
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("DB_NAME", "thermaleye")
    db_user = os.environ.get("DB_USER", "thermaleye")
    db_pass = os.environ.get("DB_PASSWORD", "thermaleye_secret")
    return f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"


DATABASE_URL: str = resolve_database_url()

API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
API_PORT: int = int(os.environ.get("API_PORT", os.environ.get("PORT", "8000")))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OSM_DISCLAIMER = "OSM coverage limitations apply"
ML_DISCLAIMER = (
    "Model provides land-cover context only, not facility identification "
    "or fire confirmation."
)
DEFAULT_LOOKBACK_DAYS = 7

# ---------------------------------------------------------------------------
# Database connection pool
# ---------------------------------------------------------------------------
pool: Optional[asyncpg.Pool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the asyncpg connection pool lifecycle and schema migrations."""
    global pool
    db_display = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
    logger.info("Connecting to PostGIS at %s", db_display)
    try:
        pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=int(os.environ.get("DB_POOL_MIN", "1")),
            max_size=int(os.environ.get("DB_POOL_MAX", "5" if os.environ.get("RENDER") else "10")),
            command_timeout=30,
        )
        logger.info("Database pool created.")

        # Ensure database schema alignment for ML columns and local OSM index
        async with pool.acquire() as conn:
            await conn.execute("ALTER TABLE candidate_events ADD COLUMN IF NOT EXISTS gradcam_path TEXT;")
            await conn.execute("ALTER TABLE candidate_events ADD COLUMN IF NOT EXISTS land_cover_confidence DOUBLE PRECISION;")
            await conn.execute("ALTER TABLE candidate_events ADD COLUMN IF NOT EXISTS ml_model_arch TEXT;")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS osm_industrial_features (
                    osm_id      TEXT PRIMARY KEY,
                    name        TEXT,
                    category    TEXT NOT NULL,
                    osm_type    TEXT NOT NULL,
                    geom        GEOMETRY(Point, 4326) NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_osm_industrial_geom
                    ON osm_industrial_features USING GIST (geom);
                CREATE INDEX IF NOT EXISTS idx_osm_industrial_geog
                    ON osm_industrial_features USING GIST ((geom::geography));
            """)
        logger.info("Database schema alignment complete (ML columns, osm_industrial_features).")
    except Exception:
        logger.exception("Failed to initialize database pool or run startup migrations.")
        sys.exit(1)
    yield
    if pool:
        await pool.close()
        logger.info("Database pool closed.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ThermalEye API",
    version="0.3.0",
    description="Geospatial intelligence API for thermal anomaly events.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static Imagery Mounting & Artifacts
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.join(_script_dir, "static")
IMAGERY_DIR = os.path.join(STATIC_DIR, "imagery")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
ARTIFACTS_DIR = os.path.join(_script_dir, "artifacts")
os.makedirs(IMAGERY_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


# ---------------------------------------------------------------------------
# Error handling — never expose raw DB errors
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _row_to_feature(row: asyncpg.Record, geometry_key: str = "geojson") -> dict:
    """Convert an asyncpg Record into a GeoJSON Feature dict."""
    import json

    props = {}
    geom = None

    for key in row.keys():
        value = row[key]
        if key == geometry_key:
            geom = json.loads(value) if value else None
            continue
        # Convert types that aren't JSON-native
        if isinstance(value, (datetime,)):
            value = value.isoformat()
        elif isinstance(value, (UUID,)):
            value = str(value)
        elif isinstance(value, (bytes, bytearray)):
            continue  # skip binary geometry columns
        props[key] = value

    return {
        "type": "Feature",
        "geometry": geom,
        "properties": props,
    }


def _feature_collection(features: list[dict]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _enrich_event_properties(props: dict) -> dict:
    """Guarantee required AI explainability fields, disclaimers, and imagery URLs."""
    if props.get("priority_reasons") is None:
        props["priority_reasons"] = {
            "note": "Priority breakdown not yet computed for this event."
        }
    props["osm_disclaimer"] = OSM_DISCLAIMER

    # Strict ML disclaimer — always present
    props["ml_disclaimer"] = props.get("ml_disclaimer") or ML_DISCLAIMER

    # Derived public imagery URLs
    rgb_path = props.get("imagery_rgb_path")
    swir_path = props.get("imagery_swir_path")
    gradcam_path = props.get("gradcam_path")

    props["imagery_rgb_url"] = f"/{rgb_path.lstrip('/')}" if rgb_path else None
    props["imagery_swir_url"] = f"/{swir_path.lstrip('/')}" if swir_path else None
    if gradcam_path:
        clean = gradcam_path.lstrip("/")
        props["gradcam_url"] = f"/{clean}" if clean.startswith("static/") else f"/static/imagery/{clean}"
    else:
        props["gradcam_url"] = None

    if "land_cover_confidence" not in props:
        props["land_cover_confidence"] = None
    if "ml_model_arch" not in props:
        props["ml_model_arch"] = None

    return props


# ---------------------------------------------------------------------------
# GET /api/v1/events/active & /api/events
# ---------------------------------------------------------------------------
@app.get("/api/v1/events/active")
@app.get("/api/events")
async def get_active_events(
    days: int = Query(
        default=DEFAULT_LOOKBACK_DAYS,
        ge=1,
        le=90,
        description="Lookback window in days.",
    ),
    limit: Optional[int] = Query(
        default=None,
        ge=1,
        description="Maximum candidate events to return.",
    ),
):
    """
    Return a GeoJSON FeatureCollection of candidate_events whose
    `last_seen` falls within the lookback window.

    Each Feature contains centroid geometry, optional footprint,
    priority_score, priority_reasons, OSM proximity, Grad-CAM attribution URL,
    ML confidence, model architecture, and the required OSM/ML disclaimer strings.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    query = """
        SELECT
            ce.id,
            ST_AsGeoJSON(ce.centroid)::text                  AS geojson,
            ST_AsGeoJSON(ce.footprint)::text                 AS footprint_geojson,
            ce.first_seen,
            ce.last_seen,
            ce.max_frp,
            ce.priority_score,
            ce.priority_reasons::text                        AS priority_reasons_raw,
            ce.nearest_osm_type,
            ce.nearest_osm_name,
            ce.nearest_osm_distance_km,
            ce.land_cover_class,
            ce.land_cover_confidence,
            ce.ml_model_arch,
            ce.ml_disclaimer,
            ce.imagery_rgb_path,
            ce.imagery_swir_path,
            ce.gradcam_path,
            ce.sentinel2_scene_id,
            ce.imagery_fetched_at,
            ce.detection_count,
            ce.created_at,
            ce.updated_at
        FROM candidate_events ce
        WHERE ce.last_seen >= $1
        ORDER BY ce.priority_score DESC NULLS LAST, ce.last_seen DESC
    """
    if limit is not None:
        query_sql = query + " LIMIT $2"
        params = (cutoff, limit)
    else:
        query_sql = query
        params = (cutoff,)

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query_sql, *params)
    except Exception:
        logger.exception("Database query failed for active events.")
        raise HTTPException(
            status_code=503,
            detail="Database temporarily unavailable.",
        )

    import json

    features = []
    for row in rows:
        feature = _row_to_feature(row, geometry_key="geojson")
        props = feature["properties"]

        # Parse priority_reasons from raw JSON text
        raw_pr = props.pop("priority_reasons_raw", None)
        props["priority_reasons"] = json.loads(raw_pr) if raw_pr else None

        # Include footprint as a separate geometry property (not the main geom)
        fp_raw = props.pop("footprint_geojson", None)
        props["footprint"] = json.loads(fp_raw) if fp_raw else None

        _enrich_event_properties(props)
        features.append(feature)

    return _feature_collection(features)


# ---------------------------------------------------------------------------
# Indian State Boundary Index (for Spatial Joins)
# ---------------------------------------------------------------------------
_STATE_GEOMETRIES: Optional[list[tuple[str, tuple[float, float, float, float], Any]]] = None


def _iso_utc(dt: Optional[datetime]) -> Optional[str]:
    """Serialize a datetime to ISO-8601 UTC string ending in 'Z'."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _frp_intensity(peak_frp: Optional[float]) -> str:
    """Categorize peak Fire Radiative Power (MW) into High / Moderate / Low."""
    if peak_frp is None:
        return "Low"
    if peak_frp >= 50.0:
        return "High"
    if peak_frp >= 20.0:
        return "Moderate"
    return "Low"


def _load_raw_state_polygons() -> list[dict]:
    """
    Load Indian state polygon definitions from data/india_states.json,
    falling back to extracting the STATES array from the landing HTML if needed.
    """
    json_path = os.path.join(_script_dir, "data", "india_states.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    for rel in (
        os.path.join("frontend", "index.html"),
        os.path.join("frontend", "ThermalEye landing page.html"),
    ):
        html_path = os.path.join(_script_dir, rel)
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                match = re.search(r"const STATES=(\[.*?\]);", f.read())
                if match:
                    return json.loads(match.group(1))
    return []


def _get_state_geometries() -> list[tuple[str, tuple[float, float, float, float], Any]]:
    """Build and cache prepared Shapely MultiPolygons + bounding boxes per state."""
    global _STATE_GEOMETRIES
    if _STATE_GEOMETRIES is not None:
        return _STATE_GEOMETRIES

    raw_states = _load_raw_state_polygons()
    geoms: list[tuple[str, tuple[float, float, float, float], Any]] = []
    for entry in raw_states:
        name = entry.get("n")
        polys_raw = entry.get("p") or []
        polys = []
        for poly in polys_raw:
            if not poly or not poly[0] or len(poly[0]) < 3:
                continue
            holes = [r for r in poly[1:] if len(r) >= 3] if len(poly) > 1 else None
            p = Polygon(poly[0], holes)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                polys.append(p)
        if not name or not polys:
            continue
        mp = MultiPolygon(polys) if len(polys) > 1 else polys[0]
        if not mp.is_valid:
            mp = mp.buffer(0)
        geoms.append((name, mp.bounds, prep(mp)))

    _STATE_GEOMETRIES = geoms
    return _STATE_GEOMETRIES


def locate_indian_state(lon: Optional[float], lat: Optional[float]) -> Optional[str]:
    """Return the Indian state name containing (lon, lat), or None if unmatched."""
    if lon is None or lat is None:
        return None
    states = _get_state_geometries()
    if not states:
        return None
    pt = Point(lon, lat)
    for name, bbox, pgeom in states:
        if bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]:
            if pgeom.covers(pt):
                return name
    return None


# ---------------------------------------------------------------------------
# GET /api/events/recent
# ---------------------------------------------------------------------------
@app.get("/api/events/recent")
async def get_recent_events():
    """Return the 8 most recent/highest-priority candidate_events."""
    query = """
        SELECT
            id,
            ST_X(centroid) AS lon,
            ST_Y(centroid) AS lat,
            max_frp,
            priority_score,
            nearest_osm_name,
            imagery_rgb_path,
            imagery_swir_path,
            land_cover_class
        FROM candidate_events
        ORDER BY (imagery_rgb_path IS NOT NULL) DESC, priority_score DESC NULLS LAST, last_seen DESC
        LIMIT 8
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query)
    except Exception:
        logger.exception("Database query failed for /api/events/recent.")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable.")

    results = []
    for row in rows:
        state_name = locate_indian_state(row["lon"], row["lat"]) or "India"
        peak_frp = round(float(row["max_frp"]), 1) if row["max_frp"] is not None else None
        if row["imagery_rgb_path"] is not None and row["nearest_osm_name"] is not None:
            evidence = "available"
        elif any(
            v is not None
            for v in (
                row["imagery_rgb_path"],
                row["imagery_swir_path"],
                row["nearest_osm_name"],
                row["land_cover_class"],
                row["priority_score"],
            )
        ):
            evidence = "partial"
        else:
            evidence = "unavailable"
        results.append(
            {
                "id": str(row["id"]),
                "state": state_name,
                "lat": float(row["lat"]) if row["lat"] is not None else None,
                "lon": float(row["lon"]) if row["lon"] is not None else None,
                "peak_frp": peak_frp,
                "evidence": evidence,
                "imagery_rgb_path": row["imagery_rgb_path"],
                "imagery_swir_path": row["imagery_swir_path"],
            }
        )

    return results


# ---------------------------------------------------------------------------
# GET /api/v1/events/{event_id} & /api/events/{event_id}
# ---------------------------------------------------------------------------
@app.get("/api/v1/events/{event_id}")
@app.get("/api/events/{event_id}")
async def get_event_dossier(event_id: UUID):
    """
    Return a detailed dossier for a single candidate event, including:
      - Full event metadata and priority breakdown
      - All linked raw FIRMS detections as a nested FeatureCollection
      - OSM proximity details
      - Sentinel-2 imagery URLs, Grad-CAM attribution overlays, and ML classification
      - Required disclaimers (OSM limitations + ML constraints)
    """
    # ---- Fetch the event itself ----
    event_query = """
        SELECT
            ce.id,
            ST_AsGeoJSON(ce.centroid)::text                  AS geojson,
            ST_AsGeoJSON(ce.footprint)::text                 AS footprint_geojson,
            ce.first_seen,
            ce.last_seen,
            ce.max_frp,
            ce.priority_score,
            ce.priority_reasons::text                        AS priority_reasons_raw,
            ce.nearest_osm_type,
            ce.nearest_osm_name,
            ce.nearest_osm_distance_km,
            ce.land_cover_class,
            ce.land_cover_confidence,
            ce.ml_model_arch,
            ce.ml_disclaimer,
            ce.imagery_rgb_path,
            ce.imagery_swir_path,
            ce.gradcam_path,
            ce.sentinel2_scene_id,
            ce.imagery_fetched_at,
            ce.detection_count,
            ce.created_at,
            ce.updated_at
        FROM candidate_events ce
        WHERE ce.id = $1
    """

    # ---- Fetch linked raw detections ----
    detections_query = """
        SELECT
            rd.id,
            ST_AsGeoJSON(rd.geom)::text AS geojson,
            rd.satellite,
            rd.acq_datetime,
            rd.frp,
            rd.confidence,
            rd.brightness,
            rd.bright_t31,
            rd.scan,
            rd.track,
            rd.daynight,
            rd.version,
            rd.ingested_at
        FROM raw_firms_detections rd
        WHERE rd.candidate_event_id = $1
        ORDER BY rd.acq_datetime DESC
    """

    import json

    try:
        async with pool.acquire() as conn:
            event_row = await conn.fetchrow(event_query, event_id)
            if event_row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Event {event_id} not found.",
                )
            detection_rows = await conn.fetch(detections_query, event_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Database query failed for event %s.", event_id)
        raise HTTPException(
            status_code=503,
            detail="Database temporarily unavailable.",
        )

    # Build event feature
    event_feature = _row_to_feature(event_row, geometry_key="geojson")
    props = event_feature["properties"]

    raw_pr = props.pop("priority_reasons_raw", None)
    props["priority_reasons"] = json.loads(raw_pr) if raw_pr else None

    fp_raw = props.pop("footprint_geojson", None)
    props["footprint"] = json.loads(fp_raw) if fp_raw else None

    _enrich_event_properties(props)

    # Build detections sub-collection
    det_features = [_row_to_feature(r, geometry_key="geojson") for r in detection_rows]

    # Compose dossier response
    dossier = {
        "type": "Feature",
        "geometry": event_feature["geometry"],
        "properties": {
            **props,
            "raw_detections": _feature_collection(det_features),
        },
    }

    return dossier


# ---------------------------------------------------------------------------
# Official Survey of India Boundary GeoJSON
# ---------------------------------------------------------------------------
@app.get("/api/v1/india-boundary")
async def get_india_boundary():
    """
    Return the official Survey of India sovereign boundary GeoJSON
    with complete territorial integrity including Jammu & Kashmir and Ladakh.
    """
    try:
        simplified_path = os.path.join(_script_dir, "data", "india_boundary_simplified.geojson")
        full_path = os.path.join(_script_dir, "data", "india_boundary.geojson")
        chosen_path = simplified_path if os.path.exists(simplified_path) else full_path
        if os.path.exists(chosen_path):
            import json
            with open(chosen_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return JSONResponse(content=data)

        async with pool.acquire() as conn:
            geojson_str = await conn.fetchval(
                "SELECT ST_AsGeoJSON(geom) FROM india_boundary LIMIT 1"
            )
            if geojson_str:
                import json
                return JSONResponse(
                    content={
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": json.loads(geojson_str),
                                "properties": {"name": "India (Official Sovereign Boundary)"},
                            }
                        ],
                    }
                )
        raise HTTPException(status_code=404, detail="India boundary data not found.")
    except Exception as exc:
        logger.exception("Error serving India boundary: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to load India boundary.")


# ---------------------------------------------------------------------------
# GET /api/summary
# ---------------------------------------------------------------------------
@app.get("/api/summary")
async def get_summary():
    """Return live summary telemetry computed strictly from PostGIS."""
    raw_query = """
        SELECT
            COUNT(*)::int AS raw_detections,
            MIN((acq_datetime AT TIME ZONE 'UTC')::date) AS min_date,
            MAX((acq_datetime AT TIME ZONE 'UTC')::date) AS max_date,
            MAX(ingested_at) AS latest_ingested_at
        FROM raw_firms_detections
    """
    events_query = """
        SELECT
            COUNT(*)::int AS candidate_events,
            MAX(max_frp) AS peak_frp,
            COUNT(*) FILTER (WHERE priority_score >= 50)::int AS high_priority,
            COUNT(*) FILTER (
                WHERE imagery_rgb_path IS NOT NULL
                   OR imagery_swir_path IS NOT NULL
                   OR nearest_osm_name IS NOT NULL
            )::int AS evidence_available,
            GREATEST(MAX(updated_at), MAX(created_at)) AS latest_event_ts
        FROM candidate_events
    """
    centroids_query = """
        SELECT ST_X(centroid) AS lon, ST_Y(centroid) AS lat
        FROM candidate_events
        WHERE centroid IS NOT NULL
    """
    try:
        async with pool.acquire() as conn:
            raw_row = await conn.fetchrow(raw_query)
            ev_row = await conn.fetchrow(events_query)
            centroid_rows = await conn.fetch(centroids_query)
    except Exception:
        logger.exception("Database query failed for /api/summary.")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable.")

    min_date = raw_row["min_date"] if raw_row else None
    max_date = raw_row["max_date"] if raw_row else None
    if min_date and max_date:
        window = (
            min_date.isoformat()
            if min_date == max_date
            else f"{min_date.isoformat()} to {max_date.isoformat()}"
        )
    else:
        window = None

    latest_ts = (ev_row["latest_event_ts"] if ev_row else None) or (
        raw_row["latest_ingested_at"] if raw_row else None
    )

    matched_states = set()
    for row in centroid_rows:
        st = locate_indian_state(row["lon"], row["lat"])
        if st:
            matched_states.add(st)

    peak_frp_raw = ev_row["peak_frp"] if ev_row else None
    peak_frp = round(float(peak_frp_raw), 1) if peak_frp_raw is not None else None

    return {
        "window": window,
        "updated_at": _iso_utc(latest_ts),
        "candidate_events": int(ev_row["candidate_events"]) if ev_row else 0,
        "raw_detections": int(raw_row["raw_detections"]) if raw_row else 0,
        "peak_frp": peak_frp,
        "states_with_events": len(matched_states),
        "high_priority": int(ev_row["high_priority"]) if ev_row else 0,
        "evidence_available": int(ev_row["evidence_available"]) if ev_row else 0,
    }


# ---------------------------------------------------------------------------
# GET /api/analytics/states
# ---------------------------------------------------------------------------
@app.get("/api/analytics/states")
async def get_analytics_states():
    """Spatially join candidate_events centroids to Indian states and aggregate."""
    query = """
        SELECT
            id,
            ST_X(centroid) AS lon,
            ST_Y(centroid) AS lat,
            max_frp,
            detection_count,
            priority_score,
            imagery_rgb_path,
            imagery_swir_path,
            nearest_osm_name
        FROM candidate_events
        WHERE centroid IS NOT NULL
        ORDER BY priority_score DESC NULLS LAST, max_frp DESC NULLS LAST
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query)
    except Exception:
        logger.exception("Database query failed for /api/analytics/states.")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable.")

    agg: dict[str, dict[str, Any]] = {}
    for row in rows:
        state_name = locate_indian_state(row["lon"], row["lat"])
        if not state_name:
            continue
        bucket = agg.get(state_name)
        if bucket is None:
            bucket = {
                "state": state_name,
                "count": 0,
                "peak_frp": None,
                "clusters": 0,
                "evidence": 0,
                "_events": [],
            }
            agg[state_name] = bucket
        bucket["count"] += 1
        det_cnt = row["detection_count"]
        if det_cnt is not None:
            bucket["clusters"] += int(det_cnt)
        frp = row["max_frp"]
        if frp is not None:
            frp_val = float(frp)
            if bucket["peak_frp"] is None or frp_val > bucket["peak_frp"]:
                bucket["peak_frp"] = frp_val
        if (
            row["imagery_rgb_path"] is not None
            or row["imagery_swir_path"] is not None
            or row["nearest_osm_name"] is not None
        ):
            bucket["evidence"] += 1
        bucket["_events"].append(
            (
                row["priority_score"],
                row["max_frp"],
                str(row["id"])[:8] if row["id"] is not None else "",
            )
        )

    results = []
    for bucket in agg.values():
        peak = round(bucket["peak_frp"], 1) if bucket["peak_frp"] is not None else None
        sorted_events = sorted(
            bucket["_events"],
            key=lambda e: (
                e[0] is not None,
                float(e[0]) if e[0] is not None else 0.0,
                e[1] is not None,
                float(e[1]) if e[1] is not None else 0.0,
            ),
            reverse=True,
        )[:5]
        top_events = [
            {
                "id": ev_id,
                "peak_frp": round(float(ev_frp), 1) if ev_frp is not None else None,
            }
            for _prio, ev_frp, ev_id in sorted_events
        ]
        results.append(
            {
                "state": bucket["state"],
                "count": bucket["count"],
                "peak_frp": peak,
                "clusters": bucket["clusters"],
                "intensity": _frp_intensity(peak),
                "evidence": bucket["evidence"],
                "events": top_events,
            }
        )

    results.sort(key=lambda item: (-item["count"], -(item["peak_frp"] or 0.0), item["state"]))
    return results


# ---------------------------------------------------------------------------
# GET /api/analytics/timeline
# ---------------------------------------------------------------------------
@app.get("/api/analytics/timeline")
async def get_analytics_timeline():
    """Return daily time-series aggregates ordered chronologically by UTC date."""
    raw_query = """
        SELECT
            (acq_datetime AT TIME ZONE 'UTC')::date AS day,
            COUNT(*)::int AS count
        FROM raw_firms_detections
        GROUP BY day
        ORDER BY day ASC
    """
    events_query = """
        SELECT
            (last_seen AT TIME ZONE 'UTC')::date AS day,
            COUNT(*)::int AS event_count,
            MAX(max_frp) AS max_frp,
            COUNT(*) FILTER (WHERE detection_count > 1)::int AS recurring_count
        FROM candidate_events
        GROUP BY day
        ORDER BY day ASC
    """
    try:
        async with pool.acquire() as conn:
            raw_rows = await conn.fetch(raw_query)
            event_rows = await conn.fetch(events_query)
    except Exception:
        logger.exception("Database query failed for /api/analytics/timeline.")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable.")

    detections = [
        {"date": r["day"].isoformat(), "value": int(r["count"])}
        for r in raw_rows
        if r["day"] is not None
    ]
    candidate_events = [
        {"date": r["day"].isoformat(), "value": int(r["event_count"])}
        for r in event_rows
        if r["day"] is not None
    ]
    frp = [
        {"date": r["day"].isoformat(), "value": round(float(r["max_frp"]), 1)}
        for r in event_rows
        if r["day"] is not None and r["max_frp"] is not None
    ]
    recurring_hotspots = [
        {"date": r["day"].isoformat(), "value": int(r["recurring_count"])}
        for r in event_rows
        if r["day"] is not None
    ]

    return {
        "detections": detections,
        "candidate_events": candidate_events,
        "frp": frp,
        "recurring_hotspots": recurring_hotspots,
    }


# ---------------------------------------------------------------------------
# GET /api/providers/health
# ---------------------------------------------------------------------------
@app.get("/api/providers/health")
async def get_providers_health():
    """Compute live provider health strictly from actual PostGIS state."""
    raw_query = """
        SELECT
            COUNT(*)::int AS total_detections,
            MAX((acq_datetime AT TIME ZONE 'UTC')::date) AS latest_acq_date
        FROM raw_firms_detections
    """
    events_query = """
        SELECT
            COUNT(*)::int AS total_events,
            COUNT(*) FILTER (WHERE nearest_osm_name IS NOT NULL)::int AS osm_enriched,
            COUNT(*) FILTER (WHERE priority_reasons::text ILIKE '%Rate Limited%')::int AS osm_rate_limited,
            COUNT(*) FILTER (WHERE imagery_rgb_path IS NOT NULL)::int AS imagery_chips,
            COUNT(*) FILTER (WHERE imagery_fetched_at IS NOT NULL)::int AS imagery_inspected,
            COUNT(*) FILTER (WHERE land_cover_class IS NOT NULL)::int AS ml_classified,
            MAX(updated_at) AS latest_updated_at
        FROM candidate_events
    """
    try:
        async with pool.acquire() as conn:
            raw_row = await conn.fetchrow(raw_query)
            ev_row = await conn.fetchrow(events_query)
            osm_has_table = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'osm_industrial_features')"
            )
            osm_local_count = (
                await conn.fetchval("SELECT COUNT(*)::int FROM osm_industrial_features")
                if osm_has_table
                else 0
            ) or 0
    except Exception:
        logger.exception("Database query failed for /api/providers/health.")
        return {
            "NASA FIRMS": {"status": "unknown", "detail": "Unable to query database."},
            "OpenStreetMap": {"status": "unknown", "detail": "Unable to query database."},
            "Copernicus CDSE": {"status": "unknown", "detail": "Unable to query database."},
            "MobileNetV2": {"status": "unknown", "detail": "Unable to query database."},
            "PostGIS": {"status": "failing", "detail": "PostgreSQL / PostGIS connection unavailable."},
            "Background worker": {"status": "unknown", "detail": "Unable to query database."},
        }

    total_det = int(raw_row["total_detections"]) if raw_row else 0
    latest_acq = raw_row["latest_acq_date"] if raw_row else None

    total_ev = int(ev_row["total_events"]) if ev_row else 0
    osm_enriched = int(ev_row["osm_enriched"]) if ev_row else 0
    osm_rate_limited = int(ev_row["osm_rate_limited"]) if ev_row else 0
    imagery_chips = int(ev_row["imagery_chips"]) if ev_row else 0
    imagery_inspected = int(ev_row["imagery_inspected"]) if ev_row else 0
    ml_classified = int(ev_row["ml_classified"]) if ev_row else 0
    latest_updated_at = ev_row["latest_updated_at"] if ev_row else None

    # 1. NASA FIRMS
    firms_status = "ok" if total_det > 0 else "unknown"
    firms_detail = (
        f"Latest acquisition {latest_acq.isoformat()}; {total_det:,} detections ingested"
        if total_det > 0 and latest_acq is not None
        else "No FIRMS detections ingested yet."
    )

    # 2. OpenStreetMap
    if osm_local_count > 0:
        osm_status = "ok"
        osm_detail = (
            f"Local index: {osm_local_count:,} industrial features; {osm_enriched:,} of {total_ev:,} candidate events enriched."
        )
    elif osm_enriched > 0:
        osm_status = "ok"
        osm_detail = (
            f"{osm_enriched:,} enriched, {osm_rate_limited:,} rate-limited of {total_ev:,} events"
            if total_ev > 0
            else "No candidate events evaluated yet."
        )
    elif osm_rate_limited > 0:
        osm_status = "degraded"
        osm_detail = (
            f"{osm_enriched:,} enriched, {osm_rate_limited:,} rate-limited of {total_ev:,} events"
            if total_ev > 0
            else "No candidate events evaluated yet."
        )
    else:
        osm_status = "unknown"
        osm_detail = (
            f"{osm_enriched:,} enriched, {osm_rate_limited:,} rate-limited of {total_ev:,} events"
            if total_ev > 0
            else "No candidate events evaluated yet."
        )

    # 3. Copernicus CDSE
    if imagery_chips > 0:
        cdse_status = "ok"
    elif imagery_inspected > 0:
        cdse_status = "degraded"
    else:
        cdse_status = "unknown"
    cdse_detail = (
        f"{imagery_chips:,} chips acquired across {imagery_inspected:,} inspected events"
        if (imagery_chips > 0 or imagery_inspected > 0)
        else "No Sentinel-2 imagery inspections recorded yet."
    )

    # 4. MobileNetV2
    ml_status = "ok" if ml_classified > 0 else "unknown"
    ml_detail = (
        f"{ml_classified:,} events classified"
        if ml_classified > 0
        else "No events classified yet."
    )

    # 5. PostGIS
    postgis_status = "ok"
    postgis_detail = "PostgreSQL 15 / PostGIS 3.4 connected"

    # 6. Background worker
    if latest_updated_at is None:
        worker_status = "unknown"
        worker_detail = "No pipeline updates recorded yet."
    else:
        ts_utc = (
            latest_updated_at.replace(tzinfo=timezone.utc)
            if latest_updated_at.tzinfo is None
            else latest_updated_at.astimezone(timezone.utc)
        )
        age = datetime.now(timezone.utc) - ts_utc
        worker_status = "ok" if age <= timedelta(hours=2) else "degraded"
        worker_detail = f"Last update {_iso_utc(ts_utc)}"

    return {
        "NASA FIRMS": {"status": firms_status, "detail": firms_detail},
        "OpenStreetMap": {"status": osm_status, "detail": osm_detail},
        "Copernicus CDSE": {"status": cdse_status, "detail": cdse_detail},
        "MobileNetV2": {"status": ml_status, "detail": ml_detail},
        "PostGIS": {"status": postgis_status, "detail": postgis_detail},
        "Background worker": {"status": worker_status, "detail": worker_detail},
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/v1/health")
async def health():
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "database": "disconnected"},
        )


# ---------------------------------------------------------------------------
# GET /api/ml/metrics
# ---------------------------------------------------------------------------
@app.get("/api/ml/metrics")
async def get_ml_metrics():
    """
    Read and return machine learning benchmark evaluations, dataset statistics,
    and empirical pipeline validation reports from disk.
    Returns null for any artifact not yet generated or missing from disk.
    """
    def _read_artifact(filename: str) -> Optional[dict]:
        file_path = os.path.join(ARTIFACTS_DIR, filename)
        if not os.path.isfile(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("Failed to read ML artifact: %s", filename)
            return None

    return {
        "dataset_stats": _read_artifact("dataset_stats.json"),
        "model_comparison": _read_artifact("model_comparison.json"),
        "evaluation_report": _read_artifact("evaluation_report.json"),
        "experimental_validation": _read_artifact("experimental_validation.json"),
    }


# ---------------------------------------------------------------------------
# Multimodal Sample Analysis & Geolocation Helpers
# ---------------------------------------------------------------------------
def extract_exif_gps(img: Image.Image) -> tuple[Optional[float], Optional[float]]:
    """Extract WGS84 GPS latitude and longitude from PIL Image EXIF tags if present."""
    try:
        exif = img.getexif()
        if not exif:
            return None, None

        gps_info = None
        if hasattr(exif, "get_ifd"):
            try:
                gps_info = exif.get_ifd(0x8825)
            except Exception:
                pass
        if not gps_info:
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                if tag_name == "GPSInfo" or tag_id == 34853:
                    gps_info = value
                    break
        if not gps_info:
            return None, None

        gps_data = {}
        for k, v in gps_info.items():
            sub_tag = ExifTags.GPSTAGS.get(k, k)
            gps_data[sub_tag] = v

        def _dms_to_deg(dms, ref):
            if isinstance(dms, (int, float)):
                deg = float(dms)
            else:
                deg = float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0
            if ref in ("S", "W"):
                deg = -deg
            return deg

        lat, lon = None, None
        if "GPSLatitude" in gps_data and "GPSLatitudeRef" in gps_data:
            lat = _dms_to_deg(gps_data["GPSLatitude"], gps_data["GPSLatitudeRef"])
        if "GPSLongitude" in gps_data and "GPSLongitudeRef" in gps_data:
            lon = _dms_to_deg(gps_data["GPSLongitude"], gps_data["GPSLongitudeRef"])

        return lat, lon
    except Exception as e:
        logger.debug("Failed to extract EXIF GPS: %s", e)
        return None, None


_SAMPLE_MODEL_BUNDLE = None


def get_sample_analysis_model() -> dict:
    """Lazy-load the fine-tuned 6-channel champion model and channel normalization stats.

    Enforces single-thread execution and downloads the checkpoint automatically from
    os.getenv('MODEL_URL') if absent on disk, optimizing for Render 512 MB RAM limit.
    """
    global _SAMPLE_MODEL_BUNDLE
    if _SAMPLE_MODEL_BUNDLE is not None:
        return _SAMPLE_MODEL_BUNDLE

    # Enforce single-thread execution to prevent memory spikes on free-tier (512 MB RAM)
    torch.set_num_threads(1)

    from ml.models import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = os.path.join(_script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    weights_path = os.path.join(models_dir, "thermaleye_best.pt")

    # If weights file is not on disk, automatically download from MODEL_URL if provided
    if not os.path.isfile(weights_path):
        model_url = os.getenv("MODEL_URL", "").strip()
        if model_url:
            logger.info("Champion weights not found at %s. Downloading from MODEL_URL: %s", weights_path, model_url)
            try:
                urllib.request.urlretrieve(model_url, weights_path)
                logger.info(
                    "Successfully downloaded champion model weights (%.2f MB).",
                    os.path.getsize(weights_path) / (1024 * 1024),
                )
            except Exception as e:
                logger.error("Failed to download model weights from %s: %s", model_url, e)

    class_names = [
        "industrial_thermal",
        "agricultural_stubble_burn",
        "forest_wildfire",
        "urban_false_positive",
    ]
    arch = "efficientnet_b0"
    in_channels = 6

    # Load normalization stats from artifacts/dataset_stats.json
    stats_path = os.path.join(ARTIFACTS_DIR, "dataset_stats.json")
    mean = [0.28, 0.25, 0.17, 0.38, 0.55, 0.62]
    std = [0.19, 0.13, 0.11, 0.23, 0.27, 0.27]
    if os.path.isfile(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                cs = d.get("channel_stats", {})
                if "mean" in cs and len(cs["mean"]) >= 6:
                    mean = cs["mean"][:6]
                if "std" in cs and len(cs["std"]) >= 6:
                    std = cs["std"][:6]
        except Exception:
            pass

    if os.path.isfile(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            arch = checkpoint.get("arch", arch)
            in_channels = checkpoint.get("in_channels", in_channels)
            class_names = checkpoint.get("class_names", class_names)
            model = build_model(arch=arch, in_channels=in_channels, num_classes=len(class_names), pretrained=False)
            model.load_state_dict(checkpoint["state_dict"])
            del checkpoint
            gc.collect()
        except Exception as e:
            logger.warning("Failed loading checkpoint from %s: %s. Using initialized model.", weights_path, e)
            model = build_model(arch=arch, in_channels=in_channels, num_classes=len(class_names), pretrained=False)
    else:
        model = build_model(arch=arch, in_channels=in_channels, num_classes=len(class_names), pretrained=False)

    model.to(device)
    model.eval()

    _SAMPLE_MODEL_BUNDLE = {
        "model": model,
        "arch": arch,
        "in_channels": in_channels,
        "class_names": class_names,
        "mean": torch.tensor(mean[:in_channels], dtype=torch.float32, device=device).view(-1, 1, 1),
        "std": torch.tensor(std[:in_channels], dtype=torch.float32, device=device).view(-1, 1, 1),
        "device": device,
    }
    return _SAMPLE_MODEL_BUNDLE


# ---------------------------------------------------------------------------
# POST /api/analyze-sample
# ---------------------------------------------------------------------------
@app.post("/api/analyze-sample")
async def analyze_sample(
    rgb_file: UploadFile = File(...),
    swir_file: Optional[UploadFile] = File(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
):
    """
    On-demand sample analysis: upload satellite RGB (and optional SWIR) chip,
    extract or accept coordinates, resolve sovereign boundary & OSM context,
    run 6-channel classification, and synthesize Grad-CAM explainability overlay.
    """
    # 1. Read & validate RGB file
    if not rgb_file or not rgb_file.filename:
        raise HTTPException(status_code=400, detail="Missing required 'rgb_file'.")

    try:
        ext = os.path.splitext(rgb_file.filename)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported RGB image format '{ext}'. Allowed: PNG, JPEG, TIF/TIFF.",
            )
        rgb_bytes = await rgb_file.read()
        if not rgb_bytes:
            raise HTTPException(status_code=400, detail="Uploaded RGB file is empty.")
        rgb_img = Image.open(io.BytesIO(rgb_bytes))
        fmt = (rgb_img.format or "").upper()
        if fmt not in ("PNG", "JPEG", "TIFF"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported RGB image format '{fmt}'. Allowed: PNG, JPEG, TIF/TIFF.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted RGB image: {str(e)}")

    # 2. Read & validate optional SWIR file
    swir_img = None
    if swir_file and swir_file.filename:
        try:
            s_ext = os.path.splitext(swir_file.filename)[1].lower()
            if s_ext not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported SWIR image format '{s_ext}'. Allowed: PNG, JPEG, TIF/TIFF.",
                )
            swir_bytes = await swir_file.read()
            if swir_bytes:
                swir_img = Image.open(io.BytesIO(swir_bytes))
                s_fmt = (swir_img.format or "").upper()
                if s_fmt not in ("PNG", "JPEG", "TIFF"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported SWIR image format '{s_fmt}'. Allowed: PNG, JPEG, TIF/TIFF.",
                    )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid or corrupted SWIR image: {str(e)}")

    # 3. Coordinate validation & EXIF extraction
    if latitude is not None or longitude is not None:
        if latitude is None or longitude is None:
            raise HTTPException(status_code=400, detail="Both latitude and longitude must be provided together.")
        if not (-90.0 <= latitude <= 90.0) or not (-180.0 <= longitude <= 180.0):
            raise HTTPException(
                status_code=400,
                detail=f"Coordinates out of bounds: latitude must be [-90, 90], longitude [-180, 180]. Got ({latitude}, {longitude}).",
            )
    else:
        # Inspect EXIF metadata
        exif_lat, exif_lon = extract_exif_gps(rgb_img)
        if exif_lat is not None and exif_lon is not None:
            if -90.0 <= exif_lat <= 90.0 and -180.0 <= exif_lon <= 180.0:
                latitude, longitude = exif_lat, exif_lon

    # 4. Geospatial Context (when coordinates are present)
    if latitude is not None and longitude is not None:
        state_name = locate_indian_state(longitude, latitude)
        inside_india = False
        nearby_events_5km = 0
        nearest_infrastructure = None

        if pool:
            try:
                async with pool.acquire() as conn:
                    if hasattr(conn, "fetchval"):
                        inside_val = await conn.fetchval(
                            "SELECT EXISTS(SELECT 1 FROM india_boundary WHERE ST_Intersects(geom, ST_SetSRID(ST_MakePoint($1, $2), 4326)))",
                            longitude, latitude
                        )
                        if inside_val is not None:
                            inside_india = bool(inside_val)

                        nearby_val = await conn.fetchval(
                            "SELECT COUNT(*) FROM candidate_events WHERE ST_DWithin(centroid::geography, ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography, 5000)",
                            longitude, latitude
                        )
                        if nearby_val is not None:
                            nearby_events_5km = int(nearby_val)

                    if hasattr(conn, "fetchrow"):
                        osm_row = await conn.fetchrow(
                            """
                            SELECT nearest_osm_name, nearest_osm_type,
                                   ST_Distance(centroid::geography, ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography) / 1000.0 AS dist_km
                            FROM candidate_events
                            WHERE nearest_osm_name IS NOT NULL
                            ORDER BY ST_Distance(centroid::geography, ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography) ASC
                            LIMIT 1
                            """,
                            longitude, latitude
                        )
                        if osm_row:
                            dist = osm_row["dist_km"]
                            name = osm_row["nearest_osm_name"]
                            infra_type = osm_row.get("nearest_osm_type")
                            type_str = f" [{infra_type}]" if infra_type else ""
                            nearest_infrastructure = f"{name}{type_str} ({dist:.1f} km)"
            except Exception as e:
                logger.warning("Geospatial context query warning: %s", e)

        if not inside_india and state_name is not None:
            inside_india = True

        location = {
            "latitude": round(latitude, 6),
            "longitude": round(longitude, 6),
            "inside_india": inside_india,
            "state": state_name,
            "nearby_events_5km": nearby_events_5km,
            "nearest_infrastructure": nearest_infrastructure or "No mapped infrastructure nearby",
        }
    else:
        location = {
            "latitude": None,
            "longitude": None,
            "inside_india": False,
            "state": None,
            "nearby_events_5km": 0,
            "nearest_infrastructure": None,
            "status": "Coordinates not provided",
        }

    # 5. Save uploaded images
    sample_id = str(uuid.uuid4())
    rgb_filename = f"{sample_id}_rgb.png"
    rgb_path = os.path.join(UPLOADS_DIR, rgb_filename)
    rgb_img.convert("RGB").save(rgb_path, format="PNG")

    swir_filename = None
    if swir_img is not None:
        swir_filename = f"{sample_id}_swir.png"
        swir_path = os.path.join(UPLOADS_DIR, swir_filename)
        swir_img.convert("RGB").save(swir_path, format="PNG")

    # 6. Multispectral Tensor Construction & Model Inference
    bundle = get_sample_analysis_model()
    model = bundle["model"]
    device = bundle["device"]

    rgb_resized = rgb_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    rgb_arr = np.array(rgb_resized, dtype=np.float32) / 255.0
    rgb_t = torch.from_numpy(rgb_arr).permute(2, 0, 1).to(device)

    if swir_img is not None:
        swir_resized = swir_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        swir_arr = np.array(swir_resized, dtype=np.float32) / 255.0
        swir_t = torch.from_numpy(swir_arr).permute(2, 0, 1).to(device)
        input_t = torch.cat([rgb_t, swir_t], dim=0)
        modality = "RGB + SWIR (Paired)"
    else:
        input_t = torch.cat([rgb_t, rgb_t], dim=0)
        modality = "RGB-Only (SWIR Synthesized)"

    norm_t = (input_t - bundle["mean"]) / (bundle["std"] + 1e-7)
    input_tensor = norm_t.unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    class_names = bundle["class_names"]
    pred_idx = int(np.argmax(probs))
    predicted_class = class_names[pred_idx]
    confidence = round(float(probs[pred_idx]), 4)
    class_probs = {
        cls_name: round(float(probs[i]), 4)
        for i, cls_name in enumerate(class_names)
    }

    # 7. Generate Grad-CAM attribution overlay
    gradcam_filename = f"{sample_id}_gradcam.png"
    gradcam_path = os.path.join(UPLOADS_DIR, gradcam_filename)
    try:
        from ml.gradcam import generate_gradcam
        generate_gradcam(
            model=model,
            input_tensor=input_tensor,
            rgb_image_path=rgb_path,
            arch=bundle["arch"],
            target_class_idx=pred_idx,
            output_path=gradcam_path,
        )
        gradcam_url = f"/static/uploads/{gradcam_filename}"
    except Exception as e:
        logger.exception("Grad-CAM generation failed for sample %s: %s", sample_id, e)
        gradcam_url = None
    finally:
        gc.collect()

    return {
        "sample_id": sample_id,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_probabilities": class_probs,
        "model_arch": bundle["arch"],
        "modality": modality,
        "inference_latency_ms": latency_ms,
        "rgb_url": f"/static/uploads/{rgb_filename}",
        "swir_url": f"/static/uploads/{swir_filename}" if swir_filename else None,
        "gradcam_url": gradcam_url,
        "location": location,
        "disclaimer": ML_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Single-Container Cloud Frontend Serving (Render / Production)
# ---------------------------------------------------------------------------
FRONTEND_DIST_DIR = os.path.join(_script_dir, "frontend", "dist")
SERVE_FRONTEND: bool = (
    os.environ.get("SERVE_FRONTEND", "").strip().lower() in ("1", "true", "yes")
    or os.path.isfile(os.path.join(FRONTEND_DIST_DIR, "index.html"))
)

if SERVE_FRONTEND and os.path.isdir(FRONTEND_DIST_DIR):
    assets_dir = os.path.join(FRONTEND_DIST_DIR, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend_assets")

    index_html_path = os.path.join(FRONTEND_DIST_DIR, "index.html")
    dashboard_html_path = os.path.join(FRONTEND_DIST_DIR, "dashboard.html")

    @app.get("/", include_in_schema=False)
    async def serve_frontend_index():
        if os.path.isfile(index_html_path):
            return FileResponse(index_html_path, media_type="text/html")
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")

    @app.get("/dashboard", include_in_schema=False)
    async def serve_frontend_dashboard():
        if os.path.isfile(dashboard_html_path):
            return FileResponse(dashboard_html_path, media_type="text/html")
        raise HTTPException(status_code=404, detail="Frontend dashboard.html not found.")

    @app.get("/{filename:path}", include_in_schema=False)
    async def serve_frontend_static_root(filename: str):
        # Exclude reserved prefixes so API, static, and assets routes are never intercepted
        if any(filename.startswith(p) for p in ("api/", "static/", "assets/")):
            raise HTTPException(status_code=404, detail="Not found.")
        target_path = os.path.join(FRONTEND_DIST_DIR, filename)
        if os.path.isfile(target_path):
            return FileResponse(target_path)
        raise HTTPException(status_code=404, detail="Not found.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
