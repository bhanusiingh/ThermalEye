-- ==============================================
-- ThermalEye — PostGIS Schema Initialization
-- Phase 1–4: Raw detections, candidate events,
--             linkage, boundary exclusion audit,
--             enrichment columns, and imagery/ML.
-- ==============================================

-- Enable PostGIS extension (idempotent)
CREATE EXTENSION IF NOT EXISTS postgis;

-- ==============================================
-- 1. Raw FIRMS Detections
--    Stores unclustered thermal anomaly points
--    ingested directly from the NASA FIRMS feed.
--
--    `id` is a deterministic UUID computed from
--    a hash of (latitude, longitude, satellite,
--    acq_date, acq_time) to guarantee idempotent
--    upserts across redundant pipeline runs.
-- ==============================================
CREATE TABLE IF NOT EXISTS raw_firms_detections (
    id              UUID PRIMARY KEY,

    -- Geospatial point (SRID 4326 = WGS-84)
    geom            GEOMETRY(Point, 4326) NOT NULL,

    -- Source satellite / instrument identifier
    satellite       TEXT NOT NULL,

    -- Acquisition timestamp (date + time merged)
    acq_datetime    TIMESTAMPTZ NOT NULL,

    -- Fire Radiative Power (MW) — NULL when absent
    frp             DOUBLE PRECISION,

    -- Confidence metric — NULL when absent
    -- Stored as TEXT because MODIS uses categorical
    -- values (e.g. "nominal", "high") while VIIRS
    -- uses integer percentages.
    confidence      TEXT,

    -- Brightness temperature channels (Kelvin)
    brightness      DOUBLE PRECISION,
    bright_t31      DOUBLE PRECISION,

    -- Scan & track pixel dimensions (km)
    scan            DOUBLE PRECISION,
    track           DOUBLE PRECISION,

    -- Day / Night flag
    daynight        CHAR(1),

    -- FIRMS data version
    version         TEXT,

    -- Foreign key to the candidate event this
    -- detection was clustered into (set later
    -- by the clustering stage).
    candidate_event_id UUID,

    -- Ingestion bookkeeping
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Spatial index for fast geospatial queries
CREATE INDEX IF NOT EXISTS idx_raw_firms_geom
    ON raw_firms_detections USING GIST (geom);

-- Temporal index for time-range scans
CREATE INDEX IF NOT EXISTS idx_raw_firms_acq_datetime
    ON raw_firms_detections (acq_datetime);

-- Index on the FK to candidate events
CREATE INDEX IF NOT EXISTS idx_raw_firms_candidate_event
    ON raw_firms_detections (candidate_event_id);


-- ==============================================
-- 2. Candidate Events
--    Grouped / clustered thermal anomalies that
--    may represent a real-world fire or industrial
--    event. Populated by a downstream clustering
--    stage (Phase 2+).
-- ==============================================
CREATE TABLE IF NOT EXISTS candidate_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Representative centroid of the cluster
    centroid        GEOMETRY(Point, 4326) NOT NULL,

    -- Optional convex-hull or concave-hull footprint
    footprint       GEOMETRY(Polygon, 4326),

    -- Temporal extent
    first_seen      TIMESTAMPTZ NOT NULL,
    last_seen       TIMESTAMPTZ NOT NULL,

    -- Peak Fire Radiative Power observed (MW)
    max_frp         DOUBLE PRECISION,

    -- Computed priority / severity score
    priority_score  DOUBLE PRECISION,

    -- JSONB breakdown of how the priority score was computed.
    -- Keys: "frp_score", "persistence_score", "infrastructure_proximity_score",
    --        "land_cover_score", etc.  Values: numeric sub-scores and textual reasons.
    priority_reasons JSONB,

    -- Nearest OSM infrastructure (populated by enrichment pipeline)
    nearest_osm_type     TEXT,          -- e.g. "industrial", "power_plant"
    nearest_osm_name     TEXT,          -- human-readable name if available
    nearest_osm_distance_km DOUBLE PRECISION,  -- distance in km (NULL if unavailable)

    -- ML land-cover classification label (NULL if model hasn't run)
    land_cover_class TEXT,

    -- Phase 4: Imagery & ML Land-Cover Columns
    imagery_rgb_path        TEXT,
    imagery_swir_path       TEXT,
    gradcam_path            TEXT,
    land_cover_confidence   DOUBLE PRECISION,
    ml_model_arch           TEXT,
    ml_disclaimer           TEXT,
    sentinel2_scene_id      TEXT,
    imagery_fetched_at      TIMESTAMPTZ,

    -- Detection count in this cluster
    detection_count  INTEGER NOT NULL DEFAULT 0,

    -- Bookkeeping
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_candidate_events_centroid
    ON candidate_events USING GIST (centroid);

CREATE INDEX IF NOT EXISTS idx_candidate_events_last_seen
    ON candidate_events (last_seen);


-- ==============================================
-- 3. Foreign Key: raw_firms_detections →
--                 candidate_events
--    Added after both tables exist.
-- ==============================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'fk_raw_firms_candidate_event'
          AND table_name      = 'raw_firms_detections'
    ) THEN
        ALTER TABLE raw_firms_detections
            ADD CONSTRAINT fk_raw_firms_candidate_event
            FOREIGN KEY (candidate_event_id)
            REFERENCES candidate_events (id)
            ON DELETE SET NULL;
    END IF;
END
$$;


-- ==============================================
-- 4. Boundary Exclusions Audit
--    Logs every FIRMS detection that failed the
--    India-boundary intersection check, preserving
--    the original data and the reason for exclusion.
-- ==============================================
CREATE TABLE IF NOT EXISTS boundary_exclusions_audit (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- The deterministic detection UUID that was excluded
    detection_id    UUID NOT NULL,

    -- Original point geometry
    geom            GEOMETRY(Point, 4326) NOT NULL,

    -- Attributes echoed for forensic review
    satellite       TEXT,
    acq_datetime    TIMESTAMPTZ,
    frp             DOUBLE PRECISION,
    confidence      TEXT,

    -- Why it was excluded
    exclusion_reason TEXT NOT NULL DEFAULT 'outside_india_boundary',

    -- When the exclusion was logged
    logged_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_boundary_exclusions_geom
    ON boundary_exclusions_audit USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_boundary_exclusions_logged_at
    ON boundary_exclusions_audit (logged_at);



-- ==============================================
-- 5. Phase 3 — Idempotent column additions
--    Safe to re-run on an existing database.
-- ==============================================
DO $$
BEGIN
    -- priority_reasons JSONB
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_events' AND column_name = 'priority_reasons'
    ) THEN
        ALTER TABLE candidate_events ADD COLUMN priority_reasons JSONB;
    END IF;

    -- nearest OSM infrastructure columns
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_events' AND column_name = 'nearest_osm_type'
    ) THEN
        ALTER TABLE candidate_events ADD COLUMN nearest_osm_type TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_events' AND column_name = 'nearest_osm_name'
    ) THEN
        ALTER TABLE candidate_events ADD COLUMN nearest_osm_name TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_events' AND column_name = 'nearest_osm_distance_km'
    ) THEN
        ALTER TABLE candidate_events ADD COLUMN nearest_osm_distance_km DOUBLE PRECISION;
    END IF;

    -- ML land-cover label
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_events' AND column_name = 'land_cover_class'
    ) THEN
        ALTER TABLE candidate_events ADD COLUMN land_cover_class TEXT;
    END IF;

    -- detection_count
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_events' AND column_name = 'detection_count'
    ) THEN
        ALTER TABLE candidate_events ADD COLUMN detection_count INTEGER NOT NULL DEFAULT 0;
    END IF;
END
$$;



-- ==============================================
-- 6. Phase 4 — Imagery & ML classification columns
--    Idempotent additions for Sentinel-2 imagery
--    paths and ML land-cover metadata.
-- ==============================================
DO $$
BEGIN
    -- RGB image path (relative to static/imagery/)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_events' AND column_name = 'imagery_rgb_path'
    ) THEN
        ALTER TABLE candidate_events ADD COLUMN imagery_rgb_path TEXT;
    END IF;

    -- SWIR image path
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_events' AND column_name = 'imagery_swir_path'
    ) THEN
        ALTER TABLE candidate_events ADD COLUMN imagery_swir_path TEXT;
    END IF;

    -- ML confidence score (0.0–1.0)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_events' AND column_name = 'land_cover_confidence'
    ) THEN
        ALTER TABLE candidate_events ADD COLUMN land_cover_confidence DOUBLE PRECISION;
    END IF;

    -- ML model disclaimer string
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_events' AND column_name = 'ml_disclaimer'
    ) THEN
        ALTER TABLE candidate_events ADD COLUMN ml_disclaimer TEXT;
    END IF;

    -- Sentinel-2 scene identifier used for imagery
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_events' AND column_name = 'sentinel2_scene_id'
    ) THEN
        ALTER TABLE candidate_events ADD COLUMN sentinel2_scene_id TEXT;
    END IF;

    -- Timestamp when imagery was fetched
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'candidate_events' AND column_name = 'imagery_fetched_at'
    ) THEN
        ALTER TABLE candidate_events ADD COLUMN imagery_fetched_at TIMESTAMPTZ;
    END IF;
END
$$;


-- ==============================================
-- 7. Local OpenStreetMap Industrial Features Index
--    Extracted from Geofabrik India OSM PBF.
-- ==============================================
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

