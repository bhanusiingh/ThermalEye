"""
ThermalEye — Local OpenStreetMap Industrial Features PBF Importer
==================================================================

Streams and parses an OpenStreetMap .osm.pbf snapshot using pyosmium,
extracting industrial, energy, and mining features into PostGIS table
`osm_industrial_features` for rapid local offline spatial indexing.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import osmium
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("thermaleye.import_osm_pbf")

_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_script_dir, ".env"))


def get_database_url() -> str:
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://thermaleye:thermaleye_secret@localhost:5432/thermaleye",
    )
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    if "+" not in url.split("://")[0]:
        try:
            import psycopg2
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        except ImportError:
            try:
                import psycopg
                url = url.replace("postgresql://", "postgresql+psycopg://", 1)
            except ImportError:
                pass
    return url


# ---------------------------------------------------------------------------
# Matching & Classification Rules
# ---------------------------------------------------------------------------
INDUSTRIAL_LANDUSE = {"industrial", "quarry", "brownfield"}
POWER_TAGS = {"plant", "generator", "substation"}
MAN_MADE_TAGS = {
    "works",
    "kiln",
    "chimney",
    "petroleum_well",
    "flare",
    "refinery",
    "gasometer",
    "storage_tank",
    "mineshaft",
    "adit",
}


def classify_feature(tags: Any) -> Optional[Tuple[str, str]]:
    """
    Determine (category, osm_type) based on tags.

    Categories:
      - 'industrial'
      - 'power_plant'
      - 'mining'
      - 'kiln_refinery'
      - 'energy_infra'

    osm_type is formatted as 'key=value'.
    """
    # 1. Power infrastructure
    power = tags.get("power")
    if power in POWER_TAGS:
        category = "power_plant" if power == "plant" else "energy_infra"
        return category, f"power={power}"

    # 2. Man-made industrial structures
    man_made = tags.get("man_made")
    if man_made in MAN_MADE_TAGS:
        if man_made in {
            "kiln",
            "refinery",
            "flare",
            "petroleum_well",
            "gasometer",
            "storage_tank",
        }:
            category = "kiln_refinery"
        elif man_made in {"mineshaft", "adit"}:
            category = "mining"
        else:
            category = "industrial"
        return category, f"man_made={man_made}"

    # 3. Explicit mining, quarrying, resources
    if "mining" in tags:
        return "mining", f"mining={tags.get('mining')}"
    if "quarry" in tags:
        return "mining", f"quarry={tags.get('quarry')}"
    if "resource" in tags:
        return "mining", f"resource={tags.get('resource')}"

    # 4. Landuse
    landuse = tags.get("landuse")
    if landuse in INDUSTRIAL_LANDUSE:
        category = "mining" if landuse == "quarry" else "industrial"
        return category, f"landuse={landuse}"

    # 5. Any generic industrial key
    if "industrial" in tags:
        return "industrial", f"industrial={tags.get('industrial')}"

    return None


# ---------------------------------------------------------------------------
# Database Schema Initialization
# ---------------------------------------------------------------------------
def ensure_osm_table(engine) -> None:
    """Ensure the osm_industrial_features table and GIST indexes exist."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
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
                """
            )
        )
    logger.info("Ensured osm_industrial_features schema and spatial indexes exist.")


class _LimitReached(Exception):
    """Internal sentinel exception to stop parsing when limit is reached."""
    pass


# ---------------------------------------------------------------------------
# Streaming Osmium Handler
# ---------------------------------------------------------------------------
class IndustrialOSMHandler(osmium.SimpleHandler):
    """
    Streaming handler that filters nodes and ways for industrial/power facilities
    and writes them in batches into PostGIS.
    """

    def __init__(
        self,
        engine,
        batch_size: int = 2000,
        limit: Optional[int] = None,
    ):
        super().__init__()
        self.engine = engine
        self.batch_size = batch_size
        self.limit = limit

        self.batch: List[Dict[str, Any]] = []
        self.total_inserted = 0
        self.nodes_matched = 0
        self.ways_matched = 0
        self.start_time = time.time()

    def _flush_batch(self) -> None:
        if not self.batch:
            return

        upsert_query = text(
            """
            INSERT INTO osm_industrial_features (osm_id, name, category, osm_type, geom)
            VALUES (:osm_id, :name, :category, :osm_type, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326))
            ON CONFLICT (osm_id) DO UPDATE SET
                name = EXCLUDED.name,
                category = EXCLUDED.category,
                osm_type = EXCLUDED.osm_type,
                geom = EXCLUDED.geom
            """
        )

        with self.engine.begin() as conn:
            conn.execute(upsert_query, self.batch)

        self.total_inserted += len(self.batch)
        self.batch.clear()

        elapsed = time.time() - self.start_time
        rate = self.total_inserted / elapsed if elapsed > 0 else 0
        logger.info(
            "Flushed batch. Total extracted: %d (Nodes: %d, Ways: %d) | Elapsed: %.1fs (%.1f feats/sec)",
            self.total_inserted,
            self.nodes_matched,
            self.ways_matched,
            elapsed,
            rate,
        )

    def node(self, n: osmium.osm.Node) -> None:
        if self.limit and (self.nodes_matched + self.ways_matched) >= self.limit:
            raise _LimitReached()

        if not n.location.valid():
            return

        classification = classify_feature(n.tags)
        if not classification:
            return

        category, osm_type = classification
        name = n.tags.get("name")
        osm_id = f"node/{n.id}"

        self.batch.append(
            {
                "osm_id": osm_id,
                "name": name,
                "category": category,
                "osm_type": osm_type,
                "lon": float(n.location.lon),
                "lat": float(n.location.lat),
            }
        )
        self.nodes_matched += 1

        if len(self.batch) >= self.batch_size:
            self._flush_batch()

        if self.limit and (self.nodes_matched + self.ways_matched) >= self.limit:
            raise _LimitReached()

    def way(self, w: osmium.osm.Way) -> None:
        if self.limit and (self.nodes_matched + self.ways_matched) >= self.limit:
            raise _LimitReached()

        classification = classify_feature(w.tags)
        if not classification:
            return

        # Compute centroid from valid constituent nodes
        lons: List[float] = []
        lats: List[float] = []
        for pt in w.nodes:
            try:
                lons.append(pt.lon)
                lats.append(pt.lat)
            except osmium.InvalidLocationError:
                continue

        if not lons:
            return

        mean_lon = sum(lons) / len(lons)
        mean_lat = sum(lats) / len(lats)

        category, osm_type = classification
        name = w.tags.get("name")
        osm_id = f"way/{w.id}"

        self.batch.append(
            {
                "osm_id": osm_id,
                "name": name,
                "category": category,
                "osm_type": osm_type,
                "lon": float(mean_lon),
                "lat": float(mean_lat),
            }
        )
        self.ways_matched += 1

        if len(self.batch) >= self.batch_size:
            self._flush_batch()

        if self.limit and (self.nodes_matched + self.ways_matched) >= self.limit:
            raise _LimitReached()

    def finalize(self) -> None:
        self._flush_batch()
        elapsed = time.time() - self.start_time
        logger.info(
            "Import complete: %d features inserted (Nodes: %d, Ways: %d) in %.2fs.",
            self.total_inserted,
            self.nodes_matched,
            self.ways_matched,
            elapsed,
        )


# ---------------------------------------------------------------------------
# PBF Path Discovery
# ---------------------------------------------------------------------------
def find_pbf_file(explicit_path: Optional[str] = None) -> str:
    """Resolve PBF file path from CLI argument or search standard locations."""
    if explicit_path:
        if os.path.exists(explicit_path):
            return os.path.abspath(explicit_path)
        raise FileNotFoundError(f"Specified PBF file does not exist: {explicit_path}")

    patterns = [
        os.path.join(_script_dir, "data", "osm", "*.osm.pbf"),
        os.path.join(_script_dir, "data", "*.osm.pbf"),
        os.path.join("data", "osm", "*.osm.pbf"),
        os.path.join("data", "*.osm.pbf"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            matches.sort(key=os.path.getsize, reverse=True)
            return os.path.abspath(matches[0])

    raise FileNotFoundError(
        "No .osm.pbf file found in data/osm/ or data/. "
        "Please provide --pbf <path> or place india-*.osm.pbf in data/osm/."
    )


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run_import(
    pbf_path: Optional[str] = None,
    batch_size: int = 2000,
    limit: Optional[int] = None,
    db_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the PBF importer end-to-end."""
    resolved_pbf = find_pbf_file(pbf_path)
    file_size_mb = os.path.getsize(resolved_pbf) / (1024 * 1024)
    logger.info("Found OSM PBF snapshot: %s (%.1f MB)", resolved_pbf, file_size_mb)

    resolved_db = db_url or get_database_url()
    engine = create_engine(resolved_db)

    ensure_osm_table(engine)

    handler = IndustrialOSMHandler(engine, batch_size=batch_size, limit=limit)
    logger.info("Starting streaming parse with apply_file(locations=True, idx='flex_mem')...")
    start = time.time()
    try:
        handler.apply_file(resolved_pbf, locations=True, idx="flex_mem")
    except _LimitReached:
        logger.info("Extraction limit of %d reached. Finishing import early.", limit)
    finally:
        handler.finalize()

    duration = time.time() - start
    return {
        "pbf_file": resolved_pbf,
        "features_inserted": handler.total_inserted,
        "nodes_matched": handler.nodes_matched,
        "ways_matched": handler.ways_matched,
        "duration_seconds": duration,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stream and import OpenStreetMap industrial/energy features from .osm.pbf to PostGIS."
    )
    parser.add_argument(
        "--pbf",
        type=str,
        default=None,
        help="Path to .osm.pbf snapshot (auto-detects data/osm/*.osm.pbf if omitted)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Batch insert chunk size (default: 2000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of features to extract",
    )
    args = parser.parse_args()

    try:
        res = run_import(pbf_path=args.pbf, batch_size=args.batch_size, limit=args.limit)
        print(f"PBF Import successfully completed: {res['features_inserted']} features imported in {res['duration_seconds']:.1f}s.")
    except Exception as exc:
        logger.exception("PBF import failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
