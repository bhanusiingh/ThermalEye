"""Configuration settings for ThermalEye data ingestion."""

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Tuple
import warnings
from dotenv import load_dotenv

# Suppress harmless third-party version mismatch warning
warnings.filterwarnings("ignore", message=".*urllib3.*")

# Load environment variables from .env file if present
load_dotenv()


@dataclass(frozen=True)
class BoundingBox:
    """Geographic bounding box defined by min_lon, min_lat, max_lon, max_lat."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def __post_init__(self) -> None:
        if not (-180.0 <= self.min_lon <= 180.0 and -180.0 <= self.max_lon <= 180.0):
            raise ValueError(f"Longitude values must be within [-180, 180]: min_lon={self.min_lon}, max_lon={self.max_lon}")
        if not (-90.0 <= self.min_lat <= 90.0 and -90.0 <= self.max_lat <= 90.0):
            raise ValueError(f"Latitude values must be within [-90, 90]: min_lat={self.min_lat}, max_lat={self.max_lat}")
        if self.min_lon >= self.max_lon:
            raise ValueError(f"min_lon ({self.min_lon}) must be less than max_lon ({self.max_lon})")
        if self.min_lat >= self.max_lat:
            raise ValueError(f"min_lat ({self.min_lat}) must be less than max_lat ({self.max_lat})")

    def to_firms_str(self) -> str:
        """Format as expected by FIRMS API: min_lon,min_lat,max_lon,max_lat."""
        return f"{self.min_lon},{self.min_lat},{self.max_lon},{self.max_lat}"


# Standard bounding box covering the Indian subcontinent:
# Longitudes ~68.1E to 97.4E, Latitudes ~6.7N to 37.1N
DEFAULT_INDIA_BBOX = BoundingBox(
    min_lon=68.1,
    min_lat=6.7,
    max_lon=97.4,
    max_lat=37.1,
)

# Supported FIRMS Near-Real-Time VIIRS products for NOAA-20 and NOAA-21
DEFAULT_VIIRS_PRODUCTS: Tuple[str, ...] = (
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
)

FIRMS_API_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
DEFAULT_DAY_RANGE = 1
DEFAULT_REQUEST_TIMEOUT = 30  # seconds
DEFAULT_OUTPUT_PATH = Path("data/samples/india_viirs_recent.csv")


def get_firms_map_key() -> str | None:
    """Retrieve FIRMS_MAP_KEY from environment without logging or exposing it."""
    key = os.getenv("FIRMS_MAP_KEY", "").strip()
    return key if key else None


# OpenStreetMap & Overpass API settings
OVERPASS_API_URL = os.getenv("OVERPASS_API_URL", "https://overpass-api.de/api/interpreter").strip()
DEFAULT_OSM_SEARCH_RADIUS_M = 3000  # 3.0 km search radius
DEFAULT_OSM_TIMEOUT = 25  # seconds
DEFAULT_OSM_CACHE_PATH = Path("data/cache/osm_cache.json")
DEFAULT_OSM_OUTPUT_PATH = Path("data/processed/thermal_events_osm.csv")
DEFAULT_MAX_OSM_EVENTS = 10  # safe limit for testing/CLI
DEFAULT_OSM_REQUEST_DELAY = 1.5  # seconds delay between live queries to respect public API rate limits

# Copernicus Data Space Ecosystem (CDSE) / Sentinel Hub settings
CDSE_TOKEN_URL = os.getenv(
    "CDSE_TOKEN_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
).strip()
CDSE_CATALOG_URL = os.getenv(
    "CDSE_CATALOG_URL",
    "https://sh.dataspace.copernicus.eu/catalog/v1/search",
).strip()
CDSE_PROCESS_URL = os.getenv(
    "CDSE_PROCESS_URL",
    "https://sh.dataspace.copernicus.eu/process/v1",
).strip()
CDSE_S2_COLLECTION = "sentinel-2-l2a"

# Patch size and output settings
DEFAULT_S2_DATE_RANGE_DAYS = 10      # days before event last_detected to search for scenes
DEFAULT_S2_MAX_CLOUD_PCT = 80.0      # maximum cloud coverage % to accept a scene
DEFAULT_S2_PATCH_WIDTH_PX = 64       # output patch width in pixels (matches EuroSAT 64x64)
DEFAULT_S2_PATCH_HEIGHT_PX = 64      # output patch height in pixels
DEFAULT_S2_PATCH_HALF_DEG = 0.003    # half-width in degrees (~333m, total bbox ~666m, matches EuroSAT 640m extent)
DEFAULT_S2_OUTPUT_DIR = Path("data/samples/sentinel2")
DEFAULT_S2_META_PATH = Path("data/processed/sentinel2_metadata.csv")
DEFAULT_S2_REQUEST_TIMEOUT = 30      # seconds


def get_cdse_credentials() -> tuple[str | None, str | None]:
    """Return CDSE (client_id, client_secret) from environment without exposing values."""
    client_id = os.getenv("CDSE_CLIENT_ID", "").strip() or None
    client_secret = os.getenv("CDSE_CLIENT_SECRET", "").strip() or None
    return client_id, client_secret


