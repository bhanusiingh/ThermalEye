"""Geospatial boundaries and filtering."""
import logging
from pathlib import Path
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Standard Natural Earth GeoJSON available via DataHub
NATURAL_EARTH_GEOJSON_URL = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
CACHE_DIR = Path("data/cache")
GEOJSON_CACHE_PATH = CACHE_DIR / "countries.geojson"

def get_country_polygon(country_name: str = "India"):
    """Fetch and return the GeoPandas polygon for a specific country."""
    try:
        import geopandas as gpd
    except ImportError:
        logger.error("geopandas not installed. Cannot retrieve country polygon.")
        raise RuntimeError("geopandas is required for boundary retrieval.")

    if not GEOJSON_CACHE_PATH.exists():
        logger.info(f"Downloading country boundaries from {NATURAL_EARTH_GEOJSON_URL}...")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        resp = requests.get(NATURAL_EARTH_GEOJSON_URL, timeout=30)
        resp.raise_for_status()
        with open(GEOJSON_CACHE_PATH, "w", encoding="utf-8") as f:
            f.write(resp.text)

    # Load GeoJSON
    world = gpd.read_file(GEOJSON_CACHE_PATH)
    
    # Check column names (DataHub GeoJSON uses ADMIN)
    country_col = 'ADMIN' if 'ADMIN' in world.columns else 'name'
    
    country_gdf = world[world[country_col] == country_name]
    if country_gdf.empty:
        raise ValueError(f"Country '{country_name}' not found in boundaries dataset.")
        
    return country_gdf

def filter_points_by_country(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude", country_name: str = "India") -> pd.DataFrame:
    """Filter a DataFrame of points, keeping only those within the country's border."""
    try:
        import geopandas as gpd
    except ImportError:
        logger.error("geopandas is required for boundary filtering but is not installed.")
        raise RuntimeError(f"Cannot perform {country_name}-specific boundary filtering without geopandas.")

    country_gdf = get_country_polygon(country_name)
    if country_gdf is None or df.empty:
        return df
        
    # Drop rows with missing coordinates so shapely doesn't crash
    df_clean = df.dropna(subset=[lat_col, lon_col]).copy()
    
    # Create GeoDataFrame
    points = gpd.points_from_xy(df_clean[lon_col], df_clean[lat_col])
    gdf = gpd.GeoDataFrame(df_clean, geometry=points, crs="EPSG:4326")
    
    # Use sjoin with intersects to filter
    filtered = gpd.sjoin(gdf, country_gdf, how="inner", predicate="intersects")
    
    # Cleanup columns added by sjoin
    cols_to_drop = [c for c in filtered.columns if c not in df.columns]
    filtered = filtered.drop(columns=cols_to_drop)
    
    logger.info(f"Filtered {len(df)} points to {len(filtered)} inside {country_name}")
    return filtered
