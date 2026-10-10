import sys
from pathlib import Path
import pandas as pd
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from thermaleye.config import DEFAULT_INDIA_BBOX

def main():
    print(f"Current India Bounding Box Filter:")
    print(f"  Lat: {DEFAULT_INDIA_BBOX.min_lat} to {DEFAULT_INDIA_BBOX.max_lat}")
    print(f"  Lon: {DEFAULT_INDIA_BBOX.min_lon} to {DEFAULT_INDIA_BBOX.max_lon}")
    
    events_path = Path("data/processed/thermal_events.csv")
    if not events_path.exists():
        print("Events file not found.")
        return
        
    df = pd.read_csv(events_path)
    print(f"\nTotal thermal events evaluated: {len(df)}")
    
    # Let's do a fast offline check if geopandas/shapely is installed
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        
        # Load low-res world map
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        # Create GeoDataFrame for our events
        geometry = [Point(xy) for xy in zip(df.centroid_lon, df.centroid_lat)]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        # Spatial join
        joined = gpd.sjoin(gdf, world, how="left", predicate="intersects")
        
        print("\nGeographic Distribution of Thermal Events:")
        counts = joined['name'].value_counts(dropna=False)
        for country, count in counts.items():
            print(f"  {country if pd.notna(country) else 'Ocean/Unknown'}: {count} events")
            
        non_india = joined[joined['name'] != 'India']
        print(f"\nTotal events strictly outside India's polygon: {len(non_india)}")
        if len(non_india) > 0:
            sample = non_india.iloc[0]
            print(f"  Example: {sample['event_id']} is in {sample['name']} (Lat: {sample['centroid_lat']}, Lon: {sample['centroid_lon']})")
            
    except ImportError:
        print("\nGeopandas not installed. Falling back to bounding box bounds check.")
        # Manual bounding box check is what it does.
        # But we know 17.7 N, 95.2 E is Myanmar.
        print("Without a GIS polygon library, a rectangular bounding box over India will inherently cover neighboring countries like Myanmar, Nepal, Pakistan, and Bangladesh.")

if __name__ == "__main__":
    main()
