import pandas as pd
from unittest.mock import patch, MagicMock
from thermaleye.geo.boundary import filter_points_by_country

def test_filter_points_geopandas_missing():
    import pytest
    df = pd.DataFrame({"latitude": [10.0, 20.0], "longitude": [80.0, 90.0]})
    with patch("builtins.__import__", side_effect=ImportError):
        with pytest.raises(RuntimeError, match="without geopandas"):
            filter_points_by_country(df)

def test_filter_points_empty_dataframe():
    import pytest
    pytest.importorskip("geopandas")
    df = pd.DataFrame(columns=["latitude", "longitude"])
    result = filter_points_by_country(df)
    assert result.empty

def test_filter_points_success():
    import pytest
    pytest.importorskip("geopandas")
        
    # Setup some test points
    # 21.0, 79.0 is inside India (Nagpur)
    # 17.7, 95.2 is outside India (Myanmar)
    # 0.0, 0.0 is outside India (Ocean)
    # 10.0, 70.0 is exactly on the boundary corner of our mock polygon
    # 30.1, 70.0 is just outside the boundary corner (lat 30.1 vs max_lat 30)
    # NaN coords should be dropped
    
    df = pd.DataFrame({
        "latitude": [21.0, 17.7, 0.0, 10.0, 30.1, None],
        "longitude": [79.0, 95.2, 0.0, 70.0, 70.0, 79.0],
        "id": [1, 2, 3, 4, 5, 6]
    })
    
    # We will patch get_country_polygon to avoid downloading large files during tests if possible,
    # but Natural Earth is small enough or we can mock it. Let's mock it to make test fast offline.
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    # Create a mock polygon roughly covering central India (min_lon=70, max_lon=88, min_lat=10, max_lat=30)
    mock_poly = Polygon([(70, 10), (88, 10), (88, 30), (70, 30), (70, 10)])
    mock_gdf = gpd.GeoDataFrame({"name": ["India"]}, geometry=[mock_poly], crs="EPSG:4326")
    
    with patch("thermaleye.geo.boundary.get_country_polygon", return_value=mock_gdf):
        result = filter_points_by_country(df, country_name="India")
        
    # Inside and on-boundary points should remain (id 1 and id 4)
    assert len(result) == 2
    kept_ids = result["id"].tolist()
    assert 1 in kept_ids
    assert 4 in kept_ids
    assert 2 not in kept_ids
    assert 5 not in kept_ids
