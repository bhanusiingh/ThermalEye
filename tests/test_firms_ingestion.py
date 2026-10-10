"""Offline unit tests for NASA FIRMS ingestion module."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import requests

from thermaleye.config import BoundingBox, DEFAULT_INDIA_BBOX
from thermaleye.ingestion.firms import (
    FirmsClient,
    FirmsAPIError,
    FirmsConfigError,
    deduplicate_observations,
    run_ingestion,
)

SAMPLE_VIIRS_CSV_NOAA20 = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
28.6139,77.2090,325.4,0.4,0.4,2026-10-09,0730,20,VIIRS,nominal,2.0NRT,295.2,12.5,D
28.6200,77.2150,330.1,0.4,0.4,2026-10-09,0730,20,VIIRS,high,2.0NRT,298.0,18.2,D
"""

SAMPLE_VIIRS_CSV_NOAA21 = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
19.0760,72.8777,318.2,0.4,0.4,2026-10-09,0815,21,VIIRS,nominal,2.0NRT,293.1,8.4,D
"""


class TestBoundingBox:
    """Test BoundingBox validation and formatting."""

    def test_valid_bbox(self) -> None:
        bbox = BoundingBox(min_lon=68.1, min_lat=6.7, max_lon=97.4, max_lat=37.1)
        assert bbox.to_firms_str() == "68.1,6.7,97.4,37.1"

    def test_invalid_lat_range(self) -> None:
        with pytest.raises(ValueError, match="Latitude values must be within"):
            BoundingBox(min_lon=68.1, min_lat=-95.0, max_lon=97.4, max_lat=37.1)

    def test_invalid_lon_order(self) -> None:
        with pytest.raises(ValueError, match="min_lon .* must be less than max_lon"):
            BoundingBox(min_lon=97.4, min_lat=6.7, max_lon=68.1, max_lat=37.1)

    def test_invalid_lat_order(self) -> None:
        with pytest.raises(ValueError, match="min_lat .* must be less than max_lat"):
            BoundingBox(min_lon=68.1, min_lat=37.1, max_lon=97.4, max_lat=6.7)


class TestFirmsClient:
    """Test FirmsClient URL building, credential safety, and response parsing."""

    def test_missing_credentials_raises_config_error(self) -> None:
        with patch("thermaleye.ingestion.firms.get_firms_map_key", return_value=None):
            with pytest.raises(FirmsConfigError, match="FIRMS_MAP_KEY is missing"):
                FirmsClient(map_key=None)

    def test_build_request_url_format(self) -> None:
        client = FirmsClient(map_key="test_secret_key_123")
        url = client.build_request_url(
            product="VIIRS_NOAA20_NRT",
            bbox=DEFAULT_INDIA_BBOX,
            day_range=1,
        )
        expected_url = (
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            "test_secret_key_123/VIIRS_NOAA20_NRT/68.1,6.7,97.4,37.1/1"
        )
        assert url == expected_url

    def test_build_request_url_with_date(self) -> None:
        client = FirmsClient(map_key="test_secret_key_123")
        url = client.build_request_url(
            product="VIIRS_NOAA21_NRT",
            bbox=DEFAULT_INDIA_BBOX,
            day_range=2,
            date_str="2026-10-09",
        )
        assert url.endswith("/2/2026-10-09")

    def test_invalid_day_range(self) -> None:
        client = FirmsClient(map_key="test_secret_key_123")
        with pytest.raises(ValueError, match="day_range must be between 1 and 10"):
            client.build_request_url("VIIRS_NOAA20_NRT", DEFAULT_INDIA_BBOX, day_range=0)

    def test_sanitize_redacts_secret_key(self) -> None:
        client = FirmsClient(map_key="SUPER_SECRET_FIRMS_TOKEN")
        text = "Request sent to https://firms.../SUPER_SECRET_FIRMS_TOKEN/VIIRS_NOAA20_NRT"
        sanitized = client._sanitize(text)
        assert "SUPER_SECRET_FIRMS_TOKEN" not in sanitized
        assert "***REDACTED***" in sanitized

    def test_parse_valid_csv(self) -> None:
        client = FirmsClient(map_key="test_key")
        df = client.parse_csv_response(SAMPLE_VIIRS_CSV_NOAA20, "VIIRS_NOAA20_NRT")
        assert len(df) == 2
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        assert "source_product" in df.columns
        assert df["source_product"].iloc[0] == "VIIRS_NOAA20_NRT"

    def test_parse_empty_csv(self) -> None:
        client = FirmsClient(map_key="test_key")
        df = client.parse_csv_response("", "VIIRS_NOAA20_NRT")
        assert df.empty
        assert "latitude" in df.columns
        assert "source_product" in df.columns

    def test_parse_csv_with_header_only(self) -> None:
        header_only = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
        client = FirmsClient(map_key="test_key")
        df = client.parse_csv_response(header_only, "VIIRS_NOAA20_NRT")
        assert df.empty
        assert "latitude" in df.columns
        assert "source_product" in df.columns

    def test_parse_html_error_response_raises(self) -> None:
        html_payload = "<!DOCTYPE html><html><body>502 Bad Gateway</body></html>"
        client = FirmsClient(map_key="test_key")
        with pytest.raises(FirmsAPIError, match="HTML response"):
            client.parse_csv_response(html_payload, "VIIRS_NOAA20_NRT")

    def test_parse_firms_plaintext_error_raises(self) -> None:
        client = FirmsClient(map_key="my_secret")
        with pytest.raises(FirmsAPIError, match="FIRMS API reported error: Invalid MAP_KEY"):
            client.parse_csv_response("Invalid MAP_KEY\n", "VIIRS_NOAA20_NRT")

    def test_parse_unrecognized_response_raises(self) -> None:
        client = FirmsClient(map_key="test_key")
        with pytest.raises(FirmsAPIError, match="Unrecognized response header|Expected latitude"):
            client.parse_csv_response("Unexpected Server Error Without Commas\n", "VIIRS_NOAA20_NRT")


class TestDeduplication:
    """Test observation deduplication logic."""

    def test_preserves_distinct_satellite_observations(self) -> None:
        client = FirmsClient(map_key="test_key")
        df1 = client.parse_csv_response(SAMPLE_VIIRS_CSV_NOAA20, "VIIRS_NOAA20_NRT")
        df2 = client.parse_csv_response(SAMPLE_VIIRS_CSV_NOAA21, "VIIRS_NOAA21_NRT")
        combined = pd.concat([df1, df2], ignore_index=True)
        deduped = deduplicate_observations(combined)
        assert len(deduped) == 3

    def test_removes_identical_observations(self) -> None:
        client = FirmsClient(map_key="test_key")
        df1 = client.parse_csv_response(SAMPLE_VIIRS_CSV_NOAA20, "VIIRS_NOAA20_NRT")
        # Duplicate df1 rows
        duplicated = pd.concat([df1, df1], ignore_index=True)
        assert len(duplicated) == 4
        deduped = deduplicate_observations(duplicated)
        assert len(deduped) == 2


class TestIngestionWorkflow:
    """Test the complete run_ingestion workflow with mocked HTTP calls."""

    @patch("requests.get")
    def test_successful_multi_product_ingestion(self, mock_get: MagicMock, tmp_path: Path) -> None:
        def side_effect(url: str, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            if "VIIRS_NOAA20_NRT" in url:
                resp.text = SAMPLE_VIIRS_CSV_NOAA20
            else:
                resp.text = SAMPLE_VIIRS_CSV_NOAA21
            return resp

        mock_get.side_effect = side_effect

        out_file = tmp_path / "test_output.csv"
        summary = run_ingestion(
            map_key="dummy_map_key",
            products=("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"),
            output_path=out_file,
        )

        assert len(summary.successful_requests) == 2
        assert len(summary.failed_requests) == 0
        assert summary.saved_rows == 3
        assert out_file.exists()

        saved_df = pd.read_csv(out_file)
        assert len(saved_df) == 3
        assert set(saved_df["source_product"].unique()) == {"VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"}

    @patch("requests.get")
    def test_partial_failure_handling(self, mock_get: MagicMock, tmp_path: Path) -> None:
        def side_effect(url: str, **kwargs):
            resp = MagicMock()
            if "VIIRS_NOAA20_NRT" in url:
                resp.status_code = 200
                resp.text = SAMPLE_VIIRS_CSV_NOAA20
            else:
                resp.status_code = 500
                resp.text = "Internal Server Error"
            return resp

        mock_get.side_effect = side_effect

        out_file = tmp_path / "test_output_partial.csv"
        summary = run_ingestion(
            map_key="dummy_map_key",
            products=("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"),
            output_path=out_file,
        )

        assert "VIIRS_NOAA20_NRT" in summary.successful_requests
        assert len(summary.failed_requests) == 1
        assert summary.failed_requests[0][0] == "VIIRS_NOAA21_NRT"
        assert summary.saved_rows == 2
        assert out_file.exists()

    @patch("requests.get")
    def test_all_products_fail_raises_api_error(self, mock_get: MagicMock, tmp_path: Path) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Invalid MAP_KEY"
        mock_get.return_value = mock_resp

        out_file = tmp_path / "test_output_fail.csv"
        with pytest.raises(FirmsAPIError, match="All FIRMS product queries failed"):
            run_ingestion(
                map_key="dummy_map_key",
                products=("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"),
                output_path=out_file,
            )
        assert not out_file.exists()
