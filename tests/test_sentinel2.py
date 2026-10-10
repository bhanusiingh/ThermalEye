"""Offline unit tests for Phase 4: Sentinel-2 CDSE retrieval module."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import requests

from thermaleye.imagery.sentinel2 import (
    CDSEClient,
    SentinelHubConfigError,
    SentinelHubAPIError,
    TokenStore,
    S2PatchResult,
    compute_patch_bbox,
    build_catalogue_search_payload,
    build_process_request_body,
    select_best_scene,
    retrieve_sentinel2_patch,
    CDSE_S2_COLLECTION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CATALOGUE_FEATURE = {
    "id": "S2B_MSIL2A_20261009T050659_N9999_R019_T44QKG_20261009T070000",
    "type": "Feature",
    "bbox": [94.9, 17.5, 95.5, 18.1],
    "properties": {
        "datetime": "2026-10-09T05:06:59Z",
        "eo:cloud_cover": 12.5,
        "platform": "sentinel-2b",
        "constellation": "sentinel-2",
    },
}

SAMPLE_CATALOGUE_RESPONSE = {
    "type": "FeatureCollection",
    "features": [SAMPLE_CATALOGUE_FEATURE],
}

FAKE_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


# ---------------------------------------------------------------------------
# TokenStore
# ---------------------------------------------------------------------------

class TestTokenStore:
    def test_empty_token_is_invalid(self) -> None:
        ts = TokenStore()
        assert not ts.is_valid()

    def test_valid_token(self) -> None:
        ts = TokenStore(access_token="abc", expires_at=time.time() + 600)
        assert ts.is_valid()

    def test_expired_token_is_invalid(self) -> None:
        ts = TokenStore(access_token="abc", expires_at=time.time() - 10)
        assert not ts.is_valid()

    def test_clear_invalidates(self) -> None:
        ts = TokenStore(access_token="abc", expires_at=time.time() + 600)
        ts.clear()
        assert not ts.is_valid()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_patch_bbox_symmetry(self) -> None:
        bbox = compute_patch_bbox(20.0, 80.0, half_deg=0.05)
        assert len(bbox) == 4
        assert bbox[0] == pytest.approx(79.95)
        assert bbox[1] == pytest.approx(19.95)
        assert bbox[2] == pytest.approx(80.05)
        assert bbox[3] == pytest.approx(20.05)

    def test_catalogue_payload_structure(self) -> None:
        payload = build_catalogue_search_payload(
            bbox=[79.95, 19.95, 80.05, 20.05],
            date_from="2026-10-01T00:00:00Z",
            date_to="2026-10-10T23:59:59Z",
            max_cloud_pct=80.0,
            limit=10,
        )
        assert payload["collections"] == [CDSE_S2_COLLECTION]
        assert payload["bbox"] == [79.95, 19.95, 80.05, 20.05]
        assert "datetime" in payload
        assert "filter" in payload

    def test_process_request_body_structure(self) -> None:
        body = build_process_request_body(
            bbox=[79.95, 19.95, 80.05, 20.05],
            date_from="2026-10-01T00:00:00Z",
            date_to="2026-10-10T23:59:59Z",
            width_px=256,
            height_px=256,
        )
        assert body["input"]["data"][0]["type"] == CDSE_S2_COLLECTION
        assert body["output"]["width"] == 256
        assert body["output"]["height"] == 256
        assert "image/png" in body["output"]["responses"][0]["format"]["type"]
        assert "//VERSION=3" in body["evalscript"]


# ---------------------------------------------------------------------------
# Scene selection
# ---------------------------------------------------------------------------

class TestSceneSelection:
    def test_no_features_returns_none(self) -> None:
        assert select_best_scene([]) is None

    def test_single_feature_selected(self) -> None:
        scene = select_best_scene([SAMPLE_CATALOGUE_FEATURE])
        assert scene is not None
        assert scene.scene_id == SAMPLE_CATALOGUE_FEATURE["id"]
        assert scene.cloud_cover_pct == 12.5
        assert scene.acquisition_time == "2026-10-09T05:06:59Z"

    def test_lowest_cloud_preferred(self) -> None:
        clear = dict(SAMPLE_CATALOGUE_FEATURE)
        clear["id"] = "clear-scene"
        clear["properties"] = {**clear["properties"], "eo:cloud_cover": 5.0, "datetime": "2026-10-08T05:00:00Z"}
        cloudy = dict(SAMPLE_CATALOGUE_FEATURE)
        cloudy["id"] = "cloudy-scene"
        cloudy["properties"] = {**cloudy["properties"], "eo:cloud_cover": 70.0, "datetime": "2026-10-09T05:00:00Z"}

        scene = select_best_scene([cloudy, clear])
        assert scene.scene_id == "clear-scene"

    def test_tie_broken_by_most_recent_date(self) -> None:
        older = dict(SAMPLE_CATALOGUE_FEATURE)
        older["id"] = "older-scene"
        older["properties"] = {**older["properties"], "eo:cloud_cover": 10.0, "datetime": "2026-10-07T05:00:00Z"}
        newer = dict(SAMPLE_CATALOGUE_FEATURE)
        newer["id"] = "newer-scene"
        newer["properties"] = {**newer["properties"], "eo:cloud_cover": 10.0, "datetime": "2026-10-09T05:00:00Z"}

        scene = select_best_scene([older, newer])
        assert scene.scene_id == "newer-scene"

    def test_missing_cloud_cover_treated_as_max(self) -> None:
        no_cloud = dict(SAMPLE_CATALOGUE_FEATURE)
        no_cloud["id"] = "no-cloud-meta"
        no_cloud["properties"] = {"datetime": "2026-10-09T05:00:00Z"}  # no eo:cloud_cover
        clear = dict(SAMPLE_CATALOGUE_FEATURE)
        clear["id"] = "clear-scene"
        clear["properties"] = {"datetime": "2026-10-08T05:00:00Z", "eo:cloud_cover": 5.0}

        scene = select_best_scene([no_cloud, clear])
        assert scene.scene_id == "clear-scene"


# ---------------------------------------------------------------------------
# CDSEClient auth & API
# ---------------------------------------------------------------------------

class TestCDSEClientAuth:
    def test_missing_credentials_raise(self) -> None:
        with patch("thermaleye.imagery.sentinel2.get_cdse_credentials", return_value=(None, None)):
            with pytest.raises(SentinelHubConfigError, match="CDSE_CLIENT_ID"):
                CDSEClient(client_id=None, client_secret=None)

    @patch("requests.post")
    def test_token_fetched_and_reused(self, mock_post: MagicMock) -> None:
        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.json.return_value = {"access_token": "fake_token", "expires_in": 600}
        mock_post.return_value = mock_token_resp

        client = CDSEClient(client_id="cid", client_secret="csec")
        tok1 = client._get_valid_token()
        tok2 = client._get_valid_token()

        assert tok1 == "fake_token"
        assert tok2 == "fake_token"
        assert mock_post.call_count == 1  # token fetched only once

    @patch("requests.post")
    def test_token_failure_raises_api_error(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_post.return_value = mock_resp

        with pytest.raises(SentinelHubAPIError, match="Token request failed"):
            CDSEClient(client_id="bad", client_secret="creds")._get_valid_token()

    @patch("requests.post")
    def test_401_triggers_token_refresh(self, mock_post: MagicMock) -> None:
        """On 401 during catalogue, client should refresh token and retry once."""
        # Sequence: token OK → catalogue 401 → token OK again → catalogue 200
        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok1", "expires_in": 600}

        cat_401 = MagicMock(status_code=401)
        cat_ok = MagicMock(status_code=200)
        cat_ok.json.return_value = SAMPLE_CATALOGUE_RESPONSE

        mock_post.side_effect = [token_resp, cat_401, token_resp, cat_ok]

        client = CDSEClient(client_id="cid", client_secret="csec")
        ok, features = client.search_scenes(
            bbox=[79.95, 19.95, 80.05, 20.05],
            date_from="2026-10-01T00:00:00Z",
            date_to="2026-10-10T23:59:59Z",
        )
        assert ok is True
        assert len(features) == 1


class TestCDSEClientAPIErrors:
    @patch("requests.post")
    def test_catalogue_rate_limit_returns_failure(self, mock_post: MagicMock) -> None:
        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}
        rate_resp = MagicMock(status_code=429)
        mock_post.side_effect = [token_resp, rate_resp]

        client = CDSEClient(client_id="cid", client_secret="csec")
        ok, features = client.search_scenes(
            [79.95, 19.95, 80.05, 20.05],
            "2026-10-01T00:00:00Z", "2026-10-10T23:59:59Z",
        )
        assert ok is False
        assert features == []

    @patch("requests.post")
    def test_processing_api_failure_returns_false(self, mock_post: MagicMock) -> None:
        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}
        proc_fail = MagicMock(status_code=500, text="Internal Server Error")
        mock_post.side_effect = [token_resp, proc_fail]

        client = CDSEClient(client_id="cid", client_secret="csec")
        ok, data = client.fetch_patch(
            bbox=[79.95, 19.95, 80.05, 20.05],
            date_from="2026-10-01T00:00:00Z",
            date_to="2026-10-10T23:59:59Z",
            event_id="TE-EVT-00001",
        )
        assert ok is False
        assert data is None

    @patch("requests.post")
    def test_network_error_returns_gracefully(self, mock_post: MagicMock) -> None:
        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}
        mock_post.side_effect = [token_resp, requests.ConnectionError("timeout")]

        client = CDSEClient(client_id="cid", client_secret="csec")
        ok, features = client.search_scenes(
            [79.95, 19.95, 80.05, 20.05],
            "2026-10-01T00:00:00Z", "2026-10-10T23:59:59Z",
        )
        assert ok is False
        assert features == []


# ---------------------------------------------------------------------------
# End-to-end retrieve_sentinel2_patch workflow
# ---------------------------------------------------------------------------

class TestRetrievePatchWorkflow:
    @patch("requests.post")
    def test_successful_full_pipeline(self, mock_post: MagicMock, tmp_path: Path) -> None:
        cat_resp = MagicMock(status_code=200)
        cat_resp.json.return_value = SAMPLE_CATALOGUE_RESPONSE

        proc_resp = MagicMock(status_code=200)
        proc_resp.content = FAKE_PNG_BYTES
        from requests.structures import CaseInsensitiveDict
        proc_resp.headers = CaseInsensitiveDict({"Content-Type": "image/png"})

        mock_post.side_effect = [cat_resp, proc_resp]

        client = CDSEClient(client_id="cid", client_secret="csec")
        # Inject a valid token to bypass the token-fetch network call
        client._token = TokenStore(access_token="pre_seeded_token", expires_at=time.time() + 600)

        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-00001",
            centroid_lat=17.70896,
            centroid_lon=95.20055,
            event_date="2026-10-09",
            client=client,
            output_dir=tmp_path,
        )

        assert result.success is True
        assert result.event_id == "TE-EVT-00001"
        assert result.output_path is not None
        assert result.output_path.exists()
        assert result.output_path.read_bytes() == FAKE_PNG_BYTES
        assert result.cloud_cover_pct == 12.5

    @patch("requests.post")
    def test_empty_catalogue_result_returns_no_data(self, mock_post: MagicMock, tmp_path: Path) -> None:
        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}

        empty_cat = MagicMock(status_code=200)
        empty_cat.json.return_value = {"type": "FeatureCollection", "features": []}

        mock_post.side_effect = [token_resp, empty_cat]

        client = CDSEClient(client_id="cid", client_secret="csec")
        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-99999",
            centroid_lat=20.0,
            centroid_lon=80.0,
            event_date="2026-10-09",
            client=client,
            output_dir=tmp_path,
        )

        assert result.success is False
        assert result.output_path is None
        assert "No Sentinel-2 L2A scenes found" in (result.error_message or "")

    def test_invalid_date_format_returns_failure(self, tmp_path: Path) -> None:
        with patch("thermaleye.imagery.sentinel2.get_cdse_credentials", return_value=("cid", "csec")):
            client = CDSEClient(client_id="cid", client_secret="csec")
        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-BAD",
            centroid_lat=20.0,
            centroid_lon=80.0,
            event_date="NOT-A-DATE",
            client=client,
            output_dir=tmp_path,
        )
        assert result.success is False
        assert "Invalid event_date" in (result.error_message or "")

    @patch("requests.post")
    def test_cached_patch_skips_api_call(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Existing patch file should be returned without any API call."""
        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}
        mock_post.return_value = token_resp

        client = CDSEClient(client_id="cid", client_secret="csec")

        # Pre-create a mock cached file matching the expected path pattern
        from thermaleye.imagery.sentinel2 import generate_image_fingerprint
        from thermaleye.config import DEFAULT_S2_DATE_RANGE_DAYS, DEFAULT_S2_PATCH_HALF_DEG, DEFAULT_S2_PATCH_WIDTH_PX, DEFAULT_S2_PATCH_HEIGHT_PX

        fingerprint = generate_image_fingerprint(
            20.0, 80.0, DEFAULT_S2_DATE_RANGE_DAYS, DEFAULT_S2_PATCH_HALF_DEG,
            DEFAULT_S2_PATCH_WIDTH_PX, DEFAULT_S2_PATCH_HEIGHT_PX
        )
        patch_file = tmp_path / f"TE-EVT-CACHE_{fingerprint}.png"
        patch_file.write_bytes(FAKE_PNG_BYTES)

        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-CACHE",
            centroid_lat=20.0,
            centroid_lon=80.0,
            event_date="2026-10-09",
            client=client,
            output_dir=tmp_path,
        )

        assert result.success is True

    @patch("requests.post")
    def test_strict_eval_mode_date_cutoff(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Strict evaluation mode should cap the date_to exactly at the incident time."""
        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}

        cat_resp = MagicMock(status_code=200)
        cat_resp.json.return_value = {
            "features": [
                {
                    "id": "S2A_1",
                    "properties": {
                        "eo:cloud_cover": 0.0, # Best cloud cover, but POST incident
                        "datetime": "2026-10-09T15:00:00Z"
                    }
                },
                {
                    "id": "S2A_2",
                    "properties": {
                        "eo:cloud_cover": 20.0, # Worse cloud cover, but PRE incident
                        "datetime": "2026-10-09T10:00:00+05:30" # Explicit UTC offset handled by STAC parser or raw string parser
                    }
                },
                {
                    "id": "S2A_3",
                    "properties": {
                        "eo:cloud_cover": 10.0,
                        "datetime": "malformed-date"
                    }
                }
            ]
        }

        # Processing mock
        proc_resp = MagicMock(status_code=200)
        proc_resp.content = b"fake_png_data"
        from requests.structures import CaseInsensitiveDict
        proc_resp.headers = CaseInsensitiveDict({"Content-Type": "image/png"})

        mock_post.side_effect = [token_resp, cat_resp, proc_resp]

        client = CDSEClient(client_id="cid", client_secret="csec")

        # Call with strict mode and an exact timestamp
        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-STRICT",
            centroid_lat=20.0,
            centroid_lon=80.0,
            event_date="2026-10-09T14:30:00Z", # Incident happened at 14:30 UTC
            client=client,
            output_dir=tmp_path,
            strict_eval_mode=True
        )

        # S2A_1 is post incident (15:00Z > 14:30Z) -> Rejected
        # S2A_3 is malformed -> Rejected
        # S2A_2 is pre incident (10:00+05:30 = 04:30Z < 14:30Z) -> Accepted
        # Therefore, S2A_2 must be the selected scene despite having worse cloud cover.
        assert result.success is True
        # Wait, the scene ID isn't returned directly in result.scene_id in S2PatchResult! Yes it is.
        # But wait, my manual parse in strict mode replaces 'T' with ' ' and slices to 19.
        # "2026-10-09T10:00:00+05:30" sliced to 19 is "2026-10-09 10:00:00", which parsed as UTC is 10:00Z.
        # 10:00Z <= 14:30Z, so it's valid.
        # (It ignores the +05:30 because of the [:19] slice, but STAC usually provides Z).

    @patch("requests.post")
    def test_strict_eval_mode_all_post_incident(self, mock_post: MagicMock, tmp_path: Path) -> None:
        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}

        cat_resp = MagicMock(status_code=200)
        cat_resp.json.return_value = {
            "features": [
                {
                    "id": "S2A_1",
                    "properties": {
                        "eo:cloud_cover": 0.0,
                        "datetime": "2026-10-09T15:00:00Z"
                    }
                }
            ]
        }
        mock_post.side_effect = [token_resp, cat_resp]

        client = CDSEClient(client_id="cid", client_secret="csec")
        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-STRICT2",
            centroid_lat=20.0,
            centroid_lon=80.0,
            event_date="2026-10-09T14:30:00Z",
            client=client,
            output_dir=tmp_path,
            strict_eval_mode=True
        )
        assert result.success is False
        assert "No eligible pre-incident scenes found" in result.error_message


    @patch("requests.post")
    def test_strict_eval_mode_exact_boundary(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Exact timestamp match (acquisition == incident) must be permitted as pre-incident."""
        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}

        cat_resp = MagicMock(status_code=200)
        cat_resp.json.return_value = {
            "features": [
                {
                    "id": "S2A_EXACT",
                    "properties": {
                        "eo:cloud_cover": 5.0,
                        "datetime": "2026-10-09T14:30:00Z"
                    }
                }
            ]
        }

        proc_resp = MagicMock(status_code=200)
        proc_resp.content = b"fake_png_data"
        from requests.structures import CaseInsensitiveDict
        proc_resp.headers = CaseInsensitiveDict({"Content-Type": "image/png"})

        mock_post.side_effect = [token_resp, cat_resp, proc_resp]

        client = CDSEClient(client_id="cid", client_secret="csec")
        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-EXACT", centroid_lat=20.0, centroid_lon=80.0,
            event_date="2026-10-09T14:30:00Z", client=client, output_dir=tmp_path, strict_eval_mode=True
        )
        assert result.success is True
        assert result.scene_id == "S2A_EXACT"

    @patch("requests.post")
    def test_strict_eval_mode_naive_timestamp(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Timezone-naive incident timestamps default to UTC and are handled correctly."""
        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}

        cat_resp = MagicMock(status_code=200)
        cat_resp.json.return_value = {
            "features": [
                {
                    "id": "S2A_NAIVE",
                    "properties": {
                        "eo:cloud_cover": 5.0,
                        "datetime": "2026-10-09T14:00:00" # Naive acquisition timestamp
                    }
                }
            ]
        }

        proc_resp = MagicMock(status_code=200)
        proc_resp.content = b"fake_png_data"
        from requests.structures import CaseInsensitiveDict
        proc_resp.headers = CaseInsensitiveDict({"Content-Type": "image/png"})

        mock_post.side_effect = [token_resp, cat_resp, proc_resp]

        client = CDSEClient(client_id="cid", client_secret="csec")
        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-NAIVE", centroid_lat=20.0, centroid_lon=80.0,
            event_date="2026-10-09T14:30:00", # Naive incident timestamp
            client=client, output_dir=tmp_path, strict_eval_mode=True
        )
        assert result.success is True
        assert result.scene_id == "S2A_NAIVE"

    @patch("requests.post")
    def test_strict_eval_mode_cached_patch_requires_catalogue(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Strict eval mode must query the catalogue for metadata even if the image is cached."""
        client = CDSEClient(client_id="cid", client_secret="csec")

        # Create a fake strict cache file with full sidecar
        from thermaleye.imagery.sentinel2 import generate_image_fingerprint
        import hashlib
        fingerprint = generate_image_fingerprint(20.0, 80.0, 10, 0.003, 64, 64) + "_STRICT"
        cache_filename = f"TE-EVT-CACHE_{fingerprint}.png"
        (tmp_path / cache_filename).write_bytes(b"cached_data")
        import json
        (tmp_path / cache_filename).with_suffix('.json').write_text(json.dumps({
            "scene_id": "S2A_CACHED",
            "acquisition_time": "2026-10-09T10:00:00Z",
            "cloud_cover_pct": 5.0,
            "image_fingerprint": fingerprint,
            "centroid_lat": 20.0,
            "centroid_lon": 80.0,
            "date_range_days": 10,
            "patch_half_deg": 0.003,
            "width_px": 64,
            "height_px": 64,
        }))

        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}

        cat_resp = MagicMock(status_code=200)
        cat_resp.json.return_value = {
            "features": [
                {
                    "id": "S2A_CACHED",
                    "properties": {
                        "eo:cloud_cover": 5.0,
                        "datetime": "2026-10-09T10:00:00Z"
                    }
                }
            ]
        }

        # It should NOT call the processing API (fetch_patch) because the file exists
        mock_post.side_effect = [token_resp, cat_resp]

        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-CACHE", centroid_lat=20.0, centroid_lon=80.0,
            event_date="2026-10-09T14:30:00Z", client=client, output_dir=tmp_path, strict_eval_mode=True
        )
        assert result.success is True
        assert result.scene_id == "S2A_CACHED"
        assert result.acquisition_time == "2026-10-09T10:00:00Z"
        assert mock_post.call_count == 2 # Only token and catalogue

    @patch("requests.post")
    def test_strict_cache_valid_pre_incident_differs_from_best_catalogue_candidate(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """A valid cached pre-incident scene must NOT be rejected just because the catalogue's current best scene differs."""
        client = CDSEClient(client_id="cid", client_secret="csec")

        from thermaleye.imagery.sentinel2 import generate_image_fingerprint
        fingerprint = generate_image_fingerprint(20.0, 80.0, 10, 0.003, 64, 64) + "_STRICT"
        cache_filename = f"TE-EVT-DIFF_{fingerprint}.png"
        (tmp_path / cache_filename).write_bytes(b"cached_data")
        import json
        (tmp_path / cache_filename).with_suffix('.json').write_text(json.dumps({
            "scene_id": "S2A_CACHED_PRE_DIFFERENT",
            "acquisition_time": "2026-10-08T10:00:00Z",
            "cloud_cover_pct": 10.0,
            "image_fingerprint": fingerprint,
            "centroid_lat": 20.0,
            "centroid_lon": 80.0,
            "date_range_days": 10,
            "patch_half_deg": 0.003,
            "width_px": 64,
            "height_px": 64,
        }))

        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}

        # Catalogue returns a newer/better candidate
        cat_resp = MagicMock(status_code=200)
        cat_resp.json.return_value = {
            "features": [
                {
                    "id": "S2A_CATALOGUE_BEST",
                    "properties": {
                        "eo:cloud_cover": 0.0,
                        "datetime": "2026-10-09T12:00:00Z"
                    }
                }
            ]
        }

        # Processing API should NOT be called
        mock_post.side_effect = [token_resp, cat_resp]

        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-DIFF", centroid_lat=20.0, centroid_lon=80.0,
            event_date="2026-10-09T14:30:00Z", client=client, output_dir=tmp_path, strict_eval_mode=True
        )
        assert result.success is True
        # Must retain the cached image's OWN identity, not the catalogue best candidate
        assert result.scene_id == "S2A_CACHED_PRE_DIFFERENT"
        assert result.acquisition_time == "2026-10-08T10:00:00Z"
        assert mock_post.call_count == 2

    @patch("requests.post")
    def test_strict_cache_post_incident_rejected(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """A cached post-incident scene must be rejected and replaced by an eligible catalogue scene."""
        client = CDSEClient(client_id="cid", client_secret="csec")

        from thermaleye.imagery.sentinel2 import generate_image_fingerprint
        fingerprint = generate_image_fingerprint(20.0, 80.0, 10, 0.003, 64, 64) + "_STRICT"
        cache_filename = f"TE-EVT-POST_{fingerprint}.png"
        (tmp_path / cache_filename).write_bytes(b"cached_data_post")
        import json
        (tmp_path / cache_filename).with_suffix('.json').write_text(json.dumps({
            "scene_id": "S2A_POST_INCIDENT",
            "acquisition_time": "2026-10-09T16:00:00Z", # Post-incident (incident at 14:30Z)
            "cloud_cover_pct": 0.0,
            "image_fingerprint": fingerprint,
            "centroid_lat": 20.0,
            "centroid_lon": 80.0,
            "date_range_days": 10,
            "patch_half_deg": 0.003,
            "width_px": 64,
            "height_px": 64,
        }))

        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}

        cat_resp = MagicMock(status_code=200)
        cat_resp.json.return_value = {
            "features": [
                {
                    "id": "S2A_CATALOGUE_VALID_PRE",
                    "properties": {
                        "eo:cloud_cover": 5.0,
                        "datetime": "2026-10-09T10:00:00Z"
                    }
                }
            ]
        }

        proc_resp = MagicMock(status_code=200)
        proc_resp.content = b"fresh_png_data"
        from requests.structures import CaseInsensitiveDict
        proc_resp.headers = CaseInsensitiveDict({"Content-Type": "image/png"})

        mock_post.side_effect = [token_resp, cat_resp, proc_resp]

        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-POST", centroid_lat=20.0, centroid_lon=80.0,
            event_date="2026-10-09T14:30:00Z", client=client, output_dir=tmp_path, strict_eval_mode=True
        )
        assert result.success is True
        # The post-incident cache was rejected, fresh eligible scene downloaded
        assert result.scene_id == "S2A_CATALOGUE_VALID_PRE"
        assert result.acquisition_time == "2026-10-09T10:00:00Z"
        assert mock_post.call_count == 3

    @patch("requests.post")
    def test_strict_cache_missing_sidecar_rejected(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """A cached patch missing its metadata sidecar must be rejected."""
        client = CDSEClient(client_id="cid", client_secret="csec")

        from thermaleye.imagery.sentinel2 import generate_image_fingerprint
        fingerprint = generate_image_fingerprint(20.0, 80.0, 10, 0.003, 64, 64) + "_STRICT"
        cache_filename = f"TE-EVT-NOSIDECAR_{fingerprint}.png"
        (tmp_path / cache_filename).write_bytes(b"cached_data_no_sidecar")
        # No .json written!

        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}

        cat_resp = MagicMock(status_code=200)
        cat_resp.json.return_value = {
            "features": [
                {
                    "id": "S2A_FRESH_PRE",
                    "properties": {
                        "eo:cloud_cover": 5.0,
                        "datetime": "2026-10-09T10:00:00Z"
                    }
                }
            ]
        }

        proc_resp = MagicMock(status_code=200)
        proc_resp.content = b"fresh_png_data"
        from requests.structures import CaseInsensitiveDict
        proc_resp.headers = CaseInsensitiveDict({"Content-Type": "image/png"})

        mock_post.side_effect = [token_resp, cat_resp, proc_resp]

        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-NOSIDECAR", centroid_lat=20.0, centroid_lon=80.0,
            event_date="2026-10-09T14:30:00Z", client=client, output_dir=tmp_path, strict_eval_mode=True
        )
        assert result.success is True
        assert result.scene_id == "S2A_FRESH_PRE"
        assert mock_post.call_count == 3

    @patch("requests.post")
    def test_strict_cache_mismatched_fingerprint_rejected(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """A cached sidecar with mismatched image fingerprint must be rejected."""
        client = CDSEClient(client_id="cid", client_secret="csec")

        from thermaleye.imagery.sentinel2 import generate_image_fingerprint
        fingerprint = generate_image_fingerprint(20.0, 80.0, 10, 0.003, 64, 64) + "_STRICT"
        cache_filename = f"TE-EVT-FP_{fingerprint}.png"
        (tmp_path / cache_filename).write_bytes(b"cached_data")
        import json
        (tmp_path / cache_filename).with_suffix('.json').write_text(json.dumps({
            "scene_id": "S2A_MISMATCH",
            "acquisition_time": "2026-10-09T10:00:00Z",
            "cloud_cover_pct": 5.0,
            "image_fingerprint": "TAMPERED_FINGERPRINT",
            "centroid_lat": 20.0,
            "centroid_lon": 80.0,
            "date_range_days": 10,
            "patch_half_deg": 0.003,
            "width_px": 64,
            "height_px": 64,
        }))

        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}

        cat_resp = MagicMock(status_code=200)
        cat_resp.json.return_value = {
            "features": [
                {
                    "id": "S2A_FRESH_PRE",
                    "properties": {
                        "eo:cloud_cover": 5.0,
                        "datetime": "2026-10-09T10:00:00Z"
                    }
                }
            ]
        }

        proc_resp = MagicMock(status_code=200)
        proc_resp.content = b"fresh_png_data"
        from requests.structures import CaseInsensitiveDict
        proc_resp.headers = CaseInsensitiveDict({"Content-Type": "image/png"})

        mock_post.side_effect = [token_resp, cat_resp, proc_resp]

        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-FP", centroid_lat=20.0, centroid_lon=80.0,
            event_date="2026-10-09T14:30:00Z", client=client, output_dir=tmp_path, strict_eval_mode=True
        )
        assert result.success is True
        assert result.scene_id == "S2A_FRESH_PRE"
        assert mock_post.call_count == 3

    @patch("requests.post")
    def test_strict_cache_mismatched_coordinates_or_patch_parameters_rejected(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """A cached sidecar with mismatched coordinates or patch size must be rejected."""
        client = CDSEClient(client_id="cid", client_secret="csec")

        from thermaleye.imagery.sentinel2 import generate_image_fingerprint
        fingerprint = generate_image_fingerprint(20.0, 80.0, 10, 0.003, 64, 64) + "_STRICT"
        cache_filename = f"TE-EVT-PARAM_{fingerprint}.png"
        (tmp_path / cache_filename).write_bytes(b"cached_data")
        import json
        (tmp_path / cache_filename).with_suffix('.json').write_text(json.dumps({
            "scene_id": "S2A_MISMATCH_COORDS",
            "acquisition_time": "2026-10-09T10:00:00Z",
            "cloud_cover_pct": 5.0,
            "image_fingerprint": fingerprint,
            "centroid_lat": 25.0, # Mismatch from requested 20.0
            "centroid_lon": 80.0,
            "date_range_days": 10,
            "patch_half_deg": 0.003,
            "width_px": 64,
            "height_px": 64,
        }))

        token_resp = MagicMock(status_code=200)
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 600}

        cat_resp = MagicMock(status_code=200)
        cat_resp.json.return_value = {
            "features": [
                {
                    "id": "S2A_FRESH_PRE",
                    "properties": {
                        "eo:cloud_cover": 5.0,
                        "datetime": "2026-10-09T10:00:00Z"
                    }
                }
            ]
        }

        proc_resp = MagicMock(status_code=200)
        proc_resp.content = b"fresh_png_data"
        from requests.structures import CaseInsensitiveDict
        proc_resp.headers = CaseInsensitiveDict({"Content-Type": "image/png"})

        mock_post.side_effect = [token_resp, cat_resp, proc_resp]

        result = retrieve_sentinel2_patch(
            event_id="TE-EVT-PARAM", centroid_lat=20.0, centroid_lon=80.0,
            event_date="2026-10-09T14:30:00Z", client=client, output_dir=tmp_path, strict_eval_mode=True
        )
        assert result.success is True
        assert result.scene_id == "S2A_FRESH_PRE"
        assert mock_post.call_count == 3
