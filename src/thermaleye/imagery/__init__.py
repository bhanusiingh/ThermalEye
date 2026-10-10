"""Satellite imagery retrieval package for ThermalEye."""

from thermaleye.imagery.sentinel2 import (
    CDSEClient,
    SentinelHubConfigError,
    SentinelHubAPIError,
    TokenStore,
    S2SceneRecord,
    S2PatchResult,
    compute_patch_bbox,
    build_catalogue_search_payload,
    build_process_request_body,
    select_best_scene,
    retrieve_sentinel2_patch,
)

__all__ = [
    "CDSEClient",
    "SentinelHubConfigError",
    "SentinelHubAPIError",
    "TokenStore",
    "S2SceneRecord",
    "S2PatchResult",
    "compute_patch_bbox",
    "build_catalogue_search_payload",
    "build_process_request_body",
    "select_best_scene",
    "retrieve_sentinel2_patch",
]
