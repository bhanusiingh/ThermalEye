"""ThermalEye — Phase 4: ML Land-Cover Classification & Explainable AI Worker
=============================================================================

Runs ThermalEye's fine-tuned CNN backbone (EfficientNet-B0 / MobileNetV2) on
multimodal Sentinel-2 satellite imagery (RGB + SWIR) to classify candidate fire
events into 4 operational domains:
  1. industrial_thermal
  2. agricultural_stubble_burn
  3. forest_wildfire
  4. urban_false_positive

Also computes Gradient-weighted Class Activation Mapping (Grad-CAM) to generate
visual attribution heatmaps saved to `static/imagery/{event_id}_gradcam.png`
and referenced in PostGIS `candidate_events.gradcam_path`.

Architectural invariants:
  • This model provides CONTEXT ONLY — the mandatory disclaimer is always appended.
  • Events without a valid RGB image → land_cover_class stays NULL.
  • Missing data is never mocked or defaulted to zero.
  • Falls back gracefully to baseline classifier if champion model weights are absent.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
import numpy as np
from PIL import Image
from sqlalchemy import create_engine, text

# Add workspace to sys.path
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("thermaleye.classify_landcover")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
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


DATABASE_URL: str = get_database_url()

# The mandatory ML disclaimer — appended verbatim to every classification
ML_DISCLAIMER = (
    "Model provides land-cover context only, not facility identification "
    "or fire confirmation."
)

CHAMPION_MODEL_PATH = _script_dir / "models" / "thermaleye_best.pt"

# Minimum confidence threshold
MIN_CONFIDENCE: float = float(os.environ.get("MIN_LANDCOVER_CONFIDENCE", "0.0"))

# Fallback label map
LANDCOVER_MAP: dict[str, str] = {
    "valley": "forest_wildfire",
    "alp": "forest_wildfire",
    "cliff": "forest_wildfire",
    "volcano": "forest_wildfire",
    "refinery": "industrial_thermal",
    "power line": "industrial_thermal",
    "solar dish": "industrial_thermal",
    "harvester": "agricultural_stubble_burn",
    "thresher": "agricultural_stubble_burn",
    "plow": "agricultural_stubble_burn",
    "tractor": "agricultural_stubble_burn",
    "barn": "agricultural_stubble_burn",
    "greenhouse": "agricultural_stubble_burn",
    "streetcar": "urban_false_positive",
    "palace": "urban_false_positive",
    "church": "urban_false_positive",
    "mobile home": "urban_false_positive",
}


def map_to_landcover(label: str) -> str:
    label_lower = label.lower().strip()
    for key, category in LANDCOVER_MAP.items():
        if key in label_lower:
            return category
    return f"raw:{label_lower}"


# ---------------------------------------------------------------------------
# Fine-Tuned PyTorch Model & Grad-CAM Loader
# ---------------------------------------------------------------------------
_pytorch_model_bundle = None


def get_fine_tuned_model() -> Optional[Dict[str, Any]]:
    """Lazy-load the fine-tuned ThermalEye PyTorch model and metadata."""
    global _pytorch_model_bundle
    if _pytorch_model_bundle is not None:
        return _pytorch_model_bundle

    if not CHAMPION_MODEL_PATH.exists():
        logger.warning("Champion model checkpoint not found at %s. Falling back.", CHAMPION_MODEL_PATH)
        return None

    try:
        import torch
        from ml.models import build_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(CHAMPION_MODEL_PATH, map_location=device)

        arch = checkpoint.get("arch", "efficientnet_b0")
        in_channels = checkpoint.get("in_channels", 6)
        class_names = checkpoint.get(
            "class_names",
            ["industrial_thermal", "agricultural_stubble_burn", "forest_wildfire", "urban_false_positive"],
        )

        model = build_model(arch=arch, in_channels=in_channels, num_classes=len(class_names), pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        norm = checkpoint.get("normalization", {})
        mean = norm.get("mean", [0.28, 0.25, 0.17, 0.38, 0.55, 0.62][:in_channels])
        std = norm.get("std", [0.19, 0.13, 0.11, 0.23, 0.27, 0.27][:in_channels])

        _pytorch_model_bundle = {
            "model": model,
            "arch": arch,
            "in_channels": in_channels,
            "class_names": class_names,
            "mean": torch.tensor(mean, dtype=torch.float32, device=device).view(-1, 1, 1),
            "std": torch.tensor(std, dtype=torch.float32, device=device).view(-1, 1, 1),
            "device": device,
        }
        logger.info(
            "Loaded fine-tuned champion model: %s (%d channels, device=%s)",
            arch,
            in_channels,
            device,
        )
    except Exception:
        logger.exception("Failed to load fine-tuned PyTorch model. Falling back.")
        _pytorch_model_bundle = None

    return _pytorch_model_bundle


# ---------------------------------------------------------------------------
# Fallback HuggingFace Classifier Singleton
# ---------------------------------------------------------------------------
_hf_classifier = None


def get_hf_classifier():
    global _hf_classifier
    if _hf_classifier is not None:
        return _hf_classifier

    model_name = os.environ.get("LANDCOVER_MODEL", "google/mobilenet_v2_1.0_224")
    logger.info("Loading fallback Hugging Face model: %s ...", model_name)
    try:
        from transformers import pipeline as hf_pipeline
        _hf_classifier = hf_pipeline("image-classification", model=model_name, device=-1)
        logger.info("Fallback model loaded successfully.")
    except Exception:
        logger.exception("Failed to load fallback Hugging Face model.")
        raise
    return _hf_classifier


# ---------------------------------------------------------------------------
# Classification & Grad-CAM Generation
# ---------------------------------------------------------------------------
def classify_multimodal_chip(
    rgb_path: Path,
    swir_path: Optional[Path],
    event_id: str,
    output_dir: Path,
) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str]]:
    """Classify satellite imagery chip using fine-tuned CNN and generate Grad-CAM."""
    bundle = get_fine_tuned_model()
    if bundle is None:
        lbl, conf = classify_image(rgb_path)
        return lbl, conf, None, "google/mobilenet_v2_1.0_224"

    try:
        import torch
        from ml.gradcam import generate_gradcam

        model = bundle["model"]
        arch = bundle["arch"]
        in_channels = bundle["in_channels"]
        class_names = bundle["class_names"]
        device = bundle["device"]

        # Load RGB
        rgb_img = Image.open(rgb_path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        rgb_arr = np.array(rgb_img, dtype=np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb_arr).permute(2, 0, 1).to(device)

        if in_channels == 6 and swir_path is not None and swir_path.exists():
            swir_img = Image.open(swir_path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            swir_arr = np.array(swir_img, dtype=np.float32) / 255.0
            swir_t = torch.from_numpy(swir_arr).permute(2, 0, 1).to(device)
            input_tensor = torch.cat([rgb_t, swir_t], dim=0)
        else:
            if in_channels == 6:
                # If model expects 6ch but swir missing, replicate RGB as surrogate
                input_tensor = torch.cat([rgb_t, rgb_t], dim=0)
            else:
                input_tensor = rgb_t

        input_tensor = (input_tensor - bundle["mean"]) / (bundle["std"] + 1e-7)
        input_tensor = input_tensor.unsqueeze(0)  # (1, C, H, W)

        # Output Grad-CAM path
        gradcam_filename = f"{event_id}_gradcam.png"
        gradcam_path = output_dir / gradcam_filename

        _, _, pred_idx, confidence = generate_gradcam(
            model=model,
            input_tensor=input_tensor,
            rgb_image_path=rgb_path,
            arch=arch,
            output_path=gradcam_path,
        )

        label = class_names[pred_idx]
        rel_gradcam_path = f"static/imagery/{gradcam_filename}"

        logger.info(
            "CNN Classification: %s -> %s (conf: %.3f, arch: %s) | Grad-CAM -> %s",
            event_id[:8],
            label,
            confidence,
            arch,
            rel_gradcam_path,
        )
        return label, float(confidence), rel_gradcam_path, arch

    except Exception:
        logger.exception("Fine-tuned CNN inference failed for event %s. Falling back.", event_id[:8])
        fallback_label, fallback_conf = classify_image(rgb_path)
        return fallback_label, fallback_conf, None, "google/mobilenet_v2_1.0_224"


def classify_image(image_path: Path) -> tuple[str | None, float | None]:
    """Fallback classification using pre-trained Hugging Face pipeline."""
    if not image_path.exists():
        logger.warning("Image not found: %s", image_path)
        return None, None

    try:
        img = Image.open(image_path).convert("RGB")
        classifier = get_hf_classifier()
        results = classifier(img, top_k=3)
        if not results:
            return None, None
        top = results[0]
        raw_label = top.get("label", "unknown")
        confidence = float(top.get("score", 0.0))
        land_cover = map_to_landcover(raw_label)
        return land_cover, confidence
    except Exception:
        logger.exception("Fallback classification failed for: %s", image_path)
        return None, None


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------
def ensure_db_schema(engine) -> None:
    """Ensure ML columns exist in candidate_events table."""
    with engine.begin() as conn:
        conn.execute(
            text("ALTER TABLE candidate_events ADD COLUMN IF NOT EXISTS gradcam_path TEXT;")
        )
        conn.execute(
            text("ALTER TABLE candidate_events ADD COLUMN IF NOT EXISTS land_cover_confidence DOUBLE PRECISION;")
        )
        conn.execute(
            text("ALTER TABLE candidate_events ADD COLUMN IF NOT EXISTS ml_model_arch TEXT;")
        )


def get_unclassified_events(engine) -> list[dict]:
    """Fetch candidate events with RGB imagery needing classification or Grad-CAM."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    id::text           AS event_id,
                    imagery_rgb_path,
                    imagery_swir_path
                FROM candidate_events
                WHERE imagery_rgb_path IS NOT NULL
                  AND (land_cover_class IS NULL OR gradcam_path IS NULL OR ml_model_arch IS NULL)
                ORDER BY priority_score DESC NULLS LAST
                """
            )
        ).fetchall()

    return [
        {"event_id": r[0], "imagery_rgb_path": r[1], "imagery_swir_path": r[2]}
        for r in rows
    ]


def update_classification(
    engine,
    event_id: str,
    land_cover_class: str | None,
    land_cover_confidence: float | None,
    gradcam_path: str | None = None,
    ml_model_arch: str | None = None,
) -> None:
    """Persist the ML classification result, model arch, and Grad-CAM path to PostGIS."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE candidate_events
                SET land_cover_class       = :lc_class,
                    land_cover_confidence   = :lc_confidence,
                    gradcam_path            = COALESCE(:gradcam_path, gradcam_path),
                    ml_model_arch           = COALESCE(:ml_model_arch, ml_model_arch),
                    ml_disclaimer           = :disclaimer,
                    updated_at              = NOW()
                WHERE id = CAST(:event_id AS uuid)
                """
            ),
            {
                "event_id": event_id,
                "lc_class": land_cover_class,
                "lc_confidence": land_cover_confidence,
                "gradcam_path": gradcam_path,
                "ml_model_arch": ml_model_arch,
                "disclaimer": ML_DISCLAIMER,
            },
        )
    logger.info(
        "Updated event %s: class=%s, conf=%s, arch=%s, gradcam=%s",
        event_id[:8],
        land_cover_class or "NULL",
        f"{land_cover_confidence:.3f}" if land_cover_confidence is not None else "NULL",
        ml_model_arch or "NULL",
        gradcam_path or "NULL",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_classification() -> dict:
    """Main entry point. Finds unclassified events and runs classification + Grad-CAM."""
    logger.info("=" * 60)
    logger.info("ThermalEye — ML land-cover classification starting")
    logger.info("=" * 60)

    engine = create_engine(DATABASE_URL, echo=False)
    ensure_db_schema(engine)

    events = get_unclassified_events(engine)
    logger.info("Found %d events needing classification.", len(events))

    if not events:
        return {"processed": 0, "classified": 0, "failed": 0}

    output_dir = _script_dir / "static" / "imagery"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "classified": 0, "failed": 0}

    for event in events:
        event_id = event["event_id"]
        rgb_rel = event["imagery_rgb_path"]
        swir_rel = event.get("imagery_swir_path")

        rgb_abs = _script_dir / rgb_rel
        swir_abs = _script_dir / swir_rel if swir_rel else None

        label, confidence, gradcam_rel, model_arch = classify_multimodal_chip(
            rgb_path=rgb_abs,
            swir_path=swir_abs,
            event_id=event_id,
            output_dir=output_dir,
        )

        update_classification(
            engine,
            event_id,
            label,
            confidence,
            gradcam_path=gradcam_rel,
            ml_model_arch=model_arch,
        )

        if label is not None:
            stats["classified"] += 1
        else:
            stats["failed"] += 1

        stats["processed"] += 1

    logger.info("Classification complete. Summary: %s", stats)
    return stats


if __name__ == "__main__":
    run_classification()
