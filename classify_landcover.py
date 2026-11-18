"""
ThermalEye — Phase 4: ML Land-Cover Classification Worker
============================================================

Runs a lightweight pre-trained vision model (MobileNetV2 via
Hugging Face transformers) on RGB Sentinel-2 image chips to
produce basic land-cover context labels for candidate events.

Architectural invariants:
  • This model provides CONTEXT ONLY — the mandatory disclaimer
    is always appended to every database record.
  • Events without a valid RGB image → land_cover_class stays NULL.
  • Missing data is never mocked or defaulted to zero.
  • Credentials loaded from .env via python-dotenv.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from sqlalchemy import create_engine, text

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


DATABASE_URL: str = get_database_url()

# The mandatory ML disclaimer — appended verbatim to every classification
ML_DISCLAIMER = (
    "Model provides land-cover context only, not facility identification "
    "or fire confirmation."
)

# Pre-trained model identifier (Hugging Face Hub)
MODEL_NAME: str = os.environ.get(
    "LANDCOVER_MODEL",
    "google/mobilenet_v2_1.0_224",
)

# Minimum confidence threshold — below this we still record the label
# but flag it as low-confidence
MIN_CONFIDENCE: float = float(os.environ.get("MIN_LANDCOVER_CONFIDENCE", "0.0"))


# ---------------------------------------------------------------------------
# Land-cover label mapping
# ---------------------------------------------------------------------------
# ImageNet labels are object-centric. We map common top predictions to
# coarse land-cover categories that are meaningful for fire intelligence.
# Unmapped labels pass through with a "raw:" prefix.
LANDCOVER_MAP: dict[str, str] = {
    # Vegetation / Forest
    "valley": "vegetation/valley",
    "alp": "vegetation/alpine",
    "cliff": "terrain/cliff",
    "volcano": "terrain/volcanic",
    "lakeside": "water/lakeside",
    "seashore": "water/coastal",
    "sandbar": "water/sandbar",
    "promontory": "terrain/promontory",
    "coral reef": "water/reef",
    # Urban / Industrial
    "greenhouse": "agriculture/greenhouse",
    "barn": "agriculture/barn",
    "palace": "urban/structure",
    "church": "urban/structure",
    "monastery": "urban/structure",
    "dam": "infrastructure/dam",
    "breakwater": "infrastructure/coastal",
    "pier": "infrastructure/port",
    "dock": "infrastructure/port",
    "mobile home": "urban/residential",
    "steel arch bridge": "infrastructure/bridge",
    "suspension bridge": "infrastructure/bridge",
    "viaduct": "infrastructure/bridge",
    "streetcar": "urban/transport",
    "freight car": "infrastructure/rail",
    "container ship": "infrastructure/shipping",
    "aircraft carrier": "infrastructure/military",
    "missile": "infrastructure/military",
    "military uniform": "infrastructure/military",
    "refinery": "industrial/refinery",
    "power line": "infrastructure/power",
    "solar dish": "infrastructure/solar",
    "harvester": "agriculture/machinery",
    "thresher": "agriculture/machinery",
    "plow": "agriculture/machinery",
    "tractor": "agriculture/machinery",
    # Natural terrain
    "mountain": "terrain/mountain",
    "ridge": "terrain/ridge",
    "geyser": "terrain/geothermal",
    "hot spring": "terrain/geothermal",
}


def map_to_landcover(imagenet_label: str) -> str:
    """Map an ImageNet label to a coarse land-cover category."""
    label_lower = imagenet_label.lower().strip()
    for key, category in LANDCOVER_MAP.items():
        if key in label_lower:
            return category
    # Pass through with prefix so the UI knows it's a raw model label
    return f"raw:{imagenet_label.lower().strip()}"


# ---------------------------------------------------------------------------
# ML Pipeline — lazy-loaded singleton
# ---------------------------------------------------------------------------
_classifier = None


def get_classifier():
    """Lazy-load the Hugging Face image classification pipeline."""
    global _classifier
    if _classifier is not None:
        return _classifier

    logger.info("Loading ML model: %s …", MODEL_NAME)
    try:
        from transformers import pipeline as hf_pipeline

        _classifier = hf_pipeline(
            "image-classification",
            model=MODEL_NAME,
            device=-1,  # CPU — keep GPU-free for other workloads
        )
        logger.info("Model loaded successfully.")
    except Exception:
        logger.exception("Failed to load ML model %s.", MODEL_NAME)
        raise

    return _classifier


# ---------------------------------------------------------------------------
# Classify a single image
# ---------------------------------------------------------------------------
def classify_image(image_path: Path) -> tuple[str | None, float | None]:
    """
    Run image classification on a single RGB image chip.

    Returns:
        (label, confidence) or (None, None) on failure.
    """
    if not image_path.exists():
        logger.warning("Image not found: %s", image_path)
        return None, None

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        logger.exception("Failed to open image: %s", image_path)
        return None, None

    try:
        classifier = get_classifier()
        results = classifier(img, top_k=3)
    except Exception:
        logger.exception("ML inference failed for: %s", image_path)
        return None, None

    if not results:
        logger.warning("Model returned no predictions for: %s", image_path)
        return None, None

    top = results[0]
    raw_label = top.get("label", "unknown")
    confidence = float(top.get("score", 0.0))

    # Map to land-cover category
    land_cover = map_to_landcover(raw_label)

    logger.info(
        "Classification: %s → %s (confidence: %.3f) | raw: %s",
        image_path.name,
        land_cover,
        confidence,
        raw_label,
    )

    # Log runner-up predictions for context
    for r in results[1:]:
        logger.debug(
            "  Runner-up: %s (%.3f)",
            r.get("label", "?"),
            r.get("score", 0),
        )

    return land_cover, confidence


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------
def get_unclassified_events(engine) -> list[dict]:
    """
    Fetch candidate events that have an RGB image but no land-cover
    classification yet.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    id::text       AS event_id,
                    imagery_rgb_path
                FROM candidate_events
                WHERE imagery_rgb_path IS NOT NULL
                  AND land_cover_class IS NULL
                ORDER BY priority_score DESC NULLS LAST
                """
            )
        ).fetchall()

    return [
        {"event_id": r[0], "imagery_rgb_path": r[1]}
        for r in rows
    ]


def update_classification(
    engine,
    event_id: str,
    land_cover_class: str | None,
    land_cover_confidence: float | None,
) -> None:
    """
    Persist the ML classification result to the database.
    Always writes the ML_DISCLAIMER regardless of classification outcome.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE candidate_events
                SET land_cover_class       = :lc_class,
                    land_cover_confidence   = :lc_confidence,
                    ml_disclaimer           = :disclaimer,
                    updated_at              = NOW()
                WHERE id = CAST(:event_id AS uuid)
                """
            ),
            {
                "event_id": event_id,
                "lc_class": land_cover_class,
                "lc_confidence": land_cover_confidence,
                "disclaimer": ML_DISCLAIMER,
            },
        )
    logger.info(
        "Updated event %s: class=%s, confidence=%s",
        event_id[:8],
        land_cover_class or "NULL",
        f"{land_cover_confidence:.3f}" if land_cover_confidence is not None else "NULL",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_classification() -> dict:
    """
    Main entry point. Finds unclassified events with RGB imagery
    and runs the ML pipeline on each.

    Returns a summary dict.
    """
    logger.info("=" * 60)
    logger.info("ThermalEye — ML land-cover classification starting")
    logger.info("=" * 60)

    engine = create_engine(DATABASE_URL, echo=False)

    events = get_unclassified_events(engine)
    logger.info("Found %d events needing classification.", len(events))

    if not events:
        return {"processed": 0, "classified": 0, "failed": 0}

    # Pre-load the model once before looping
    try:
        get_classifier()
    except Exception:
        logger.critical("Cannot load ML model. Aborting classification run.")
        return {"processed": 0, "classified": 0, "failed": len(events)}

    stats = {"processed": 0, "classified": 0, "failed": 0}

    for event in events:
        event_id = event["event_id"]
        rgb_rel_path = event["imagery_rgb_path"]

        # Resolve relative path from project root
        rgb_abs_path = Path(_script_dir) / rgb_rel_path

        label, confidence = classify_image(rgb_abs_path)

        update_classification(engine, event_id, label, confidence)

        if label is not None:
            stats["classified"] += 1
        else:
            stats["failed"] += 1

        stats["processed"] += 1

    logger.info("Classification complete. Summary: %s", stats)
    logger.info("ML Disclaimer: %s", ML_DISCLAIMER)
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_classification()
