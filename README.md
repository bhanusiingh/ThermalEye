# ThermalEye

### Geospatial Intelligence & Explainable Deep Learning Platform for Satellite Thermal Anomaly Analysis Across India

[![Live Demo](https://img.shields.io/badge/Live%20Demo-thermaleye.onrender.com-b86f1c?style=flat-square)](https://thermaleye.onrender.com/)

[![3D Dashboard](https://img.shields.io/badge/3D%20Dashboard-%2Fdashboard-2b6e72?style=flat-square)](https://thermaleye.onrender.com/dashboard)

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

[![PyTorch](https://img.shields.io/badge/PyTorch-6--Ch%20CNN%20%2B%20Grad--CAM-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)

[![PostGIS](https://img.shields.io/badge/PostGIS-142%2C090%20OSM%20Features-336791?style=flat-square&logo=postgresql&logoColor=white)](https://postgis.net/)

[![Tests](https://img.shields.io/badge/Tests-66%20Passing-2ea44f?style=flat-square)](#running-automated-tests)

ThermalEye is an end-to-end geospatial intelligence and explainable computer vision platform built for the sovereign territory of India. It ingests NASA FIRMS VIIRS 375 m active fire detections, groups proximate hotspots via ST-DBSCAN, resolves industrial infrastructure locally against a 142,090-feature PostGIS OpenStreetMap index, retrieves 10 m Copernicus Sentinel-2 RGB + SWIR imagery, and classifies terrain context with 6-channel PyTorch CNNs and Grad-CAM saliency attribution.

**Live Application:** [https://thermaleye.onrender.com/](https://thermaleye.onrender.com/)  
**3D Anomaly Dashboard:** [https://thermaleye.onrender.com/dashboard](https://thermaleye.onrender.com/dashboard)  
*(Hosted on Render Free Tier; initial cold start after inactivity takes ~50 seconds.)*

---

## System Architecture

```text
External & Local Sources          Worker & Deep Learning Pipeline           Storage & API                        Web Client & Hosting
────────────────────────          ───────────────────────────────           ─────────────                        ────────────────────
NASA FIRMS VIIRS (375 m)    ───▶  ST-DBSCAN Spatiotemporal Clustering  ───▶ PostgreSQL / PostGIS            ───▶ Render Cloud (Dockerfile.render)
14,595 thermal detections         7,517 candidate events (48h window)       Neon Cloud (PostGIS 3.6)             or Local Nginx (30 MB upload cap)
                                                                            Local Docker (PostGIS 3.4)
Geofabrik India OSM PBF     ───▶  Local PostGIS Spatial Enrichment     ───▶ osm_industrial_features         ───▶ Landing Page & XAI Lab (/)
1.64 GB stream-filtered           0 HTTP 429 limits (~25-30s batch)         142,090 GIST-indexed assets          Interactive India Relief & #analyze
                                  0-100 Priority Threat Index               48.4k nodes / 93.7k ways
Copernicus Sentinel-2 L2A   ───▶  6-Ch PyTorch CNN + Grad-CAM XAI      ───▶ FastAPI Backend (Uvicorn)       ───▶ 3D Anomaly Dashboard (/dashboard)
10 m RGB + SWIR chips             EfficientNet-B0 Champion (4.01M)          /api/events/recent, /api/summary     Deck.gl v9 + MapLibre GL v4
116 chips / 111 events            ~238 ms CPU inference                     /api/analyze-sample, /api/ml/metrics Event Dossier & Sample Analyzer
```

---

## Key Features

- **Survey of India Sovereign Demarcation:** Enforces official Survey of India territorial boundaries (complete Jammu & Kashmir, Ladakh, Aksai Chin, and Arunachal Pradesh) with forensic logging of out-of-boundary detections (`boundary_exclusions_audit`).
- **Zero-Rate-Limit Local PostGIS OSM Index:** Stream-filters Geofabrik's 1.64 GB India `.osm.pbf` snapshot via PyOsmium into 142,090 GIST-indexed industrial geometries (`74,240` energy infrastructure, `39,135` industrial, `12,490` kiln/refinery, `12,003` mining, `4,222` power plants), eliminating Overpass API HTTP 429 failures.
- **Auditable 0-100 Investigation Threat Index:** Weighted scoring combining Thermal Intensity FRP (0-40 pts), Temporal Persistence (0-30 pts), and Local OSM Infrastructure Proximity (0-30 pts), with per-event JSON breakdowns.
- **6-Channel Multispectral CNN + Grad-CAM Attribution:** Expands ImageNet-pretrained backbones from 3 to 6 input channels (`RGB` B04-B03-B02 + `SWIR` B12-B11-B8A) and renders spatial Grad-CAM heatmaps inside both the Event Dossier and on-demand Sample Analyzer.
- **Unified Dual-Theme Web Interface:** Cohesive dark (charcoal-teal) and light (coastal-teal and warm sand) design system across:
  - **Landing Portal (`/`):** Interactive state-level relief explorer, timeline trends, ML validation benchmarks, and inline `#analyze` Sample Analyzer.
  - **3D Geospatial Dashboard (`/dashboard`):** Deck.gl 3D FRP columns with two-tier opacity, warm-gold active selection highlights, pulsing target rings, unobscured country labels, Event Dossier panel, and theme-synced Sample Analyzer modal.

---

## Experimental Validation & Benchmarks

### 1. Priority Scoring Ablation (`N = 6,183` Events, 51 High-Consequence)

| Metric | Full Pipeline (Local PostGIS OSM) | OSM-Ablated Baseline | Lift |
| :--- | :---: | :---: | :---: |
| **Precision@10** | **90.0%** | 60.0% | **+30.0 pp** |
| **Precision@25** | **96.0%** | 44.0% | **+52.0 pp** |
| **Precision@50** | **50.0%** | 24.0% | **+26.0 pp** |
| **Precision@100** | **25.0%** | 17.0% | **+8.0 pp** |
| **Batch Enrichment Time** | **~25-30 sec (0 rate limits)** | Blocked / >60 min | **100% Local** |

### 2. PyTorch 4-Class Multispectral CNN Benchmark

| Model Architecture | Channels | Parameters | Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| MobileNetV2 (Baseline) | 3 (RGB) | 2.23M | 25.0% | 19.6% | 17.9% |
| MobileNetV2 | 6 (RGB+SWIR) | 2.23M | 25.0% | 17.4% | 15.7% |
| ResNet18 | 6 (RGB+SWIR) | 11.19M | 25.0% | 20.8% | 23.6% |
| **EfficientNet-B0 (Champion)** | **6 (RGB+SWIR)** | **4.01M** | **33.3%** | **37.5%** | **35.8%** |

*`efficientnet_b0_6ch` achieves a +90.9% relative Macro F1 lift over the 3-channel baseline and is the only evaluated architecture with non-zero F1 across all four classes (`industrial_thermal`, `agricultural_stubble_burn`, `forest_wildfire`, `urban_false_positive`). Champion weights (`thermaleye_best.pt`, 15.6 MB) are hosted via [GitHub Releases (`v1.0.0-model`)](https://github.com/bhanusiingh/ThermalEye/releases/tag/v1.0.0-model).*

---

## Tech Stack

- **Backend & API:** Python 3.12, FastAPI, Uvicorn, asyncpg, psycopg2-binary, SQLAlchemy 2.0, GeoAlchemy2, GeoPandas, Shapely 2.0, PyOsmium
- **Deep Learning & XAI:** PyTorch (CPU), Torchvision (`EfficientNet-B0`, `ResNet18`, `MobileNetV2`), Grad-CAM, scikit-learn, Pillow, NumPy
- **Spatial Database:** PostgreSQL 15-18 + PostGIS 3.4-3.6 (GIST spatial indexing)
- **Frontend & 3D GIS:** React 18, TypeScript, Vite 6, Deck.gl v9, MapLibre GL JS v4, Tailwind CSS v4 (Fraunces + Public Sans)
- **Infrastructure:** Docker Compose (4-container local stack), Nginx Alpine, Neon Serverless PostGIS, Render Web Service (`Dockerfile.render`)

---

## Quick Start (Local Docker Compose)

### 1. Prerequisites
- Docker & Docker Compose
- Free NASA FIRMS `MAP_KEY` ([Request key](https://firms.modaps.eosdis.nasa.gov/api/map_key))
- Optional Copernicus CDSE credentials for Sentinel-2 chip retrieval

### 2. Configure Environment
```bash
cp .env.example .env
```
Populate `NASA_FIRMS_MAP_KEY` / `FIRMS_MAP_KEY` (and optional CDSE credentials) inside `.env`.

### 3. Build & Launch Stack
```bash
docker compose up -d --build
```
Services start at:
- **Landing Page & Sample Analyzer:** `http://localhost/`
- **3D Geospatial Dashboard:** `http://localhost/dashboard`
- **FastAPI OpenAPI Docs:** `http://localhost:8000/docs`
- **PostGIS Database:** `localhost:5432`

---

## Cloud Deployment (Render + Neon PostGIS)

ThermalEye ships with a multi-stage single-container cloud build (`Dockerfile.render`) engineered for 512 MB RAM environments (CPU-only PyTorch wheels, `torch.set_num_threads(1)`, single Uvicorn worker, explicit `gc.collect()` after Grad-CAM, and automatic model download on first inference).

Required environment variables on Render:
- `DATABASE_URL` — Neon pooled PostgreSQL connection string (`sslmode=require`)
- `SERVE_FRONTEND=1` — Serves compiled `frontend/dist` directly from FastAPI
- `MODEL_URL=https://github.com/bhanusiingh/ThermalEye/releases/download/v1.0.0-model/thermaleye_best.pt`
- `NASA_FIRMS_MAP_KEY`, `CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET`

---

## Local Development & Pipeline Commands

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Apply PostGIS schema
psql -U thermaleye -d thermaleye -f schema.sql

# 3. Optional one-time OSM PBF import (Geofabrik India snapshot)
python import_osm_pbf.py

# 4. Run full ingestion, clustering, enrichment, imagery & ML scoring pipeline
python run_pipeline.py

# 5. Start API server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# 6. Start frontend dev server
cd frontend && npm install && npm run dev
```

---

## Running Automated Tests

ThermalEye includes 66 automated unit and integration tests covering API analytics, PostGIS spatial enrichment, STAC imagery handling, 6-channel ML inference, Grad-CAM attribution, and cloud DSN/frontend serving:

```bash
python -m unittest discover tests -v
```

---

## Author

**Bhanu Pratap Singh** — School of Computer Science and Engineering, Lovely Professional University (`CSE471: Deep Learning for Computer Vision`).

## License

This project is licensed under the MIT License.
