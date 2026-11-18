# ThermalEye 🛰️🔥
### Autonomous Geospatial Intelligence & Thermal Anomaly Monitoring Platform for India

ThermalEye is a near-real-time satellite thermal anomaly monitoring and investigation platform built specifically for the sovereign territory of the Republic of India. It autonomously ingests spaceborne infrared data, clusters anomalies into discrete fire/heat events, enriches them with nearby infrastructure context from OpenStreetMap, acquires high-resolution Sentinel-2 multispectral imagery, and runs machine-learning land-cover classification.

---

## Architecture Overview

ThermalEye implements a 5-stage sequential processing pipeline backed by PostGIS:

```
NASA FIRMS (VIIRS/MODIS) 
  ↳ 1. Ingestion & Deterministic Hashing (SHA-256)
  ↳ 2. Spatio-Temporal Clustering (ST-DBSCAN: 2km radius, 24h window)
  ↳ 3. OpenStreetMap Enrichment (Overpass API 3,000m radius + 0-100 Priority Scoring)
  ↳ 4. Multispectral Satellite Retrieval (Copernicus CDSE Sentinel-2 L2A RGB & SWIR)
  ↳ 5. ML Land-Cover Classification (MobileNetV2 Context Inference)
  ↳ Delivery: FastAPI Backend + Deck.gl v9 / MapLibre 3D Intelligence Dashboard
```

---

## Key Features

- **Survey of India Boundary Demarcation:** Strictly adheres to the constitutional sovereign boundaries of India (including full Jammu & Kashmir, Ladakh, Aksai Chin, and Arunachal Pradesh). All out-of-boundary detections are recorded in a forensic audit log (`boundary_exclusions_audit`).
- **Investigation Priority Score (0–100):** Multi-attribute scoring heuristic combining thermal radiative power (40%), temporal persistence across passes (30%), and infrastructure proximity (30%).
- **Rate-Limit Resilience:** Decouples core scoring from external Overpass API availability with automated exponential backoff, jitter, and fallback scoring (`0/30 (Overpass Rate Limited)`).
- **Dual-Surface Web Client:**
  - **Landing Portal (`/`):** Pure HTML5/Canvas organic relief map of India with interactive state rankings, timeline analytics, and provider health matrix.
  - **Geospatial Dashboard (`/dashboard`):** React 18, Deck.gl v9, MapLibre GL, 3D FRP column extrusions, dual dark/light theme tokens, and a 6-module Bento Grid Event Dossier panel.
- **Production Containerization:** Single-command Docker Compose stack running PostGIS 15, FastAPI, background daemon worker, and multi-stage Nginx static web server.

---

## Tech Stack

- **Backend:** Python 3.12, FastAPI, Uvicorn, SQLAlchemy 2.0, GeoAlchemy2, asyncpg, psycopg2-binary, GeoPandas, Shapely 2.0, scikit-learn, Tenacity
- **Machine Learning:** PyTorch, Hugging Face Transformers (`MobileNetV2`), Pillow
- **Database:** PostgreSQL 15 + PostGIS 3.4
- **Frontend:** React 18, TypeScript, Deck.gl v9, MapLibre GL v4, Tailwind CSS v4, Vite 6
- **Typography:** Fraunces (600 serif) & Public Sans (sans-serif)
- **Deployment:** Docker & Docker Compose, Nginx Alpine

---

## Quick Start

### 1. Prerequisites
- Docker & Docker Compose (or Python 3.12 + Node.js 20 + PostgreSQL/PostGIS 15 locally)
- Free NASA FIRMS MAP_KEY ([Get one here](https://firms.modaps.eosdis.nasa.gov/api/map_key))

### 2. Configure Environment
Copy the example environment configuration:
```bash
cp .env.example .env
```
Edit `.env` and fill in your `FIRMS_MAP_KEY` and optional Copernicus CDSE credentials.

### 3. Run with Docker Compose (Single Command)
```bash
docker compose up -d --build
```
The services will be available at:
- **Landing Page & Web Dashboard:** `http://localhost/` (Dashboard: `http://localhost/dashboard`)
- **FastAPI Documentation:** `http://localhost:8000/docs`
- **PostGIS Database:** `localhost:5432`

---

## Local Development Setup

### Backend & Ingestion Pipeline
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run database migrations / schema
psql -U thermaleye -d thermaleye -f schema.sql

# 3. Execute the full sequential pipeline once
python run_pipeline.py

# Or run continuously as a daemon (hourly polling)
python run_pipeline.py --daemon --interval 3600

# 4. Start the FastAPI API server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173/` or `http://localhost:5173/dashboard`.

---

## Running Automated Tests

ThermalEye includes 38 comprehensive unit and integration tests:
```bash
python -m unittest discover tests -v
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
