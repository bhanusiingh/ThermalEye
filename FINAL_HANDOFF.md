# ThermalEye Phase 17 — Final Handoff & Demo Guide

## Prerequisites & Environment Setup
- Python 3.9 - 3.12
- Git for Windows (or equivalent)
- Sufficient disk space for Sentinel-2 satellite patches (~1 MB per image).
- **Environment Variables**: The system expects a `.env` file in the root directory. Copy `.env.example` to `.env` and fill in:
  - `FIRMS_MAP_KEY` (Free from NASA FIRMS)
  - `CDSE_CLIENT_ID` & `CDSE_CLIENT_SECRET` (For Sentinel-2 imagery via Copernicus Data Space Ecosystem).

## Exact Installation and Launch Commands
```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard locally
python run_dashboard.py
```
*Note: The dashboard will start automatically on `http://localhost:8501`. If it does not, run `streamlit run src/thermaleye/dashboard/app.py` directly.*

## Demo Workflow
For a repeatable, visually compelling offline demo without relying on API credentials, the project contains a fully processed historical pilot dataset. This dataset spans geographically-bounded candidate detections associated with documented historical incidents (Baghjan Blowout, UPL Jhagadia, HPCL Visakh, Yashashvi Rasayan, and regional fires in Sangrur and Simlipal).

1. Ensure the `data/pilot/` directory is fully populated with `fusion_all.csv` (this was done in Phase 17.2).
2. Launch the dashboard (`python run_dashboard.py`).
3. The dashboard will automatically detect and load the `data/pilot/fusion_all.csv` database without making live API requests.
4. **Overview Tab**: Show the National Map, emphasizing that thermal candidate clusters are plotted at their satellite-derived centroids (which may be offset from the reported incident coordinates). Note the total event count and the breakdown of priority scorings.
5. **Investigation Table**: Drill down into `BAGHJAN_TE-EVT-00001` or `UPL_TE-EVT-00001`. Point out the evidence coverage flags. 
6. Show how missing OpenStreetMap (OSM) data cleanly renders as `Unavailable` and does not negatively penalize the candidate, and how the dashboard gracefully handles limited evidence.

## Data Limitations & Known Issues
1. **The 2.0 km / 24-hour Rule vs Ground Truth**: Evaluation matching dictates that historical incident coordinates must align within a strict 2.0 km and 24-hour window of the satellite pixel centroid to be considered a confirmed match. Industrial estates (e.g., Dahej SEZ, UPL Jhagadia) are massive, meaning approximate journalistic coordinates can be offset by 6 to 12 km. The 2.0 km rule is a configured methodology, not an assertion of perfect scientific ground truth. Exploratory sensitivity tests at 15 km / 100-hours successfully linked these candidates, but such tests do not establish formal detection accuracy and should not be treated as the default.
2. **Heuristic Triage, Not Probability**: The `High`, `Medium`, and `Low` fusion scores are uncalibrated triage heuristics. They are *not* scientific fire probabilities.
3. **EuroSAT as Context, Not Classifier**: The Sentinel-2 computer vision component uses EuroSAT strictly to provide industrial land-cover context. It is not an active fire detection model.
4. **Regional Events**: District-wide stubble burning (Sangrur) and biosphere fires (Simlipal) generate dozens of legitimate candidate clusters. Point-based 1-to-1 distance evaluation cannot validly map one coordinate pin to 50 regional detections.
5. **Overpass API Rate Limits**: The OSM enrichment module queries the public Overpass API and will frequently hit `HTTP 429` (Rate Limited) timeouts during bulk ingestion. The pipeline gracefully converts this to `Unavailable` evidence.

## Deferred Tasks
- **3D / UI Overhaul**: Advanced UI polishing (like 3D effects on the PyDeck map or dark-mode overrides) was intentionally deferred to prioritize backend data integrity and metric honesty before handoff.
- **Model Training**: Fine-tuning a custom industrial-fire vision model is deferred to a future phase.
