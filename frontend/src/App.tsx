import React, { useState, useEffect, useCallback, useMemo } from "react";
import { createRoot } from "react-dom/client";
import { Map } from "react-map-gl/maplibre";
import DeckGL from "@deck.gl/react";
import { GeoJsonLayer } from "@deck.gl/layers";
import { ColumnLayer } from "@deck.gl/layers";
import { LightingEffect, AmbientLight, DirectionalLight } from "@deck.gl/core";
import "maplibre-gl/dist/maplibre-gl.css";
import "./index.css";
import EventDossierPanel from "./components/EventDossierPanel";
import SampleAnalyzerModal from "./components/SampleAnalyzerModal";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? "";
const MAP_STYLE =
  (import.meta as any).env?.VITE_MAP_STYLE ??
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

type ThemeMode = "dark" | "light";

interface MapPalette {
  bg: string;
  bg2: string;
  landSubtle: string;
  depth: string;
  line: string;
  road: string;
  tx: string;
  mut: string;
  hotRgb: [number, number, number];
  emberRgb: [number, number, number];
  coolRgb: [number, number, number];
  unkRgb: [number, number, number];
}

// Map style configuration synced to :root / :root[data-theme="light"] tokens
const MAP_PALETTES: Record<ThemeMode, MapPalette> = {
  dark: {
    bg: "#0d1a1f",
    bg2: "#12232a",
    landSubtle: "#162a32",
    depth: "#091418",
    line: "#26393f",
    road: "#1d3037",
    tx: "#e8edeb",
    mut: "#93a6a7",
    hotRgb: [224, 164, 88],
    emberRgb: [208, 105, 76],
    coolRgb: [116, 169, 174],
    unkRgb: [111, 132, 134],
  },
  light: {
    bg: "#f0f3f1",
    bg2: "#fbfcfb",
    landSubtle: "#e8efec",
    depth: "#b9c8c4",
    line: "#d3dcd8",
    road: "#dce4e0",
    tx: "#15292f",
    mut: "#566c71",
    hotRgb: [180, 112, 31],
    emberRgb: [173, 79, 55],
    coolRgb: [47, 106, 115],
    unkRgb: [123, 143, 147],
  },
};

// Initial camera — centered on India
const INITIAL_VIEW_STATE = {
  longitude: 78.9629,
  latitude: 22.5937,
  zoom: 4.5,
  pitch: 45,
  bearing: -15,
  minZoom: 3,
  maxZoom: 18,
};

// State-level camera presets aligned with Survey of India state names
const INDIAN_STATE_VIEWPORTS: Record<
  string,
  { longitude: number; latitude: number; zoom: number }
> = {
  "Andaman and Nicobar Islands": { longitude: 92.88, latitude: 12.865, zoom: 7.8 },
  "Andhra Pradesh": { longitude: 80.76, latitude: 15.875, zoom: 6.2 },
  "Arunachal Pradesh": { longitude: 94.475, latitude: 28.055, zoom: 6.6 },
  "Assam": { longitude: 92.86, latitude: 26.06, zoom: 6.6 },
  "Bihar": { longitude: 85.815, latitude: 25.905, zoom: 6.8 },
  "Chandigarh": { longitude: 76.775, latitude: 30.72, zoom: 10.0 },
  "Chhattisgarh": { longitude: 82.335, latitude: 20.945, zoom: 6.4 },
  "Dadra and Nagar Haveli and Daman and Diu": { longitude: 73.07, latitude: 20.21, zoom: 9.0 },
  "Delhi": { longitude: 77.1, latitude: 28.635, zoom: 9.5 },
  "Goa": { longitude: 74.01, latitude: 15.325, zoom: 8.5 },
  "Gujarat": { longitude: 71.285, latitude: 22.415, zoom: 6.3 },
  "Haryana": { longitude: 76.03, latitude: 29.275, zoom: 7.2 },
  "Himachal Pradesh": { longitude: 77.305, latitude: 31.815, zoom: 7.2 },
  "Jammu and Kashmir": { longitude: 75.08, latitude: 33.71, zoom: 7.0 },
  "Jharkhand": { longitude: 85.645, latitude: 23.655, zoom: 6.8 },
  "Karnataka": { longitude: 76.34, latitude: 15.015, zoom: 6.3 },
  "Kerala": { longitude: 76.135, latitude: 10.525, zoom: 7.0 },
  "Ladakh": { longitude: 76.43, latitude: 34.71, zoom: 6.2 },
  "Lakshadweep": { longitude: 72.6369, latitude: 10.5667, zoom: 8.2 },
  "Madhya Pradesh": { longitude: 78.43, latitude: 23.975, zoom: 6.0 },
  "Maharashtra": { longitude: 76.775, latitude: 18.82, zoom: 6.0 },
  "Manipur": { longitude: 93.86, latitude: 24.725, zoom: 7.8 },
  "Meghalaya": { longitude: 91.305, latitude: 25.575, zoom: 7.5 },
  "Mizoram": { longitude: 92.845, latitude: 23.26, zoom: 7.5 },
  "Nagaland": { longitude: 94.28, latitude: 26.115, zoom: 7.8 },
  "Odisha": { longitude: 84.44, latitude: 20.19, zoom: 6.5 },
  "Puducherry": { longitude: 79.785, latitude: 10.915, zoom: 9.0 },
  "Punjab": { longitude: 75.405, latitude: 31.035, zoom: 7.2 },
  "Rajasthan": { longitude: 73.875, latitude: 26.625, zoom: 5.9 },
  "Sikkim": { longitude: 88.48, latitude: 27.585, zoom: 8.2 },
  "Tamil Nadu": { longitude: 78.285, latitude: 10.82, zoom: 6.5 },
  "Telangana": { longitude: 79.275, latitude: 17.875, zoom: 6.9 },
  "Tripura": { longitude: 91.745, latitude: 23.75, zoom: 8.0 },
  "Uttar Pradesh": { longitude: 80.865, latitude: 27.135, zoom: 6.1 },
  "Uttarakhand": { longitude: 79.315, latitude: 30.09, zoom: 7.3 },
  "West Bengal": { longitude: 87.845, latitude: 24.39, zoom: 6.5 },
};

function findStateViewport(stateName: string) {
  const normalized = stateName.trim().toLowerCase();
  for (const [name, viewport] of Object.entries(INDIAN_STATE_VIEWPORTS)) {
    if (name.toLowerCase() === normalized) {
      return viewport;
    }
  }
  return null;
}

function getInitialTheme(): ThemeMode {
  try {
    const attr = document.documentElement.dataset.theme;
    if (attr === "light" || attr === "dark") return attr;
    const saved = localStorage.getItem("te-theme");
    if (saved === "light" || saved === "dark") return saved;
    if (window.matchMedia("(prefers-color-scheme: light)").matches) return "light";
  } catch {
    // Ignore storage access errors
  }
  return "dark";
}

// ---------------------------------------------------------------------------
// Color scale: priority_score 0-100 -> var(--hot) to var(--ember)
// ---------------------------------------------------------------------------
function priorityToColor(
  score: number | null,
  theme: ThemeMode
): [number, number, number, number] {
  const pal = MAP_PALETTES[theme];
  if (score == null) return [...pal.unkRgb, 185];
  const t = Math.max(0, Math.min(1, score / 100));
  const [r1, g1, b1] = pal.hotRgb;
  const [r2, g2, b2] = pal.emberRgb;
  return [
    Math.round(r1 + (r2 - r1) * t),
    Math.round(g1 + (g2 - g1) * t),
    Math.round(b1 + (b2 - b1) * t),
    Math.round(212 + 28 * t),
  ];
}

// ---------------------------------------------------------------------------
// Theme & Sanitize MapLibre Basemap Style
// ---------------------------------------------------------------------------
function buildThemedMapStyle(rawStyle: any, theme: ThemeMode) {
  if (!rawStyle || !Array.isArray(rawStyle.layers)) return rawStyle;
  const pal = MAP_PALETTES[theme];
  const majorLabelIds = new Set([
    "place_country_1",
    "place_capital_dot_z7",
    "place_city_r5",
    "place_city_r6",
  ]);

  const layers = rawStyle.layers.map((layer: any) => {
    const id: string = layer.id || "";
    const paint = { ...(layer.paint || {}) };

    // Hide international boundary layers that misrepresent Indian sovereignty
    if (
      id === "boundary_country_inner" ||
      id === "boundary_country_outline" ||
      id === "boundary_state" ||
      id === "boundary_county"
    ) {
      return {
        ...layer,
        layout: { ...(layer.layout || {}), visibility: "none" },
      };
    }

    if (layer.type === "background") {
      paint["background-color"] = pal.bg2;
      paint["background-opacity"] = 0.92;
      return { ...layer, paint };
    }

    if (id === "water" || id === "water_shadow") {
      paint["fill-color"] = pal.depth;
      paint["fill-opacity"] = id === "water_shadow" ? 0.5 : 0.94;
      return { ...layer, paint };
    }

    if (id === "waterway") {
      paint["line-color"] = pal.depth;
      return { ...layer, paint };
    }

    if (
      id === "landcover" ||
      id.startsWith("park_") ||
      id.startsWith("landuse")
    ) {
      paint["fill-color"] = pal.landSubtle;
      return { ...layer, paint };
    }

    if (id === "building" || id === "building-top") {
      paint["fill-color"] = pal.landSubtle;
      if ("fill-outline-color" in paint) {
        paint["fill-outline-color"] = pal.line;
      }
      return { ...layer, paint };
    }

    if (
      id.startsWith("road_") ||
      id.startsWith("tunnel_") ||
      id.startsWith("bridge_") ||
      id.startsWith("aeroway")
    ) {
      if ("line-color" in paint) {
        paint["line-color"] =
          id.includes("mot") || id.includes("trunk") || id.includes("pri")
            ? pal.line
            : pal.road;
      }
      return { ...layer, paint };
    }

    if (layer.type === "symbol") {
      if ("text-color" in paint) {
        paint["text-color"] = majorLabelIds.has(id) ? pal.tx : pal.mut;
      }
      if ("text-halo-color" in paint) {
        paint["text-halo-color"] = pal.bg;
      }
      if ("icon-color" in paint) {
        paint["icon-color"] = pal.mut;
      }
      return { ...layer, paint };
    }

    return layer;
  });

  return { ...rawStyle, layers };
}

// ---------------------------------------------------------------------------
// 3D Lighting
// ---------------------------------------------------------------------------
const ambientLight = new AmbientLight({ color: [255, 255, 255], intensity: 1.0 });
const dirLight = new DirectionalLight({
  color: [255, 255, 255],
  intensity: 0.9,
  direction: [-3, -9, -1],
});
const lightingEffect = new LightingEffect({ ambientLight, dirLight });

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface EventProperties {
  id: string;
  first_seen: string;
  last_seen: string;
  max_frp: number | null;
  priority_score: number | null;
  priority_reasons: Record<string, unknown> | null;
  nearest_osm_type: string | null;
  nearest_osm_name: string | null;
  nearest_osm_distance_km: number | null;
  land_cover_class: string | null;
  land_cover_confidence?: number | null;
  ml_model_arch?: string | null;
  ml_disclaimer?: string | null;
  imagery_rgb_url?: string | null;
  imagery_swir_url?: string | null;
  gradcam_url?: string | null;
  detection_count: number | null;
  footprint: GeoJSON.Geometry | null;
  osm_disclaimer: string;
}

interface EventFeatureCollection
  extends GeoJSON.FeatureCollection<GeoJSON.Point, EventProperties> {}

// ---------------------------------------------------------------------------
// App Component
// ---------------------------------------------------------------------------
export default function App() {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);
  const [eventsData, setEventsData] = useState<EventFeatureCollection | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [indiaBoundary, setIndiaBoundary] = useState<any>(null);
  const [rawMapStyle, setRawMapStyle] = useState<any>(null);
  const [deepLinkEventCentered, setDeepLinkEventCentered] = useState(false);
  const [isAnalyzeOpen, setIsAnalyzeOpen] = useState(false);
  const [isPickingLocation, setIsPickingLocation] = useState(false);
  const [pickedCoordinates, setPickedCoordinates] = useState<{ lat: number; lon: number } | null>(null);

  // ---- Sync Theme Attribute & Persistence ---------------------------------
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("te-theme", theme);
    } catch {
      // Ignore storage access errors
    }
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }, []);

  // ---- Deep-linking: ?event=<id>, ?state=<name>, ?analyze=1 ---------------
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const eventParam = params.get("event");
    const stateParam = params.get("state");
    const analyzeParam = params.get("analyze");

    if (analyzeParam === "1") {
      setIsAnalyzeOpen(true);
    }

    if (eventParam && eventParam.trim()) {
      setSelectedEventId(eventParam.trim());
    }

    if (stateParam && stateParam.trim()) {
      const target = findStateViewport(stateParam);
      if (target) {
        setViewState((prev) => ({
          ...prev,
          longitude: target.longitude,
          latitude: target.latitude,
          zoom: target.zoom,
        }));
        setDeepLinkEventCentered(true);
      }
    }
  }, []);

  // Cancel location picking mode on Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isPickingLocation) {
        setIsPickingLocation(false);
        setIsAnalyzeOpen(true);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isPickingLocation]);

  // Center on deep-linked event once active events load (if no ?state was given)
  useEffect(() => {
    if (!selectedEventId || deepLinkEventCentered || !eventsData?.features) return;
    const params = new URLSearchParams(window.location.search);
    if (!params.get("event") || params.get("state")) return;
    const match = eventsData.features.find((f) => f.properties.id === selectedEventId);
    if (match?.geometry?.type === "Point") {
      const [lon, lat] = match.geometry.coordinates;
      setViewState((prev) => ({
        ...prev,
        longitude: lon,
        latitude: lat,
        zoom: Math.max(prev.zoom, 8.5),
      }));
      setDeepLinkEventCentered(true);
    }
  }, [selectedEventId, deepLinkEventCentered, eventsData]);

  // ---- Fetch Base Basemap Style Once --------------------------------------
  useEffect(() => {
    fetch(MAP_STYLE)
      .then((res) => res.json())
      .then((style) => {
        if (style && Array.isArray(style.layers)) {
          setRawMapStyle(style);
        }
      })
      .catch((err) => {
        console.warn("[ThermalEye] Could not customize basemap style:", err);
        setRawMapStyle(null);
      });
  }, []);

  const customMapStyle = useMemo(() => {
    if (!rawMapStyle) return MAP_STYLE;
    return buildThemedMapStyle(rawMapStyle, theme);
  }, [rawMapStyle, theme]);

  // ---- Fetch Official Survey of India Boundary ----------------------------
  useEffect(() => {
    fetch("/india_boundary.geojson")
      .then((res) => {
        if (!res.ok) throw new Error("Local boundary asset not found");
        return res.json();
      })
      .then((data) => setIndiaBoundary(data))
      .catch(() => {
        fetch(`${API_BASE}/api/v1/india-boundary`)
          .then((res) => res.json())
          .then((data) => setIndiaBoundary(data))
          .catch((err) => console.error("[ThermalEye] Boundary fetch error:", err));
      });
  }, []);

  // ---- Fetch active events ------------------------------------------------
  const fetchActiveEvents = useCallback(async () => {
    setLoading(true);
    setError(null);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);
    try {
      const res = await fetch(`${API_BASE}/api/v1/events/active?days=7`, {
        signal: controller.signal,
      });
      if (!res.ok) {
        console.error(`[ThermalEye] Events API request failed with HTTP ${res.status}: ${res.statusText}`);
        throw new Error(`API returned ${res.status}: ${res.statusText}`);
      }
      const data: EventFeatureCollection = await res.json();
      setEventsData(data);
    } catch (err: unknown) {
      const isAbort = err instanceof Error && err.name === "AbortError";
      const message = isAbort
        ? "Request timed out after 15 seconds."
        : err instanceof Error
        ? err.message
        : "Failed to fetch events.";
      setError(message);
      console.error("[ThermalEye] Fetch error:", err);
    } finally {
      clearTimeout(timeoutId);
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchActiveEvents();
    // Auto-refresh every 5 minutes
    const interval = setInterval(fetchActiveEvents, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchActiveEvents]);

  // ---- Build Column data from centroids -----------------------------------
  const columnData = useMemo(() => {
    if (!eventsData?.features) return [];
    return eventsData.features
      .filter((f) => f.geometry?.type === "Point")
      .map((f) => ({
        position: f.geometry.coordinates as [number, number],
        frp: f.properties.max_frp ?? 1,
        priorityScore: f.properties.priority_score,
        id: f.properties.id,
        detectionCount: f.properties.detection_count ?? 1,
      }));
  }, [eventsData]);

  // ---- Build footprint FeatureCollection for polygons ---------------------
  const footprintData = useMemo(() => {
    if (!eventsData?.features) return { type: "FeatureCollection" as const, features: [] };
    const polyFeatures = eventsData.features
      .filter((f) => f.properties.footprint != null)
      .map((f) => ({
        type: "Feature" as const,
        geometry: f.properties.footprint!,
        properties: {
          id: f.properties.id,
          priority_score: f.properties.priority_score,
        },
      }));
    return { type: "FeatureCollection" as const, features: polyFeatures };
  }, [eventsData]);

  // ---- Click handler ------------------------------------------------------
  const handleClick = useCallback((info: any) => {
    if (isPickingLocation && info?.coordinate) {
      const [lon, lat] = info.coordinate;
      setPickedCoordinates({ lat, lon });
      setIsPickingLocation(false);
      setIsAnalyzeOpen(true);
      return;
    }
    if (info?.object) {
      const id = info.object.id ?? info.object.properties?.id;
      if (id) {
        setSelectedEventId(id);
      }
    }
  }, [isPickingLocation]);

  // ---- Deck.gl layers -----------------------------------------------------
  const layers = useMemo(() => {
    const result: any[] = [];
    const pal = MAP_PALETTES[theme];

    // 0) Official Survey of India Sovereign Boundary Layers
    if (indiaBoundary) {
      // Subtle soft drop-shadow glow in var(--cool)
      result.push(
        new GeoJsonLayer({
          id: "official-india-boundary-glow",
          data: indiaBoundary,
          pickable: false,
          stroked: true,
          filled: false,
          lineWidthMinPixels: 4.5,
          getLineWidth: 5,
          getLineColor: [...pal.coolRgb, theme === "dark" ? 48 : 38],
          updateTriggers: {
            getLineColor: [theme],
          },
        })
      );
      // Refined 1.8px var(--cool) sovereign boundary line + subtle territory tint
      result.push(
        new GeoJsonLayer({
          id: "official-india-boundary-core",
          data: indiaBoundary,
          pickable: false,
          stroked: true,
          filled: true,
          lineWidthMinPixels: 1.8,
          getLineWidth: 1.8,
          getLineColor: [...pal.coolRgb, theme === "dark" ? 235 : 225],
          getFillColor: [...pal.coolRgb, theme === "dark" ? 14 : 10],
          updateTriggers: {
            getLineColor: [theme],
            getFillColor: [theme],
          },
        })
      );
    }

    // 1) Footprint polygons
    if (footprintData.features.length > 0) {
      result.push(
        new GeoJsonLayer({
          id: "event-footprints",
          data: footprintData as any,
          pickable: true,
          stroked: true,
          filled: true,
          lineWidthMinPixels: 1,
          getLineColor: [...pal.emberRgb, 190],
          getFillColor: (f: any) => {
            const score = f.properties?.priority_score;
            return priorityToColor(score, theme);
          },
          getLineWidth: 1.5,
          onClick: handleClick,
          updateTriggers: {
            getLineColor: [theme],
            getFillColor: [footprintData, theme],
          },
        })
      );
    }

    // 2) 3D Column layer — FRP extrusion
    if (columnData.length > 0) {
      const maxFrp = Math.max(...columnData.map((d) => d.frp || 1), 1);

      result.push(
        new ColumnLayer({
          id: "frp-columns",
          data: columnData,
          pickable: true,
          diskResolution: 12,
          radius: 8000,
          extruded: true,
          elevationScale: 200,
          getPosition: (d: any) => d.position,
          getElevation: (d: any) => ((d.frp || 1) / maxFrp) * 500,
          getFillColor: (d: any) => priorityToColor(d.priorityScore, theme),
          onClick: handleClick,
          updateTriggers: {
            getFillColor: [columnData, theme],
            getElevation: [columnData],
          },
        })
      );
    }

    return result;
  }, [columnData, footprintData, handleClick, indiaBoundary, theme]);

  // ---- Tooltip ------------------------------------------------------------
  const getTooltip = useCallback((info: any) => {
    if (!info?.object) return null;
    const obj = info.object;
    const score = obj.priorityScore ?? obj.properties?.priority_score;
    const frp = obj.frp ?? obj.properties?.max_frp;
    const id = obj.id ?? obj.properties?.id ?? "—";

    return {
      html: `
        <div style="font-family: var(--font-sans); font-size: 13px; line-height: 1.45; min-width: 175px;">
          <div style="font-family: var(--font-display); font-weight: 600; font-size: 15px; color: var(--hot); margin-bottom: 6px; letter-spacing: -0.015em;">
            Event #${String(id).substring(0, 8).toUpperCase()}
          </div>
          <dl style="display: grid; grid-template-columns: 1fr auto; gap: 3px 12px; margin: 0;">
            <dt style="color: var(--mut);">Threat score</dt>
            <dd style="margin: 0; color: var(--tx); font-weight: 600; font-variant-numeric: tabular-nums;">${
              score != null ? `${score.toFixed(1)} / 100` : "N/A"
            }</dd>
            <dt style="color: var(--mut);">Peak FRP</dt>
            <dd style="margin: 0; color: var(--tx); font-weight: 600; font-variant-numeric: tabular-nums;">${
              frp != null ? `${frp.toFixed(1)} MW` : "N/A"
            }</dd>
          </dl>
        </div>
      `,
      style: {
        backgroundColor: "color-mix(in srgb, var(--bg) 92%, transparent)",
        backdropFilter: "blur(8px)",
        color: "var(--tx)",
        border: "1px solid var(--line)",
        borderRadius: "10px",
        padding: "12px 14px",
        boxShadow: "0 12px 28px rgba(0, 0, 0, 0.28)",
      },
    };
  }, []);

  const activeCount = eventsData?.features?.length
    ? eventsData.features.length.toLocaleString()
    : "0";

  return (
    <div className="te-dashboard">
      {/* ------------------------------------------------------------------ */}
      {/* Ambient Background Blobs & Rotating Contour Rings (Landing Theme) */}
      {/* ------------------------------------------------------------------ */}
      <div className="bg" aria-hidden="true">
        <i className="b1" />
        <i className="b2" />
        <i className="b3" />
        <i className="ct" />
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Dashboard Top Navigation Bar */}
      {/* ------------------------------------------------------------------ */}
      <header className="te-nav te-enter-nav">
        <div className="te-nav-inner">
          <div className="te-brand">
            <span className="te-brand-mark" aria-hidden="true">
              <svg
                className="w-4 h-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z" />
              </svg>
            </span>
            <div className="te-brand-text">
              <span className="te-logo">ThermalEye</span>
              <span className="te-subtitle">Geospatial Intelligence Platform</span>
            </div>
          </div>

          <div className="te-nav-actions">
            {loading && (
              <div className="te-status-inline is-warn" role="status" aria-live="polite">
                <span className="te-dot te-dot-warn" />
                <span className="te-btn-label">Syncing</span>
              </div>
            )}

            {error && (
              <div className="te-status-inline is-bad" role="alert">
                <span className="te-dot te-dot-bad" />
                <span className="te-btn-label">Connection error</span>
              </div>
            )}

            <div className="te-chip" aria-label={`${activeCount} active events`}>
              <span className="te-chip-count">{activeCount}</span>
              <span className="te-chip-label">active events</span>
            </div>

            <a
              href="/"
              className="btn ghost"
              title="Return to Landing Page"
              aria-label="Home"
            >
              <svg
                className="w-3.5 h-3.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
                <polyline points="9 22 9 12 15 12 15 22" />
              </svg>
              <span className="te-btn-label">Home</span>
            </a>

            <button
              type="button"
              onClick={fetchActiveEvents}
              disabled={loading}
              className="btn"
              title="Refresh active thermal events"
              aria-label="Refresh"
            >
              <svg
                className={`w-3.5 h-3.5 ${loading ? "te-spin" : ""}`}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
                <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
                <path d="M16 21h5v-5" />
              </svg>
              <span className="te-btn-label">Refresh</span>
            </button>

            <button
              type="button"
              onClick={() => setIsAnalyzeOpen(true)}
              className="btn"
              title="Upload and analyze satellite image sample"
              aria-label="Analyze Sample"
            >
              <svg
                className="w-3.5 h-3.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                <line x1="12" y1="22.08" x2="12" y2="12" />
              </svg>
              <span className="te-btn-label">Analyze Sample</span>
            </button>

            <button
              type="button"
              onClick={toggleTheme}
              className="tgl"
              aria-label="Switch colour theme"
              title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            >
              {theme === "dark" ? (
                <svg
                  className="w-3.5 h-3.5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle cx="12" cy="12" r="4" />
                  <path d="M12 2v2" />
                  <path d="M12 20v2" />
                  <path d="m4.93 4.93 1.41 1.41" />
                  <path d="m17.66 17.66 1.41 1.41" />
                  <path d="M2 12h2" />
                  <path d="M20 12h2" />
                  <path d="m6.34 17.66-1.41 1.41" />
                  <path d="m19.07 4.93-1.41 1.41" />
                </svg>
              ) : (
                <svg
                  className="w-3.5 h-3.5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
                </svg>
              )}
              <span className="te-btn-label">
                {theme === "dark" ? "Light mode" : "Dark mode"}
              </span>
            </button>
          </div>
        </div>
      </header>

      {/* Location Picking Crosshair Floating HUD */}
      {isPickingLocation && (
        <div className="fixed top-20 left-1/2 -translate-x-1/2 z-40 flex items-center gap-3 px-4 py-2.5 rounded-full bg-amber-500 text-slate-950 font-semibold text-xs shadow-2xl border border-amber-300 backdrop-blur-md">
          <svg className="w-4 h-4 animate-pulse" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <circle cx="12" cy="12" r="10" />
            <line x1="22" y1="12" x2="18" y2="12" />
            <line x1="6" y1="12" x2="2" y2="12" />
            <line x1="12" y1="6" x2="12" y2="2" />
            <line x1="12" y1="22" x2="12" y2="18" />
          </svg>
          <span>Crosshair Active: Click anywhere on the 3D map to capture coordinates</span>
          <button
            type="button"
            onClick={() => {
              setIsPickingLocation(false);
              setIsAnalyzeOpen(true);
            }}
            className="ml-2 px-2.5 py-1 rounded-full bg-slate-950/20 hover:bg-slate-950/30 text-[11px] font-bold"
          >
            Cancel
          </button>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Deck.gl 3D Geospatial Map Canvas */}
      {/* ------------------------------------------------------------------ */}
      <div className="te-map-shell te-enter-map">
        <DeckGL
          viewState={viewState}
          onViewStateChange={(e: any) => setViewState(e.viewState)}
          controller={true}
          layers={layers}
          effects={[lightingEffect]}
          getTooltip={getTooltip}
          onClick={(info: any) => {
            if (isPickingLocation && info?.coordinate) {
              const [lon, lat] = info.coordinate;
              setPickedCoordinates({ lat, lon });
              setIsPickingLocation(false);
              setIsAnalyzeOpen(true);
            }
          }}
          style={{ position: "absolute", inset: "0px" }}
        >
          <Map mapStyle={customMapStyle} attributionControl={false} />
        </DeckGL>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Bottom HUD Cards: Priority Scale Legend & Official Boundary Badge */}
      {/* ------------------------------------------------------------------ */}
      <div className="te-hud-bottom">
        {/* HUD: Priority Threat Scale Legend (Bottom-Left) */}
        <section
          className="te-card te-legend-card te-enter-panel-1"
          aria-label="Priority Threat Scale Legend"
        >
          <h2 className="te-card-title">Priority Threat Scale</h2>
          <div className="te-scale-bar" aria-hidden="true" />
          <div className="te-scale-ticks">
            <span>0 (Low)</span>
            <span>50</span>
            <span>100 (Max)</span>
          </div>
          <div className="te-scale-note">Column height = FRP (MW)</div>
        </section>

        {/* HUD: Official Sovereign Boundary Demarcation (Bottom-Right) */}
        <section
          className="te-card te-boundary-card te-enter-panel-2"
          aria-label="Survey of India Official Boundary"
        >
          <span className="te-dot te-dot-cool" aria-hidden="true" />
          <div>
            <div className="te-boundary-title">SURVEY OF INDIA OFFICIAL BOUNDARY</div>
            <div className="te-boundary-sub">
              Sovereign territorial integrity · J&amp;K and Ladakh complete demarcation
            </div>
          </div>
        </section>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Active Event Dossier Drawer Panel */}
      {/* ------------------------------------------------------------------ */}
      {selectedEventId && (
        <EventDossierPanel
          eventId={selectedEventId}
          apiBase={API_BASE}
          onClose={() => setSelectedEventId(null)}
        />
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Sample Analyzer & Geolocation Modal */}
      {/* ------------------------------------------------------------------ */}
      <SampleAnalyzerModal
        isOpen={isAnalyzeOpen}
        onClose={() => setIsAnalyzeOpen(false)}
        apiBase={API_BASE}
        isPickingLocation={isPickingLocation}
        setIsPickingLocation={setIsPickingLocation}
        pickedCoordinates={pickedCoordinates}
        onClearPickedCoordinates={() => setPickedCoordinates(null)}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Mount
// ---------------------------------------------------------------------------
const rootEl = document.getElementById("root");
if (rootEl) {
  createRoot(rootEl).render(<App />);
}
