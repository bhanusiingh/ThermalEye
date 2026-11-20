import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { createRoot } from "react-dom/client";
import { Map } from "react-map-gl/maplibre";
import DeckGL from "@deck.gl/react";
import { GeoJsonLayer, ColumnLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import { LightingEffect, AmbientLight, DirectionalLight, FlyToInterpolator } from "@deck.gl/core";
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
    bg: "#0d1a1d",
    bg2: "#12232a",
    landSubtle: "#162628",
    depth: "#0d1a1d",
    line: "#243b3f",
    road: "#243b3f",
    tx: "#e8edeb",
    mut: "#9ab0b3",
    hotRgb: [224, 164, 88],
    emberRgb: [208, 105, 76],
    coolRgb: [79, 163, 165],
    unkRgb: [111, 132, 134],
  },
  light: {
    bg: "#f2ece1",
    bg2: "#fbfcfb",
    landSubtle: "#f2ece1",
    depth: "#cfe0df",
    line: "#d4cbb9",
    road: "#d4cbb9",
    tx: "#23383b",
    mut: "#566c71",
    hotRgb: [184, 111, 28],
    emberRgb: [173, 79, 55],
    coolRgb: [43, 110, 114],
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
// Apply Dynamic MapLibre Basemap Styling & Sovereign Exclusion
// ---------------------------------------------------------------------------
function applyBasemapTheme(map: any, theme: ThemeMode) {
  if (!map || !map.isStyleLoaded || !map.isStyleLoaded()) return;

  const style = map.getStyle();
  if (!style || !Array.isArray(style.layers)) return;

  const isDark = theme === "dark";

  style.layers.forEach((layer: any) => {
    const id: string = layer.id || "";
    const type: string = layer.type || "";

    // Hide boundary layers that misrepresent Indian sovereignty
    if (
      id === "boundary_country_inner" ||
      id === "boundary_country_outline" ||
      id === "boundary_state" ||
      id === "boundary_county"
    ) {
      if (map.getLayer(id)) {
        map.setLayoutProperty(id, "visibility", "none");
      }
      return;
    }

    if (type === "background") {
      if (map.getLayer(id)) {
        map.setPaintProperty(id, "background-color", isDark ? "#0d1a1d" : "#cfe0df");
      }
      return;
    }

    if (id === "water" || id === "water_shadow") {
      if (map.getLayer(id)) {
        map.setPaintProperty(id, "fill-color", isDark ? "#0d1a1d" : "#cfe0df");
        map.setPaintProperty(id, "fill-opacity", 1.0);
      }
      return;
    }

    if (id === "waterway") {
      if (map.getLayer(id)) {
        map.setPaintProperty(id, "line-color", isDark ? "#0d1a1d" : "#b9d2d1");
      }
      return;
    }

    if (
      id === "landcover" ||
      id.startsWith("park_") ||
      id.startsWith("landuse") ||
      id === "land"
    ) {
      if (map.getLayer(id)) {
        let color = isDark ? "#162628" : "#f2ece1";
        if (!isDark) {
          if (id.startsWith("park_") || id.includes("green") || id.includes("wood")) {
            color = "#e1e9e2";
          } else if (
            id.includes("urban") ||
            id.includes("residential") ||
            id.includes("commercial")
          ) {
            color = "#eae2d4";
          }
        } else {
          if (id.includes("secondary") || id.includes("landuse")) {
            color = "#1b2e31";
          }
        }
        map.setPaintProperty(id, "fill-color", color);
      }
      return;
    }

    if (id === "building" || id === "building-top") {
      if (map.getLayer(id)) {
        map.setPaintProperty(id, "fill-color", isDark ? "#162628" : "#eae2d4");
        if (layer.paint && "fill-outline-color" in layer.paint) {
          map.setPaintProperty(id, "fill-outline-color", isDark ? "#243b3f" : "#d4cbb9");
        }
      }
      return;
    }

    if (
      id.startsWith("road_") ||
      id.startsWith("tunnel_") ||
      id.startsWith("bridge_") ||
      id.startsWith("aeroway") ||
      id.startsWith("boundary_")
    ) {
      if (map.getLayer(id)) {
        const roadColor = isDark ? "#243b3f" : "#d4cbb9";
        if (type === "line") {
          map.setPaintProperty(id, "line-color", roadColor);
          map.setPaintProperty(id, "line-opacity", 0.45);
        }
      }
      return;
    }

    if (type === "symbol") {
      if (map.getLayer(id)) {
        const textColor = isDark ? "#9ab0b3" : "#23383b";
        const haloColor = isDark ? "#0d1a1d" : "#f2ece1";
        map.setPaintProperty(id, "text-color", textColor);
        map.setPaintProperty(id, "text-halo-color", haloColor);
        map.setPaintProperty(id, "text-halo-width", 1.8);

        // Exclude India and Bangladesh across all common CARTO/OpenMapTiles name properties
        const textField = map.getLayoutProperty(id, "text-field");
        const textFieldStr = JSON.stringify(textField || "");
        const isCountryOrPlace = /country|place/i.test(id) || textFieldStr.includes("name");

        if (isCountryOrPlace) {
          const currentFilter = map.getFilter(id);
          const filterStr = JSON.stringify(currentFilter || "");
          if (!filterStr.includes('"india"')) {
            const excludeExpr: any = [
              "!",
              [
                "in",
                [
                  "downcase",
                  [
                    "coalesce",
                    ["get", "name_en"],
                    ["get", "name:en"],
                    ["get", "name_int"],
                    ["get", "name"],
                    "",
                  ],
                ],
                ["literal", ["india", "bangladesh"]],
              ],
            ];
            const newFilter = currentFilter
              ? ["all", currentFilter, excludeExpr]
              : excludeExpr;
            map.setFilter(id, newFilter);
          }
        }
      }
      return;
    }
  });
}

// ---------------------------------------------------------------------------
// Initial MapLibre Static Style Sanitizer
// ---------------------------------------------------------------------------
function buildThemedMapStyle(rawStyle: any, theme: ThemeMode) {
  if (!rawStyle || !Array.isArray(rawStyle.layers)) return rawStyle;
  const isDark = theme === "dark";

  const layers = rawStyle.layers.map((layer: any) => {
    const id: string = layer.id || "";
    const type: string = layer.type || "";
    const paint = { ...(layer.paint || {}) };

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

    if (type === "background") {
      paint["background-color"] = isDark ? "#0d1a1d" : "#cfe0df";
      return { ...layer, paint };
    }

    if (id === "water" || id === "water_shadow") {
      paint["fill-color"] = isDark ? "#0d1a1d" : "#cfe0df";
      paint["fill-opacity"] = 1.0;
      return { ...layer, paint };
    }

    if (id === "waterway") {
      paint["line-color"] = isDark ? "#0d1a1d" : "#b9d2d1";
      return { ...layer, paint };
    }

    if (
      id === "landcover" ||
      id.startsWith("park_") ||
      id.startsWith("landuse") ||
      id === "land"
    ) {
      let color = isDark ? "#162628" : "#f2ece1";
      if (!isDark) {
        if (id.startsWith("park_") || id.includes("green") || id.includes("wood")) {
          color = "#e1e9e2";
        } else if (
          id.includes("urban") ||
          id.includes("residential") ||
          id.includes("commercial")
        ) {
          color = "#eae2d4";
        }
      } else {
        if (id.includes("secondary") || id.includes("landuse")) {
          color = "#1b2e31";
        }
      }
      paint["fill-color"] = color;
      return { ...layer, paint };
    }

    if (id === "building" || id === "building-top") {
      paint["fill-color"] = isDark ? "#162628" : "#eae2d4";
      if ("fill-outline-color" in paint) {
        paint["fill-outline-color"] = isDark ? "#243b3f" : "#d4cbb9";
      }
      return { ...layer, paint };
    }

    if (
      id.startsWith("road_") ||
      id.startsWith("tunnel_") ||
      id.startsWith("bridge_") ||
      id.startsWith("aeroway") ||
      id.startsWith("boundary_")
    ) {
      if (type === "line") {
        paint["line-color"] = isDark ? "#243b3f" : "#d4cbb9";
        paint["line-opacity"] = 0.45;
      }
      return { ...layer, paint };
    }

    if (type === "symbol") {
      paint["text-color"] = isDark ? "#9ab0b3" : "#23383b";
      paint["text-halo-color"] = isDark ? "#0d1a1d" : "#f2ece1";
      paint["text-halo-width"] = 1.8;
      let filter = layer.filter;
      const textFieldStr = JSON.stringify(layer.layout?.["text-field"] || "");
      const isCountryOrPlace = /country|place/i.test(id) || textFieldStr.includes("name");
      if (isCountryOrPlace) {
        const filterStr = JSON.stringify(filter || "");
        if (!filterStr.includes('"india"')) {
          const excludeExpr: any = [
            "!",
            [
              "in",
              [
                "downcase",
                [
                  "coalesce",
                  ["get", "name_en"],
                  ["get", "name:en"],
                  ["get", "name_int"],
                  ["get", "name"],
                  "",
                ],
              ],
              ["literal", ["india", "bangladesh"]],
            ],
          ];
          filter = filter ? ["all", filter, excludeExpr] : excludeExpr;
        }
      }
      return { ...layer, paint, filter };
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
  const [pulseProgress, setPulseProgress] = useState(0);

  // ---- Selected Event Geometry Resolution ----------------------------------
  const selectedEvent = useMemo(() => {
    if (!selectedEventId || !eventsData?.features) return null;
    const match = eventsData.features.find((f) => f.properties.id === selectedEventId);
    if (!match) return null;
    let lon = 0;
    let lat = 0;
    if (match.geometry?.type === "Point") {
      [lon, lat] = match.geometry.coordinates;
    } else if (match.properties.footprint?.type === "Polygon") {
      const ring = (match.properties.footprint as any).coordinates[0];
      lon = ring.reduce((s: number, p: number[]) => s + p[0], 0) / ring.length;
      lat = ring.reduce((s: number, p: number[]) => s + p[1], 0) / ring.length;
    } else {
      return null;
    }
    return {
      id: match.properties.id,
      longitude: lon,
      latitude: lat,
      frp: match.properties.max_frp,
      priorityScore: match.properties.priority_score,
    };
  }, [selectedEventId, eventsData]);

  // ---- Smooth Camera Pan (Locate & Select) ---------------------------------
  const focusOnEvent = useCallback(
    (event: { longitude: number; latitude: number } | null) => {
      if (!event) return;
      const prefersReducedMotion =
        typeof window !== "undefined" &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      setViewState((prev: any) => {
        const targetZoom = Math.max(prev.zoom || 4.5, 7.5);
        // Slightly offset center eastward on desktop so event is not obscured by the right drawer
        const isDesktop = typeof window !== "undefined" && window.innerWidth >= 900;
        const lonOffset = isDesktop ? (200 * 360) / (256 * Math.pow(2, targetZoom)) : 0;
        const latOffset = !isDesktop ? (-100 * 180) / (256 * Math.pow(2, targetZoom)) : 0;
        return {
          ...prev,
          longitude: event.longitude + lonOffset,
          latitude: event.latitude + latOffset,
          zoom: targetZoom,
          pitch: 45,
          transitionDuration: prefersReducedMotion ? 0 : 650,
          transitionInterpolator: new FlyToInterpolator(),
        };
      });
    },
    []
  );

  // ---- Radar Pulse Halo Animation Loop -------------------------------------
  useEffect(() => {
    if (!selectedEventId) {
      setPulseProgress(0);
      return;
    }
    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced) {
      setPulseProgress(0.5);
      return;
    }

    let animId: number;
    const duration = 1200; // ms
    let start: number | null = null;

    const frame = (time: number) => {
      if (start === null) start = time;
      const elapsed = time - start;
      const progress = (elapsed % duration) / duration;
      setPulseProgress(progress);
      animId = requestAnimationFrame(frame);
    };

    animId = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(animId);
  }, [selectedEventId]);

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
      focusOnEvent({ longitude: lon, latitude: lat });
      setDeepLinkEventCentered(true);
    }
  }, [selectedEventId, deepLinkEventCentered, eventsData, focusOnEvent]);

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

  const mapRef = useRef<any>(null);

  const handleMapLoad = useCallback(
    (e: any) => {
      applyBasemapTheme(e.target, theme);
    },
    [theme]
  );

  const handleMapStyleData = useCallback(
    (e: any) => {
      applyBasemapTheme(e.target, theme);
    },
    [theme]
  );

  useEffect(() => {
    if (mapRef.current) {
      const map = mapRef.current.getMap ? mapRef.current.getMap() : mapRef.current;
      if (map && map.isStyleLoaded && map.isStyleLoaded()) {
        applyBasemapTheme(map, theme);
      }
    }
  }, [theme]);

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
  const handleClick = useCallback(
    (info: any) => {
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
          const pos = info.object.position;
          if (pos && Array.isArray(pos) && pos.length >= 2) {
            focusOnEvent({ longitude: pos[0], latitude: pos[1] });
          } else if (info.coordinate) {
            focusOnEvent({ longitude: info.coordinate[0], latitude: info.coordinate[1] });
          }
        }
      }
    },
    [isPickingLocation, focusOnEvent]
  );

  // ---- Deck.gl layers -----------------------------------------------------
  const layers = useMemo(() => {
    const result: any[] = [];
    const pal = MAP_PALETTES[theme];

    // 0) Official Survey of India Sovereign Boundary Layers
    if (indiaBoundary) {
      // Subtle outer edge
      result.push(
        new GeoJsonLayer({
          id: "official-india-boundary-glow",
          data: indiaBoundary,
          pickable: false,
          stroked: true,
          filled: false,
          lineWidthMinPixels: theme === "dark" ? 3.5 : 3.0,
          getLineWidth: theme === "dark" ? 3.5 : 3.0,
          getLineColor: theme === "dark" ? [43, 110, 114, 70] : [43, 110, 114, 40],
          updateTriggers: {
            getLineColor: [theme],
          },
        })
      );
      // Crisp primary stroke in muted sage-teal (dark) / deep slate-teal (light) + subtle interior tint
      result.push(
        new GeoJsonLayer({
          id: "official-india-boundary-core",
          data: indiaBoundary,
          pickable: false,
          stroked: true,
          filled: true,
          lineWidthMinPixels: theme === "dark" ? 1.6 : 1.7,
          getLineWidth: theme === "dark" ? 1.6 : 1.7,
          getLineColor: theme === "dark" ? [79, 163, 165, 220] : [43, 110, 114, 215],
          getFillColor: theme === "dark" ? [43, 110, 114, 34] : [78, 135, 128, 20],
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
            const isDark = theme === "dark";
            const score = f.properties?.priority_score;
            const base = priorityToColor(score, theme);
            if (selectedEventId !== null) {
              if (f.properties?.id === selectedEventId) {
                return isDark ? [253, 224, 140, 255] : [184, 111, 28, 255];
              }
              const dimAlpha = isDark ? 48 : 42;
              return [base[0], base[1], base[2], dimAlpha];
            }
            const defaultAlpha = isDark ? 125 : 110;
            return [base[0], base[1], base[2], defaultAlpha];
          },
          getLineWidth: 1.5,
          onClick: handleClick,
          updateTriggers: {
            getLineColor: [theme],
            getFillColor: [footprintData, theme, selectedEventId],
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
          autoHighlight: true,
          highlightColor: [248, 250, 252, 180],
          diskResolution: 12,
          radius: 8000,
          extruded: true,
          elevationScale: 200,
          getPosition: (d: any) => d.position,
          getElevation: (d: any) => {
            const baseHeight = ((d.frp || 1) / maxFrp) * 500;
            if (selectedEventId && d.id === selectedEventId) {
              return Math.max(baseHeight * 1.25, 45);
            }
            return baseHeight;
          },
          getFillColor: (d: any) => {
            const isDark = theme === "dark";
            if (selectedEventId !== null) {
              if (d.id === selectedEventId) {
                return isDark ? [253, 224, 140, 255] : [184, 111, 28, 255];
              }
              const base = priorityToColor(d.priorityScore, theme);
              const dimAlpha = isDark ? 48 : 42;
              return [base[0], base[1], base[2], dimAlpha];
            }
            const base = priorityToColor(d.priorityScore, theme);
            const defaultAlpha = isDark ? 125 : 110;
            return [base[0], base[1], base[2], defaultAlpha];
          },
          onClick: handleClick,
          updateTriggers: {
            getFillColor: [columnData, theme, selectedEventId],
            getElevation: [columnData, selectedEventId],
          },
        })
      );
    }

    // 3) Dedicated Selection Target Layers (Warm Luminous Gold)
    if (selectedEvent) {
      const prefersReduced =
        typeof window !== "undefined" &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      // Concentric Ground Target Ring
      result.push(
        new ScatterplotLayer({
          id: "selected-event-ring",
          data: [selectedEvent],
          pickable: false,
          getPosition: (d: any) => [d.longitude, d.latitude],
          filled: false,
          stroked: true,
          lineWidthUnits: "pixels",
          getLineWidth: 2.5,
          radiusUnits: "pixels",
          getRadius: 22,
          getLineColor: theme === "dark" ? [253, 224, 140, 240] : [184, 111, 28, 240],
          updateTriggers: {
            getPosition: [selectedEvent.id],
            getLineColor: [theme],
          },
        })
      );

      // Pulsing Radar Halo
      const pulseRadius = prefersReduced ? 28 : 16 + (44 - 16) * pulseProgress;
      const pulseAlpha = prefersReduced ? 160 : Math.round(220 * (1 - pulseProgress));
      const goldRgb = theme === "dark" ? [253, 224, 140] : [184, 111, 28];
      result.push(
        new ScatterplotLayer({
          id: "selected-event-pulse",
          data: [selectedEvent],
          pickable: false,
          getPosition: (d: any) => [d.longitude, d.latitude],
          filled: false,
          stroked: true,
          lineWidthUnits: "pixels",
          getLineWidth: 2,
          radiusUnits: "pixels",
          getRadius: pulseRadius,
          getLineColor: [goldRgb[0], goldRgb[1], goldRgb[2], pulseAlpha] as [number, number, number, number],
          updateTriggers: {
            getPosition: [selectedEvent.id],
            getRadius: [pulseRadius],
            getLineColor: [theme, pulseAlpha],
          },
        })
      );

      // Persistent Pinned Event Tag
      result.push(
        new TextLayer({
          id: "selected-event-label",
          data: [selectedEvent],
          pickable: false,
          getPosition: (d: any) => [d.longitude, d.latitude],
          getText: (d: any) => `Event #${String(d.id).substring(0, 8).toUpperCase()}`,
          sizeUnits: "pixels",
          getSize: 12,
          getColor: [248, 250, 252, 255],
          getPixelOffset: [0, -28],
          getTextAnchor: "middle",
          getAlignmentBaseline: "bottom",
          fontFamily: "Inter, system-ui, sans-serif",
          fontWeight: 600,
          background: true,
          getBackgroundColor: [15, 23, 42, 235],
          backgroundBorderRadius: 6,
          backgroundPadding: [8, 4, 8, 4],
          getBorderColor: theme === "dark" ? [253, 224, 140, 255] : [184, 111, 28, 255],
          getBorderWidth: 1.5,
          updateTriggers: {
            getPosition: [selectedEvent.id],
            getText: [selectedEvent.id],
            getBorderColor: [theme],
          },
        })
      );
    }

    // 4) Sovereign Country Labels (Rendered top-most after boundary & ColumnLayer)
    const currentZoom = viewState.zoom || 4.5;
    const zoomFade =
      currentZoom > 5.8
        ? 0
        : currentZoom <= 5.0
        ? 1
        : Math.max(0, (5.8 - currentZoom) / 0.8);

    if (zoomFade > 0.02) {
      const isDark = theme === "dark";
      const textColor: [number, number, number, number] = isDark
        ? [232, 238, 237, Math.round(230 * zoomFade)]
        : [24, 42, 45, Math.round(235 * zoomFade)];
      const outlineColor: [number, number, number, number] = isDark
        ? [13, 26, 29, Math.round(240 * zoomFade)]
        : [242, 236, 225, Math.round(245 * zoomFade)];

      result.push(
        new TextLayer({
          id: "sovereign-country-labels",
          data: [
            {
              text: "I N D I A",
              coordinates: [78.9, 21.2],
              size: 13,
              color: textColor,
              outlineColor: outlineColor,
            },
            {
              text: "BANGLADESH",
              coordinates: [90.45, 23.75],
              size: 10.5,
              color: textColor,
              outlineColor: outlineColor,
            },
          ],
          pickable: false,
          billboard: true,
          parameters: { depthTest: false },
          getPosition: (d: any) => [d.coordinates[0], d.coordinates[1], 0],
          getText: (d: any) => d.text,
          getSize: (d: any) => d.size,
          getColor: (d: any) => d.color,
          getTextAnchor: "middle",
          getAlignmentBaseline: "center",
          fontFamily: '"Public Sans", system-ui, sans-serif',
          fontWeight: 700,
          fontSettings: { sdf: true },
          outlineWidth: 4,
          outlineColor: outlineColor,
          updateTriggers: {
            getColor: [theme, zoomFade],
          },
        })
      );
    }

    return result;
  }, [
    columnData,
    footprintData,
    handleClick,
    indiaBoundary,
    theme,
    selectedEventId,
    selectedEvent,
    pulseProgress,
    viewState.zoom,
  ]);

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
            <a href="/" className="te-logo-link" title="ThermalEye Home">
              <svg
                viewBox="0 0 32 32"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                className="w-[26px] h-[26px] flex-none"
              >
                <circle cx="16" cy="16" r="13" />
                <circle cx="16" cy="16" r="6" stroke="var(--hot)" />
                <circle cx="16" cy="16" r="1.8" fill="var(--hot)" stroke="none" />
              </svg>
              <span className="te-logo-text">ThermalEye</span>
            </a>
          </div>

          <div className="te-nav-actions">
            {/* 1. Active events status pill */}
            <div className="te-chip" aria-label={`${activeCount.toLocaleString()} active events`}>
              <span className="te-chip-count">{activeCount.toLocaleString()}</span>
              <span className="te-chip-label">active events</span>
            </div>

            {/* 2. Home ghost/outline link */}
            <a
              href="/"
              className="btn ghost"
              title="Return to ThermalEye landing page"
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

            {/* 3. Refresh ghost/outline button */}
            <button
              type="button"
              onClick={fetchActiveEvents}
              disabled={loading}
              className="btn ghost"
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

            {/* 3. Primary CTA: Analyze Sample */}
            <button
              type="button"
              onClick={() => setIsAnalyzeOpen(true)}
              className="btn btn-primary"
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

            {/* 4. Pill-outlined theme toggle */}
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
          getCursor={({ isHovering, isDragging }: any) =>
            isDragging ? "grabbing" : isHovering ? "pointer" : "grab"
          }
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
          <Map
            ref={mapRef}
            mapStyle={customMapStyle}
            attributionControl={false}
            onLoad={handleMapLoad}
            onStyleData={handleMapStyleData}
          />
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
          <span className="te-dot te-dot-soi" aria-hidden="true" />
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
          onFocusEvent={() => selectedEvent && focusOnEvent(selectedEvent)}
        />
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Sample Analyzer & Geolocation Modal */}
      {/* ------------------------------------------------------------------ */}
      <SampleAnalyzerModal
        isOpen={isAnalyzeOpen}
        onClose={() => setIsAnalyzeOpen(false)}
        apiBase={API_BASE}
        theme={theme}
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
