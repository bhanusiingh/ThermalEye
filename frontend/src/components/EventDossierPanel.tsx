import React, { useState, useEffect, useCallback } from "react";

// ---------------------------------------------------------------------------
// TypeScript Interfaces
// ---------------------------------------------------------------------------
export interface PriorityReasons {
  [key: string]: unknown;
}

export interface RawDetectionProperties {
  id: string;
  satellite: string | null;
  acq_datetime: string | null;
  frp: number | null;
  confidence: string | null;
  brightness: number | null;
  bright_t31: number | null;
  scan: number | null;
  track: number | null;
  daynight: string | null;
}

export interface EventProperties {
  id: string;
  first_seen: string;
  last_seen: string;
  max_frp: number | null;
  priority_score: number | null;
  priority_reasons: PriorityReasons | null;
  nearest_osm_type: string | null;
  nearest_osm_name: string | null;
  nearest_osm_distance_km: number | null;
  land_cover_class: string | null;
  land_cover_confidence: number | null;
  ml_disclaimer: string | null;
  imagery_rgb_url: string | null;
  imagery_swir_url: string | null;
  sentinel2_scene_id: string | null;
  detection_count: number | null;
  footprint: GeoJSON.Geometry | null;
  osm_disclaimer: string;
  raw_detections?: {
    type: "FeatureCollection";
    features: Array<{
      type: "Feature";
      geometry: GeoJSON.Point;
      properties: RawDetectionProperties;
    }>;
  };
}

export interface EventDossier {
  type: "Feature";
  geometry: GeoJSON.Point | null;
  properties: EventProperties;
}

interface EventDossierPanelProps {
  eventId: string;
  apiBase: string;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Inline Clean SVGs (Zero Emojis)
// ---------------------------------------------------------------------------
const Icons = {
  Cross: () => (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  ),
  Copy: () => (
    <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  ),
  Check: () => (
    <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  ),
  Flame: () => (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z" />
    </svg>
  ),
  Satellite: () => (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M13 7 9 3 5 7l4 4" />
      <path d="m17 11 4 4-4 4-4-4" />
      <path d="m8 12 4 4 6-6-4-4Z" />
      <path d="m16 8 3-3" />
      <path d="M9 21a6 6 0 0 0-6-6" />
    </svg>
  ),
  Crosshair: () => (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="22" y1="12" x2="18" y2="12" />
      <line x1="6" y1="12" x2="2" y2="12" />
      <line x1="12" y1="6" x2="12" y2="2" />
      <line x1="12" y1="22" x2="12" y2="18" />
    </svg>
  ),
  Building: () => (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="2" width="16" height="20" rx="2" ry="2" />
      <path d="M9 22v-4h6v4" />
      <path d="M8 6h.01" />
      <path d="M16 6h.01" />
      <path d="M12 6h.01" />
      <path d="M12 10h.01" />
      <path d="M12 14h.01" />
      <path d="M16 10h.01" />
      <path d="M16 14h.01" />
      <path d="M8 10h.01" />
      <path d="M8 14h.01" />
    </svg>
  ),
  Cpu: () => (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <line x1="9" y1="1" x2="9" y2="4" />
      <line x1="15" y1="1" x2="15" y2="4" />
      <line x1="9" y1="20" x2="9" y2="23" />
      <line x1="15" y1="20" x2="15" y2="23" />
      <line x1="20" y1="9" x2="23" y2="9" />
      <line x1="20" y1="14" x2="23" y2="14" />
      <line x1="1" y1="9" x2="4" y2="9" />
      <line x1="1" y1="14" x2="4" y2="14" />
    </svg>
  ),
  AlertTriangle: () => (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  ),
  Info: () => (
    <svg className="w-3.5 h-3.5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  ),
  Layers: () => (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </svg>
  ),
};

// ---------------------------------------------------------------------------
// Helper Formatters
// ---------------------------------------------------------------------------
function formatUtcTimestamp(iso: string | null): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toISOString().replace("T", " ").replace(/\.\d+Z$/, " UTC");
  } catch {
    return iso;
  }
}

function parseScoreComponent(raw: unknown, defaultMax: number): { value: number; max: number; note?: string } {
  if (typeof raw !== "string") {
    const num = typeof raw === "number" ? raw : 0;
    return { value: Math.max(0, num), max: defaultMax };
  }
  const match = raw.match(/^([\d.]+)\/([\d.]+)(?:\s*\((.*)\))?/);
  if (match) {
    const val = parseFloat(match[1]);
    const mx = parseFloat(match[2]);
    const note = match[3]?.trim();
    return { value: isNaN(val) ? 0 : val, max: isNaN(mx) ? defaultMax : mx, note };
  }
  const valOnly = parseFloat(raw);
  return { value: isNaN(valOnly) ? 0 : valOnly, max: defaultMax, note: raw };
}

function getThreatClassification(score: number | null): {
  label: string;
  pillClass: string;
  dotClass: string;
  barClass: string;
  accentVar: string;
} {
  if (score == null) {
    return {
      label: "Unscored",
      pillClass: "te-pill te-pill-unk",
      dotClass: "te-dot te-dot-unk",
      barClass: "te-bar-unk",
      accentVar: "var(--unk)",
    };
  }
  if (score >= 67) {
    return {
      label: "Critical threat",
      pillClass: "te-pill te-pill-bad",
      dotClass: "te-dot te-dot-bad",
      barClass: "te-bar-ember",
      accentVar: "var(--bad)",
    };
  }
  if (score >= 34) {
    return {
      label: "Elevated concern",
      pillClass: "te-pill te-pill-warn",
      dotClass: "te-dot te-dot-warn",
      barClass: "te-bar-hot",
      accentVar: "var(--warn)",
    };
  }
  return {
    label: "Low advisory",
    pillClass: "te-pill te-pill-ok",
    dotClass: "te-dot te-dot-ok",
    barClass: "te-bar-ok",
    accentVar: "var(--ok)",
  };
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
export default function EventDossierPanel({ eventId, apiBase, onClose }: EventDossierPanelProps) {
  const [dossier, setDossier] = useState<EventDossier | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copiedCoords, setCopiedCoords] = useState(false);
  const [copiedId, setCopiedId] = useState(false);

  // Keyboard accessibility — close on Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  // Fetch dossier
  useEffect(() => {
    let cancelled = false;
    const fetchDossier = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${apiBase}/api/v1/events/${eventId}`);
        if (!res.ok) {
          throw new Error(`API ${res.status}: ${res.statusText}`);
        }
        const data: EventDossier = await res.json();
        if (!cancelled) setDossier(data);
      } catch (err: unknown) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load dossier.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchDossier();
    return () => {
      cancelled = true;
    };
  }, [eventId, apiBase]);

  const copyToClipboard = useCallback((text: string, type: "coords" | "id") => {
    navigator.clipboard.writeText(text);
    if (type === "coords") {
      setCopiedCoords(true);
      setTimeout(() => setCopiedCoords(false), 2000);
    } else {
      setCopiedId(true);
      setTimeout(() => setCopiedId(false), 2000);
    }
  }, []);

  // Loading Skeleton
  if (loading) {
    return (
      <aside
        role="dialog"
        aria-label="Event Dossier Loading"
        aria-modal="true"
        className="te-drawer"
      >
        <div className="te-drawer-handle" aria-hidden="true" />
        <div className="te-drawer-head">
          <div className="h-5 w-44 te-skeleton" />
          <button
            type="button"
            onClick={onClose}
            aria-label="Close dossier"
            className="te-icon-btn"
          >
            <Icons.Cross />
          </button>
        </div>
        <div className="te-drawer-body">
          <div className="h-32 te-skeleton" />
          <div className="h-44 te-skeleton" />
          <div className="h-40 te-skeleton" />
        </div>
      </aside>
    );
  }

  // Error State
  if (error || !dossier) {
    return (
      <aside
        role="dialog"
        aria-label="Event Dossier Error"
        aria-modal="true"
        className="te-drawer"
      >
        <div className="te-drawer-handle" aria-hidden="true" />
        <div className="te-drawer-head">
          <div className="flex items-center gap-2" style={{ color: "var(--bad)" }}>
            <Icons.AlertTriangle />
            <h2 className="te-card-title" style={{ color: "var(--bad)" }}>
              Dossier unavailable
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close dossier"
            className="te-icon-btn"
          >
            <Icons.Cross />
          </button>
        </div>
        <div className="te-drawer-body items-center justify-center text-center">
          <div
            className="rounded-full p-3.5 mb-2"
            style={{
              background: "color-mix(in srgb, var(--bad) 14%, var(--bg2))",
              border: "1px solid color-mix(in srgb, var(--bad) 40%, var(--line))",
              color: "var(--bad)",
            }}
          >
            <Icons.AlertTriangle />
          </div>
          <h3 className="te-card-title text-lg">Unable to load intelligence dossier</h3>
          <p className="text-sm te-muted max-w-sm">
            {error ?? "Event record not found or database unreachable."}
          </p>
          <button type="button" onClick={onClose} className="btn ghost mt-2">
            Dismiss
          </button>
        </div>
      </aside>
    );
  }

  const p = dossier.properties;
  const coords = dossier.geometry?.coordinates;
  const lon = coords ? coords[0] : null;
  const lat = coords ? coords[1] : null;
  const coordsStr = lat != null && lon != null ? `${lat.toFixed(5)}° N, ${lon.toFixed(5)}° E` : null;

  const threat = getThreatClassification(p.priority_score);
  const detections = p.raw_detections?.features ?? [];

  // Parse breakdown
  const reasons = p.priority_reasons || {};
  const frpComponent = parseScoreComponent(reasons["Thermal Intensity (FRP)"], 40);
  const persistenceComponent = parseScoreComponent(reasons["Persistence"], 30);
  const infraComponent = parseScoreComponent(reasons["Infrastructure Proximity"], 30);

  return (
    <aside
      role="dialog"
      aria-label={`Event Dossier ${eventId}`}
      aria-modal="true"
      className="te-drawer"
    >
      <div className="te-drawer-handle" aria-hidden="true" />

      {/* ------------------------------------------------------------------ */}
      {/* Dossier Header Bar */}
      {/* ------------------------------------------------------------------ */}
      <header className="te-drawer-head">
        <div className="flex items-center gap-3 min-w-0">
          <span className="te-brand-mark" aria-hidden="true">
            <Icons.Flame />
          </span>
          <div className="min-w-0">
            <h2 className="te-card-title text-lg">Thermal Anomaly Dossier</h2>
            <div className="flex items-center gap-1.5 mt-0.5 text-xs">
              <span className="te-num font-semibold" style={{ color: "var(--hot)" }}>
                Event #{eventId.substring(0, 8).toUpperCase()}
              </span>
              <button
                type="button"
                onClick={() => copyToClipboard(eventId, "id")}
                title="Copy full UUID"
                aria-label="Copy full event UUID"
                className="te-icon-btn"
                style={{ padding: "3px", borderColor: "transparent" }}
              >
                {copiedId ? <Icons.Check /> : <Icons.Copy />}
              </button>
            </div>
          </div>
        </div>

        <button
          type="button"
          onClick={onClose}
          aria-label="Close dossier"
          className="te-icon-btn"
        >
          <Icons.Cross />
        </button>
      </header>

      {/* ------------------------------------------------------------------ */}
      {/* Modular Card Content Body */}
      {/* ------------------------------------------------------------------ */}
      <div className="te-drawer-body">
        {/* Module 1: Geolocation & Cluster Telemetry */}
        <section className="te-card">
          <div className="te-section-head">
            <div className="te-section-label">
              <Icons.Crosshair />
              <h3>Target coordinates &amp; cluster</h3>
            </div>
            <span className="te-pill">
              {p.detection_count ?? detections.length} detections
            </span>
          </div>

          <dl className="te-dl">
            <dt>WGS84 centroid</dt>
            <dd>
              {coordsStr ? (
                <span className="inline-flex items-center gap-1.5">
                  <span>{coordsStr}</span>
                  <button
                    type="button"
                    onClick={() => copyToClipboard(`${lat?.toFixed(5)}, ${lon?.toFixed(5)}`, "coords")}
                    title="Copy coordinates"
                    aria-label="Copy coordinates"
                    className="te-icon-btn"
                    style={{ padding: "3px" }}
                  >
                    {copiedCoords ? <Icons.Check /> : <Icons.Copy />}
                  </button>
                </span>
              ) : (
                <span style={{ color: "var(--warn)" }}>Data unavailable</span>
              )}
            </dd>

            <dt>Max thermal intensity</dt>
            <dd style={{ color: "var(--hot)" }}>
              {p.max_frp != null ? `${p.max_frp.toFixed(1)} MW (FRP)` : "—"}
            </dd>

            <dt>First seen</dt>
            <dd>{formatUtcTimestamp(p.first_seen)}</dd>

            <dt>Last seen</dt>
            <dd>{formatUtcTimestamp(p.last_seen)}</dd>
          </dl>
        </section>

        {/* Module 2: Priority Score Breakdown */}
        <section className="te-card">
          <div className="te-section-head">
            <div className="te-section-label">
              <Icons.Flame />
              <h3>Priority score breakdown</h3>
            </div>
            <span className={threat.pillClass}>
              <span className={threat.dotClass} />
              <span>{threat.label}</span>
            </span>
          </div>

          <div className="te-subcard flex items-center gap-4 mb-3">
            <div
              className="flex flex-col items-center justify-center w-16 h-16 rounded-lg shrink-0"
              style={{
                background: `color-mix(in srgb, ${threat.accentVar} 16%, var(--bg))`,
                border: `1px solid color-mix(in srgb, ${threat.accentVar} 42%, var(--line))`,
              }}
            >
              <span className="te-big-num text-2xl" style={{ color: threat.accentVar }}>
                {p.priority_score != null ? Math.round(p.priority_score) : "—"}
              </span>
              <span className="text-[11px] te-muted">/ 100</span>
            </div>
            <div className="flex-1 space-y-2 min-w-0">
              <dl className="te-dl">
                <dt>Investigation threat index</dt>
                <dd>{p.priority_score != null ? `${p.priority_score.toFixed(1)}%` : "0.0%"}</dd>
              </dl>
              <div className="te-bar-track">
                <div
                  className={`te-bar-fill ${threat.barClass}`}
                  style={{ width: `${Math.min(100, Math.max(0, p.priority_score ?? 0))}%` }}
                />
              </div>
            </div>
          </div>

          {/* Component Progress Bars */}
          <div className="space-y-2.5">
            {/* Component A: Thermal Intensity (FRP) */}
            <div className="te-subcard space-y-2">
              <dl className="te-dl">
                <dt>Thermal intensity (FRP)</dt>
                <dd>
                  {frpComponent.value.toFixed(1)} / {frpComponent.max}
                </dd>
              </dl>
              <div className="te-bar-track">
                <div
                  className="te-bar-fill te-bar-ember"
                  style={{ width: `${Math.min(100, (frpComponent.value / frpComponent.max) * 100)}%` }}
                />
              </div>
            </div>

            {/* Component B: Temporal Persistence */}
            <div className="te-subcard space-y-2">
              <dl className="te-dl">
                <dt>Temporal persistence</dt>
                <dd>
                  {persistenceComponent.value.toFixed(0)} / {persistenceComponent.max}
                </dd>
              </dl>
              <div className="te-bar-track">
                <div
                  className="te-bar-fill te-bar-warn"
                  style={{ width: `${Math.min(100, (persistenceComponent.value / persistenceComponent.max) * 100)}%` }}
                />
              </div>
            </div>

            {/* Component C: Infrastructure Proximity */}
            <div className="te-subcard space-y-2">
              <dl className="te-dl">
                <dt>Infrastructure proximity</dt>
                <dd>
                  {infraComponent.value.toFixed(0)} / {infraComponent.max}
                </dd>
              </dl>
              <div className="te-bar-track">
                <div
                  className="te-bar-fill te-bar-cool"
                  style={{ width: `${Math.min(100, (infraComponent.value / infraComponent.max) * 100)}%` }}
                />
              </div>
              {infraComponent.note && (
                <div className="flex items-center gap-1.5 pt-1 text-xs" style={{ color: "var(--warn)" }}>
                  <Icons.Info />
                  <span>{infraComponent.note}</span>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Module 3: Sentinel-2 Optical & SWIR Imagery */}
        <section className="te-card">
          <div className="te-section-head">
            <div className="te-section-label">
              <Icons.Satellite />
              <h3>Sentinel-2 L2A imagery</h3>
            </div>
            {p.sentinel2_scene_id ? (
              <span
                className="text-xs te-muted te-num truncate max-w-[190px]"
                title={p.sentinel2_scene_id}
              >
                {p.sentinel2_scene_id}
              </span>
            ) : null}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
            {/* True Color RGB Card */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs te-muted">
                <span>True color (RGB)</span>
                <span className="te-num">B04 · B03 · B02</span>
              </div>
              <div className="te-img-frame">
                {p.imagery_rgb_url ? (
                  <img
                    src={`${apiBase}${p.imagery_rgb_url}`}
                    alt="Sentinel-2 True Color RGB Chip"
                    loading="lazy"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="te-notice is-warn m-3 flex-col items-center justify-center text-center h-[calc(100%-24px)] w-[calc(100%-24px)]">
                    <Icons.AlertTriangle />
                    <span>Data unavailable — Optical/SWIR chip not acquired</span>
                  </div>
                )}
              </div>
            </div>

            {/* SWIR False Color Card */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs te-muted">
                <span>SWIR thermal penetration</span>
                <span className="te-num">B12 · B11 · B8A</span>
              </div>
              <div className="te-img-frame">
                {p.imagery_swir_url ? (
                  <img
                    src={`${apiBase}${p.imagery_swir_url}`}
                    alt="Sentinel-2 SWIR Spectral Chip"
                    loading="lazy"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="te-notice is-warn m-3 flex-col items-center justify-center text-center h-[calc(100%-24px)] w-[calc(100%-24px)]">
                    <Icons.AlertTriangle />
                    <span>Data unavailable — Optical/SWIR chip not acquired</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="te-notice mt-3">
            <Icons.Info />
            <span>Target chips sampled at 10m Ground Sample Distance (GSD). Cloud exclusion threshold: 20%.</span>
          </div>
        </section>

        {/* Module 4: ML Land-Cover Context */}
        <section className="te-card">
          <div className="te-section-head">
            <div className="te-section-label">
              <Icons.Cpu />
              <h3>ML land-cover context</h3>
            </div>
            {p.land_cover_class ? (
              <span className="te-pill te-pill-ok">
                <span className="te-dot te-dot-ok" />
                <span>Classified</span>
              </span>
            ) : null}
          </div>

          {p.land_cover_class ? (
            <div className="te-subcard mb-3">
              <dl className="te-dl">
                <dt>Predicted terrain class</dt>
                <dd>{p.land_cover_class}</dd>

                <dt>Model confidence</dt>
                <dd style={{ color: "var(--ok)" }}>
                  {p.land_cover_confidence != null
                    ? `${(p.land_cover_confidence * 100).toFixed(1)}%`
                    : "—"}
                </dd>
              </dl>
            </div>
          ) : (
            <div className="te-notice is-warn mb-3">
              <Icons.AlertTriangle />
              <span>Data unavailable — Land-cover inference not computed for this event.</span>
            </div>
          )}

          {/* Mandatory ML Disclaimer Banner */}
          <div className="te-notice">
            <Icons.Info />
            <span>
              {p.ml_disclaimer ||
                "Model provides land-cover context only, not facility identification or fire confirmation."}
            </span>
          </div>
        </section>

        {/* Module 5: Infrastructure Proximity */}
        <section className="te-card">
          <div className="te-section-head">
            <div className="te-section-label">
              <Icons.Building />
              <h3>Nearest OpenStreetMap infrastructure</h3>
            </div>
          </div>

          {p.nearest_osm_type || p.nearest_osm_name ? (
            <div className="te-subcard mb-3">
              <dl className="te-dl">
                <dt>Facility name</dt>
                <dd className="truncate max-w-[240px]">
                  {p.nearest_osm_name || "Unnamed infrastructure"}
                </dd>

                <dt>Facility category</dt>
                <dd style={{ color: "var(--cool)" }}>
                  {p.nearest_osm_type || "General industrial"}
                </dd>

                {p.nearest_osm_distance_km != null && (
                  <>
                    <dt>Proximity distance</dt>
                    <dd>{p.nearest_osm_distance_km.toFixed(2)} km</dd>
                  </>
                )}
              </dl>
            </div>
          ) : (
            <div className="te-notice is-warn mb-3 justify-between items-center">
              <div className="flex items-start gap-2">
                <Icons.AlertTriangle />
                <span>Data unavailable — No facility within 3,000m or API rate-limited</span>
              </div>
              <span className="te-pill te-pill-unk shrink-0">Unavailable</span>
            </div>
          )}

          <div className="te-notice">
            <Icons.Info />
            <span>{p.osm_disclaimer || "OSM coverage limitations apply"}</span>
          </div>
        </section>

        {/* Module 6: Raw Detections Cluster */}
        <section className="te-card">
          <div className="te-section-head">
            <div className="te-section-label">
              <Icons.Layers />
              <h3>Linked FIRMS detections ({detections.length})</h3>
            </div>
          </div>

          {detections.length === 0 ? (
            <div className="te-notice is-warn">
              <Icons.AlertTriangle />
              <span>No linked raw detections found.</span>
            </div>
          ) : (
            <div
              className="overflow-x-auto rounded-lg"
              style={{ border: "1px solid var(--line)" }}
            >
              <table className="te-table">
                <thead>
                  <tr>
                    <th>Acquisition (UTC)</th>
                    <th>Sensor</th>
                    <th style={{ textAlign: "right" }}>FRP (MW)</th>
                    <th style={{ textAlign: "center" }}>Conf.</th>
                    <th style={{ textAlign: "center" }}>D/N</th>
                  </tr>
                </thead>
                <tbody>
                  {detections.slice(0, 50).map((det, i) => {
                    const dp = det.properties;
                    return (
                      <tr key={dp.id ?? i}>
                        <td>
                          {dp.acq_datetime
                            ? dp.acq_datetime.slice(0, 16).replace("T", " ")
                            : "—"}
                        </td>
                        <td className="te-muted">{dp.satellite ?? "—"}</td>
                        <td
                          style={{
                            textAlign: "right",
                            fontWeight: 600,
                            color: "var(--hot)",
                          }}
                        >
                          {dp.frp != null ? dp.frp.toFixed(1) : "—"}
                        </td>
                        <td className="te-muted" style={{ textAlign: "center" }}>
                          {dp.confidence ?? "—"}
                        </td>
                        <td className="te-muted" style={{ textAlign: "center" }}>
                          {dp.daynight ?? "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {detections.length > 50 && (
                <div
                  className="px-3 py-2 text-xs te-muted"
                  style={{
                    background: "var(--bg2)",
                    borderTop: "1px solid var(--line)",
                  }}
                >
                  Displaying first 50 of {detections.length} detections in cluster.
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </aside>
  );
}
