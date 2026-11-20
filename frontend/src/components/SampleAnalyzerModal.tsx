import React, { useState, useEffect, useRef, useCallback } from "react";

// ---------------------------------------------------------------------------
// TypeScript Interfaces
// ---------------------------------------------------------------------------
export interface SampleLocationContext {
  latitude: number | null;
  longitude: number | null;
  inside_india: boolean;
  state: string | null;
  nearby_events_5km: number;
  nearest_infrastructure: string | null;
  status?: string;
}

export interface SampleAnalysisResponse {
  sample_id: string;
  predicted_class: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  model_arch: string;
  modality: string;
  inference_latency_ms: number;
  rgb_url: string;
  swir_url: string | null;
  gradcam_url: string | null;
  location: SampleLocationContext;
  disclaimer: string;
}

export interface RecentEventPreset {
  id: string;
  lat: number;
  lon: number;
  max_frp?: number;
  peak_frp?: number;
  priority_score?: number;
  nearest_osm_name?: string | null;
  state?: string | null;
  imagery_rgb_path?: string | null;
  imagery_swir_path?: string | null;
  land_cover_class?: string | null;
}

interface SampleAnalyzerModalProps {
  isOpen: boolean;
  onClose: () => void;
  apiBase: string;
  theme?: "dark" | "light";
  isPickingLocation: boolean;
  setIsPickingLocation: (val: boolean) => void;
  pickedCoordinates: { lat: number; lon: number } | null;
  onClearPickedCoordinates: () => void;
}

interface ClassStyle {
  label: string;
  colorDark: string;
  colorLight: string;
  badgeDark: { bg: string; border: string; text: string };
  badgeLight: { bg: string; border: string; text: string };
  barDark: string;
  barLight: string;
}

const CLASS_CONFIG: Record<string, ClassStyle> = {
  industrial_thermal: {
    label: "Industrial Thermal",
    colorDark: "#e5a246",
    colorLight: "#b86f1c",
    badgeDark: { bg: "rgba(224, 164, 88, 0.15)", border: "rgba(224, 164, 88, 0.4)", text: "#e5a246" },
    badgeLight: { bg: "rgba(184, 111, 28, 0.12)", border: "rgba(184, 111, 28, 0.35)", text: "#b86f1c" },
    barDark: "#e5a246",
    barLight: "#b86f1c",
  },
  agricultural_stubble_burn: {
    label: "Agricultural Burn",
    colorDark: "#f39c12",
    colorLight: "#ca621a",
    badgeDark: { bg: "rgba(230, 126, 34, 0.15)", border: "rgba(230, 126, 34, 0.4)", text: "#f39c12" },
    badgeLight: { bg: "rgba(202, 98, 26, 0.12)", border: "rgba(202, 98, 26, 0.35)", text: "#ca621a" },
    barDark: "#f39c12",
    barLight: "#ca621a",
  },
  forest_wildfire: {
    label: "Forest Wildfire",
    colorDark: "#e74c3c",
    colorLight: "#ad4f37",
    badgeDark: { bg: "rgba(208, 105, 76, 0.15)", border: "rgba(208, 105, 76, 0.4)", text: "#e74c3c" },
    badgeLight: { bg: "rgba(173, 79, 55, 0.12)", border: "rgba(173, 79, 55, 0.35)", text: "#ad4f37" },
    barDark: "#e74c3c",
    barLight: "#ad4f37",
  },
  urban_false_positive: {
    label: "Urban False Positive",
    colorDark: "#74a9ae",
    colorLight: "#2b6e72",
    badgeDark: { bg: "rgba(116, 169, 174, 0.15)", border: "rgba(116, 169, 174, 0.4)", text: "#74a9ae" },
    badgeLight: { bg: "rgba(43, 110, 114, 0.12)", border: "rgba(43, 110, 114, 0.35)", text: "#2b6e72" },
    barDark: "#74a9ae",
    barLight: "#2b6e72",
  },
};

export default function SampleAnalyzerModal({
  isOpen,
  onClose,
  apiBase,
  theme = "dark",
  isPickingLocation,
  setIsPickingLocation,
  pickedCoordinates,
  onClearPickedCoordinates,
}: SampleAnalyzerModalProps) {
  const isDark = theme !== "light";

  // Input states
  const [rgbFile, setRgbFile] = useState<File | null>(null);
  const [rgbPreview, setRgbPreview] = useState<string | null>(null);
  const [swirFile, setSwirFile] = useState<File | null>(null);
  const [swirPreview, setSwirPreview] = useState<string | null>(null);
  const [latitude, setLatitude] = useState<string>("");
  const [longitude, setLongitude] = useState<string>("");

  // Presets and execution state
  const [presets, setPresets] = useState<RecentEventPreset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>("");
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SampleAnalysisResponse | null>(null);

  // Drag states
  const [isRgbDragging, setIsRgbDragging] = useState(false);
  const [isSwirDragging, setIsSwirDragging] = useState(false);

  const rgbInputRef = useRef<HTMLInputElement>(null);
  const swirInputRef = useRef<HTMLInputElement>(null);

  // Sync picked coordinates from Deck.gl map click
  useEffect(() => {
    if (pickedCoordinates) {
      setLatitude(pickedCoordinates.lat.toFixed(6));
      setLongitude(pickedCoordinates.lon.toFixed(6));
      onClearPickedCoordinates();
    }
  }, [pickedCoordinates, onClearPickedCoordinates]);

  // Load sample presets from /api/events/recent
  useEffect(() => {
    if (!isOpen) return;
    let isMounted = true;
    fetch(`${apiBase}/api/events/recent`)
      .then((res) => (res.ok ? res.json() : []))
      .then((data: RecentEventPreset[]) => {
        if (isMounted && Array.isArray(data)) {
          setPresets(data);
        }
      })
      .catch((err) => {
        console.warn("Could not fetch recent events presets:", err);
      });
    return () => {
      isMounted = false;
    };
  }, [isOpen, apiBase]);

  // Handle Preset Selection
  const handleSelectPreset = useCallback(
    async (eventId: string) => {
      setSelectedPresetId(eventId);
      if (!eventId) return;

      const ev = presets.find((p) => String(p.id) === String(eventId));
      if (!ev) return;

      if (ev.lat !== undefined && ev.lat !== null) setLatitude(String(ev.lat));
      if (ev.lon !== undefined && ev.lon !== null) setLongitude(String(ev.lon));

      // Attempt to preload RGB chip if available
      if (ev.imagery_rgb_path) {
        try {
          const filename = ev.imagery_rgb_path.split(/[/\\]/).pop();
          const rgbUrl = `${apiBase}/static/imagery/${filename}`;
          const res = await fetch(rgbUrl);
          if (res.ok) {
            const blob = await res.blob();
            const file = new File([blob], filename || "preset_rgb.png", {
              type: blob.type || "image/png",
            });
            setRgbFile(file);
            setRgbPreview(URL.createObjectURL(blob));
          }
        } catch (e) {
          console.warn("Preset RGB preload failed:", e);
        }
      }

      // Attempt to preload SWIR chip if available
      if (ev.imagery_swir_path) {
        try {
          const filename = ev.imagery_swir_path.split(/[/\\]/).pop();
          const swirUrl = `${apiBase}/static/imagery/${filename}`;
          const res = await fetch(swirUrl);
          if (res.ok) {
            const blob = await res.blob();
            const file = new File([blob], filename || "preset_swir.png", {
              type: blob.type || "image/png",
            });
            setSwirFile(file);
            setSwirPreview(URL.createObjectURL(blob));
          }
        } catch (e) {
          console.warn("Preset SWIR preload failed:", e);
        }
      }
    },
    [presets, apiBase]
  );

  // File Handlers
  const handleRgbSelect = (file: File) => {
    setRgbFile(file);
    setRgbPreview(URL.createObjectURL(file));
    setError(null);
  };

  const handleSwirSelect = (file: File) => {
    setSwirFile(file);
    setSwirPreview(URL.createObjectURL(file));
  };

  const clearRgb = () => {
    setRgbFile(null);
    if (rgbPreview) URL.revokeObjectURL(rgbPreview);
    setRgbPreview(null);
  };

  const clearSwir = () => {
    setSwirFile(null);
    if (swirPreview) URL.revokeObjectURL(swirPreview);
    setSwirPreview(null);
  };

  // Submit Analysis Request
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!rgbFile) {
      setError("Please select or drop an RGB satellite image (PNG, JPEG, or TIF/TIFF).");
      return;
    }

    setAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append("rgb_file", rgbFile);
    if (swirFile) {
      formData.append("swir_file", swirFile);
    }
    if (latitude.trim()) {
      formData.append("latitude", latitude.trim());
    }
    if (longitude.trim()) {
      formData.append("longitude", longitude.trim());
    }

    try {
      const res = await fetch(`${apiBase}/api/analyze-sample`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => null);
        throw new Error(errJson?.detail || `Analysis failed with HTTP status ${res.status}`);
      }

      const data: SampleAnalysisResponse = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err?.message || "Failed to analyze sample. Please check file and coordinates.");
    } finally {
      setAnalyzing(false);
    }
  };

  // Reset Form
  const handleReset = () => {
    clearRgb();
    clearSwir();
    setLatitude("");
    setLongitude("");
    setSelectedPresetId("");
    setResult(null);
    setError(null);
  };

  if (!isOpen) return null;

  return (
    <div
      className="sam-backdrop"
      data-theme={theme}
      role="dialog"
      aria-modal="true"
      aria-labelledby="sample-analyzer-title"
    >
      <div className="sam-dialog">
        {/* Header Bar */}
        <div className="sam-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <div className="sam-header-icon" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                <line x1="12" y1="22.08" x2="12" y2="12" />
              </svg>
            </div>
            <div>
              <h2 id="sample-analyzer-title" className="sam-title">
                Sample Analyzer &amp; Geolocation
              </h2>
              <p className="sam-sub">
                On-demand 6-channel satellite classification, sovereign demarcation &amp; Grad-CAM attribution
              </p>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            {result && (
              <button
                type="button"
                onClick={handleReset}
                className="sam-btn-cancel"
                style={{ height: "32px", padding: "0 0.75rem", fontSize: "0.75rem" }}
              >
                Reset Form
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              className="sam-btn-cancel"
              style={{ height: "32px", width: "32px", padding: 0 }}
              aria-label="Close modal"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="sam-body" style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          {/* Preset Selector Banner */}
          <div
            className="sam-inset-panel"
            style={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "0.75rem",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.75rem" }}>
              <svg
                width="16"
                height="16"
                style={{ color: isDark ? "#e5a246" : "#b86f1c", flexShrink: 0 }}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="16" x2="12" y2="12" />
                <line x1="12" y1="8" x2="12.01" y2="8" />
              </svg>
              <span style={{ color: isDark ? "#e8eeed" : "#182a2d" }}>
                Load sample telemetry and chips from recent high-priority events:
              </span>
            </div>
            <select
              value={selectedPresetId}
              onChange={(e) => handleSelectPreset(e.target.value)}
              className="sam-select"
              style={{ maxWidth: "280px" }}
              aria-label="Load Sample Preset"
            >
              <option value="">Load Sample Preset...</option>
              {presets.map((ev) => (
                <option key={ev.id} value={ev.id}>
                  Event #{ev.id.slice(0, 8)} &middot; {ev.state || "India"} &middot; {ev.max_frp || ev.peak_frp || 0} MW
                </option>
              ))}
            </select>
          </div>

          {/* Upload Form */}
          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                gap: "1.25rem",
              }}
            >
              {/* RGB Upload Zone */}
              <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <label
                    style={{
                      fontSize: "0.75rem",
                      fontWeight: 600,
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                      color: isDark ? "#e8eeed" : "#182a2d",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.25rem",
                    }}
                  >
                    <span>RGB Satellite Image</span>
                    <span style={{ color: isDark ? "#e5a246" : "#b86f1c" }}>*</span>
                  </label>
                  {rgbFile && (
                    <button
                      type="button"
                      onClick={clearRgb}
                      style={{
                        background: "none",
                        border: "none",
                        color: isDark ? "#f43f5e" : "#be123c",
                        fontSize: "0.6875rem",
                        cursor: "pointer",
                      }}
                    >
                      Clear
                    </button>
                  )}
                </div>

                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsRgbDragging(true);
                  }}
                  onDragLeave={() => setIsRgbDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsRgbDragging(false);
                    if (e.dataTransfer.files?.[0]) handleRgbSelect(e.dataTransfer.files[0]);
                  }}
                  onClick={() => rgbInputRef.current?.click()}
                  className={`sam-dropzone ${isRgbDragging ? "is-dragging" : ""} ${
                    rgbPreview ? "has-preview" : ""
                  }`}
                >
                  <input
                    ref={rgbInputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/tiff,.tif,.tiff"
                    style={{ display: "none" }}
                    onChange={(e) => {
                      if (e.target.files?.[0]) handleRgbSelect(e.target.files[0]);
                    }}
                  />
                  {rgbPreview ? (
                    <div style={{ position: "relative", width: "100%", height: "100%" }}>
                      <img
                        src={rgbPreview}
                        alt="RGB Preview"
                        style={{ width: "100%", height: "100%", objectFit: "contain", padding: "0.5rem" }}
                      />
                    </div>
                  ) : (
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: "0.5rem",
                        padding: "1rem",
                        textAlign: "center",
                      }}
                    >
                      <svg
                        width="32"
                        height="32"
                        style={{ color: isDark ? "#9ab0b3" : "#4b6164" }}
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.5"
                      >
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17 8 12 3 7 8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                      </svg>
                      <p
                        style={{
                          fontSize: "0.75rem",
                          fontWeight: 500,
                          color: isDark ? "#e8eeed" : "#182a2d",
                          margin: 0,
                        }}
                      >
                        Drag &amp; drop RGB chip or{" "}
                        <span style={{ color: isDark ? "#e5a246" : "#b86f1c", fontWeight: 600 }}>browse</span>
                      </p>
                      <p
                        style={{
                          fontSize: "0.6875rem",
                          color: isDark ? "#9ab0b3" : "#4b6164",
                          margin: 0,
                        }}
                      >
                        PNG, JPEG, TIF/TIFF &middot; Max 20MB
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* SWIR Upload Zone */}
              <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <label
                    style={{
                      fontSize: "0.75rem",
                      fontWeight: 600,
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                      color: isDark ? "#9ab0b3" : "#4b6164",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.25rem",
                    }}
                  >
                    <span>SWIR Band Image</span>
                    <span style={{ fontSize: "0.625rem", textTransform: "lowercase", opacity: 0.8 }}>(optional)</span>
                  </label>
                  {swirFile && (
                    <button
                      type="button"
                      onClick={clearSwir}
                      style={{
                        background: "none",
                        border: "none",
                        color: isDark ? "#f43f5e" : "#be123c",
                        fontSize: "0.6875rem",
                        cursor: "pointer",
                      }}
                    >
                      Clear
                    </button>
                  )}
                </div>

                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsSwirDragging(true);
                  }}
                  onDragLeave={() => setIsSwirDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsSwirDragging(false);
                    if (e.dataTransfer.files?.[0]) handleSwirSelect(e.dataTransfer.files[0]);
                  }}
                  onClick={() => swirInputRef.current?.click()}
                  className={`sam-dropzone ${isSwirDragging ? "is-dragging" : ""} ${
                    swirPreview ? "has-preview" : ""
                  }`}
                >
                  <input
                    ref={swirInputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/tiff,.tif,.tiff"
                    style={{ display: "none" }}
                    onChange={(e) => {
                      if (e.target.files?.[0]) handleSwirSelect(e.target.files[0]);
                    }}
                  />
                  {swirPreview ? (
                    <div style={{ position: "relative", width: "100%", height: "100%" }}>
                      <img
                        src={swirPreview}
                        alt="SWIR Preview"
                        style={{ width: "100%", height: "100%", objectFit: "contain", padding: "0.5rem" }}
                      />
                    </div>
                  ) : (
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: "0.5rem",
                        padding: "1rem",
                        textAlign: "center",
                      }}
                    >
                      <svg
                        width="32"
                        height="32"
                        style={{ color: isDark ? "#9ab0b3" : "#4b6164" }}
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.5"
                      >
                        <polygon points="12 2 2 7 12 12 22 7 12 2" />
                        <polyline points="2 17 12 22 22 17" />
                        <polyline points="2 12 12 17 22 12" />
                      </svg>
                      <p
                        style={{
                          fontSize: "0.75rem",
                          fontWeight: 500,
                          color: isDark ? "#e8eeed" : "#182a2d",
                          margin: 0,
                        }}
                      >
                        Paired SWIR2/SWIR1/NIR chip or{" "}
                        <span style={{ color: isDark ? "#e5a246" : "#b86f1c", fontWeight: 600 }}>browse</span>
                      </p>
                      <p
                        style={{
                          fontSize: "0.6875rem",
                          color: isDark ? "#9ab0b3" : "#4b6164",
                          margin: 0,
                        }}
                      >
                        Auto-synthesized if omitted
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Coordinates & Map Picking Bar */}
            <div className="sam-inset-panel" style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "0.75rem",
                }}
              >
                <span
                  style={{
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    color: isDark ? "#e8eeed" : "#182a2d",
                  }}
                >
                  WGS84 Coordinates &amp; Geolocation Context
                </span>

                <button
                  type="button"
                  onClick={() => setIsPickingLocation(!isPickingLocation)}
                  className={`sam-btn-pick ${isPickingLocation ? "is-active" : ""}`}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="22" y1="12" x2="18" y2="12" />
                    <line x1="6" y1="12" x2="2" y2="12" />
                    <line x1="12" y1="6" x2="12" y2="2" />
                    <line x1="12" y1="22" x2="12" y2="18" />
                  </svg>
                  <span>{isPickingLocation ? "Click Map to Capture..." : "Pick From Map"}</span>
                </button>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.75rem" }}>
                <div>
                  <label
                    style={{
                      display: "block",
                      fontSize: "0.6875rem",
                      color: isDark ? "#9ab0b3" : "#4b6164",
                      marginBottom: "0.25rem",
                    }}
                  >
                    Latitude (&deg;N / &deg;S)
                  </label>
                  <input
                    type="number"
                    step="any"
                    placeholder="e.g. 23.795700 (or extracted from EXIF)"
                    value={latitude}
                    onChange={(e) => setLatitude(e.target.value)}
                    className="sam-input"
                  />
                </div>
                <div>
                  <label
                    style={{
                      display: "block",
                      fontSize: "0.6875rem",
                      color: isDark ? "#9ab0b3" : "#4b6164",
                      marginBottom: "0.25rem",
                    }}
                  >
                    Longitude (&deg;E / &deg;W)
                  </label>
                  <input
                    type="number"
                    step="any"
                    placeholder="e.g. 86.430400 (or extracted from EXIF)"
                    value={longitude}
                    onChange={(e) => setLongitude(e.target.value)}
                    className="sam-input"
                  />
                </div>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="sam-error" role="alert">
                <svg width="16" height="16" style={{ flexShrink: 0 }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                <span>{error}</span>
              </div>
            )}

            {/* Action Bar */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: "0.75rem", paddingTop: "0.5rem" }}>
              <button type="button" onClick={onClose} className="sam-btn-cancel">
                Cancel
              </button>
              <button type="submit" disabled={analyzing || !rgbFile} className="sam-btn-primary">
                {analyzing ? (
                  <>
                    <svg
                      width="14"
                      height="14"
                      className="te-spin"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                    </svg>
                    <span>Running Inference &amp; Grad-CAM...</span>
                  </>
                ) : (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                    </svg>
                    <span>Analyze Sample</span>
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Results Panel */}
          {result && (
            <div
              style={{
                paddingTop: "1.25rem",
                borderTop: isDark ? "1px solid #243b3f" : "1px solid #d4cbb9",
                display: "flex",
                flexDirection: "column",
                gap: "1.25rem",
              }}
            >
              {/* Header Telemetry Pill */}
              <div
                className="sam-inset-panel"
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "0.75rem",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                  {(() => {
                    const cfg = CLASS_CONFIG[result.predicted_class];
                    const badge = isDark ? cfg?.badgeDark : cfg?.badgeLight;
                    return (
                      <span
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          padding: "0.25rem 0.75rem",
                          borderRadius: "9999px",
                          fontSize: "0.75rem",
                          fontWeight: 600,
                          background: badge?.bg || "rgba(128,128,128,0.1)",
                          border: `1px solid ${badge?.border || "#888"}`,
                          color: badge?.text || (isDark ? "#e8eeed" : "#182a2d"),
                        }}
                      >
                        {cfg?.label || result.predicted_class}
                      </span>
                    );
                  })()}
                  <span
                    style={{
                      fontSize: "0.875rem",
                      fontWeight: 600,
                      color: isDark ? "#e8eeed" : "#182a2d",
                    }}
                  >
                    {(result.confidence * 100).toFixed(1)}% Confidence
                  </span>
                </div>

                <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0.5rem" }}>
                  <span className="sam-chip">
                    <span className="sam-chip-label">Latency:</span>
                    <strong>{result.inference_latency_ms} ms</strong>
                  </span>
                  <span className="sam-chip">
                    <span className="sam-chip-label">Modality:</span>
                    <strong>{result.modality}</strong>
                  </span>
                  <span className="sam-chip">
                    <span className="sam-chip-label">Arch:</span>
                    <strong>{result.model_arch}</strong>
                  </span>
                </div>
              </div>

              {/* Side-by-side Cards */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                  gap: "1rem",
                }}
              >
                {/* 1. Input RGB Card */}
                <div className="sam-img-card">
                  <div className="sam-img-card-header">
                    <span>Input RGB (Visual)</span>
                    <span style={{ fontSize: "0.625rem", color: isDark ? "#9ab0b3" : "#4b6164" }}>B04 / B03 / B02</span>
                  </div>
                  <div className="sam-img-card-body">
                    <img
                      src={`${apiBase}${result.rgb_url}`}
                      alt="Input RGB Sample"
                      style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: "0.5rem" }}
                    />
                  </div>
                </div>

                {/* 2. SWIR Band Card */}
                <div className="sam-img-card">
                  <div className="sam-img-card-header">
                    <span>SWIR Band (False Color)</span>
                    <span style={{ fontSize: "0.625rem", color: isDark ? "#9ab0b3" : "#4b6164" }}>B12 / B11 / B8A</span>
                  </div>
                  <div className="sam-img-card-body">
                    {result.swir_url ? (
                      <img
                        src={`${apiBase}${result.swir_url}`}
                        alt="SWIR Band Sample"
                        style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: "0.5rem" }}
                      />
                    ) : (
                      <div
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          alignItems: "center",
                          justifyContent: "center",
                          textAlign: "center",
                          padding: "1rem",
                          color: isDark ? "#e5a246" : "#b86f1c",
                        }}
                      >
                        <svg
                          width="32"
                          height="32"
                          style={{ marginBottom: "0.5rem" }}
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.5"
                        >
                          <polygon points="12 2 2 7 12 12 22 7 12 2" />
                          <polyline points="2 17 12 22 22 17" />
                          <polyline points="2 12 12 17 22 12" />
                        </svg>
                        <p style={{ fontSize: "0.75rem", fontWeight: 600, margin: 0 }}>SWIR Not Provided</p>
                        <p
                          style={{
                            fontSize: "0.6875rem",
                            color: isDark ? "#9ab0b3" : "#4b6164",
                            marginTop: "0.25rem",
                            maxWidth: "180px",
                          }}
                        >
                          Synthesized by replicating RGB channels into bands 4-6
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {/* 3. Grad-CAM Overlay Card */}
                <div className="sam-img-card">
                  <div className="sam-img-card-header">
                    <span>Grad-CAM Attribution</span>
                    <span
                      style={{
                        fontSize: "0.625rem",
                        color: isDark ? "#e5a246" : "#b86f1c",
                        fontWeight: 600,
                      }}
                    >
                      Salient Features
                    </span>
                  </div>
                  <div className="sam-img-card-body">
                    {result.gradcam_url ? (
                      <img
                        src={`${apiBase}${result.gradcam_url}`}
                        alt="Grad-CAM Attribution Heatmap"
                        style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: "0.5rem" }}
                      />
                    ) : (
                      <div style={{ fontSize: "0.75rem", color: isDark ? "#9ab0b3" : "#4b6164" }}>
                        Grad-CAM unavailable
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Class Probability Distributions */}
              <div className="sam-inset-panel" style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                <span
                  style={{
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    color: isDark ? "#e8eeed" : "#182a2d",
                  }}
                >
                  Class Probability Distribution
                </span>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                    gap: "0.875rem",
                  }}
                >
                  {Object.entries(result.class_probabilities).map(([cName, prob]) => {
                    const cfg = CLASS_CONFIG[cName];
                    const percent = (prob * 100).toFixed(1);
                    const barColor = isDark ? cfg?.barDark || "#74a9ae" : cfg?.barLight || "#2b6e72";
                    return (
                      <div key={cName} style={{ display: "flex", flexDirection: "column", gap: "0.375rem" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem" }}>
                          <span style={{ color: isDark ? "#e8eeed" : "#182a2d", fontWeight: 500 }}>
                            {cfg?.label || cName}
                          </span>
                          <span style={{ color: isDark ? "#9ab0b3" : "#4b6164", fontFamily: "monospace" }}>
                            {percent}%
                          </span>
                        </div>
                        <div className="sam-prob-track">
                          <div
                            style={{
                              height: "100%",
                              borderRadius: "9999px",
                              backgroundColor: barColor,
                              width: `${percent}%`,
                              transition: "width 0.5s ease",
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Geolocation & Territorial Demarcation Card */}
              <div className="sam-inset-panel" style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                <span
                  style={{
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    color: isDark ? "#e8eeed" : "#182a2d",
                  }}
                >
                  Geospatial Verification &amp; Territorial Demarcation
                </span>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                    gap: "0.75rem",
                  }}
                >
                  <div className="sam-demarcation-card">
                    <span className="sam-demarcation-label">WGS84 Coordinates</span>
                    <span className="sam-demarcation-val" style={{ fontFamily: "monospace" }}>
                      {result.location.latitude !== null && result.location.longitude !== null
                        ? `${result.location.latitude.toFixed(4)}, ${result.location.longitude.toFixed(4)}`
                        : "Not specified"}
                    </span>
                  </div>

                  <div className="sam-demarcation-card">
                    <span className="sam-demarcation-label">Survey of India Boundary</span>
                    <span
                      className="sam-demarcation-val"
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.375rem",
                        color: result.location.inside_india
                          ? isDark
                            ? "#4ade80"
                            : "#16a34a"
                          : isDark
                          ? "#9ab0b3"
                          : "#4b6164",
                      }}
                    >
                      <span
                        style={{
                          width: "8px",
                          height: "8px",
                          borderRadius: "50%",
                          backgroundColor: result.location.inside_india ? "#22c55e" : "#94a3b8",
                        }}
                      />
                      <span>
                        {result.location.inside_india ? "Inside Sovereign India" : "Outside Territory / Unmapped"}
                      </span>
                    </span>
                  </div>

                  <div className="sam-demarcation-card">
                    <span className="sam-demarcation-label">Matched State / UT</span>
                    <span className="sam-demarcation-val">{result.location.state || "Not identified"}</span>
                  </div>

                  <div className="sam-demarcation-card">
                    <span className="sam-demarcation-label">5 km Thermal Context</span>
                    <span className="sam-demarcation-val">{result.location.nearby_events_5km} candidate events</span>
                  </div>
                </div>

                {result.location.nearest_infrastructure && (
                  <div
                    style={{
                      padding: "0.625rem 0.875rem",
                      borderRadius: "0.5rem",
                      fontSize: "0.75rem",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                      background: isDark ? "rgba(13, 26, 29, 0.6)" : "#ffffff",
                      border: isDark ? "1px solid #243b3f" : "1px solid #cfc6b4",
                    }}
                  >
                    <span style={{ color: isDark ? "#9ab0b3" : "#4b6164" }}>Nearest Infrastructure Context:</span>
                    <strong style={{ color: isDark ? "#e8eeed" : "#182a2d" }}>
                      {result.location.nearest_infrastructure}
                    </strong>
                  </div>
                )}
              </div>

              {/* Mandatory ML Disclaimer Banner */}
              <div className="sam-disclaimer">
                <svg
                  width="16"
                  height="16"
                  style={{ flexShrink: 0, marginTop: "2px", color: isDark ? "#e5a246" : "#b86f1c" }}
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                  <line x1="12" y1="9" x2="12" y2="13" />
                  <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
                <span>{result.disclaimer}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
