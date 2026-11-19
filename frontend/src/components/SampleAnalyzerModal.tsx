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
  isPickingLocation: boolean;
  setIsPickingLocation: (val: boolean) => void;
  pickedCoordinates: { lat: number; lon: number } | null;
  onClearPickedCoordinates: () => void;
}

const CLASS_CONFIG: Record<
  string,
  { label: string; color: string; badgeClass: string; barClass: string }
> = {
  industrial_thermal: {
    label: "Industrial Thermal",
    color: "#e0a458",
    badgeClass: "border-amber-500/40 bg-amber-500/10 text-amber-300",
    barClass: "bg-amber-500",
  },
  agricultural_stubble_burn: {
    label: "Agricultural Burn",
    color: "#e67e22",
    badgeClass: "border-orange-500/40 bg-orange-500/10 text-orange-300",
    barClass: "bg-orange-500",
  },
  forest_wildfire: {
    label: "Forest Wildfire",
    color: "#d0694c",
    badgeClass: "border-rose-500/40 bg-rose-500/10 text-rose-300",
    barClass: "bg-rose-500",
  },
  urban_false_positive: {
    label: "Urban False Positive",
    color: "#74a9ae",
    badgeClass: "border-cyan-500/40 bg-cyan-500/10 text-cyan-300",
    barClass: "bg-cyan-500",
  },
};

export default function SampleAnalyzerModal({
  isOpen,
  onClose,
  apiBase,
  isPickingLocation,
  setIsPickingLocation,
  pickedCoordinates,
  onClearPickedCoordinates,
}: SampleAnalyzerModalProps) {
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
            const file = new File([blob], filename || "preset_rgb.png", { type: blob.type || "image/png" });
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
            const file = new File([blob], filename || "preset_swir.png", { type: blob.type || "image/png" });
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
      className="fixed inset-0 z-50 flex items-center justify-center p-3 md:p-6 overflow-y-auto bg-slate-950/75 backdrop-blur-md"
      role="dialog"
      aria-modal="true"
      aria-labelledby="sample-analyzer-title"
    >
      <div className="relative w-full max-w-5xl my-auto rounded-2xl bg-slate-950/90 backdrop-blur-xl border border-slate-800 text-slate-100 shadow-2xl overflow-hidden transition-all">
        {/* Header Bar */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/80 bg-slate-900/50">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400">
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                <line x1="12" y1="22.08" x2="12" y2="12" />
              </svg>
            </div>
            <div>
              <h2 id="sample-analyzer-title" className="text-lg font-semibold tracking-tight text-slate-100">
                Sample Analyzer & Geolocation
              </h2>
              <p className="text-xs text-slate-400">
                On-demand 6-channel satellite classification, sovereign demarcation & Grad-CAM attribution
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {result && (
              <button
                type="button"
                onClick={handleReset}
                className="px-2.5 py-1 text-xs font-medium rounded-lg text-slate-400 hover:text-slate-200 border border-slate-800 hover:border-slate-700 transition-colors"
              >
                Reset Form
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-transparent hover:border-slate-700/80 transition-colors"
              aria-label="Close modal"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="p-6 max-h-[82vh] overflow-y-auto space-y-6">
          {/* Preset Selector Banner */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 rounded-xl bg-slate-900/40 border border-slate-800/70">
            <div className="flex items-center gap-2.5 text-xs text-slate-300">
              <svg className="w-4 h-4 text-amber-400 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="16" x2="12" y2="12" />
                <line x1="12" y1="8" x2="12.01" y2="8" />
              </svg>
              <span>Load sample telemetry and chips from recent high-priority events:</span>
            </div>
            <select
              value={selectedPresetId}
              onChange={(e) => handleSelectPreset(e.target.value)}
              className="px-3 py-1.5 text-xs rounded-lg bg-slate-900 border border-slate-700 text-slate-200 focus:outline-none focus:border-amber-500"
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
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* RGB Upload Zone */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                    <span>RGB Satellite Image</span>
                    <span className="text-amber-400">*</span>
                  </label>
                  {rgbFile && (
                    <button
                      type="button"
                      onClick={clearRgb}
                      className="text-[11px] text-rose-400 hover:text-rose-300"
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
                  className={`relative flex flex-col items-center justify-center h-44 rounded-xl border-2 border-dashed transition-all cursor-pointer overflow-hidden ${
                    isRgbDragging
                      ? "border-amber-500 bg-amber-500/10"
                      : rgbPreview
                      ? "border-slate-700 bg-slate-900/30"
                      : "border-slate-800 hover:border-slate-700 bg-slate-900/20"
                  }`}
                >
                  <input
                    ref={rgbInputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/tiff,.tif,.tiff"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files?.[0]) handleRgbSelect(e.target.files[0]);
                    }}
                  />
                  {rgbPreview ? (
                    <div className="relative w-full h-full group">
                      <img
                        src={rgbPreview}
                        alt="RGB Preview"
                        className="w-full h-full object-contain p-2"
                      />
                      <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-xs font-medium text-slate-200">
                        Click to change image
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-2 p-4 text-center">
                      <svg className="w-8 h-8 text-slate-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17 8 12 3 7 8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                      </svg>
                      <p className="text-xs text-slate-300 font-medium">
                        Drag & drop RGB chip or <span className="text-amber-400">browse</span>
                      </p>
                      <p className="text-[10px] text-slate-500">PNG, JPEG, TIF/TIFF &middot; Max 20MB</p>
                    </div>
                  )}
                </div>
              </div>

              {/* SWIR Upload Zone */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                    <span>SWIR Band Image</span>
                    <span className="text-[10px] lowercase text-slate-500">(optional)</span>
                  </label>
                  {swirFile && (
                    <button
                      type="button"
                      onClick={clearSwir}
                      className="text-[11px] text-rose-400 hover:text-rose-300"
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
                  className={`relative flex flex-col items-center justify-center h-44 rounded-xl border-2 border-dashed transition-all cursor-pointer overflow-hidden ${
                    isSwirDragging
                      ? "border-amber-500 bg-amber-500/10"
                      : swirPreview
                      ? "border-slate-700 bg-slate-900/30"
                      : "border-slate-800 hover:border-slate-700 bg-slate-900/20"
                  }`}
                >
                  <input
                    ref={swirInputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/tiff,.tif,.tiff"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files?.[0]) handleSwirSelect(e.target.files[0]);
                    }}
                  />
                  {swirPreview ? (
                    <div className="relative w-full h-full group">
                      <img
                        src={swirPreview}
                        alt="SWIR Preview"
                        className="w-full h-full object-contain p-2"
                      />
                      <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-xs font-medium text-slate-200">
                        Click to change image
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-2 p-4 text-center">
                      <svg className="w-8 h-8 text-slate-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <polygon points="12 2 2 7 12 12 22 7 12 2" />
                        <polyline points="2 17 12 22 22 17" />
                        <polyline points="2 12 12 17 22 12" />
                      </svg>
                      <p className="text-xs text-slate-400 font-medium">
                        Paired SWIR2/SWIR1/NIR chip or <span className="text-slate-300">browse</span>
                      </p>
                      <p className="text-[10px] text-slate-500">Auto-synthesized if omitted</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Coordinates & Map Picking Bar */}
            <div className="p-4 rounded-xl bg-slate-900/30 border border-slate-800/80 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                  WGS84 Coordinates & Geolocation Context
                </span>

                <button
                  type="button"
                  onClick={() => setIsPickingLocation(!isPickingLocation)}
                  className={`inline-flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-lg border transition-all ${
                    isPickingLocation
                      ? "bg-amber-500 text-slate-950 border-amber-400 font-semibold"
                      : "bg-slate-900 hover:bg-slate-800 text-slate-300 border-slate-700"
                  }`}
                >
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="22" y1="12" x2="18" y2="12" />
                    <line x1="6" y1="12" x2="2" y2="12" />
                    <line x1="12" y1="6" x2="12" y2="2" />
                    <line x1="12" y1="22" x2="12" y2="18" />
                  </svg>
                  <span>{isPickingLocation ? "Click Map to Capture..." : "Pick From Map"}</span>
                </button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] text-slate-400 mb-1">Latitude (&deg;N / &deg;S)</label>
                  <input
                    type="number"
                    step="any"
                    placeholder="e.g. 23.795700 (or extracted from EXIF)"
                    value={latitude}
                    onChange={(e) => setLatitude(e.target.value)}
                    className="w-full px-3 py-2 text-xs rounded-lg bg-slate-950/80 border border-slate-800 text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-amber-500"
                  />
                </div>
                <div>
                  <label className="block text-[11px] text-slate-400 mb-1">Longitude (&deg;E / &deg;W)</label>
                  <input
                    type="number"
                    step="any"
                    placeholder="e.g. 86.430400 (or extracted from EXIF)"
                    value={longitude}
                    onChange={(e) => setLongitude(e.target.value)}
                    className="w-full px-3 py-2 text-xs rounded-lg bg-slate-950/80 border border-slate-800 text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-amber-500"
                  />
                </div>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="flex items-center gap-2.5 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
                <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                <span>{error}</span>
              </div>
            )}

            {/* Action Bar */}
            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-xs font-medium rounded-lg text-slate-400 hover:text-slate-200 border border-slate-800 hover:border-slate-700 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={analyzing || !rgbFile}
                className="inline-flex items-center gap-2 px-5 py-2 text-xs font-semibold rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-950 disabled:opacity-50 disabled:cursor-not-allowed shadow-md transition-all"
              >
                {analyzing ? (
                  <>
                    <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                    </svg>
                    <span>Running Inference & Grad-CAM...</span>
                  </>
                ) : (
                  <>
                    <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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
            <div className="pt-6 border-t border-slate-800/80 space-y-6">
              {/* Header Telemetry Pill */}
              <div className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-xl bg-slate-900/40 border border-slate-800">
                <div className="flex items-center gap-3">
                  <span
                    className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border ${
                      CLASS_CONFIG[result.predicted_class]?.badgeClass || "border-slate-700 text-slate-200"
                    }`}
                  >
                    {CLASS_CONFIG[result.predicted_class]?.label || result.predicted_class}
                  </span>
                  <span className="text-sm font-semibold text-slate-100">
                    {(result.confidence * 100).toFixed(1)}% Confidence
                  </span>
                </div>

                <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800">
                    <span className="text-slate-500">Latency:</span>
                    <strong className="text-slate-200">{result.inference_latency_ms} ms</strong>
                  </span>
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800">
                    <span className="text-slate-500">Modality:</span>
                    <strong className="text-slate-200">{result.modality}</strong>
                  </span>
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800">
                    <span className="text-slate-500">Arch:</span>
                    <strong className="text-slate-200">{result.model_arch}</strong>
                  </span>
                </div>
              </div>

              {/* Side-by-side Aspect-Square Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* 1. Input RGB Card */}
                <div className="flex flex-col rounded-xl overflow-hidden border border-slate-800 bg-slate-900/30">
                  <div className="px-3.5 py-2 text-xs font-semibold tracking-wide text-slate-300 border-b border-slate-800/80 flex items-center justify-between">
                    <span>Input RGB (Visual)</span>
                    <span className="text-[10px] text-slate-500">B04 / B03 / B02</span>
                  </div>
                  <div className="aspect-square relative bg-slate-950 flex items-center justify-center p-1">
                    <img
                      src={`${apiBase}${result.rgb_url}`}
                      alt="Input RGB Sample"
                      className="w-full h-full object-contain rounded-lg"
                    />
                  </div>
                </div>

                {/* 2. SWIR Band Card */}
                <div className="flex flex-col rounded-xl overflow-hidden border border-slate-800 bg-slate-900/30">
                  <div className="px-3.5 py-2 text-xs font-semibold tracking-wide text-slate-300 border-b border-slate-800/80 flex items-center justify-between">
                    <span>SWIR Band (False Color)</span>
                    <span className="text-[10px] text-slate-500">B12 / B11 / B8A</span>
                  </div>
                  <div className="aspect-square relative bg-slate-950 flex items-center justify-center p-2">
                    {result.swir_url ? (
                      <img
                        src={`${apiBase}${result.swir_url}`}
                        alt="SWIR Band Sample"
                        className="w-full h-full object-contain rounded-lg"
                      />
                    ) : (
                      <div className="flex flex-col items-center justify-center text-center p-4 rounded-xl border border-amber-500/20 bg-amber-500/5 text-amber-300">
                        <svg className="w-8 h-8 mb-2 text-amber-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                          <polygon points="12 2 2 7 12 12 22 7 12 2" />
                          <polyline points="2 17 12 22 22 17" />
                          <polyline points="2 12 12 17 22 12" />
                        </svg>
                        <p className="text-xs font-semibold">SWIR Not Provided</p>
                        <p className="text-[11px] text-slate-400 mt-1 max-w-[180px]">
                          Synthesized by replicating RGB channels into bands 4-6
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {/* 3. Grad-CAM Overlay Card */}
                <div className="flex flex-col rounded-xl overflow-hidden border border-slate-800 bg-slate-900/30">
                  <div className="px-3.5 py-2 text-xs font-semibold tracking-wide text-slate-300 border-b border-slate-800/80 flex items-center justify-between">
                    <span>Grad-CAM Attribution</span>
                    <span className="text-[10px] text-amber-400">Salient Features</span>
                  </div>
                  <div className="aspect-square relative bg-slate-950 flex items-center justify-center p-1">
                    {result.gradcam_url ? (
                      <img
                        src={`${apiBase}${result.gradcam_url}`}
                        alt="Grad-CAM Attribution Heatmap"
                        className="w-full h-full object-contain rounded-lg"
                      />
                    ) : (
                      <div className="text-xs text-slate-500">Grad-CAM unavailable</div>
                    )}
                  </div>
                </div>
              </div>

              {/* Class Probability Distributions */}
              <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-300 block">
                  Class Probability Distribution
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                  {Object.entries(result.class_probabilities).map(([cName, prob]) => {
                    const cfg = CLASS_CONFIG[cName];
                    const percent = (prob * 100).toFixed(1);
                    return (
                      <div key={cName} className="space-y-1.5">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-300 font-medium">{cfg?.label || cName}</span>
                          <span className="text-slate-400 font-mono">{percent}%</span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-slate-800/80 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${cfg?.barClass || "bg-slate-400"} transition-all duration-500`}
                            style={{ width: `${percent}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Geolocation & Territorial Demarcation Card */}
              <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-300 block">
                  Geospatial Verification & Territorial Demarcation
                </span>

                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                  <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                    <span className="text-slate-500 block text-[11px]">WGS84 Coordinates</span>
                    <span className="font-mono text-slate-200 mt-1 block">
                      {result.location.latitude !== null && result.location.longitude !== null
                        ? `${result.location.latitude.toFixed(4)}, ${result.location.longitude.toFixed(4)}`
                        : "Not specified"}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                    <span className="text-slate-500 block text-[11px]">Survey of India Boundary</span>
                    <span className="mt-1 flex items-center gap-1.5 font-medium">
                      <span
                        className={`w-2 h-2 rounded-full ${
                          result.location.inside_india ? "bg-emerald-400" : "bg-slate-500"
                        }`}
                      />
                      <span className={result.location.inside_india ? "text-emerald-300" : "text-slate-400"}>
                        {result.location.inside_india ? "Inside Sovereign India" : "Outside Territory / Unmapped"}
                      </span>
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                    <span className="text-slate-500 block text-[11px]">Matched State / UT</span>
                    <span className="font-medium text-slate-200 mt-1 block">
                      {result.location.state || "Not identified"}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                    <span className="text-slate-500 block text-[11px]">5 km Thermal Context</span>
                    <span className="font-medium text-slate-200 mt-1 block">
                      {result.location.nearby_events_5km} candidate events
                    </span>
                  </div>
                </div>

                {result.location.nearest_infrastructure && (
                  <div className="p-2.5 rounded-lg bg-slate-950/40 border border-slate-800/60 text-xs text-slate-300 flex items-center gap-2">
                    <span className="text-slate-500">Nearest Infrastructure Context:</span>
                    <strong className="text-slate-200">{result.location.nearest_infrastructure}</strong>
                  </div>
                )}
              </div>

              {/* Mandatory ML Disclaimer Banner */}
              <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-200/90 text-xs flex items-start gap-2.5">
                <svg className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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
