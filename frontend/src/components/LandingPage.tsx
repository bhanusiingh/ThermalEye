import React, { useState, useEffect, useRef } from "react";

export interface LandingPageProps {
  onEnterDashboard: () => void;
  onNavigateSection?: (section: "events" | "methodology" | "about") => void;
  activeEventCount?: number;
}

export default function LandingPage({
  onEnterDashboard,
  onNavigateSection,
  activeEventCount = 3407,
}: LandingPageProps) {
  const [activeModal, setActiveModal] = useState<"methodology" | "about" | null>(null);
  const [mouseOffset, setMouseOffset] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number | null>(null);

  // Smooth pointer parallax
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const { clientX, clientY } = e;
    const { innerWidth, innerHeight } = window;
    const targetX = ((clientX / innerWidth) - 0.5) * 16;
    const targetY = ((clientY / innerHeight) - 0.5) * 14;

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      setMouseOffset({ x: targetX, y: targetY });
    });
  };

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMouseMove}
      className="relative min-h-screen w-full bg-[#242834] text-slate-100 flex flex-col justify-between overflow-x-hidden select-none"
    >
      {/* Ambient Radial Lighting Glow */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(circle at 65% 50%, rgba(43, 48, 62, 0.95) 0%, rgba(36, 40, 52, 0.8) 55%, #242834 100%)",
        }}
      />

      {/* -------------------------------------------------------------------- */}
      {/* Shared Container for strict vertical alignment */}
      {/* -------------------------------------------------------------------- */}
      <div className="w-full max-w-7xl mx-auto px-8 sm:px-12 lg:px-16 flex flex-col flex-1 relative z-20">
        {/* Top Navigation Bar */}
        <header className="w-full pt-8 pb-4 flex items-center justify-between">
          {/* Left Cluster: Brand Emblem & Social Icons */}
          <div className="flex items-center gap-5 sm:gap-6">
            {/* Circular Brand Avatar / Emblem */}
            <div className="w-9 h-9 rounded-full border border-slate-500/40 bg-slate-800/80 flex items-center justify-center text-slate-200 shadow-md shadow-black/20 overflow-hidden hover:border-emerald-400/60 transition-colors">
              <svg
                className="w-4 h-4 text-emerald-400"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="10" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                <path d="M2 12h20" />
              </svg>
            </div>

            {/* Social / Ecosystem Links */}
            <div className="flex items-center gap-4 text-slate-400">
              {/* GitHub Link */}
              <a
                href="https://github.com"
                target="_blank"
                rel="noreferrer"
                className="hover:text-emerald-400 transition-colors p-1"
                title="GitHub Repository"
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                </svg>
              </a>

              {/* Twitter / X Link */}
              <a
                href="https://twitter.com"
                target="_blank"
                rel="noreferrer"
                className="hover:text-emerald-400 transition-colors p-1"
                title="Twitter / X"
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                </svg>
              </a>

              {/* API Docs Link */}
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="hover:text-emerald-400 transition-colors p-1"
                title="FastAPI Interactive Docs"
              >
                <svg
                  className="w-3.5 h-3.5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z" />
                  <path d="M6 6h10" />
                  <path d="M6 10h10" />
                </svg>
              </a>
            </div>
          </div>

          {/* Right Cluster: Navigation Items */}
          <nav className="flex items-center gap-6 sm:gap-9 text-xs font-semibold tracking-[0.2em] uppercase">
            <button
              onClick={() => {}}
              className="text-emerald-400 transition-colors cursor-pointer relative py-1"
            >
              <span>HOME</span>
              <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-emerald-400 rounded-full" />
            </button>

            <button
              onClick={() => {
                if (onNavigateSection) onNavigateSection("events");
                onEnterDashboard();
              }}
              className="text-slate-400 hover:text-slate-100 transition-colors cursor-pointer py-1"
            >
              EVENTS
            </button>

            <button
              onClick={() => {
                if (onNavigateSection) onNavigateSection("methodology");
                setActiveModal("methodology");
              }}
              className="text-slate-400 hover:text-slate-100 transition-colors cursor-pointer py-1"
            >
              METHODOLOGY
            </button>

            <button
              onClick={() => {
                if (onNavigateSection) onNavigateSection("about");
                setActiveModal("about");
              }}
              className="text-slate-400 hover:text-slate-100 transition-colors cursor-pointer py-1"
            >
              ABOUT
            </button>
          </nav>
        </header>

        {/* ------------------------------------------------------------------ */}
        {/* Main Split Hero Section: 12-Column Grid */}
        {/* ------------------------------------------------------------------ */}
        <main className="w-full flex-1 min-h-[calc(100vh-88px)] grid grid-cols-1 lg:grid-cols-12 items-center py-6 lg:py-0">
          {/* Left Hero Column: lg:col-span-5 with generous vertical breathing room */}
          <div className="lg:col-span-5 flex flex-col justify-center z-20 py-8 lg:py-0">
            {/* Background Watermark */}
            <div className="text-6xl sm:text-7xl font-extrabold text-[#2E3342] tracking-tight leading-none select-none pointer-events-none uppercase">
              THERMAL
            </div>

            {/* Foreground Overlapping Heading */}
            <h1 className="text-5xl sm:text-6xl font-bold tracking-tight text-[#F8FAFC] -mt-6 sm:-mt-8 relative z-10 leading-none">
              ThermalEye
            </h1>

            {/* Accent Divider Bar (Explicit block element) */}
            <div className="w-8 h-[2px] bg-[#34D399] my-6 block rounded-full shadow-[0_0_8px_rgba(52,211,153,0.5)]" />

            {/* Description Paragraph */}
            <p className="text-slate-300/85 text-sm sm:text-[15px] leading-relaxed max-w-sm mb-8 font-normal">
              Satellite thermal anomaly intelligence and geospatial threat detection across India.
              Powered by NASA VIIRS, Sentinel-2 imagery, and spatial ML.
            </p>

            {/* Primary CTA Button & Subtle Active Counter Badge */}
            <div className="flex flex-wrap items-center">
              <button
                onClick={onEnterDashboard}
                className="inline-flex items-center gap-2.5 bg-[#34D399] hover:bg-[#2EBD85] active:scale-[0.98] text-slate-950 font-bold text-[11px] tracking-[0.18em] uppercase px-7 py-3.5 rounded-[4px] shadow-lg shadow-emerald-950/40 hover:shadow-emerald-500/20 transition-all cursor-pointer group"
              >
                <span>GET STARTED</span>
                <svg
                  className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M5 12h14" />
                  <path d="m12 5 7 7-7 7" />
                </svg>
              </button>

              <div className="ml-4 flex items-center gap-2 text-[11px] font-mono text-slate-400">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
                <span>
                  <strong className="text-slate-200 font-semibold">{activeEventCount.toLocaleString()}</strong> ACTIVE EVENTS
                </span>
              </div>
            </div>
          </div>

          {/* Right Column: lg:col-span-7 Organic Urban Street-Grid & River Network SVG */}
          <div className="lg:col-span-7 h-[500px] sm:h-[600px] lg:h-[700px] relative flex items-center justify-center pointer-events-none lg:pointer-events-auto overflow-visible">
            {/* Pointer Parallax Container */}
            <div
              className="w-full h-full flex items-center justify-center transition-transform duration-200 ease-out"
              style={{
                transform: `translate3d(${mouseOffset.x}px, ${mouseOffset.y}px, 0)`,
              }}
            >
              {/* Elliptical Radial CSS Mask for Smooth Border Fading */}
              <div
                className="w-full h-full relative"
                style={{
                  WebkitMaskImage:
                    "radial-gradient(ellipse 72% 72% at 55% 50%, #000 42%, transparent 92%)",
                  maskImage:
                    "radial-gradient(ellipse 72% 72% at 55% 50%, #000 42%, transparent 92%)",
                }}
              >
                <svg
                  viewBox="0 0 800 700"
                  className="w-full h-full overflow-visible"
                  preserveAspectRatio="xMidYMid meet"
                >
                  <defs>
                    {/* Emerald Hotspot Glow Filter */}
                    <filter id="emerald-glow" x="-50%" y="-50%" width="200%" height="200%">
                      <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur" />
                      <feColorMatrix
                        type="matrix"
                        values="0 0 0 0 0.204  0 0 0 0 0.827  0 0 0 0 0.6  0 0 0 1 0"
                      />
                      <feMerge>
                        <feMergeNode />
                        <feMergeNode in="SourceGraphic" />
                      </feMerge>
                    </filter>
                  </defs>

                  {/* -------------------------------------------------------- */}
                  {/* Layer 1: Winding River & Canal Paths                     */}
                  {/* -------------------------------------------------------- */}
                  {/* Organic River Base Channels (#2D4A53) */}
                  <g fill="none" stroke="#2D4A53" strokeLinecap="round" strokeLinejoin="round">
                    {/* Eastern Meandering River Channel */}
                    <path
                      d="M 680 10 Q 720 90 680 180 T 730 290 T 665 410 T 740 540 T 700 690"
                      strokeWidth="3.5"
                    />
                    {/* Northern Canal Branch */}
                    <path
                      d="M 180 80 Q 320 130 460 160 T 680 180"
                      strokeWidth="2.5"
                    />
                    {/* Southern Canal Branch */}
                    <path
                      d="M 665 410 Q 560 480 430 540 T 170 640"
                      strokeWidth="2.5"
                    />
                  </g>

                  {/* River Accent Streamlines (#34D399 with subtle opacity) */}
                  <g fill="none" stroke="#34D399" strokeWidth="1.5" strokeOpacity="0.28" strokeLinecap="round">
                    <path d="M 680 10 Q 720 90 680 180 T 730 290 T 665 410 T 740 540 T 700 690" />
                    <path d="M 180 80 Q 320 130 460 160 T 680 180" />
                    <path d="M 665 410 Q 560 480 430 540 T 170 640" />
                  </g>

                  {/* -------------------------------------------------------- */}
                  {/* Layer 2: Dense Fine Street Mesh (60+ segments in #313746) */}
                  {/* -------------------------------------------------------- */}
                  <g fill="none" stroke="#313746" strokeWidth="0.75" strokeLinecap="round">
                    {/* Horizontal Block Streets */}
                    <path d="M 250 210 H 610" />
                    <path d="M 270 230 H 590" />
                    <path d="M 300 270 H 560" />
                    <path d="M 330 295 H 530" />
                    <path d="M 360 340 H 500" />
                    <path d="M 360 390 H 500" />
                    <path d="M 330 435 H 530" />
                    <path d="M 300 460 H 560" />
                    <path d="M 270 500 H 590" />
                    <path d="M 250 525 H 610" />

                    {/* Outer Quadrant Horizontal Lines */}
                    <path d="M 180 150 H 680" />
                    <path d="M 190 190 H 670" />
                    <path d="M 190 545 H 670" />
                    <path d="M 180 580 H 680" />
                    <path d="M 120 270 H 260" />
                    <path d="M 120 310 H 280" />
                    <path d="M 120 420 H 280" />
                    <path d="M 120 460 H 260" />
                    <path d="M 600 270 H 740" />
                    <path d="M 580 310 H 740" />
                    <path d="M 580 420 H 740" />
                    <path d="M 600 460 H 740" />

                    {/* Vertical City Cross Streets */}
                    <path d="M 270 170 V 560" />
                    <path d="M 300 210 V 520" />
                    <path d="M 340 250 V 480" />
                    <path d="M 380 280 V 450" />
                    <path d="M 405 300 V 430" />
                    <path d="M 455 300 V 430" />
                    <path d="M 480 280 V 450" />
                    <path d="M 520 250 V 480" />
                    <path d="M 560 210 V 520" />
                    <path d="M 590 170 V 560" />

                    {/* Outer Quadrant Vertical Lines */}
                    <path d="M 210 120 V 610" />
                    <path d="M 240 140 V 590" />
                    <path d="M 620 140 V 590" />
                    <path d="M 650 120 V 610" />
                    <path d="M 360 100 V 220" />
                    <path d="M 500 100 V 220" />
                    <path d="M 360 510 V 630" />
                    <path d="M 500 510 V 630" />
                    <path d="M 160 240 V 490" />
                    <path d="M 700 240 V 490" />

                    {/* Neighborhood Diagonal Interconnects */}
                    <path d="M 260 210 Q 310 260 280 320" />
                    <path d="M 600 210 Q 550 260 580 320" />
                    <path d="M 280 410 Q 310 470 260 520" />
                    <path d="M 580 410 Q 550 470 600 520" />
                    <path d="M 370 140 Q 430 180 490 140" />
                    <path d="M 370 590 Q 430 550 490 590" />
                    <path d="M 160 280 Q 210 330 190 380" />
                    <path d="M 700 280 Q 650 330 670 380" />
                    <path d="M 310 320 L 350 365 L 310 410" />
                    <path d="M 550 320 L 510 365 L 550 410" />
                    <path d="M 370 280 L 430 320 L 490 280" />
                    <path d="M 370 450 L 430 410 L 490 450" />
                  </g>

                  {/* -------------------------------------------------------- */}
                  {/* Layer 3: Ring Roads & Arterial Highways (#3E4659, 1.3px) */}
                  {/* -------------------------------------------------------- */}
                  <g fill="none" stroke="#3E4659" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
                    {/* Ring 2: Inner City Loop */}
                    <path d="M 370 320 Q 430 310 490 320 Q 510 365 490 410 Q 430 420 370 410 Q 350 365 370 320 Z" />

                    {/* Ring 3: Middle City Expressway */}
                    <path d="M 310 250 Q 430 230 550 250 Q 585 365 550 480 Q 430 500 310 480 Q 275 365 310 250 Z" />

                    {/* Ring 4: Outer City Beltway */}
                    <path d="M 230 170 Q 430 140 630 170 Q 680 365 630 560 Q 430 590 230 560 Q 180 365 230 170 Z" />

                    {/* Ring 5: Regional Perimeter Loop */}
                    <path d="M 140 90 Q 430 50 720 90 Q 780 365 720 640 Q 430 680 140 640 Q 80 365 140 90 Z" />

                    {/* Major Intersecting Diagonal Avenues */}
                    <path d="M 430 30 L 430 670" />
                    <path d="M 90 365 L 770 365" />
                    <path d="M 150 110 Q 280 230 430 365 T 710 620" />
                    <path d="M 710 110 Q 580 230 430 365 T 150 620" />
                    <path d="M 310 40 Q 370 200 430 365 T 550 660" />
                    <path d="M 550 40 Q 490 200 430 365 T 310 660" />
                  </g>

                  {/* -------------------------------------------------------- */}
                  {/* Layer 4: Active Corridor Flow Lines (Animated Dashes)     */}
                  {/* -------------------------------------------------------- */}
                  <g fill="none" stroke="#34D399" strokeWidth="1.6" strokeOpacity="0.8" strokeLinecap="round">
                    {/* Ring 3 Flow Path */}
                    <path
                      d="M 310 250 Q 430 230 550 250 Q 585 365 550 480 Q 430 500 310 480 Q 275 365 310 250 Z"
                      strokeDasharray="8 14"
                      className="animate-dash-flow"
                    />

                    {/* Diagonal Highway Flow Path */}
                    <path
                      d="M 150 110 Q 280 230 430 365 T 710 620"
                      strokeDasharray="8 14"
                      className="animate-dash-flow"
                    />

                    {/* East-West Boulevard Flow Path */}
                    <path
                      d="M 180 365 L 680 365"
                      strokeDasharray="8 14"
                      className="animate-dash-flow"
                    />
                  </g>

                  {/* -------------------------------------------------------- */}
                  {/* Layer 5: Staggered Radar Pulse Waves (5 Major Hubs)       */}
                  {/* -------------------------------------------------------- */}
                  <g fill="none" stroke="#34D399" strokeWidth="1.5">
                    {/* 1. Central Core Hub (430, 365) */}
                    <circle cx="430" cy="365" r="10" className="animate-radar-pulse" />
                    <circle cx="430" cy="365" r="10" style={{ animationDelay: "1.3s" }} className="animate-radar-pulse" />

                    {/* 2. Northeast Junction (520, 260) */}
                    <circle cx="520" cy="260" r="8" style={{ animationDelay: "0.5s" }} className="animate-radar-pulse" />

                    {/* 3. Southwest Junction (350, 440) */}
                    <circle cx="350" cy="440" r="8" style={{ animationDelay: "1.0s" }} className="animate-radar-pulse" />

                    {/* 4. East Boulevard Hub (550, 380) */}
                    <circle cx="550" cy="380" r="7" style={{ animationDelay: "1.7s" }} className="animate-radar-pulse" />

                    {/* 5. Northwest Junction (330, 280) */}
                    <circle cx="330" cy="280" r="7" style={{ animationDelay: "0.2s" }} className="animate-radar-pulse" />
                  </g>

                  {/* -------------------------------------------------------- */}
                  {/* Layer 6: Clustered Glowing Emerald Hotspots (35+ nodes)  */}
                  {/* -------------------------------------------------------- */}
                  {/* Underlay: Gaussian Blur Glow Nodes */}
                  <g fill="#34D399" filter="url(#emerald-glow)" opacity="0.88">
                    {/* Dense Central Urban Core */}
                    <circle cx="430" cy="365" r="6" />
                    <circle cx="442" cy="355" r="5" />
                    <circle cx="418" cy="375" r="5" />
                    <circle cx="438" cy="380" r="5.5" />
                    <circle cx="415" cy="355" r="4.5" />
                    <circle cx="450" cy="370" r="4.5" />
                    <circle cx="425" cy="345" r="4" />
                    <circle cx="452" cy="348" r="4" />
                    <circle cx="408" cy="385" r="4" />
                    <circle cx="445" cy="392" r="3.5" />
                    <circle cx="400" cy="362" r="3.5" />
                    <circle cx="460" cy="360" r="3.5" />

                    {/* Mid-City Arterial Junctions */}
                    <circle cx="520" cy="260" r="5.5" />
                    <circle cx="532" cy="252" r="4" />
                    <circle cx="510" cy="270" r="3.5" />

                    <circle cx="350" cy="440" r="5" />
                    <circle cx="338" cy="450" r="3.5" />
                    <circle cx="362" cy="430" r="3.5" />

                    <circle cx="550" cy="380" r="5" />
                    <circle cx="330" cy="280" r="4.5" />
                    <circle cx="600" cy="440" r="4.5" />
                    <circle cx="500" cy="480" r="4" />
                    <circle cx="460" cy="240" r="4" />
                    <circle cx="400" cy="490" r="4" />
                    <circle cx="370" cy="240" r="4" />
                    <circle cx="620" cy="310" r="4" />
                    <circle cx="260" cy="340" r="4" />
                    <circle cx="280" cy="410" r="3.5" />
                    <circle cx="580" cy="490" r="3.5" />

                    {/* Outer Edge Corridors */}
                    <circle cx="660" cy="210" r="3.5" />
                    <circle cx="690" cy="420" r="3.5" />
                    <circle cx="230" cy="200" r="3.5" />
                    <circle cx="630" cy="530" r="3" />
                    <circle cx="220" cy="520" r="3" />
                    <circle cx="430" cy="150" r="3" />
                    <circle cx="430" cy="580" r="3" />
                    <circle cx="170" cy="365" r="3" />
                    <circle cx="710" cy="365" r="3" />
                  </g>

                  {/* Foreground: Crisp White-Emerald Centers */}
                  <g fill="#A7F3D0" stroke="#10B981" strokeWidth="0.8">
                    {/* Core Nodes */}
                    <circle cx="430" cy="365" r="3.8" fill="#FFFFFF" />
                    <circle cx="442" cy="355" r="3" fill="#FFFFFF" />
                    <circle cx="418" cy="375" r="3" fill="#FFFFFF" />
                    <circle cx="438" cy="380" r="3.2" />
                    <circle cx="415" cy="355" r="2.8" />
                    <circle cx="450" cy="370" r="2.8" />
                    <circle cx="425" cy="345" r="2.5" />
                    <circle cx="452" cy="348" r="2.5" />
                    <circle cx="408" cy="385" r="2.5" />
                    <circle cx="445" cy="392" r="2.2" />
                    <circle cx="400" cy="362" r="2.2" />
                    <circle cx="460" cy="360" r="2.2" />

                    {/* Secondary Hubs */}
                    <circle cx="520" cy="260" r="3.5" fill="#FFFFFF" />
                    <circle cx="532" cy="252" r="2.5" />
                    <circle cx="510" cy="270" r="2.2" />

                    <circle cx="350" cy="440" r="3.2" fill="#FFFFFF" />
                    <circle cx="338" cy="450" r="2.2" />
                    <circle cx="362" cy="430" r="2.2" />

                    <circle cx="550" cy="380" r="3.2" fill="#FFFFFF" />
                    <circle cx="330" cy="280" r="2.8" />
                    <circle cx="600" cy="440" r="2.8" />
                    <circle cx="500" cy="480" r="2.5" />
                    <circle cx="460" cy="240" r="2.5" />
                    <circle cx="400" cy="490" r="2.5" />
                    <circle cx="370" cy="240" r="2.5" />
                    <circle cx="620" cy="310" r="2.5" />
                    <circle cx="260" cy="340" r="2.5" />
                    <circle cx="280" cy="410" r="2.2" />
                    <circle cx="580" cy="490" r="2.2" />

                    {/* Outer Nodes */}
                    <circle cx="660" cy="210" r="2.2" />
                    <circle cx="690" cy="420" r="2.2" />
                    <circle cx="230" cy="200" r="2.2" />
                    <circle cx="630" cy="530" r="2" />
                    <circle cx="220" cy="520" r="2" />
                    <circle cx="430" cy="150" r="2" />
                    <circle cx="430" cy="580" r="2" />
                    <circle cx="170" cy="365" r="2" />
                    <circle cx="710" cy="365" r="2" />
                  </g>
                </svg>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* -------------------------------------------------------------------- */}
      {/* Modal: Methodology Pipeline View */}
      {/* -------------------------------------------------------------------- */}
      {activeModal === "methodology" && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md">
          <div className="relative w-full max-w-2xl bg-slate-900/95 border border-slate-700/80 rounded-2xl p-6 sm:p-8 shadow-2xl text-slate-100">
            {/* Close Button */}
            <button
              onClick={() => setActiveModal(null)}
              className="absolute top-5 right-5 p-1.5 text-slate-400 hover:text-slate-100 rounded-lg hover:bg-slate-800 transition-colors"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6 6 18M6 6l12 12" />
              </svg>
            </button>

            <div className="flex items-center gap-2 text-xs font-mono text-emerald-400 uppercase tracking-widest mb-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span>Pipeline Architecture</span>
            </div>

            <h3 className="text-2xl font-bold tracking-tight text-slate-100 mb-6 font-sans">
              ThermalEye 5-Stage Intelligence Engine
            </h3>

            <div className="space-y-4 text-xs font-mono text-slate-300">
              <div className="p-3.5 rounded-xl bg-slate-800/60 border border-slate-700/60">
                <div className="flex items-center justify-between text-emerald-400 font-bold mb-1">
                  <span>STAGE 1: FIRMS SATELLITE INGESTION</span>
                  <span className="text-slate-500">VIIRS 375m</span>
                </div>
                <p className="text-slate-400 font-sans text-xs">
                  Continuous polling of NASA VIIRS active fire data across India. Clipped precisely to the official Survey of India boundary.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-800/60 border border-slate-700/60">
                <div className="flex items-center justify-between text-emerald-400 font-bold mb-1">
                  <span>STAGE 2: SPATIOTEMPORAL DBSCAN</span>
                  <span className="text-slate-500">2km / 12h</span>
                </div>
                <p className="text-slate-400 font-sans text-xs">
                  Groups multi-sensor detections into discrete candidate thermal threat events using spatial neighborhood epsilon and sliding temporal windows.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-800/60 border border-slate-700/60">
                <div className="flex items-center justify-between text-emerald-400 font-bold mb-1">
                  <span>STAGE 3: OSM INFRASTRUCTURE ENRICHMENT</span>
                  <span className="text-slate-500">Overpass QL</span>
                </div>
                <p className="text-slate-400 font-sans text-xs">
                  Identifies nearby high-voltage grid corridors, substations, industrial plants, and rail links with rate-limit resilient circuit breakers.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-800/60 border border-slate-700/60">
                <div className="flex items-center justify-between text-emerald-400 font-bold mb-1">
                  <span>STAGE 4: SENTINEL-2 L2A STAC IMAGERY</span>
                  <span className="text-slate-500">Planetary Computer</span>
                </div>
                <p className="text-slate-400 font-sans text-xs">
                  Fetches 10-meter resolution true-color (B04, B03, B02) and false-color infrared satellite imagery for visual confirmation.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-800/60 border border-slate-700/60">
                <div className="flex items-center justify-between text-emerald-400 font-bold mb-1">
                  <span>STAGE 5: ML LAND-COVER CLASSIFICATION</span>
                  <span className="text-slate-500">MobileNetV2</span>
                </div>
                <p className="text-slate-400 font-sans text-xs">
                  Classifies post-fire burn footprints into Forest, Cropland/Stubble, Industrial, Shrubland, or Urban categories.
                </p>
              </div>
            </div>

            <div className="mt-6 flex justify-end">
              <button
                onClick={() => {
                  setActiveModal(null);
                  onEnterDashboard();
                }}
                className="px-6 py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs uppercase tracking-wider transition-colors cursor-pointer"
              >
                Launch Dashboard
              </button>
            </div>
          </div>
        </div>
      )}

      {/* -------------------------------------------------------------------- */}
      {/* Modal: About View */}
      {/* -------------------------------------------------------------------- */}
      {activeModal === "about" && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md">
          <div className="relative w-full max-w-xl bg-slate-900/95 border border-slate-700/80 rounded-2xl p-6 sm:p-8 shadow-2xl text-slate-100">
            <button
              onClick={() => setActiveModal(null)}
              className="absolute top-5 right-5 p-1.5 text-slate-400 hover:text-slate-100 rounded-lg hover:bg-slate-800 transition-colors"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6 6 18M6 6l12 12" />
              </svg>
            </button>

            <div className="flex items-center gap-2 text-xs font-mono text-emerald-400 uppercase tracking-widest mb-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span>Mission Architecture</span>
            </div>

            <h3 className="text-2xl font-bold tracking-tight text-slate-100 mb-4 font-sans">
              About ThermalEye
            </h3>

            <div className="space-y-4 text-xs font-mono text-slate-300 leading-relaxed">
              <p>
                ThermalEye is an open, high-frequency satellite situational awareness system purpose-built for India.
                It detects thermal anomalies, forest fires, agricultural residue burning, and industrial flares in near-real-time.
              </p>
              <div className="p-3.5 rounded-xl bg-slate-800/60 border border-cyan-500/30">
                <div className="font-bold text-cyan-300 mb-1">SOVEREIGN BORDER RECOGNITION</div>
                <p className="text-slate-400 font-sans text-xs">
                  Every geospatial query is aligned with the sovereign international boundary of India as recognized by the Survey of India, including complete demarcations of Jammu &amp; Kashmir and Ladakh.
                </p>
              </div>
              <p className="text-slate-400">
                Tech Stack: React 18, Deck.gl, MapLibre GL, Tailwind CSS, FastAPI, PostgreSQL 15 + PostGIS 3.4, Docker Compose.
              </p>
            </div>

            <div className="mt-6 flex justify-end">
              <button
                onClick={() => {
                  setActiveModal(null);
                  onEnterDashboard();
                }}
                className="px-6 py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs uppercase tracking-wider transition-colors cursor-pointer"
              >
                Explore Live Map
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
