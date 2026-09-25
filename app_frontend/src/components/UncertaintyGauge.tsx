"use client";

import React from "react";
import { ShieldCheck, AlertOctagon } from "lucide-react";

interface UncertaintyGaugeProps {
  uncertaintyBps: number | null;
  tauOod: number | null;
  isOod: boolean;
  isFallback?: boolean;
}

export default function UncertaintyGauge({
  uncertaintyBps,
  tauOod = 2.17,
  isOod,
  isFallback,
}: UncertaintyGaugeProps) {
  const currentUnc = uncertaintyBps ?? 0;
  const threshold = tauOod ?? 2.17;
  const maxVal = 5.0;

  // Map value to angle in semicircle [-180, 0] degrees
  const clampedVal = Math.min(Math.max(currentUnc, 0), maxVal);
  const frac = clampedVal / maxVal;
  const angleDeg = -180 + frac * 180;
  const angleRad = (angleDeg * Math.PI) / 180;

  // Threshold angle
  const threshFrac = Math.min(Math.max(threshold, 0), maxVal) / maxVal;
  const threshDeg = -180 + threshFrac * 180;
  const threshRad = (threshDeg * Math.PI) / 180;

  // SVG Geometry
  const cx = 130;
  const cy = 125;
  const r = 90;
  const strokeWidth = 14;

  // Needle coordinates
  const needleLen = 72;
  const needleX = cx + needleLen * Math.cos(angleRad);
  const needleY = cy + needleLen * Math.sin(angleRad);

  // Threshold marker coordinates
  const tInnerX = cx + (r - strokeWidth / 2 - 4) * Math.cos(threshRad);
  const tInnerY = cy + (r - strokeWidth / 2 - 4) * Math.sin(threshRad);
  const tOuterX = cx + (r + strokeWidth / 2 + 6) * Math.cos(threshRad);
  const tOuterY = cy + (r + strokeWidth / 2 + 6) * Math.sin(threshRad);

  // Arc path generator
  const describeArc = (x: number, y: number, radius: number, startAngle: number, endAngle: number) => {
    const start = {
      x: x + radius * Math.cos((startAngle * Math.PI) / 180),
      y: y + radius * Math.sin((startAngle * Math.PI) / 180),
    };
    const end = {
      x: x + radius * Math.cos((endAngle * Math.PI) / 180),
      y: y + radius * Math.sin((endAngle * Math.PI) / 180),
    };
    const largeArc = endAngle - startAngle <= 180 ? "0" : "1";
    return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x} ${end.y}`;
  };

  const bgArc = describeArc(cx, cy, r, -180, 0);

  return (
    <div className="flex flex-col items-center justify-between p-4 bg-zinc-900 border border-zinc-800 rounded-xl shadow-lg">
      <div className="flex items-center justify-between w-full mb-1">
        <span className="text-xs font-mono font-medium tracking-wider text-zinc-400 uppercase">
          Epistemic Uncertainty (σ)
        </span>
        <span className="text-xs font-mono text-zinc-400">
          τ_OOD = {threshold.toFixed(2)} bps
        </span>
      </div>

      <div className="relative flex items-center justify-center my-1">
        <svg width="260" height="145" viewBox="0 0 260 145" className="overflow-visible">
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="25%" stopColor="#22c55e" />
              <stop offset="45%" stopColor="#eab308" />
              <stop offset="65%" stopColor="#f97316" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
          </defs>

          {/* Background track */}
          <path
            d={bgArc}
            fill="none"
            stroke="#27272a"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />

          {/* Colored gradient arc */}
          <path
            d={bgArc}
            fill="none"
            stroke="url(#gaugeGradient)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />

          {/* Threshold marker */}
          <line
            x1={tInnerX}
            y1={tInnerY}
            x2={tOuterX}
            y2={tOuterY}
            stroke="#ffffff"
            strokeWidth="3"
            strokeDasharray="3 2"
          />

          {/* Needle pivot */}
          <circle cx={cx} cy={cy} r="6" fill="#f4f4f5" />

          {/* Needle line */}
          <line
            x1={cx}
            y1={cy}
            x2={needleX}
            y2={needleY}
            stroke="#f4f4f5"
            strokeWidth="3.5"
            strokeLinecap="round"
          />
        </svg>

        {/* Center digital readout */}
        <div className="absolute flex flex-col items-center bottom-1">
          <span className="text-2xl font-bold font-mono text-zinc-100">
            {uncertaintyBps !== null ? `${uncertaintyBps.toFixed(2)}` : "--"}
          </span>
          <span className="text-[11px] font-mono text-zinc-400">basis points (bps)</span>
        </div>
      </div>

      {/* Ticks legend */}
      <div className="flex justify-between w-full px-5 text-[11px] font-mono text-zinc-400">
        <span>0.0 bps</span>
        <span className="text-amber-400 font-medium">τ_OOD = {threshold.toFixed(2)} bps</span>
        <span>5.0 bps</span>
      </div>

      {/* OOD & Governance Status Badge */}
      <div className="w-full mt-3">
        {isFallback ? (
          <div className="flex items-center justify-center gap-2 px-3 py-2 text-xs font-mono font-medium text-amber-300 bg-amber-950/40 border border-amber-800/80 rounded-lg">
            <AlertOctagon className="w-4 h-4 text-amber-400 shrink-0" />
            <span>STATUS: FALLBACK ACTIVE [NUMERICAL PDE ROUTE]</span>
          </div>
        ) : isOod ? (
          <div className="flex items-center justify-center gap-2 px-3 py-2 text-xs font-mono font-medium text-red-300 bg-red-950/40 border border-red-800/80 rounded-lg">
            <AlertOctagon className="w-4 h-4 text-red-400 shrink-0" />
            <span>STATUS: OUT-OF-DISTRIBUTION [σ &gt; τ_OOD]</span>
          </div>
        ) : (
          <div className="flex items-center justify-center gap-2 px-3 py-2 text-xs font-mono font-medium text-emerald-300 bg-emerald-950/40 border border-emerald-800/80 rounded-lg">
            <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>STATUS: IN-DISTRIBUTION [σ ≤ τ_OOD]</span>
          </div>
        )}
      </div>
    </div>
  );
}
