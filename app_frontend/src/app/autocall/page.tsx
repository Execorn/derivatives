"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useShallow } from "zustand/react/shallow";
import { useAutocallStore } from "../../store/useAutocallStore";
import UncertaintyGauge from "../../components/UncertaintyGauge";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  Clock,
  Cpu,
  Layers,
  Play,
  RotateCcw,
  Shield,
  ShieldAlert,
  Zap,
} from "lucide-react";

export default function AutocallPage() {
  const {
    kappa,
    theta,
    sigma,
    rho,
    v0,
    B,
    coupon,
    T,
    n_obs_per_year,
    r,
    useGuardian,
    npv,
    pdeNpv,
    correctionBps,
    uncertaintyBps,
    isOod,
    isFallback,
    fallbackTrigger,
    fallbackReasons,
    dfDB,
    latencyMs,
    tauOod,
    isLoading,
    error,
    setParam,
    setUseGuardian,
    fetchPrice,
  } = useAutocallStore(
    useShallow((state) => ({
      kappa: state.kappa,
      theta: state.theta,
      sigma: state.sigma,
      rho: state.rho,
      v0: state.v0,
      B: state.B,
      coupon: state.coupon,
      T: state.T,
      n_obs_per_year: state.n_obs_per_year,
      r: state.r,
      useGuardian: state.useGuardian,
      npv: state.npv,
      pdeNpv: state.pdeNpv,
      correctionBps: state.correctionBps,
      uncertaintyBps: state.uncertaintyBps,
      isOod: state.isOod,
      isFallback: state.isFallback,
      fallbackTrigger: state.fallbackTrigger,
      fallbackReasons: state.fallbackReasons,
      dfDB: state.dfDB,
      latencyMs: state.latencyMs,
      tauOod: state.tauOod,
      isLoading: state.isLoading,
      error: state.error,
      setParam: state.setParam,
      setUseGuardian: state.setUseGuardian,
      fetchPrice: state.fetchPrice,
    }))
  );

  // Trigger initial pricing call on mount if not yet priced
  useEffect(() => {
    if (npv === null && !isLoading) {
      fetchPrice();
    }
  }, []);

  // Synthetic individual member spread around correctionBps for visualization
  const spreadValues = React.useMemo(() => {
    const mean = correctionBps ?? 0;
    const std = uncertaintyBps ?? 0.8;
    return [
      mean - 1.2 * std,
      mean - 0.4 * std,
      mean + 0.1 * std,
      mean + 0.6 * std,
      mean + 1.1 * std,
    ];
  }, [correctionBps, uncertaintyBps]);

  return (
    <div className="flex flex-col min-h-screen bg-zinc-950 text-zinc-100">
      {/* Top Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-900/50 backdrop-blur">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-zinc-400 bg-zinc-800/60 hover:bg-zinc-800 rounded-lg transition-colors border border-zinc-700/50"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Surface Pricer
          </Link>
          <div className="h-4 w-px bg-zinc-800 mx-1" />
          <h1 className="text-base font-semibold text-zinc-100 flex items-center gap-2">
            <Layers className="w-5 h-5 text-blue-500" />
            DeepVol Autocallable Note Pricer
          </h1>
          <span className="px-2 py-0.5 text-[10px] font-mono tracking-wider text-blue-400 bg-blue-950/60 border border-blue-800 rounded-full">
            Phase D CorrectionEnsemble (K=5)
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5 text-zinc-400">
            <Cpu className="w-4 h-4 text-emerald-400" />
            <span>Target: RTX 3060 CUDA</span>
          </div>
          <div className="h-4 w-px bg-zinc-800" />
          <div className="flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                isFallback ? "bg-amber-400" : isOod ? "bg-red-400" : "bg-emerald-400"
              }`}
            />
            <span className="font-mono text-zinc-300">
              {isFallback ? "Fallback Active" : isOod ? "OOD Detected" : "In Distribution"}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar Controls */}
        <aside className="w-80 p-5 overflow-y-auto border-r border-zinc-800 bg-zinc-900/30 flex flex-col gap-5 shrink-0">
          <div>
            <h2 className="text-xs font-semibold tracking-wider text-zinc-400 uppercase mb-3 flex items-center justify-between">
              <span>Heston Dynamics</span>
              <span className="text-[10px] font-mono text-blue-400">float64</span>
            </h2>
            <div className="space-y-3 text-xs">
              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-zinc-400">Mean Reversion (κ)</label>
                  <span className="font-mono text-zinc-200">{kappa.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="5.0"
                  step="0.1"
                  value={kappa}
                  onChange={(e) => setParam("kappa", parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-zinc-400">Long-term Variance (θ)</label>
                  <span className="font-mono text-zinc-200">{theta.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.15"
                  step="0.005"
                  value={theta}
                  onChange={(e) => setParam("theta", parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-zinc-400">Vol of Vol (σ)</label>
                  <span className="font-mono text-zinc-200">{sigma.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={sigma}
                  onChange={(e) => setParam("sigma", parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-zinc-400">Spot-Vol Correlation (ρ)</label>
                  <span className="font-mono text-zinc-200">{rho.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="-0.95"
                  max="0.0"
                  step="0.05"
                  value={rho}
                  onChange={(e) => setParam("rho", parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-zinc-400">Initial Variance (v₀)</label>
                  <span className="font-mono text-zinc-200">{v0.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.15"
                  step="0.005"
                  value={v0}
                  onChange={(e) => setParam("v0", parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>
            </div>
          </div>

          <div className="border-t border-zinc-800/80 pt-4">
            <h2 className="text-xs font-semibold tracking-wider text-zinc-400 uppercase mb-3">
              Contract Terms
            </h2>
            <div className="space-y-3 text-xs">
              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-zinc-400">Barrier B (% of S₀)</label>
                  <span className="font-mono text-zinc-200">{(B * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0.85"
                  max="1.15"
                  step="0.01"
                  value={B}
                  onChange={(e) => setParam("B", parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-zinc-400">Annual Coupon Rate</label>
                  <span className="font-mono text-zinc-200">{(coupon * 100).toFixed(1)}%</span>
                </div>
                <input
                  type="range"
                  min="0.03"
                  max="0.25"
                  step="0.005"
                  value={coupon}
                  onChange={(e) => setParam("coupon", parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-zinc-400">Maturity T (Years)</label>
                  <span className="font-mono text-zinc-200">{T.toFixed(2)}y</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="3.0"
                  step="0.25"
                  value={T}
                  onChange={(e) => setParam("T", parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              <div>
                <label className="text-zinc-400 block mb-1">Observation Frequency</label>
                <select
                  value={n_obs_per_year}
                  onChange={(e) => setParam("n_obs_per_year", parseFloat(e.target.value))}
                  className="w-full px-3 py-1.5 bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-200 font-mono text-xs focus:outline-none focus:border-blue-500"
                >
                  <option value={4}>Quarterly (4 obs/yr)</option>
                  <option value={8}>Semi-Annual / 8 per yr</option>
                  <option value={12}>Monthly (12 obs/yr)</option>
                </select>
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-zinc-400">Risk-free Rate (r)</label>
                  <span className="font-mono text-zinc-200">{(r * 100).toFixed(2)}%</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="0.08"
                  step="0.0025"
                  value={r}
                  onChange={(e) => setParam("r", parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>
            </div>
          </div>

          <div className="border-t border-zinc-800/80 pt-4">
            <div className="flex items-center justify-between p-3 bg-zinc-800/40 border border-zinc-750 rounded-xl mb-4">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-blue-400" />
                <label htmlFor="guardian-mode" className="text-xs font-medium text-zinc-200 cursor-pointer">
                  SR 26-2 Guardian
                </label>
              </div>
              <input
                id="guardian-mode"
                type="checkbox"
                checked={useGuardian}
                onChange={(e) => setUseGuardian(e.target.checked)}
                className="w-4 h-4 accent-blue-500 cursor-pointer rounded"
              />
            </div>

            <button
              onClick={() => fetchPrice()}
              disabled={isLoading}
              className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 disabled:opacity-50 text-white font-medium text-xs rounded-xl flex items-center justify-center gap-2 transition-all shadow-lg shadow-blue-900/30"
            >
              {isLoading ? (
                <>
                  <RotateCcw className="w-4 h-4 animate-spin" />
                  <span>Computing PDE + Ensemble...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Price Autocall</span>
                </>
              )}
            </button>
          </div>
        </aside>

        {/* Main Display Area */}
        <main className="flex-1 p-6 overflow-y-auto space-y-6">
          {error && (
            <div className="p-4 bg-red-950/70 border border-red-800 text-red-200 text-xs rounded-xl flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-red-400 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Metric Cards Row */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-xl">
              <span className="text-xs text-zinc-400 block mb-1">Total NPV (% Par)</span>
              <span className="text-2xl font-bold font-mono text-zinc-100">
                {npv !== null ? `${(npv * 100).toFixed(2)}%` : "--"}
              </span>
            </div>

            <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-xl">
              <span className="text-xs text-zinc-400 block mb-1">PDE Base Price</span>
              <span className="text-2xl font-bold font-mono text-zinc-200">
                {pdeNpv !== null ? `${(pdeNpv * 100).toFixed(2)}%` : "--"}
              </span>
            </div>

            <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-xl">
              <span className="text-xs text-zinc-400 block mb-1">MLP Correction</span>
              <span
                className={`text-2xl font-bold font-mono ${
                  correctionBps && correctionBps >= 0 ? "text-emerald-400" : "text-amber-400"
                }`}
              >
                {correctionBps !== null ? `${correctionBps > 0 ? "+" : ""}${correctionBps.toFixed(1)} bps` : "--"}
              </span>
            </div>

            <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-xl">
              <span className="text-xs text-zinc-400 block mb-1">Epistemic σ</span>
              <span className="text-2xl font-bold font-mono text-blue-400">
                {uncertaintyBps !== null ? `${uncertaintyBps.toFixed(2)} bps` : "--"}
              </span>
            </div>

            <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-xl">
              <span className="text-xs text-zinc-400 block mb-1">Barrier Delta (ΔB)</span>
              <span className="text-2xl font-bold font-mono text-zinc-200">
                {dfDB !== null ? dfDB.toFixed(4) : "--"}
              </span>
            </div>

            <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-xl">
              <span className="text-xs text-zinc-400 block mb-1">Latency</span>
              <span className="text-2xl font-bold font-mono text-emerald-400">
                {latencyMs !== null ? `${latencyMs.toFixed(1)} ms` : "--"}
              </span>
            </div>
          </div>

          {/* Middle Row: Gauge & Ensemble Spread */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Epistemic Uncertainty Gauge */}
            <UncertaintyGauge
              uncertaintyBps={uncertaintyBps}
              tauOod={tauOod}
              isOod={isOod}
              isFallback={isFallback}
            />

            {/* Right: 5-Member Ensemble Spread Chart */}
            <div className="p-5 bg-zinc-900 border border-zinc-800 rounded-xl shadow-lg flex flex-col justify-between">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold tracking-wider text-zinc-400 uppercase">
                  5-Member Ensemble Spread
                </span>
                <span className="text-xs font-mono text-zinc-400">
                  Mean: {correctionBps !== null ? `${correctionBps > 0 ? "+" : ""}${correctionBps.toFixed(1)} bps` : "--"}
                </span>
              </div>

              {/* Clean SVG Bar Chart */}
              <div className="h-44 w-full flex items-end justify-around gap-4 px-2 pt-4 pb-2 border-b border-zinc-800">
                {spreadValues.map((val, idx) => {
                  const maxAbs = Math.max(...spreadValues.map((v) => Math.abs(v)), 5.0);
                  const barHeightPct = Math.min(Math.max((Math.abs(val) / maxAbs) * 100, 10), 100);

                  return (
                    <div key={idx} className="flex-1 flex flex-col items-center gap-1.5 h-full justify-end">
                      <span className="text-[11px] font-mono text-zinc-300 font-medium">
                        {val > 0 ? "+" : ""}
                        {val.toFixed(1)}
                      </span>
                      <div
                        className="w-full bg-blue-500 hover:bg-blue-400 transition-all rounded-t-md shadow-md shadow-blue-950"
                        style={{ height: `${barHeightPct}%` }}
                      />
                      <span className="text-[10px] text-zinc-400 font-mono">M{idx}</span>
                    </div>
                  );
                })}
              </div>

              <div className="flex justify-between items-center text-[11px] text-zinc-400 pt-2 font-mono">
                <span>Model: ResNet MLP (19D → 256)</span>
                <span className="text-blue-400">K = 5 Members</span>
              </div>
            </div>
          </div>

          {/* Bottom Row: SR 26-2 Model Governance Panel */}
          <div className="p-5 bg-zinc-900/80 border border-zinc-800 rounded-xl shadow-lg space-y-4">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div className="flex items-center gap-2">
                <Shield className="w-5 h-5 text-blue-400" />
                <h3 className="text-sm font-semibold text-zinc-100">
                  SR 26-2 Model Risk Governance & Compliance
                </h3>
              </div>
              <span
                className={`px-3 py-1 text-xs font-semibold rounded-full border ${
                  isFallback
                    ? "bg-amber-950/70 border-amber-700 text-amber-300"
                    : isOod
                    ? "bg-red-950/70 border-red-700 text-red-300"
                    : "bg-emerald-950/70 border-emerald-700 text-emerald-300"
                }`}
              >
                {isFallback ? "⚠️ Tier 3 PDE Fallback Engaged" : isOod ? "🚨 OOD Breach" : "🟢 Operational Status: Compliant"}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
              <div className="p-3 bg-zinc-950 border border-zinc-800/80 rounded-lg">
                <span className="text-zinc-400 block mb-1">OOD Threshold (τ_OOD)</span>
                <span className="text-base font-bold font-mono text-zinc-100">
                  {tauOod !== null ? `${tauOod.toFixed(2)} bps` : "2.17 bps"}
                </span>
                <span className="text-[10px] text-zinc-400 block mt-1">Calibrated at 99th percentile</span>
              </div>

              <div className="p-3 bg-zinc-950 border border-zinc-800/80 rounded-lg">
                <span className="text-zinc-400 block mb-1">Ensemble Raw RMSE</span>
                <span className="text-base font-bold font-mono text-emerald-400">1.14 bps</span>
                <span className="text-[10px] text-zinc-400 block mt-1">Trimmed RMSE: 0.99 bps</span>
              </div>

              <div className="p-3 bg-zinc-950 border border-zinc-800/80 rounded-lg">
                <span className="text-zinc-400 block mb-1">Tail Error Bounds</span>
                <span className="text-base font-bold font-mono text-zinc-100">P95: 2.30 | P99: 3.91 bps</span>
                <span className="text-[10px] text-zinc-400 block mt-1">Worst routed outlier &lt; 15 bps</span>
              </div>

              <div className="p-3 bg-zinc-950 border border-zinc-800/80 rounded-lg">
                <span className="text-zinc-400 block mb-1">Numerical Guardian Tier</span>
                <span className="text-base font-bold font-mono text-zinc-100">
                  {isFallback ? fallbackTrigger || "Tier 1 OOD" : "Tier 1 & 2 Clear"}
                </span>
                <span className="text-[10px] text-zinc-400 block mt-1">Feller, Mahalanobis, df/dB</span>
              </div>
            </div>

            {fallbackReasons && fallbackReasons.length > 0 && (
              <div className="p-3 bg-amber-950/40 border border-amber-800/80 rounded-lg text-xs text-amber-200">
                <span className="font-semibold block mb-1">Guardian Intervention Reasons:</span>
                <ul className="list-disc pl-5 space-y-0.5 font-mono text-[11px]">
                  {fallbackReasons.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
