"use client";

import React, { useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { useRiskStore, GreeksData } from "../store/useRiskStore";
import { useWebSocket } from "../hooks/useWebSocket";
import { checkArbitrage } from "../utils/arbitrage";
import VolSurfaceWebGL from "../components/VolSurfaceWebGL";
import {
  Activity,
  AlertTriangle,
  Play,
  Square,
  TrendingUp,
  Percent,
  Clock,
  Shield,
  Zap,
  RefreshCw,
  Sliders,
} from "lucide-react";

const TGrid = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0];
const KGrid = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5];

// Helper to generate a default/mock surface for initial preview
function generatePreviewSurface(spot: number, modelName: string, params: Record<string, number | number[]>): GreeksData {
  const nT = TGrid.length;
  const nK = KGrid.length;

  const iv_surface: number[][] = [];
  const delta: number[][] = [];
  const gamma: number[][] = [];
  const vega: number[][] = [];
  const vanna: number[][] = [];
  const volga: number[][] = [];

  const v0 = params.v0 as number | undefined;
  const alpha = params.alpha as number | undefined;
  const H = (params.H as number) || 0.08;
  const baseVol = v0 ? Math.sqrt(v0) : alpha || 0.20;

  for (let i = 0; i < nT; i++) {
    const t = TGrid[i];
    const ivRow: number[] = [];
    const deltaRow: number[] = [];
    const gammaRow: number[] = [];
    const vegaRow: number[] = [];
    const vannaRow: number[] = [];
    const volgaRow: number[] = [];

    for (let j = 0; j < nK; j++) {
      const k = KGrid[j];
      
      // Generate a mock smile using a simple formula: baseVol + smile + term structure skew
      const timeDecay = Math.pow(t, H - 0.5);
      const skew = -0.15 * k * timeDecay;
      const smile = 0.35 * k * k * timeDecay;
      const vol = Math.max(0.01, baseVol + skew + smile);
      
      ivRow.push(vol);

      // Simple mock Greeks for visualization
      deltaRow.push(0.5 + 0.5 * Math.sin(k) * Math.exp(-0.2 * t));
      gammaRow.push(Math.exp(-0.5 * k * k) / (vol * Math.sqrt(t)));
      vegaRow.push(spot * Math.sqrt(t) * Math.exp(-0.5 * k * k));
      vannaRow.push(-Math.sin(k) * Math.exp(-0.5 * t));
      volgaRow.push(spot * Math.sqrt(t) * k * k);
    }
    iv_surface.push(ivRow);
    delta.push(deltaRow);
    gamma.push(gammaRow);
    vega.push(vegaRow);
    vanna.push(vannaRow);
    volga.push(volgaRow);
  }

  return {
    iv_surface,
    delta,
    gamma,
    vega,
    vanna,
    volga,
  };
}

export default function Dashboard() {
  const {
    currency,
    modelName,
    spot,
    parameters,
    latencyMs,
    connectionStatus,
    isStreaming,
    greeks,
    stressResult,
    stressSpot,
    setCurrency,
    setModelName,
    setParameters,
    resetStore,
  } = useRiskStore(
    useShallow((state) => ({
      currency: state.currency,
      modelName: state.modelName,
      spot: state.spot,
      parameters: state.parameters,
      latencyMs: state.latencyMs,
      connectionStatus: state.connectionStatus,
      isStreaming: state.isStreaming,
      greeks: state.greeks,
      stressResult: state.stressResult,
      stressSpot: state.stressSpot,
      setCurrency: state.setCurrency,
      setModelName: state.setModelName,
      setParameters: state.setParameters,
      resetStore: state.resetStore,
    }))
  );

  const { connect, disconnect, runStressTest } = useWebSocket();

  // Local UI states
  const [surfaceTab, setSurfaceTab] = useState<"iv_surface" | "delta" | "gamma" | "vega" | "vanna" | "volga" >("iv_surface");
  const [showViolations, setShowViolations] = useState(true);
  const [viewMode, setViewMode] = useState<"live" | "stress">("live");
  
  // Stress test inputs
  const [stressShock, setStressShock] = useState("-10");
  const [stressR, setStressR] = useState("0.05");
  const [stressQ, setStressQ] = useState("0.00");

  // Generate preview data if there's no live or stress data yet
  const displayGreeks = viewMode === "live" 
    ? (greeks || generatePreviewSurface(spot, modelName, parameters))
    : (stressResult || generatePreviewSurface(stressSpot || spot, modelName, parameters));

  const activeSpot = viewMode === "live" ? spot : (stressSpot || spot);

  // Compute arbitrage violations on display IV surface
  const arbResult = displayGreeks.iv_surface
    ? checkArbitrage(displayGreeks.iv_surface, TGrid, KGrid)
    : {
        hasCalendarArb: false,
        hasButterflyArb: false,
        calendarViolations: Array.from({ length: 8 }, () => Array(11).fill(false)),
        butterflyViolations: Array.from({ length: 8 }, () => Array(11).fill(false)),
        gValues: Array.from({ length: 8 }, () => Array(11).fill(1)),
      };

  const handleParamChange = (key: string, val: number) => {
    // Cast values to numbers for safety since sliders output numbers
    const updated: Record<string, number | number[]> = {};
    Object.keys(parameters).forEach((k) => {
      updated[k] = k === key ? val : (parameters[k] as number);
    });
    
    setParameters(updated);
    
    // If streaming, send updated subscription settings
    if (isStreaming) {
      // Cast parameter values to numbers
      const numParams: Record<string, number> = {};
      Object.keys(updated).forEach((k) => {
        numParams[k] = updated[k] as number;
      });
      connect(currency, modelName, numParams);
    }
  };

  const toggleConnection = () => {
    if (connectionStatus === "connected") {
      disconnect();
    } else {
      const numParams: Record<string, number> = {};
      Object.keys(parameters).forEach((k) => {
        numParams[k] = parameters[k] as number;
      });
      connect(currency, modelName, numParams);
    }
  };

  const executeStressTest = () => {
    const shockMultiplier = 1 + parseFloat(stressShock) / 100;
    const targetSpot = spot * shockMultiplier;
    const numParams: Record<string, number> = {};
    Object.keys(parameters).forEach((k) => {
      numParams[k] = parameters[k] as number;
    });
    runStressTest(modelName, numParams, targetSpot, parseFloat(stressR), parseFloat(stressQ));
    setViewMode("stress");
  };

  const resetAll = () => {
    disconnect();
    resetStore();
    setViewMode("live");
    setSurfaceTab("iv_surface");
  };

  // Define parameter meta definitions for sliders
  const parameterMeta: Record<string, Record<string, { label: string; min: number; max: number; step: number }>> = {
    rough_heston: {
      kappa: { label: "Mean Reversion (κ)", min: 0.2, max: 8.0, step: 0.1 },
      theta: { label: "Long-run Var (θ)", min: 0.02, max: 0.4, step: 0.01 },
      sigma: { label: "Vol of Vol (σ)", min: 0.1, max: 1.8, step: 0.05 },
      rho: { label: "Correlation (ρ)", min: -0.95, max: 0.0, step: 0.05 },
      v0: { label: "Initial Var (v0)", min: 0.02, max: 0.4, step: 0.01 },
      H: { label: "Hurst (H)", min: 0.04, max: 0.2, step: 0.01 },
    },
    heston: {
      kappa: { label: "Mean Reversion (κ)", min: 0.2, max: 8.0, step: 0.1 },
      theta: { label: "Long-run Var (θ)", min: 0.02, max: 0.4, step: 0.01 },
      sigma: { label: "Vol of Vol (σ)", min: 0.1, max: 1.8, step: 0.05 },
      rho: { label: "Correlation (ρ)", min: -0.95, max: 0.0, step: 0.05 },
      v0: { label: "Initial Var (v0)", min: 0.02, max: 0.4, step: 0.01 },
    },
    sabr: {
      alpha: { label: "Initial Vol (α)", min: 0.05, max: 0.8, step: 0.01 },
      rho: { label: "Correlation (ρ)", min: -0.95, max: 0.95, step: 0.05 },
      nu: { label: "Vol of Vol (ν)", min: 0.05, max: 1.5, step: 0.05 },
    },
    ssvi: {
      rho: { label: "Correlation (ρ)", min: -0.95, max: 0.95, step: 0.05 },
      eta: { label: "Slope (η)", min: 0.1, max: 4.0, step: 0.1 },
      gamma: { label: "Power (γ)", min: 0.1, max: 1.5, step: 0.05 },
    },
    rbergomi: {
      v0: { label: "Initial Var (v0)", min: 0.02, max: 0.4, step: 0.01 },
      H: { label: "Hurst (H)", min: 0.02, max: 0.48, step: 0.01 },
      eta: { label: "Vol of Vol (η)", min: 0.1, max: 3.0, step: 0.05 },
      rho: { label: "Correlation (ρ)", min: -0.95, max: 0.0, step: 0.05 },
    },
  };

  return (
    <div className="flex-1 bg-gray-950 text-gray-100 flex flex-col min-h-screen">
      {/* HEADER */}
      <header className="border-b border-gray-900 bg-gray-950/70 backdrop-blur px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center space-x-3">
          <div className="bg-emerald-500/10 p-2 rounded-lg border border-emerald-500/20">
            <TrendingUp className="h-6 w-6 text-emerald-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-emerald-400 to-teal-300 bg-clip-text text-transparent">
              DeepVol Pricing & Risk Hub
            </h1>
            <p className="text-xs text-gray-500 font-mono">
              Vectorized FNO Volatility Calibration Framework
            </p>
          </div>
        </div>

        {/* CONNECTION STATUS & STATS */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 bg-gray-900/50 border border-gray-800/80 px-3 py-1.5 rounded-lg font-mono text-xs">
            <Activity className="h-4 w-4 text-gray-400" />
            <span className="text-gray-400">Status:</span>
            {connectionStatus === "connected" ? (
              <span className="text-emerald-400 flex items-center">
                <span className="w-2 h-2 rounded-full bg-emerald-500 mr-1.5 animate-pulse" />
                Live Stream
              </span>
            ) : connectionStatus === "connecting" ? (
              <span className="text-amber-400 flex items-center">
                <span className="w-2 h-2 rounded-full bg-amber-500 mr-1.5 animate-ping" />
                Connecting
              </span>
            ) : (
              <span className="text-gray-500 flex items-center">
                <span className="w-2 h-2 rounded-full bg-gray-600 mr-1.5" />
                Offline (Preview)
              </span>
            )}
          </div>

          <div className="flex items-center space-x-2 bg-gray-900/50 border border-gray-800/80 px-3 py-1.5 rounded-lg font-mono text-xs">
            <Clock className="h-4 w-4 text-gray-400" />
            <span className="text-gray-400">Latency:</span>
            <span className="text-teal-400 font-bold">{latencyMs.toFixed(1)} ms</span>
          </div>

          <button
            onClick={resetAll}
            className="p-1.5 bg-gray-900 hover:bg-gray-800 text-gray-400 hover:text-gray-200 border border-gray-800 rounded-lg transition-colors"
            title="Reset store and disconnect"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </header>

      {/* DASHBOARD LAYOUT */}
      <div className="flex-1 flex overflow-hidden">
        {/* SIDEBAR */}
        <aside className="w-80 border-r border-gray-900 bg-gray-950 p-6 space-y-6 overflow-y-auto shrink-0">
          {/* CONTROL SWITCHES */}
          <div className="space-y-4">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Feed Selection
            </h2>
            <div className="grid grid-cols-2 gap-2">
              {["BTC", "ETH"].map((c) => (
                <button
                  key={c}
                  onClick={() => setCurrency(c)}
                  className={`py-2 px-3 rounded-lg border font-medium text-sm transition-all ${
                    currency === c
                      ? "bg-emerald-500/10 border-emerald-500/50 text-emerald-400"
                      : "bg-gray-900/40 border-gray-800 hover:bg-gray-900 text-gray-400"
                  }`}
                >
                  {c} Feed
                </button>
              ))}
            </div>
          </div>

          {/* CALIBRATION MODEL */}
          <div className="space-y-4">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Calibration Model
            </h2>
            <select
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="w-full bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm text-gray-300 focus:outline-none focus:border-emerald-500/50"
            >
              <option value="rough_heston">Rough Heston (6D)</option>
              <option value="heston">Classic Heston (5D)</option>
              <option value="sabr">SABR Model</option>
              <option value="ssvi">SSVI Surface</option>
              <option value="rbergomi">rough Bergomi</option>
            </select>
          </div>

          {/* PARAMETER SLIDERS */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center">
                <Sliders className="h-3.5 w-3.5 mr-1 text-gray-400" />
                Model Parameters
              </h2>
              <span className="text-[10px] text-gray-500 font-mono">
                {isStreaming ? "Auto-updating" : "Manual skew"}
              </span>
            </div>

            <div className="space-y-4 bg-gray-900/30 border border-gray-900/80 p-4 rounded-xl">
              {Object.keys(parameterMeta[modelName] || {}).map((key) => {
                const meta = parameterMeta[modelName][key];
                const value = (parameters[key] as number) ?? meta.min;
                return (
                  <div key={key} className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-gray-400">{meta.label}</span>
                      <span className="text-emerald-400 font-bold">
                        {value.toFixed(meta.step < 0.05 ? 3 : 2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={meta.min}
                      max={meta.max}
                      step={meta.step}
                      value={value}
                      onChange={(e) => handleParamChange(key, parseFloat(e.target.value))}
                      className="w-full accent-emerald-500 h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                );
              })}
            </div>
          </div>

          {/* STREAM SWITCH */}
          <div className="pt-2">
            <button
              onClick={toggleConnection}
              className={`w-full flex items-center justify-center space-x-2 py-3 px-4 rounded-xl border font-semibold text-sm transition-all shadow-lg ${
                connectionStatus === "connected"
                  ? "bg-red-500/10 border-red-500/40 text-red-400 hover:bg-red-500/20"
                  : "bg-emerald-500/15 border-emerald-500/40 text-emerald-400 hover:bg-emerald-500/25"
              }`}
            >
              {connectionStatus === "connected" ? (
                <>
                  <Square className="h-4 w-4 fill-current" />
                  <span>Disconnect Feed</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-current" />
                  <span>Stream Live Data</span>
                </>
              )}
            </button>
          </div>

          {/* SCENARIO STRESS TEST */}
          <div className="space-y-4 pt-4 border-t border-gray-900">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center">
              <Shield className="h-3.5 w-3.5 mr-1 text-gray-400" />
              Scenario Stress Test
            </h2>

            <div className="space-y-3 bg-gray-900/30 border border-gray-900/80 p-4 rounded-xl">
              <div className="space-y-1">
                <label className="text-[10px] font-mono text-gray-400">Spot Shock (%)</label>
                <select
                  value={stressShock}
                  onChange={(e) => setStressShock(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg px-2.5 py-1.5 text-xs text-gray-300 focus:outline-none"
                >
                  <option value="-20">-20% Market Crash</option>
                  <option value="-10">-10% Downturn</option>
                  <option value="-5">-5% Slight Dip</option>
                  <option value="5">+5% Upward Bump</option>
                  <option value="10">+10% Rally</option>
                  <option value="20">+20% Short Squeeze</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <label className="text-[10px] font-mono text-gray-400">Rate (r)</label>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="0.2"
                    value={stressR}
                    onChange={(e) => setStressR(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg px-2.5 py-1.5 text-xs text-gray-300 focus:outline-none"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] font-mono text-gray-400">Yield (q)</label>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="0.2"
                    value={stressQ}
                    onChange={(e) => setStressQ(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg px-2.5 py-1.5 text-xs text-gray-300 focus:outline-none"
                  />
                </div>
              </div>

              <button
                onClick={executeStressTest}
                disabled={connectionStatus !== "connected"}
                className="w-full py-2 px-3 bg-teal-500/10 border border-teal-500/30 text-teal-400 hover:bg-teal-500/20 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium text-xs transition-colors flex items-center justify-center space-x-1.5"
              >
                <Zap className="h-3.5 w-3.5" />
                <span>Run stress scenario</span>
              </button>
            </div>
          </div>
        </aside>

        {/* MAIN PANEL */}
        <main className="flex-1 bg-gray-950 flex flex-col p-6 space-y-6 overflow-y-auto">
          {/* LIVE METRICS CARDS */}
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-gray-900/20 border border-gray-900 p-4 rounded-xl flex items-center space-x-4">
              <div className="bg-emerald-500/10 p-3 rounded-lg">
                <Percent className="h-6 w-6 text-emerald-400" />
              </div>
              <div>
                <p className="text-xs text-gray-500 font-medium">Underlying Spot</p>
                <p className="text-xl font-bold text-gray-100 font-mono">
                  ${activeSpot.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </p>
              </div>
            </div>

            <div className="bg-gray-900/20 border border-gray-900 p-4 rounded-xl flex items-center space-x-4">
              <div className="bg-purple-500/10 p-3 rounded-lg">
                <Sliders className="h-6 w-6 text-purple-400" />
              </div>
              <div>
                <p className="text-xs text-gray-500 font-medium">Hurst Parameter (H)</p>
                <p className="text-xl font-bold text-gray-100 font-mono">
                  {parameters.H ? (parameters.H as number).toFixed(3) : "N/A"}
                </p>
              </div>
            </div>

            <div className="bg-gray-900/20 border border-gray-900 p-4 rounded-xl flex items-center space-x-4">
              <div className="bg-amber-500/10 p-3 rounded-lg">
                <AlertTriangle className="h-6 w-6 text-amber-400" />
              </div>
              <div>
                <p className="text-xs text-gray-500 font-medium">Arbitrage Violations</p>
                <p className="text-xl font-bold font-mono text-gray-100">
                  {arbResult.hasCalendarArb || arbResult.hasButterflyArb ? (
                    <span className="text-red-400">Breach Detected</span>
                  ) : (
                    <span className="text-emerald-400">None</span>
                  )}
                </p>
              </div>
            </div>

            <div className="bg-gray-900/20 border border-gray-900 p-4 rounded-xl flex items-center space-x-4">
              <div className="bg-blue-500/10 p-3 rounded-lg">
                <Activity className="h-6 w-6 text-blue-400" />
              </div>
              <div>
                <p className="text-xs text-gray-500 font-medium">Model Type</p>
                <p className="text-lg font-bold text-gray-100 uppercase tracking-tight">
                  {modelName.replace("_", " ")}
                </p>
              </div>
            </div>
          </div>

          {/* DUAL MODE TABS: LIVE VS STRESS VIEW */}
          <div className="flex justify-between items-center bg-gray-900/40 border border-gray-900 px-4 py-2 rounded-xl">
            <div className="flex space-x-2">
              <button
                onClick={() => setViewMode("live")}
                className={`py-1.5 px-4 rounded-lg text-xs font-semibold transition-all ${
                  viewMode === "live"
                    ? "bg-gray-800 text-gray-100 border border-gray-700"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                Live Calibrated Surface
              </button>
              <button
                onClick={() => setViewMode("stress")}
                disabled={!stressResult}
                className={`py-1.5 px-4 rounded-lg text-xs font-semibold transition-all flex items-center space-x-1.5 ${
                  viewMode === "stress"
                    ? "bg-gray-800 text-gray-100 border border-gray-700"
                    : "text-gray-400 hover:text-gray-200 disabled:opacity-30 disabled:hover:text-gray-400"
                }`}
              >
                <span>Stressed Surface</span>
                {stressResult && <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />}
              </button>
            </div>

            <div className="flex items-center space-x-4">
              <label className="flex items-center space-x-2 text-xs text-gray-400 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showViolations}
                  onChange={(e) => setShowViolations(e.target.checked)}
                  className="rounded border-gray-850 accent-emerald-500 bg-gray-900"
                />
                <span>Highlight Arbitrage Breaches</span>
              </label>
            </div>
          </div>

          {/* MAIN VISUALIZATION CARD */}
          <div className="bg-gray-900/10 border border-gray-900/60 rounded-2xl flex flex-col flex-1 overflow-hidden min-h-[600px]">
            {/* SURFACE TABS */}
            <div className="flex border-b border-gray-900 bg-gray-950/40 p-2 overflow-x-auto space-x-1 shrink-0">
              {[
                { id: "iv_surface", label: "Implied Vol (σ)" },
                { id: "delta", label: "Delta (Δ)" },
                { id: "gamma", label: "Gamma (Γ)" },
                { id: "vega", label: "Vega (ν)" },
                { id: "vanna", label: "Vanna (dΔ/dσ)" },
                { id: "volga", label: "Volga (dν/dσ)" },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setSurfaceTab(tab.id as "iv_surface" | "delta" | "gamma" | "vega" | "vanna" | "volga")}
                  className={`py-2 px-4 rounded-lg text-xs font-mono font-medium transition-all shrink-0 ${
                    surfaceTab === tab.id
                      ? "bg-emerald-500/10 border border-emerald-500/30 text-emerald-400"
                      : "text-gray-400 hover:text-gray-200 border border-transparent"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* WEBGL PLOT CONTAINER */}
            <div className="flex-1 flex items-center justify-center p-4 bg-gray-950/20 relative">
              <VolSurfaceWebGL
                surfaceType={surfaceTab}
                zData={displayGreeks[surfaceTab]}
                TGrid={TGrid}
                KGrid={KGrid}
                showViolations={showViolations}
                calendarViolations={arbResult.calendarViolations}
                butterflyViolations={arbResult.butterflyViolations}
              />
            </div>
          </div>

          {/* ARBITRAGE MONITOR PANEL */}
          <div className="bg-gray-900/10 border border-gray-900/60 p-6 rounded-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-gray-900 pb-3">
              <h3 className="text-sm font-semibold text-gray-200 flex items-center">
                <Shield className="h-4 w-4 mr-2 text-emerald-400" />
                Static Arbitrage Risk Guardian
              </h3>
              <span className="text-xs text-gray-500 font-mono">
                Durrleman & Calendar Spread Metrics
              </span>
            </div>

            <div className="grid grid-cols-2 gap-6">
              {/* Calendar Spread Arbitrage */}
              <div className="space-y-3 bg-gray-950/40 p-4 border border-gray-900 rounded-xl">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-gray-400">Calendar Spread Check</span>
                  {arbResult.hasCalendarArb ? (
                    <span className="bg-red-500/10 border border-red-500/30 text-red-400 text-[10px] font-bold px-2 py-0.5 rounded-full">
                      BREACHED
                    </span>
                  ) : (
                    <span className="bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[10px] font-bold px-2 py-0.5 rounded-full">
                      COMPLIANT
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-500">
                  Total variance $w(k, T) = \sigma^2 T$ must be non-decreasing in maturity ($T$).
                </p>
                <div className="text-xs font-mono text-gray-400">
                  {arbResult.hasCalendarArb ? (
                    <div className="text-red-400/90 space-y-1 max-h-24 overflow-y-auto pr-1">
                      {TGrid.slice(1).map((t, i) => {
                        const idx = i + 1;
                        let count = 0;
                        for (let j = 0; j < KGrid.length; j++) {
                          if (arbResult.calendarViolations[idx][j]) count++;
                        }
                        if (count > 0) {
                          return (
                            <div key={t} className="flex justify-between">
                              <span>Maturity T={t}:</span>
                              <span>{count} strike points decreased in variance</span>
                            </div>
                          );
                        }
                        return null;
                      })}
                    </div>
                  ) : (
                    <span className="text-emerald-500/80">No calendar spread violations detected. ∂w/∂T ≥ 0 holds.</span>
                  )}
                </div>
              </div>

              {/* Butterfly / Strike Arbitrage */}
              <div className="space-y-3 bg-gray-950/40 p-4 border border-gray-900 rounded-xl">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-gray-400">Butterfly Spread Check</span>
                  {arbResult.hasButterflyArb ? (
                    <span className="bg-red-500/10 border border-red-500/30 text-red-400 text-[10px] font-bold px-2 py-0.5 rounded-full">
                      BREACHED
                    </span>
                  ) : (
                    <span className="bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[10px] font-bold px-2 py-0.5 rounded-full">
                      COMPLIANT
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-500">
                  Call option pricing function must be convex in strike, requiring Durrleman density $g(k) \geq 0$.
                </p>
                <div className="text-xs font-mono text-gray-400">
                  {arbResult.hasButterflyArb ? (
                    <div className="text-red-400/90 space-y-1 max-h-24 overflow-y-auto pr-1">
                      {TGrid.map((t, i) => {
                        let count = 0;
                        for (let j = 0; j < KGrid.length; j++) {
                          if (arbResult.butterflyViolations[i][j]) count++;
                        }
                        if (count > 0) {
                          return (
                            <div key={t} className="flex justify-between">
                              <span>Maturity T={t}:</span>
                              <span>{count} strikes violate density convexity</span>
                            </div>
                          );
                        }
                        return null;
                      })}
                    </div>
                  ) : (
                    <span className="text-emerald-500/80">No butterfly violations detected. Implied density g(k) ≥ 0.</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
