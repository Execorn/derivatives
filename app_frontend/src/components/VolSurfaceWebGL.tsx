"use client";

import React, { useEffect, useRef, useState } from "react";

interface VolSurfaceWebGLProps {
  surfaceType: "iv_surface" | "delta" | "gamma" | "vega" | "vanna" | "volga";
  zData: number[][]; // 2D array of shape (nT, nK)
  TGrid: number[]; // length nT
  KGrid: number[]; // length nK
  showViolations: boolean;
  calendarViolations: boolean[][]; // Shape: (nT, nK)
  butterflyViolations: boolean[][]; // Shape: (nT, nK)
}

interface PlotlyRenderer {
  react: (
    el: HTMLDivElement | null,
    data: Array<Record<string, unknown>>,
    layout: Record<string, unknown>,
    config?: Record<string, unknown>
  ) => void;
}

export default function VolSurfaceWebGL({
  surfaceType,
  zData,
  TGrid,
  KGrid,
  showViolations,
  calendarViolations,
  butterflyViolations,
}: VolSurfaceWebGLProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [plotly, setPlotly] = useState<PlotlyRenderer | null>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  // Dynamically load Plotly on client-side
  useEffect(() => {
    import("plotly.js-dist-min").then((mod) => {
      setPlotly((mod.default || mod) as unknown as PlotlyRenderer);
    });
  }, []);

  // Handle resizing
  useEffect(() => {
    if (!containerRef.current) return;
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height || 500,
        });
      }
    });
    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  useEffect(() => {
    if (!plotly || !containerRef.current || !zData || zData.length === 0) return;

    const nT = TGrid.length;
    const nK = KGrid.length;

    // Surface trace
    const surfaceTrace: Record<string, unknown> = {
      type: "surface",
      x: KGrid,
      y: TGrid,
      z: zData,
      colorscale: surfaceType === "iv_surface" ? "Viridis" : "RdBu",
      showscale: true,
      colorbar: {
        title: {
          text: surfaceType.toUpperCase().replace("_", " "),
          side: "right",
        },
        thickness: 15,
        len: 0.8,
      },
      hovertemplate:
        "Strike (k): %{x:.2f}<br>Maturity (T): %{y:.2f}<br>Value: %{z:.4f}<extra></extra>",
    };

    const traces: Array<Record<string, unknown>> = [surfaceTrace];

    // Violation overlays
    if (showViolations) {
      const calX: number[] = [];
      const calY: number[] = [];
      const calZ: number[] = [];

      const butX: number[] = [];
      const butY: number[] = [];
      const butZ: number[] = [];

      for (let i = 0; i < nT; i++) {
        for (let j = 0; j < nK; j++) {
          if (calendarViolations[i] && calendarViolations[i][j]) {
            calX.push(KGrid[j]);
            calY.push(TGrid[i]);
            calZ.push(zData[i][j]);
          }
          if (butterflyViolations[i] && butterflyViolations[i][j]) {
            butX.push(KGrid[j]);
            butY.push(TGrid[i]);
            butZ.push(zData[i][j]);
          }
        }
      }

      if (calX.length > 0) {
        traces.push({
          type: "scatter3d",
          mode: "markers",
          x: calX,
          y: calY,
          z: calZ,
          name: "Calendar Arb Violation",
          marker: {
            size: 6,
            color: "#EF4444", // red-500
            symbol: "circle",
            line: {
              color: "#FFFFFF",
              width: 1,
            },
          },
          hovertemplate:
            "<b>Calendar Arb Breach</b><br>Strike: %{x:.2f}<br>Maturity: %{y:.2f}<br>IV: %{z:.4f}<extra></extra>",
        });
      }

      if (butX.length > 0) {
        traces.push({
          type: "scatter3d",
          mode: "markers",
          x: butX,
          y: butY,
          z: butZ,
          name: "Butterfly Arb Violation",
          marker: {
            size: 6,
            color: "#F59E0B", // amber-500
            symbol: "diamond",
            line: {
              color: "#FFFFFF",
              width: 1,
            },
          },
          hovertemplate:
            "<b>Butterfly Arb Breach</b><br>Strike: %{x:.2f}<br>Maturity: %{y:.2f}<br>IV: %{z:.4f}<extra></extra>",
        });
      }
    }

    const layout = {
      title: {
        text: `3D WebGL ${surfaceType.toUpperCase().replace("_", " ")} Surface`,
        font: { color: "#F3F4F6", size: 16 }, // gray-100
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      scene: {
        xaxis: {
          title: { text: "Log-Strike (k)", font: { color: "#9CA3AF" } },
          gridcolor: "#374151",
          zerolinecolor: "#4B5563",
          tickfont: { color: "#9CA3AF" },
        },
        yaxis: {
          title: { text: "Maturity (T)", font: { color: "#9CA3AF" } },
          gridcolor: "#374151",
          zerolinecolor: "#4B5563",
          tickfont: { color: "#9CA3AF" },
        },
        zaxis: {
          title: { text: "Value", font: { color: "#9CA3AF" } },
          gridcolor: "#374151",
          zerolinecolor: "#4B5563",
          tickfont: { color: "#9CA3AF" },
        },
        camera: {
          eye: { x: 1.5, y: 1.5, z: 1.2 },
        },
      },
      margin: { l: 0, r: 0, b: 0, t: 40 },
      autosize: true,
      width: dimensions.width || undefined,
      height: dimensions.height || 500,
      showlegend: showViolations,
      legend: {
        x: 0,
        y: 1,
        font: { color: "#9CA3AF" },
        bgcolor: "rgba(17, 24, 39, 0.8)",
        bordercolor: "#374151",
        borderwidth: 1,
      },
    };

    const config = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["toImage", "sendDataToCloud"],
    };

    plotly.react(containerRef.current, traces, layout, config);
  }, [plotly, zData, TGrid, KGrid, showViolations, calendarViolations, butterflyViolations, surfaceType, dimensions]);

  return (
    <div className="relative w-full h-full min-h-[500px]" ref={containerRef}>
      {!plotly && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-50 border border-gray-800 rounded-lg">
          <div className="flex flex-col items-center space-y-3">
            <div className="w-10 h-10 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
            <span className="text-gray-400 text-sm font-medium">Initializing WebGL Engine...</span>
          </div>
        </div>
      )}
    </div>
  );
}
