import { create } from "zustand";

export interface GreeksData {
  delta: number[][];
  gamma: number[][];
  vega: number[][];
  vanna: number[][];
  volga: number[][];
  iv_surface: number[][];
}

export interface RiskState {
  currency: string;
  modelName: string;
  parameters: Record<string, number | number[]>;
  spot: number;
  timestamp: number;
  greeks: GreeksData | null;
  latencyMs: number;
  connectionStatus: "disconnected" | "connecting" | "connected";
  isStreaming: boolean;
  stressResult: GreeksData | null;
  stressSpot: number | null;
  
  // Actions
  setCurrency: (currency: string) => void;
  setModelName: (modelName: string) => void;
  setParameters: (params: Record<string, number | number[]>) => void;
  setConnectionStatus: (status: "disconnected" | "connecting" | "connected") => void;
  setIsStreaming: (isStreaming: boolean) => void;
  updateFromWebSocket: (data: {
    spot: number;
    timestamp: number;
    parameters: Record<string, number | number[]>;
    greeks: GreeksData;
    latency_ms: number;
  }) => void;
  setStressResult: (result: GreeksData | null, spot: number | null) => void;
  resetStore: () => void;
}

const initialParams: Record<string, Record<string, number>> = {
  rough_heston: { kappa: 2.0, theta: 0.05, sigma: 0.3, rho: -0.6, v0: 0.05, H: 0.08 },
  heston: { kappa: 2.0, theta: 0.05, sigma: 0.3, rho: -0.6, v0: 0.05 },
  sabr: { alpha: 0.20, rho: -0.40, nu: 0.40 },
  ssvi: { rho: -0.40, eta: 1.0, gamma: 0.5 }, // theta_atm is custom or handled separately, we can also default it
  rbergomi: { v0: 0.08, H: 0.07, eta: 1.5, rho: -0.70 }
};

export const useRiskStore = create<RiskState>((set) => ({
  currency: "BTC",
  modelName: "rough_heston",
  parameters: { ...initialParams.rough_heston },
  spot: 65000.0,
  timestamp: 0,
  greeks: null,
  latencyMs: 0,
  connectionStatus: "disconnected",
  isStreaming: false,
  stressResult: null,
  stressSpot: null,

  setCurrency: (currency) => set({ currency, spot: currency === "BTC" ? 65000.0 : 3500.0 }),
  
  setModelName: (modelName) => set(() => ({
    modelName,
    parameters: { ...(initialParams[modelName.toLowerCase()] || {}) }
  })),
  
  setParameters: (parameters) => set({ parameters }),
  
  setConnectionStatus: (connectionStatus) => set({ connectionStatus }),
  
  setIsStreaming: (isStreaming) => set({ isStreaming }),
  
  updateFromWebSocket: (data) => set({
    spot: data.spot,
    timestamp: data.timestamp,
    parameters: data.parameters,
    greeks: data.greeks,
    latencyMs: data.latency_ms,
  }),

  setStressResult: (stressResult, stressSpot) => set({ stressResult, stressSpot }),
  
  resetStore: () => set({
    currency: "BTC",
    modelName: "rough_heston",
    parameters: { ...initialParams.rough_heston },
    spot: 65000.0,
    timestamp: 0,
    greeks: null,
    latencyMs: 0,
    connectionStatus: "disconnected",
    isStreaming: false,
    stressResult: null,
    stressSpot: null,
  })
}));
