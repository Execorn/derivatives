import { describe, it, expect, beforeEach } from "vitest";
import { useRiskStore } from "./useRiskStore";

describe("useRiskStore", () => {
  beforeEach(() => {
    useRiskStore.getState().resetStore();
  });

  it("should initialize with default states", () => {
    const state = useRiskStore.getState();
    expect(state.currency).toBe("BTC");
    expect(state.modelName).toBe("rough_heston");
    expect(state.spot).toBe(65000.0);
    expect(state.connectionStatus).toBe("disconnected");
    expect(state.isStreaming).toBe(false);
    expect(state.greeks).toBeNull();
  });

  it("should update currency and spot price when setCurrency is called", () => {
    useRiskStore.getState().setCurrency("ETH");
    expect(useRiskStore.getState().currency).toBe("ETH");
    expect(useRiskStore.getState().spot).toBe(3500.0);
  });

  it("should update model name and parameters when setModelName is called", () => {
    useRiskStore.getState().setModelName("sabr");
    const state = useRiskStore.getState();
    expect(state.modelName).toBe("sabr");
    expect(state.parameters).toEqual({ alpha: 0.20, rho: -0.40, nu: 0.40 });
  });

  it("should update connection status when setConnectionStatus is called", () => {
    useRiskStore.getState().setConnectionStatus("connected");
    expect(useRiskStore.getState().connectionStatus).toBe("connected");
  });

  it("should update store values from websocket data payload", () => {
    const mockData = {
      spot: 64250.0,
      timestamp: 1718900000.0,
      parameters: { kappa: 2.1, theta: 0.04, H: 0.09 },
      greeks: {
        delta: [[0.5]],
        gamma: [[0.01]],
        vega: [[100.0]],
        vanna: [[-1.2]],
        volga: [[15.5]],
        iv_surface: [[0.22]],
      },
      latency_ms: 2.5,
    };

    useRiskStore.getState().updateFromWebSocket(mockData);

    const state = useRiskStore.getState();
    expect(state.spot).toBe(64250.0);
    expect(state.timestamp).toBe(1718900000.0);
    expect(state.latencyMs).toBe(2.5);
    expect(state.parameters).toEqual(mockData.parameters);
    expect(state.greeks).toEqual(mockData.greeks);
  });
});
