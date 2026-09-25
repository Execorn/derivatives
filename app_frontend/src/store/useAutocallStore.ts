import { create } from "zustand";

export interface AutocallPriceResponseData {
  npv: number;
  pde_npv: number;
  correction_bps: number;
  uncertainty_bps: number;
  is_ood: boolean;
  is_fallback: boolean;
  fallback_trigger: string | null;
  fallback_reasons: string[];
  df_dB: number;
  latency_ms: number;
  tau_ood: number;
}

export interface AutocallState {
  // Params
  kappa: number;
  theta: number;
  sigma: number;
  rho: number;
  v0: number;
  B: number;
  coupon: number;
  T: number;
  n_obs_per_year: number;
  r: number;
  useGuardian: boolean;

  // Results
  npv: number | null;
  pdeNpv: number | null;
  correctionBps: number | null;
  uncertaintyBps: number | null;
  isOod: boolean;
  isFallback: boolean;
  fallbackTrigger: string | null;
  fallbackReasons: string[];
  dfDB: number | null;
  latencyMs: number | null;
  tauOod: number | null;

  // UI State
  isLoading: boolean;
  error: string | null;

  // Actions
  setParam: (key: string, value: number) => void;
  setUseGuardian: (v: boolean) => void;
  fetchPrice: () => Promise<void>;
}

export const useAutocallStore = create<AutocallState>((set, get) => ({
  kappa: 2.0,
  theta: 0.04,
  sigma: 0.3,
  rho: -0.7,
  v0: 0.04,
  B: 1.0,
  coupon: 0.10,
  T: 1.5,
  n_obs_per_year: 4.0,
  r: 0.03,
  useGuardian: true,

  npv: null,
  pdeNpv: null,
  correctionBps: null,
  uncertaintyBps: null,
  isOod: false,
  isFallback: false,
  fallbackTrigger: null,
  fallbackReasons: [],
  dfDB: null,
  latencyMs: null,
  tauOod: 2.17,

  isLoading: false,
  error: null,

  setParam: (key, value) => {
    set((state) => ({ ...state, [key]: value }));
  },

  setUseGuardian: (useGuardian) => {
    set({ useGuardian });
  },

  fetchPrice: async () => {
    set({ isLoading: true, error: null });
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
    } = get();

    const payload = {
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
      use_guardian: useGuardian,
    };

    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    try {
      const resp = await fetch(`${baseUrl}/autocall/price`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const errDetail = await resp.text();
        throw new Error(`Pricing API error (${resp.status}): ${errDetail}`);
      }

      const data: AutocallPriceResponseData = await resp.json();

      set({
        npv: data.npv,
        pdeNpv: data.pde_npv,
        correctionBps: data.correction_bps,
        uncertaintyBps: data.uncertainty_bps,
        isOod: data.is_ood,
        isFallback: data.is_fallback,
        fallbackTrigger: data.fallback_trigger,
        fallbackReasons: data.fallback_reasons || [],
        dfDB: data.df_dB,
        latencyMs: data.latency_ms,
        tauOod: data.tau_ood,
        isLoading: false,
        error: null,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      set({ isLoading: false, error: msg });
    }
  },
}));
