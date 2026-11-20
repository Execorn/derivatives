from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import importlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from deepvol.api.eq_autocall_arrow import (
    decode_eq_autocall_arrow_response,
    encode_eq_autocall_arrow_request,
)
from deepvol.api.eq_autocall_runtime import (
    DEFAULT_CHECKPOINT_PATH,
    EQAutocallRuntime,
    _load_surrogate_from_ckpt,
)
from deepvol.api.schemas_eq_autocall import (
    EQAutocallableTradeRequest,
    parse_murex_autocall_dag,
)
from deepvol.models.autocall.analytical_baseline import compute_analytical_autocall_baseline
from deepvol.models.autocall.dupire_qmc_solver import price_eq_autocall_dupire_qmc
from deepvol.models.autocall.feature_layout import FEATURE_SCHEMA_VERSION
import deepvol.models.autocall.greeks_engine as _greeks_mod
from deepvol.models.autocall.greeks_engine import compute_batched_surrogate_greeks
import deepvol.models.autocall.pre_post_processor as _preproc_mod
from deepvol.models.autocall.pre_post_processor import (
    AutocallPreprocState,
    assemble_autocall_cashflows_and_pv,
    preprocess_eq_autocall,
)
from deepvol.surrogates.eq_autocall_egno import (
    MaskedEQAutocallSurrogate,
    assemble_surrogate_product_value,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_REQUEST_PATH = (
    REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt"
    if (REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt").exists()
    else REPO_ROOT / "eq_autocallable_request.txt"
)

GATE_G2_1_PROXY_M1_US = 50.0
GATE_G2_1_PROXY_M12_US = 150.0
GATE_G2_2_FWD_P50_MS = 2.5
GATE_G2_3A_BATCH_GREEKS_P50_MS = 8.0
GATE_G2_3B_COLD_PVCF_M1_MS = 5.0
GATE_G2_3B_COLD_PVCF_M12_MS = 8.0
GATE_G2_3B_WARM_SHOCK_PVCF_M12_MS = 3.5
GATE_G2_3B_COLD_GREEKS_M1_MS = 12.0
GATE_G2_3B_COLD_GREEKS_M12_MS = 20.0
GATE_G2_3B_WARM_SHOCK_GREEKS_M1_MS = 3.5
GATE_G2_3B_WARM_SHOCK_GREEKS_M12_MS = 8.0
GATE_G2_4_ARROW_M1_MS_PER_TRADE = 0.025
GATE_G2_4_ARROW_M12_MS_PER_TRADE = 0.045
GATE_GR5_ARROW_D8M48_MS_PER_TRADE = 0.085
GATE_G2_4_JSON_M12_MS_PER_TRADE = 1.5
GATE_GR5_PEAK_VRAM_MB = 1500.0
GATE_QMC_GPU_M1_MS = 10.0
GATE_QMC_GPU_M12_MS = 30.0
GATE_QMC_GPU_PEAK_VRAM_MB = 250.0


@dataclass(frozen=True)
class BenchmarkMetric:
    """Structured SLA latency, throughput, and VRAM metric emitted by `eq_autocall_bench`."""

    gate_id: str
    suite_name: str
    description: str
    unit: str
    p50: float
    p95: float
    p99: float
    threshold: float
    mc_baseline_same_unit: float
    speedup_vs_mc: float
    passed: bool
    peak_vram_mb: float = 0.0


def _clear_runtime_caches(states: Sequence[AutocallPreprocState] = ()) -> None:
    """Clears all preprocessing, market, surface, and Greek caches for honest Cold timings."""
    for st in states:
        sc = getattr(st, "_greeks_result_cache", None)
        if sc is not None and hasattr(sc, "clear"):
            sc.clear()
    for attr in (
        "_PREPROC_STATE_CACHE",
        "_MARKET_OBJECT_CACHE",
        "_CURVE_CACHE",
        "_SURFACE_CACHE",
    ):
        cache_obj = getattr(_preproc_mod, attr, None)
        if cache_obj is not None and hasattr(cache_obj, "clear"):
            cache_obj.clear()
    for attr in (
        "_SURFACE_GEOM_CACHE",
        "_SURFACE_QUOTE_CACHE",
        "_CURVE_DELTA_CACHE",
        "_GLOBAL_SURFACE_TORCH_CACHE",
        "_JACOBIAN_CACHE",
        "_GREEKS_CACHE",
    ):
        cache_obj = getattr(_greeks_mod, attr, None)
        if cache_obj is not None and hasattr(cache_obj, "clear"):
            cache_obj.clear()


def _load_surrogate(
    device: torch.device | str = "cpu",
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    strict_ckpt: bool = False,
) -> Tuple[MaskedEQAutocallSurrogate, bool]:
    """Loads the trained surrogate via `weights_only=True, strict=True`, failing if `--strict-ckpt` is missing."""
    dev = torch.device(device)
    if checkpoint_path.is_file():
        model, _metrics = _load_surrogate_from_ckpt(
            checkpoint_path,
            device=dev,
            dtype=torch.float32,
        )
        return model.eval(), True
    if strict_ckpt:
        raise FileNotFoundError(
            f"--strict-ckpt enabled but checkpoint not found: {checkpoint_path}"
        )
    fallback = MaskedEQAutocallSurrogate(
        d_max=8,
        m_max=48,
        d_model=128,
        n_egno_layers=3,
        n_sched_layers=2,
        n_heads=4,
        d_node=16,
        d_global=16,
    ).to(device=dev, dtype=torch.float32)
    fallback.zero_init_residual_heads(std=0.0)
    return fallback.eval(), False


def _load_reference_request() -> EQAutocallableTradeRequest:
    raw_text = REF_REQUEST_PATH.read_text(encoding="utf-8")
    req, _ = parse_murex_autocall_dag(raw_text)
    return req


def _build_synthetic_schedule_request(
    base_req: EQAutocallableTradeRequest,
    *,
    target_mrem: int,
    target_d: int,
    in_domain_guardian: bool = False,
) -> EQAutocallableTradeRequest:
    """Builds a valid cloned `EQAutocallableTradeRequest` with `(D=target_d, M_rem=target_mrem)`."""
    data = base_req.model_dump(mode="python", by_alias=True)
    as_of: datetime = base_req.Model.MarketDataSet.AsOfDate
    prod = data["Product"]
    obs_dates: List[datetime] = []
    pay_dates: List[datetime] = []
    for k in range(1, target_mrem + 1):
        obs_d = as_of + timedelta(days=30 * k)
        obs_dates.append(obs_d)
        pay_dates.append(obs_d + timedelta(days=3))
    prod["StartDate"] = as_of
    prod["ObservationDates"] = obs_dates
    prod["PaymentDates"] = pay_dates
    prod["AutoCallTriggerLevels"] = [1.00] * target_mrem
    if prod.get("CouponBarrierLevels") is not None:
        prod["CouponBarrierLevels"] = [0.75] * target_mrem
    if isinstance(prod.get("FixedRate"), list):
        prod["FixedRate"] = [0.04] * target_mrem
    elif isinstance(prod.get("FixedCouponAmount"), list):
        prod["FixedCouponAmount"] = [0.04] * target_mrem

    mds = data["Model"]["MarketDataSet"]
    mds["Fixings"] = {}
    existing_ids = list(prod["UnderlyingIdentifiers"])
    existing_refs = [float(r) for r in prod["UnderlyingReferenceSpots"]]
    while len(existing_ids) < target_d:
        idx_src = len(existing_ids) % len(prod["UnderlyingIdentifiers"])
        src_id = existing_ids[idx_src]
        new_id = f"SYNTH_{len(existing_ids)}_RUB"
        mds["Spots"][new_id] = float(mds["Spots"][src_id])
        if src_id in mds["VolatilitySurfaces"]:
            mds["VolatilitySurfaces"][new_id] = dict(mds["VolatilitySurfaces"][src_id])
        if src_id in mds.get("Dividends", {}):
            mds["Dividends"][new_id] = dict(mds["Dividends"][src_id])
        if src_id in mds.get("RepoSpreadCurves", {}):
            mds["RepoSpreadCurves"][new_id] = dict(mds["RepoSpreadCurves"][src_id])
        existing_ids.append(new_id)
        existing_refs.append(existing_refs[idx_src])

    chosen_ids = existing_ids[:target_d]
    prod["UnderlyingIdentifiers"] = chosen_ids
    prod["UnderlyingReferenceSpots"] = existing_refs[:target_d]

    if in_domain_guardian:
        # Calibrate moderate ATM in-domain basket so conformal_half_width_bps <= 60.0 and D_M^2 < q_ood
        prod["CallabilityMonitoringIndex"] = "Average"
        prod["CouponBarrierMonitoringIndex"] = "Average"
        prod["OptionMonitoringIndex"] = "Average"
        prod["OptionDownAndInBarrierLevel"] = None
        for aid, ref_s in zip(chosen_ids, prod["UnderlyingReferenceSpots"]):
            mds["Spots"][aid] = float(ref_s)
            if aid in mds["VolatilitySurfaces"]:
                vs = dict(mds["VolatilitySurfaces"][aid])
                vs["VolatilityQuotes"] = [[0.22 for _ in row] for row in vs["VolatilityQuotes"]]
                mds["VolatilitySurfaces"][aid] = vs

    corr_map: Dict[str, float] = {}
    for i, a1 in enumerate(chosen_ids):
        for j in range(i + 1, len(chosen_ids)):
            a2 = chosen_ids[j]
            pair_key = f"{a1}&{a2}"
            rev_key = f"{a2}&{a1}"
            existing_val = mds.get("Correlations", {}).get(
                pair_key, mds.get("Correlations", {}).get(rev_key, 0.45)
            )
            corr_map[pair_key] = float(existing_val)
    mds["Correlations"] = corr_map
    return EQAutocallableTradeRequest.model_validate(data)


def _run_crn_qmc_fd_greeks(
    state: AutocallPreprocState,
    *,
    n_paths: int,
    gpu_solver: Any | None = None,
) -> None:
    """Honest CRN QMC spot-delta+gamma baseline (base + 2*D CRN evaluations) when M5 CRN is absent."""
    d_act = max(int(state.d_assets), 1)
    if gpu_solver is not None:
        from deepvol.models.autocall.dupire_qmc_gpu import DupireQMCConfig

        cfg = DupireQMCConfig(n_paths=n_paths, seed=42)
        gpu_solver.simulate_and_price(state, config=cfg)
        base_spots = state.spots_t0.clone()
        try:
            for i in range(d_act):
                h = float(base_spots[i].item()) * 1e-3
                state.spots_t0 = base_spots.clone()
                state.spots_t0[i] = base_spots[i] + h
                gpu_solver.simulate_and_price(state, config=cfg)
                state.spots_t0 = base_spots.clone()
                state.spots_t0[i] = base_spots[i] - h
                gpu_solver.simulate_and_price(state, config=cfg)
        finally:
            state.spots_t0 = base_spots
    else:
        price_eq_autocall_dupire_qmc(state, n_paths=n_paths, seed=42)
        base_spots = state.spots_t0.clone()
        try:
            for i in range(d_act):
                h = float(base_spots[i].item()) * 1e-3
                state.spots_t0 = base_spots.clone()
                state.spots_t0[i] = base_spots[i] + h
                price_eq_autocall_dupire_qmc(state, n_paths=n_paths, seed=42)
                state.spots_t0 = base_spots.clone()
                state.spots_t0[i] = base_spots[i] - h
                price_eq_autocall_dupire_qmc(state, n_paths=n_paths, seed=42)
        finally:
            state.spots_t0 = base_spots


def _measure_real_qmc_suites(
    state_m1: AutocallPreprocState,
    state_m12: AutocallPreprocState,
    *,
    n_paths: int = 32768,
    n_iter: int = 5,
) -> Tuple[float, float, List[BenchmarkMetric]]:
    """Measures genuine n_iter>=5 CPU QMC forward, GPU RQMC M1/M12 forward, and CRN QMC Greeks."""
    metrics: List[BenchmarkMetric] = []

    price_eq_autocall_dupire_qmc(state_m12, n_paths=4096)
    cpu_samples_ms: List[float] = []
    for _ in range(n_iter):
        _clear_runtime_caches([state_m12])
        t0 = time.perf_counter_ns()
        price_eq_autocall_dupire_qmc(state_m12, n_paths=n_paths)
        cpu_samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    p50_cpu, p95_cpu, p99_cpu = (float(x) for x in np.percentile(cpu_samples_ms, [50, 95, 99]))
    metrics.append(
        BenchmarkMetric(
            gate_id="QMC-CPU",
            suite_name="qmc",
            description=f"Dupire QMC CPU Forward (M_rem=12, N={n_paths}, n_iter={n_iter})",
            unit="ms",
            p50=p50_cpu,
            p95=p95_cpu,
            p99=p99_cpu,
            threshold=10000.0,
            mc_baseline_same_unit=p50_cpu,
            speedup_vs_mc=1.0,
            passed=True,
        )
    )

    gpu_solver: Any = None
    if torch.cuda.is_available():
        try:
            from deepvol.models.autocall.dupire_qmc_gpu import (
                DupireQMCConfig,
                DupireQMCGpuSolver,
            )

            gpu_solver = DupireQMCGpuSolver(device="cuda")
            for gid, st_q, thresh_ms, label in (
                ("QMC-GPU-M1", state_m1, GATE_QMC_GPU_M1_MS, "M_rem=1"),
                ("QMC-GPU", state_m12, GATE_QMC_GPU_M12_MS, "M_rem=12"),
            ):
                torch.cuda.reset_peak_memory_stats()
                gpu_solver.simulate_and_price(st_q, config=DupireQMCConfig(n_paths=4096))
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                gpu_samples_ms: List[float] = []
                cfg_gpu = DupireQMCConfig(n_paths=n_paths)
                for _ in range(n_iter):
                    _clear_runtime_caches([st_q])
                    t0 = time.perf_counter_ns()
                    gpu_solver.simulate_and_price(st_q, config=cfg_gpu)
                    torch.cuda.synchronize()
                    gpu_samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
                peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                p50_g, p95_g, p99_g = (float(x) for x in np.percentile(gpu_samples_ms, [50, 95, 99]))
                metrics.append(
                    BenchmarkMetric(
                        gate_id=gid,
                        suite_name="qmc",
                        description=f"Dupire RQMC GPU Forward ({label}, N={n_paths}, VRAM={peak_gpu_mb:.1f} MB)",
                        unit="ms",
                        p50=p50_g,
                        p95=p95_g,
                        p99=p99_g,
                        threshold=thresh_ms,
                        mc_baseline_same_unit=p50_cpu,
                        speedup_vs_mc=p50_cpu / max(p50_g, 1e-9),
                        passed=(p50_g <= thresh_ms and peak_gpu_mb <= GATE_QMC_GPU_PEAK_VRAM_MB),
                        peak_vram_mb=peak_gpu_mb,
                    )
                )
        except Exception:
            gpu_solver = None

    # Real CRN QMC Greeks baseline (n_iter >= 5, cache cleared per iter, never cached proxy)
    crn_fn = getattr(_greeks_mod, "compute_qmc_crn_greeks", None)
    queries_all = ["Delta", "Gamma", "MultivariateAggregatedCashGamma", "Vega", "Rho", "Theta", "Cega"]
    crn_samples_ms: List[float] = []
    if callable(crn_fn):
        crn_fn(state_m12, queries=queries_all)
        for _ in range(n_iter):
            _clear_runtime_caches([state_m12])
            t0 = time.perf_counter_ns()
            crn_fn(state_m12, queries=queries_all)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            crn_samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    else:
        _run_crn_qmc_fd_greeks(state_m12, n_paths=n_paths, gpu_solver=gpu_solver)
        for _ in range(n_iter):
            _clear_runtime_caches([state_m12])
            t0 = time.perf_counter_ns()
            _run_crn_qmc_fd_greeks(state_m12, n_paths=n_paths, gpu_solver=gpu_solver)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            crn_samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    p50_crn, p95_crn, p99_crn = (float(x) for x in np.percentile(crn_samples_ms, [50, 95, 99]))
    metrics.append(
        BenchmarkMetric(
            gate_id="QMC-CRN",
            suite_name="qmc",
            description=f"Measured Dupire CRN QMC Greeks (M_rem=12, N={n_paths}, n_iter={n_iter})",
            unit="ms",
            p50=p50_crn,
            p95=p95_crn,
            p99=p99_crn,
            threshold=30000.0,
            mc_baseline_same_unit=p50_crn,
            speedup_vs_mc=1.0,
            passed=True,
        )
    )

    return p50_cpu, p50_crn, metrics


def run_proxy_suite(
    state_m1: AutocallPreprocState,
    state_m12: AutocallPreprocState,
    mc_fwd_ms: float,
    n_warmup: int = 400,
    n_iter: int = 1000,
) -> List[BenchmarkMetric]:
    """Benchmarks Tier-1 Analytical Proxy on `M_rem=1` (`45476647`) and `M_rem=12`."""
    results: List[BenchmarkMetric] = []
    mc_us = mc_fwd_ms * 1e3
    for label, st, thresh in (
        ("G2.1", state_m1, GATE_G2_1_PROXY_M1_US),
        ("G2.1-M12", state_m12, GATE_G2_1_PROXY_M12_US),
    ):
        for _ in range(n_warmup):
            compute_analytical_autocall_baseline(st, use_fast_cpu=True)
        samples_us: List[float] = []
        for _ in range(n_iter):
            t0 = time.perf_counter_ns()
            compute_analytical_autocall_baseline(st, use_fast_cpu=True)
            samples_us.append((time.perf_counter_ns() - t0) / 1e3)
        p50, p95, p99 = (float(x) for x in np.percentile(samples_us, [50, 95, 99]))
        results.append(
            BenchmarkMetric(
                gate_id=label,
                suite_name="proxy",
                description=f"Tier-1 Analytical Proxy (M_rem={st.m_rem}, D={st.d_assets})",
                unit="us",
                p50=p50,
                p95=p95,
                p99=p99,
                threshold=thresh,
                mc_baseline_same_unit=mc_us,
                speedup_vs_mc=mc_us / max(p50, 1e-9),
                passed=(p50 <= thresh),
            )
        )
    return results


def run_fwd_suite(
    state_m1: AutocallPreprocState,
    state_m12: AutocallPreprocState,
    state_d8_m48: AutocallPreprocState,
    mc_fwd_ms: float,
    *,
    checkpoint_path: Path,
    strict_ckpt: bool,
    n_warmup: int = 40,
    n_iter: int = 200,
) -> List[BenchmarkMetric]:
    """Benchmarks Single-Trade Surrogate Forward across `M_rem=1`, `M_rem=12`, and `D=8,M=48`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate, _ = _load_surrogate(device, checkpoint_path=checkpoint_path, strict_ckpt=strict_ckpt)

    compiled_runner: Any = None
    if device.type == "cuda":
        try:
            compiled_mod = importlib.import_module("deepvol.surrogates.compiled_runner")
            runner_cls = getattr(compiled_mod, "CompiledSurrogateRunner", None)
            if runner_cls is not None:
                compiled_runner = runner_cls(
                    surrogate,
                    use_tf32=True,
                    enable_cudagraphs=True,
                    enable_inductor=True,
                )
        except Exception:
            compiled_runner = None

    results: List[BenchmarkMetric] = []
    configs = (
        ("G2.2", state_m1, GATE_G2_2_FWD_P50_MS),
        ("G2.2-M12", state_m12, GATE_G2_2_FWD_P50_MS),
        ("G2.2-D8M48", state_d8_m48, 5.0),
    )
    for gate_id, st, thresh in configs:
        d_act = max(int(st.d_assets), 1)
        m_act = max(int(st.m_rem), 1)
        t2 = st.to_tier2_float32_tensors()
        nf = t2["node_feats"][:, :d_act, :].to(device).contiguous()
        ef = t2["edge_feats"][:, :d_act, :d_act, :].to(device).contiguous()
        am = t2["asset_mask"][:, :d_act].to(device).contiguous()
        sf = t2["sched_feats"][:, :m_act, :].to(device).contiguous()
        sm = t2["sched_mask"][:, :m_act].to(device).contiguous()
        gf = t2["global_feats"].to(device).contiguous()

        def _fwd_once() -> None:
            if compiled_runner is not None and device.type == "cuda":
                dh, dc, dp = compiled_runner.predict_residuals(nf, ef, am, sf, sm, gf)
                _ = (dh, dc, dp)
            else:
                out = surrogate(nf, ef, am, sf, sm, gf, all_active=True)
                assemble_surrogate_product_value(out, sf, sm, gf)

        with torch.inference_mode():
            for _ in range(n_warmup):
                _fwd_once()
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples_ms: List[float] = []
            for _ in range(n_iter):
                t0 = time.perf_counter_ns()
                _fwd_once()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                samples_ms.append((time.perf_counter_ns() - t0) / 1e6)

        p50, p95, p99 = (float(x) for x in np.percentile(samples_ms, [50, 95, 99]))
        results.append(
            BenchmarkMetric(
                gate_id=gate_id,
                suite_name="fwd",
                description=f"Single-Trade Surrogate Forward (B=1, D={d_act}, M={m_act}, {device.type.upper()})",
                unit="ms",
                p50=p50,
                p95=p95,
                p99=p99,
                threshold=thresh,
                mc_baseline_same_unit=mc_fwd_ms,
                speedup_vs_mc=mc_fwd_ms / max(p50, 1e-9),
                passed=(p50 <= thresh),
            )
        )
    return results


def run_full_greeks_suite(
    req_m1: EQAutocallableTradeRequest,
    req_m12: EQAutocallableTradeRequest,
    state_m1: AutocallPreprocState,
    state_m12: AutocallPreprocState,
    mc_aad_ms: float,
    *,
    checkpoint_path: Path,
    strict_ckpt: bool,
    n_warmup: int = 15,
    n_iter: int = 60,
) -> List[BenchmarkMetric]:
    """Benchmarks Spec AG3b/AG3c/GR9 E2E Cold/Warm-Shock PV+CF, All-8 Greeks, and G2.3a."""
    rt = EQAutocallRuntime(
        checkpoint_path=checkpoint_path,
        strict_checkpoint=strict_ckpt,
        d_max=8,
        m_max=48,
    )
    surrogate = rt.surrogate
    dev = rt.device
    results: List[BenchmarkMetric] = []

    # G2.3a: Single-trade batched 1st-order Greeks + CashGamma + CashFlows on M_rem=12
    d_act = max(int(state_m12.d_assets), 1)
    m_act = max(int(state_m12.m_rem), 1)
    nf64 = state_m12.node_feats[:, :d_act, :].to(dev).contiguous()
    ef64 = state_m12.edge_feats[:, :d_act, :d_act, :].to(dev).contiguous()
    am64 = state_m12.asset_mask[:, :d_act].to(dev).contiguous()
    sf64 = state_m12.sched_feats[:, :m_act, :].to(dev).contiguous()
    sm64 = state_m12.sched_mask[:, :m_act].to(dev).contiguous()
    gf64 = state_m12.global_feats.to(dev).contiguous()
    sp_b = state_m12.spots_t0.unsqueeze(0).to(dev)
    ref_b = state_m12.ref_spots.unsqueeze(0).to(dev)

    def _step_g23a() -> None:
        bg = compute_batched_surrogate_greeks(
            surrogate,
            nf64,
            ef64,
            am64,
            sf64,
            sm64,
            gf64,
            spots_t0=sp_b,
            ref_spots=ref_b,
            compute_cash_gamma=True,
        )
        assemble_autocall_cashflows_and_pv(
            state_m12,
            p_stop=bg.stopping_probs[:, :m_act].squeeze(0).cpu(),
            q_cpn=bg.coupon_probs[:, :m_act].squeeze(0).cpu(),
            l_put=bg.put_loss.squeeze(0).cpu(),
        )

    for _ in range(n_warmup):
        _step_g23a()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    samples_g23a: List[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        _step_g23a()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        samples_g23a.append((time.perf_counter_ns() - t0) / 1e6)
    p50_a, p95_a, p99_a = (float(x) for x in np.percentile(samples_g23a, [50, 95, 99]))
    results.append(
        BenchmarkMetric(
            gate_id="G2.3a",
            suite_name="full_greeks",
            description=f"Single-Trade 1st-Order Greeks + CashGamma + CashFlows (M_rem=12, {dev.type.upper()})",
            unit="ms",
            p50=p50_a,
            p95=p95_a,
            p99=p99_a,
            threshold=GATE_G2_3A_BATCH_GREEKS_P50_MS,
            mc_baseline_same_unit=mc_aad_ms,
            speedup_vs_mc=mc_aad_ms / max(p50_a, 1e-9),
            passed=(p50_a <= GATE_G2_3A_BATCH_GREEKS_P50_MS),
        )
    )

    # G2.3b Cold & Warm Spot-Shock (+1bp/iter) on PV+CF (AG3b / GR9) and All 8 Greeks (AG3c)
    cold_iters = min(20, n_iter)
    pvcf_queries = ["ProductValue", "CashFlows"]
    for label_m, req_base, cold_pvcf_thr, cold_grk_thr, warm_grk_thr in (
        ("M1", req_m1, GATE_G2_3B_COLD_PVCF_M1_MS, GATE_G2_3B_COLD_GREEKS_M1_MS, GATE_G2_3B_WARM_SHOCK_GREEKS_M1_MS),
        ("M12", req_m12, GATE_G2_3B_COLD_PVCF_M12_MS, GATE_G2_3B_COLD_GREEKS_M12_MS, GATE_G2_3B_WARM_SHOCK_GREEKS_M12_MS),
    ):
        req_pvcf = req_base.model_copy(update={"Queries": pvcf_queries})
        base_spots = dict(req_base.Model.MarketDataSet.Spots)

        rt.price_trade(req_pvcf, engine="auto", include_all_greeks=False)
        cold_pvcf_ms: List[float] = []
        warm_pvcf_ms: List[float] = []
        for idx in range(cold_iters):
            _clear_runtime_caches()
            t0 = time.perf_counter_ns()
            rt.price_trade(req_pvcf, engine="auto", include_all_greeks=False)
            cold_pvcf_ms.append((time.perf_counter_ns() - t0) / 1e6)

            shocked_pvcf = req_pvcf.model_copy(deep=True)
            for k, val in base_spots.items():
                shocked_pvcf.Model.MarketDataSet.Spots[k] = float(val) * (1.0 + 1e-4 * (idx + 1))
            t1 = time.perf_counter_ns()
            rt.price_trade(shocked_pvcf, engine="auto", include_all_greeks=False)
            warm_pvcf_ms.append((time.perf_counter_ns() - t1) / 1e6)

        p50_cp, p95_cp, p99_cp = (float(x) for x in np.percentile(cold_pvcf_ms, [50, 95, 99]))
        results.append(
            BenchmarkMetric(
                gate_id=f"G2.3b-Cold-PVCF-{label_m}",
                suite_name="full_greeks",
                description=f"Cold E2E price_trade PV+CashFlows ({label_m})",
                unit="ms",
                p50=p50_cp,
                p95=p95_cp,
                p99=p99_cp,
                threshold=cold_pvcf_thr,
                mc_baseline_same_unit=mc_aad_ms,
                speedup_vs_mc=mc_aad_ms / max(p50_cp, 1e-9),
                passed=(p50_cp <= cold_pvcf_thr),
            )
        )
        if label_m == "M12":
            p50_wp, p95_wp, p99_wp = (float(x) for x in np.percentile(warm_pvcf_ms, [50, 95, 99]))
            results.append(
                BenchmarkMetric(
                    gate_id="G2.3b-WarmShock-PVCF-M12",
                    suite_name="full_greeks",
                    description="Warm Spot-Shock (+1bp/iter) price_trade PV+CashFlows (M12)",
                    unit="ms",
                    p50=p50_wp,
                    p95=p95_wp,
                    p99=p99_wp,
                    threshold=GATE_G2_3B_WARM_SHOCK_PVCF_M12_MS,
                    mc_baseline_same_unit=mc_aad_ms,
                    speedup_vs_mc=mc_aad_ms / max(p50_wp, 1e-9),
                    passed=(p50_wp <= GATE_G2_3B_WARM_SHOCK_PVCF_M12_MS),
                )
            )

        rt.price_trade(req_base, engine="auto", include_all_greeks=True)
        cold_ms: List[float] = []
        warm_shock_ms: List[float] = []
        for idx in range(cold_iters):
            _clear_runtime_caches()
            t0 = time.perf_counter_ns()
            rt.price_trade(req_base, engine="auto", include_all_greeks=True)
            cold_ms.append((time.perf_counter_ns() - t0) / 1e6)

            shocked_req = req_base.model_copy(deep=True)
            for k, val in base_spots.items():
                shocked_req.Model.MarketDataSet.Spots[k] = float(val) * (1.0 + 1e-4 * (idx + 1))
            t1 = time.perf_counter_ns()
            rt.price_trade(shocked_req, engine="auto", include_all_greeks=True)
            warm_shock_ms.append((time.perf_counter_ns() - t1) / 1e6)

        p50_c, p95_c, p99_c = (float(x) for x in np.percentile(cold_ms, [50, 95, 99]))
        p50_w, p95_w, p99_w = (float(x) for x in np.percentile(warm_shock_ms, [50, 95, 99]))
        results.append(
            BenchmarkMetric(
                gate_id=f"G2.3b-Cold-{label_m}",
                suite_name="full_greeks",
                description=f"Cold E2E price_trade + All 8 Greeks (cache cleared, {label_m})",
                unit="ms",
                p50=p50_c,
                p95=p95_c,
                p99=p99_c,
                threshold=cold_grk_thr,
                mc_baseline_same_unit=mc_aad_ms,
                speedup_vs_mc=mc_aad_ms / max(p50_c, 1e-9),
                passed=(p50_c <= cold_grk_thr),
            )
        )
        results.append(
            BenchmarkMetric(
                gate_id=f"G2.3b-WarmShock-{label_m}",
                suite_name="full_greeks",
                description=f"Warm Spot-Shock (+1bp/iter) price_trade + All 8 Greeks ({label_m})",
                unit="ms",
                p50=p50_w,
                p95=p95_w,
                p99=p99_w,
                threshold=warm_grk_thr,
                mc_baseline_same_unit=mc_aad_ms,
                speedup_vs_mc=mc_aad_ms / max(p50_w, 1e-9),
                passed=(p50_w <= warm_grk_thr),
            )
        )
    return results


def run_batch_suite(
    req_m12_in_domain: EQAutocallableTradeRequest,
    state_m1: AutocallPreprocState,
    state_m12: AutocallPreprocState,
    state_d8_m48: AutocallPreprocState,
    mc_aad_ms: float,
    *,
    checkpoint_path: Path,
    strict_ckpt: bool,
    batch_size: int = 1024,
    n_warmup: int = 8,
    n_iter: int = 30,
) -> List[BenchmarkMetric]:
    """Benchmarks Arrow E2E (Spec AG3d), `D=8,M=48` chunked peak VRAM (GR5), and M=12 JSON batching (AG3e)."""
    rt = EQAutocallRuntime(
        checkpoint_path=checkpoint_path,
        strict_checkpoint=strict_ckpt,
        d_max=8,
        m_max=48,
    )
    dev = rt.device
    results: List[BenchmarkMetric] = []

    for label, st, thresh in (
        ("G2.4-M1", state_m1, GATE_G2_4_ARROW_M1_MS_PER_TRADE),
        ("G2.4-M12", state_m12, GATE_G2_4_ARROW_M12_MS_PER_TRADE),
        ("GR5-D8M48", state_d8_m48, GATE_GR5_ARROW_D8M48_MS_PER_TRADE),
    ):
        nf_np = st.node_feats.repeat(batch_size, 1, 1).cpu().numpy()
        ef_np = st.edge_feats.repeat(batch_size, 1, 1, 1).cpu().numpy()
        am_np = st.asset_mask.repeat(batch_size, 1).cpu().numpy()
        sf_np = st.sched_feats.repeat(batch_size, 1, 1).cpu().numpy()
        sm_np = st.sched_mask.repeat(batch_size, 1).cpu().numpy()
        gf_np = st.global_feats.repeat(batch_size, 1).cpu().numpy()
        d_max = int(st.node_feats.shape[1])
        sp_np = np.ones((batch_size, d_max), dtype=np.float64)
        sp_np[:, : st.d_assets] = st.spots_t0.cpu().numpy()[None, :]
        ref_np = np.ones((batch_size, d_max), dtype=np.float64)
        ref_np[:, : st.d_assets] = st.ref_spots.cpu().numpy()[None, :]

        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for _ in range(n_warmup):
            ipc_in = encode_eq_autocall_arrow_request(
                nf_np, ef_np, am_np, sf_np, sm_np, gf_np,
                {"schema_version": FEATURE_SCHEMA_VERSION},
                spots_t0=sp_np, ref_spots=ref_np,
            )
            ipc_out = rt.price_arrow_ipc(ipc_in, compute_greeks=True)
            _ = decode_eq_autocall_arrow_response(ipc_out)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        samples_ms_per_trade: List[float] = []
        for _ in range(n_iter):
            t0 = time.perf_counter_ns()
            ipc_in = encode_eq_autocall_arrow_request(
                nf_np, ef_np, am_np, sf_np, sm_np, gf_np,
                {"schema_version": FEATURE_SCHEMA_VERSION},
                spots_t0=sp_np, ref_spots=ref_np,
            )
            ipc_out = rt.price_arrow_ipc(ipc_in, compute_greeks=True)
            _ = decode_eq_autocall_arrow_response(ipc_out)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            samples_ms_per_trade.append((time.perf_counter_ns() - t0) / 1e6 / batch_size)

        peak_mb = (
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if dev.type == "cuda"
            else 0.0
        )
        p50, p95, p99 = (float(x) for x in np.percentile(samples_ms_per_trade, [50, 95, 99]))
        vram_ok = (peak_mb <= GATE_GR5_PEAK_VRAM_MB) if dev.type == "cuda" else True
        results.append(
            BenchmarkMetric(
                gate_id=label,
                suite_name="batch",
                description=(
                    f"Arrow E2E Round-Trip + Greeks (B={batch_size}, D={st.d_assets}, "
                    f"M={st.m_rem}, VRAM={peak_mb:.1f} MB)"
                ),
                unit="ms/trade",
                p50=p50,
                p95=p95,
                p99=p99,
                threshold=thresh,
                mc_baseline_same_unit=mc_aad_ms,
                speedup_vs_mc=mc_aad_ms / max(p50, 1e-9),
                passed=(p50 <= thresh and vram_ok),
                peak_vram_mb=peak_mb,
            )
        )

    # JSON Batch throughput on in-domain M_rem=12 (Spec AG3e <= 1.5 ms/trade)
    json_list = [req_m12_in_domain.model_dump(mode="python", by_alias=True) for _ in range(batch_size)]
    warm_res = rt.price_batch_json(json_list, compute_greeks=False, engine="auto")
    if warm_res and warm_res[0].fallback_triggered:
        raise RuntimeError(
            f"G2.4-JSON benchmark fixture triggered unexpected Guardian fallback: {warm_res[0].fallback_reasons}"
        )
    json_samples: List[float] = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        rt.price_batch_json(json_list, compute_greeks=False, engine="auto")
        json_samples.append((time.perf_counter_ns() - t0) / 1e6 / batch_size)
    p50_j, p95_j, p99_j = (float(x) for x in np.percentile(json_samples, [50, 95, 99]))
    results.append(
        BenchmarkMetric(
            gate_id="G2.4-JSON",
            suite_name="batch",
            description=f"JSON Batch Valuation (B={batch_size}, M_rem=12 in-domain, market-dedup, SR 26-2 audit)",
            unit="ms/trade",
            p50=p50_j,
            p95=p95_j,
            p99=p99_j,
            threshold=GATE_G2_4_JSON_M12_MS_PER_TRADE,
            mc_baseline_same_unit=mc_aad_ms,
            speedup_vs_mc=mc_aad_ms / max(p50_j, 1e-9),
            passed=(p50_j <= GATE_G2_4_JSON_M12_MS_PER_TRADE),
        )
    )
    return results


def _format_markdown_report(
    metrics: List[BenchmarkMetric],
    mc_fwd_ms: float,
    mc_aad_ms: float,
    strict_ckpt: bool,
    ckpt_loaded: bool,
) -> str:
    dev_str = f"CUDA ({torch.cuda.get_device_name(0)}) / CPU" if torch.cuda.is_available() else "CPU"
    lines = [
        "# SPEC-DEEPVOL-EQ-AUTOCALL-002 — End-to-End Latency, Throughput & VRAM Benchmark Report",
        "",
        f"- **Device**: `{dev_str}`",
        f"- **Strict Checkpoint**: `strict_ckpt={strict_ckpt}`, `loaded={ckpt_loaded}`",
        f"- **32,768-Path Dupire QMC Baseline (Measured, n_iter=5)**: `{mc_fwd_ms:.2f} ms` (forward) / `{mc_aad_ms:.2f} ms` (CRN QMC Greeks)",
        "",
        "| Gate ID | Suite | Operation | Unit | p50 | p95 | p99 | Gate Threshold | Speedup vs MC | Status |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for m in metrics:
        status_str = "PASS" if m.passed else "FAIL"
        lines.append(
            f"| `{m.gate_id}` | `{m.suite_name}` | {m.description} | `{m.unit}` | "
            f"`{m.p50:.4f}` | `{m.p95:.4f}` | `{m.p99:.4f}` | "
            f"`<= {m.threshold:.4f}` | `{m.speedup_vs_mc:,.1f}x` | **{status_str}** |"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-End Latency, Throughput & VRAM Benchmark Suite for SPEC-DEEPVOL-EQ-AUTOCALL-002."
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=(
            "all",
            "proxy",
            "pricing",
            "fwd",
            "surrogate",
            "full_greeks",
            "greeks",
            "batch",
            "api",
            "qmc",
        ),
        default=["all"],
        help="Benchmark suite(s) to execute (default: all).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to EGNO surrogate checkpoint.",
    )
    parser.add_argument(
        "--strict-ckpt",
        action="store_true",
        help="Fail immediately if checkpoint cannot be loaded strictly with weights_only=True.",
    )
    args = parser.parse_args(argv)

    _surr_check, ckpt_loaded = _load_surrogate(
        "cpu",
        checkpoint_path=args.checkpoint,
        strict_ckpt=bool(args.strict_ckpt),
    )
    del _surr_check

    req_m1 = _load_reference_request()
    req_m12 = _build_synthetic_schedule_request(
        req_m1, target_mrem=12, target_d=5, in_domain_guardian=False
    )
    req_m12_in_domain = _build_synthetic_schedule_request(
        req_m1, target_mrem=12, target_d=5, in_domain_guardian=True
    )
    req_d8_m48 = _build_synthetic_schedule_request(req_m1, target_mrem=48, target_d=8)

    state_m1 = preprocess_eq_autocall(req_m1, d_max=8, m_max=48)
    state_m12 = preprocess_eq_autocall(req_m12, d_max=8, m_max=48)
    state_d8_m48 = preprocess_eq_autocall(req_d8_m48, d_max=8, m_max=48)

    suite_aliases = {
        "pricing": "proxy",
        "surrogate": "fwd",
        "greeks": "full_greeks",
        "api": "batch",
    }
    suites = {suite_aliases.get(s, s) for s in args.suite}

    metrics: List[BenchmarkMetric] = []
    proxy_raw: List[BenchmarkMetric] = []
    if "all" in suites or "proxy" in suites:
        proxy_raw = run_proxy_suite(state_m1, state_m12, mc_fwd_ms=1000.0)

    mc_fwd_ms, mc_crn_ms, qmc_metrics = _measure_real_qmc_suites(state_m1, state_m12, n_paths=32768, n_iter=5)

    if proxy_raw:
        mc_us = mc_fwd_ms * 1e3
        for pm in proxy_raw:
            metrics.append(
                BenchmarkMetric(
                    gate_id=pm.gate_id,
                    suite_name=pm.suite_name,
                    description=pm.description,
                    unit=pm.unit,
                    p50=pm.p50,
                    p95=pm.p95,
                    p99=pm.p99,
                    threshold=pm.threshold,
                    mc_baseline_same_unit=mc_us,
                    speedup_vs_mc=mc_us / max(pm.p50, 1e-9),
                    passed=pm.passed,
                )
            )
    if "all" in suites or "qmc" in suites:
        metrics.extend(qmc_metrics)
    if "all" in suites or "fwd" in suites:
        metrics.extend(
            run_fwd_suite(
                state_m1,
                state_m12,
                state_d8_m48,
                mc_fwd_ms,
                checkpoint_path=args.checkpoint,
                strict_ckpt=bool(args.strict_ckpt),
            )
        )
    if "all" in suites or "full_greeks" in suites:
        metrics.extend(
            run_full_greeks_suite(
                req_m1,
                req_m12,
                state_m1,
                state_m12,
                mc_crn_ms,
                checkpoint_path=args.checkpoint,
                strict_ckpt=bool(args.strict_ckpt),
            )
        )
    if "all" in suites or "batch" in suites:
        metrics.extend(
            run_batch_suite(
                req_m12_in_domain,
                state_m1,
                state_m12,
                state_d8_m48,
                mc_crn_ms,
                checkpoint_path=args.checkpoint,
                strict_ckpt=bool(args.strict_ckpt),
            )
        )

    print(_format_markdown_report(metrics, mc_fwd_ms, mc_crn_ms, bool(args.strict_ckpt), ckpt_loaded))

    failed = [m for m in metrics if not m.passed]
    if failed:
        for m in failed:
            print(
                f"ERROR: Gate {m.gate_id} ({m.suite_name}) failed: "
                f"measured p50={m.p50:.4f} {m.unit} > threshold {m.threshold:.4f} {m.unit} "
                f"(VRAM={m.peak_vram_mb:.1f} MB)",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
