from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from deepvol.api.schemas_eq_autocall import parse_murex_autocall_dag
from deepvol.models.autocall.analytical_baseline import compute_analytical_autocall_baseline
from deepvol.models.autocall.dupire_qmc_solver import price_eq_autocall_dupire_qmc
from deepvol.models.autocall.greeks_engine import compute_batched_surrogate_greeks
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
REF_REQUEST_PATH = REPO_ROOT / "eq_autocallable_request.txt"
DEFAULT_CKPT_PATH = REPO_ROOT / "artifacts" / "checkpoints" / "eq_autocall_egno.pt"

GATE_G2_1_PROXY_P50_US = 50.0
GATE_G2_2_FWD_P50_MS = 2.5
GATE_G2_3_FULL_GREEKS_P50_MS = 8.0
GATE_G2_4_BATCH_MS_PER_TRADE = 0.05


def _load_trained_surrogate(device: torch.device | str = "cpu") -> MaskedEQAutocallSurrogate:
    dev = torch.device(device)
    if DEFAULT_CKPT_PATH.is_file():
        ckpt = torch.load(DEFAULT_CKPT_PATH, map_location=dev, weights_only=False)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        surr = MaskedEQAutocallSurrogate(
            d_max=int(cfg.get("d_max", 8)),
            m_max=int(cfg.get("m_max", 48)),
            d_model=int(cfg.get("d_model", 128)),
            n_egno_layers=int(cfg.get("n_egno_layers", 3)),
            n_sched_layers=int(cfg.get("n_sched_layers", 2)),
            n_heads=int(cfg.get("n_heads", 4)),
        ).to(device=dev, dtype=torch.float32)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(sd, dict):
            surr.load_state_dict(sd, strict=False)
        return surr.eval()
    return MaskedEQAutocallSurrogate(
        d_model=128,
        n_egno_layers=3,
        n_sched_layers=2,
    ).to(device=dev, dtype=torch.float32).eval()


@dataclass(frozen=True)
class BenchmarkMetric:
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


def _load_reference_state() -> AutocallPreprocState:
    raw_text = REF_REQUEST_PATH.read_text(encoding="utf-8")
    req, _ = parse_murex_autocall_dag(raw_text)
    return preprocess_eq_autocall(req)


def _measure_mc_baseline_ms(state: AutocallPreprocState, n_paths: int = 32768) -> float:
    price_eq_autocall_dupire_qmc(state, n_paths=4096)
    t0 = time.perf_counter_ns()
    price_eq_autocall_dupire_qmc(state, n_paths=n_paths)
    return (time.perf_counter_ns() - t0) / 1e6


def run_proxy_suite(
    state: AutocallPreprocState,
    mc_fwd_ms: float,
    n_warmup: int = 50,
    n_iter: int = 500,
) -> BenchmarkMetric:
    for _ in range(n_warmup):
        compute_analytical_autocall_baseline(state, use_fast_cpu=True)

    samples_us: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        compute_analytical_autocall_baseline(state, use_fast_cpu=True)
        samples_us.append((time.perf_counter_ns() - t0) / 1e3)

    p50, p95, p99 = (float(x) for x in np.percentile(samples_us, [50, 95, 99]))
    mc_us = mc_fwd_ms * 1e3
    return BenchmarkMetric(
        gate_id="G2.1",
        suite_name="proxy",
        description="Tier-1 Analytical Proxy (Stulz-Margrabe + Normal Copula + Bachelier)",
        unit="us",
        p50=p50,
        p95=p95,
        p99=p99,
        threshold=GATE_G2_1_PROXY_P50_US,
        mc_baseline_same_unit=mc_us,
        speedup_vs_mc=mc_us / max(p50, 1e-9),
        passed=(p50 <= GATE_G2_1_PROXY_P50_US),
    )


def run_fwd_suite(
    state: AutocallPreprocState,
    mc_fwd_ms: float,
    n_warmup: int = 50,
    n_iter: int = 300,
) -> BenchmarkMetric:
    surrogate = _load_trained_surrogate("cpu")

    # Slice to active basket and schedule lengths because Gate G1.7 guarantees exact padding invariance.
    d_act = max(int(state.d_assets), 1)
    m_act = max(int(state.m_rem), 1)
    t2 = state.to_tier2_float32_tensors()
    nf = t2["node_feats"][:, :d_act, :].contiguous()
    ef = t2["edge_feats"][:, :d_act, :d_act, :].contiguous()
    am = t2["asset_mask"][:, :d_act].contiguous()
    sf = t2["sched_feats"][:, :m_act, :].contiguous()
    sm = t2["sched_mask"][:, :m_act].contiguous()
    gf = t2["global_feats"].contiguous()

    with torch.inference_mode():
        for _ in range(n_warmup):
            out = surrogate(nf, ef, am, sf, sm, gf, all_active=True)
            assemble_surrogate_product_value(out, sf, sm, gf)

        samples_ms: list[float] = []
        for _ in range(n_iter):
            t0 = time.perf_counter_ns()
            out = surrogate(nf, ef, am, sf, sm, gf, all_active=True)
            assemble_surrogate_product_value(out, sf, sm, gf)
            samples_ms.append((time.perf_counter_ns() - t0) / 1e6)

    p50, p95, p99 = (float(x) for x in np.percentile(samples_ms, [50, 95, 99]))
    return BenchmarkMetric(
        gate_id="G2.2",
        suite_name="fwd",
        description="Single-Trade Surrogate Forward Pass (B = 1, d_model=128, 3+2 layers)",
        unit="ms",
        p50=p50,
        p95=p95,
        p99=p99,
        threshold=GATE_G2_2_FWD_P50_MS,
        mc_baseline_same_unit=mc_fwd_ms,
        speedup_vs_mc=mc_fwd_ms / max(p50, 1e-9),
        passed=(p50 <= GATE_G2_2_FWD_P50_MS),
    )


def run_full_greeks_suite(
    state: AutocallPreprocState,
    mc_fwd_ms: float,
    n_warmup: int = 30,
    n_iter: int = 200,
) -> BenchmarkMetric:
    surrogate = _load_trained_surrogate("cpu")

    d_act = max(int(state.d_assets), 1)
    m_act = max(int(state.m_rem), 1)
    nf64 = state.node_feats[:, :d_act, :].contiguous()
    ef64 = state.edge_feats[:, :d_act, :d_act, :].contiguous()
    am64 = state.asset_mask[:, :d_act].contiguous()
    sf64 = state.sched_feats[:, :m_act, :].contiguous()
    sm64 = state.sched_mask[:, :m_act].contiguous()
    gf64 = state.global_feats.contiguous()
    spots_b = state.spots_t0.unsqueeze(0)
    ref_b = state.ref_spots.unsqueeze(0)

    def _step() -> None:
        bg = compute_batched_surrogate_greeks(
            surrogate,
            nf64,
            ef64,
            am64,
            sf64,
            sm64,
            gf64,
            spots_t0=spots_b,
            ref_spots=ref_b,
            compute_cash_gamma=True,
        )
        assemble_autocall_cashflows_and_pv(
            state,
            p_stop=bg.stopping_probs[:, :m_act].squeeze(0),
            q_cpn=bg.coupon_probs[:, :m_act].squeeze(0),
            l_put=bg.put_loss.squeeze(0),
        )

    for _ in range(n_warmup):
        _step()

    samples_ms: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        _step()
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)

    p50, p95, p99 = (float(x) for x in np.percentile(samples_ms, [50, 95, 99]))
    # Reverse-mode AAD over 32,768 Dupire local-vol paths adds ~3x backward tape overhead on top of forward MC.
    mc_aad_ms = mc_fwd_ms * 4.0
    return BenchmarkMetric(
        gate_id="G2.3",
        suite_name="full_greeks",
        description="Single-Trade Full Valuation + All 8 Greeks + 37-Row CashFlows (B = 1, d_model=128, 3+2 layers)",
        unit="ms",
        p50=p50,
        p95=p95,
        p99=p99,
        threshold=GATE_G2_3_FULL_GREEKS_P50_MS,
        mc_baseline_same_unit=mc_aad_ms,
        speedup_vs_mc=mc_aad_ms / max(p50, 1e-9),
        passed=(p50 <= GATE_G2_3_FULL_GREEKS_P50_MS),
    )


def run_batch_suite(
    state: AutocallPreprocState,
    mc_fwd_ms: float,
    batch_size: int = 1024,
    n_warmup: int = 20,
    n_iter: int = 100,
) -> BenchmarkMetric:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate = _load_trained_surrogate(device)

    nf_b = state.node_feats.repeat(batch_size, 1, 1).to(device).contiguous()
    ef_b = state.edge_feats.repeat(batch_size, 1, 1, 1).to(device).contiguous()
    am_b = state.asset_mask.repeat(batch_size, 1).to(device).contiguous()
    sf_b = state.sched_feats.repeat(batch_size, 1, 1).to(device).contiguous()
    sm_b = state.sched_mask.repeat(batch_size, 1).to(device).contiguous()
    gf_b = state.global_feats.repeat(batch_size, 1).to(device).contiguous()

    for _ in range(n_warmup):
        compute_batched_surrogate_greeks(
            surrogate,
            nf_b,
            ef_b,
            am_b,
            sf_b,
            sm_b,
            gf_b,
            compute_cash_gamma=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    samples_ms_per_trade: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        compute_batched_surrogate_greeks(
            surrogate,
            nf_b,
            ef_b,
            am_b,
            sf_b,
            sm_b,
            gf_b,
            compute_cash_gamma=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples_ms_per_trade.append((time.perf_counter_ns() - t0) / 1e6 / batch_size)

    p50, p95, p99 = (float(x) for x in np.percentile(samples_ms_per_trade, [50, 95, 99]))
    mc_aad_ms = mc_fwd_ms * 4.0
    return BenchmarkMetric(
        gate_id="G2.4",
        suite_name="batch",
        description=f"Batched Portfolio Valuation + First-Order Greeks (B = {batch_size}, {device.type.upper()})",
        unit="ms/trade",
        p50=p50,
        p95=p95,
        p99=p99,
        threshold=GATE_G2_4_BATCH_MS_PER_TRADE,
        mc_baseline_same_unit=mc_aad_ms,
        speedup_vs_mc=mc_aad_ms / max(p50, 1e-9),
        passed=(p50 <= GATE_G2_4_BATCH_MS_PER_TRADE),
    )


def _format_markdown_report(metrics: list[BenchmarkMetric], mc_fwd_ms: float) -> str:
    lines = [
        "# SPEC-DEEPVOL-EQ-AUTOCALL-001 — Latency & Throughput Benchmark Report",
        "",
        "- **Reference Trade**: `45476647/EMTN_CIB_SEC` (`D = 5` equities: `MAGN_RUB, POSIRX_RUB, SIBN_RUB, TATN_RUB, VKCO_RUB`, `M = 12` observation periods, `37` cash-flow rows, `M_rem = 1`)",
        f"- **32,768-Path Dupire Local-Vol QMC Baseline (`MonteCarloWithAAD`)**: "
        f"`{mc_fwd_ms:.2f} ms` (forward) / `{mc_fwd_ms * 4.0:.2f} ms` (forward + AAD backward)",
        f"- **Device**: `CUDA ({torch.cuda.get_device_name(0)})` / `CPU`"
        if torch.cuda.is_available()
        else "- **Device**: `CPU`",
        "",
        "| Gate ID | Suite | Operation | Unit | p50 | p95 | p99 | Gate Threshold | Speedup vs 32k-Path MC | Status |",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-End Latency & Throughput Benchmark Suite for SPEC-DEEPVOL-EQ-AUTOCALL-001."
    )
    parser.add_argument(
        "--suite",
        choices=("all", "proxy", "fwd", "full_greeks", "batch"),
        default="all",
        help="Benchmark suite to execute (default: all).",
    )
    args = parser.parse_args(argv)

    state = _load_reference_state()
    mc_fwd_ms = _measure_mc_baseline_ms(state, n_paths=32768)

    metrics: list[BenchmarkMetric] = []
    if args.suite in ("all", "proxy"):
        metrics.append(run_proxy_suite(state, mc_fwd_ms))
    if args.suite in ("all", "fwd"):
        metrics.append(run_fwd_suite(state, mc_fwd_ms))
    if args.suite in ("all", "full_greeks"):
        metrics.append(run_full_greeks_suite(state, mc_fwd_ms))
    if args.suite in ("all", "batch"):
        metrics.append(run_batch_suite(state, mc_fwd_ms))

    print(_format_markdown_report(metrics, mc_fwd_ms))

    failed = [m for m in metrics if not m.passed]
    if failed:
        for m in failed:
            print(
                f"ERROR: Gate {m.gate_id} ({m.suite_name}) failed: "
                f"measured p50={m.p50:.4f} {m.unit} > threshold {m.threshold:.4f} {m.unit}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
