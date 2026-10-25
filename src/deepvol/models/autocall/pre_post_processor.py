"""pre_post_processor.py — Tier 1 Deterministic float64 Pre- and Post-Processor (Task 1.1c).

Implements SPEC-DEEPVOL-EQ-AUTOCALL-001 §1.2, §1.3.A, and Symbolic Proof W1 in strict
IEEE-754 float64 precision (`torch.float64` / `np.float64`):
  - Fixed-padding feature coordinate enums (`AutocallNodeCol`, `AutocallEdgeCol`,
    `AutocallSchedCol`, `AutocallGlobalCol`) with D_max = 8 and M_max = 48.
  - `preprocess_eq_autocall(...)`: end-to-end differentiable `torch.float64` pre-processor
    constructing `AutocallPreprocState` from `EQAutocallableTradeRequest`, raw Murex
    `ConstantMessageMap` text/files (`eq_autocallable_request.txt`), or structured dicts.
  - `.to_tier2_float32_tensors()`: single precision-segregation boundary (REM-S0-01 & Gate G4.4)
    casting only the 6 padded neural input tensors to `torch.float32` while preserving autograd.
  - `assemble_autocall_cashflows_and_pv(...)`: exact affine stopping-time cash-flow & PV
    post-processor (`W1`), reproducing all 37 `CashFlows` rows of `eq_autocallable_request.txt`
    (Gate G1.3).
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
import datetime
from enum import IntEnum
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch

from deepvol.models.autocall.analytical_baseline import (
    AutocallBaselineOutput,
    compute_analytical_autocall_baseline,
)
from deepvol.models.autocall.dupire_qmc_solver import RegularizedVolSurface
from deepvol.models.autocall.escrowed_dividend import (
    EscrowedDividendOutput,
    build_discount_rate_evaluator,
    compute_escrowed_dividend_forwards,
)
from deepvol.models.autocall.seasoned_collapser import (
    CashFlowItem,
    SeasonedScheduleState,
    _to_utc_datetime,
    _year_fraction,
    collapse_seasoned_schedule,
)

D_MAX_DEFAULT: int = 8
M_MAX_DEFAULT: int = 48


class AutocallNodeCol(IntEnum):
    """14-column node feature layout for X_node in R^{B x D_max x 14} (§1.3.A.1)."""

    LOG_MONEYNESS = 0
    LOG_FWD_CARRY_1Y = 1
    CASH_DIV_LEVERAGE = 2
    REPO_SPREAD_AVG = 3
    IV_SHORT_ATM = 4
    IV_TERM_ATM = 5
    IV_TERM_SLOPE = 6
    SSVI_SKEW_RHO = 7
    SSVI_CURV_ETA = 8
    SMILE_RESIDUAL_PC1 = 9
    SMILE_RESIDUAL_PC2 = 10
    DIST_TO_PUT_STRIKE = 11
    PROXY_DELTA_WEIGHT = 12
    ASSET_ACTIVE_MASK = 13


class AutocallEdgeCol(IntEnum):
    """4-column edge feature layout for X_edge in R^{B x D_max x D_max x 4} (§1.3.A.2)."""

    CORR_RHO = 0
    COVARIANCE_TERM = 1
    LOG_PERF_DIFF = 2
    OFF_DIAG_MASK = 3


class AutocallSchedCol(IntEnum):
    """12-column schedule feature layout for X_sched in R^{B x M_max x 12} (§1.3.A.3)."""

    TAU_OBS = 0
    DELTA_TAU = 1
    DF_PAY = 2
    LOG_KO_BARRIER = 3
    IS_CALLABLE_FLAG = 4
    LOG_CPN_BARRIER = 5
    IS_GUARANTEED_CPN = 6
    COUPON_RATE = 7
    STULZ_FWD_W = 8
    STULZ_VOL_W = 9
    PROXY_KO_PROB = 10
    PROXY_CPN_PROB = 11


class AutocallGlobalCol(IntEnum):
    """10-column global context feature layout for X_global in R^{B x 10} (§1.3.A.4)."""

    K_PUT = 0
    B_KI = 1
    HAS_KI_FLAG = 2
    I_KI_PAST = 3
    G_PUT = 4
    C_MEM_0 = 5
    IS_MEMORY_FLAG = 6
    ABS_KO_SPREAD = 7
    ABS_CPN_SPREAD = 8
    L_PUT_PROXY = 9


class _LazySurfaceJacobianDict(dict[str, npt.NDArray[np.float64]]):
    """Lazily computes and caches surface quote Jacobians only if accessed."""

    def __init__(self, builder: Callable[[str], npt.NDArray[np.float64]], keys: Sequence[str]) -> None:
        super().__init__()
        self._builder = builder
        self._keys = list(keys)

    def _populate_all(self) -> None:
        for k in self._keys:
            if not super().__contains__(k):
                super().__setitem__(k, self._builder(k))

    def __getitem__(self, key: str) -> npt.NDArray[np.float64]:
        if not super().__contains__(key) and key in self._keys:
            super().__setitem__(key, self._builder(key))
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._keys:
            return self[key]
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._keys or super().__contains__(key)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Any:
        self._populate_all()
        return super().__iter__()

    def keys(self) -> Any:
        self._populate_all()
        return super().keys()

    def values(self) -> Any:
        self._populate_all()
        return super().values()

    def items(self) -> Any:
        self._populate_all()
        return super().items()


@dataclass
class AutocallPreprocState:
    """Complete Tier 1 pre-processed state in IEEE-754 float64 (`torch.float64` / `np.float64`)."""

    # Padded feature tensors in torch.float64: shapes (B, D_max, 14), (B, D_max, D_max, 4), etc.
    node_feats: torch.Tensor
    edge_feats: torch.Tensor
    asset_mask: torch.Tensor
    sched_feats: torch.Tensor
    sched_mask: torch.Tensor
    global_feats: torch.Tensor

    # Dimensions & canonical ordering
    d_assets: int
    m_rem: int
    d_max: int
    m_max: int
    original_identifiers: list[str]
    sorted_identifiers: list[str]
    lex_perm: npt.NDArray[np.int64]
    inv_lex_perm: npt.NDArray[np.int64]

    # Active unpadded float64 tensors for analytical baseline, Greeks, and W1 post-processor
    spots_t0: torch.Tensor
    ref_spots: torch.Tensor
    moneyness_t0: torch.Tensor
    corr_matrix: torch.Tensor
    tau_obs: torch.Tensor
    delta_tau: torch.Tensor
    tau_pay: torch.Tensor
    df_obs: torch.Tensor
    df_pay: torch.Tensor
    ko_barriers: torch.Tensor
    is_callable: torch.Tensor
    cpn_barriers: torch.Tensor
    is_guaranteed_cpn: torch.Tensor
    coupon_amounts: torch.Tensor
    forward_rel: torch.Tensor
    forward_perf: torch.Tensor
    cash_div_leverage: torch.Tensor
    iv_at_fwd: torch.Tensor
    v_perf: torch.Tensor

    # Sub-engine states & Jacobians
    seasoned_state: SeasonedScheduleState
    dividend_schedule: EscrowedDividendOutput
    vol_surfaces: dict[str, RegularizedVolSurface]
    surface_feature_jacobians: dict[str, npt.NDArray[np.float64]]
    iv_at_fwd_jacobians: dict[str, npt.NDArray[np.float64]]
    discount_curve: Any
    baseline: AutocallBaselineOutput

    # Product payoff & contract terms
    strike: float = 0.90
    option_gearing: float = 1.11111
    lower_strike: float | None = None
    ki_barrier: float | None = None
    i_ki_past: float = 0.0
    c_mem_0: float = 0.0
    is_memory: bool = True
    ko_spread: float = -0.15
    cpn_spread: float = -0.15
    ki_spread: float = -0.15
    no_notional_repayment: bool = False
    autocall_coupon: float = 0.0
    force_coupons_on_autocall: bool = False
    allow_autocall_on_expiry: bool = False
    participation_call_strike: float | None = None
    participation_call_rate: float = 0.0
    one_star_strike: float | None = None
    one_star_coupon: float = 0.0
    is_long_basket: bool = False
    payment_currency: str = "RUB"
    callability_index: str = "WorstOf"
    coupon_index: str | None = None
    option_index: str | None = None
    funding_leg: dict[str, Any] | None = None
    pv_in_transit: float = 0.0
    already_autocalled: bool = False
    trade_info: str | None = None
    queries: list[str] = field(default_factory=lambda: ["ProductValue", "CashFlows"])
    coupon_barrier_type: str = "In"
    coupon_barrier_direction: str = "Up"

    def to_tier2_float32_tensors(self) -> dict[str, torch.Tensor]:
        """Precision boundary segregation (REM-S0-01 & Gate G4.4).

        Casts ONLY the 6 fixed-padding feature/mask tensors to `torch.float32` for Tier 2
        neural surrogate inference while maintaining autograd connectivity to upstream
        `torch.float64` tensors.
        """
        return {
            "node_feats": self.node_feats.to(dtype=torch.float32),
            "edge_feats": self.edge_feats.to(dtype=torch.float32),
            "asset_mask": self.asset_mask.to(dtype=torch.float32),
            "sched_feats": self.sched_feats.to(dtype=torch.float32),
            "sched_mask": self.sched_mask.to(dtype=torch.float32),
            "global_feats": self.global_feats.to(dtype=torch.float32),
        }

    @property
    def canonical_codes(self) -> list[str]:
        return self.sorted_identifiers

    @property
    def schedule_state(self) -> SeasonedScheduleState:
        return self.seasoned_state

    @property
    def escrowed_dividends(self) -> EscrowedDividendOutput:
        return self.dividend_schedule

    @property
    def node_features(self) -> torch.Tensor:
        return self.node_feats

    @property
    def edge_features(self) -> torch.Tensor:
        return self.edge_feats

    @property
    def schedule_features(self) -> torch.Tensor:
        return self.sched_feats

    @property
    def global_context(self) -> torch.Tensor:
        return self.global_feats

    def unpermute_assets(self, tensor_or_array: torch.Tensor | npt.NDArray[np.float64]) -> torch.Tensor | npt.NDArray[np.float64]:
        """Applies the inverse lexicographical permutation pi_lex^{-1} along dim 0."""
        if isinstance(tensor_or_array, torch.Tensor):
            inv_idx = torch.as_tensor(self.inv_lex_perm, dtype=torch.long, device=tensor_or_array.device)
            return tensor_or_array.index_select(0, inv_idx)
        return tensor_or_array[self.inv_lex_perm]


@dataclass
class AutocallPostprocOutput:
    """Output of the W1 affine stopping-time cash-flow and ProductValue assembler in float64."""

    product_value: torch.Tensor
    notional_pv: torch.Tensor
    coupon_pv: torch.Tensor
    option_pv: torch.Tensor
    funding_pv: torch.Tensor
    pv_in_transit: torch.Tensor
    survival_probs: torch.Tensor
    p_stop: torch.Tensor
    q_cpn: torch.Tensor
    l_put: torch.Tensor
    l_call: torch.Tensor
    cashflows: list[CashFlowItem]
    standard_error: float = 0.0
    reporting_currency: str = "RUB"
    cash_to_settle_today: float = 0.0
    l_onestar: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0, dtype=torch.float64))

    @property
    def cash_flows(self) -> list[CashFlowItem]:
        return self.cashflows


# ── Built-in Murex ConstantMessageMap DAG Fallback Parser ────────────────────


def _parse_murex_block_body(body_str: str) -> dict[str, Any]:
    """Parses a Murex ConstantMessageMap block body into a Python dictionary."""
    s = body_str.strip()
    # Replace `<skipped>` with None
    s = re.sub(r"<skipped>", "None", s)
    # Replace `@<40-hex>` references with string literals `"@<40-hex>"`
    s = re.sub(r"@([0-9a-f]{40})", r'"@\1"', s)
    # Quote unquoted keys before colons (including keys with `&` like `MAGN_RUB&POSIRX_RUB:`)
    s = re.sub(r"(?m)^(\s*)([A-Za-z0-9_&]+)\s*:", r'\1"\2":', s)
    # Insert commas at the end of value lines inside lists or dicts
    lines = s.splitlines()
    out_lines: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Look ahead for next non-empty line
        next_stripped = ""
        for j in range(i + 1, len(lines)):
            if lines[j].strip():
                next_stripped = lines[j].strip()
                break
        needs_comma = (
            not stripped.endswith(("{", "[", ":", ","))
            and next_stripped != ""
            and not next_stripped.startswith(("}", "]"))
        )
        if needs_comma:
            out_lines.append(line + ",")
        else:
            out_lines.append(line)

    py_expr = "\n".join(out_lines)
    # Replace datetime.datetime(...) calls with ISO strings before literal_eval
    def _dt_repl(match: re.Match[str]) -> str:
        args = [int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()]
        y, mo, d = args[0], args[1], args[2]
        hh = args[3] if len(args) > 3 else 0
        mm = args[4] if len(args) > 4 else 0
        ss = args[5] if len(args) > 5 else 0
        return f'"{y:04d}-{mo:02d}-{d:02d}T{hh:02d}:{mm:02d}:{ss:02d}+00:00"'

    py_expr = re.sub(r"datetime\.datetime\(([^)]+)\)", _dt_repl, py_expr)
    parsed = ast.literal_eval(py_expr)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed ConstantMessageMap block is not a dict.")
    return parsed


def parse_murex_text_or_dict_fallback(
    raw_input: str | Path | Mapping[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    """Parses a Murex ConstantMessageMap dump (or file path / dict) into resolved request & reference result."""
    try:
        from deepvol.api.schemas_eq_autocall import parse_murex_autocall_dag

        if isinstance(raw_input, Path):
            raw_text = raw_input.read_text(encoding="utf-8")
        else:
            raw_text = raw_input  # type: ignore[assignment]
        req_obj, ref_dict = parse_murex_autocall_dag(raw_text)
        return req_obj, ref_dict
    except Exception:
        pass

    if isinstance(raw_input, Mapping):
        return dict(raw_input), None

    if isinstance(raw_input, Path):
        text = raw_input.read_text(encoding="utf-8")
    elif isinstance(raw_input, str) and len(raw_input) < 512 and "\n" not in raw_input and Path(raw_input).is_file():
        text = Path(raw_input).read_text(encoding="utf-8")
    else:
        text = str(raw_input)

    # Split root reference `@<40-hex>` if present
    root_match = re.search(r"(?m)^@([0-9a-f]{40})\s*$", text)
    ref_result: dict[str, Any] | None = None
    if root_match:
        root_id = root_match.group(1)
        prefix_text = text[: root_match.start()].strip()
        if prefix_text.startswith("{"):
            ref_result = _parse_murex_block_body(prefix_text)
        dag_text = text[root_match.end() :].strip()
    else:
        root_id = ""
        dag_text = text.strip()

    block_matches = list(
        re.finditer(
            r"(?m)^([0-9a-f]{40})\s*\nclass:\s*ConstantMessageMap\s*\n(\{.*?\n\})",
            dag_text,
            re.DOTALL,
        )
    )
    if not block_matches:
        # Single dictionary payload
        return _parse_murex_block_body(dag_text), ref_result

    nodes: dict[str, dict[str, Any]] = {}
    for bm in block_matches:
        nid = bm.group(1)
        nbody = bm.group(2)
        nodes[nid] = _parse_murex_block_body(nbody)
        if not root_id:
            root_id = nid

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, str) and obj.startswith("@") and len(obj) == 41:
            ref_id = obj[1:]
            if ref_id in nodes:
                return _resolve(nodes[ref_id])
            return obj
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(v) for v in obj]
        return obj

    resolved_root = _resolve(nodes[root_id])
    return resolved_root, ref_result


def _extract_attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Extracts the first matching key or attribute from a dict or Pydantic model."""
    if obj is None:
        return default
    if isinstance(obj, (dict, Mapping)):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
        return default
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return default


def _build_correlation_matrix(
    sorted_ids: Sequence[str],
    corr_input: Mapping[str, float] | Sequence[Sequence[float]] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Constructs the (D, D) correlation matrix in canonical sorted asset order in torch.float64."""
    d_assets = len(sorted_ids)
    if isinstance(corr_input, torch.Tensor):
        return corr_input.to(dtype=torch.float64, device=device)
    if not isinstance(corr_input, Mapping):
        return torch.as_tensor(np.asarray(corr_input, dtype=np.float64), dtype=torch.float64, device=device)

    mat = np.eye(d_assets, dtype=np.float64)
    for i in range(d_assets):
        for j in range(i + 1, d_assets):
            id_i = sorted_ids[i]
            id_j = sorted_ids[j]
            k1 = f"{id_i}&{id_j}"
            k2 = f"{id_j}&{id_i}"
            if k1 in corr_input:
                c_val = float(corr_input[k1])
            elif k2 in corr_input:
                c_val = float(corr_input[k2])
            else:
                raise KeyError(f"Missing pairwise correlation for {k1!r}.")
            mat[i, j] = c_val
            mat[j, i] = c_val
    return torch.as_tensor(mat, dtype=torch.float64, device=device)


_PREPROC_STATE_CACHE: dict[Any, AutocallPreprocState] = {}


def _try_make_preproc_cache_key(
    *,
    d_max: int,
    m_max: int,
    as_of_dt: datetime.datetime,
    start_dt: datetime.datetime,
    payment_ccy: str,
    dcc: str,
    underlying_ids: list[str],
    ref_spots: list[float],
    obs_dates: list[Any],
    pay_dates: list[Any],
    ko_levels: list[float],
    cpn_levels: list[float],
    cpn_amounts: list[float] | None,
    fixed_cpn_amt: Any,
    fixed_rate: Any,
    is_memory: bool,
    callability_index: str,
    coupon_index: str | None,
    option_index: str | None,
    strike: float,
    option_gearing: float,
    lower_strike: float | None,
    ki_barrier: float | None,
    ko_spread: float,
    cpn_spread: float,
    ki_spread: float,
    no_notional_repayment: bool,
    autocall_coupon: float,
    force_coupons_on_autocall: bool,
    allow_autocall_on_expiry: bool,
    is_long_basket: bool,
    part_strike: float | None,
    part_rate: float,
    one_star_strike: float | None,
    one_star_coupon: float,
    cpn_barrier_type: str,
    cpn_barrier_direction: str,
    funding_leg_dict: dict[str, Any] | None,
    disc_curve_spec: Any,
    spots_map: Any,
    fixings_map: Any,
    repo_curves_map: Any,
    dividends_map: Any,
    predef_divs_map: Any,
    vol_surfaces_map: Any,
    correlations_map: Any,
) -> Any | None:
    try:
        if not isinstance(spots_map, (dict, Mapping)) or any(
            isinstance(v, torch.Tensor) and v.requires_grad for v in spots_map.values()
        ):
            return None
        spots_k = tuple((str(k), float(v)) for k, v in sorted(spots_map.items()))
        corr_k = (
            tuple((str(k), float(v)) for k, v in sorted(correlations_map.items()))
            if isinstance(correlations_map, (dict, Mapping))
            else str(correlations_map)
        )
        if isinstance(disc_curve_spec, (dict, Mapping)):
            disc_k: Any = str(disc_curve_spec)
        else:
            rs = _extract_attr(disc_curve_spec, "RatesSchedule", "rates_schedule")
            disc_k = (
                str(_extract_attr(disc_curve_spec, "Identifier", "identifier")),
                str(_extract_attr(disc_curve_spec, "InterpolationMethod", "interpolation_method")),
                str(_extract_attr(disc_curve_spec, "InterpolationSpaceCoordinates", "space_coords")),
                tuple(_extract_attr(rs, "Rates", "rates", default=()) or ())
                if rs is not None
                else (),
                tuple(_extract_attr(rs, "Spreads", "spreads", default=()) or ())
                if rs is not None and _extract_attr(rs, "Spreads", "spreads") is not None
                else (),
            )
        divs_k = (
            tuple(
                (
                    str(u),
                    tuple(
                        float(_extract_attr(it, "Amount", "amount", default=0.0))
                        for it in (
                            _extract_attr(
                                dividends_map[u], "CashDividendList", "cash_dividend_list", default=()
                            )
                            or ()
                        )
                    ),
                )
                for u in sorted(dividends_map.keys())
            )
            if isinstance(dividends_map, (dict, Mapping))
            else None
        )
        pdivs_k = str(predef_divs_map) if predef_divs_map else None
        surf_list: list[Any] = []
        if isinstance(vol_surfaces_map, (dict, Mapping)):
            for u in sorted(vol_surfaces_map.keys()):
                vq: Any = _extract_attr(vol_surfaces_map[u], "VolatilityQuotes", "volatility_quotes", default=()) or ()
                if len(vq) > 0 and len(vq[0]) > 0:
                    surf_list.append((str(u), len(vq), len(vq[0]), float(vq[0][0]), float(vq[-1][-1])))
                else:
                    surf_list.append((str(u), 0))
        surf_k = tuple(surf_list)
        repo_k = (
            tuple(sorted(repo_curves_map.keys()))
            if isinstance(repo_curves_map, (dict, Mapping))
            else None
        )
        fix_k = (
            tuple(
                (
                    str(u),
                    len(_extract_attr(fixings_map[u], "FixingList", "fixing_list", default=()) or ()),
                )
                for u in sorted(fixings_map.keys())
            )
            if isinstance(fixings_map, (dict, Mapping))
            else None
        )
        f_leg_k = (
            (
                funding_leg_dict.get("rate_identifier"),
                funding_leg_dict.get("funding_spread"),
                tuple(str(d) for d in funding_leg_dict.get("payment_dates", ())),
            )
            if funding_leg_dict is not None
            else None
        )
        return (
            d_max,
            m_max,
            as_of_dt,
            start_dt,
            payment_ccy,
            dcc,
            tuple(underlying_ids),
            tuple(ref_spots),
            tuple(str(d) for d in obs_dates),
            tuple(str(d) for d in pay_dates),
            tuple(ko_levels),
            tuple(cpn_levels),
            tuple(cpn_amounts) if cpn_amounts is not None else None,
            fixed_cpn_amt,
            fixed_rate,
            is_memory,
            callability_index,
            coupon_index,
            option_index,
            strike,
            option_gearing,
            lower_strike,
            ki_barrier,
            ko_spread,
            cpn_spread,
            ki_spread,
            no_notional_repayment,
            autocall_coupon,
            force_coupons_on_autocall,
            allow_autocall_on_expiry,
            is_long_basket,
            part_strike,
            part_rate,
            one_star_strike,
            one_star_coupon,
            cpn_barrier_type,
            cpn_barrier_direction,
            f_leg_k,
            disc_k,
            spots_k,
            corr_k,
            divs_k,
            pdivs_k,
            surf_k,
            repo_k,
            fix_k,
        )
    except Exception:
        return None


def preprocess_eq_autocall(
    req_or_raw: Any,
    *,
    spots_override: torch.Tensor | None = None,
    corr_override: torch.Tensor | None = None,
    df_pay_override: torch.Tensor | None = None,
    dividend_amount_overrides: Mapping[str, torch.Tensor] | None = None,
    as_of_date_override: datetime.datetime | datetime.date | str | None = None,
    d_max: int = D_MAX_DEFAULT,
    m_max: int = M_MAX_DEFAULT,
) -> AutocallPreprocState:
    """Executes the Tier 1 deterministic `torch.float64` pre-processor on an Autocallable trade request.

    Accepts:
      - `EQAutocallableTradeRequest` or `EQAutocallableMLRequest` (Pydantic V2 models),
      - Raw Murex `ConstantMessageMap` string or `Path` (`eq_autocallable_request.txt`),
      - Structured Python `dict`.
    """
    if isinstance(req_or_raw, (str, Path)):
        req_obj, _ = parse_murex_text_or_dict_fallback(req_or_raw)
    else:
        req_obj = req_or_raw

    product = _extract_attr(req_obj, "Product", "product")
    model = _extract_attr(req_obj, "Model", "model")
    mkt = _extract_attr(model, "MarketDataSet", "market_data_set") if model is not None else None
    if mkt is None:
        mkt = _extract_attr(req_obj, "MarketDataSet", "market_data")

    if product is None or mkt is None:
        raise ValueError("Request must contain Product and Model.MarketDataSet (or market_data).")

    trade_info = _extract_attr(req_obj, "TradeInfo", "trade_info", "trade_id")
    queries_raw = _extract_attr(req_obj, "Queries", "queries", default=["ProductValue", "CashFlows"])
    queries = [str(getattr(q, "value", q)) for q in queries_raw]

    as_of_raw = (
        as_of_date_override
        if as_of_date_override is not None
        else _extract_attr(mkt, "AsOfDate", "as_of_date")
    )
    as_of_dt = _to_utc_datetime(as_of_raw)

    start_dt = _to_utc_datetime(_extract_attr(product, "StartDate", "start_date"))
    payment_ccy = str(_extract_attr(product, "PaymentCurrency", "payment_currency", default="RUB"))
    dcc_raw = _extract_attr(product, "DayCountConvention", "day_count_convention", default="Act360")
    dcc = str(getattr(dcc_raw, "value", dcc_raw))

    underlying_ids = [
        str(x) for x in _extract_attr(product, "UnderlyingIdentifiers", "underlying_identifiers")
    ]
    ref_spots = [
        float(x) for x in _extract_attr(product, "UnderlyingReferenceSpots", "underlying_reference_spots")
    ]

    # Extract observation & payment dates and barrier schedules
    periods_obj = _extract_attr(product, "periods")
    if periods_obj is not None:
        obs_dates = [_extract_attr(p, "observation_date") for p in periods_obj]
        pay_dates = [_extract_attr(p, "payment_date") for p in periods_obj]
        ko_levels = [
            float(_extract_attr(p, "autocall_barrier", default=500.0) or 500.0) for p in periods_obj
        ]
        cpn_levels = [float(_extract_attr(p, "coupon_barrier", default=1e-5)) for p in periods_obj]
        cpn_amounts: list[float] | None = [
            float(_extract_attr(p, "coupon_amount", default=0.0)) for p in periods_obj
        ]
        fixed_cpn_amt = None
        fixed_rate = None
    else:
        obs_dates = list(_extract_attr(product, "ObservationDates", "observation_dates"))
        pay_dates = list(_extract_attr(product, "PaymentDates", "payment_dates"))
        ko_levels = [float(x) for x in _extract_attr(product, "AutoCallTriggerLevels", "autocall_trigger_levels")]
        raw_cpn_levels = _extract_attr(product, "CouponBarrierLevels", "coupon_barrier_levels")
        cpn_levels = [float(x) for x in raw_cpn_levels] if raw_cpn_levels is not None else [1e-5] * len(obs_dates)
        cpn_amounts = None
        fixed_cpn_amt = _extract_attr(product, "FixedCouponAmount", "fixed_coupon_amount")
        fixed_rate = _extract_attr(product, "FixedRate", "fixed_rate")

    is_memory = bool(_extract_attr(product, "IsMemory", "is_memory", default=False))
    call_idx_raw = _extract_attr(
        product, "CallabilityMonitoringIndex", "callability_index", default="WorstOf"
    )
    callability_index = str(getattr(call_idx_raw, "value", call_idx_raw))

    cpn_idx_raw = _extract_attr(product, "CouponBarrierMonitoringIndex", "coupon_index")
    coupon_index = str(getattr(cpn_idx_raw, "value", cpn_idx_raw)) if cpn_idx_raw is not None else None

    opt_idx_raw = _extract_attr(product, "OptionMonitoringIndex", "option_index")
    option_index = str(getattr(opt_idx_raw, "value", opt_idx_raw)) if opt_idx_raw is not None else None

    # Downside put & modifiers
    downside_put = _extract_attr(product, "downside_put")
    modifiers = _extract_attr(product, "modifiers")

    strike = float(
        _extract_attr(downside_put, "strike")
        if downside_put is not None
        else _extract_attr(product, "Strike", "strike", default=0.90)
    )
    option_gearing = float(
        _extract_attr(downside_put, "gearing")
        if downside_put is not None
        else _extract_attr(product, "OptionGearing", "option_gearing", default=1.11111)
    )
    lower_strike_raw = (
        _extract_attr(downside_put, "lower_strike")
        if downside_put is not None
        else _extract_attr(product, "LowerStrike", "lower_strike")
    )
    lower_strike = float(lower_strike_raw) if lower_strike_raw is not None else None

    ki_barrier_raw = (
        _extract_attr(downside_put, "knock_in_barrier")
        if downside_put is not None
        else _extract_attr(product, "OptionDownAndInBarrierLevel", "knock_in_barrier")
    )
    ki_barrier = float(ki_barrier_raw) if ki_barrier_raw is not None else None

    ko_spread = float(
        _extract_attr(
            product,
            "AutoCallTriggerAbsoluteDigitalSpread",
            "autocall_digital_spread",
            default=-0.15,
        )
    )
    cpn_spread_raw = _extract_attr(
        product,
        "CouponBarrierAbsoluteDigitalSpread",
        "coupon_digital_spread",
    )
    cpn_spread = float(cpn_spread_raw) if cpn_spread_raw is not None else ko_spread

    ki_spread_raw = (
        _extract_attr(downside_put, "knock_in_digital_spread")
        if downside_put is not None
        else _extract_attr(product, "OptionDownAndInBarrierAbsoluteDigitalSpread")
    )
    ki_spread = float(ki_spread_raw) if ki_spread_raw is not None else ko_spread

    allow_autocall_on_expiry = bool(
        _extract_attr(modifiers, "allow_autocall_on_expiry")
        if modifiers is not None
        else _extract_attr(product, "AllowAutocallOnExpiry", "allow_autocall_on_expiry", default=False)
    )
    force_coupons_on_autocall = bool(
        _extract_attr(modifiers, "force_coupons_on_autocall")
        if modifiers is not None
        else _extract_attr(product, "ForceCouponsOnAutocall", "force_coupons_on_autocall", default=False)
    )
    autocall_coupon = float(
        _extract_attr(modifiers, "autocall_redemption_coupon", default=0.0)
        if modifiers is not None
        else _extract_attr(product, "AutoCallCoupon", "autocall_coupon", default=0.0)
    )
    no_notional_repayment = bool(
        _extract_attr(modifiers, "no_notional_repayment")
        if modifiers is not None
        else _extract_attr(product, "NoNotionalRepayment", "no_notional_repayment", default=False)
    )
    is_long_basket = bool(
        _extract_attr(modifiers, "is_long_basket")
        if modifiers is not None
        else _extract_attr(product, "IsLongBasket", "is_long_basket", default=False)
    )
    part_strike_raw = (
        _extract_attr(modifiers, "participation_call_strike")
        if modifiers is not None
        else _extract_attr(product, "ParticipationCallStrike", "participation_call_strike")
    )
    part_strike = float(part_strike_raw) if part_strike_raw is not None else None
    part_rate = float(
        _extract_attr(modifiers, "participation_call_rate", default=0.0)
        if modifiers is not None
        else _extract_attr(product, "ParticipationCallRate", "participation_call_rate", default=0.0)
    )
    one_star_strike_raw = (
        _extract_attr(modifiers, "one_star_strike")
        if modifiers is not None
        else _extract_attr(product, "OneStarStrike", "one_star_strike")
    )
    one_star_strike = float(one_star_strike_raw) if one_star_strike_raw is not None else None
    one_star_coupon = float(
        _extract_attr(modifiers, "one_star_coupon", default=0.0)
        if modifiers is not None
        else _extract_attr(product, "OneStarCoupon", "one_star_coupon", default=0.0)
    )

    cpn_barrier_type_raw = _extract_attr(
        product, "CouponBarrierType", "coupon_barrier_type", default="In"
    )
    cpn_barrier_type = str(getattr(cpn_barrier_type_raw, "value", cpn_barrier_type_raw) or "In")
    cpn_barrier_dir_raw = _extract_attr(
        product, "CouponBarrierDirection", "coupon_barrier_direction", default="Up"
    )
    cpn_barrier_direction = str(getattr(cpn_barrier_dir_raw, "value", cpn_barrier_dir_raw) or "Up")

    # Floating FundingLeg extraction (supports both floating RateIdentifier and fixed-spread FundingLeg)
    funding_leg_obj = _extract_attr(product, "funding_leg")
    rate_id = _extract_attr(product, "RateIdentifier", "rate_identifier")
    funding_leg_dates = _extract_attr(product, "FundingLegPaymentDates", "funding_leg_payment_dates")
    funding_spread = float(_extract_attr(product, "FundingSpread", "funding_spread", default=0.0))
    funding_leg_dict: dict[str, Any] | None = None
    if funding_leg_obj is not None:
        raw_f_dates = _extract_attr(funding_leg_obj, "payment_dates", default=[])
        if raw_f_dates:
            raw_rid = _extract_attr(funding_leg_obj, "rate_identifier")
            funding_leg_dict = {
                "rate_identifier": str(raw_rid) if raw_rid is not None else None,
                "funding_spread": float(_extract_attr(funding_leg_obj, "funding_spread", default=0.0)),
                "payment_dates": [_to_utc_datetime(d) for d in raw_f_dates],
            }
    elif funding_leg_dates:
        funding_leg_dict = {
            "rate_identifier": str(rate_id) if rate_id is not None else None,
            "funding_spread": funding_spread,
            "payment_dates": [_to_utc_datetime(d) for d in funding_leg_dates],
        }

    # Resolve discount curve from MarketDataSet
    rates_bundles = _extract_attr(mkt, "RatesCurvesBundles", "rates_curves_bundles")
    disc_curve_spec = _extract_attr(mkt, "discount_curve")
    if disc_curve_spec is None and rates_bundles:
        fin_ids = _extract_attr(model, "FinancingCurveIdentifiers", "financing_curve_identifiers")
        if fin_ids and len(fin_ids) > 0 and fin_ids[0] in rates_bundles:
            disc_curve_spec = rates_bundles[fin_ids[0]]
        else:
            disc_curve_spec = next(iter(rates_bundles.values()))

    spots_map = _extract_attr(mkt, "Spots", "spots")
    fixings_map = _extract_attr(mkt, "Fixings", "fixings")
    repo_curves_map = _extract_attr(mkt, "RepoSpreadCurves", "repo_spread_curves", "repo_curves", default={})
    dividends_map = _extract_attr(mkt, "Dividends", "dividends", default={})
    predef_divs_map = _extract_attr(
        product,
        "PredefinedDividends",
        "predefined_dividends",
        default=_extract_attr(mkt, "predefined_dividends", default={}),
    )
    vol_surfaces_map = _extract_attr(mkt, "VolatilitySurfaces", "volatility_surfaces", default={})
    correlations_map = _extract_attr(mkt, "Correlations", "correlations")

    cache_key: Any | None = None
    if (
        spots_override is None
        and corr_override is None
        and df_pay_override is None
        and dividend_amount_overrides is None
        and as_of_date_override is None
    ):
        cache_key = _try_make_preproc_cache_key(
            d_max=d_max,
            m_max=m_max,
            as_of_dt=as_of_dt,
            start_dt=start_dt,
            payment_ccy=payment_ccy,
            dcc=dcc,
            underlying_ids=underlying_ids,
            ref_spots=ref_spots,
            obs_dates=obs_dates,
            pay_dates=pay_dates,
            ko_levels=ko_levels,
            cpn_levels=cpn_levels,
            cpn_amounts=cpn_amounts,
            fixed_cpn_amt=fixed_cpn_amt,
            fixed_rate=fixed_rate,
            is_memory=is_memory,
            callability_index=callability_index,
            coupon_index=coupon_index,
            option_index=option_index,
            strike=strike,
            option_gearing=option_gearing,
            lower_strike=lower_strike,
            ki_barrier=ki_barrier,
            ko_spread=ko_spread,
            cpn_spread=cpn_spread,
            ki_spread=ki_spread,
            no_notional_repayment=no_notional_repayment,
            autocall_coupon=autocall_coupon,
            force_coupons_on_autocall=force_coupons_on_autocall,
            allow_autocall_on_expiry=allow_autocall_on_expiry,
            is_long_basket=is_long_basket,
            part_strike=part_strike,
            part_rate=part_rate,
            one_star_strike=one_star_strike,
            one_star_coupon=one_star_coupon,
            cpn_barrier_type=cpn_barrier_type,
            cpn_barrier_direction=cpn_barrier_direction,
            funding_leg_dict=funding_leg_dict,
            disc_curve_spec=disc_curve_spec,
            spots_map=spots_map,
            fixings_map=fixings_map,
            repo_curves_map=repo_curves_map,
            dividends_map=dividends_map,
            predef_divs_map=predef_divs_map,
            vol_surfaces_map=vol_surfaces_map,
            correlations_map=correlations_map,
        )
        if cache_key is not None and cache_key in _PREPROC_STATE_CACHE:
            cached_template = _PREPROC_STATE_CACHE[cache_key]
            cloned = copy.copy(cached_template)
            cloned.queries = list(queries)
            cloned.trade_info = str(trade_info) if trade_info is not None else None
            return cloned

    # Wrap in ClampedYieldCurve if available from Worker A's eq_curves.py
    disc_curve_obj: Any = disc_curve_spec
    try:
        from deepvol.market.eq_curves import ClampedYieldCurve

        if not isinstance(disc_curve_spec, ClampedYieldCurve):
            disc_curve_obj = ClampedYieldCurve(disc_curve_spec)
    except Exception:
        disc_curve_obj = disc_curve_spec

    if funding_leg_dict is not None:
        f_rid = funding_leg_dict.get("rate_identifier")
        f_curve_obj: Any = None
        if f_rid is not None:
            f_spec = rates_bundles.get(f_rid) if isinstance(rates_bundles, Mapping) else None
            if f_spec is not None:
                try:
                    from deepvol.market.eq_curves import ClampedYieldCurve

                    f_curve_obj = f_spec if isinstance(f_spec, ClampedYieldCurve) else ClampedYieldCurve(f_spec)
                except Exception:
                    f_curve_obj = f_spec
            else:
                f_curve_obj = disc_curve_obj
        funding_leg_dict["funding_curve"] = f_curve_obj

    _, df_fin_fn = build_discount_rate_evaluator(disc_curve_obj, as_of_dt)

    # 1. Seasoned Schedule Partitioning & Markovian Collapse
    seasoned = collapse_seasoned_schedule(
        as_of_date=as_of_dt,
        start_date=start_dt,
        observation_dates=obs_dates,
        payment_dates=pay_dates,
        underlying_identifiers=underlying_ids,
        underlying_reference_spots=ref_spots,
        spots_t0=spots_override if spots_override is not None else spots_map,
        ko_barriers=ko_levels,
        cpn_barriers=cpn_levels,
        fixed_coupon_amount=float(fixed_cpn_amt) if fixed_cpn_amt is not None else None,
        fixed_rate=float(fixed_rate) if fixed_rate is not None else None,
        coupon_amounts=cpn_amounts,
        day_count_convention=dcc,
        time_day_count_convention="Act365",
        is_memory=is_memory,
        callability_index=callability_index,
        coupon_index=coupon_index,
        option_index=option_index,
        fixings=fixings_map,
        ki_barrier=ki_barrier,
        ki_monitored_past=False,
        allow_autocall_on_expiry=allow_autocall_on_expiry,
        force_coupons_on_autocall=force_coupons_on_autocall,
        autocall_coupon=autocall_coupon,
        no_notional_repayment=no_notional_repayment,
        payment_currency=payment_ccy,
        df_pay_fn=df_fin_fn,
    )

    sorted_ids = seasoned.sorted_identifiers
    d_assets = len(sorted_ids)
    m_rem = len(seasoned.future_indices)
    device = seasoned.spots_t0.device

    # 2. Buehler Escrowed/Affine Cash Dividend Forward & Leverage Engine
    div_out = compute_escrowed_dividend_forwards(
        identifiers=sorted_ids,
        as_of_date=as_of_dt,
        observation_dates=seasoned.future_obs_dates,
        spots_t0=seasoned.spots_t0,
        ref_spots=seasoned.ref_spots,
        discount_curve=disc_curve_obj,
        repo_curves=repo_curves_map,
        market_dividends=dividends_map,
        predefined_dividends=predef_divs_map,
        dividend_amount_overrides=dividend_amount_overrides,
        payment_currency=payment_ccy,
        vol_surfaces=vol_surfaces_map,
        correlations=correlations_map,
    )

    # 3. Regularized Volatility Surfaces & Schedule Implied Volatilities
    reg_surfaces: dict[str, RegularizedVolSurface] = {}
    iv_at_fwd_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    d_iv_dk_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    node_surf_feats_np = np.zeros((d_assets, 7), dtype=np.float64)
    fwd_price_np = div_out.forward_price.detach().cpu().numpy().astype(np.float64)
    fwd_req_grad = bool(div_out.forward_price.requires_grad)

    for i, ticker in enumerate(sorted_ids):
        if ticker not in vol_surfaces_map:
            raise KeyError(f"Missing EquityVolatilitySmileCollection for underlying {ticker!r}.")
        surf_i = RegularizedVolSurface(vol_surfaces_map[ticker], as_of_date=as_of_dt, bc_type="clamped")
        reg_surfaces[ticker] = surf_i

        if m_rem > 0:
            for m in range(m_rem):
                t_m = float(seasoned.tau_obs[m])
                f_im = float(fwd_price_np[i, m])
                iv_at_fwd_np[i, m] = float(surf_i.iv(t_m, f_im))
                if fwd_req_grad:
                    eps_k = max(1e-5 * f_im, 1e-6)
                    iv_hi = float(surf_i.iv(t_m, f_im + eps_k))
                    iv_lo = float(surf_i.iv(t_m, max(f_im - eps_k, 1e-6)))
                    d_iv_dk_np[i, m] = (iv_hi - iv_lo) / ((f_im + eps_k) - max(f_im - eps_k, 1e-6))

            t_short = float(seasoned.tau_obs[0])
            f_short = float(fwd_price_np[i, 0])
            t_term = float(seasoned.tau_obs[-1])
            f_term = float(fwd_price_np[i, -1])
            node_surf_feats_np[i, :] = surf_i.extract_node_surface_features_values_only(
                t_short,
                f_short,
                t_term,
                f_term,
                iv_short_val=float(iv_at_fwd_np[i, 0]),
                iv_term_val=float(iv_at_fwd_np[i, -1]),
            )

    id_to_idx = {ticker: idx for idx, ticker in enumerate(sorted_ids)}

    def _build_surf_feat_jac(ticker: str) -> npt.NDArray[np.float64]:
        surf_i = reg_surfaces[ticker]
        if m_rem == 0:
            return np.zeros((7, surf_i.n_exp, surf_i.n_raw_strikes), dtype=np.float64)
        idx = id_to_idx[ticker]
        _, jac_7d = surf_i.extract_node_surface_features(
            float(seasoned.tau_obs[0]),
            float(fwd_price_np[idx, 0]),
            float(seasoned.tau_obs[-1]),
            float(fwd_price_np[idx, -1]),
        )
        return jac_7d

    def _build_iv_fwd_jac(ticker: str) -> npt.NDArray[np.float64]:
        surf_i = reg_surfaces[ticker]
        if m_rem == 0:
            return np.zeros((0, surf_i.n_exp, surf_i.n_raw_strikes), dtype=np.float64)
        idx = id_to_idx[ticker]
        return np.stack(
            [
                surf_i.iv_and_d_iv_d_quotes(float(seasoned.tau_obs[m]), float(fwd_price_np[idx, m]))[1]
                for m in range(m_rem)
            ],
            axis=0,
        )

    surf_feat_jacs: dict[str, npt.NDArray[np.float64]] = _LazySurfaceJacobianDict(_build_surf_feat_jac, sorted_ids)
    iv_fwd_jacs: dict[str, npt.NDArray[np.float64]] = _LazySurfaceJacobianDict(_build_iv_fwd_jac, sorted_ids)

    iv_at_fwd_t = torch.as_tensor(iv_at_fwd_np, dtype=torch.float64, device=device)
    if div_out.forward_price.requires_grad:
        d_iv_dk_t = torch.as_tensor(d_iv_dk_np, dtype=torch.float64, device=device)
        iv_at_fwd_t = iv_at_fwd_t + d_iv_dk_t * (
            div_out.forward_price - div_out.forward_price.detach()
        )
    tau_obs_t = torch.as_tensor(seasoned.tau_obs, dtype=torch.float64, device=device)
    delta_tau_t = torch.as_tensor(seasoned.delta_tau, dtype=torch.float64, device=device)
    tau_pay_t = torch.as_tensor(seasoned.tau_pay, dtype=torch.float64, device=device)

    if d_assets > 0:
        df_obs_t = div_out.df_fin_obs[0]
    else:
        df_obs_np = np.array([df_fin_fn(d) for d in seasoned.future_obs_dates], dtype=np.float64)
        df_obs_t = torch.as_tensor(df_obs_np, dtype=torch.float64, device=device)

    if df_pay_override is not None:
        df_pay_t = df_pay_override.to(dtype=torch.float64, device=device)
    else:
        df_batch_fn = getattr(df_fin_fn, "_eval_tau_batch", None)
        if df_batch_fn is not None and getattr(disc_curve_obj, "dcc", "Act365") == "Act365":
            df_pay_np = df_batch_fn(np.asarray(seasoned.tau_pay, dtype=np.float64))
        else:
            df_pay_np = np.array([df_fin_fn(d) for d in seasoned.future_pay_dates], dtype=np.float64)
        df_pay_t = torch.as_tensor(df_pay_np, dtype=torch.float64, device=device)

    v_perf_t = iv_at_fwd_t * torch.sqrt(tau_obs_t.clamp(min=1e-12)).unsqueeze(0)

    if corr_override is not None:
        corr_t = corr_override.to(dtype=torch.float64, device=device)
    else:
        corr_t = _build_correlation_matrix(sorted_ids, correlations_map, device)

    assert seasoned.sentinel_barriers is not None
    sb = seasoned.sentinel_barriers
    ko_barriers_t = torch.as_tensor(sb.ko_barriers_raw, dtype=torch.float64, device=device)
    is_callable_t = torch.as_tensor(sb.is_callable, dtype=torch.float64, device=device)
    cpn_barriers_t = torch.as_tensor(sb.cpn_barriers_raw, dtype=torch.float64, device=device)
    is_guaranteed_t = torch.as_tensor(sb.is_guaranteed_cpn, dtype=torch.float64, device=device)
    coupon_amounts_t = torch.as_tensor(seasoned.future_coupon_rates, dtype=torch.float64, device=device)

    # 4. Analytical Stulz-Margrabe Moment-Matched Basket Proxy
    proxy_inputs = {
        "forward_perf": div_out.forward_perf,
        "v_perf": v_perf_t,
        "corr_matrix": corr_t,
        "df_pay": df_pay_t,
        "ko_barriers": ko_barriers_t,
        "is_callable": is_callable_t,
        "cpn_barriers": cpn_barriers_t,
        "is_guaranteed_cpn": is_guaranteed_t,
        "coupon_amounts": coupon_amounts_t,
        "strike": strike,
        "option_gearing": option_gearing,
        "lower_strike": lower_strike,
        "ki_barrier": ki_barrier,
        "i_ki_past": seasoned.i_ki_past,
        "c_mem_0": seasoned.c_mem_0,
        "is_memory": is_memory,
        "ko_spread": ko_spread,
        "cpn_spread": cpn_spread,
        "ki_spread": ki_spread,
        "no_notional_repayment": no_notional_repayment,
        "autocall_coupon": autocall_coupon,
        "force_coupons_on_autocall": force_coupons_on_autocall,
        "coupon_barrier_type": cpn_barrier_type,
        "coupon_barrier_direction": cpn_barrier_direction,
        "is_long_basket": is_long_basket,
        "pv_in_transit": seasoned.pv_in_transit,
        "already_autocalled": seasoned.already_autocalled,
        "callability_index": callability_index,
        "participation_call_strike": part_strike,
        "participation_call_rate": part_rate,
        "one_star_strike": one_star_strike,
        "one_star_coupon": one_star_coupon,
    }
    baseline = compute_analytical_autocall_baseline(proxy_inputs)

    # 5. Construct Fixed-Padding Feature Tensors in torch.float64 (preserving autograd graph!)
    d_pad = max(d_max, d_assets)
    m_pad = max(m_max, m_rem)

    # 5a. Node Features: (1, d_pad, 14)
    log_m0 = torch.log(seasoned.moneyness_t0.clamp(min=1e-12))  # (D,)
    if m_rem > 0:
        tau_term = tau_obs_t[-1].clamp(min=1e-6)
        log_fwd_carry_1y = torch.log(div_out.forward_rel[:, -1].clamp(min=1e-12)) / tau_term
        log_div_lev = torch.log(div_out.cash_div_leverage[:, -1].clamp(min=1.0))
        f_p_term = div_out.forward_perf[:, -1].clamp(min=1e-12)
        v_term = v_perf_t[:, -1].clamp(min=1e-6)
        k_eff = max(strike, 1e-4)
        dist_to_put = torch.log(f_p_term / k_eff) / v_term
    else:
        log_fwd_carry_1y = torch.zeros(d_assets, dtype=torch.float64, device=device)
        log_div_lev = torch.zeros(d_assets, dtype=torch.float64, device=device)
        dist_to_put = torch.zeros(d_assets, dtype=torch.float64, device=device)

    surf_feats_t = torch.as_tensor(node_surf_feats_np, dtype=torch.float64, device=device)  # (D, 7)
    active_ones = torch.ones(d_assets, dtype=torch.float64, device=device)

    node_active = torch.stack(
        [
            log_m0,
            log_fwd_carry_1y,
            log_div_lev,
            div_out.repo_spread_avg,
            surf_feats_t[:, 0],
            surf_feats_t[:, 1],
            surf_feats_t[:, 2],
            surf_feats_t[:, 3],
            surf_feats_t[:, 4],
            surf_feats_t[:, 5],
            surf_feats_t[:, 6],
            dist_to_put,
            baseline.proxy_delta_weights,
            active_ones,
        ],
        dim=1,
    )  # (D, 14)

    if d_pad > d_assets:
        node_pad = torch.zeros((d_pad - d_assets, 14), dtype=torch.float64, device=device)
        node_feats = torch.cat([node_active, node_pad], dim=0).unsqueeze(0)
        asset_mask = torch.cat(
            [active_ones, torch.zeros(d_pad - d_assets, dtype=torch.float64, device=device)],
            dim=0,
        ).unsqueeze(0)
    else:
        node_feats = node_active.unsqueeze(0)
        asset_mask = active_ones.unsqueeze(0)

    # 5b. Edge Features: (1, d_pad, d_pad, 4)
    if m_rem > 0:
        v_term_vec = v_perf_t[:, -1]
        log_fp_term = torch.log(div_out.forward_perf[:, -1].clamp(min=1e-12))
    else:
        v_term_vec = torch.zeros(d_assets, dtype=torch.float64, device=device)
        log_fp_term = log_m0

    cov_active = corr_t * v_term_vec.unsqueeze(1) * v_term_vec.unsqueeze(0)
    log_perf_diff_active = torch.abs(log_fp_term.unsqueeze(1) - log_fp_term.unsqueeze(0))
    off_diag_active = 1.0 - torch.eye(d_assets, dtype=torch.float64, device=device)

    edge_active = torch.stack(
        [corr_t, cov_active, log_perf_diff_active, off_diag_active],
        dim=-1,
    )  # (D, D, 4)

    if d_pad > d_assets:
        edge_full = torch.zeros((d_pad, d_pad, 4), dtype=torch.float64, device=device)
        edge_full[:d_assets, :d_assets, :] = edge_active
        edge_feats = edge_full.unsqueeze(0)
    else:
        edge_feats = edge_active.unsqueeze(0)

    # 5c. Schedule Features: (1, m_pad, 12)
    if m_rem > 0:
        log_ko_feat_t = torch.as_tensor(sb.log_ko_barrier_feat, dtype=torch.float64, device=device)
        log_cpn_feat_t = torch.as_tensor(sb.log_cpn_barrier_feat, dtype=torch.float64, device=device)
        log_fw_t = torch.log(baseline.f_w.clamp(min=1e-12))
        sched_active = torch.stack(
            [
                tau_obs_t,
                delta_tau_t,
                df_pay_t,
                log_ko_feat_t,
                is_callable_t,
                log_cpn_feat_t,
                is_guaranteed_t,
                coupon_amounts_t,
                log_fw_t,
                baseline.v_w,
                baseline.p_proxy,
                baseline.q_proxy,
            ],
            dim=1,
        )  # (M_rem, 12)
        sched_ones = torch.ones(m_rem, dtype=torch.float64, device=device)
        if m_pad > m_rem:
            sched_pad = torch.zeros((m_pad - m_rem, 12), dtype=torch.float64, device=device)
            sched_feats = torch.cat([sched_active, sched_pad], dim=0).unsqueeze(0)
            sched_mask = torch.cat(
                [sched_ones, torch.zeros(m_pad - m_rem, dtype=torch.float64, device=device)],
                dim=0,
            ).unsqueeze(0)
        else:
            sched_feats = sched_active.unsqueeze(0)
            sched_mask = sched_ones.unsqueeze(0)
    else:
        sched_feats = torch.zeros((1, m_pad, 12), dtype=torch.float64, device=device)
        sched_mask = torch.zeros((1, m_pad), dtype=torch.float64, device=device)

    # 5d. Global Context Vector: (1, 10)
    has_ki = 1.0 if (ki_barrier is not None and ki_barrier > 0.0) else 0.0
    b_ki_val = float(ki_barrier) if (ki_barrier is not None and ki_barrier > 0.0) else float(strike)
    global_vec = torch.stack(
        [
            torch.tensor(float(strike), dtype=torch.float64, device=device),
            torch.tensor(b_ki_val, dtype=torch.float64, device=device),
            torch.tensor(has_ki, dtype=torch.float64, device=device),
            torch.tensor(float(seasoned.i_ki_past), dtype=torch.float64, device=device),
            torch.tensor(float(option_gearing), dtype=torch.float64, device=device),
            torch.tensor(float(seasoned.c_mem_0), dtype=torch.float64, device=device),
            torch.tensor(1.0 if is_memory else 0.0, dtype=torch.float64, device=device),
            torch.tensor(abs(float(ko_spread)), dtype=torch.float64, device=device),
            torch.tensor(abs(float(cpn_spread)), dtype=torch.float64, device=device),
            baseline.l_put_proxy,
        ],
        dim=0,
    ).unsqueeze(0)

    state_out = AutocallPreprocState(
        node_feats=node_feats,
        edge_feats=edge_feats,
        asset_mask=asset_mask,
        sched_feats=sched_feats,
        sched_mask=sched_mask,
        global_feats=global_vec,
        d_assets=d_assets,
        m_rem=m_rem,
        d_max=d_pad,
        m_max=m_pad,
        original_identifiers=seasoned.original_identifiers,
        sorted_identifiers=sorted_ids,
        lex_perm=seasoned.lex_perm,
        inv_lex_perm=seasoned.inv_lex_perm,
        spots_t0=seasoned.spots_t0,
        ref_spots=seasoned.ref_spots,
        moneyness_t0=seasoned.moneyness_t0,
        corr_matrix=corr_t,
        tau_obs=tau_obs_t,
        delta_tau=delta_tau_t,
        tau_pay=tau_pay_t,
        df_obs=df_obs_t,
        df_pay=df_pay_t,
        ko_barriers=ko_barriers_t,
        is_callable=is_callable_t,
        cpn_barriers=cpn_barriers_t,
        is_guaranteed_cpn=is_guaranteed_t,
        coupon_amounts=coupon_amounts_t,
        forward_rel=div_out.forward_rel,
        forward_perf=div_out.forward_perf,
        cash_div_leverage=div_out.cash_div_leverage,
        iv_at_fwd=iv_at_fwd_t,
        v_perf=v_perf_t,
        seasoned_state=seasoned,
        dividend_schedule=div_out,
        vol_surfaces=reg_surfaces,
        surface_feature_jacobians=surf_feat_jacs,
        iv_at_fwd_jacobians=iv_fwd_jacs,
        discount_curve=disc_curve_obj,
        baseline=baseline,
        strike=strike,
        option_gearing=option_gearing,
        lower_strike=lower_strike,
        ki_barrier=ki_barrier,
        i_ki_past=seasoned.i_ki_past,
        c_mem_0=seasoned.c_mem_0,
        is_memory=is_memory,
        ko_spread=ko_spread,
        cpn_spread=cpn_spread,
        ki_spread=ki_spread,
        no_notional_repayment=no_notional_repayment,
        autocall_coupon=autocall_coupon,
        force_coupons_on_autocall=force_coupons_on_autocall,
        allow_autocall_on_expiry=allow_autocall_on_expiry,
        participation_call_strike=part_strike,
        participation_call_rate=part_rate,
        one_star_strike=one_star_strike,
        one_star_coupon=one_star_coupon,
        is_long_basket=is_long_basket,
        payment_currency=payment_ccy,
        callability_index=callability_index,
        coupon_index=coupon_index,
        option_index=option_index,
        funding_leg=funding_leg_dict,
        pv_in_transit=seasoned.pv_in_transit,
        already_autocalled=seasoned.already_autocalled,
        trade_info=str(trade_info) if trade_info is not None else None,
        queries=queries,
        coupon_barrier_type=cpn_barrier_type,
        coupon_barrier_direction=cpn_barrier_direction,
    )
    if cache_key is not None:
        if len(_PREPROC_STATE_CACHE) > 256:
            _PREPROC_STATE_CACHE.clear()
        setattr(state_out, "_cache_origin", state_out)
        _PREPROC_STATE_CACHE[cache_key] = state_out
        return copy.copy(state_out)
    return state_out


def assemble_autocall_cashflows_and_pv(
    preproc_state: AutocallPreprocState,
    *,
    p_stop: torch.Tensor | None = None,
    q_cpn: torch.Tensor | None = None,
    l_put: torch.Tensor | None = None,
    l_call: torch.Tensor | None = None,
    l_onestar: torch.Tensor | float | None = None,
    h_prob: torch.Tensor | None = None,
    q_prob: torch.Tensor | None = None,
    option_se: float = 0.0,
    total_se: float | None = None,
    df_fin_fn_override: Callable[[datetime.datetime], torch.Tensor | float] | None = None,
    build_cashflow_items: bool = True,
) -> AutocallPostprocOutput:
    """Executes the W1 affine stopping-time cash-flow and ProductValue post-processor in torch.float64.

    If `p_stop`, `q_cpn`, or `l_put` are omitted, defaults to the analytical Stulz-Margrabe baseline
    predictions in `preproc_state.baseline`.
    """
    m_rem = preproc_state.m_rem
    device = preproc_state.spots_t0.device
    ccy = preproc_state.payment_currency

    if p_stop is None and h_prob is not None:
        h_t = (
            h_prob.to(dtype=torch.float64, device=device).reshape(-1)[:m_rem]
            * preproc_state.is_callable
        )
        one_minus_h = 1.0 - h_t.clamp(0.0, 1.0)
        surv_h = torch.cumprod(one_minus_h, dim=0)
        surv_prev_h = torch.cat(
            [torch.ones(1, dtype=torch.float64, device=device), surv_h[:-1]],
            dim=0,
        )
        p_stop = surv_prev_h * h_t
    if q_cpn is None and q_prob is not None:
        q_cpn = q_prob

    p_t = (
        preproc_state.baseline.p_proxy
        if p_stop is None
        else p_stop.to(dtype=torch.float64, device=device).reshape(-1)[:m_rem]
    )
    q_t = (
        preproc_state.baseline.q_proxy
        if q_cpn is None
        else q_cpn.to(dtype=torch.float64, device=device).reshape(-1)[:m_rem]
    )
    l_put_t = (
        preproc_state.baseline.l_put_proxy
        if l_put is None
        else l_put.to(dtype=torch.float64, device=device).reshape(())
    )
    l_call_t = (
        getattr(preproc_state.baseline, "l_call_proxy", torch.tensor(0.0, dtype=torch.float64, device=device))
        if l_call is None
        else l_call.to(dtype=torch.float64, device=device).reshape(())
    )
    l_onestar_t = (
        getattr(preproc_state.baseline, "l_onestar_proxy", torch.tensor(0.0, dtype=torch.float64, device=device))
        if l_onestar is None
        else torch.as_tensor(l_onestar, dtype=torch.float64, device=device).reshape(())
    )

    # Enforce callable mask on stopping probabilities
    p_t = p_t * preproc_state.is_callable

    cashflows: list[CashFlowItem] = (
        list(preproc_state.seasoned_state.past_cashflows) if build_cashflow_items else []
    )
    pv_in_transit_t = torch.tensor(preproc_state.pv_in_transit, dtype=torch.float64, device=device)

    if m_rem == 0 or preproc_state.already_autocalled:
        z_s = torch.tensor(0.0, dtype=torch.float64, device=device)
        if build_cashflow_items:
            for m in range(m_rem):
                pay_dt = preproc_state.seasoned_state.future_pay_dates[m]
                cashflows.append(
                    CashFlowItem("FundingLeg", pay_dt, 0.0, 0.0, ccy, 0, "Estimate")
                )
                cashflows.append(
                    CashFlowItem("Coupon", pay_dt, 0.0, 0.0, ccy, 0, "Estimate")
                )
                if m == m_rem - 1:
                    cashflows.append(
                        CashFlowItem("Option", pay_dt, 0.0, 0.0, ccy, 0, "Estimate")
                    )
                cashflows.append(
                    CashFlowItem("Notional", pay_dt, 0.0, 0.0, ccy, 0, "Estimate")
                )
        return AutocallPostprocOutput(
            product_value=pv_in_transit_t,
            notional_pv=z_s,
            coupon_pv=z_s,
            option_pv=z_s,
            funding_pv=z_s,
            pv_in_transit=pv_in_transit_t,
            survival_probs=torch.zeros(m_rem, dtype=torch.float64, device=device),
            p_stop=p_t,
            q_cpn=q_t,
            l_put=l_put_t,
            l_call=l_call_t,
            cashflows=cashflows,
            standard_error=0.0,
            reporting_currency=ccy,
            cash_to_settle_today=0.0,
            l_onestar=l_onestar_t,
        )

    # Survival probabilities S_m = 1 - sum_{k=1}^m p_k
    cum_p = torch.cumsum(p_t, dim=0)
    survival_probs = (1.0 - cum_p).clamp(min=0.0)
    surv_prev = torch.cat(
        [torch.ones(1, dtype=torch.float64, device=device), survival_probs[:-1]],
        dim=0,
    )

    df_pay = preproc_state.df_pay
    cpn_rates = preproc_state.coupon_amounts

    # Floating / Fixed-Spread FundingLeg schedule per period if present
    funding_cf_per_period: list[torch.Tensor] = [
        torch.tensor(0.0, dtype=torch.float64, device=device) for _ in range(m_rem)
    ]
    if preproc_state.funding_leg is not None:
        f_leg = preproc_state.funding_leg
        f_rid = f_leg.get("rate_identifier")
        f_curve = f_leg.get("funding_curve")
        if df_fin_fn_override is None:
            cache_key_w = (preproc_state.seasoned_state.as_of_date, m_rem)
            cached_w_pair = f_leg.get("_cached_w_fund_per_period")
            if cached_w_pair is not None and cached_w_pair[0] == cache_key_w:
                w_fund_list: list[float] = cached_w_pair[1]
            else:
                cached_fin = f_leg.get("_cached_df_fin_fn")
                if cached_fin is None:
                    _, cached_fin = build_discount_rate_evaluator(
                        preproc_state.discount_curve, preproc_state.seasoned_state.as_of_date
                    )
                    f_leg["_cached_df_fin_fn"] = cached_fin
                df_fin_fn = cached_fin
                if f_rid is not None and f_curve is not None and f_curve is not preproc_state.discount_curve:
                    cached_fwd = f_leg.get("_cached_df_fwd_fn")
                    if cached_fwd is None:
                        _, cached_fwd = build_discount_rate_evaluator(
                            f_curve, preproc_state.seasoned_state.as_of_date
                        )
                        f_leg["_cached_df_fwd_fn"] = cached_fwd
                    df_fwd_fn = cached_fwd
                else:
                    df_fwd_fn = df_fin_fn
                f_spread = float(f_leg.get("funding_spread", 0.0))
                f_dates = f_leg.get("payment_dates", [])
                w_fund_list = [0.0] * m_rem
                prev_f_dt = preproc_state.seasoned_state.as_of_date
                for f_dt in f_dates:
                    if f_dt <= preproc_state.seasoned_state.as_of_date:
                        prev_f_dt = f_dt
                        continue
                    alpha_j = _year_fraction(
                        prev_f_dt, f_dt, preproc_state.seasoned_state.day_count_convention
                    )
                    df_curr: Any = float(df_fin_fn(f_dt))
                    if f_rid is None:
                        r_fwd_j_val = 0.0
                    else:
                        df_prev_fwd: Any = float(df_fwd_fn(prev_f_dt))
                        df_curr_fwd: Any = float(df_fwd_fn(f_dt))
                        r_fwd_j_val = (df_prev_fwd / max(df_curr_fwd, 1e-12) - 1.0) / max(alpha_j, 1e-8)
                    m_idx = m_rem - 1
                    for m in range(m_rem):
                        if preproc_state.seasoned_state.future_pay_dates[m] >= f_dt:
                            m_idx = m
                            break
                    w_fund_list[m_idx] += df_curr * (r_fwd_j_val + f_spread) * alpha_j
                    prev_f_dt = f_dt
                f_leg["_cached_w_fund_per_period"] = (cache_key_w, w_fund_list)
            for m_idx, w_val in enumerate(w_fund_list):
                if w_val != 0.0:
                    funding_cf_per_period[m_idx] = surv_prev[m_idx] * w_val
        else:
            df_fin_fn = df_fin_fn_override
            if f_rid is not None and f_curve is not None and f_curve is not preproc_state.discount_curve:
                cached_fwd = f_leg.get("_cached_df_fwd_fn")
                if cached_fwd is None:
                    _, cached_fwd = build_discount_rate_evaluator(
                        f_curve, preproc_state.seasoned_state.as_of_date
                    )
                    f_leg["_cached_df_fwd_fn"] = cached_fwd
                df_fwd_fn = cached_fwd
            else:
                df_fwd_fn = df_fin_fn
            f_spread = float(f_leg.get("funding_spread", 0.0))
            f_dates = f_leg.get("payment_dates", [])
            prev_f_dt = preproc_state.seasoned_state.as_of_date
            for f_dt in f_dates:
                if f_dt <= preproc_state.seasoned_state.as_of_date:
                    prev_f_dt = f_dt
                    continue
                alpha_j = _year_fraction(prev_f_dt, f_dt, preproc_state.seasoned_state.day_count_convention)
                df_curr = df_fin_fn(f_dt)
                if f_rid is None:
                    r_fwd_j: Any = 0.0
                else:
                    df_prev_fwd = df_fwd_fn(prev_f_dt)
                    df_curr_fwd = df_fwd_fn(f_dt)
                    if isinstance(df_curr_fwd, torch.Tensor):
                        r_fwd_j = (df_prev_fwd / df_curr_fwd.clamp(min=1e-12) - 1.0) / max(alpha_j, 1e-8)
                    else:
                        r_fwd_j = (df_prev_fwd / max(float(df_curr_fwd), 1e-12) - 1.0) / max(alpha_j, 1e-8)
                # Map to nearest active period m
                m_idx = m_rem - 1
                for m in range(m_rem):
                    if preproc_state.seasoned_state.future_pay_dates[m] >= f_dt:
                        m_idx = m
                        break
                cf_fund_j = surv_prev[m_idx] * df_curr * (r_fwd_j + f_spread) * alpha_j
                funding_cf_per_period[m_idx] = funding_cf_per_period[m_idx] + cf_fund_j
                prev_f_dt = f_dt

    notional_cf_list: list[torch.Tensor] = []
    coupon_cf_list: list[torch.Tensor] = []
    mem_acc = torch.tensor(preproc_state.c_mem_0, dtype=torch.float64, device=device)

    for m in range(m_rem):
        # Notional Redemption Row at T_m (§1.2.E.1)
        if preproc_state.no_notional_repayment:
            cf_notional_m = torch.tensor(0.0, dtype=torch.float64, device=device)
        elif m < m_rem - 1:
            cf_notional_m = df_pay[m] * p_t[m]
        else:
            cf_notional_m = df_pay[m] * (p_t[m] + survival_probs[m])
        notional_cf_list.append(cf_notional_m)

        # Coupon Row at T_m (§1.2.E.1)
        c_m = cpn_rates[m]
        eff_c_m = (c_m + mem_acc) if preproc_state.is_memory else c_m
        if preproc_state.force_coupons_on_autocall:
            eff_q_m = torch.maximum(q_t[m], p_t[m] / surv_prev[m].clamp(min=1e-12))
        else:
            eff_q_m = q_t[m]
        cf_cpn_m = df_pay[m] * (
            surv_prev[m] * eff_q_m * eff_c_m + p_t[m] * float(preproc_state.autocall_coupon)
        )
        coupon_cf_list.append(cf_cpn_m)

        if preproc_state.is_memory:
            mem_acc = eff_c_m * (1.0 - eff_q_m)

    # Terminal Option Row at T_{M_rem} (§1.2.E.1)
    put_sign = 1.0 if preproc_state.is_long_basket else -1.0
    cf_option_term = df_pay[-1] * (
        put_sign * float(preproc_state.option_gearing) * l_put_t
        + float(preproc_state.participation_call_rate) * l_call_t
        + l_onestar_t
    )

    notional_pv = torch.stack(notional_cf_list).sum()
    coupon_pv = torch.stack(coupon_cf_list).sum()
    funding_pv = torch.stack(funding_cf_per_period).sum()
    option_pv = cf_option_term

    product_value = pv_in_transit_t + notional_pv + coupon_pv + option_pv - funding_pv

    # Emit CashFlowItem rows for all future periods m = 0 .. M_rem - 1
    if build_cashflow_items:
        for m in range(m_rem):
            pay_dt = preproc_state.seasoned_state.future_pay_dates[m]
            fund_amt = float(funding_cf_per_period[m].detach().cpu().item())
            cpn_amt = float(coupon_cf_list[m].detach().cpu().item())
            notional_amt = float(notional_cf_list[m].detach().cpu().item())

            cashflows.append(
                CashFlowItem(
                    Description="FundingLeg",
                    Date=pay_dt,
                    Amount=-fund_amt if fund_amt != 0.0 else 0.0,
                    StandardError=0.0,
                    Currency=ccy,
                    Leg=0,
                    StatusText="Estimate",
                )
            )
            cashflows.append(
                CashFlowItem(
                    Description="Coupon",
                    Date=pay_dt,
                    Amount=cpn_amt,
                    StandardError=0.0,
                    Currency=ccy,
                    Leg=0,
                    StatusText="Estimate",
                )
            )
            if m == m_rem - 1:
                opt_amt = float(cf_option_term.detach().cpu().item())
                cashflows.append(
                    CashFlowItem(
                        Description="Option",
                        Date=pay_dt,
                        Amount=opt_amt,
                        StandardError=float(option_se),
                        Currency=ccy,
                        Leg=0,
                        StatusText="Estimate",
                    )
                )
            cashflows.append(
                CashFlowItem(
                    Description="Notional",
                    Date=pay_dt,
                    Amount=notional_amt,
                    StandardError=0.0,
                    Currency=ccy,
                    Leg=0,
                    StatusText="Estimate",
                )
            )

    eff_total_se = float(total_se) if total_se is not None else float(option_se)
    return AutocallPostprocOutput(
        product_value=product_value,
        notional_pv=notional_pv,
        coupon_pv=coupon_pv,
        option_pv=option_pv,
        funding_pv=funding_pv,
        pv_in_transit=pv_in_transit_t,
        survival_probs=survival_probs,
        p_stop=p_t,
        q_cpn=q_t,
        l_put=l_put_t,
        l_call=l_call_t,
        cashflows=cashflows,
        standard_error=eff_total_se,
        reporting_currency=ccy,
        cash_to_settle_today=0.0,
        l_onestar=l_onestar_t,
    )


__all__ = [
    "D_MAX_DEFAULT",
    "M_MAX_DEFAULT",
    "AutocallNodeCol",
    "AutocallEdgeCol",
    "AutocallSchedCol",
    "AutocallGlobalCol",
    "AutocallPreprocState",
    "AutocallPostprocOutput",
    "parse_murex_text_or_dict_fallback",
    "preprocess_eq_autocall",
    "assemble_autocall_cashflows_and_pv",
]
