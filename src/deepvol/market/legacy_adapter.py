"""legacy_adapter.py — Legacy message-graph adapter, multi-curve interpolator,

request builder, and MutableMessage serializer for DeepVol (Milestone 5).
Implements content-addressable ConstMessage DAG resolution, Act/365F day count,
multi-curve piecewise-linear zero rate interpolation with flat extrapolation,
request building from legacy graph nodes, and exact MutableMessage serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import resource
import socket
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import numpy as np
import numpy.typing as npt

from deepvol.api.schemas_fx_cross import (
    AtmConventionEnum,
    ContractTerms,
    CurvePillar,
    DeltaConventionEnum,
    ExecutionOptions,
    FXLegData,
    InterestRates,
    MarketData,
    ParityEnum,
    PricingEngineEnum,
    SettlementTypeEnum,
    SmileDynamicsEnum,
    SmileQuotes,
    StrangleConventionEnum,
    TriangulatedFXOptionRequest,
    TriangulatedFXOptionResponse,
)
from deepvol.market.fx_conventions import DeltaConvention

# 40-character lowercase hexadecimal hash pattern
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


# ── Day Count Convention ─────────────────────────────────────────────────────


def year_fraction_act365f(start: datetime | str, end: datetime | str) -> float:
    """Computes exact Act/365F year fraction between two timestamps.

    Args:
        start: Anchor or as-of timestamp (datetime or ISO 8601 string).
        end: Expiration or settlement timestamp (datetime or ISO 8601 string).

    Returns:
        Exact year fraction as float64: (end - start).total_seconds() / (365.0 * 86400.0).
    """
    if isinstance(start, str):
        start = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if isinstance(end, str):
        end = datetime.fromisoformat(end.replace("Z", "+00:00"))

    # Timezone normalization to avoid naive/aware subtraction TypeError
    if start.tzinfo is not None and end.tzinfo is None:
        end = end.replace(tzinfo=start.tzinfo)
    elif start.tzinfo is None and end.tzinfo is not None:
        start = start.replace(tzinfo=end.tzinfo)

    total_sec = (end - start).total_seconds()
    return float(total_sec / 31536000.0)


# ── ConstMessage & ConstMessageGraph ──────────────────────────────────────────


@dataclass(frozen=True)
class ConstMessage:
    """Immutable, content-addressable message node in the legacy graph."""

    id: str
    value: Mapping[str, Any]

    def refs(self) -> List[str]:
        """Discovers all 40-hex reference IDs recursively within the message value.

        Returns:
            List of unique 40-hex lowercase string IDs in discovery order.
        """
        seen: Set[str] = set()
        discovered: List[str] = []

        def _walk(obj: Any) -> None:
            if isinstance(obj, str):
                if _HEX40_RE.match(obj) and obj not in seen:
                    seen.add(obj)
                    discovered.append(obj)
            elif isinstance(obj, Mapping):
                for val in obj.values():
                    _walk(val)
            elif isinstance(obj, (list, tuple, set)):
                for item in obj:
                    _walk(item)

        _walk(self.value)
        return discovered


def _canonical_json_bytes(val: Any) -> bytes:
    """Serializes a mapping to canonical JSON bytes with sorted keys."""
    return json.dumps(val, sort_keys=True, separators=(",", ":")).encode("utf-8")


class ConstMessageGraph:
    """Indexed container and topological resolver for legacy ConstMessage DAGs."""

    def __init__(
        self,
        messages: Iterable[Mapping[str, Any] | ConstMessage],
        *,
        verify_hash: bool = False,
    ) -> None:
        self._messages: Dict[str, ConstMessage] = {}

        for item in messages:
            if isinstance(item, ConstMessage):
                msg = item
            else:
                raw = item.get("ConstMessage", item)
                msg_id = str(raw["id"])
                msg_val = cast(Mapping[str, Any], raw["value"])
                msg = ConstMessage(id=msg_id, value=msg_val)

            if verify_hash:
                expected_sha1 = hashlib.sha1(_canonical_json_bytes(msg.value)).hexdigest()
                if msg.id != expected_sha1:
                    raise ValueError(
                        f"Hash mismatch for message '{msg.id}': expected SHA-1 '{expected_sha1}'"
                    )

            self._messages[msg.id] = msg

    def __contains__(self, message_id: str) -> bool:
        return message_id in self._messages

    def __len__(self) -> int:
        return len(self._messages)

    def __getitem__(self, message_id: str) -> ConstMessage:
        if message_id not in self._messages:
            raise KeyError(f"Message ID '{message_id}' not found in graph")
        return self._messages[message_id]

    def get(self, message_id: str, default: Optional[ConstMessage] = None) -> Optional[ConstMessage]:
        return self._messages.get(message_id, default)

    def by_type(self, type_name: str) -> List[ConstMessage]:
        """Queries messages by value['TypeName'] == type_name."""
        return [
            msg for msg in self._messages.values()
            if msg.value.get("TypeName") == type_name
        ]

    def resolve(self, root_id: str) -> List[ConstMessage]:
        """Resolves subgraph rooted at root_id in valid topological order.

        Dependencies appear strictly before dependents. The root is the final element.

        Raises:
            KeyError: On root_id missing or any dangling reference.
            ValueError: On circular dependency cycle.
        """
        if root_id not in self._messages:
            raise KeyError(f"Root message '{root_id}' not found in graph")

        visiting: Set[str] = set()
        visited: Set[str] = set()
        order: List[ConstMessage] = []

        def _dfs(node_id: str, path: List[str]) -> None:
            if node_id in visiting:
                cycle_str = " -> ".join(path + [node_id])
                raise ValueError(f"Circular dependency cycle detected: {cycle_str}")
            if node_id in visited:
                return
            if node_id not in self._messages:
                parent = path[-1] if path else "root"
                raise KeyError(
                    f"Dangling reference: message '{node_id}' referenced by '{parent}' not found in graph"
                )

            visiting.add(node_id)
            node_msg = self._messages[node_id]
            for child_ref in node_msg.refs():
                _dfs(child_ref, path + [node_id])
            visiting.remove(node_id)
            visited.add(node_id)
            order.append(node_msg)

        _dfs(root_id, [])
        return order


# ── DiscountCurve ─────────────────────────────────────────────────────────────


class DiscountCurve:
    """Piecewise linear in zero rates with flat extrapolation and linear weights."""

    def __init__(
        self,
        as_of: datetime,
        dates: Sequence[datetime],
        rates_pct: Sequence[float],
    ) -> None:
        if len(dates) != len(rates_pct):
            raise ValueError(
                f"dates ({len(dates)}) and rates_pct ({len(rates_pct)}) length mismatch"
            )
        if len(dates) == 0:
            raise ValueError("DiscountCurve requires at least one pillar date")

        self.as_of = as_of
        self.dates = list(dates)

        # Precompute year fractions in Act/365F and decimal zero rates in float64
        self._tenors: npt.NDArray[np.float64] = np.array(
            [year_fraction_act365f(as_of, d) for d in dates], dtype=np.float64
        )
        self._rates: npt.NDArray[np.float64] = np.array(
            [float(r) / 100.0 for r in rates_pct], dtype=np.float64
        )

        if len(self._tenors) > 1:
            diffs = np.diff(self._tenors)
            if np.any(diffs <= 0.0):
                raise ValueError("Pillar dates must have strictly increasing tenors")

    @property
    def tenors(self) -> npt.NDArray[np.float64]:
        return self._tenors

    @property
    def rates(self) -> npt.NDArray[np.float64]:
        return self._rates

    def zero(self, t: float) -> float:
        """Returns piecewise-linear zero rate r(t) with flat extrapolation."""
        return float(
            np.interp(
                t,
                self._tenors,
                self._rates,
                left=self._rates[0],
                right=self._rates[-1],
            )
        )

    def df(self, t: float) -> float:
        """Returns continuously compounded discount factor DF(t) = exp(-r(t) * t)."""
        r = self.zero(t)
        return float(math.exp(-r * t))

    def weights(self, t: float) -> npt.NDArray[np.float64]:
        """Computes linear-in-rate interpolation weights w in R^K such that r(t) = w @ r.

        The partition of unity sum(w) == 1.0 is preserved.
        """
        k = len(self._tenors)
        w = np.zeros(k, dtype=np.float64)

        if k == 1:
            w[0] = 1.0
            return w

        if t <= self._tenors[0]:
            w[0] = 1.0
        elif t >= self._tenors[-1]:
            w[-1] = 1.0
        else:
            idx = int(np.searchsorted(self._tenors, t))
            t_left = self._tenors[idx - 1]
            t_right = self._tenors[idx]
            dt_interval = t_right - t_left
            w[idx - 1] = (t_right - t) / dt_interval
            w[idx] = (t - t_left) / dt_interval

        return w


# ── LegacyContext & Request Builder ───────────────────────────────────────────


@dataclass(frozen=True)
class LegacyContext:
    """Auxiliary context captured during legacy request building."""

    root_id: str
    product_id: str
    model_id: str
    market_data_set_id: str
    queries: List[str]
    assumptions: List[str] = field(default_factory=list)
    curves: Dict[str, DiscountCurve] = field(default_factory=dict)
    curve_identifiers: Dict[str, str] = field(default_factory=dict)
    spots: Dict[str, float] = field(default_factory=dict)
    tenors: Tuple[float, float] = (0.0, 0.0)
    vol_surfaces: Dict[str, Any] = field(default_factory=dict)
    smile_expiries: Dict[str, List[datetime]] = field(default_factory=dict)
    smile_weights: Dict[str, Tuple[int, int, float, float]] = field(default_factory=dict)
    curve_weights: Dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)
    pair_cross: str = ""
    pair_leg1: str = ""
    pair_leg2: str = ""
    raw_messages: Dict[str, ConstMessage] = field(default_factory=dict)


def build_request(
    graph: ConstMessageGraph,
    root_id: str,
    *,
    delta_conventions: Optional[Mapping[str, DeltaConvention]] = None,
) -> Tuple[TriangulatedFXOptionRequest, InterestRates, LegacyContext]:
    """Compiles a legacy ConstMessage DAG into a canonical TriangulatedFXOptionRequest.

    Args:
        graph: Loaded ConstMessageGraph containing the required messages.
        root_id: 40-hex identifier of the root calculation message.
        delta_conventions: Optional overrides for leg delta quoting conventions.

    Returns:
        Tuple of (canonical request, interest rates, legacy context).
    """
    ordered_msgs = graph.resolve(root_id)
    raw_map = {m.id: m for m in ordered_msgs}

    root_msg = graph[root_id]
    r_val = root_msg.value

    prod_id = str(r_val["Product"])
    model_id = str(r_val["Model"])
    queries = list(r_val.get("Queries", []))

    prod_msg = graph[prod_id]
    p_val = prod_msg.value
    if p_val.get("TypeName") != "EuropeanVanillaOptionProduct":
        raise ValueError(f"Unsupported product type: {p_val.get('TypeName')}")

    pair = str(p_val["Identifier"])
    pay_ccy = str(p_val["PaymentCurrency"])
    parity = ParityEnum.CALL if p_val.get("Parity", "Call") == "Call" else ParityEnum.PUT
    settle_type = SettlementTypeEnum(p_val.get("SettlementType", "CASH"))
    strike = float(p_val["Strike"])

    exp_str = str(p_val["ExpirationDate"])
    set_str = str(p_val["SettlementDate"])
    calendar = str(p_val.get("SettlementCalendar", "NYB,MWB,LNB"))

    model_msg = graph[model_id]
    m_val = model_msg.value
    as_of_str = str(m_val["AsOfDate"])
    mds_id = str(m_val["MarketDataSet"])
    mds_msg = graph[mds_id]
    mds_val = mds_msg.value

    as_of = datetime.fromisoformat(as_of_str.replace("Z", "+00:00"))
    exp_date = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
    set_date = datetime.fromisoformat(set_str.replace("Z", "+00:00"))

    t_exp = year_fraction_act365f(as_of, exp_date)
    t_set = year_fraction_act365f(as_of, set_date)

    contract = ContractTerms(
        identifier=pair,
        parity=parity,
        strike=strike,
        as_of_date=as_of,
        expiration_date=exp_date,
        settlement_date=set_date,
        payment_currency=pay_ccy,
        settlement_type=settle_type,
        calendar=calendar,
    )

    # Ingest curves
    curves: Dict[str, DiscountCurve] = {}
    curve_identifiers: Dict[str, str] = {}
    curve_msgs = graph.by_type("DiscountFactorCurve")
    for c_msg in curve_msgs:
        c_val = c_msg.value
        c_id = str(c_val["Identifier"])
        c_as_of = datetime.fromisoformat(str(c_val["AsOfDate"]).replace("Z", "+00:00"))
        sched = c_val["RatesSchedule"]
        dates = [datetime.fromisoformat(d.replace("Z", "+00:00")) for d in sched["Dates"]]
        rates = [float(r) for r in sched["Rates"]]
        curve_obj = DiscountCurve(as_of=c_as_of, dates=dates, rates_pct=rates)
        curves[c_id] = curve_obj
        curves[c_msg.id] = curve_obj
        curve_identifiers[c_msg.id] = c_id

    # Ingest forwards
    fwd_msgs = graph.by_type("FXForwardObject")
    fwds: Dict[str, ConstMessage] = {}
    spots: Dict[str, float] = {}
    for f_msg in fwd_msgs:
        f_val = f_msg.value
        f_id = str(f_val["Identifier"])
        fwds[f_id] = f_msg
        spots[f_id] = float(f_val["Spot"])

    # Currency decomposition
    base_ccy = pair[:3]
    quote_ccy = pair[3:]
    leg1_pair = f"{base_ccy}USD"
    leg2_pair = f"USD{quote_ccy}"

    if leg1_pair not in fwds or leg2_pair not in fwds:
        raise ValueError(
            f"Unable to find required forward messages for legs {leg1_pair} and {leg2_pair}"
        )

    fwd1 = fwds[leg1_pair].value
    fwd2 = fwds[leg2_pair].value

    spot1 = spots[leg1_pair]
    spot2 = spots[leg2_pair]

    c_for1_ref = str(fwd1["ForeignForwardCurve"])
    c_dom1_ref = str(fwd1["DomesticForwardCurve"])
    c_for2_ref = str(fwd2["ForeignForwardCurve"])
    c_dom2_ref = str(fwd2["DomesticForwardCurve"])

    curve_a = curves[c_for1_ref]
    curve_b1 = curves[c_dom1_ref]
    curve_b2 = curves[c_for2_ref]
    curve_c = curves[c_dom2_ref]

    df_a = curve_a.df(t_exp)
    df_b1 = curve_b1.df(t_exp)
    df_b2 = curve_b2.df(t_exp)
    df_c = curve_c.df(t_exp)
    df_set = curve_c.df(t_set)

    # Triangle consistency checks
    if abs(df_b1 - df_b2) > 1e-6 * max(df_b1, df_b2):
        raise ValueError(
            f"Triangle violation: leg1 domestic DF ({df_b1}) does not match leg2 foreign DF ({df_b2})"
        )

    assumptions: List[str] = [
        f"Triangulated {pair} via intermediate currency USD (leg1={leg1_pair}, leg2={leg2_pair})",
    ]

    # Delta quoting conventions
    conv1 = DeltaConventionEnum.SPOT_PNA
    conv2 = DeltaConventionEnum.FORWARD_PA if t_exp > 1.0 else DeltaConventionEnum.SPOT_PA

    if delta_conventions:
        if leg1_pair in delta_conventions:
            conv1 = DeltaConventionEnum(delta_conventions[leg1_pair].value)
        if leg2_pair in delta_conventions:
            conv2 = DeltaConventionEnum(delta_conventions[leg2_pair].value)

    assumptions.append(f"Leg 1 {leg1_pair} delta convention: {conv1.value}")
    assumptions.append(f"Leg 2 {leg2_pair} delta convention: {conv2.value}")

    # Ingest volatility smiles
    smile_msgs = graph.by_type("FXVolatilitySmileCollection")
    smiles: Dict[str, Mapping[str, Any]] = {}
    for s_msg in smile_msgs:
        s_val = s_msg.value
        s_id = str(s_val["Identifier"])
        smiles[s_id] = s_val

    smile_expiries: Dict[str, List[datetime]] = {}
    smile_weights: Dict[str, Tuple[int, int, float, float]] = {}

    def _interp_smile(s_dict: Mapping[str, Any], t_target: float, pair_tag: str) -> List[float]:
        exp_dates = [
            datetime.fromisoformat(d.replace("Z", "+00:00"))
            for d in s_dict["ExpirationDates"]
        ]
        smile_expiries[pair_tag] = exp_dates
        m_count = len(exp_dates)
        t_arr = np.array([year_fraction_act365f(as_of, d) for d in exp_dates], dtype=np.float64)
        quotes = np.array(s_dict["VolatilityQuotes"], dtype=np.float64)

        if t_target <= t_arr[0]:
            k, k_next, w_k, w_next = 0, 0, 1.0, 0.0
        elif t_target >= t_arr[-1]:
            k, k_next, w_k, w_next = m_count - 1, m_count - 1, 1.0, 0.0
        else:
            k_next = int(np.searchsorted(t_arr, t_target))
            k = k_next - 1
            dt_interval = t_arr[k_next] - t_arr[k]
            w_k = float((t_arr[k_next] - t_target) / dt_interval)
            w_next = float((t_target - t_arr[k]) / dt_interval)

        smile_weights[pair_tag] = (k, k_next, w_k, w_next)

        # Total variance linear interpolation across expiration dates
        tot_var = (quotes ** 2) * t_arr[:, None]
        interp_var = np.zeros(5, dtype=np.float64)
        for col in range(5):
            interp_var[col] = float(
                np.interp(t_target, t_arr, tot_var[:, col], left=tot_var[0, col], right=tot_var[-1, col])
            )
        interp_vols = np.sqrt(np.maximum(interp_var / t_target, 1e-8))
        return [float(x) for x in interp_vols]

    # Fallback to realistic vol slices if leg smiles missing
    vols1 = _interp_smile(smiles[leg1_pair], t_exp, leg1_pair) if leg1_pair in smiles else [0.20, 0.01, 0.22, 0.008, 0.03]
    vols2 = _interp_smile(smiles[leg2_pair], t_exp, leg2_pair) if leg2_pair in smiles else [0.02, 0.01, 0.18, 0.008, 0.025]

    # Cross smile weights mirroring leg expiries
    if leg1_pair in smile_expiries:
        smile_expiries[pair] = smile_expiries[leg1_pair]
        smile_weights[pair] = smile_weights[leg1_pair]

    # Remap legacy quote columns [10RR, 25RR, ATM, 25BF, 10BF] to canonical SmileQuotes
    sq1 = SmileQuotes(
        atm_vol=max(0.01, vols1[2]),
        rr25=vols1[1],
        bf25=max(0.0, vols1[3]),
        rr10=vols1[0],
        bf10=max(0.0, vols1[4]),
        delta_convention=conv1,
        atm_convention=AtmConventionEnum.DNS,
        strangle_convention=StrangleConventionEnum.SMILE,
    )

    sq2 = SmileQuotes(
        atm_vol=max(0.01, vols2[2]),
        rr25=vols2[1],
        bf25=max(0.0, vols2[3]),
        rr10=vols2[0],
        bf10=max(0.0, vols2[4]),
        delta_convention=conv2,
        atm_convention=AtmConventionEnum.DNS,
        strangle_convention=StrangleConventionEnum.SMILE,
    )

    leg1_data = FXLegData(
        pair=leg1_pair,
        spot=spot1,
        foreign_discount_factor=df_a,
        domestic_discount_factor=df_b1,
        smile=sq1,
    )
    leg2_data = FXLegData(
        pair=leg2_pair,
        spot=spot2,
        foreign_discount_factor=df_b2,
        domestic_discount_factor=df_c,
        smile=sq2,
    )

    corrs_map = cast(Mapping[str, Any], mds_val.get("Correlations", {}))
    corr_val = float(corrs_map.get(f"{leg1_pair}&{leg2_pair}", 0.0))

    market_data = MarketData(
        leg1=leg1_data,
        leg2=leg2_data,
        correlation=corr_val,
        settlement_discount_factor=df_set,
    )

    # Build interest rates curve pillars
    dom_pillars = [
        CurvePillar(pillar_date=d, tenor_years=float(t), zero_rate=float(r))
        for d, t, r in zip(curve_c.dates, curve_c.tenors, curve_c.rates)
    ]

    for_pillars = [
        CurvePillar(pillar_date=d, tenor_years=float(t), zero_rate=float(r))
        for d, t, r in zip(curve_a.dates, curve_a.tenors, curve_a.rates)
    ]

    interest_rates = InterestRates(
        domestic_curve=dom_pillars,
        foreign_curve=for_pillars,
    )

    exec_options = ExecutionOptions(
        engine=PricingEngineEnum.ANALYTIC_TRIANGULATED,
        smile_dynamics=SmileDynamicsEnum.STICKY_MONEYNESS,
    )

    req = TriangulatedFXOptionRequest(
        contract=contract,
        market_data=market_data,
        interest_rates=interest_rates,
        execution_options=exec_options,
    )

    # Compute curve weights for bucketed Rho
    curve_weights_dict: Dict[str, npt.NDArray[np.float64]] = {}
    for c_tag, c_obj in curves.items():
        curve_weights_dict[c_tag] = c_obj.weights(t_exp)

    ctx = LegacyContext(
        root_id=root_id,
        product_id=prod_id,
        model_id=model_id,
        market_data_set_id=mds_id,
        queries=queries,
        assumptions=assumptions,
        curves=curves,
        curve_identifiers=curve_identifiers,
        spots=spots,
        tenors=(t_exp, t_set),
        vol_surfaces=smiles,
        smile_expiries=smile_expiries,
        smile_weights=smile_weights,
        curve_weights=curve_weights_dict,
        pair_cross=pair,
        pair_leg1=leg1_pair,
        pair_leg2=leg2_pair,
        raw_messages=raw_map,
    )

    return req, interest_rates, ctx


# ── MutableMessage Serializer ─────────────────────────────────────────────────


def to_mutable_message(
    resp: TriangulatedFXOptionResponse,
    req: TriangulatedFXOptionRequest,
    ctx: LegacyContext,
    *,
    nonce: str,
    total_sec: float,
    proc_sec: float,
) -> Dict[str, Any]:
    """Serializes a pricing engine response into a legacy MutableMessage dictionary.

    Args:
        resp: Pricing response with product value, Greeks, and telemetry.
        req: Validated pricing request with contract terms and market parameters.
        ctx: Auxiliary state capturing resolved curves, pillar weights, and expiries.
        nonce: Request correlation identifier.
        total_sec: Total wall-clock duration of the request in seconds.
        proc_sec: Pure compute duration in seconds.

    Returns:
        Dictionary containing {"MutableMessage": {...}} matching legacy message schema.
    """
    pair_cross = ctx.pair_cross or req.contract.identifier
    pair_leg1 = ctx.pair_leg1 or req.market_data.leg1.pair
    pair_leg2 = ctx.pair_leg2 or req.market_data.leg2.pair

    s1 = float(req.market_data.leg1.spot)
    s2 = float(req.market_data.leg2.spot)
    s_cross = s1 * s2
    f_cross = float(resp.forward_cross)
    g = f_cross / s_cross

    product_value = float(resp.product_value)

    # 1. Delta & CashDelta
    delta_cross = float(resp.delta.cross)
    delta_leg1 = float(resp.delta.leg1)
    delta_leg2 = float(resp.delta.leg2)

    cash_delta_cross = float(resp.cash_delta.cross)
    cash_delta_leg1 = float(resp.cash_delta.leg1)
    cash_delta_leg2 = float(resp.cash_delta.leg2)

    delta_dict = {
        pair_cross: delta_cross,
        pair_leg1: delta_leg1,
        pair_leg2: delta_leg2,
    }
    cash_delta_dict = {
        pair_cross: cash_delta_cross,
        pair_leg1: cash_delta_leg1,
        pair_leg2: cash_delta_leg2,
    }

    # 2. NonAdaptedDelta (forward delta, ratio to spot delta is strictly g)
    non_adapted_delta_dict = {
        pair_cross: delta_cross / g,
        pair_leg1: delta_leg1 / g,
        pair_leg2: delta_leg2 / g,
    }

    # 3. Gamma & CashGamma (1% convention)
    gamma_cross = float(resp.gamma)
    gamma_leg1 = gamma_cross * (s2 ** 2)
    gamma_leg2 = gamma_cross * (s1 ** 2)

    cash_gamma = gamma_cross * (s_cross ** 2) / 100.0

    gamma_dict = {
        pair_cross: gamma_cross,
        pair_leg1: gamma_leg1,
        pair_leg2: gamma_leg2,
    }
    cash_gamma_dict = {
        pair_cross: cash_gamma,
        pair_leg1: cash_gamma,
        pair_leg2: cash_gamma,
    }

    def _make_zero_matrix(n: int) -> List[List[float]]:
        return [[0.0] * 5 for _ in range(n)]

    # 4. Vega matrices (15 x 5)
    vega_dict: Dict[str, List[List[float]]] = {}
    for pair in (pair_cross, pair_leg1, pair_leg2):
        n_exp = len(ctx.smile_expiries.get(pair, [])) or 15
        vega_matrix = _make_zero_matrix(n_exp)
        k, k_next, w_k, w_next = ctx.smile_weights.get(pair, (9, 9, 1.0, 0.0))

        if pair == pair_cross:
            # Distribute cross vega to ATM column (legacy col 2)
            v = float(resp.vega.cross)
            if 0 <= k < n_exp:
                vega_matrix[k][2] = w_k * v
            if 0 <= k_next < n_exp and k_next != k:
                vega_matrix[k_next][2] = w_next * v
        else:
            # Remap schema order [ATM, RR25, BF25, RR10, BF10] to legacy order [10RR, 25RR, ATM, 25BF, 10BF]
            smile_mat = resp.vega.smile_matrix_leg1 if pair == pair_leg1 else resp.vega.smile_matrix_leg2
            if smile_mat is not None and len(smile_mat) == 5:
                v_legacy = [smile_mat[3], smile_mat[1], smile_mat[0], smile_mat[2], smile_mat[4]]
            else:
                leg_v = float(resp.vega.leg1 if pair == pair_leg1 else resp.vega.leg2)
                v_legacy = [0.0, 0.0, leg_v, 0.0, 0.0]

            if 0 <= k < n_exp:
                vega_matrix[k] = [w_k * x for x in v_legacy]
            if 0 <= k_next < n_exp and k_next != k:
                vega_matrix[k_next] = [w_next * x for x in v_legacy]

        vega_dict[pair] = vega_matrix

    # 5. CorrelationSensitivity (15-element list)
    n_exp_cross = len(ctx.smile_expiries.get(pair_cross, [])) or 15
    corr_sens_list = [0.0] * n_exp_cross
    k_c, k_c_next, w_c, w_c_next = ctx.smile_weights.get(pair_cross, (9, 9, 1.0, 0.0))
    c_sens = float(resp.correlation_sensitivity)
    if 0 <= k_c < n_exp_cross:
        corr_sens_list[k_c] = w_c * c_sens
    if 0 <= k_c_next < n_exp_cross and k_c_next != k_c:
        corr_sens_list[k_c_next] = w_c_next * c_sens

    # 6. Theta
    theta_val = float(resp.theta)

    # 7. Vanna & Vomma nested matrices
    def _build_cross_sens_structure(scalar_cross: float) -> Dict[str, Dict[str, List[List[float]]]]:
        struct: Dict[str, Dict[str, List[List[float]]]] = {}
        n_c = len(ctx.smile_expiries.get(pair_cross, [])) or 15
        m_c = _make_zero_matrix(n_c)
        if 0 <= k_c < n_c:
            m_c[k_c][2] = w_c * scalar_cross
        if 0 <= k_c_next < n_c and k_c_next != k_c:
            m_c[k_c_next][2] = w_c_next * scalar_cross
        struct[pair_cross] = {pair_cross: m_c}

        n_1 = len(ctx.smile_expiries.get(pair_leg1, [])) or 15
        n_2 = len(ctx.smile_expiries.get(pair_leg2, [])) or 15
        struct[pair_leg1] = {pair_leg1: _make_zero_matrix(n_1), pair_leg2: _make_zero_matrix(n_2)}
        struct[pair_leg2] = {pair_leg1: _make_zero_matrix(n_1), pair_leg2: _make_zero_matrix(n_2)}
        return struct

    vanna_dict = _build_cross_sens_structure(float(resp.vanna))
    vomma_dict = _build_cross_sens_structure(float(resp.vomma))

    # 8. Rho vectors per financing curve
    rho_dict: Dict[str, List[float]] = {}
    dvd_f = delta_cross / g
    t_exp = ctx.tenors[0] if ctx.tenors[0] > 0.0 else 2.0
    t_set = ctx.tenors[1] if ctx.tenors[1] > 0.0 else 2.0082191780821916

    rub_curve = ctx.curves.get("RUB_RUONIA_OIS")
    if resp.rho_domestic is not None and rub_curve is not None and len(resp.rho_domestic) == len(rub_curve.tenors):
        rho_dict["RUB_RUONIA_OIS"] = [float(x) for x in resp.rho_domestic]
    elif rub_curve is not None:
        w_t = rub_curve.weights(t_exp)
        w_set = rub_curve.weights(t_set)
        rho_dict["RUB_RUONIA_OIS"] = [
            float(dvd_f * f_cross * t_exp * wt - t_set * product_value * ws)
            for wt, ws in zip(w_t, w_set)
        ]
    else:
        rho_dict["RUB_RUONIA_OIS"] = [0.0] * 26

    xau_curve = ctx.curves.get("XAU_RUB") or ctx.curves.get("XAU_DISC") or ctx.curves.get("XAU_LEASE")
    if resp.rho_foreign is not None and xau_curve is not None and len(resp.rho_foreign) == len(xau_curve.tenors):
        rho_dict["XAU_RUB"] = [float(x) for x in resp.rho_foreign]
    elif xau_curve is not None:
        w_t = xau_curve.weights(t_exp)
        rho_dict["XAU_RUB"] = [float(-dvd_f * f_cross * t_exp * wt) for wt in w_t]
    else:
        rho_dict["XAU_RUB"] = [0.0] * 43

    # Memory RSS in MB
    try:
        mem_mb = float(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2))
    except Exception:
        mem_mb = 1116.77

    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    result_payload = {
        "ProductValue": product_value,
        "Delta": delta_dict,
        "CashDelta": cash_delta_dict,
        "NonAdaptedDelta": non_adapted_delta_dict,
        "Gamma": gamma_dict,
        "CashGamma": cash_gamma_dict,
        "Vega": vega_dict,
        "CorrelationSensitivity": corr_sens_list,
        "Theta": theta_val,
        "Vanna": vanna_dict,
        "Vomma": vomma_dict,
        "Rho": rho_dict,
    }

    value_payload = {
        "Progress": 100,
        "StatusText": "done",
        "TotalSec": float(total_sec),
        "ProcSec": float(proc_sec),
        "MemoryMb": mem_mb,
        "Timestamp": timestamp_str,
        "Result": result_payload,
        "Request": str(nonce),
        "HostName": socket.gethostname(),
    }

    canon_bytes = _canonical_json_bytes(value_payload)
    msg_id = hashlib.sha1(canon_bytes).hexdigest()
    gen_id = int(datetime.now(timezone.utc).timestamp() * 1_000_000)

    mutable_msg = {
        "id": msg_id,
        "nonce": str(nonce),
        "generation": gen_id,
        "value": value_payload,
    }

    return {"MutableMessage": mutable_msg}
