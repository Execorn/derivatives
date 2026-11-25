"""Canonical feature tensor layout and monitoring index codes for EQAutocallable."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Literal, Optional

FEATURE_SCHEMA_VERSION: str = "2.0"
D_MAX_DEFAULT: int = 8
M_MAX_DEFAULT: int = 48


class MonitoringIndexCode(IntEnum):
    """Canonical integer codes for all 18 Murex monitoring functionals (0..17)."""

    WORST_OF = 0
    BEST_OF = 1
    AVERAGE_OF = 2
    DISPERSION = 3
    RATIO_AB = 4
    RATIO_BA = 5
    RATIO_BEST_WORST = 6
    RATIO_BEST_AVERAGE = 7
    RATIO_AVERAGE_WORST = 8
    DIFF_AB = 9
    DIFF_BA = 10
    DIFF_BEST_WORST = 11
    DIFF_BEST_AVERAGE = 12
    DIFF_AVERAGE_WORST = 13
    TWIN_WIN = 14
    OTHER_EXOTIC = 14  # Alias retained for checkpoints expecting n_func_codes >= 15.
    SECOND_WORST_OF = 15
    THIRD_WORST_OF = 16
    SECOND_BEST_OF = 17


_MONITORING_INDEX_MAP: Dict[str, MonitoringIndexCode] = {
    "WorstOf": MonitoringIndexCode.WORST_OF,
    "Lookback": MonitoringIndexCode.WORST_OF,
    "BestOf": MonitoringIndexCode.BEST_OF,
    "Average": MonitoringIndexCode.AVERAGE_OF,
    "AverageOf": MonitoringIndexCode.AVERAGE_OF,
    "MultiPerf": MonitoringIndexCode.AVERAGE_OF,
    "EquallyWeightedBasket": MonitoringIndexCode.AVERAGE_OF,
    "Dispersion": MonitoringIndexCode.DISPERSION,
    "RatioAB": MonitoringIndexCode.RATIO_AB,
    "Ratio": MonitoringIndexCode.RATIO_AB,
    "RatioBA": MonitoringIndexCode.RATIO_BA,
    "RatioBestWorst": MonitoringIndexCode.RATIO_BEST_WORST,
    "RatioBestAverage": MonitoringIndexCode.RATIO_BEST_AVERAGE,
    "RatioAverageWorst": MonitoringIndexCode.RATIO_AVERAGE_WORST,
    "DiffAB": MonitoringIndexCode.DIFF_AB,
    "Outperformance": MonitoringIndexCode.DIFF_AB,
    "Spread": MonitoringIndexCode.DIFF_AB,
    "DiffBA": MonitoringIndexCode.DIFF_BA,
    "DiffBestWorst": MonitoringIndexCode.DIFF_BEST_WORST,
    "Rainbow": MonitoringIndexCode.DIFF_BEST_WORST,
    "SpreadBestWorst": MonitoringIndexCode.DIFF_BEST_WORST,
    "DiffBestAverage": MonitoringIndexCode.DIFF_BEST_AVERAGE,
    "SpreadBestAverage": MonitoringIndexCode.DIFF_BEST_AVERAGE,
    "DiffAverageWorst": MonitoringIndexCode.DIFF_AVERAGE_WORST,
    "SpreadAverageWorst": MonitoringIndexCode.DIFF_AVERAGE_WORST,
    "TwinWin": MonitoringIndexCode.TWIN_WIN,
    "SecondWorstOf": MonitoringIndexCode.SECOND_WORST_OF,
    "ThirdWorstOf": MonitoringIndexCode.THIRD_WORST_OF,
    "SecondBestOf": MonitoringIndexCode.SECOND_BEST_OF,
}

PAIR_ORDER_DEPENDENT_INDICES: frozenset[str] = frozenset(
    {
        "RatioAB",
        "Ratio",
        "RatioBA",
        "DiffAB",
        "DiffBA",
        "Outperformance",
        "Spread",
    }
)


def monitoring_index_to_code(index_name: Optional[str]) -> int:
    """Maps a Murex monitoring index name to its canonical integer code."""
    if not index_name:
        return int(MonitoringIndexCode.WORST_OF)
    cleaned = index_name.strip()
    if not cleaned:
        return int(MonitoringIndexCode.WORST_OF)
    return int(_MONITORING_INDEX_MAP.get(cleaned, MonitoringIndexCode.OTHER_EXOTIC))


class AutocallNodeCol(IntEnum):
    """Per-asset node feature columns: X_node in R^{B x D_max x 16}."""

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
    ROLE_LEG_A = 14
    ROLE_LEG_B = 15


class AutocallEdgeCol(IntEnum):
    """Pairwise asset edge feature columns: X_edge in R^{B x D_max x D_max x 4}."""

    CORR_RHO = 0
    COVARIANCE_TERM = 1
    LOG_PERF_DIFF = 2
    OFF_DIAG_MASK = 3


class AutocallSchedCol(IntEnum):
    """Per-observation-period schedule feature columns: X_sched in R^{B x M_max x 12}."""

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
    """Contract-level global context feature columns: X_global in R^{B x 16}."""

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
    LOWER_STRIKE = 10
    CALL_INDEX_ID = 11
    CPN_INDEX_ID = 12
    OPT_INDEX_ID = 13
    IS_LONG_BASKET = 14
    ABS_KI_SPREAD = 15


@dataclass(frozen=True)
class CanonicalTier3Config:
    """Query-invariant Tier-3/4 base configuration guaranteeing 0.00 bps pv_only==all_greeks drift."""

    n_paths: int
    n_replicates: int
    n_substeps: int
    lv_grid_nt: int
    lv_grid_ny: int
    cv_mode: Literal["multifidelity_gbm", "bs1d_exact", "none"]
    n_cheap_mf: int
    seed: int
    adaptive_substeps: bool = False


def resolve_canonical_tier3_config(
    d_assets: int,
    m_rem: int,
    *,
    device_type: Literal["cuda", "cpu"] | str,
    n_fallback_paths: Optional[int] = None,
    seed: int = 42,
) -> CanonicalTier3Config:
    """Deterministic resolver depending strictly on (d_assets, m_rem, device_type, n_fallback_paths), NEVER on Queries."""
    d_val = int(d_assets)
    m_val = int(m_rem)
    dev_tok = "cuda" if str(device_type).strip().lower().startswith("cuda") else "cpu"
    short_tenor = 0 < m_val <= 10
    cpu_very_short = 0 < m_val <= 6
    if n_fallback_paths is not None:
        eff_paths = max(2, int(n_fallback_paths))
        n_sub = 8 if short_tenor else 4
    elif dev_tok == "cuda":
        eff_paths = 4096 if m_val > 20 else 8192
        n_sub = 8 if short_tenor else 4
    elif cpu_very_short:
        eff_paths = 8192
        n_sub = 8
    elif m_val > 24:
        eff_paths = 2048
        n_sub = 4
    elif d_val >= 6 and m_val <= 12:
        eff_paths = 8192
        n_sub = 4
    else:
        eff_paths = 4096
        n_sub = 4
    eff_r = min(max(1, eff_paths // 2), 8 if eff_paths >= 8192 else 4)
    nt, ny = 32, 64
    n_cheap = (
        max(262144, 8 * eff_paths)
        if (dev_tok == "cuda" or (n_fallback_paths is not None and short_tenor) or (dev_tok == "cpu" and cpu_very_short))
        else (max(16384, 8 * eff_paths) if (dev_tok == "cpu" and d_val * m_val >= 200) else max(32768, 8 * eff_paths))
    )
    return CanonicalTier3Config(
        n_paths=eff_paths,
        n_replicates=eff_r,
        n_substeps=n_sub,
        lv_grid_nt=nt,
        lv_grid_ny=ny,
        cv_mode="multifidelity_gbm",
        n_cheap_mf=n_cheap,
        seed=int(seed),
        adaptive_substeps=False,
    )


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "D_MAX_DEFAULT",
    "M_MAX_DEFAULT",
    "PAIR_ORDER_DEPENDENT_INDICES",
    "MonitoringIndexCode",
    "monitoring_index_to_code",
    "AutocallNodeCol",
    "AutocallEdgeCol",
    "AutocallSchedCol",
    "AutocallGlobalCol",
    "CanonicalTier3Config",
    "resolve_canonical_tier3_config",
]
