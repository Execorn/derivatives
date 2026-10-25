from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import orjson
import pyarrow as pa  # type: ignore[import-untyped]
import torch

from deepvol.models.autocall.greeks_engine import BatchedSurrogateGreeksOutput
from deepvol.models.autocall.pre_post_processor import D_MAX_DEFAULT, M_MAX_DEFAULT

ARROW_MIME: Final[str] = "application/vnd.apache.arrow.stream"
EQ_AUTOCALL_HEADER_KEY: Final[bytes] = b"deepvol.eq_autocall.header"
HEADER_METADATA_KEY: Final[bytes] = b"deepvol.header"

D_MAX_ARROW: Final[int] = D_MAX_DEFAULT
M_MAX_ARROW: Final[int] = M_MAX_DEFAULT
MAX_BATCH_TRADES: Final[int] = 16384


def _zero_copy_tensor_from_numpy(arr: npt.NDArray[np.float64]) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(arr)


@dataclass(frozen=True)
class DecodedEQAutocallArrowBatch:
    """Contiguous `torch.float64` batch extracted from an Apache Arrow IPC stream request."""

    node_feats: torch.Tensor
    edge_feats: torch.Tensor
    asset_mask: torch.Tensor
    sched_feats: torch.Tensor
    sched_mask: torch.Tensor
    global_feats: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    spots_t0: torch.Tensor | None = None
    ref_spots: torch.Tensor | None = None
    pv_in_transit: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.node_feats.shape[0])

    @property
    def d_max(self) -> int:
        return int(self.node_feats.shape[1])

    @property
    def m_max(self) -> int:
        return int(self.sched_feats.shape[1])

    @property
    def metadata_dict(self) -> dict[str, Any]:
        return self.metadata

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "node_feats": self.node_feats,
            "edge_feats": self.edge_feats,
            "asset_mask": self.asset_mask,
            "sched_feats": self.sched_feats,
            "sched_mask": self.sched_mask,
            "global_feats": self.global_feats,
            "metadata": self.metadata,
        }
        if self.spots_t0 is not None:
            out["spots_t0"] = self.spots_t0
        if self.ref_spots is not None:
            out["ref_spots"] = self.ref_spots
        if self.pv_in_transit is not None:
            out["pv_in_transit"] = self.pv_in_transit
        return out

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _to_contiguous_f64_np(
    val: torch.Tensor | npt.NDArray[Any] | Sequence[Any],
    *,
    name: str,
) -> npt.NDArray[np.float64]:
    """Converts a PyTorch tensor, NumPy array, or nested sequence to a C-contiguous float64 NumPy array."""
    if isinstance(val, torch.Tensor):
        arr = val.detach().cpu().to(dtype=torch.float64).contiguous().numpy()
    else:
        arr = np.ascontiguousarray(val, dtype=np.float64)
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"Input tensor '{name}' contains non-finite (NaN or Inf) values.")
    return arr


def _make_fixed_size_f64_column(
    arr: npt.NDArray[np.float64],
    batch_size: int,
    inner_size: int,
) -> pa.Array:
    """Constructs a zero-copy FixedSizeListArray(float64, inner_size) of length `batch_size`."""
    flat = np.ascontiguousarray(arr.reshape(batch_size * inner_size), dtype=np.float64)
    val_buf = pa.py_buffer(flat)
    flat_arr = pa.Array.from_buffers(pa.float64(), batch_size * inner_size, [None, val_buf])
    return pa.FixedSizeListArray.from_arrays(flat_arr, inner_size)


def _make_1d_f64_column(
    arr: npt.NDArray[np.float64],
    batch_size: int,
) -> pa.Array:
    """Constructs a zero-copy 1D float64 Arrow Array of length `batch_size`."""
    flat = np.ascontiguousarray(arr.reshape(batch_size), dtype=np.float64)
    val_buf = pa.py_buffer(flat)
    return pa.Array.from_buffers(pa.float64(), batch_size, [None, val_buf])


def _extract_column_f64_np(
    table: pa.Table,
    col_name: str,
    expected_shape: tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Extracts a column from `table` into a contiguous `np.float64` array of `expected_shape` via zero-copy `np.frombuffer`."""
    if col_name not in table.column_names:
        raise ValueError(f"Missing required Arrow column: '{col_name}'.")

    chunked = table.column(col_name)
    combined = chunked.combine_chunks() if chunked.num_chunks > 1 else (
        chunked.chunk(0) if chunked.num_chunks == 1 else pa.array([], type=pa.float64())
    )

    flat_arr = (
        combined.values
        if isinstance(combined.type, (pa.FixedSizeListType, pa.ListType))
        else combined
    )
    expected_numel = int(np.prod(expected_shape))

    if (
        flat_arr.type == pa.float64()
        and flat_arr.null_count == 0
        and len(flat_arr.buffers()) >= 2
        and flat_arr.buffers()[1] is not None
    ):
        if len(flat_arr) != expected_numel:
            raise ValueError(
                f"Column '{col_name}' element count ({len(flat_arr)}) does not match expected shape {expected_shape} ({expected_numel})."
            )
        val_buf = flat_arr.buffers()[1]
        out = np.frombuffer(
            val_buf,
            dtype=np.float64,
            count=expected_numel,
            offset=int(flat_arr.offset) * 8,
        ).reshape(expected_shape)
    else:
        raw_np = flat_arr.to_numpy(zero_copy_only=False)
        if raw_np.size != expected_numel:
            raise ValueError(
                f"Column '{col_name}' element count ({raw_np.size}) does not match expected shape {expected_shape} ({expected_numel})."
            )
        out = np.ascontiguousarray(raw_np, dtype=np.float64).reshape(expected_shape)

    if not bool(np.all(np.isfinite(out))):
        raise ValueError(f"Decoded Arrow column '{col_name}' contains non-finite (NaN or Inf) values.")
    return out


def encode_eq_autocall_arrow_request(
    node_feats: torch.Tensor | npt.NDArray[Any] | Sequence[Any],
    edge_feats: torch.Tensor | npt.NDArray[Any] | Sequence[Any],
    asset_mask: torch.Tensor | npt.NDArray[Any] | Sequence[Any],
    sched_feats: torch.Tensor | npt.NDArray[Any] | Sequence[Any],
    sched_mask: torch.Tensor | npt.NDArray[Any] | Sequence[Any],
    global_feats: torch.Tensor | npt.NDArray[Any] | Sequence[Any],
    metadata_dict: Mapping[str, Any] | None = None,
    *,
    spots_t0: torch.Tensor | npt.NDArray[Any] | Sequence[Any] | None = None,
    ref_spots: torch.Tensor | npt.NDArray[Any] | Sequence[Any] | None = None,
    pv_in_transit: torch.Tensor | npt.NDArray[Any] | Sequence[Any] | float | None = None,
) -> bytes:
    """Serializes a batch of B trades into an Apache Arrow IPC stream (`pa.ipc.new_stream`).

    All feature & mask tensors are serialized in IEEE-754 `float64` precision (REM-S0-01):
      - `node_feats`: `(B, D_max, 14)` (typically `D_max = 8`)
      - `edge_feats`: `(B, D_max, D_max, 4)`
      - `asset_mask`: `(B, D_max)`
      - `sched_feats`: `(B, M_max, 12)` (typically `M_max = 48`)
      - `sched_mask`: `(B, M_max)`
      - `global_feats`: `(B, 10)`
      - `spots_t0` (optional): `(B, D_max)`
      - `ref_spots` (optional): `(B, D_max)`
      - `pv_in_transit` (optional): `(B,)`
    """
    node_np = _to_contiguous_f64_np(node_feats, name="node_feats")
    edge_np = _to_contiguous_f64_np(edge_feats, name="edge_feats")
    amask_np = _to_contiguous_f64_np(asset_mask, name="asset_mask")
    sched_np = _to_contiguous_f64_np(sched_feats, name="sched_feats")
    smask_np = _to_contiguous_f64_np(sched_mask, name="sched_mask")
    glob_np = _to_contiguous_f64_np(global_feats, name="global_feats")

    # Support unbatched single-trade inputs by prepending batch dimension B = 1
    if node_np.ndim == 2:
        node_np = node_np[None, ...]
    if edge_np.ndim == 3:
        edge_np = edge_np[None, ...]
    if amask_np.ndim == 1:
        amask_np = amask_np[None, ...]
    if sched_np.ndim == 2:
        sched_np = sched_np[None, ...]
    if smask_np.ndim == 1:
        smask_np = smask_np[None, ...]
    if glob_np.ndim == 1:
        glob_np = glob_np[None, ...]

    if node_np.ndim != 3 or node_np.shape[2] != 14:
        raise ValueError(f"Expected node_feats of shape (B, D_max, 14), got {node_np.shape}.")
    batch_size, d_max, _ = node_np.shape
    if batch_size < 1 or batch_size > MAX_BATCH_TRADES:
        raise ValueError(f"Batch size {batch_size} out of bounds [1, {MAX_BATCH_TRADES}].")

    if edge_np.shape != (batch_size, d_max, d_max, 4):
        raise ValueError(
            f"Expected edge_feats of shape {(batch_size, d_max, d_max, 4)}, got {edge_np.shape}."
        )
    if amask_np.shape != (batch_size, d_max):
        raise ValueError(
            f"Expected asset_mask of shape {(batch_size, d_max)}, got {amask_np.shape}."
        )
    if sched_np.ndim != 3 or sched_np.shape[0] != batch_size or sched_np.shape[2] != 12:
        raise ValueError(
            f"Expected sched_feats of shape (B={batch_size}, M_max, 12), got {sched_np.shape}."
        )
    m_max = int(sched_np.shape[1])
    if smask_np.shape != (batch_size, m_max):
        raise ValueError(
            f"Expected sched_mask of shape {(batch_size, m_max)}, got {smask_np.shape}."
        )
    if glob_np.shape != (batch_size, 10):
        raise ValueError(
            f"Expected global_feats of shape {(batch_size, 10)}, got {glob_np.shape}."
        )

    cols = [
        _make_fixed_size_f64_column(node_np, batch_size, d_max * 14),
        _make_fixed_size_f64_column(edge_np, batch_size, d_max * d_max * 4),
        _make_fixed_size_f64_column(amask_np, batch_size, d_max),
        _make_fixed_size_f64_column(sched_np, batch_size, m_max * 12),
        _make_fixed_size_f64_column(smask_np, batch_size, m_max),
        _make_fixed_size_f64_column(glob_np, batch_size, 10),
    ]
    names = [
        "node_feats",
        "edge_feats",
        "asset_mask",
        "sched_feats",
        "sched_mask",
        "global_feats",
    ]

    if spots_t0 is not None:
        spots_np = _to_contiguous_f64_np(spots_t0, name="spots_t0")
        if spots_np.ndim == 1:
            spots_np = spots_np[None, ...]
        if spots_np.shape != (batch_size, d_max):
            raise ValueError(f"Expected spots_t0 of shape {(batch_size, d_max)}, got {spots_np.shape}.")
        cols.append(_make_fixed_size_f64_column(spots_np, batch_size, d_max))
        names.append("spots_t0")

    if ref_spots is not None:
        ref_np = _to_contiguous_f64_np(ref_spots, name="ref_spots")
        if ref_np.ndim == 1:
            ref_np = ref_np[None, ...]
        if ref_np.shape != (batch_size, d_max):
            raise ValueError(f"Expected ref_spots of shape {(batch_size, d_max)}, got {ref_np.shape}.")
        cols.append(_make_fixed_size_f64_column(ref_np, batch_size, d_max))
        names.append("ref_spots")

    if pv_in_transit is not None:
        if isinstance(pv_in_transit, (int, float)):
            pvit_np = np.full(batch_size, float(pv_in_transit), dtype=np.float64)
        else:
            pvit_np = _to_contiguous_f64_np(pv_in_transit, name="pv_in_transit")
            if pvit_np.ndim == 0:
                pvit_np = np.full(batch_size, float(pvit_np), dtype=np.float64)
            elif pvit_np.shape != (batch_size,):
                raise ValueError(f"Expected pv_in_transit of shape {(batch_size,)}, got {pvit_np.shape}.")
        cols.append(_make_1d_f64_column(pvit_np, batch_size))
        names.append("pv_in_transit")

    user_meta = dict(metadata_dict) if metadata_dict is not None else {}
    header_obj: dict[str, Any] = {
        **user_meta,
        "batch_size": int(batch_size),
        "d_max": int(d_max),
        "m_max": int(m_max),
        "metadata": user_meta,
    }
    header_bytes = orjson.dumps(header_obj, option=orjson.OPT_SERIALIZE_NUMPY)

    batch = pa.RecordBatch.from_arrays(cols, names=names)
    schema_with_meta = batch.schema.with_metadata(
        {
            EQ_AUTOCALL_HEADER_KEY: header_bytes,
            HEADER_METADATA_KEY: header_bytes,
        }
    )
    batch = batch.replace_schema_metadata(schema_with_meta.metadata)

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    res_bytes: bytes = sink.getvalue().to_pybytes()
    return res_bytes


def decode_eq_autocall_arrow_request(body: bytes) -> DecodedEQAutocallArrowBatch:
    """Decodes an Apache Arrow IPC stream request into contiguous `torch.float64` tensors."""
    if not body:
        raise ValueError("Empty Arrow IPC request payload.")

    try:
        reader = pa.ipc.open_stream(body)
        table = reader.read_all()
    except Exception as exc:
        raise ValueError(f"Invalid Apache Arrow IPC stream: {exc}") from exc

    raw_meta = reader.schema.metadata or {}
    header_raw_bytes = raw_meta.get(EQ_AUTOCALL_HEADER_KEY) or raw_meta.get(HEADER_METADATA_KEY)
    header_dict: dict[str, Any] = {}
    if header_raw_bytes is not None:
        try:
            loaded = orjson.loads(header_raw_bytes)
            if isinstance(loaded, dict):
                header_dict = loaded
        except Exception as exc:
            raise ValueError(f"Corrupt JSON in Arrow header metadata: {exc}") from exc

    batch_size = int(table.num_rows)
    if batch_size < 1 or batch_size > MAX_BATCH_TRADES:
        raise ValueError(f"Arrow batch row count {batch_size} out of bounds [1, {MAX_BATCH_TRADES}].")

    d_max = int(header_dict.get("d_max", D_MAX_ARROW))
    m_max = int(header_dict.get("m_max", M_MAX_ARROW))

    node_np = _extract_column_f64_np(table, "node_feats", (batch_size, d_max, 14))
    edge_np = _extract_column_f64_np(table, "edge_feats", (batch_size, d_max, d_max, 4))
    amask_np = _extract_column_f64_np(table, "asset_mask", (batch_size, d_max))
    sched_np = _extract_column_f64_np(table, "sched_feats", (batch_size, m_max, 12))
    smask_np = _extract_column_f64_np(table, "sched_mask", (batch_size, m_max))
    glob_np = _extract_column_f64_np(table, "global_feats", (batch_size, 10))

    spots_t0_t: torch.Tensor | None = None
    if "spots_t0" in table.column_names:
        spots_t0_t = _zero_copy_tensor_from_numpy(
            _extract_column_f64_np(table, "spots_t0", (batch_size, d_max))
        )

    ref_spots_t: torch.Tensor | None = None
    if "ref_spots" in table.column_names:
        ref_spots_t = _zero_copy_tensor_from_numpy(
            _extract_column_f64_np(table, "ref_spots", (batch_size, d_max))
        )

    pv_in_transit_t: torch.Tensor | None = None
    if "pv_in_transit" in table.column_names:
        pv_in_transit_t = _zero_copy_tensor_from_numpy(
            _extract_column_f64_np(table, "pv_in_transit", (batch_size,))
        )

    user_meta: dict[str, Any] = {}
    if isinstance(header_dict.get("metadata"), dict):
        user_meta.update(header_dict["metadata"])
    for k, v in header_dict.items():
        if k != "metadata" and k not in user_meta:
            user_meta[k] = v

    return DecodedEQAutocallArrowBatch(
        node_feats=_zero_copy_tensor_from_numpy(node_np),
        edge_feats=_zero_copy_tensor_from_numpy(edge_np),
        asset_mask=_zero_copy_tensor_from_numpy(amask_np),
        sched_feats=_zero_copy_tensor_from_numpy(sched_np),
        sched_mask=_zero_copy_tensor_from_numpy(smask_np),
        global_feats=_zero_copy_tensor_from_numpy(glob_np),
        metadata=user_meta,
        spots_t0=spots_t0_t,
        ref_spots=ref_spots_t,
        pv_in_transit=pv_in_transit_t,
    )


def encode_eq_autocall_arrow_response(
    batch_out: BatchedSurrogateGreeksOutput,
    latency_ms: float = 0.0,
    engine_used: str = "MaskedEGNO_Surrogate",
    metadata_dict: Mapping[str, Any] | None = None,
) -> bytes:
    """Serializes `BatchedSurrogateGreeksOutput` into an Apache Arrow IPC stream response in `float64`.

    Serializes:
      - `product_value`: `(B,)`
      - `stopping_probs`: `(B, M_max)`
      - `survival_probs`: `(B, M_max)`
      - `coupon_probs`: `(B, M_max)`
      - `put_loss`: `(B,)`
      - `delta`: `(B, D_max)`
      - `vega_atm`: `(B, D_max, 2)`
      - `vega_surface_feats`: `(B, D_max, 7)`
      - `rho_df`: `(B, M_max)`
      - `cega`: `(B, D_max, D_max)`
      - `cash_gamma_euler`: `(B,)`
    """
    pv_np = _to_contiguous_f64_np(batch_out.product_value, name="product_value")
    stop_np = _to_contiguous_f64_np(batch_out.stopping_probs, name="stopping_probs")
    surv_np = _to_contiguous_f64_np(batch_out.survival_probs, name="survival_probs")
    cpn_np = _to_contiguous_f64_np(batch_out.coupon_probs, name="coupon_probs")
    put_np = _to_contiguous_f64_np(batch_out.put_loss, name="put_loss")
    delta_np = _to_contiguous_f64_np(batch_out.delta, name="delta")
    vega_atm_np = _to_contiguous_f64_np(batch_out.vega_atm, name="vega_atm")
    vega_surf_np = _to_contiguous_f64_np(batch_out.vega_surface_feats, name="vega_surface_feats")
    rho_df_np = _to_contiguous_f64_np(batch_out.rho_df, name="rho_df")
    cega_np = _to_contiguous_f64_np(batch_out.cega, name="cega")

    batch_size = int(pv_np.shape[0])
    d_max = int(delta_np.shape[1])
    m_max = int(stop_np.shape[1])

    has_cash_gamma = batch_out.cash_gamma_euler is not None
    if batch_out.cash_gamma_euler is not None:
        cg_np = _to_contiguous_f64_np(batch_out.cash_gamma_euler, name="cash_gamma_euler")
    else:
        cg_np = np.zeros(batch_size, dtype=np.float64)

    cols = [
        _make_1d_f64_column(pv_np, batch_size),
        _make_fixed_size_f64_column(stop_np, batch_size, m_max),
        _make_fixed_size_f64_column(surv_np, batch_size, m_max),
        _make_fixed_size_f64_column(cpn_np, batch_size, m_max),
        _make_1d_f64_column(put_np, batch_size),
        _make_fixed_size_f64_column(delta_np, batch_size, d_max),
        _make_fixed_size_f64_column(vega_atm_np, batch_size, d_max * 2),
        _make_fixed_size_f64_column(vega_surf_np, batch_size, d_max * 7),
        _make_fixed_size_f64_column(rho_df_np, batch_size, m_max),
        _make_fixed_size_f64_column(cega_np, batch_size, d_max * d_max),
        _make_1d_f64_column(cg_np, batch_size),
    ]
    names = [
        "product_value",
        "stopping_probs",
        "survival_probs",
        "coupon_probs",
        "put_loss",
        "delta",
        "vega_atm",
        "vega_surface_feats",
        "rho_df",
        "cega",
        "cash_gamma_euler",
    ]

    user_meta = dict(metadata_dict) if metadata_dict is not None else {}
    header_obj: dict[str, Any] = {
        **user_meta,
        "status": "done",
        "batch_size": batch_size,
        "d_max": d_max,
        "m_max": m_max,
        "latency_ms": float(latency_ms),
        "engine_used": str(engine_used),
        "has_cash_gamma_euler": bool(has_cash_gamma),
        "metadata": user_meta,
    }
    header_bytes = orjson.dumps(header_obj, option=orjson.OPT_SERIALIZE_NUMPY)

    batch = pa.RecordBatch.from_arrays(cols, names=names)
    schema_with_meta = batch.schema.with_metadata(
        {
            EQ_AUTOCALL_HEADER_KEY: header_bytes,
            HEADER_METADATA_KEY: header_bytes,
            b"status": b"done",
            b"engine_used": str(engine_used).encode("utf-8"),
            b"latency_ms": f"{float(latency_ms):.4f}".encode("utf-8"),
        }
    )
    batch = batch.replace_schema_metadata(schema_with_meta.metadata)

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    res_bytes: bytes = sink.getvalue().to_pybytes()
    return res_bytes


def decode_eq_autocall_arrow_response(body: bytes) -> dict[str, Any]:
    """Decodes an Apache Arrow IPC response back into `np.float64` arrays and metadata."""
    if not body:
        raise ValueError("Empty Arrow IPC response payload.")

    try:
        reader = pa.ipc.open_stream(body)
        table = reader.read_all()
    except Exception as exc:
        raise ValueError(f"Invalid Apache Arrow IPC response stream: {exc}") from exc

    raw_meta = reader.schema.metadata or {}
    header_raw_bytes = raw_meta.get(EQ_AUTOCALL_HEADER_KEY) or raw_meta.get(HEADER_METADATA_KEY)
    header_dict: dict[str, Any] = {}
    if header_raw_bytes is not None:
        try:
            loaded = orjson.loads(header_raw_bytes)
            if isinstance(loaded, dict):
                header_dict = loaded
        except Exception as exc:
            raise ValueError(f"Corrupt JSON in Arrow response header metadata: {exc}") from exc

    batch_size = int(table.num_rows)
    d_max = int(header_dict.get("d_max", D_MAX_ARROW))
    m_max = int(header_dict.get("m_max", M_MAX_ARROW))
    latency_ms = float(header_dict.get("latency_ms", 0.0))
    engine_used = str(header_dict.get("engine_used", "MaskedEGNO_Surrogate"))
    status = str(header_dict.get("status", "done"))

    pv_np = _extract_column_f64_np(table, "product_value", (batch_size,))
    stop_np = _extract_column_f64_np(table, "stopping_probs", (batch_size, m_max))
    surv_np = _extract_column_f64_np(table, "survival_probs", (batch_size, m_max))
    cpn_np = _extract_column_f64_np(table, "coupon_probs", (batch_size, m_max))
    put_np = _extract_column_f64_np(table, "put_loss", (batch_size,))
    delta_np = _extract_column_f64_np(table, "delta", (batch_size, d_max))
    vega_atm_np = _extract_column_f64_np(table, "vega_atm", (batch_size, d_max, 2))
    rho_df_np = _extract_column_f64_np(table, "rho_df", (batch_size, m_max))
    cega_np = _extract_column_f64_np(table, "cega", (batch_size, d_max, d_max))
    cg_np = _extract_column_f64_np(table, "cash_gamma_euler", (batch_size,))

    out: dict[str, Any] = {
        "status": status,
        "batch_size": batch_size,
        "d_max": d_max,
        "m_max": m_max,
        "latency_ms": latency_ms,
        "engine_used": engine_used,
        "product_value": pv_np,
        "stopping_probs": stop_np,
        "survival_probs": surv_np,
        "coupon_probs": cpn_np,
        "put_loss": put_np,
        "delta": delta_np,
        "vega_atm": vega_atm_np,
        "rho_df": rho_df_np,
        "cega": cega_np,
        "cash_gamma_euler": cg_np,
        "metadata": header_dict.get("metadata", header_dict),
    }
    if "vega_surface_feats" in table.column_names:
        out["vega_surface_feats"] = _extract_column_f64_np(
            table, "vega_surface_feats", (batch_size, d_max, 7)
        )
    return out


__all__ = [
    "ARROW_MIME",
    "EQ_AUTOCALL_HEADER_KEY",
    "HEADER_METADATA_KEY",
    "D_MAX_ARROW",
    "M_MAX_ARROW",
    "MAX_BATCH_TRADES",
    "DecodedEQAutocallArrowBatch",
    "encode_eq_autocall_arrow_request",
    "decode_eq_autocall_arrow_request",
    "encode_eq_autocall_arrow_response",
    "decode_eq_autocall_arrow_response",
]
