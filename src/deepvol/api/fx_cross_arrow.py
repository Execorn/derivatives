"""fx_cross_arrow.py — Apache Arrow IPC binary stream transport codec for triangulated FX cross options.

Provides high-throughput binary serialization and deserialization for columnar batches
of European cross options (up to 10,000 contracts across 64 market states), bypassing
JSON parsing overhead and eliminating Python heap list allocations.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Final, List, Mapping, Optional, Union

import numpy as np
from numpy.typing import NDArray
import orjson
import pyarrow as pa  # type: ignore[import-untyped]
from pydantic import TypeAdapter

from deepvol.api.schemas_fx_cross import (
    ExecutionOptions,
    MarketStateColumnar,
    TriangulatedFXOptionColumnarBatchRequest,
    CCY_PATTERN,
    MAX_CONTRACTS,
    MAX_STATES,
    PAIR_PATTERN,
)

ARROW_MIME: Final[str] = "application/vnd.apache.arrow.stream"
HEADER_METADATA_KEY: Final[bytes] = b"deepvol.header"

# Cached module-level type adapters for low-latency validation
_states_adapter: TypeAdapter[List[MarketStateColumnar]] = TypeAdapter(
    List[MarketStateColumnar]
)
_options_adapter: TypeAdapter[ExecutionOptions] = TypeAdapter(ExecutionOptions)


@dataclass(frozen=True)
class DecodedBatch:
    """Pre-validated batch extracted into contiguous NumPy arrays for engine consumption."""

    states: List[MarketStateColumnar]
    state_index: NDArray[np.int64]
    strike: NDArray[np.float64]
    parity: NDArray[np.float64]
    notional: Optional[NDArray[np.float64]]
    options: ExecutionOptions
    identifier: str = ""
    payment_currency: str = ""


def decode_json_batch(body: bytes) -> DecodedBatch:
    """Decode and validate a JSON columnar batch request into a DecodedBatch.

    Uses Pydantic V2 model validation (Rust-backed item constraints) for
    TriangulatedFXOptionColumnarBatchRequest, converting arrays to NumPy.
    """
    req = TriangulatedFXOptionColumnarBatchRequest.model_validate_json(body)
    return DecodedBatch(
        states=req.states,
        state_index=np.asarray(req.state_index, dtype=np.int64),
        strike=np.asarray(req.strike, dtype=np.float64),
        parity=np.asarray(req.parity_sign, dtype=np.float64),
        notional=np.asarray(req.notional, dtype=np.float64)
        if req.notional is not None
        else None,
        options=req.execution_options,
        identifier=req.identifier,
        payment_currency=req.payment_currency,
    )


def decode_arrow_batch(body: bytes) -> DecodedBatch:
    """Decode and validate an Apache Arrow IPC stream batch request.

    Extracts JSON metadata header 'deepvol.header' from schema metadata and validates
    market states via Pydantic TypeAdapter. Enforces vectorized NumPy checks on
    the contract arrays (state_index, strike, parity_sign, notional).
    """
    try:
        reader = pa.ipc.open_stream(body)
    except Exception as exc:
        raise ValueError(f"Invalid Arrow IPC stream: {exc}") from exc

    metadata = reader.schema.metadata
    if metadata is None or HEADER_METADATA_KEY not in metadata:
        raise ValueError("Missing 'deepvol.header' in Arrow schema metadata")

    try:
        header_raw: Any = orjson.loads(metadata[HEADER_METADATA_KEY])
    except Exception as exc:
        raise ValueError(f"Invalid JSON in 'deepvol.header': {exc}") from exc

    if not isinstance(header_raw, dict):
        raise ValueError("'deepvol.header' must be a JSON object")

    identifier = header_raw.get("identifier")
    if not isinstance(identifier, str) or not re.match(PAIR_PATTERN, identifier):
        raise ValueError(f"Invalid or missing identifier in header: {identifier}")

    payment_currency = header_raw.get("payment_currency")
    if not isinstance(payment_currency, str) or not re.match(
        CCY_PATTERN, payment_currency
    ):
        raise ValueError(
            f"Invalid or missing payment_currency in header: {payment_currency}"
        )

    raw_states = header_raw.get("states")
    if (
        not isinstance(raw_states, list)
        or len(raw_states) < 1
        or len(raw_states) > MAX_STATES
    ):
        raise ValueError(f"states must be a list of 1 to {MAX_STATES} elements")

    states = _states_adapter.validate_python(raw_states)

    # Cross-consistency checks across states and request identifier
    for st in states:
        md = st.market_data
        if identifier != md.leg1.base + md.leg2.quote:
            raise ValueError(
                f"state {st.state_id}: identifier != leg1 base + leg2 quote"
            )
        if payment_currency != md.leg2.quote:
            raise ValueError(
                f"state {st.state_id}: payment_currency must equal the cross quote ccy C"
            )
        if not (st.as_of_date < st.expiration_date <= st.settlement_date):
            raise ValueError(
                f"state {st.state_id}: require as_of_date < expiration_date <= settlement_date"
            )

    raw_opts = header_raw.get("execution_options")
    options = (
        _options_adapter.validate_python(raw_opts)
        if raw_opts is not None
        else ExecutionOptions()
    )

    try:
        table = reader.read_all()
    except Exception as exc:
        raise ValueError(f"Failed to read Arrow record batches: {exc}") from exc

    for req_col in ("state_index", "strike", "parity_sign"):
        if req_col not in table.column_names:
            raise ValueError(f"Missing required columnar array: '{req_col}'")

    state_index = table.column("state_index").to_numpy(zero_copy_only=False)
    strike = table.column("strike").to_numpy(zero_copy_only=False)
    parity_sign = table.column("parity_sign").to_numpy(zero_copy_only=False)
    notional = (
        table.column("notional").to_numpy(zero_copy_only=False)
        if "notional" in table.column_names
        else None
    )

    n_contracts = len(strike)
    if n_contracts < 1 or n_contracts > MAX_CONTRACTS:
        raise ValueError(
            f"Contract count {n_contracts} out of bounds [1, {MAX_CONTRACTS}]"
        )

    if (
        len(state_index) != n_contracts
        or len(parity_sign) != n_contracts
        or (notional is not None and len(notional) != n_contracts)
    ):
        raise ValueError("columnar arrays must have equal length")

    if not np.isfinite(strike).all():
        raise ValueError("strike contains non-finite values (NaN or Inf)")

    if not bool(((strike >= 1e-6) & (strike <= 1e9)).all()):
        raise ValueError("strike out of range [1e-6, 1e9]")

    n_states = len(states)
    if not bool(((state_index >= 0) & (state_index < n_states)).all()):
        raise ValueError(f"state_index out of range [0, {n_states})")

    if not bool(((parity_sign == 1) | (parity_sign == -1)).all()):
        raise ValueError("parity_sign must contain only +1 or -1")

    if notional is not None:
        if not np.isfinite(notional).all():
            raise ValueError("notional contains non-finite values (NaN or Inf)")

    return DecodedBatch(
        states=states,
        state_index=state_index
        if state_index.dtype == np.int64
        else state_index.astype(np.int64),
        strike=strike if strike.dtype == np.float64 else strike.astype(np.float64),
        parity=parity_sign
        if parity_sign.dtype == np.float64
        else parity_sign.astype(np.float64),
        notional=notional
        if (notional is None or notional.dtype == np.float64)
        else notional.astype(np.float64),
        options=options,
        identifier=identifier,
        payment_currency=payment_currency,
    )


_EMPTY_DICT_VALS: Final[pa.Array] = pa.array([""], type=pa.string())
_ZERO_INT8_BUF: Final[NDArray[np.int8]] = np.zeros(MAX_CONTRACTS, dtype=np.int8)
_ZERO_INT8_ARRAY: Final[pa.Array] = pa.Array.from_buffers(
    pa.int8(), MAX_CONTRACTS, [None, pa.py_buffer(_ZERO_INT8_BUF)]
)
_ZERO_BOOL_ARRAY: Final[pa.Array] = pa.array(
    np.zeros(MAX_CONTRACTS, dtype=bool), type=pa.bool_()
)


def encode_arrow_response(
    cols: Mapping[str, Any],
    meta: Mapping[str, str],
) -> bytes:
    """Encode columnar response arrays and metadata into an Apache Arrow IPC stream.

    Constructs a pyarrow.RecordBatch containing the columnar response fields as
    float64/bool/dictionary-encoded columns, with per-state fields and scalar metadata
    stored in the schema metadata.
    """
    arrow_arrays: List[Union[pa.Array, pa.ChunkedArray]] = []
    field_names: List[str] = []

    fb_mask = cols.get("fallback_mask")
    no_fallbacks = (
        isinstance(fb_mask, np.ndarray)
        and (fb_mask.dtype == np.bool_ or fb_mask.dtype == bool)
        and not bool(fb_mask.any())
    )

    for name, col in cols.items():
        field_names.append(name)
        if name == "fallback_reason":
            n_rows = len(col)
            if isinstance(col, (pa.Array, pa.ChunkedArray)):
                arrow_arrays.append(col)
            elif (no_fallbacks and n_rows <= MAX_CONTRACTS) or (
                isinstance(col, (list, tuple))
                and n_rows <= MAX_CONTRACTS
                and (n_rows == 0 or all(x == "" for x in col))
            ):
                idx_arr = _ZERO_INT8_ARRAY.slice(0, n_rows)
                arrow_arrays.append(
                    pa.DictionaryArray.from_arrays(idx_arr, _EMPTY_DICT_VALS)
                )
            else:
                arrow_arrays.append(pa.array(col, type=pa.string()).dictionary_encode())
        elif (
            isinstance(col, np.ndarray)
            and col.dtype == np.float64
            and col.flags.c_contiguous
        ):
            buf = pa.py_buffer(col)
            arrow_arrays.append(
                pa.Array.from_buffers(pa.float64(), len(col), [None, buf])
            )
        elif (
            isinstance(col, np.ndarray)
            and col.dtype == np.int64
            and col.flags.c_contiguous
        ):
            buf = pa.py_buffer(col)
            arrow_arrays.append(
                pa.Array.from_buffers(pa.int64(), len(col), [None, buf])
            )
        elif isinstance(col, np.ndarray) and (
            col.dtype == np.bool_ or col.dtype == bool
        ):
            if no_fallbacks and name == "fallback_mask" and len(col) <= MAX_CONTRACTS:
                arrow_arrays.append(_ZERO_BOOL_ARRAY.slice(0, len(col)))
            else:
                arrow_arrays.append(pa.array(col, type=pa.bool_()))
        elif np.issubdtype(col.dtype, np.floating):
            arrow_arrays.append(pa.array(col, type=pa.float64()))
        elif np.issubdtype(col.dtype, np.integer):
            arrow_arrays.append(pa.array(col, type=pa.int64()))
        else:
            arrow_arrays.append(pa.array(col))

    batch = pa.RecordBatch.from_arrays(arrow_arrays, names=field_names)
    meta_bytes = {k.encode("utf-8"): v.encode("utf-8") for k, v in meta.items()}
    schema_with_meta = batch.schema.with_metadata(meta_bytes)
    batch = batch.replace_schema_metadata(schema_with_meta.metadata)

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    res: bytes = sink.getvalue().to_pybytes()
    return res


def encode_json_response(
    cols: Mapping[str, Any],
    scalars: Mapping[str, object],
) -> bytes:
    """Encode columnar response arrays and scalar/per-state metadata into JSON.

    Uses orjson.dumps with orjson.OPT_SERIALIZE_NUMPY for direct C-level vector
    serialization of NumPy buffers without Python list/float object allocation.
    """
    merged: dict[str, object] = {}
    for k, v in scalars.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind in ("U", "S", "O"):
                merged[k] = v.tolist()
            else:
                merged[k] = np.ascontiguousarray(v) if not v.flags.c_contiguous else v
        else:
            merged[k] = v
    for k, v in cols.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind in ("U", "S", "O"):
                merged[k] = v.tolist()
            else:
                merged[k] = np.ascontiguousarray(v) if not v.flags.c_contiguous else v
        else:
            merged[k] = v
    return orjson.dumps(merged, option=orjson.OPT_SERIALIZE_NUMPY)
