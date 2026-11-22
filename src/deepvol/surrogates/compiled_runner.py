"""Thread-safe static-shape Inductor + shared-pool CUDAGraph surrogate runner."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Dict, Optional, Tuple, cast

import torch

from deepvol.surrogates.eq_autocall_egno import MaskedEQAutocallSurrogate

# Drop B=1024 from CUDAGraph buckets (compute-bound & Arrow B=1024 uses autograd); keeps VRAM <= 150 MB.
BATCH_BUCKETS: Tuple[int, ...] = (1, 16, 64, 256)
GEOM_BUCKETS: Tuple[Tuple[int, int], ...] = ((5, 1), (5, 12), (8, 12), (8, 48), (8, 64))

_ResidualsFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
_BucketKey = Tuple[int, int, int, torch.dtype, torch.device]


@dataclass
class _CapturedBucketGraph:
    graph: Any
    in_node: torch.Tensor
    in_edge: torch.Tensor
    in_amask: torch.Tensor
    in_sched: torch.Tensor
    in_smask: torch.Tensor
    in_glob: torch.Tensor
    out_dhaz: torch.Tensor
    out_dcpn: torch.Tensor
    out_dput: torch.Tensor


class CompiledSurrogateRunner:
    """Thread-safe static-shape Inductor + shared-pool CUDAGraph runner with mandatory .clone() safety."""

    def __init__(
        self,
        model: MaskedEQAutocallSurrogate,
        use_tf32: bool = True,
        enable_cudagraphs: bool = True,
        enable_inductor: bool = True,
    ) -> None:
        self.model = model
        self.model.eval()
        self.model.freeze_eval_buffers()
        self.use_tf32 = bool(use_tf32)
        self.enable_cudagraphs = bool(enable_cudagraphs)
        self.enable_inductor = bool(enable_inductor)
        self._lock = threading.Lock()
        self._graphs: Dict[_BucketKey, _CapturedBucketGraph] = {}
        self._graph_pool: Optional[Any] = None
        if self.use_tf32 and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
        if self.enable_inductor:
            self._residuals_fn: _ResidualsFn = cast(
                _ResidualsFn,
                torch.compile(self.model.forward_residuals, fullgraph=True),
            )
        else:
            self._residuals_fn = self.model.forward_residuals

    def invalidate_graphs(self) -> None:
        """Clear captured CUDAGraphs after weight reload or freeze_eval_buffers()."""
        with self._lock:
            self._graphs.clear()
            self._graph_pool = None

    def predict_residuals(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward residual prediction; thread-safe under self._lock with .clone() output."""
        if (
            not self.enable_cudagraphs
            or not node_feats.is_cuda
            or torch.is_grad_enabled()
        ):
            out = self.model.forward_residuals(
                node_feats, edge_feats, asset_mask, sched_feats, sched_mask, global_feats
            )
            return out[0], out[1], out[2]
        b_real = int(node_feats.shape[0])
        d_real = int(node_feats.shape[1])
        m_real = int(sched_feats.shape[1])
        b_bkt = next((b for b in BATCH_BUCKETS if b >= b_real), None)
        geom = next(((d, m) for (d, m) in GEOM_BUCKETS if d >= d_real and m >= m_real), None)
        if b_bkt is None or geom is None:
            out = self._residuals_fn(
                node_feats, edge_feats, asset_mask, sched_feats, sched_mask, global_feats
            )
            return out[0], out[1], out[2]
        key: _BucketKey = (b_bkt, geom[0], geom[1], node_feats.dtype, node_feats.device)
        with self._lock:
            cap = self._graphs.get(key)
            if cap is None:
                cap = self._capture_bucket(key)
                self._graphs[key] = cap
            self._copy_padded_into_static(
                cap, node_feats, edge_feats, asset_mask, sched_feats, sched_mask, global_feats
            )
            cap.graph.replay()
            # Clone inside lock so subsequent replays never overwrite returned static buffers.
            return (
                cap.out_dhaz[:b_real, :m_real].clone(),
                cap.out_dcpn[:b_real, :m_real].clone(),
                cap.out_dput[:b_real].clone(),
            )

    def _capture_bucket(self, key: _BucketKey) -> _CapturedBucketGraph:
        b_bkt, d_bkt, m_bkt, dtype, device = key
        cuda_mod: Any = torch.cuda
        if self._graph_pool is None:
            self._graph_pool = cuda_mod.graph_pool_handle()
        in_node = torch.zeros((b_bkt, d_bkt, self.model.d_node), dtype=dtype, device=device)
        in_edge = torch.zeros((b_bkt, d_bkt, d_bkt, self.model.d_edge), dtype=dtype, device=device)
        in_amask = torch.zeros((b_bkt, d_bkt), dtype=dtype, device=device)
        in_sched = torch.zeros((b_bkt, m_bkt, self.model.d_sched), dtype=dtype, device=device)
        in_smask = torch.zeros((b_bkt, m_bkt), dtype=dtype, device=device)
        in_glob = torch.zeros((b_bkt, self.model.d_global), dtype=dtype, device=device)
        in_amask[:, 0] = 1.0
        in_smask[:, 0] = 1.0
        stream = cuda_mod.Stream(device=device)
        stream.wait_stream(cuda_mod.current_stream(device=device))
        with cuda_mod.stream(stream), torch.no_grad():
            for _ in range(3):
                _ = self._residuals_fn(
                    in_node, in_edge, in_amask, in_sched, in_smask, in_glob
                )
        cuda_mod.current_stream(device=device).wait_stream(stream)
        graph = cuda_mod.CUDAGraph()
        with cuda_mod.graph(graph, pool=self._graph_pool), torch.no_grad():
            out_dhaz, out_dcpn, out_dput = self._residuals_fn(
                in_node, in_edge, in_amask, in_sched, in_smask, in_glob
            )
        return _CapturedBucketGraph(
            graph=graph,
            in_node=in_node,
            in_edge=in_edge,
            in_amask=in_amask,
            in_sched=in_sched,
            in_smask=in_smask,
            in_glob=in_glob,
            out_dhaz=out_dhaz,
            out_dcpn=out_dcpn,
            out_dput=out_dput,
        )

    @staticmethod
    def _copy_padded_into_static(
        cap: _CapturedBucketGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> None:
        b_r, d_r, m_r = int(node_feats.shape[0]), int(node_feats.shape[1]), int(sched_feats.shape[1])
        cap.in_node.zero_()
        cap.in_edge.zero_()
        cap.in_amask.zero_()
        cap.in_sched.zero_()
        cap.in_smask.zero_()
        cap.in_glob.zero_()
        cap.in_node[:b_r, :d_r].copy_(node_feats)
        cap.in_edge[:b_r, :d_r, :d_r].copy_(edge_feats)
        cap.in_amask[:b_r, :d_r].copy_(asset_mask)
        cap.in_sched[:b_r, :m_r].copy_(sched_feats)
        cap.in_smask[:b_r, :m_r].copy_(sched_mask)
        cap.in_glob[:b_r].copy_(global_feats)
        # Keep padded batch rows consistent with capture-time non-empty mask invariants.
        if b_r < int(cap.in_amask.shape[0]):
            cap.in_amask[b_r:, 0] = 1.0
            cap.in_smask[b_r:, 0] = 1.0


__all__ = [
    "BATCH_BUCKETS",
    "GEOM_BUCKETS",
    "CompiledSurrogateRunner",
]
