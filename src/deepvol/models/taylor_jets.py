"""taylor_jets.py — Second-order Taylor jets along scalar log-strike k with forward tangents.

Propagates truncated Taylor triples (value, d/dk, d2/dk2) and P first-order forward tangents
through analytic operations and stacked MLP GEMMs within a single torch.compile region.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Union
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class Jet(NamedTuple):
    """Container for 2nd-order Taylor jet along scalar k and P optional forward tangents."""

    v: Tensor
    d: Tensor
    dd: Tensor
    t: Optional[Tensor] = None


def const(x: Tensor, P: Optional[int] = None) -> Jet:
    """Wraps tensor x as a constant Jet with zero derivatives and sparse None tangents unless P > 0 is given."""
    z = torch.zeros_like(x)
    t = (
        torch.zeros(x.shape[0], P, dtype=x.dtype, device=x.device)
        if (P is not None and P > 0)
        else None
    )
    return Jet(v=x, d=z, dd=z, t=t)


def seed_k(k: Tensor, P: Optional[int] = None) -> Jet:
    """Seeds a Jet along scalar log-strike direction k (d/dk = 1, d2/dk2 = 0, sparse t = None unless P > 0)."""
    t = (
        torch.zeros(k.shape[0], P, dtype=k.dtype, device=k.device)
        if (P is not None and P > 0)
        else None
    )
    return Jet(
        v=k,
        d=torch.ones_like(k),
        dd=torch.zeros_like(k),
        t=t,
    )


def seed_tangent(x: Tensor, j: int, P: int, eye: Optional[Tensor] = None) -> Jet:
    """Seeds a Jet with value x and unit forward tangent along parameter index j."""
    z = torch.zeros_like(x)
    if eye is not None:
        t = x.unsqueeze(-1) * 0.0 + eye[j]
    else:
        t = torch.zeros(x.shape[0], P, dtype=x.dtype, device=x.device)
        t[:, j] = 1.0
    return Jet(v=x, d=z, dd=z, t=t)


def _add_t(ta: Optional[Tensor], tb: Optional[Tensor]) -> Optional[Tensor]:
    """Adds two optional tangent tensors, preserving None sparsity."""
    if ta is None:
        return tb
    if tb is None:
        return ta
    return ta + tb


def _scale_t(t: Optional[Tensor], c: Union[Tensor, float]) -> Optional[Tensor]:
    """Scales an optional tangent tensor by a scalar or column vector, preserving None sparsity."""
    if t is None:
        return None
    if isinstance(c, Tensor) and c.dim() >= 1:
        return t * c.unsqueeze(-1)
    return t * c


def add(a: Jet, b: Jet) -> Jet:
    """Linear addition of two Jets."""
    return Jet(v=a.v + b.v, d=a.d + b.d, dd=a.dd + b.dd, t=_add_t(a.t, b.t))


def scale(a: Jet, c: Union[Tensor, float]) -> Jet:
    """Scales a Jet by a scalar or broadcastable tensor."""
    return Jet(v=a.v * c, d=a.d * c, dd=a.dd * c, t=_scale_t(a.t, c))


def mul(a: Jet, b: Jet) -> Jet:
    """Product of two Jets using the Leibniz rule."""
    v = a.v * b.v
    d = a.d * b.v + a.v * b.d
    dd = a.dd * b.v + 2.0 * a.d * b.d + a.v * b.dd
    t = _add_t(_scale_t(a.t, b.v), _scale_t(b.t, a.v))
    return Jet(v=v, d=d, dd=dd, t=t)


def unary(a: Jet, f: Tensor, f1: Tensor, f2: Tensor) -> Jet:
    """Univariate chain rule on Jet a given f(a.v), f'(a.v), and f''(a.v)."""
    d = f1 * a.d
    dd = f2 * (a.d * a.d) + f1 * a.dd
    t = _scale_t(a.t, f1)
    return Jet(v=f, d=d, dd=dd, t=t)


def sqrt(a: Jet) -> Jet:
    """Square root Jet."""
    r = torch.sqrt(a.v)
    f1 = 0.5 / r
    f2 = -0.25 / (r * a.v)
    return unary(a, r, f1, f2)


def recip(a: Jet) -> Jet:
    """Reciprocal Jet (1 / a)."""
    r = 1.0 / a.v
    f1 = -r * r
    f2 = 2.0 * r * r * r
    return unary(a, r, f1, f2)


def exp(a: Jet) -> Jet:
    """Exponential Jet."""
    e = torch.exp(a.v)
    return unary(a, e, e, e)


def tanh(a: Jet) -> Jet:
    """Hyperbolic tangent Jet."""
    t = torch.tanh(a.v)
    u = 1.0 - t * t
    return unary(a, t, u, -2.0 * t * u)


def silu(a: Jet) -> Jet:
    """SiLU (Swish-1) activation Jet."""
    s = torch.sigmoid(a.v)
    f = a.v * s
    f1 = s + a.v * s * (1.0 - s)
    f2 = s * (1.0 - s) * (2.0 + a.v * (1.0 - 2.0 * s))
    return unary(a, f, f1, f2)


def softfloor(a: Jet, lo: float, beta: float) -> Jet:
    """Smooth lower bound floor: lo + softplus_beta(a - lo)."""
    diff = a.v - lo
    s = torch.sigmoid(beta * diff)
    f = lo + F.softplus(diff, beta=beta)
    f1 = s
    f2 = beta * s * (1.0 - s)
    return unary(a, f, f1, f2)


def mlp_jet(layers: Sequence[nn.Module], cols: Sequence[Jet]) -> Jet:
    """Propagates a sequence of feature Jets through an MLP using stacked GEMMs."""
    B = cols[0].v.shape[0]
    P = 0
    for c in cols:
        if c.t is not None:
            P = c.t.shape[1]
            break

    v = torch.stack([c.v for c in cols], dim=-1)
    d = torch.stack([c.d for c in cols], dim=-1)
    dd = torch.stack([c.dd for c in cols], dim=-1)
    if P > 0:
        zt = torch.zeros((B, P), dtype=v.dtype, device=v.device)
        t = torch.stack([c.t if c.t is not None else zt for c in cols], dim=-1)
        t_flat = t.reshape(B * P, -1)
        H = torch.cat([v, d, dd, t_flat], dim=0).float()
    else:
        H = torch.cat([v, d, dd], dim=0).float()

    for layer in layers:
        if isinstance(layer, nn.Linear):
            H = H @ layer.weight.t()
            if layer.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                H = torch.cat([H[:B] + layer.bias, H[B:]], dim=0)
        elif isinstance(layer, nn.SiLU):
            t_in = H[3 * B :].reshape(B, P, -1).transpose(1, 2) if P > 0 else None
            a = silu(Jet(v=H[:B], d=H[B : 2 * B], dd=H[2 * B : 3 * B], t=t_in))
            if P > 0 and a.t is not None:
                H = torch.cat([a.v, a.d, a.dd, a.t.transpose(1, 2).reshape(B * P, -1)], dim=0)
            else:
                H = torch.cat([a.v, a.d, a.dd], dim=0)

    H_64 = H.double()
    v_out = H_64[:B, 0]
    d_out = H_64[B : 2 * B, 0]
    dd_out = H_64[2 * B : 3 * B, 0]
    t_out: Optional[Tensor] = H_64[3 * B :, 0].reshape(B, P) if P > 0 else None

    return Jet(v=v_out, d=d_out, dd=dd_out, t=t_out)
