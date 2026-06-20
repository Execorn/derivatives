"""
src/api package — Rough Heston FNO REST API.
"""
from __future__ import annotations

__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from api.server import app
        return app
    raise AttributeError(name)
