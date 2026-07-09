"""Lightweight typing aliases used across modules."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

JSONDict = Dict[str, Any]
JSONList = List[Any]
JSONMapping = Mapping[str, Any]
JSONSequence = Sequence[Any]

__all__ = ["JSONDict", "JSONList", "JSONMapping", "JSONSequence"]
