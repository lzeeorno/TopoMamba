#!/usr/bin/env python
"""Lightweight Synapse 3D model registry.

This module intentionally avoids importing torch so the legacy Synapse
entrypoints can decide whether to route into the 3D pipeline before loading the
heavy 2D training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Synapse3DModelSpec:
    name: str
    module: str
    config_key: str


_REGISTRY: dict[str, Synapse3DModelSpec] = {
    "TopoMamba_3D_t": Synapse3DModelSpec(
        name="TopoMamba_3D_t",
        module="models.TopoMamba_3D",
        config_key="topomamba_3d_t_config",
    ),
    "topomamba_3d_t": Synapse3DModelSpec(
        name="TopoMamba_3D_t",
        module="models.TopoMamba_3D",
        config_key="topomamba_3d_t_config",
    ),
}


def list_synapse3d_models() -> list[str]:
    return sorted(_REGISTRY)


def is_synapse3d_model(model_name: str | None) -> bool:
    return bool(model_name) and str(model_name) in _REGISTRY


def require_synapse3d_model(model_name: str) -> Synapse3DModelSpec:
    try:
        return _REGISTRY[str(model_name)]
    except KeyError as exc:
        supported = ", ".join(list_synapse3d_models())
        raise ValueError(f"Unsupported Synapse 3D model {model_name!r}. Registered models: {supported}") from exc


def canonical_synapse3d_model_name(model_name: str) -> str:
    return require_synapse3d_model(model_name).name


def synapse3d_config_key(model_name: str) -> str:
    return require_synapse3d_model(model_name).config_key


def create_synapse3d_model(model_name: str, **kwargs):
    spec = require_synapse3d_model(model_name)
    module = import_module(spec.module)
    return module.create_model(spec.name, **kwargs)
