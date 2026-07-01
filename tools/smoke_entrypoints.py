"""Early --smoke dispatch helpers for train/test entrypoints."""

from __future__ import annotations

import importlib
import os
import sys
import types


def install_smoke_import_stubs() -> None:
    """Install tiny stubs for optional modules not needed by smoke checks."""
    sys.modules.setdefault("SimpleITK", types.ModuleType("SimpleITK"))
    if "medpy" not in sys.modules:
        medpy = types.ModuleType("medpy")
        metric = types.ModuleType("medpy.metric")
        binary = types.SimpleNamespace(dc=lambda *_args, **_kwargs: 0.0, hd95=lambda *_args, **_kwargs: 0.0)
        metric.binary = binary
        medpy.metric = metric
        sys.modules["medpy"] = medpy
        sys.modules["medpy.metric"] = metric


def configure_smoke_environment_defaults() -> None:
    """Keep entrypoint smoke checks lightweight unless explicitly overridden."""
    os.environ.setdefault("TOPO_SMOKE_FAST", "1")
    os.environ.setdefault("TOPO_SMOKE_DEVICE", "cpu")


def run_smoke_from_config(config_module: str, script_name: str) -> None:
    """Load a config and run shared smoke checks, then exit the process."""
    configure_smoke_environment_defaults()
    install_smoke_import_stubs()
    config = importlib.import_module(config_module).setting_config
    from tools.smoke_topomamba import smoke_entrypoint

    smoke_entrypoint(config, script_name)
    sys.exit(0)
