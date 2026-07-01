"""Shared CLI override helpers for ACMMM experiment scripts."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any, List


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def add_shared_experiment_args(parser: ArgumentParser) -> ArgumentParser:
    """Add optional experiment overrides shared by train/test entrypoints."""
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--branch_mode",
        type=str,
        default=None,
        choices=["both", "traditional_only", "dzz_only"],
        help="Override TopoMamba scan branch mode",
    )
    parser.add_argument(
        "--fusion_method",
        type=str,
        default=None,
        choices=[
            "hsic",
            "hsic_gate",
            "add",
            "sum",
            "eltwise",
            "conv1x1",
            "conv",
            "se",
            "se_fuse",
            "se_gate",
            "cos",
            "cosine",
            "bftt3d",
            "bftt3d_cvpr2024",
            "i2pmae",
            "i2pmae_cvpr2023",
            "dtrg",
            "dtrg_tip2022",
            "hsic_opt",
            "optimized_hsic",
        ],
        help="Override dual-branch fusion method",
    )
    parser.add_argument("--hsic_proj_dim", type=int, default=None, help="Override HSIC projection dimension")
    parser.add_argument("--hsic_temperature", type=float, default=None, help="Override HSIC gate temperature")
    parser.add_argument("--hsic_residual", type=float, default=None, help="Override HSIC residual shortcut weight")
    parser.add_argument("--enable_cache", type=str2bool, default=None, help="Enable or disable ScanCache")
    parser.add_argument("--enable_gating", type=str2bool, default=None, help="Enable or disable fusion gating")
    parser.add_argument("--load_pretrained", type=str2bool, default=None, help="Enable or disable model pretraining")
    parser.add_argument("--topology_loss_enabled", type=str2bool, default=None, help="Enable or disable topology-aware loss")
    parser.add_argument("--postprocess_enabled", type=str2bool, default=None, help="Enable or disable post-processing")
    return parser


def apply_shared_experiment_overrides(config: Any, args: Namespace) -> List[str]:
    """Apply shared overrides to a config class/object and return change notes."""
    changed: List[str] = []
    model_cfg = getattr(config, "model_config", None)
    if model_cfg is None:
        model_cfg = {}
        config.model_config = model_cfg

    top_level = {
        "seed": getattr(args, "seed", None),
        "topology_loss_enabled": getattr(args, "topology_loss_enabled", None),
        "postprocess_enabled": getattr(args, "postprocess_enabled", None),
    }
    for key, value in top_level.items():
        if value is not None:
            setattr(config, key, value)
            changed.append(f"{key}={value}")

    model_keys = [
        "branch_mode",
        "fusion_method",
        "hsic_proj_dim",
        "hsic_temperature",
        "hsic_residual",
        "enable_cache",
        "enable_gating",
        "load_pretrained",
    ]
    for key in model_keys:
        value = getattr(args, key, None)
        if value is not None:
            model_cfg[key] = value
            changed.append(f"model_config.{key}={value}")

    criterion = getattr(config, "criterion", None)
    if getattr(args, "topology_loss_enabled", None) is not None and hasattr(criterion, "enabled"):
        criterion.enabled = bool(args.topology_loss_enabled)
        changed.append(f"criterion.enabled={criterion.enabled}")

    return changed


def print_shared_override_report(changed: List[str]) -> None:
    if not changed:
        return
    print("ACMMM shared overrides:")
    for item in changed:
        print(f"  - {item}")
