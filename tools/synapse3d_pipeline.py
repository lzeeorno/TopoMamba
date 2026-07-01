#!/usr/bin/env python
"""Dedicated TopoMamba-3D Synapse train/test helpers.

The original Synapse entrypoints are 2D slice trainers.  This module keeps the
3D path separate while allowing ``train_synapse.py`` and ``test_synapse.py`` to
route ``TopoMamba_3D_t`` into a volumetric crop trainer and a volume evaluator.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import os
import random
import re
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib
from scipy.ndimage import zoom

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from datasets.synapse3d_dataset import Synapse3DCaseDataset, collate_single_case, sample_case_crop
from losses import CombinedSegTopologyLoss
from tools.synapse3d_registry import (
    canonical_synapse3d_model_name,
    create_synapse3d_model,
    is_synapse3d_model,
    synapse3d_config_key,
)
from tools.synapse3d_preprocess import load_cached_case, load_test_case_volume, prepare_synapse_case_cache, read_case_ids


SLICE_RE = re.compile(r"^(case\d+)_slice(\d+)$")
ORGAN_NAMES = [
    "Aorta",
    "Gallbladder",
    "Left_Kidney",
    "Right_Kidney",
    "Liver",
    "Pancreas",
    "Spleen",
    "Stomach",
]


class DiceCrossEntropyLoss(nn.Module):
    """Small CE+Dice loss that works for 5D ``(B,C,D,H,W)`` logits."""

    def __init__(self, num_classes: int, smooth: float = 1e-5) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.smooth = float(smooth)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        ce = self.ce(logits, target)
        prob = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target.clamp_min(0), num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 4, 1, 2, 3).to(dtype=prob.dtype, device=prob.device)
        dims = tuple(range(2, prob.dim()))
        intersection = torch.sum(prob * one_hot, dim=dims)
        denominator = torch.sum(prob + one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if self.num_classes > 1:
            dice = dice[:, 1:]
        return ce + (1.0 - dice.mean())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str = "auto") -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "" or value.lower() == "all":
        return default
    return int(value)


def _env_tuple(name: str, default: Sequence[int]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if not value:
        return tuple(int(v) for v in default)
    parts = value.replace(",", " ").split()
    return tuple(int(part) for part in parts)


def _env_3tuple_optional(name: str) -> tuple[int, int, int] | None:
    value = os.environ.get(name)
    if not value:
        return None
    parsed = tuple(int(part) for part in value.replace(",", " ").split())
    if len(parsed) != 3:
        raise ValueError(f"{name} must contain exactly 3 integers, got {value!r}")
    return parsed


def _env_case_ids(name: str) -> list[str]:
    value = os.environ.get(name, "")
    if not value:
        return []
    return [part.strip() for part in re.split(r"[,\s]+", value) if part.strip()]


def _env_axes(name: str, default: Sequence[int] = (0, 1, 2)) -> tuple[int, ...]:
    value = os.environ.get(name)
    if not value:
        return tuple(int(v) for v in default)
    if value.lower() in {"none", "off", "false", "0"}:
        return ()
    axes = tuple(int(part) for part in value.replace(",", " ").split())
    invalid = [axis for axis in axes if axis not in (0, 1, 2)]
    if invalid:
        raise ValueError(f"{name} only supports axes 0, 1, 2; got {invalid}")
    return axes


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _validate_synapse_mirror_tta_safety(mirror_tta: bool, mirror_axes: Sequence[int]) -> None:
    if not mirror_tta:
        return
    axes = tuple(dict.fromkeys(int(axis) for axis in mirror_axes))
    if set(axes) == {0, 1, 2} and not _env_bool("SYNAPSE3D_ALLOW_UNSAFE_MIRROR_TTA", False):
        raise ValueError(
            "Unsafe Synapse3D mirror TTA requested: all-axis mirror TTA is disabled by default because "
            "Synapse multi-organ labels include anatomy-specific classes such as Left_Kidney and Right_Kidney. "
            "The current implementation flips logits back but does not swap left/right class channels, which can "
            "collapse small-organ predictions. Use SYNAPSE3D_MIRROR_TTA=0 for validated testing, or set "
            "SYNAPSE3D_ALLOW_UNSAFE_MIRROR_TTA=1 only for an explicit ablation."
        )
    print(
        "⚠️  Synapse3D mirror TTA is experimental: multi-organ labels include side-specific anatomy. "
        "Use validation before reporting results, and avoid all-axis mirror unless channel swaps are implemented.",
        flush=True,
    )


def _as_3tuple(value: Sequence[int] | str | None, default: Sequence[int]) -> tuple[int, int, int]:
    source = default if value is None else value
    if isinstance(source, str):
        parts = source.replace(",", " ").split()
        parsed = tuple(int(part) for part in parts)
    else:
        parsed = tuple(int(part) for part in source)
    if len(parsed) != 3:
        raise ValueError(f"Expected a D,H,W tuple, got {source!r}")
    return parsed


def _default_cache_root() -> str:
    return os.environ.get("SYNAPSE3D_CACHE_ROOT", "data/Synapse/topomamba3d_nnunetlite")


def _dataloader_workers(threads: int) -> int:
    return max(0, int(threads))


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default, help=help_text)
    else:
        parser.add_argument(name, type=str2bool, default=default, help=help_text)


def _format_count(value: int) -> str:
    return f"{int(value):,} ({int(value) / 1e6:.2f}M)"


def _model_param_count(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def _print_cuda_report(device: torch.device) -> None:
    print("#----------GPU init----------#")
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    print(f"🔧 CUDA版本: {torch.version.cuda}")
    print(f"🔧 PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        current = torch.cuda.current_device()
        print(f"🚀 GPU设备: {torch.cuda.get_device_name(current)}")
        print(f"🚀 GPU数量: {torch.cuda.device_count()}")
        print(f"🚀 当前GPU内存: {torch.cuda.get_device_properties(current).total_memory / 1024**3:.1f} GB")
        print(f"🚀 3D运行设备: {device}")
    else:
        print("⚠️  CUDA不可用，TopoMamba-3D 将在CPU上运行。请检查CUDA驱动或环境。")
        print(f"🚀 3D运行设备: {device}")
    if device.type == "cpu" and torch.cuda.is_available():
        print("⚠️  检测到CUDA可用但当前选择CPU。若要使用GPU，请不要设置 SYNAPSE3D_DEVICE=cpu。")


def _print_pretrained_report(report: dict) -> None:
    if not report:
        print("[Pretrained Loading] no report available")
        return
    print("[Pretrained Loading - TopoMamba3D]")
    print(f" - path: {report.get('path')}")
    print(f" - loaded: {report.get('loaded')}")
    print(f" - backbone_params: {_format_count(int(report.get('backbone_params', report.get('model_params', 0))))}")
    print(f" - actually_loaded_params: {_format_count(int(report.get('matched_params', 0)))}")
    print(f" - source_weight_params: {_format_count(int(report.get('checkpoint_params', 0)))}")
    print(f" - mapped_keys: {report.get('matched_keys', 0)}/{report.get('checkpoint_keys', '?')}")
    ratio = float(report.get("matched_backbone_ratio", 0.0)) * 100.0
    print(f" - loading_ratio(params): {ratio:.2f}%")


def _write_train_config_txt(path: Path, args, model: torch.nn.Module, model_cfg: dict, groups_count: int, device: torch.device) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TopoMamba-3D Synapse Training Configuration\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: TopoMamba_3D_t\n")
        f.write(f"Device: {device}\n")
        f.write(f"Parameters: {_format_count(_model_param_count(model))}\n")
        f.write(f"Train cases: {groups_count}\n")
        f.write(f"Training mode: {getattr(args, 'training_mode', '3d_fullres')}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Steps per epoch: {args.steps_per_epoch}\n")
        f.write(f"Batch size: {getattr(args, 'batch_size', 1)} volumetric crops\n")
        f.write(f"Crop: D={args.crop_depth}, H/W={args.crop_size}\n")
        if hasattr(args, "planner_report"):
            f.write(f"Planner report: {json.dumps(args.planner_report, ensure_ascii=False)}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Scheduler: {getattr(args, 'scheduler', 'none')}\n")
        f.write(f"Save interval: {getattr(args, 'save_interval', 0)}\n")
        f.write(f"Print interval: {getattr(args, 'print_interval', 1)}\n")
        f.write(f"Resume: {getattr(args, 'resume', False)}\n")
        f.write(f"Preprocessed root: {getattr(args, 'preprocessed_root', '')}\n")
        f.write(f"Model config: {json.dumps(model_cfg, ensure_ascii=False)}\n")


def _foreground_dice_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)
        target = target.long()
        scores = []
        for cls in range(1, num_classes):
            pred_mask = pred == cls
            target_mask = target == cls
            denom = pred_mask.sum() + target_mask.sum()
            if denom.item() == 0:
                continue
            score = (2.0 * (pred_mask & target_mask).sum().float() / denom.float()).detach()
            scores.append(float(score.cpu()))
        return float(np.mean(scores)) if scores else 0.0


def _normalize_slice_for_png(image_slice: np.ndarray) -> np.ndarray:
    image_slice = image_slice.astype(np.float32, copy=False)
    if image_slice.size == 0:
        return image_slice
    lo, hi = np.percentile(image_slice, [1, 99])
    if hi <= lo:
        hi = float(image_slice.max()) if float(image_slice.max()) > lo else lo + 1.0
    return np.clip((image_slice - lo) / (hi - lo), 0.0, 1.0)


def save_volume_prediction_png(case: str, image: np.ndarray, label: np.ndarray, pred: np.ndarray, out_dir: Path) -> Path:
    foreground = np.where(label > 0)[0]
    slice_idx = int(np.median(foreground)) if foreground.size else image.shape[0] // 2
    image_slice = _normalize_slice_for_png(image[slice_idx])
    label_slice = label[slice_idx]
    pred_slice = pred[slice_idx]
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Input Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(image_slice, cmap="gray")
    axes[1].imshow(label_slice, cmap="tab20", alpha=0.65, vmin=0, vmax=max(1, int(label.max())))
    axes[1].set_title("Ground Truth", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    axes[2].imshow(image_slice, cmap="gray")
    axes[2].imshow(pred_slice, cmap="tab20", alpha=0.65, vmin=0, vmax=max(1, int(label.max()), int(pred.max())))
    axes[2].set_title("Prediction", fontsize=14, fontweight="bold")
    axes[2].axis("off")
    fig.suptitle(f"{case} - Slice {slice_idx:03d}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path = out_dir / f"{case}_slice{slice_idx:03d}_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


def _eval_config_node(node: ast.AST, context: dict):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return context[node.id]
    if isinstance(node, ast.List):
        return [_eval_config_node(item, context) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_config_node(item, context) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _eval_config_node(key, context): _eval_config_node(value, context)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_config_node(node.operand, context)
    if isinstance(node, ast.BinOp):
        left = _eval_config_node(node.left, context)
        right = _eval_config_node(node.right, context)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append(str(_eval_config_node(value.value, context)))
        return "".join(parts)
    if isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name == "range":
            return range(*[_eval_config_node(arg, context) for arg in node.args])
        if func_name == "tuple" and len(node.args) == 1:
            return tuple(_eval_config_node(node.args[0], context))
        if func_name == "list" and len(node.args) == 1:
            return list(_eval_config_node(node.args[0], context))
    raise ValueError(f"Unsupported config expression: {ast.dump(node)}")


def load_synapse_config_defaults(config_path: Optional[Path] = None) -> dict:
    """Read simple ``setting_config`` constants without importing heavy 2D deps."""
    path = config_path or (Path(__file__).resolve().parents[1] / "configs" / "config_setting_synapse.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    setting_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "setting_config"
    )
    context: dict = {}
    for stmt in setting_class.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        name = stmt.targets[0].id
        try:
            context[name] = _eval_config_node(stmt.value, context)
        except Exception:
            continue
    configured_network = str(context.get("network", "TopoMamba_2D_t"))
    network = configured_network if is_synapse3d_model(configured_network) else "TopoMamba_3D_t"
    model_cfg = dict(context.get(synapse3d_config_key(network), {}))
    model_cfg.setdefault("model_name", canonical_synapse3d_model_name(network))
    datasets_name = context.get("datasets_name", "synapse")
    work_dir = context.get("work_dir") if is_synapse3d_model(configured_network) else f"results/{network}_{datasets_name}/"
    test_weights_path = (
        context.get("test_weights_path")
        if is_synapse3d_model(configured_network)
        else f"results/{network}_{datasets_name}/checkpoints/best.pth"
    )
    epochs = int(context.get("synapse3d_epochs", 1000))
    target_patch = _as_3tuple(context.get("synapse3d_target_patch"), (128, 128, 128))
    patch_candidates = tuple(
        _as_3tuple(candidate, target_patch)
        for candidate in context.get("synapse3d_patch_candidates", (target_patch,))
    )
    return {
        "network": network,
        "datasets_name": datasets_name,
        "data_path": context.get("data_path", "data/Synapse/train_npz"),
        "list_dir": context.get("list_dir", "data/Synapse/lists/lists_Synapse"),
        "volume_path": context.get("volume_path", "data/Synapse/test_vol_h5"),
        "work_dir": work_dir or f"results/{network}_{datasets_name}/",
        "test_weights_path": test_weights_path or f"results/{network}_{datasets_name}/checkpoints/best.pth",
        "epochs": epochs,
        "batch_size": int(context.get("synapse3d_batch_size", 2)),
        "max_batch_size": int(context.get("synapse3d_max_batch_size", 4)),
        "training_mode": context.get("synapse3d_training_mode", "3d_fullres"),
        "planner": context.get("synapse3d_planner", "auto"),
        "target_patch": target_patch,
        "patch_candidates": patch_candidates,
        "steps_per_epoch": int(context.get("synapse3d_steps_per_epoch", 250)),
        "num_workers": int(context.get("num_workers", 1)),
        "seed": int(context.get("seed", 2050)),
        "resume": bool(context.get("resume_training", True)),
        "print_interval": int(context.get("print_interval", 20)),
        "val_interval": int(context.get("val_interval", 20)),
        "save_interval": int(context.get("save_interval", 100)),
        "lr": float(context.get("lr", 3e-4)),
        "weight_decay": float(context.get("weight_decay", 0.01)),
        "scheduler": context.get("sch", "CosineAnnealingLR"),
        "t_max": int(context.get("synapse3d_t_max", epochs)),
        "eta_min": float(context.get("eta_min", 6e-7)),
        "last_epoch": int(context.get("last_epoch", -1)),
        "gradient_clip_norm": float(context.get("gradient_clip_norm", 12.0)),
        "topology_loss_enabled": bool(context.get("topology_loss_enabled", True)),
        "topology_loss_weight": float(context.get("topology_loss_weight", 0.05)),
        "topology_focal_gamma": float(context.get("topology_focal_gamma", 2.0)),
        "topology_critical_weight": float(context.get("topology_critical_weight", 4.0)),
        "topology_loss_max_elements": int(context.get("topology_loss_max_elements", 65536)),
        "model_config": model_cfg,
    }


def default_synapse_network() -> str:
    return str(load_synapse_config_defaults().get("network", "TopoMamba_3D_t"))


def build_case_index(data_dir: Path, list_file: Path):
    groups = defaultdict(list)
    names = [line.strip() for line in list_file.read_text().splitlines() if line.strip()]
    for name in names:
        match = SLICE_RE.match(name)
        if not match:
            continue
        case, slice_idx = match.group(1), int(match.group(2))
        path = data_dir / f"{name}.npz"
        if path.exists():
            groups[case].append((slice_idx, path))
    groups = {case: sorted(items) for case, items in groups.items() if items}
    if not groups:
        raise RuntimeError(f"No Synapse train_npz slices found from {list_file} under {data_dir}")
    return groups


def _pad_or_crop_spatial(volume: np.ndarray, label: np.ndarray, size: int):
    _, height, width = volume.shape
    pad_h = max(0, size - height)
    pad_w = max(0, size - width)
    if pad_h or pad_w:
        before_h, after_h = pad_h // 2, pad_h - pad_h // 2
        before_w, after_w = pad_w // 2, pad_w - pad_w // 2
        volume = np.pad(volume, ((0, 0), (before_h, after_h), (before_w, after_w)), mode="constant")
        label = np.pad(label, ((0, 0), (before_h, after_h), (before_w, after_w)), mode="constant")
        _, height, width = volume.shape

    foreground = np.argwhere(label > 0)
    if foreground.size:
        center_h = int(np.median(foreground[:, 1]))
        center_w = int(np.median(foreground[:, 2]))
        jitter = max(1, size // 8)
        center_h += random.randint(-jitter, jitter)
        center_w += random.randint(-jitter, jitter)
    else:
        center_h = random.randint(size // 2, max(size // 2, height - size // 2))
        center_w = random.randint(size // 2, max(size // 2, width - size // 2))
    start_h = min(max(0, center_h - size // 2), height - size)
    start_w = min(max(0, center_w - size // 2), width - size)
    return volume[:, start_h:start_h + size, start_w:start_w + size], label[:, start_h:start_h + size, start_w:start_w + size]


def sample_synapse_crop(groups, depth: int, size: int, device: torch.device):
    case = random.choice(list(groups.keys()))
    items = groups[case]
    if len(items) >= depth:
        start = random.randint(0, len(items) - depth)
        selected = items[start:start + depth]
    else:
        selected = list(items)
        while len(selected) < depth:
            selected.append(items[-1])

    images, labels = [], []
    for _, path in selected:
        data = np.load(path)
        images.append(data["image"].astype(np.float32))
        labels.append(data["label"].astype(np.int64))
    image = normalize_ct_like(np.stack(images, axis=0))
    label = np.stack(labels, axis=0)
    image, label = _pad_or_crop_spatial(image, label, size)
    image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    label_t = torch.from_numpy(label).unsqueeze(0).to(device=device, dtype=torch.long)
    return case, image_t, label_t


def normalize_ct_like(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    if image.size and (float(image.max()) > 20.0 or float(image.min()) < -20.0):
        image = np.clip(image, -125.0, 275.0)
        image = (image + 125.0) / 400.0
    return image


def iter_legacy_test_cases(label_dir: Path, test_list_file: Path, limit: int | None):
    case_ids = read_case_ids(Path(test_list_file))
    if limit is not None:
        case_ids = case_ids[:limit]
    if not case_ids:
        raise RuntimeError(f"No Synapse test cases listed in {test_list_file}")
    for case_id in case_ids:
        image, label, path = load_test_case_volume(Path(label_dir), case_id)
        yield case_id, normalize_ct_like(image), label, path


def build_topomamba3d(model_cfg: dict, device: torch.device):
    cfg = dict(model_cfg)
    model_name = cfg.pop("model_name", "TopoMamba_3D_t")
    model = create_synapse3d_model(model_name, **cfg).to(device)
    return model, int(cfg.get("num_classes", 9))


def _build_criterion(num_classes: int, args) -> CombinedSegTopologyLoss:
    base = DiceCrossEntropyLoss(num_classes)
    return CombinedSegTopologyLoss(
        base,
        num_classes=num_classes,
        enabled=bool(args.topology_loss_enabled),
        topology_weight=float(args.topology_loss_weight),
        focal_gamma=float(args.topology_focal_gamma),
        critical_weight=float(args.topology_critical_weight),
        foreground_classes=tuple(range(1, num_classes)),
        max_elements=int(args.topology_loss_max_elements),
    )


def _is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _clear_probe_state(model: torch.nn.Module, device: torch.device) -> None:
    model.zero_grad(set_to_none=True)
    cache = getattr(model, "cache", None)
    if hasattr(cache, "_cache"):
        cache._cache.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _candidate_plan(args) -> list[tuple[tuple[int, int, int], int]]:
    target = _as_3tuple(getattr(args, "target_patch", None), (128, 128, 128))
    patches = [_as_3tuple(candidate, target) for candidate in getattr(args, "patch_candidates", (target,))]
    for patch in (target, *patches):
        if patch[1] != patch[2]:
            raise ValueError(f"Synapse3D training patches must have equal H/W, got {patch}")
    if target not in patches:
        patches.insert(0, target)
    default_batch = max(1, int(getattr(args, "default_batch_size", 2)))
    max_batch = max(default_batch, int(getattr(args, "max_batch_size", default_batch)))
    candidates: list[tuple[tuple[int, int, int], int]] = []
    for batch in range(max_batch, default_batch - 1, -1):
        candidates.append((target, batch))
    for patch in patches:
        if patch == target:
            continue
        candidates.append((patch, default_batch))
    final_patch = patches[-1]
    if default_batch > 1:
        candidates.append((final_patch, 1))

    unique: list[tuple[tuple[int, int, int], int]] = []
    seen = set()
    for item in candidates:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def _probe_training_candidate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    patch: tuple[int, int, int],
    batch_size: int,
) -> None:
    depth, height, width = patch
    x = torch.zeros((int(batch_size), 1, depth, height, width), device=device, dtype=torch.float32)
    target = torch.zeros((int(batch_size), depth, height, width), device=device, dtype=torch.long)
    if num_classes > 1:
        target[:, depth // 4: max(depth // 4 + 1, depth // 2), height // 4: max(height // 4 + 1, height // 2), width // 4: max(width // 4 + 1, width // 2)] = 1
    logits = model(x)
    loss = criterion(logits, target)
    loss.backward()
    del logits, loss, target, x


def _resolve_train_runtime_options(args) -> None:
    defaults = load_synapse_config_defaults()
    args.network = canonical_synapse3d_model_name(getattr(args, "network", None) or defaults["network"])
    args.training_mode = getattr(args, "training_mode", None) or defaults["training_mode"]
    args.planner = getattr(args, "planner", None) or defaults["planner"]
    args.target_patch = _as_3tuple(getattr(args, "target_patch", defaults["target_patch"]), defaults["target_patch"])
    patch_candidate_source = getattr(args, "patch_candidates", None) or defaults["patch_candidates"]
    args.patch_candidates = tuple(
        _as_3tuple(candidate, args.target_patch)
        for candidate in patch_candidate_source
    )
    for patch in (args.target_patch, *args.patch_candidates):
        if patch[1] != patch[2]:
            raise ValueError(f"Synapse3D training patches must have equal H/W, got {patch}")
    args.default_batch_size = int(getattr(args, "default_batch_size", None) or defaults["batch_size"])
    args.max_batch_size = int(getattr(args, "max_batch_size", None) or defaults["max_batch_size"])

    patch_env = _env_3tuple_optional("SYNAPSE3D_PATCH_SIZE")
    crop_depth_env = os.environ.get("SYNAPSE3D_CROP_DEPTH")
    crop_size_env = os.environ.get("SYNAPSE3D_CROP_SIZE")
    batch_env = os.environ.get("SYNAPSE3D_BATCH_SIZE")
    manual_crop = bool(patch_env or crop_depth_env or crop_size_env or getattr(args, "crop_depth", None) is not None or getattr(args, "crop_size", None) is not None)
    manual_batch = bool(batch_env or getattr(args, "batch_size", None) is not None)

    args.steps_per_epoch = _env_int("SYNAPSE3D_STEPS_PER_EPOCH", getattr(args, "steps_per_epoch", None))
    if args.steps_per_epoch is None:
        args.steps_per_epoch = int(defaults["steps_per_epoch"])

    if patch_env:
        args.crop_depth = int(patch_env[0])
        if patch_env[1] != patch_env[2]:
            raise ValueError("SYNAPSE3D_PATCH_SIZE for training must have equal H/W because the current crop sampler uses one crop_size")
        args.crop_size = int(patch_env[1])
    else:
        args.crop_depth = _env_int("SYNAPSE3D_CROP_DEPTH", getattr(args, "crop_depth", None))
        args.crop_size = _env_int("SYNAPSE3D_CROP_SIZE", getattr(args, "crop_size", None))
    args.batch_size = _env_int("SYNAPSE3D_BATCH_SIZE", getattr(args, "batch_size", None))

    args._manual_crop = manual_crop
    args._manual_batch = manual_batch
    args._manual_plan = manual_crop or manual_batch
    if args._manual_plan:
        if args.crop_depth is None:
            args.crop_depth = int(args.target_patch[0])
        if args.crop_size is None:
            args.crop_size = int(args.target_patch[1])
        if args.batch_size is None:
            args.batch_size = int(args.default_batch_size)


def _finalize_auto_train_plan(
    args,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict:
    planner_candidates = _candidate_plan(args)
    plan = {
        "training_mode": getattr(args, "training_mode", "3d_fullres"),
        "planner": getattr(args, "planner", "auto"),
        "manual": bool(getattr(args, "_manual_plan", False)),
        "planner_candidates": [
            {"patch": list(patch), "batch_size": int(batch)}
            for patch, batch in planner_candidates
        ],
        "attempts": [],
    }
    if getattr(args, "_manual_plan", False):
        patch = (int(args.crop_depth), int(args.crop_size), int(args.crop_size))
        plan.update({"selected_patch": list(patch), "batch_size": int(args.batch_size), "reason": "manual_override"})
        return plan

    if device.type != "cuda":
        patch = (32, 96, 96)
        args.crop_depth, args.crop_size, args.batch_size = patch[0], patch[1], 1
        plan.update({"selected_patch": list(patch), "batch_size": 1, "reason": "cpu_default"})
        return plan

    if str(getattr(args, "planner", "auto")).lower() not in {"auto", "cuda-auto"}:
        patch = _as_3tuple(getattr(args, "target_patch", None), (128, 128, 128))
        args.crop_depth, args.crop_size, args.batch_size = patch[0], patch[1], int(args.default_batch_size)
        plan.update({"selected_patch": list(patch), "batch_size": int(args.batch_size), "reason": "planner_disabled"})
        return plan

    previous_training = model.training
    model.train()
    for patch, batch in planner_candidates:
        try:
            _clear_probe_state(model, device)
            _probe_training_candidate(model, criterion, device, num_classes, patch, batch)
            _clear_probe_state(model, device)
            args.crop_depth, args.crop_size, args.batch_size = int(patch[0]), int(patch[1]), int(batch)
            plan["attempts"].append({"patch": list(patch), "batch_size": int(batch), "status": "ok"})
            plan.update({"selected_patch": list(patch), "batch_size": int(batch), "reason": "cuda_probe"})
            model.train(previous_training)
            return plan
        except RuntimeError as exc:
            _clear_probe_state(model, device)
            if not _is_cuda_oom(exc):
                model.train(previous_training)
                raise
            plan["attempts"].append({"patch": list(patch), "batch_size": int(batch), "status": "oom"})
            print(f"[synapse3d-planner] OOM for patch={patch}, batch={batch}; trying smaller candidate.", flush=True)

    fallback_patch, fallback_batch = planner_candidates[-1]
    args.crop_depth, args.crop_size, args.batch_size = int(fallback_patch[0]), int(fallback_patch[1]), int(fallback_batch)
    plan.update({"selected_patch": list(fallback_patch), "batch_size": int(fallback_batch), "reason": "fallback_after_all_oom"})
    model.train(previous_training)
    return plan


def _load_model_checkpoint(model: torch.nn.Module, checkpoint: Path):
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    else:
        state = ckpt
    own = model.state_dict()
    matched = {}
    ignored = []
    for key, value in state.items():
        clean = key.replace("module.", "", 1)
        candidates = [clean]
        if not clean.startswith("backbone."):
            candidates.append(f"backbone.{clean}")
        target_key = next((candidate for candidate in candidates if candidate in own), None)
        if torch.is_tensor(value) and target_key is not None and tuple(own[target_key].shape) == tuple(value.shape):
            matched[target_key] = value
        else:
            ignored.append(clean)
    missing, unexpected = model.load_state_dict(matched, strict=False)
    return {
        "path": str(checkpoint),
        "matched_keys": len(matched),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "ignored_keys": len(ignored),
        "matched_params": int(sum(v.numel() for v in matched.values())),
        "model_params": int(sum(v.numel() for v in own.values())),
    }


def _next_case_from_loader(loader_iter, train_loader):
    try:
        return loader_iter, next(loader_iter)
    except StopIteration:
        loader_iter = iter(train_loader)
        return loader_iter, next(loader_iter)


def sample_case_crop_batch(loader_iter, train_loader, batch_size: int, crop_depth: int, crop_size: int, device: torch.device):
    case_ids = []
    images = []
    labels = []
    for _ in range(max(1, int(batch_size))):
        loader_iter, case_batch = _next_case_from_loader(loader_iter, train_loader)
        case_id, x, target = sample_case_crop(case_batch, int(crop_depth), int(crop_size), device)
        case_ids.append(case_id)
        images.append(x)
        labels.append(target)
    return loader_iter, case_ids, torch.cat(images, dim=0), torch.cat(labels, dim=0)


def run_synapse3d_training(args) -> dict:
    args.threads = _env_int("SYNAPSE3D_THREADS", int(args.threads))
    args.device = os.environ.get("SYNAPSE3D_DEVICE", args.device)
    args.preprocessed_root = os.environ.get("SYNAPSE3D_CACHE_ROOT", args.preprocessed_root)
    args.rebuild_cache = _env_bool("SYNAPSE3D_REBUILD_CACHE", bool(getattr(args, "rebuild_cache", False)))
    _resolve_train_runtime_options(args)
    if getattr(args, "t_max", None) is None:
        args.t_max = int(args.epochs)
    set_seed(int(args.seed))
    torch.set_num_threads(max(1, int(args.threads)))
    device = _device(args.device)
    print("#----------Creating logger----------#")
    _print_cuda_report(device)
    cache_summary = prepare_synapse_case_cache(
        train_npz_dir=Path(args.data_dir),
        test_h5_dir=Path(args.test_h5_dir),
        list_dir=Path(args.list_dir),
        out_root=Path(args.preprocessed_root),
        rebuild=bool(args.rebuild_cache),
    )
    cache_manifest = cache_summary["manifest"]
    train_case_ids = read_case_ids(Path(cache_manifest["cache_root"]) / "manifests" / "train_cases.txt")
    excluded_train_cases = set(_env_case_ids("SYNAPSE3D_TRAIN_EXCLUDE_CASES"))
    if excluded_train_cases:
        before_count = len(train_case_ids)
        train_case_ids = [case_id for case_id in train_case_ids if case_id not in excluded_train_cases]
        print(
            "[synapse3d] excluded validation cases from training: "
            f"{sorted(excluded_train_cases)} ({before_count} -> {len(train_case_ids)})"
        )
    train_dataset = Synapse3DCaseDataset(Path(cache_manifest["train_case_dir"]), train_case_ids)
    if len(train_dataset) == 0:
        raise RuntimeError("Synapse3D cache did not produce any train cases")
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=_dataloader_workers(int(args.threads)),
        collate_fn=collate_single_case,
    )
    steps_per_epoch = int(args.steps_per_epoch)
    args.steps_per_epoch = steps_per_epoch
    base_model_cfg = dict(getattr(args, "model_config", None) or load_synapse_config_defaults().get("model_config", {}))
    model_cfg = {
        **base_model_cfg,
        "model_name": canonical_synapse3d_model_name(args.network),
        "num_classes": int(args.num_classes),
        "hsic_proj_dim": int(args.hsic_proj_dim),
        "hsic_alpha": float(args.hsic_alpha),
        "hsic_temperature": float(args.hsic_temperature),
        "hsic_residual": float(args.hsic_residual),
        "enable_cache": bool(args.enable_cache),
        "load_pretrained": bool(args.load_pretrained),
        "pretrained_path": str(args.checkpoint) if args.checkpoint else None,
    }

    print("#----------Preparing 3D dataset----------#")
    print(f"✅ Synapse 3D训练病例数: {len(train_case_ids)}")
    print(f"✅ train_npz路径: {args.data_dir}")
    print(f"✅ train list: {Path(args.list_dir) / 'train.txt'}")
    print(f"✅ 3D case cache路径: {cache_manifest['train_case_dir']}")
    print(f"✅ split contract: {cache_manifest['split_contract']}")
    if device.type == "cuda" and steps_per_epoch <= len(train_case_ids):
        print("ℹ️  当前每轮3D crop数较少，nvidia-smi瞬时GPU利用率可能偏低；paper-scale运行可增大 SYNAPSE3D_STEPS_PER_EPOCH / CROP_SIZE。")

    print("#----------Prepareing 3D Models----------#")
    checkpoint_abs = os.path.abspath(str(args.checkpoint)) if args.checkpoint else None
    print(f"[CONFIG] load_pretrained={bool(args.load_pretrained)}  pretrained_path={checkpoint_abs}")
    model, num_classes = build_topomamba3d(model_cfg, device)
    print(f"✅ TopoMamba_3D模型创建完成: {model_cfg['model_name']} | fusion=hsic")
    print(f"📊 模型统计信息已记录: {_model_param_count(model) / 1e6:.2f}M 参数")
    _print_pretrained_report(getattr(model, "_pretrained_load_report", {}))
    model.train()
    print("#----------Prepareing loss, opt, sch and amp----------#")
    criterion = _build_criterion(num_classes, args)
    planner_report = _finalize_auto_train_plan(args, model, criterion, device, num_classes)
    args.planner_report = planner_report
    print("=" * 80)
    print("🚀 3D训练配置:")
    print("=" * 80)
    print(f"  模型: {model_cfg['model_name']}")
    print(f"  Training mode: {getattr(args, 'training_mode', '3d_fullres')}")
    print(f"  Planner: {planner_report.get('planner')} ({planner_report.get('reason')})")
    print(f"  批次大小: {args.batch_size} volumetric crops")
    print(f"  训练轮数: {args.epochs}")
    print(f"  每轮采样3D crop数: {steps_per_epoch}")
    print(f"  3D crop尺寸: D={args.crop_depth}, H/W={args.crop_size}")
    print(f"  学习率: {args.lr}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  数据工作进程/线程: {args.threads}")
    print(f"  保存间隔: {getattr(args, 'save_interval', 0)}")
    print(f"  打印间隔: {getattr(args, 'print_interval', 1)}")
    print(f"  恢复训练: {getattr(args, 'resume', False)}")
    print(f"  工作目录: {args.work_dir}")
    print("=" * 80)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = None
    if getattr(args, "scheduler", "none") == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(getattr(args, "t_max", args.epochs) or args.epochs),
            eta_min=float(getattr(args, "eta_min", 0.0)),
            last_epoch=int(getattr(args, "last_epoch", -1)),
        )
    elif str(getattr(args, "scheduler", "none")).lower() not in {"none", "null", ""}:
        raise ValueError(f"Unsupported TopoMamba-3D scheduler: {args.scheduler}")

    work_dir = Path(args.work_dir)
    checkpoint_dir = work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pth"
    final_path = checkpoint_dir / "final.pth"
    summary_path = work_dir / "train3d_summary.json"
    record_path = work_dir / "train_record.csv"
    config_path = work_dir / "train3d_config.json"
    config_txt_path = work_dir / "train_config_3d.txt"

    history = []
    best_loss = float("inf")
    start_epoch = 1
    resume_path = final_path
    reset_optimizer = bool(getattr(args, "reset_optimizer", False)) or os.environ.get("SYNAPSE3D_RESET_OPTIMIZER", "0").lower() in {"1", "true", "yes"}
    if bool(getattr(args, "resume", False)) and resume_path.exists():
        checkpoint = torch.load(str(resume_path), map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        if reset_optimizer:
            print("⚠️  SYNAPSE3D_RESET_OPTIMIZER=1/--reset-optimizer: 仅恢复模型权重，optimizer/scheduler 使用当前config重新初始化。")
        elif "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if not reset_optimizer and scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = list(checkpoint.get("history", []))
        if history:
            best_loss = min(float(row["loss"]) for row in history if "loss" in row)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"[synapse3d] resumed from {resume_path} at epoch {start_epoch}")

    actual_lr = float(optimizer.param_groups[0]["lr"])
    if abs(actual_lr - float(args.lr)) > 1e-12:
        print(
            "⚠️  resume checkpoint中的optimizer lr与当前config lr不同: "
            f"optimizer_lr={actual_lr}, config_lr={args.lr}. "
            "默认遵循checkpoint optimizer状态；如需使用当前config lr，请加 --resume false 或设置 SYNAPSE3D_RESET_OPTIMIZER=1。"
        )

    print("#----------Set other params----------#")
    print("#----------调试信息----------#")
    print(f"✅ min_loss: {best_loss if best_loss < float('inf') else 999}")
    print(f"✅ start_epoch: {start_epoch}")
    print(f"✅ resume_model 路径: {resume_path}")
    print(f"✅ 文件是否存在: {resume_path.exists()}")
    print(f"✅ config.resume_training: {getattr(args, 'resume', False)}")
    if start_epoch == 1:
        print("#----------从零开始训练----------#")
        print("🚀 没有检测到checkpoint文件，从epoch 1开始全新训练")
    else:
        print("#----------恢复训练----------#")
        print(f"🚀 从epoch {start_epoch}继续训练到 {args.epochs}")

    config_path.write_text(json.dumps({
        "data_dir": str(args.data_dir),
        "list_dir": str(args.list_dir),
        "train_list_file": str(Path(args.list_dir) / "train.txt"),
        "test_h5_dir": str(args.test_h5_dir),
        "preprocessed_root": str(args.preprocessed_root),
        "split_contract": cache_manifest["split_contract"],
        "work_dir": str(work_dir),
        "epochs": int(args.epochs),
        "start_epoch": start_epoch,
        "steps_per_epoch": steps_per_epoch,
        "crop": [int(args.crop_depth), int(args.crop_size), int(args.crop_size)],
        "selected_patch": [int(args.crop_depth), int(args.crop_size), int(args.crop_size)],
        "batch_size": int(args.batch_size),
        "excluded_train_cases": sorted(excluded_train_cases),
        "training_mode": getattr(args, "training_mode", "3d_fullres"),
        "planner_report": planner_report,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "scheduler": getattr(args, "scheduler", "none"),
        "t_max": int(getattr(args, "t_max", args.epochs) or args.epochs),
        "eta_min": float(getattr(args, "eta_min", 0.0)),
        "print_interval": int(getattr(args, "print_interval", 1)),
        "save_interval": int(getattr(args, "save_interval", 0)),
        "resume": bool(getattr(args, "resume", False)),
        "reset_optimizer": reset_optimizer,
        "actual_optimizer_lr": actual_lr,
        "model_config": model_cfg,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_train_config_txt(config_txt_path, args, model, model_cfg, len(train_case_ids), device)

    record_exists = record_path.exists() and start_epoch > 1
    start_time = time.time()
    print(
        "[synapse3d] training config: "
        f"epochs={args.epochs}, start_epoch={start_epoch}, steps_per_epoch={steps_per_epoch}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, scheduler={getattr(args, 'scheduler', 'none')}, "
        f"print_interval={getattr(args, 'print_interval', 1)}, save_interval={getattr(args, 'save_interval', 0)}",
        flush=True,
    )
    print("✅ 执行到训练前检查点")
    print(f"✅ config.network: {model_cfg['model_name']}")
    print("✅ 即将进入训练循环...")
    print("#----------Training----------#")
    print("🚗 使用TopoMamba_3D训练逻辑")
    print(f"✅ 即将开始TopoMamba_3D训练循环，epoch范围: {start_epoch} 到 {args.epochs}")
    printed_device_check = False
    loader_iter = iter(train_loader)
    for epoch in range(start_epoch, int(args.epochs) + 1):
        print(f"开始第 {epoch} 轮3D训练...")
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_grad = 0.0
        epoch_dice = 0.0
        last_case = None
        for step_idx in range(1, steps_per_epoch + 1):
            loader_iter, case_ids, x, target = sample_case_crop_batch(
                loader_iter,
                train_loader,
                int(args.batch_size),
                int(args.crop_depth),
                int(args.crop_size),
                device,
            )
            last_case = ",".join(case_ids)
            if not printed_device_check:
                model_device = next(model.parameters()).device
                print(f"✅ 3D model device: {model_device}")
                print(f"✅ 3D batch device: image={x.device}, label={target.device}")
                if device.type == "cuda" and (model_device.type != "cuda" or x.device.type != "cuda"):
                    print("⚠️  期望使用CUDA，但模型或输入没有在CUDA上。")
                printed_device_check = True
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite TopoMamba-3D loss at epoch {epoch}: {float(loss.detach().cpu())}")
            loss.backward()
            total_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.gradient_clip_norm))
            if not torch.isfinite(total_grad) or float(total_grad.detach().cpu()) <= 0:
                raise RuntimeError(f"Invalid TopoMamba-3D gradient norm at epoch {epoch}: {total_grad}")
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_grad += float(total_grad.detach().cpu())
            batch_dice = _foreground_dice_from_logits(logits, target, num_classes)
            epoch_dice += batch_dice
            now_lr = float(optimizer.param_groups[0]["lr"])
            if step_idx % max(1, int(getattr(args, "print_interval", 1))) == 0:
                components = getattr(criterion, "last_components", {})
                topo_txt = f", base: {components.get('base', 0):.4f}, topo: {components.get('topology', 0):.4f}" if components else ""
                print(
                    f"train3d: epoch {epoch}, iter:{step_idx}, loss: {float(loss.detach().cpu()):.4f}"
                    f"{topo_txt}, dice: {batch_dice:.4f}, lr: {now_lr}, case: {last_case}",
                    flush=True,
                )

        current_lr = float(optimizer.param_groups[0]["lr"])
        avg_loss = epoch_loss / max(1, steps_per_epoch)
        avg_dice = epoch_dice / max(1, steps_per_epoch)
        row = {
            "epoch": epoch,
            "case": last_case,
            "lr": current_lr,
            "loss": avg_loss,
            "avg_dice": avg_dice,
            "grad_norm": epoch_grad / max(1, steps_per_epoch),
            "components": getattr(criterion, "last_components", {}),
        }
        history.append(row)
        should_print = (
            epoch == start_epoch
            or epoch == int(args.epochs)
            or epoch % max(1, int(getattr(args, "print_interval", 1))) == 0
        )
        if should_print:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        print(f"Finish one epoch train3d: epoch {epoch}, loss: {avg_loss:.4f}, avg_dice: {avg_dice:.4f}, time(s): {time.time() - epoch_start:.2f}")
        print(f"训练完成，损失: {avg_loss:.4f}, 平均Dice: {avg_dice:.4f}")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "checkpoint_report": getattr(model, "_pretrained_load_report", {}),
            "history": history,
            "model_config": model_cfg,
            "planner_report": planner_report,
        }
        if avg_loss <= best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, best_path)
        save_interval = int(getattr(args, "save_interval", 0) or 0)
        if save_interval > 0 and epoch % save_interval == 0:
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:04d}.pth")
        record_mode = "a" if record_exists else "w"
        with record_path.open(record_mode, newline="", encoding="utf-8") as f:
            fieldnames = ["epoch", "lr", "loss", "avg_dice", "grad_norm", "case", "base_loss", "topology_loss", "total_loss"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not record_exists:
                writer.writeheader()
                record_exists = True
            components = row.get("components", {})
            writer.writerow({
                "epoch": epoch,
                "lr": current_lr,
                "loss": avg_loss,
                "avg_dice": avg_dice,
                "grad_norm": row["grad_norm"],
                "case": last_case,
                "base_loss": components.get("base"),
                "topology_loss": components.get("topology"),
                "total_loss": components.get("total"),
            })
        print(f"训练记录已保存到: {record_path}")
        print(f"第 {epoch} 轮3D训练完成")
        if scheduler is not None:
            scheduler.step()

    torch.save(
        {
            "epoch": int(args.epochs),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "checkpoint_report": getattr(model, "_pretrained_load_report", {}),
            "history": history,
            "model_config": model_cfg,
            "planner_report": planner_report,
        },
        final_path,
    )
    summary = {
        "status": "ok",
        "device": str(device),
        "epochs": int(args.epochs),
        "steps_per_epoch": steps_per_epoch,
        "crop": [int(args.crop_depth), int(args.crop_size), int(args.crop_size)],
        "selected_patch": [int(args.crop_depth), int(args.crop_size), int(args.crop_size)],
        "batch_size": int(args.batch_size),
        "training_mode": getattr(args, "training_mode", "3d_fullres"),
        "planner_report": planner_report,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "scheduler": getattr(args, "scheduler", "none"),
        "print_interval": int(getattr(args, "print_interval", 1)),
        "save_interval": int(getattr(args, "save_interval", 0)),
        "checkpoint_report": getattr(model, "_pretrained_load_report", {}),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "preprocessed_root": str(args.preprocessed_root),
        "split_contract": cache_manifest["split_contract"],
        "record_path": str(record_path),
        "config_path": str(config_path),
        "config_txt_path": str(config_txt_path),
        "elapsed_sec": time.time() - start_time,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[synapse3d] wrote {summary_path}")
    return summary


def _starts(length: int, patch: int, stride: int):
    if length <= patch:
        return [0]
    starts = list(range(0, max(1, length - patch + 1), stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def _mirror_axis_combinations(axes: Sequence[int]) -> list[tuple[int, ...]]:
    axes = tuple(dict.fromkeys(int(axis) for axis in axes))
    combos: list[tuple[int, ...]] = [()]
    for r in range(1, len(axes) + 1):
        combos.extend(tuple(combo) for combo in combinations(axes, r))
    return combos


def gaussian_importance_map(patch_dhw: Sequence[int], sigma_scale: float = 1.0 / 8.0) -> np.ndarray:
    weights = np.ones(tuple(int(v) for v in patch_dhw), dtype=np.float32)
    for axis, size in enumerate(patch_dhw):
        coord = np.arange(int(size), dtype=np.float32)
        center = (float(size) - 1.0) / 2.0
        sigma = max(float(size) * float(sigma_scale), 1e-6)
        axis_weight = np.exp(-0.5 * ((coord - center) / sigma) ** 2).astype(np.float32)
        shape = [1, 1, 1]
        shape[axis] = int(size)
        weights *= axis_weight.reshape(shape)
    weights /= max(float(weights.max()), 1e-6)
    return np.maximum(weights, 1e-6).astype(np.float32, copy=False)


def _predict_patch_logits(
    model,
    crop: np.ndarray,
    patch_dhw: Sequence[int],
    device: torch.device,
    mirror_axes: Sequence[int] = (),
) -> torch.Tensor:
    pd, ph, pw = (int(v) for v in patch_dhw)
    inp = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    logits_sum = None
    combos = _mirror_axis_combinations(mirror_axes)
    for axes in combos:
        dims = [2 + int(axis) for axis in axes]
        aug_inp = torch.flip(inp, dims=dims) if dims else inp
        logits = model(aug_inp)
        if logits.shape[-3:] != (pd, ph, pw):
            logits = F.interpolate(logits, size=(pd, ph, pw), mode="trilinear", align_corners=False)
        if dims:
            logits = torch.flip(logits, dims=dims)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / max(1, len(combos))


def predict_3d_quick_resize(model, volume: np.ndarray, patch_dhw, device: torch.device):
    depth, height, width = volume.shape
    pd, ph, pw = patch_dhw
    resized = zoom(volume, (pd / depth, ph / height, pw / width), order=1)
    x = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        if logits.shape[-3:] != (pd, ph, pw):
            logits = F.interpolate(logits, size=(pd, ph, pw), mode="trilinear", align_corners=False)
        pred_small = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.int16)
    pred = zoom(pred_small, (depth / pd, height / ph, width / pw), order=0)
    return pred.astype(np.int16)


def predict_3d_sliding(
    model,
    volume: np.ndarray,
    num_classes: int,
    patch_dhw,
    stride_dhw,
    device: torch.device,
    gaussian_blending: bool = False,
    mirror_tta: bool = False,
    mirror_axes: Sequence[int] = (0, 1, 2),
):
    depth, height, width = volume.shape
    pd, ph, pw = patch_dhw
    sd, sh, sw = stride_dhw
    pad_d, pad_h, pad_w = max(0, pd - depth), max(0, ph - height), max(0, pw - width)
    if pad_d or pad_h or pad_w:
        volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
    padded_shape = volume.shape
    logits_sum = np.zeros((num_classes, *padded_shape), dtype=np.float32)
    counts = np.zeros(padded_shape, dtype=np.float32)
    patch_weight = gaussian_importance_map(patch_dhw) if gaussian_blending else np.ones((pd, ph, pw), dtype=np.float32)
    active_mirror_axes = tuple(mirror_axes) if mirror_tta else ()
    model.eval()
    with torch.no_grad():
        for z in _starts(padded_shape[0], pd, sd):
            for y in _starts(padded_shape[1], ph, sh):
                for x0 in _starts(padded_shape[2], pw, sw):
                    crop = volume[z:z + pd, y:y + ph, x0:x0 + pw]
                    logits = _predict_patch_logits(model, crop, patch_dhw, device, active_mirror_axes)
                    logits_sum[:, z:z + pd, y:y + ph, x0:x0 + pw] += logits.squeeze(0).cpu().numpy() * patch_weight[None]
                    counts[z:z + pd, y:y + ph, x0:x0 + pw] += patch_weight
    prediction = np.argmax(logits_sum / np.maximum(counts[None], 1e-6), axis=0).astype(np.int16)
    return prediction[:depth, :height, :width]


def iter_h5_cases(label_dir: Path, limit: int | None):
    files = sorted(label_dir.glob("*.h5")) + sorted(label_dir.glob("*.hdf5"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"No Synapse h5 volumes found under {label_dir}")
    for path in files:
        case = path.stem.replace(".npy", "")
        with h5py.File(path, "r") as f:
            yield case, normalize_ct_like(f["image"][:]), f["label"][:]


def binary_dice_hd95(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    from medpy import metric as medpy_metric

    pred = pred.astype(np.uint8, copy=True)
    gt = gt.astype(np.uint8, copy=True)
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = medpy_metric.binary.dc(pred, gt)
        hd95 = medpy_metric.binary.hd95(pred, gt)
        return float(dice), float(hd95)
    if pred.sum() > 0 and gt.sum() == 0:
        return 1.0, 0.0
    return 0.0, 0.0


def _component_mask(mask: np.ndarray, keep_largest: bool, min_component_size: int | None, min_component_fraction: float) -> np.ndarray:
    if not np.any(mask) or (not keep_largest and min_component_size is None):
        return mask.astype(bool, copy=False)
    from scipy import ndimage as ndi

    structure = ndi.generate_binary_structure(mask.ndim, 1)
    labels, count = ndi.label(mask.astype(bool), structure=structure)
    if count <= 1:
        return mask.astype(bool, copy=False)
    sizes = ndi.sum(mask.astype(np.int32), labels, index=np.arange(1, count + 1))
    if keep_largest:
        keep_ids = np.array([int(np.argmax(sizes) + 1)])
    else:
        largest_size = float(np.max(sizes)) if sizes.size else 0.0
        threshold = max(float(min_component_size or 0), largest_size * float(min_component_fraction))
        keep_ids = np.nonzero(sizes >= threshold)[0] + 1
        if keep_ids.size == 0:
            keep_ids = np.array([int(np.argmax(sizes) + 1)])
    return np.isin(labels, keep_ids)


def postprocess_prediction(pred: np.ndarray, classes: Sequence[int], config: dict | None) -> np.ndarray:
    if not config or not config.get("enabled", True):
        return pred
    per_class = config.get("per_class") or {}
    default_cfg = config.get("default") or {}
    out = np.zeros_like(pred, dtype=pred.dtype)
    try:
        from scipy import ndimage as ndi
    except Exception:
        ndi = None
    for cls in classes:
        cls_cfg = {**default_cfg, **(per_class.get(str(cls)) or per_class.get(int(cls), {}) or {})}
        mask = pred == cls
        if bool(cls_cfg.get("fill_holes", False)) and ndi is not None:
            mask = ndi.binary_fill_holes(mask)
        mask = _component_mask(
            mask,
            keep_largest=bool(cls_cfg.get("keep_largest", False)),
            min_component_size=cls_cfg.get("min_component_size"),
            min_component_fraction=float(cls_cfg.get("min_component_fraction", 0.0) or 0.0),
        )
        out[mask] = cls
    return out


def load_postprocess_config(path: str | None) -> dict | None:
    if not path:
        return None
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Postprocess config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("source_path", str(config_path))
    return config


def summarize_case_metrics(case_rows: list[dict], classes: Sequence[int]):
    dice_per_organ = []
    hd95_per_organ = []
    for cls in classes:
        dice_per_organ.append(float(np.mean([row["per_class"][str(cls)]["dice"] for row in case_rows])))
        hd95_per_organ.append(float(np.mean([row["per_class"][str(cls)]["hd95"] for row in case_rows])))
    sample_dice = [float(np.mean([row["per_class"][str(cls)]["dice"] for cls in classes])) for row in case_rows]
    sample_hd95 = [float(np.mean([row["per_class"][str(cls)]["hd95"] for cls in classes])) for row in case_rows]
    return {
        "avg_dice": float(np.mean(dice_per_organ)),
        "std_dice": float(np.std(sample_dice)),
        "mean_hd95": float(np.mean(hd95_per_organ)),
        "std_hd95": float(np.std(sample_hd95)),
        "dice_per_organ": dice_per_organ,
        "hd95_per_organ": hd95_per_organ,
    }


def _train_config_path_from_weights(weights: Path | None) -> Path | None:
    if weights is None:
        return None
    if weights.parent.name == "checkpoints":
        return weights.parent.parent / "train3d_config.json"
    return weights.parent / "train3d_config.json"


def _patch_from_train_config(weights: Path | None) -> tuple[int, int, int] | None:
    config_path = _train_config_path_from_weights(weights)
    if config_path is None or not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    patch = data.get("selected_patch") or data.get("crop")
    if patch is None:
        planner_report = data.get("planner_report") or {}
        patch = planner_report.get("selected_patch")
    if patch is None:
        return None
    return _as_3tuple(patch, (128, 128, 128))


def _resolve_test_patch(args, weights: Path | None) -> tuple[int, int, int]:
    env_patch = _env_3tuple_optional("SYNAPSE3D_PATCH_SIZE")
    if env_patch is not None:
        return env_patch
    if getattr(args, "patch_size", None) is not None:
        return _as_3tuple(args.patch_size, (128, 128, 128))
    train_patch = _patch_from_train_config(weights)
    if train_patch is not None:
        return train_patch
    return (128, 128, 128)


def run_synapse3d_test(args) -> dict:
    args.threads = _env_int("SYNAPSE3D_THREADS", int(args.threads))
    args.device = os.environ.get("SYNAPSE3D_DEVICE", args.device)
    args.preprocessed_root = os.environ.get("SYNAPSE3D_CACHE_ROOT", args.preprocessed_root)
    args.rebuild_cache = _env_bool("SYNAPSE3D_REBUILD_CACHE", bool(getattr(args, "rebuild_cache", False)))
    if os.environ.get("SYNAPSE3D_INFER_MODE"):
        args.mode = os.environ["SYNAPSE3D_INFER_MODE"]
    if os.environ.get("SYNAPSE3D_OUT_DIR"):
        args.out_dir = os.environ["SYNAPSE3D_OUT_DIR"]
    if os.environ.get("SYNAPSE3D_EVAL_SPLIT"):
        args.eval_split = os.environ["SYNAPSE3D_EVAL_SPLIT"]
    if os.environ.get("SYNAPSE3D_POSTPROCESS_CONFIG"):
        args.postprocess_config = os.environ["SYNAPSE3D_POSTPROCESS_CONFIG"]
    if os.environ.get("SYNAPSE3D_SAVE_VISUALIZATIONS"):
        args.save_visualizations = os.environ["SYNAPSE3D_SAVE_VISUALIZATIONS"].lower() in {"1", "true", "yes"}
    args.allow_pretrained_only = bool(
        getattr(args, "allow_pretrained_only", False)
        or os.environ.get("SYNAPSE3D_ALLOW_PRETRAINED_ONLY", "0").lower() in {"1", "true", "yes"}
    )
    limit_value = os.environ.get("SYNAPSE3D_LIMIT")
    if limit_value:
        args.limit = None if limit_value.lower() == "all" else int(limit_value)
    args.gaussian_blending = _env_bool("SYNAPSE3D_GAUSSIAN_BLENDING", bool(getattr(args, "gaussian_blending", False)))
    args.mirror_tta = _env_bool("SYNAPSE3D_MIRROR_TTA", bool(getattr(args, "mirror_tta", False)))
    args.mirror_axes = _env_axes("SYNAPSE3D_MIRROR_AXES", getattr(args, "mirror_axes", (0, 1, 2)))
    _validate_synapse_mirror_tta_safety(bool(args.mirror_tta), tuple(args.mirror_axes))
    selected_cases = _env_case_ids("SYNAPSE3D_CASES")
    set_seed(int(args.seed))
    torch.set_num_threads(max(1, int(args.threads)))
    device = _device(args.device)
    cache_summary = prepare_synapse_case_cache(
        train_npz_dir=Path(args.train_npz_dir),
        test_h5_dir=Path(args.label_dir),
        list_dir=Path(args.list_dir),
        out_root=Path(args.preprocessed_root),
        rebuild=bool(args.rebuild_cache),
    )
    cache_manifest = cache_summary["manifest"]
    eval_split = getattr(args, "eval_split", "test")
    if eval_split not in {"test", "train"}:
        raise ValueError(f"Unsupported SYNAPSE3D_EVAL_SPLIT={eval_split!r}; expected 'test' or 'train'")
    manifest_name = "test_cases.txt" if eval_split == "test" else "train_cases.txt"
    test_case_ids = read_case_ids(Path(cache_manifest["cache_root"]) / "manifests" / manifest_name)
    if selected_cases:
        missing_cases = [case for case in selected_cases if case not in set(test_case_ids)]
        if missing_cases:
            raise ValueError(f"SYNAPSE3D_CASES contains cases outside {eval_split} split: {missing_cases}")
        test_case_ids = selected_cases
    if args.limit is not None:
        test_case_ids = test_case_ids[:int(args.limit)]
    if not test_case_ids:
        raise RuntimeError("Synapse3D cache did not produce any test cases")
    weights = Path(args.weights) if args.weights else None
    weights_exist = weights is not None and weights.exists()
    if not weights_exist and not args.allow_pretrained_only:
        raise FileNotFoundError(f"3D test weights not found: {weights}")
    patch = _resolve_test_patch(args, weights)
    stride_env = _env_3tuple_optional("SYNAPSE3D_STRIDE")
    if stride_env is not None:
        args.stride = stride_env
    stride = tuple(int(v) for v in args.stride) if args.stride else tuple(max(1, p // 2) for p in patch)
    load_pretrained_for_test = bool(args.allow_pretrained_only and not weights_exist)
    print("#----------创建3D测试环境----------#")
    network = canonical_synapse3d_model_name(getattr(args, "network", "TopoMamba_3D_t"))
    print(f"🔧 当前网络架构: {network}")
    print(f"🔧 权重路径: {args.weights}")
    _print_cuda_report(device)
    print("#----------准备3D测试数据----------#")
    print(f"✅ Synapse test volume路径: {args.label_dir}")
    print(f"✅ Synapse test list: {Path(args.list_dir) / 'test_vol.txt'}")
    print(f"✅ 3D case cache路径: {cache_manifest['test_case_dir']}")
    print(f"✅ split contract: {cache_manifest['split_contract']}")
    print(f"✅ 推理模式: {args.mode}")
    print(f"✅ eval split: {eval_split}")
    print(f"✅ patch_size: {patch}")
    print(f"✅ stride: {stride}")
    print(f"✅ gaussian_blending: {bool(args.gaussian_blending)}")
    print(f"✅ mirror_tta: {bool(args.mirror_tta)} axes={tuple(args.mirror_axes)}")
    base_model_cfg = dict(getattr(args, "model_config", None) or load_synapse_config_defaults().get("model_config", {}))
    model_cfg = {
        **base_model_cfg,
        "model_name": network,
        "num_classes": int(args.num_classes),
        "hsic_proj_dim": int(args.hsic_proj_dim),
        "hsic_alpha": float(args.hsic_alpha),
        "hsic_temperature": float(args.hsic_temperature),
        "hsic_residual": float(args.hsic_residual),
        "enable_cache": bool(args.enable_cache),
        "load_pretrained": load_pretrained_for_test,
        "pretrained_path": str(args.pretrained_path) if args.pretrained_path else None,
    }
    print("#----------准备3D模型----------#")
    checkpoint_abs = os.path.abspath(str(args.pretrained_path)) if args.pretrained_path else None
    print(f"[CONFIG] load_pretrained={load_pretrained_for_test}  pretrained_path={checkpoint_abs}")
    model, num_classes = build_topomamba3d(model_cfg, device)
    print(f"✅ TopoMamba_3D模型创建完成: {network}")
    print(f"📊 模型统计信息已记录: {_model_param_count(model) / 1e6:.2f}M 参数")
    _print_pretrained_report(getattr(model, "_pretrained_load_report", {}))
    load_report = None
    if weights_exist:
        load_report = _load_model_checkpoint(model, weights)
        print(json.dumps({"loaded_weights": load_report}, ensure_ascii=False), flush=True)
    else:
        print("[synapse3d] using pretrained-only model for inference")

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    prediction_vis_dir = out_dir / "prediction_visualization"
    comparison_dir = out_dir / "comparison_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "save_visualizations", True):
        prediction_vis_dir.mkdir(parents=True, exist_ok=True)
        comparison_dir.mkdir(parents=True, exist_ok=True)

    classes = tuple(range(1, int(args.num_classes)))
    postprocess_config = load_postprocess_config(getattr(args, "postprocess_config", None))
    if postprocess_config:
        print(f"✅ postprocess config: {postprocess_config.get('source_path')}")
    case_rows = []
    print("#----------开始3D测试----------#")
    for case in test_case_ids:
        case_dir = Path(cache_manifest["test_case_dir"] if eval_split == "test" else cache_manifest["train_case_dir"])
        image, label, _ = load_cached_case(case_dir, case)
        if args.mode == "quick-resize":
            pred = predict_3d_quick_resize(model, image, patch, device)
        else:
            pred = predict_3d_sliding(
                model,
                image,
                num_classes,
                patch,
                stride,
                device,
                gaussian_blending=bool(args.gaussian_blending),
                mirror_tta=bool(args.mirror_tta),
                mirror_axes=tuple(args.mirror_axes),
            )
        pred = postprocess_prediction(pred, classes, postprocess_config)
        if args.save_predictions:
            np.savez_compressed(pred_dir / f"{case}.npz", prediction=pred.astype(np.int16))
        vis_path = None
        if getattr(args, "save_visualizations", True):
            vis_path = save_volume_prediction_png(case, image, label, pred, prediction_vis_dir)
            save_volume_prediction_png(case, image, label, pred, comparison_dir)
        per_class = {}
        for cls in classes:
            dice, hd95 = binary_dice_hd95(pred == cls, label == cls)
            per_class[str(cls)] = {"dice": dice, "hd95": hd95}
        row = {
            "case": case,
            "shape": list(pred.shape),
            "label_shape": list(label.shape),
            "per_class": per_class,
            "mean_dice": float(np.mean([v["dice"] for v in per_class.values()])),
            "mean_hd95": float(np.mean([v["hd95"] for v in per_class.values()])),
            "visualization": str(vis_path) if vis_path is not None else None,
        }
        case_rows.append(row)
        print(json.dumps({"case": case, "mean_dice": row["mean_dice"], "mean_hd95": row["mean_hd95"]}, ensure_ascii=False), flush=True)

    summary = summarize_case_metrics(case_rows, classes)
    result = {
        "model_info": {
            "network": network,
            "model_config": model_cfg,
            "test_weight_path": str(weights) if weights else None,
            "load_report": load_report,
        },
        "summary": summary,
        "organ_details": {
            ORGAN_NAMES[i]: {
                "dice": float(summary["dice_per_organ"][i]),
                "hd95": float(summary["hd95_per_organ"][i]),
            }
            for i in range(len(ORGAN_NAMES))
        },
        "test_info": {
            "device": str(device),
            "mode": args.mode,
            "eval_split": eval_split,
            "patch_size": list(patch),
            "stride": list(stride),
            "gaussian_blending": bool(args.gaussian_blending),
            "mirror_tta": bool(args.mirror_tta),
            "mirror_axes": list(args.mirror_axes),
            "postprocess_config": postprocess_config,
            "split_contract": cache_manifest["split_contract"],
            "preprocessed_root": str(args.preprocessed_root),
            "test_case_ids": test_case_ids,
            "num_test_cases": len(case_rows),
            "output_dir": str(out_dir),
            "prediction_dir": str(pred_dir) if args.save_predictions else None,
            "prediction_visualization_dir": str(prediction_vis_dir) if getattr(args, "save_visualizations", True) else None,
            "comparison_dir": str(comparison_dir) if getattr(args, "save_visualizations", True) else None,
        },
        "cases": case_rows,
    }
    (out_dir / "test_results_detailed.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "test_results_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "avg_dice", "std_dice", "mean_hd95", "std_hd95"]
        fieldnames += [f"dice_{name}" for name in ORGAN_NAMES]
        fieldnames += [f"hd95_{name}" for name in ORGAN_NAMES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        row = {
            "model": network,
            "avg_dice": summary["avg_dice"],
            "std_dice": summary["std_dice"],
            "mean_hd95": summary["mean_hd95"],
            "std_hd95": summary["std_hd95"],
        }
        for i, name in enumerate(ORGAN_NAMES):
            row[f"dice_{name}"] = summary["dice_per_organ"][i]
            row[f"hd95_{name}"] = summary["hd95_per_organ"][i]
        writer.writerow(row)
    with (out_dir / "organ_metrics.md").open("w", encoding="utf-8") as f:
        f.write("| Organ | Dice | HD95 |\n|---|---:|---:|\n")
        for i, name in enumerate(ORGAN_NAMES):
            f.write(f"| {name} | {summary['dice_per_organ'][i]:.4f} | {summary['hd95_per_organ'][i]:.4f} |\n")
        f.write(f"| Mean | {summary['avg_dice']:.4f} | {summary['mean_hd95']:.4f} |\n")
    print("🧹 3D测试完成，已清理缓存")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"📊 平均Dice: {summary['avg_dice']:.4f} ± {summary['std_dice']:.4f}")
    print(f"📊 平均HD95: {summary['mean_hd95']:.4f} ± {summary['std_hd95']:.4f}")
    print(f"📁 结果保存路径: {out_dir}")
    print(f"🖼️  3D预测可视化: {prediction_vis_dir if getattr(args, 'save_visualizations', True) else 'disabled'}")
    print(f"[synapse3d] wrote {out_dir}")
    return result


def build_train_parser() -> argparse.ArgumentParser:
    defaults = load_synapse_config_defaults()
    model_cfg = dict(defaults.get("model_config", {}))
    parser = argparse.ArgumentParser(description="TopoMamba-3D Synapse volumetric crop training")
    parser.add_argument("--network", "--model-name", dest="network", default=defaults["network"])
    parser.add_argument("--data-dir", default=defaults["data_path"])
    parser.add_argument("--test-h5-dir", default=defaults["volume_path"])
    parser.add_argument("--list-dir", default=defaults["list_dir"])
    parser.add_argument("--list-file", default=str(Path(defaults["list_dir"]) / "train.txt"))
    parser.add_argument("--preprocessed-root", default=_default_cache_root())
    parser.add_argument("--checkpoint", default=model_cfg.get("pretrained_path", "pre_trained_weights/tmp_model_ep799_0.8498.pt"))
    parser.add_argument("--work-dir", default=defaults["work_dir"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--crop-depth", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--training-mode", default=defaults["training_mode"])
    parser.add_argument("--planner", default=defaults["planner"])
    parser.add_argument("--target-patch", type=int, nargs=3, default=defaults["target_patch"])
    parser.set_defaults(
        default_batch_size=defaults["batch_size"],
        max_batch_size=defaults["max_batch_size"],
        patch_candidates=defaults["patch_candidates"],
    )
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--threads", type=int, default=max(1, defaults["num_workers"]))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-classes", type=int, default=int(model_cfg.get("num_classes", defaults.get("num_classes", 9))))
    parser.add_argument("--hsic-proj-dim", type=int, default=int(model_cfg.get("hsic_proj_dim", 32)))
    parser.add_argument("--hsic-alpha", type=float, default=float(model_cfg.get("hsic_alpha", 0.8)))
    parser.add_argument("--hsic-temperature", type=float, default=float(model_cfg.get("hsic_temperature", 1.5)))
    parser.add_argument("--hsic-residual", type=float, default=float(model_cfg.get("hsic_residual", 0.2)))
    add_bool_arg(parser, "--enable-cache", bool(model_cfg.get("enable_cache", True)), "Enable ScanCache")
    add_bool_arg(parser, "--load-pretrained", bool(model_cfg.get("load_pretrained", True)), "Load the SegMamba-compatible pretrained checkpoint")
    add_bool_arg(parser, "--topology-loss-enabled", defaults["topology_loss_enabled"], "Enable topology-aware focal loss")
    add_bool_arg(parser, "--resume", defaults["resume"], "Resume from work-dir/checkpoints/final.pth if it exists")
    add_bool_arg(parser, "--reset-optimizer", False, "Resume model weights but reset optimizer/scheduler to current config")
    add_bool_arg(parser, "--rebuild-cache", False, "Force rebuilding the Synapse3D case cache")
    parser.add_argument("--topology-loss-weight", type=float, default=defaults["topology_loss_weight"])
    parser.add_argument("--topology-focal-gamma", type=float, default=defaults["topology_focal_gamma"])
    parser.add_argument("--topology-critical-weight", type=float, default=defaults["topology_critical_weight"])
    parser.add_argument("--topology-loss-max-elements", type=int, default=defaults["topology_loss_max_elements"])
    parser.add_argument("--gradient-clip-norm", type=float, default=defaults["gradient_clip_norm"])
    parser.add_argument("--print-interval", "--print_interval", dest="print_interval", type=int, default=defaults["print_interval"])
    parser.add_argument("--save-interval", "--save_interval", dest="save_interval", type=int, default=defaults["save_interval"])
    parser.add_argument("--scheduler", "--sch", dest="scheduler", default=defaults["scheduler"])
    parser.add_argument("--t-max", "--T_max", dest="t_max", type=int, default=None)
    parser.add_argument("--eta-min", "--eta_min", dest="eta_min", type=float, default=defaults["eta_min"])
    parser.add_argument("--last-epoch", "--last_epoch", dest="last_epoch", type=int, default=defaults["last_epoch"])
    return parser


def build_test_parser() -> argparse.ArgumentParser:
    defaults = load_synapse_config_defaults()
    model_cfg = dict(defaults.get("model_config", {}))
    parser = argparse.ArgumentParser(description="TopoMamba-3D Synapse volume testing")
    parser.add_argument("--network", "--model-name", dest="network", default=defaults["network"])
    parser.add_argument("--weights", default=defaults["test_weights_path"])
    parser.add_argument("--pretrained-path", default=model_cfg.get("pretrained_path", "pre_trained_weights/tmp_model_ep799_0.8498.pt"))
    parser.add_argument("--allow-pretrained-only", action="store_true")
    parser.add_argument("--train-npz-dir", default=defaults["data_path"])
    parser.add_argument("--label-dir", default=defaults["volume_path"])
    parser.add_argument("--list-dir", default=defaults["list_dir"])
    parser.add_argument("--preprocessed-root", default=_default_cache_root())
    parser.add_argument("--out-dir", default="test_results/TopoMamba_3D_t_synapse")
    parser.add_argument("--patch-size", type=int, nargs=3, default=None)
    parser.add_argument("--stride", type=int, nargs=3, default=None)
    parser.add_argument("--mode", choices=["quick-resize", "sliding"], default="sliding")
    parser.add_argument("--eval-split", choices=["test", "train"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2050)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--hsic-proj-dim", type=int, default=32)
    parser.add_argument("--hsic-alpha", type=float, default=0.8)
    parser.add_argument("--hsic-temperature", type=float, default=1.5)
    parser.add_argument("--hsic-residual", type=float, default=0.2)
    add_bool_arg(parser, "--enable-cache", True, "Enable ScanCache")
    add_bool_arg(parser, "--rebuild-cache", False, "Force rebuilding the Synapse3D case cache")
    add_bool_arg(parser, "--gaussian-blending", False, "Use Gaussian patch weighting for sliding-window inference")
    add_bool_arg(parser, "--mirror-tta", False, "Use experimental mirror TTA; disabled by default for Synapse side-specific organs")
    parser.add_argument("--mirror-axes", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--postprocess-config", default=None)
    add_bool_arg(parser, "--save-predictions", True, "Save per-case prediction npz files")
    add_bool_arg(parser, "--save-visualizations", True, "Save middle-slice prediction PNG comparisons")
    return parser


def _copy_known_attrs(source, target, mapping: dict[str, str | None]) -> None:
    for source_name, target_name in mapping.items():
        value = getattr(source, source_name, None)
        if value is not None and target_name is not None:
            setattr(target, target_name, value)


def build_synapse3d_train_args_from_legacy_argv(argv: Sequence[str]):
    """Parse the legacy ``train_synapse.py`` CLI surface for early 3D routing."""
    args = build_train_parser().parse_args([])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--network", default=None)
    parser.add_argument("--data_path", "--data-path", dest="data_dir", default=None)
    parser.add_argument("--list_dir", "--list-dir", dest="list_dir", default=None)
    parser.add_argument("--volume_path", "--volume-path", dest="test_h5_dir", default=None)
    parser.add_argument("--preprocessed_root", "--preprocessed-root", dest="preprocessed_root", default=None)
    parser.add_argument("--work_dir", dest="work_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps_per_epoch", "--steps-per-epoch", dest="steps_per_epoch", type=int, default=None)
    parser.add_argument("--crop_depth", "--crop-depth", dest="crop_depth", type=int, default=None)
    parser.add_argument("--crop_size", "--crop-size", dest="crop_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hsic_proj_dim", type=int, default=None)
    parser.add_argument("--hsic_temperature", type=float, default=None)
    parser.add_argument("--hsic_residual", type=float, default=None)
    parser.add_argument("--enable_cache", type=str2bool, default=None)
    parser.add_argument("--load_pretrained", type=str2bool, default=None)
    parser.add_argument("--topology_loss_enabled", type=str2bool, default=None)
    parser.add_argument("--resume", type=str2bool, default=None)
    parser.add_argument("--reset_optimizer", "--reset-optimizer", dest="reset_optimizer", type=str2bool, default=None)
    parser.add_argument("--rebuild_cache", "--rebuild-cache", dest="rebuild_cache", type=str2bool, default=None)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--val_interval", default=None)
    parser.add_argument("--save_interval", "--save-interval", dest="save_interval", type=int, default=None)
    parser.add_argument("--print_interval", "--print-interval", dest="print_interval", type=int, default=None)
    parser.add_argument("--sch", "--scheduler", dest="scheduler", default=None)
    parser.add_argument("--T_max", "--t-max", dest="t_max", type=int, default=None)
    parser.add_argument("--eta_min", "--eta-min", dest="eta_min", type=float, default=None)
    parser.add_argument("--last_epoch", "--last-epoch", dest="last_epoch", type=int, default=None)
    parser.add_argument("--amp", default=None)
    parser.add_argument("--branch_mode", default=None)
    parser.add_argument("--fusion_method", default=None)
    parser.add_argument("--enable_gating", default=None)
    parser.add_argument("--postprocess_enabled", default=None)
    known, _ = parser.parse_known_args(list(argv))
    _copy_known_attrs(
        known,
        args,
        {
            "network": "network",
            "data_dir": "data_dir",
            "list_dir": "list_dir",
            "test_h5_dir": "test_h5_dir",
            "preprocessed_root": "preprocessed_root",
            "work_dir": "work_dir",
            "epochs": "epochs",
            "steps_per_epoch": "steps_per_epoch",
            "crop_depth": "crop_depth",
            "crop_size": "crop_size",
            "lr": "lr",
            "weight_decay": "weight_decay",
            "seed": "seed",
            "checkpoint": "checkpoint",
            "hsic_proj_dim": "hsic_proj_dim",
            "hsic_temperature": "hsic_temperature",
            "hsic_residual": "hsic_residual",
            "enable_cache": "enable_cache",
            "load_pretrained": "load_pretrained",
            "topology_loss_enabled": "topology_loss_enabled",
            "resume": "resume",
            "reset_optimizer": "reset_optimizer",
            "rebuild_cache": "rebuild_cache",
            "batch_size": "batch_size",
            "save_interval": "save_interval",
            "print_interval": "print_interval",
            "scheduler": "scheduler",
            "t_max": "t_max",
            "eta_min": "eta_min",
            "last_epoch": "last_epoch",
        },
    )
    if getattr(args, "list_dir", None):
        args.list_file = str(Path(args.list_dir) / "train.txt")
    if known.num_workers is not None:
        args.threads = max(1, int(known.num_workers))
    args.threads = _env_int("SYNAPSE3D_THREADS", int(args.threads))
    args.device = os.environ.get("SYNAPSE3D_DEVICE", args.device)
    args.preprocessed_root = os.environ.get("SYNAPSE3D_CACHE_ROOT", args.preprocessed_root)
    args.rebuild_cache = _env_bool("SYNAPSE3D_REBUILD_CACHE", bool(getattr(args, "rebuild_cache", False)))
    return args


def run_synapse3d_training_from_legacy_argv(argv: Sequence[str]) -> dict:
    args = build_synapse3d_train_args_from_legacy_argv(argv)
    print("[synapse3d] early routing train_synapse.py before 2D imports")
    return run_synapse3d_training(args)


def run_synapse3d_test_from_legacy_argv(argv: Sequence[str]) -> dict:
    """Parse the legacy ``test_synapse.py`` CLI surface for early 3D routing."""
    args = build_test_parser().parse_args([])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--network", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data_path", "--data-path", dest="train_npz_dir", default=None)
    parser.add_argument("--list_dir", "--list-dir", dest="list_dir", default=None)
    parser.add_argument("--preprocessed_root", "--preprocessed-root", dest="preprocessed_root", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--patch_size", "--patch-size", dest="patch_size", type=int, nargs=3, default=None)
    parser.add_argument("--stride", type=int, nargs=3, default=None)
    parser.add_argument("--eval_split", "--eval-split", dest="eval_split", default=None)
    parser.add_argument("--gaussian_blending", "--gaussian-blending", dest="gaussian_blending", type=str2bool, default=None)
    parser.add_argument("--mirror_tta", "--mirror-tta", dest="mirror_tta", type=str2bool, default=None)
    parser.add_argument("--mirror_axes", "--mirror-axes", dest="mirror_axes", type=int, nargs="+", default=None)
    parser.add_argument("--postprocess_config", "--postprocess-config", dest="postprocess_config", default=None)
    parser.add_argument("--hsic_proj_dim", type=int, default=None)
    parser.add_argument("--hsic_temperature", type=float, default=None)
    parser.add_argument("--hsic_residual", type=float, default=None)
    parser.add_argument("--enable_cache", type=str2bool, default=None)
    parser.add_argument("--rebuild_cache", "--rebuild-cache", dest="rebuild_cache", type=str2bool, default=None)
    parser.add_argument("--save_visualizations", "--save-visualizations", dest="save_visualizations", type=str2bool, default=None)
    parser.add_argument("--load_pretrained", default=None)
    parser.add_argument("--topology_loss_enabled", default=None)
    parser.add_argument("--branch_mode", default=None)
    parser.add_argument("--fusion_method", default=None)
    parser.add_argument("--enable_gating", default=None)
    parser.add_argument("--postprocess_enabled", default=None)
    known, _ = parser.parse_known_args(list(argv))
    _copy_known_attrs(
        known,
        args,
        {
            "network": "network",
            "weights": "weights",
            "train_npz_dir": "train_npz_dir",
            "list_dir": "list_dir",
            "preprocessed_root": "preprocessed_root",
            "seed": "seed",
            "patch_size": "patch_size",
            "stride": "stride",
            "eval_split": "eval_split",
            "gaussian_blending": "gaussian_blending",
            "mirror_tta": "mirror_tta",
            "mirror_axes": "mirror_axes",
            "postprocess_config": "postprocess_config",
            "hsic_proj_dim": "hsic_proj_dim",
            "hsic_temperature": "hsic_temperature",
            "hsic_residual": "hsic_residual",
            "enable_cache": "enable_cache",
            "rebuild_cache": "rebuild_cache",
            "save_visualizations": "save_visualizations",
        },
    )
    if known.num_workers is not None:
        args.threads = max(1, int(known.num_workers))
    args.allow_pretrained_only = os.environ.get("SYNAPSE3D_ALLOW_PRETRAINED_ONLY", "0").lower() in {"1", "true", "yes"}
    print("[synapse3d] early routing test_synapse.py before 2D imports")
    return run_synapse3d_test(args)


def run_synapse3d_training_from_config(config) -> dict:
    defaults = load_synapse_config_defaults()
    model_cfg = dict(getattr(config, "model_config", {}))
    network = getattr(config, "network", defaults["network"])
    if not is_synapse3d_model(network):
        network = defaults["network"]
        model_cfg = dict(defaults.get("model_config", {}))
    elif not is_synapse3d_model(model_cfg.get("model_name")):
        model_cfg = dict(defaults.get("model_config", {}))
    args = SimpleNamespace(
        network=network,
        model_config=model_cfg,
        data_dir=getattr(config, "data_path", "data/Synapse/train_npz"),
        test_h5_dir=getattr(config, "volume_path", "data/Synapse/test_vol_h5"),
        list_dir=getattr(config, "list_dir", "data/Synapse/lists/lists_Synapse"),
        list_file=str(Path(getattr(config, "list_dir", "data/Synapse/lists/lists_Synapse")) / "train.txt"),
        preprocessed_root=_default_cache_root(),
        checkpoint=model_cfg.get("pretrained_path", "pre_trained_weights/tmp_model_ep799_0.8498.pt"),
        work_dir=getattr(config, "work_dir", "results/TopoMamba_3D_t_synapse"),
        epochs=int(getattr(config, "synapse3d_epochs", defaults["epochs"])),
        steps_per_epoch=None,
        crop_size=None,
        crop_depth=None,
        batch_size=None,
        default_batch_size=int(getattr(config, "synapse3d_batch_size", defaults["batch_size"])),
        max_batch_size=int(getattr(config, "synapse3d_max_batch_size", defaults["max_batch_size"])),
        training_mode=getattr(config, "synapse3d_training_mode", defaults["training_mode"]),
        planner=getattr(config, "synapse3d_planner", defaults["planner"]),
        target_patch=_as_3tuple(getattr(config, "synapse3d_target_patch", defaults["target_patch"]), defaults["target_patch"]),
        patch_candidates=tuple(
            _as_3tuple(candidate, defaults["target_patch"])
            for candidate in getattr(config, "synapse3d_patch_candidates", defaults["patch_candidates"])
        ),
        lr=float(getattr(config, "lr", 3e-4)),
        weight_decay=float(getattr(config, "weight_decay", 0.01)),
        seed=int(getattr(config, "seed", 2050)),
        threads=_env_int("SYNAPSE3D_THREADS", max(1, int(getattr(config, "num_workers", 1)))),
        device=os.environ.get("SYNAPSE3D_DEVICE", "auto"),
        num_classes=int(model_cfg.get("num_classes", getattr(config, "num_classes", 9))),
        hsic_proj_dim=int(model_cfg.get("hsic_proj_dim", 32)),
        hsic_alpha=float(model_cfg.get("hsic_alpha", 0.8)),
        hsic_temperature=float(model_cfg.get("hsic_temperature", 1.5)),
        hsic_residual=float(model_cfg.get("hsic_residual", 0.2)),
        enable_cache=bool(model_cfg.get("enable_cache", True)),
        load_pretrained=bool(model_cfg.get("load_pretrained", True)),
        topology_loss_enabled=bool(getattr(config, "topology_loss_enabled", True)),
        topology_loss_weight=float(getattr(config, "topology_loss_weight", 0.05)),
        topology_focal_gamma=float(getattr(config, "topology_focal_gamma", 2.0)),
        topology_critical_weight=float(getattr(config, "topology_critical_weight", 4.0)),
        topology_loss_max_elements=int(getattr(config, "topology_loss_max_elements", 65536)),
        gradient_clip_norm=float(getattr(config, "gradient_clip_norm", 12.0)),
        print_interval=int(getattr(config, "print_interval", 20)),
        save_interval=int(getattr(config, "save_interval", 100)),
        scheduler=getattr(config, "sch", "CosineAnnealingLR"),
        t_max=int(getattr(config, "synapse3d_t_max", getattr(config, "synapse3d_epochs", defaults["epochs"]))),
        eta_min=float(getattr(config, "eta_min", 6e-7)),
        last_epoch=int(getattr(config, "last_epoch", -1)),
        resume=bool(getattr(config, "resume_training", True)),
        rebuild_cache=_env_bool("SYNAPSE3D_REBUILD_CACHE", False),
        reset_optimizer=os.environ.get("SYNAPSE3D_RESET_OPTIMIZER", "0").lower() in {"1", "true", "yes"},
    )
    print("[synapse3d] routing train_synapse.py to dedicated TopoMamba-3D volumetric trainer")
    return run_synapse3d_training(args)


def run_synapse3d_test_from_config(config) -> dict:
    defaults = load_synapse_config_defaults()
    model_cfg = dict(getattr(config, "model_config", {}))
    network = canonical_synapse3d_model_name(getattr(config, "network", "TopoMamba_3D_t"))
    if not is_synapse3d_model(model_cfg.get("model_name")):
        model_cfg = dict(defaults.get("model_config", {}))
    args = SimpleNamespace(
        network=network,
        model_config=model_cfg,
        weights=getattr(config, "test_weights_path", "results/TopoMamba_3D_t_synapse/checkpoints/best.pth"),
        pretrained_path=model_cfg.get("pretrained_path", "pre_trained_weights/tmp_model_ep799_0.8498.pt"),
        allow_pretrained_only=os.environ.get("SYNAPSE3D_ALLOW_PRETRAINED_ONLY", "0").lower() in {"1", "true", "yes"},
        train_npz_dir=getattr(config, "data_path", "data/Synapse/train_npz"),
        label_dir=getattr(config, "volume_path", "data/Synapse/test_vol_h5"),
        list_dir=getattr(config, "list_dir", "data/Synapse/lists/lists_Synapse"),
        preprocessed_root=_default_cache_root(),
        out_dir=os.environ.get("SYNAPSE3D_OUT_DIR", f"test_results/{network}_synapse"),
        patch_size=None,
        stride=None,
        mode="sliding",
        eval_split=os.environ.get("SYNAPSE3D_EVAL_SPLIT", "test"),
        limit=None,
        seed=int(getattr(config, "seed", 2050)),
        threads=_env_int("SYNAPSE3D_THREADS", max(1, int(getattr(config, "num_workers", 1)))),
        device=os.environ.get("SYNAPSE3D_DEVICE", "auto"),
        num_classes=int(model_cfg.get("num_classes", getattr(config, "num_classes", 9))),
        hsic_proj_dim=int(model_cfg.get("hsic_proj_dim", 32)),
        hsic_alpha=float(model_cfg.get("hsic_alpha", 0.8)),
        hsic_temperature=float(model_cfg.get("hsic_temperature", 1.5)),
        hsic_residual=float(model_cfg.get("hsic_residual", 0.2)),
        enable_cache=bool(model_cfg.get("enable_cache", True)),
        gaussian_blending=_env_bool("SYNAPSE3D_GAUSSIAN_BLENDING", False),
        mirror_tta=_env_bool("SYNAPSE3D_MIRROR_TTA", False),
        mirror_axes=_env_axes("SYNAPSE3D_MIRROR_AXES", (0, 1, 2)),
        postprocess_config=os.environ.get("SYNAPSE3D_POSTPROCESS_CONFIG"),
        rebuild_cache=_env_bool("SYNAPSE3D_REBUILD_CACHE", False),
        save_predictions=True,
        save_visualizations=os.environ.get("SYNAPSE3D_SAVE_VISUALIZATIONS", "1").lower() not in {"0", "false", "no"},
    )
    print("[synapse3d] routing test_synapse.py to dedicated TopoMamba-3D volume evaluator")
    return run_synapse3d_test(args)
