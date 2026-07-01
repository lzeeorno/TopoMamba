#!/usr/bin/env python
"""Select Synapse3D post-processing from held-out validation predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.synapse3d_pipeline import postprocess_prediction


def _load_prediction(path: Path) -> np.ndarray:
    with np.load(path) as data:
        for key in ("prediction", "pred", "label"):
            if key in data:
                arr = data[key]
                if arr.ndim == 4:
                    arr = arr.argmax(axis=0)
                return arr.astype(np.int16)
    raise KeyError(f"No prediction array found in {path}")


def _load_label(cache_dir: Path, case: str) -> np.ndarray:
    with np.load(cache_dir / f"{case}.npz") as data:
        return np.asarray(data["label"], dtype=np.int16)


def _dice(pred_mask: np.ndarray, label_mask: np.ndarray) -> float:
    pred_sum = int(pred_mask.sum())
    label_sum = int(label_mask.sum())
    if pred_sum and label_sum:
        return float(2.0 * np.logical_and(pred_mask, label_mask).sum() / (pred_sum + label_sum))
    if pred_sum and not label_sum:
        return 1.0
    return 0.0


def _candidate_configs() -> list[dict]:
    candidates = [
        {"name": "none", "keep_largest": False, "fill_holes": False, "min_component_size": None, "min_component_fraction": 0.0},
        {"name": "keep_largest", "keep_largest": True, "fill_holes": False, "min_component_size": None, "min_component_fraction": 0.0},
        {"name": "fill_holes", "keep_largest": False, "fill_holes": True, "min_component_size": None, "min_component_fraction": 0.0},
        {"name": "keep_largest_fill_holes", "keep_largest": True, "fill_holes": True, "min_component_size": None, "min_component_fraction": 0.0},
    ]
    for size in (64, 128, 256, 512):
        for fraction in (0.00025, 0.0005, 0.001):
            candidates.append(
                {
                    "name": f"mild_size_{size}_frac_{fraction:g}",
                    "keep_largest": False,
                    "fill_holes": False,
                    "min_component_size": size,
                    "min_component_fraction": fraction,
                }
            )
            candidates.append(
                {
                    "name": f"fill_holes_mild_size_{size}_frac_{fraction:g}",
                    "keep_largest": False,
                    "fill_holes": True,
                    "min_component_size": size,
                    "min_component_fraction": fraction,
                }
            )
    return candidates


def _case_files(pred_dir: Path, cases: Sequence[str] | None) -> list[tuple[str, Path]]:
    if cases:
        return [(case, pred_dir / f"{case}.npz") for case in cases]
    return [(path.stem, path) for path in sorted(pred_dir.glob("*.npz"))]


def select_postprocess(pred_dir: Path, label_cache_dir: Path, classes: Sequence[int], cases: Sequence[str] | None) -> tuple[dict, list[dict]]:
    matched = [(case, path) for case, path in _case_files(pred_dir, cases) if path.exists()]
    if not matched:
        raise RuntimeError(f"No prediction files found in {pred_dir}")
    candidates = _candidate_configs()
    loaded = [(case, _load_prediction(path), _load_label(label_cache_dir, case)) for case, path in matched]
    per_class = {}
    rows = []
    for cls in classes:
        best = None
        for candidate in candidates:
            scores = []
            cfg = {"enabled": True, "per_class": {str(cls): candidate}}
            for case, pred, label in loaded:
                pred_pp = postprocess_prediction(pred, [cls], cfg)
                scores.append(_dice(pred_pp == cls, label == cls))
            mean_dice = float(np.mean(scores))
            row = {"class": int(cls), "candidate": candidate["name"], "mean_dice": mean_dice}
            rows.append(row)
            if best is None or mean_dice > best["mean_dice"]:
                best = {**row, "config": candidate}
        per_class[str(cls)] = {k: v for k, v in best["config"].items() if k != "name"}
        per_class[str(cls)]["selected_variant"] = best["candidate"]
        per_class[str(cls)]["validation_mean_dice"] = best["mean_dice"]
    config = {
        "enabled": True,
        "selection": "validation_gated_per_class_mean_dice",
        "prediction_dir": str(pred_dir),
        "label_cache_dir": str(label_cache_dir),
        "cases": [case for case, _ in matched],
        "classes": [int(cls) for cls in classes],
        "per_class": per_class,
    }
    return config, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Select validation-gated Synapse3D post-processing config")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--label-cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--classes", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--cases", nargs="+", default=None)
    args = parser.parse_args()

    config, rows = select_postprocess(args.pred_dir, args.label_cache_dir, args.classes, args.cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    rows_path = args.out.with_name(args.out.stem + "_candidates.json")
    rows_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[select_synapse3d_postprocess] wrote {args.out}")
    print(f"[select_synapse3d_postprocess] wrote {rows_path}")


if __name__ == "__main__":
    main()
