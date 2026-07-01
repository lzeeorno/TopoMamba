#!/usr/bin/env python
"""nnUNet-lite Synapse3D case cache helpers."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


LEGACY_SPLIT_CONTRACT = "legacy_synapse_lists"
LEGACY_PREPROCESS_VERSION = 2
LEGACY_NORMALIZATION = "legacy_hu_clip_-125_275_to_0_1_if_needed"
LEGACY_CT_HU_CLIP = (-125.0, 275.0)
LEGACY_TRAIN_ORIENTATION = "train_npz_hflip_axis1_to_test_vol"
LEGACY_TEST_ORIENTATION = "test_vol_native"
MAX_CLASS_LOCATIONS = 2048


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_case_ids(path: Path) -> list[str]:
    return read_nonempty_lines(path)


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _case_from_slice_name(slice_name: str) -> str:
    if "_slice" not in slice_name:
        raise ValueError(f"Unsupported Synapse slice name: {slice_name}")
    return slice_name.split("_slice", 1)[0]


def _slice_index(slice_name: str) -> int:
    if "_slice" not in slice_name:
        raise ValueError(f"Unsupported Synapse slice name: {slice_name}")
    return int(slice_name.rsplit("_slice", 1)[1])


def build_legacy_split_manifest(train_list_file: Path, test_list_file: Path) -> dict:
    train_slice_names = read_nonempty_lines(train_list_file)
    test_cases = read_nonempty_lines(test_list_file)
    train_cases = _ordered_unique(_case_from_slice_name(name) for name in train_slice_names)
    train_case_set = set(train_cases)
    overlap_cases = [case for case in test_cases if case in train_case_set]
    return {
        "split_contract": LEGACY_SPLIT_CONTRACT,
        "train_case_count": len(train_cases),
        "test_case_count": len(test_cases),
        "overlap_case_count": len(overlap_cases),
        "train_cases": train_cases,
        "test_cases": test_cases,
        "overlap_cases": overlap_cases,
        "train_list_file": str(Path(train_list_file)),
        "test_list_file": str(Path(test_list_file)),
    }


def build_train_case_groups(train_list_file: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for slice_name in read_nonempty_lines(train_list_file):
        groups[_case_from_slice_name(slice_name)].append(slice_name)
    return {case_id: sorted(slice_names, key=_slice_index) for case_id, slice_names in groups.items()}


def load_train_case_volume(train_npz_dir: Path, slice_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for slice_name in sorted(slice_names, key=_slice_index):
        with np.load(Path(train_npz_dir) / f"{slice_name}.npz") as data:
            images.append(data["image"].astype(np.float32))
            labels.append(data["label"].astype(np.int16))
    return np.stack(images, axis=0), np.stack(labels, axis=0)


def normalize_legacy_synapse_image(image: np.ndarray) -> np.ndarray:
    """Map legacy Synapse CT arrays onto the 0..1 range used by test_vol_h5."""

    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=float(LEGACY_CT_HU_CLIP[1]),
        neginf=float(LEGACY_CT_HU_CLIP[0]),
    )
    if image.size == 0:
        return image
    if float(image.min()) >= -1e-6 and float(image.max()) <= 1.0 + 1e-6:
        return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    lo, hi = LEGACY_CT_HU_CLIP
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)


def align_legacy_train_case_to_test_protocol(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert stacked train_npz slices to the test_vol_h5 orientation/intensity protocol."""

    image = normalize_legacy_synapse_image(image)
    return image[:, ::-1, :], np.asarray(label, dtype=np.int16)[:, ::-1, :]


def resolve_test_case_path(test_h5_dir: Path, case_id: str) -> Path:
    candidates = [
        Path(test_h5_dir) / f"{case_id}.npy.h5",
        Path(test_h5_dir) / f"{case_id}.h5",
        Path(test_h5_dir) / f"{case_id}.hdf5",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No Synapse test volume found for {case_id} under {test_h5_dir}")


def load_test_case_volume(test_h5_dir: Path, case_id: str) -> tuple[np.ndarray, np.ndarray, Path]:
    path = resolve_test_case_path(test_h5_dir, case_id)
    with h5py.File(path, "r") as f:
        image = np.asarray(f["image"][:], dtype=np.float32)
        label = np.asarray(f["label"][:], dtype=np.int16)
    return image, label, path


def _sample_class_locations(label: np.ndarray, max_locations_per_class: int = MAX_CLASS_LOCATIONS) -> dict[int, np.ndarray]:
    class_locations: dict[int, np.ndarray] = {}
    for cls_value in np.unique(label):
        cls = int(cls_value)
        if cls <= 0:
            continue
        coords = np.argwhere(label == cls)
        if coords.shape[0] == 0:
            continue
        if coords.shape[0] > max_locations_per_class:
            keep = np.linspace(0, coords.shape[0] - 1, num=max_locations_per_class, dtype=np.int64)
            coords = coords[keep]
        dtype = np.int16 if max(label.shape) < np.iinfo(np.int16).max else np.int32
        class_locations[cls] = coords.astype(dtype, copy=False)
    return class_locations


def compute_case_properties(
    case_id: str,
    image: np.ndarray,
    label: np.ndarray,
    source_split: str,
    source_format: str,
    orientation: str,
) -> dict:
    foreground = np.argwhere(label > 0)
    if foreground.shape[0] > 0:
        foreground_bbox = [
            [int(foreground[:, axis].min()), int(foreground[:, axis].max()) + 1]
            for axis in range(label.ndim)
        ]
    else:
        foreground_bbox = [[0, int(size)] for size in label.shape]
    present_classes = [int(cls) for cls in np.unique(label) if int(cls) > 0]
    return {
        "case_id": case_id,
        "image_shape": [int(size) for size in image.shape],
        "label_shape": [int(size) for size in label.shape],
        "source_split": source_split,
        "source_format": source_format,
        "preprocess_version": LEGACY_PREPROCESS_VERSION,
        "normalization": LEGACY_NORMALIZATION,
        "ct_hu_clip": [float(LEGACY_CT_HU_CLIP[0]), float(LEGACY_CT_HU_CLIP[1])],
        "orientation": orientation,
        "present_classes": present_classes,
        "foreground_bbox": foreground_bbox,
        "class_locations": _sample_class_locations(label),
    }


def save_case_cache(
    case_id: str,
    image: np.ndarray,
    label: np.ndarray,
    out_dir: Path,
    source_split: str,
    source_format: str,
    orientation: str,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    properties = compute_case_properties(case_id, image, label, source_split, source_format, orientation)
    npz_path = out_dir / f"{case_id}.npz"
    pkl_path = out_dir / f"{case_id}.pkl"
    npz_tmp = out_dir / f"{case_id}.npz.tmp"
    pkl_tmp = out_dir / f"{case_id}.pkl.tmp"
    with npz_tmp.open("wb") as f:
        np.savez_compressed(f, image=image.astype(np.float32), label=label.astype(np.int16))
    npz_tmp.replace(npz_path)
    with pkl_tmp.open("wb") as f:
        pickle.dump(properties, f, protocol=pickle.HIGHEST_PROTOCOL)
    pkl_tmp.replace(pkl_path)
    return properties


def load_cached_case(case_dir: Path, case_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    case_dir = Path(case_dir)
    with np.load(case_dir / f"{case_id}.npz") as data:
        image = np.asarray(data["image"], dtype=np.float32)
        label = np.asarray(data["label"], dtype=np.int16)
    with (case_dir / f"{case_id}.pkl").open("rb") as f:
        properties = pickle.load(f)
    return image, label, properties


def _write_case_list(path: Path, case_ids: Sequence[str]) -> None:
    text = "\n".join(case_ids)
    if text:
        text += "\n"
    Path(path).write_text(text, encoding="utf-8")


def _manifest_matches_current_protocol(manifest: dict) -> bool:
    return (
        manifest.get("preprocess_version") == LEGACY_PREPROCESS_VERSION
        and manifest.get("normalization") == LEGACY_NORMALIZATION
        and manifest.get("train_orientation") == LEGACY_TRAIN_ORIENTATION
        and manifest.get("test_orientation") == LEGACY_TEST_ORIENTATION
    )


def _case_cache_is_valid(case_dir: Path, case_id: str, expected_orientation: str) -> bool:
    npz_path = Path(case_dir) / f"{case_id}.npz"
    pkl_path = Path(case_dir) / f"{case_id}.pkl"
    if not npz_path.exists() or not pkl_path.exists():
        return False
    try:
        with np.load(npz_path) as data:
            if "image" not in data or "label" not in data:
                return False
            image_shape = tuple(data["image"].shape)
            label_shape = tuple(data["label"].shape)
            if image_shape != label_shape or len(image_shape) != 3:
                return False
        with pkl_path.open("rb") as f:
            properties = pickle.load(f)
    except Exception:
        return False
    return (
        properties.get("preprocess_version") == LEGACY_PREPROCESS_VERSION
        and properties.get("normalization") == LEGACY_NORMALIZATION
        and properties.get("orientation") == expected_orientation
    )


def prepare_synapse_case_cache(
    train_npz_dir: Path,
    test_h5_dir: Path,
    list_dir: Path,
    out_root: Path,
    rebuild: bool = False,
) -> dict:
    train_npz_dir = Path(train_npz_dir)
    test_h5_dir = Path(test_h5_dir)
    list_dir = Path(list_dir)
    out_root = Path(out_root)
    manifests_dir = out_root / "manifests"
    train_case_dir = out_root / "train_cases"
    test_case_dir = out_root / "test_cases"
    dataset_manifest_path = manifests_dir / "dataset_manifest.json"
    train_cases_path = manifests_dir / "train_cases.txt"
    test_cases_path = manifests_dir / "test_cases.txt"
    legacy_report_path = manifests_dir / "legacy_split_report.json"

    train_list_file = list_dir / "train.txt"
    test_list_file = list_dir / "test_vol.txt"
    split_manifest = build_legacy_split_manifest(train_list_file, test_list_file)
    dataset_manifest = {
        **split_manifest,
        "cache_root": str(out_root),
        "train_case_dir": str(train_case_dir),
        "test_case_dir": str(test_case_dir),
        "preprocess_version": LEGACY_PREPROCESS_VERSION,
        "normalization": LEGACY_NORMALIZATION,
        "ct_hu_clip": [float(LEGACY_CT_HU_CLIP[0]), float(LEGACY_CT_HU_CLIP[1])],
        "train_orientation": LEGACY_TRAIN_ORIENTATION,
        "test_orientation": LEGACY_TEST_ORIENTATION,
        "train_source": str(train_npz_dir),
        "test_source": str(test_h5_dir),
    }

    if not rebuild and dataset_manifest_path.exists() and train_cases_path.exists() and test_cases_path.exists():
        train_cases = read_case_ids(train_cases_path)
        test_cases = read_case_ids(test_cases_path)
        train_ready = all(_case_cache_is_valid(train_case_dir, case_id, LEGACY_TRAIN_ORIENTATION) for case_id in train_cases)
        test_ready = all(_case_cache_is_valid(test_case_dir, case_id, LEGACY_TEST_ORIENTATION) for case_id in test_cases)
        if train_ready and test_ready:
            cached_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
            if _manifest_matches_current_protocol(cached_manifest):
                return {
                    "manifest": cached_manifest,
                    "train_case_dir": str(train_case_dir),
                    "test_case_dir": str(test_case_dir),
                    "reused": True,
                }
            rebuild = True
        else:
            rebuild = True

    manifests_dir.mkdir(parents=True, exist_ok=True)
    train_case_dir.mkdir(parents=True, exist_ok=True)
    test_case_dir.mkdir(parents=True, exist_ok=True)

    train_groups = build_train_case_groups(train_list_file)
    for case_id in split_manifest["train_cases"]:
        npz_path = train_case_dir / f"{case_id}.npz"
        pkl_path = train_case_dir / f"{case_id}.pkl"
        if rebuild or not npz_path.exists() or not pkl_path.exists():
            image, label = load_train_case_volume(train_npz_dir, train_groups[case_id])
            image, label = align_legacy_train_case_to_test_protocol(image, label)
            save_case_cache(
                case_id,
                image,
                label,
                train_case_dir,
                source_split="train",
                source_format="train_npz_stack_hflip_huclip",
                orientation=LEGACY_TRAIN_ORIENTATION,
            )

    for case_id in split_manifest["test_cases"]:
        npz_path = test_case_dir / f"{case_id}.npz"
        pkl_path = test_case_dir / f"{case_id}.pkl"
        if rebuild or not npz_path.exists() or not pkl_path.exists():
            image, label, _ = load_test_case_volume(test_h5_dir, case_id)
            image = normalize_legacy_synapse_image(image)
            save_case_cache(
                case_id,
                image,
                label,
                test_case_dir,
                source_split="test",
                source_format="test_vol_h5_huclip",
                orientation=LEGACY_TEST_ORIENTATION,
            )

    _write_case_list(train_cases_path, split_manifest["train_cases"])
    _write_case_list(test_cases_path, split_manifest["test_cases"])
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    legacy_report_path.write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "manifest": dataset_manifest,
        "train_case_dir": str(train_case_dir),
        "test_case_dir": str(test_case_dir),
        "reused": False,
    }
