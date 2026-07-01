#!/usr/bin/env python
"""Focused checks for Synapse3D nnUNet-lite case cache behavior.

Run:
  PYTHONDONTWRITEBYTECODE=1 python tools/test_synapse3d_cache_pipeline.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_build_legacy_split_manifest_reports_overlap_exactly():
    from tools.synapse3d_preprocess import build_legacy_split_manifest

    manifest = build_legacy_split_manifest(
        ROOT / "data" / "Synapse" / "lists" / "lists_Synapse" / "train.txt",
        ROOT / "data" / "Synapse" / "lists" / "lists_Synapse" / "test_vol.txt",
    )

    assert manifest["split_contract"] == "legacy_synapse_lists"
    assert manifest["train_case_count"] == 30
    assert manifest["test_case_count"] == 12
    assert manifest["overlap_case_count"] == 12
    assert manifest["overlap_cases"] == manifest["test_cases"]


def test_prepare_synapse_case_cache_writes_manifest_and_exact_case_lists():
    from tools.synapse3d_preprocess import prepare_synapse_case_cache

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        train_npz_dir = root / "train_npz"
        test_h5_dir = root / "test_vol_h5"
        list_dir = root / "lists_Synapse"
        out_root = root / "cache"

        train_npz_dir.mkdir(parents=True, exist_ok=True)
        test_h5_dir.mkdir(parents=True, exist_ok=True)
        list_dir.mkdir(parents=True, exist_ok=True)

        train_slices = {
            "case0001_slice000": (
                np.linspace(-125, 275, 16, dtype=np.float32).reshape(4, 4),
                np.arange(16, dtype=np.int16).reshape(4, 4),
            ),
            "case0001_slice001": (
                np.full((4, 4), 275, dtype=np.float32),
                (np.arange(16, dtype=np.int16).reshape(4, 4) + 20),
            ),
            "case0002_slice000": (
                np.full((4, 4), 3, dtype=np.float32),
                np.full((4, 4), 2, dtype=np.int16),
            ),
        }
        for slice_name, (image, label) in train_slices.items():
            np.savez_compressed(train_npz_dir / f"{slice_name}.npz", image=image, label=label)

        with h5py.File(test_h5_dir / "case0001.npy.h5", "w") as f:
            f.create_dataset("image", data=np.ones((2, 4, 4), dtype=np.float32))
            f.create_dataset("label", data=np.ones((2, 4, 4), dtype=np.uint8))
        with h5py.File(test_h5_dir / "case0003.npy.h5", "w") as f:
            f.create_dataset("image", data=np.full((2, 4, 4), 5, dtype=np.float32))
            f.create_dataset("label", data=np.full((2, 4, 4), 3, dtype=np.uint8))

        (list_dir / "train.txt").write_text("case0001_slice000\ncase0001_slice001\ncase0002_slice000\n", encoding="utf-8")
        (list_dir / "test_vol.txt").write_text("case0001\ncase0003\n", encoding="utf-8")

        summary = prepare_synapse_case_cache(
            train_npz_dir=train_npz_dir,
            test_h5_dir=test_h5_dir,
            list_dir=list_dir,
            out_root=out_root,
            rebuild=True,
        )

        manifest_path = out_root / "manifests" / "dataset_manifest.json"
        train_cases_path = out_root / "manifests" / "train_cases.txt"
        test_cases_path = out_root / "manifests" / "test_cases.txt"
        legacy_report_path = out_root / "manifests" / "legacy_split_report.json"

        assert manifest_path.exists()
        assert train_cases_path.exists()
        assert test_cases_path.exists()
        assert legacy_report_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_cases = [line.strip() for line in train_cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        test_cases = [line.strip() for line in test_cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        assert summary["manifest"]["split_contract"] == "legacy_synapse_lists"
        assert manifest["split_contract"] == "legacy_synapse_lists"
        assert manifest["preprocess_version"] == 2
        assert manifest["normalization"] == "legacy_hu_clip_-125_275_to_0_1_if_needed"
        assert manifest["train_orientation"] == "train_npz_hflip_axis1_to_test_vol"
        assert len(train_cases) == 2
        assert len(test_cases) == 2
        assert test_cases == manifest["test_cases"]
        assert all((out_root / "train_cases" / f"{case}.npz").exists() for case in train_cases)
        assert all((out_root / "train_cases" / f"{case}.pkl").exists() for case in train_cases)
        assert all((out_root / "test_cases" / f"{case}.npz").exists() for case in test_cases)
        assert all((out_root / "test_cases" / f"{case}.pkl").exists() for case in test_cases)

        train_case = np.load(out_root / "train_cases" / "case0001.npz")
        with (out_root / "train_cases" / "case0001.pkl").open("rb") as f:
            properties = pickle.load(f)
        assert train_case["image"].shape == (2, 4, 4)
        assert train_case["label"].shape == (2, 4, 4)
        expected_image0 = np.clip((train_slices["case0001_slice000"][0] + 125.0) / 400.0, 0.0, 1.0)[::-1, :]
        expected_label0 = train_slices["case0001_slice000"][1][::-1, :]
        assert np.allclose(train_case["image"][0], expected_image0)
        assert np.array_equal(train_case["label"][0], expected_label0)
        assert properties["case_id"] == "case0001"
        assert properties["source_split"] == "train"
        assert properties["source_format"] == "train_npz_stack_hflip_huclip"
        assert properties["orientation"] == "train_npz_hflip_axis1_to_test_vol"


def test_prepare_synapse_case_cache_rebuilds_invalid_existing_npz():
    from tools.synapse3d_preprocess import prepare_synapse_case_cache

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        train_npz_dir = root / "train_npz"
        test_h5_dir = root / "test_vol_h5"
        list_dir = root / "lists_Synapse"
        out_root = root / "cache"

        train_npz_dir.mkdir(parents=True, exist_ok=True)
        test_h5_dir.mkdir(parents=True, exist_ok=True)
        list_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            train_npz_dir / "case0001_slice000.npz",
            image=np.ones((4, 4), dtype=np.float32),
            label=np.ones((4, 4), dtype=np.int16),
        )
        with h5py.File(test_h5_dir / "case0003.npy.h5", "w") as f:
            f.create_dataset("image", data=np.ones((1, 4, 4), dtype=np.float32))
            f.create_dataset("label", data=np.full((1, 4, 4), 3, dtype=np.uint8))

        (list_dir / "train.txt").write_text("case0001_slice000\n", encoding="utf-8")
        (list_dir / "test_vol.txt").write_text("case0003\n", encoding="utf-8")

        prepare_synapse_case_cache(train_npz_dir, test_h5_dir, list_dir, out_root, rebuild=True)
        broken = out_root / "test_cases" / "case0003.npz"
        broken.write_bytes(b"")

        summary = prepare_synapse_case_cache(train_npz_dir, test_h5_dir, list_dir, out_root, rebuild=False)

        assert summary["reused"] is False
        with np.load(broken) as data:
            assert data["image"].shape == (1, 4, 4)
            assert data["label"].shape == (1, 4, 4)


def test_synapse3d_pipeline_references_cache_and_dataloader_flow():
    text = (ROOT / "tools" / "synapse3d_pipeline.py").read_text(encoding="utf-8")
    assert "prepare_synapse_case_cache" in text
    assert "Synapse3DCaseDataset" in text
    assert "DataLoader" in text


def test_synapse3d_all_axis_mirror_tta_is_guarded_by_default():
    from tools.synapse3d_pipeline import _validate_synapse_mirror_tta_safety

    old = os.environ.pop("SYNAPSE3D_ALLOW_UNSAFE_MIRROR_TTA", None)
    try:
        try:
            _validate_synapse_mirror_tta_safety(True, (0, 1, 2))
        except ValueError as exc:
            assert "Unsafe Synapse3D mirror TTA requested" in str(exc)
        else:
            raise AssertionError("all-axis mirror TTA should be guarded by default")

        _validate_synapse_mirror_tta_safety(False, (0, 1, 2))
        os.environ["SYNAPSE3D_ALLOW_UNSAFE_MIRROR_TTA"] = "1"
        _validate_synapse_mirror_tta_safety(True, (0, 1, 2))
    finally:
        if old is None:
            os.environ.pop("SYNAPSE3D_ALLOW_UNSAFE_MIRROR_TTA", None)
        else:
            os.environ["SYNAPSE3D_ALLOW_UNSAFE_MIRROR_TTA"] = old


def test_synapse3d_binary_dice_hd95_matches_legacy_medpy_semantics():
    from medpy import metric as medpy_metric
    from tools.synapse3d_pipeline import binary_dice_hd95

    pred = np.zeros((8, 8, 8), dtype=np.uint8)
    gt = np.zeros((8, 8, 8), dtype=np.uint8)
    pred[1:4, 1:4, 1:4] = 1
    gt[2:5, 1:4, 1:4] = 1

    dice, hd95 = binary_dice_hd95(pred, gt)

    assert abs(dice - float(medpy_metric.binary.dc(pred, gt))) < 1e-7
    assert abs(hd95 - float(medpy_metric.binary.hd95(pred, gt))) < 1e-7
    assert binary_dice_hd95(pred, np.zeros_like(pred)) == (1.0, 0.0)
    assert binary_dice_hd95(np.zeros_like(pred), np.zeros_like(pred)) == (0.0, 0.0)


def main():
    test_build_legacy_split_manifest_reports_overlap_exactly()
    test_prepare_synapse_case_cache_writes_manifest_and_exact_case_lists()
    test_prepare_synapse_case_cache_rebuilds_invalid_existing_npz()
    test_synapse3d_pipeline_references_cache_and_dataloader_flow()
    test_synapse3d_all_axis_mirror_tta_is_guarded_by_default()
    test_synapse3d_binary_dice_hd95_matches_legacy_medpy_semantics()
    print("tools/test_synapse3d_cache_pipeline.py: PASS")


if __name__ == "__main__":
    main()
