#!/usr/bin/env python
"""Case-level cache dataset and crop sampling for Synapse3D."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from tools.synapse3d_preprocess import load_cached_case


class Synapse3DCaseDataset(Dataset):
    def __init__(self, case_dir: Path, case_ids: Sequence[str]):
        self.case_dir = Path(case_dir)
        self.case_ids = list(case_ids)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict:
        case_id = self.case_ids[index]
        image, label, properties = load_cached_case(self.case_dir, case_id)
        return {
            "case_id": case_id,
            "image": image,
            "label": label,
            "properties": properties,
        }


def collate_single_case(batch: list[dict]) -> dict:
    return batch[0]


def _pick_crop_center(label: np.ndarray, properties: dict, crop_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    class_locations = properties.get("class_locations", {}) or {}
    available = [np.asarray(coords) for coords in class_locations.values() if len(coords)]
    if available:
        coords = random.choice(available)
        center = np.asarray(coords[random.randrange(len(coords))], dtype=np.int64)
    else:
        center = np.asarray([dim // 2 for dim in label.shape], dtype=np.int64)
    jitter = np.asarray([max(1, size // 8) for size in crop_shape], dtype=np.int64)
    for axis in range(len(center)):
        center[axis] = int(np.clip(center[axis] + random.randint(-int(jitter[axis]), int(jitter[axis])), 0, label.shape[axis] - 1))
    return int(center[0]), int(center[1]), int(center[2])


def _pad_to_shape(image: np.ndarray, label: np.ndarray, crop_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    pad_spec = []
    offsets = []
    for dim, size in zip(image.shape, crop_shape):
        pad = max(0, size - dim)
        before = pad // 2
        after = pad - before
        pad_spec.append((before, after))
        offsets.append(before)
    if any(before or after for before, after in pad_spec):
        image = np.pad(image, pad_spec, mode="constant")
        label = np.pad(label, pad_spec, mode="constant")
    return image, label, (int(offsets[0]), int(offsets[1]), int(offsets[2]))


def _crop_around_center(array: np.ndarray, center: tuple[int, int, int], crop_shape: tuple[int, int, int]) -> np.ndarray:
    starts = []
    for axis, (dim, mid, size) in enumerate(zip(array.shape, center, crop_shape)):
        start = max(0, min(int(mid) - size // 2, dim - size))
        starts.append(start)
    slices = tuple(slice(start, start + size) for start, size in zip(starts, crop_shape))
    return array[slices]


def sample_case_crop(case_batch: dict, crop_depth: int, crop_size: int, device: torch.device) -> tuple[str, torch.Tensor, torch.Tensor]:
    case_id = str(case_batch["case_id"])
    image = np.asarray(case_batch["image"], dtype=np.float32)
    label = np.asarray(case_batch["label"], dtype=np.int16)
    properties = dict(case_batch.get("properties", {}))
    crop_shape = (int(crop_depth), int(crop_size), int(crop_size))
    center = _pick_crop_center(label, properties, crop_shape)
    image, label, offsets = _pad_to_shape(image, label, crop_shape)
    padded_center = tuple(int(value + offset) for value, offset in zip(center, offsets))
    crop_image = _crop_around_center(image, padded_center, crop_shape)
    crop_label = _crop_around_center(label, padded_center, crop_shape)
    crop_image = np.nan_to_num(crop_image, nan=0.0, posinf=1.0, neginf=0.0)
    x = torch.from_numpy(crop_image).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(crop_label).unsqueeze(0).to(device=device, dtype=torch.long)
    return case_id, x, y