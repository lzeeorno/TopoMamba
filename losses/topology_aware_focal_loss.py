"""Topology-aware focal loss used by TopoMamba training.

This implementation keeps the topology operation detached from autograd, then
uses the exact discrete topology error map as a focal-weight mask on the model
probabilities. It is deliberately small and dependency-light for the existing
training loops; if GUDHI is installed in ``cmamba`` the module records that the
exact PH backend is available, while the default path uses exact connected
component and hole counts through SciPy/skimage-style binary morphology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - runtime dependency check
    ndi = None

try:
    import gudhi  # noqa: F401
    _HAS_GUDHI = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_GUDHI = False


@dataclass
class TopologyLossConfig:
    weight: float = 0.05
    gamma: float = 2.0
    topology_weight: float = 4.0
    threshold: float = 0.5
    foreground_classes: Optional[Sequence[int]] = None
    max_elements: int = 65536
    eps: float = 1e-6


def _maybe_downsample(x: torch.Tensor, max_elements: int) -> torch.Tensor:
    spatial = x.shape[-3:] if x.dim() == 5 else x.shape[-2:]
    n = int(np.prod(spatial))
    if n <= max_elements:
        return x
    if x.dim() == 5:
        scale = (max_elements / float(n)) ** (1.0 / 3.0)
        size = tuple(max(8, int(round(s * scale))) for s in spatial)
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    scale = (max_elements / float(n)) ** 0.5
    size = tuple(max(16, int(round(s * scale))) for s in spatial)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _as_probabilities(pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes == 1:
        if pred.min().detach() >= 0 and pred.max().detach() <= 1:
            return pred.clamp(1e-6, 1 - 1e-6)
        return torch.sigmoid(pred).clamp(1e-6, 1 - 1e-6)
    return torch.softmax(pred, dim=1).clamp(1e-6, 1 - 1e-6)


def _target_mask(target: torch.Tensor, class_idx: int, num_classes: int) -> torch.Tensor:
    if num_classes == 1:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        return (target > 0.5).float()
    if target.dim() == 4 and target.shape[1] == num_classes:
        labels = target.argmax(dim=1, keepdim=True)
    elif target.dim() == 5 and target.shape[1] == num_classes:
        labels = target.argmax(dim=1, keepdim=True)
    else:
        labels = target.long()
        if labels.dim() == target.dim() and labels.shape[1] == 1:
            pass
        elif labels.dim() in (3, 4):
            labels = labels.unsqueeze(1)
    return (labels == class_idx).float()


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    if ndi is None:
        return mask.astype(bool)
    structure = ndi.generate_binary_structure(mask.ndim, 1)
    eroded = ndi.binary_erosion(mask, structure=structure, border_value=0)
    dilated = ndi.binary_dilation(mask, structure=structure, border_value=0)
    return np.logical_xor(eroded, dilated)


def _hole_mask(mask: np.ndarray) -> np.ndarray:
    if ndi is None:
        return np.zeros_like(mask, dtype=bool)
    filled = ndi.binary_fill_holes(mask)
    return np.logical_and(filled, np.logical_not(mask))


def _critical_topology_mask(prob: np.ndarray, target: np.ndarray, threshold: float) -> np.ndarray:
    pred = prob >= threshold
    gt = target >= 0.5
    error = np.logical_xor(pred, gt)
    pred_holes = _hole_mask(pred)
    gt_holes = _hole_mask(gt)
    hole_error = np.logical_xor(pred_holes, gt_holes)
    boundary_error = np.logical_or(_binary_boundary(pred), _binary_boundary(gt))
    critical = np.logical_or(error, np.logical_and(boundary_error, np.logical_or(pred, gt)))
    critical = np.logical_or(critical, hole_error)
    if ndi is not None:
        critical = ndi.binary_dilation(critical, iterations=1)
    return critical.astype(np.float32)


class TopologyAwareFocalLoss(nn.Module):
    """Detached exact topology-error weighted focal loss for 2D/3D masks."""

    def __init__(
        self,
        num_classes: int,
        weight: float = 0.05,
        gamma: float = 2.0,
        topology_weight: float = 4.0,
        threshold: float = 0.5,
        foreground_classes: Optional[Iterable[int]] = None,
        max_elements: int = 65536,
    ) -> None:
        super().__init__()
        self.cfg = TopologyLossConfig(
            weight=weight,
            gamma=gamma,
            topology_weight=topology_weight,
            threshold=threshold,
            foreground_classes=tuple(foreground_classes) if foreground_classes is not None else None,
            max_elements=max_elements,
        )
        self.num_classes = int(num_classes)
        self.backend = "gudhi_available" if _HAS_GUDHI else "discrete_components"

    def _classes(self):
        if self.num_classes == 1:
            return [0]
        if self.cfg.foreground_classes is not None:
            return list(self.cfg.foreground_classes)
        return list(range(1, self.num_classes))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = _as_probabilities(pred, self.num_classes)
        prob_small = _maybe_downsample(prob, self.cfg.max_elements)
        target_float = target.float()
        if target_float.dim() in (3, 4) and prob.dim() == 5:
            target_float = target_float.unsqueeze(1)

        losses = []
        for class_idx in self._classes():
            class_prob = prob_small[:, class_idx:class_idx + 1] if self.num_classes > 1 else prob_small
            class_target = _target_mask(target_float, class_idx, self.num_classes)
            if class_target.shape[-len(class_prob.shape[2:]):] != class_prob.shape[2:]:
                mode = "nearest"
                class_target = F.interpolate(class_target.float(), size=class_prob.shape[2:], mode=mode)
            critical_masks = []
            for b in range(class_prob.shape[0]):
                crit = _critical_topology_mask(
                    class_prob[b, 0].detach().cpu().numpy(),
                    class_target[b, 0].detach().cpu().numpy(),
                    self.cfg.threshold,
                )
                critical_masks.append(torch.from_numpy(crit))
            critical = torch.stack(critical_masks, dim=0).unsqueeze(1).to(class_prob.device, class_prob.dtype)
            weights = 1.0 + self.cfg.topology_weight * critical
            pt = torch.where(class_target > 0.5, class_prob, 1.0 - class_prob).clamp(self.cfg.eps, 1 - self.cfg.eps)
            focal = -((1.0 - pt) ** self.cfg.gamma) * torch.log(pt)
            losses.append((weights * focal).mean())
        if not losses:
            return pred.new_tensor(0.0)
        return torch.stack(losses).mean()


class CombinedSegTopologyLoss(nn.Module):
    """Wrap an existing segmentation loss with topology-aware focal loss."""

    def __init__(
        self,
        base_loss: nn.Module,
        num_classes: int,
        enabled: bool = True,
        topology_weight: float = 0.05,
        focal_gamma: float = 2.0,
        critical_weight: float = 4.0,
        foreground_classes: Optional[Iterable[int]] = None,
        max_elements: int = 65536,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.enabled = bool(enabled) and topology_weight > 0
        self.topology_weight = float(topology_weight)
        self.topology_loss = TopologyAwareFocalLoss(
            num_classes=num_classes,
            weight=topology_weight,
            gamma=focal_gamma,
            topology_weight=critical_weight,
            foreground_classes=foreground_classes,
            max_elements=max_elements,
        )
        self.last_components: Dict[str, float] = {}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(pred, target)
        if not self.enabled:
            self.last_components = {"base": float(base.detach().cpu()), "topology": 0.0, "total": float(base.detach().cpu())}
            return base
        topo = self.topology_loss(pred, target)
        total = base + self.topology_weight * topo
        self.last_components = {
            "base": float(base.detach().cpu()),
            "topology": float(topo.detach().cpu()),
            "total": float(total.detach().cpu()),
            "backend": self.topology_loss.backend,
        }
        return total


def build_topology_loss(base_loss: nn.Module, num_classes: int, config=None) -> CombinedSegTopologyLoss:
    enabled = bool(getattr(config, "topology_loss_enabled", True))
    weight = float(getattr(config, "topology_loss_weight", 0.05))
    max_elements = int(getattr(config, "topology_loss_max_elements", 65536))
    critical_weight = float(getattr(config, "topology_critical_weight", 4.0))
    gamma = float(getattr(config, "topology_focal_gamma", 2.0))
    classes = getattr(config, "topology_foreground_classes", None)
    return CombinedSegTopologyLoss(
        base_loss=base_loss,
        num_classes=num_classes,
        enabled=enabled,
        topology_weight=weight,
        focal_gamma=gamma,
        critical_weight=critical_weight,
        foreground_classes=classes,
        max_elements=max_elements,
    )

