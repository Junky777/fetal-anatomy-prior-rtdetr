"""Anatomical prior rules for fetal echocardiographic structure detection.

The rules here are intentionally post-processing rules. They do not change the
RT-DETR architecture or retrain the detector; they make the clinically motivated
constraints explicit and reproducible for ablation analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CLASS_NAMES = [
    "Spine", "Stomach", "Heart", "LA", "RA", "LV", "RV", "IVS",
    "ECD", "LVOT", "RVOT", "Asc_AO", "Desc_AO", "PA", "Arch", "DA",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


PLANE_ALLOWED_STRUCTURES = {
    "1": ["Spine", "Stomach", "Desc_AO"],
    "2": ["Heart", "LA", "RA", "Desc_AO", "RV", "LV", "Spine", "ECD", "IVS"],
    "3": ["Heart", "IVS", "LVOT", "Asc_AO", "Spine"],
    "4": ["RVOT", "PA", "Asc_AO"],
    "5": ["Spine", "Arch", "DA", "Desc_AO", "PA", "Asc_AO"],
}
PLANE_ALLOWED_CLASS_IDS = {
    plane: {CLASS_TO_ID[name] for name in names}
    for plane, names in PLANE_ALLOWED_STRUCTURES.items()
}


# Normalized [x_min, y_min, x_max, y_max] regions. This reproduces the old
# clinical prior used to remove predictions in the lower-left ultrasound inset.
DEFAULT_SPATIAL_EXCLUSION_REGIONS = [
    (0.0, 0.70, 0.25, 1.0),
]


@dataclass(frozen=True)
class PriorArm:
    key: str
    label: str
    use_spatial: bool = False
    use_plane: bool = False
    use_top1: bool = False


PRIOR_ARMS = [
    PriorArm("baseline", "RT-DETR without anatomical prior"),
    PriorArm("spatial_only", "RT-DETR + spatial exclusion constraint", use_spatial=True),
    PriorArm("plane_only", "RT-DETR + plane semantic constraint", use_plane=True),
    PriorArm("plane_top1", "RT-DETR + plane semantic + top-1 constraint", use_plane=True, use_top1=True),
    PriorArm("full_prior", "RT-DETR + full anatomical prior", use_spatial=True, use_plane=True, use_top1=True),
]
PRIOR_ARM_BY_KEY = {arm.key: arm for arm in PRIOR_ARMS}


def allowed_class_ids_for_plane(plane_id: str | int | None) -> set[int]:
    if plane_id is None:
        return set()
    return set(PLANE_ALLOWED_CLASS_IDS.get(str(plane_id), set()))


def centers_in_exclusion_regions(
    boxes: np.ndarray,
    image_shape: tuple[int, int],
    regions: list[tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    """Return a boolean mask for boxes whose centers fall in excluded regions."""
    boxes = np.asarray(boxes, dtype=float)
    mask = np.zeros((len(boxes),), dtype=bool)
    if len(boxes) == 0:
        return mask
    height, width = image_shape
    if height <= 0 or width <= 0:
        return mask
    active_regions = regions if regions is not None else DEFAULT_SPATIAL_EXCLUSION_REGIONS
    cx = ((boxes[:, 0] + boxes[:, 2]) / 2.0) / float(width)
    cy = ((boxes[:, 1] + boxes[:, 3]) / 2.0) / float(height)
    for x_min, y_min, x_max, y_max in active_regions:
        mask |= (cx >= x_min) & (cx <= x_max) & (cy >= y_min) & (cy <= y_max)
    return mask


def duplicate_prediction_count(classes: np.ndarray) -> int:
    """Count same-class extra predictions beyond the highest-scored instance."""
    if len(classes) == 0:
        return 0
    _, counts = np.unique(np.asarray(classes, dtype=int), return_counts=True)
    return int(sum(max(0, int(count) - 1) for count in counts))


def apply_anatomical_prior(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    plane_id: str | int | None,
    image_shape: tuple[int, int],
    arm: PriorArm,
    spatial_regions: list[tuple[float, float, float, float]] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply an ablation arm and return kept indices plus removal counts."""
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    classes = np.asarray(classes, dtype=int)
    keep = np.ones((len(classes),), dtype=bool)
    stats = {
        "raw_predictions": int(len(classes)),
        "removed_by_spatial": 0,
        "removed_by_plane": 0,
        "removed_by_top1": 0,
    }
    if len(classes) == 0:
        return np.array([], dtype=int), stats

    if arm.use_spatial:
        spatial_mask = ~centers_in_exclusion_regions(boxes, image_shape, spatial_regions)
        stats["removed_by_spatial"] = int(np.sum(keep & ~spatial_mask))
        keep &= spatial_mask

    if arm.use_plane:
        allowed = allowed_class_ids_for_plane(plane_id)
        if allowed:
            plane_mask = np.array([int(cls) in allowed for cls in classes], dtype=bool)
            stats["removed_by_plane"] = int(np.sum(keep & ~plane_mask))
            keep &= plane_mask

    if arm.use_top1:
        top1 = np.zeros((len(classes),), dtype=bool)
        for cls_id in sorted(set(int(x) for x in classes[keep])):
            candidate_idx = np.where(keep & (classes == cls_id))[0]
            if len(candidate_idx) == 0:
                continue
            best = candidate_idx[np.argmax(scores[candidate_idx])]
            top1[best] = True
        stats["removed_by_top1"] = int(np.sum(keep & ~top1))
        keep &= top1

    return np.where(keep)[0], stats


def count_logic_violations(classes: np.ndarray, plane_id: str | int | None) -> int:
    allowed = allowed_class_ids_for_plane(plane_id)
    if not allowed:
        return 0
    return int(sum(1 for cls in np.asarray(classes, dtype=int) if int(cls) not in allowed))
