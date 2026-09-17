"""Study evaluation semantics, retained without numerical changes."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
import math
import numpy as np
from .rules import (CLASS_NAMES, PriorArm, apply_anatomical_prior,
                    count_logic_violations, duplicate_prediction_count)
IOU_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]
BACKGROUND = "Background"


@dataclass
class ImageRecord:
    fold: int
    case_id: str
    diagnosis: str
    plane: str
    image: str
    label: str
    shape: tuple[int, int]
    gt_boxes: np.ndarray
    gt_classes: np.ndarray
    pred_boxes: np.ndarray
    pred_scores: np.ndarray
    pred_classes: np.ndarray



def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")



def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)



def summarize(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return mean(clean), stdev(clean)



def mean_sd_rows(
    rows: list[dict[str, object]],
    group_fields: list[str],
    metric_fields: list[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    out = []
    for key, items in sorted(grouped.items(), key=lambda x: tuple(str(v) for v in x[0])):
        out_row = {field: value for field, value in zip(group_fields, key)}
        for metric in metric_fields:
            mu, sd = summarize([safe_float(item.get(metric, float("nan"))) for item in items])
            out_row[f"{metric}_mean"] = mu
            out_row[f"{metric}_sd"] = sd
            out_row[f"{metric}_mean_sd"] = f"{mu:.4f} ± {sd:.4f}" if not math.isnan(mu) else "NA"
        out.append(out_row)
    return out



def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        return image_path.with_suffix(".txt")



def load_gt(label_path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    boxes: list[list[float]] = []
    classes: list[int] = []
    if not label_path.exists():
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int)
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int)
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
        boxes.append([
            (cx - bw / 2.0) * width,
            (cy - bh / 2.0) * height,
            (cx + bw / 2.0) * width,
            (cy + bh / 2.0) * height,
        ])
        classes.append(cls)
    return np.asarray(boxes, dtype=float), np.asarray(classes, dtype=int)



def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=float)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    denom = area_a + area_b - inter
    out = np.zeros_like(inter, dtype=float)
    valid = denom > 0
    out[valid] = inter[valid] / denom[valid]
    return out



def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))



def arm_predictions(record: ImageRecord, arm: PriorArm) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    keep, stats = apply_anatomical_prior(
        record.pred_boxes,
        record.pred_scores,
        record.pred_classes,
        record.plane,
        record.shape,
        arm,
    )
    return (
        record.pred_boxes[keep],
        record.pred_scores[keep],
        record.pred_classes[keep],
        stats,
    )



def detection_counts(
    records: list[ImageRecord],
    arm: PriorArm,
    conf_threshold: float,
    iou_threshold: float = 0.50,
) -> tuple[dict[int, Counter[str]], Counter[str]]:
    per_class: dict[int, Counter[str]] = {i: Counter() for i in range(len(CLASS_NAMES))}
    totals: Counter[str] = Counter()
    for record in records:
        boxes, scores, classes, _ = arm_predictions(record, arm)
        keep = scores >= conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
        order = np.argsort(-scores) if len(scores) else np.asarray([], dtype=int)
        matched_gt: set[int] = set()
        for pred_idx in order:
            pred_cls = int(classes[pred_idx])
            same_gt = np.where(record.gt_classes == pred_cls)[0]
            same_gt = np.asarray([idx for idx in same_gt if idx not in matched_gt], dtype=int)
            if len(same_gt) == 0:
                per_class[pred_cls]["fp"] += 1
                totals["fp"] += 1
                continue
            overlaps = compute_iou(boxes[pred_idx], record.gt_boxes[same_gt])
            best_local = int(np.argmax(overlaps)) if len(overlaps) else -1
            if best_local >= 0 and float(overlaps[best_local]) >= iou_threshold:
                matched = int(same_gt[best_local])
                matched_gt.add(matched)
                per_class[pred_cls]["tp"] += 1
                totals["tp"] += 1
            else:
                per_class[pred_cls]["fp"] += 1
                totals["fp"] += 1
        for gt_idx, gt_cls in enumerate(record.gt_classes):
            if gt_idx not in matched_gt:
                per_class[int(gt_cls)]["fn"] += 1
                totals["fn"] += 1
    return per_class, totals



def ap_for_class(records: list[ImageRecord], arm: PriorArm, class_id: int, iou_threshold: float) -> float:
    n_gt = sum(int(np.sum(record.gt_classes == class_id)) for record in records)
    if n_gt == 0:
        return float("nan")
    predictions: list[tuple[float, int, np.ndarray]] = []
    gt_by_image: dict[int, np.ndarray] = {}
    for image_idx, record in enumerate(records):
        boxes, scores, classes, _ = arm_predictions(record, arm)
        class_pred = np.where(classes == class_id)[0]
        for idx in class_pred:
            predictions.append((float(scores[idx]), image_idx, boxes[idx].copy()))
        gt_by_image[image_idx] = np.where(record.gt_classes == class_id)[0]
    predictions.sort(key=lambda x: x[0], reverse=True)
    matched: dict[int, set[int]] = defaultdict(set)
    tp = np.zeros((len(predictions),), dtype=float)
    fp = np.zeros((len(predictions),), dtype=float)
    for pred_i, (_, image_idx, box) in enumerate(predictions):
        record = records[image_idx]
        gt_indices = [idx for idx in gt_by_image[image_idx] if int(idx) not in matched[image_idx]]
        if not gt_indices:
            fp[pred_i] = 1.0
            continue
        gt_indices_np = np.asarray(gt_indices, dtype=int)
        overlaps = compute_iou(box, record.gt_boxes[gt_indices_np])
        best_local = int(np.argmax(overlaps)) if len(overlaps) else -1
        if best_local >= 0 and float(overlaps[best_local]) >= iou_threshold:
            matched_gt = int(gt_indices_np[best_local])
            matched[image_idx].add(matched_gt)
            tp[pred_i] = 1.0
        else:
            fp[pred_i] = 1.0
    if len(predictions) == 0:
        return 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(1, n_gt)
    precision = cum_tp / np.maximum(1, cum_tp + cum_fp)
    return average_precision(recall, precision)



def map_metrics(records: list[ImageRecord], arm: PriorArm) -> tuple[dict[int, dict[str, float]], float, float]:
    per_class_ap: dict[int, dict[str, float]] = {}
    for class_id in range(len(CLASS_NAMES)):
        ap_values = [ap_for_class(records, arm, class_id, threshold) for threshold in IOU_THRESHOLDS]
        per_class_ap[class_id] = {
            "AP50": ap_values[0],
            "AP50_95": float(np.nanmean(ap_values)) if not all(math.isnan(x) for x in ap_values) else float("nan"),
        }
    valid_ap50 = [row["AP50"] for row in per_class_ap.values() if not math.isnan(row["AP50"])]
    valid_ap5095 = [row["AP50_95"] for row in per_class_ap.values() if not math.isnan(row["AP50_95"])]
    return (
        per_class_ap,
        float(mean(valid_ap50)) if valid_ap50 else float("nan"),
        float(mean(valid_ap5095)) if valid_ap5095 else float("nan"),
    )



def evaluate_records(
    records: list[ImageRecord],
    arm: PriorArm,
    conf_threshold: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    per_class_counts, totals = detection_counts(records, arm, conf_threshold)
    per_class_ap, map50, map5095 = map_metrics(records, arm)
    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    overall = {
        "precision": precision,
        "recall": recall,
        "F1": f1_score(precision, recall),
        "mAP50": map50,
        "mAP50_95": map5095,
        "TP": int(totals["tp"]),
        "FP": int(totals["fp"]),
        "FN": int(totals["fn"]),
    }
    per_class_rows = []
    for class_id, name in enumerate(CLASS_NAMES):
        counts = per_class_counts[class_id]
        p = counts["tp"] / max(1, counts["tp"] + counts["fp"])
        r = counts["tp"] / max(1, counts["tp"] + counts["fn"])
        per_class_rows.append({
            "structure": name,
            "precision": p,
            "recall": r,
            "F1": f1_score(p, r),
            "AP50": per_class_ap[class_id]["AP50"],
            "AP50_95": per_class_ap[class_id]["AP50_95"],
            "TP": int(counts["tp"]),
            "FP": int(counts["fp"]),
            "FN": int(counts["fn"]),
        })
    return overall, per_class_rows



def prediction_quality_counts(
    records: list[ImageRecord],
    arm: PriorArm,
    conf_threshold: float,
) -> Counter[str]:
    out: Counter[str] = Counter()
    for record in records:
        boxes, scores, classes, stats = arm_predictions(record, arm)
        high = scores >= conf_threshold
        high_classes = classes[high]
        out["raw_predictions"] += int(stats["raw_predictions"])
        out["removed_by_spatial"] += int(stats["removed_by_spatial"])
        out["removed_by_plane"] += int(stats["removed_by_plane"])
        out["removed_by_top1"] += int(stats["removed_by_top1"])
        out["predictions_at_threshold"] += int(np.sum(high))
        out["logic_violations"] += count_logic_violations(high_classes, record.plane)
        out["redundant_predictions"] += duplicate_prediction_count(high_classes)
    return out



def clinical_structure_metrics(
    records: list[ImageRecord],
    arm: PriorArm,
    conf_threshold: float,
    iou_threshold: float = 0.50,
) -> list[dict[str, object]]:
    rows = []
    for class_id, name in enumerate(CLASS_NAMES):
        counts = Counter()
        for record in records:
            gt_idx = np.where(record.gt_classes == class_id)[0]
            boxes, scores, classes, _ = arm_predictions(record, arm)
            pred_idx = np.where((classes == class_id) & (scores >= conf_threshold))[0]
            gt_present = len(gt_idx) > 0
            pred_correct = False
            if gt_present and len(pred_idx) > 0:
                for pidx in pred_idx:
                    if float(np.max(compute_iou(boxes[pidx], record.gt_boxes[gt_idx]))) >= iou_threshold:
                        pred_correct = True
                        break
            pred_present = len(pred_idx) > 0
            if gt_present and pred_correct:
                counts["TP"] += 1
            elif gt_present and not pred_correct:
                counts["FN"] += 1
            elif (not gt_present) and pred_present:
                counts["FP"] += 1
            else:
                counts["TN"] += 1
        sensitivity = counts["TP"] / max(1, counts["TP"] + counts["FN"])
        specificity = counts["TN"] / max(1, counts["TN"] + counts["FP"])
        ppv = counts["TP"] / max(1, counts["TP"] + counts["FP"])
        npv = counts["TN"] / max(1, counts["TN"] + counts["FN"])
        rows.append({
            "arm": arm.key,
            "arm_label": arm.label,
            "structure": name,
            "TP": int(counts["TP"]),
            "FP": int(counts["FP"]),
            "TN": int(counts["TN"]),
            "FN": int(counts["FN"]),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "PPV": ppv,
            "NPV": npv,
            "F1": f1_score(ppv, sensitivity),
        })
    return rows



def confusion_counts(
    records: list[ImageRecord],
    arm: PriorArm,
    conf_threshold: float,
    iou_threshold: float = 0.50,
) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        boxes, scores, classes, _ = arm_predictions(record, arm)
        keep = np.where(scores >= conf_threshold)[0]
        order = keep[np.argsort(-scores[keep])] if len(keep) else np.asarray([], dtype=int)
        matched_gt: set[int] = set()
        for pred_idx in order:
            pred_cls = int(classes[pred_idx])
            available = [idx for idx in range(len(record.gt_classes)) if idx not in matched_gt]
            if not available:
                counts[(BACKGROUND, CLASS_NAMES[pred_cls])] += 1
                continue
            available_np = np.asarray(available, dtype=int)
            overlaps = compute_iou(boxes[pred_idx], record.gt_boxes[available_np])
            best_local = int(np.argmax(overlaps)) if len(overlaps) else -1
            if best_local >= 0 and float(overlaps[best_local]) >= iou_threshold:
                gt_idx = int(available_np[best_local])
                matched_gt.add(gt_idx)
                counts[(CLASS_NAMES[int(record.gt_classes[gt_idx])], CLASS_NAMES[pred_cls])] += 1
            else:
                counts[(BACKGROUND, CLASS_NAMES[pred_cls])] += 1
        for gt_idx, gt_cls in enumerate(record.gt_classes):
            if gt_idx not in matched_gt:
                counts[(CLASS_NAMES[int(gt_cls)], BACKGROUND)] += 1
    return counts

