# Anatomical-prior-constrained fetal structure detection

Research code for RT-DETR-L detection of 16 anatomical classes in fetal ultrasound
key frames from normal and tetralogy-of-Fallot (TOF) examinations.

**This is not a TOF classifier, a video pipeline or a clinical diagnostic device.**
The standard view is supplied as metadata. The anatomical prior is deterministic
post-processing; it does not introduce a new detection network or retrain it.

## Included

- Patient-level, normal/TOF-stratified five-fold splitting with an inner validation subset.
- RT-DETR-L training, single-image inference and held-out prediction caching.
- Spatial exclusion, view-specific filtering and top-1 selection, in that order.
- The five reported ablation arms, subgroup results, box-level metrics,
  pooled image-structure sensitivity/specificity/F1, and confusion counts.
- A separate native-validator fixed-split comparison for RT-DETR-L and YOLO models.
- CPU unit tests using synthetic fixtures only.

No clinical images, patient identifiers, annotations, split assignments, trained
weights or prediction caches are distributed. Image and checkpoint files are
ignored by git as an additional precaution; that is not a substitute for a
manual privacy review before uploading.

## Installation

The study used Python 3.11.3, PyTorch 2.9.1+cu126, CUDA 12.6, Ultralytics 8.4.113,
and an NVIDIA GeForce RTX 2080 Ti. Use a new virtual environment; do not replace
packages in an existing study environment without recording the change.

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[detector]"
python -m unittest discover -s tests -v
```

The CPU-only rule/metric tests need only `pip install -e .`, not CUDA or PyTorch.
Pinning versions and a seed does not guarantee bitwise-identical GPU training.
The detector uses operations whose determinism can depend on device and kernels.

## Private data layout

Place private data outside the tracked repository or in the ignored `data/`
directory. A CSV manifest requires the following columns:

```text
case_id,diagnosis,plane,image,label
```

- `case_id`: pseudonymous fetus identifier, shared by its five images.
- `diagnosis`: `normal` or `tof`, used for stratification and subgroup reporting only.
- `plane`: explicit `1` abdominal, `2` four-chamber, `3` LVOT, `4` RVOT, or `5` 3VT.
- `image`, `label`: absolute paths or paths relative to the manifest.
- Each fetus contributes one image per view. YOLO label lines are
  `class_id center_x center_y width height`, normalized to the **uncropped original image**.
- The 16 class IDs are fixed by `src/fetal_anatomy/rules.py`. Do not reorder them.
- Organize labels in the standard parallel `images/` and `labels/` folders for
  Ultralytics training. The manifest label path is used explicitly for custom evaluation.

View labels are **not** automatically predicted and are never obtained from
ground-truth structure boxes. The fixed spatial rule assumes the study's original
display layout. It must not be applied uncritically to already cropped frames,
different scanners or changed coordinate systems.

## Workflow

### 1. Split by fetus

```bash
python -m fetal_anatomy.split --manifest data/manifest.csv --out data/cv5
```

Outer splitting uses `StratifiedKFold(5, shuffle=True, random_state=2026)`.
Within each training/validation portion, `StratifiedShuffleSplit` uses 20% for
validation and seed `2026 + fold`. Cases are lexicographically ordered before
splitting, as in the study. Changing case identifiers/order can change splits:
retain the original approved private split assignments for exact reproduction.
Every fetus appears in one held-out test fold, with no within-fold leakage.

### 2. Train each fold

Obtain the appropriate upstream pretrained RT-DETR-L checkpoint separately and
place it at a local path. Train folds 1 through 5, changing the data/output paths:

```bash
python -m fetal_anatomy.train --data data/cv5/fold1/data.yaml --weights weights/rtdetr-l.pt --output runs/rtdetr_fold1
```

Defaults are in `configs/rtdetr_l.yaml`: 100 epochs maximum, patience 30,
960 x 960 input, batch 2, AdamW, initial learning rate 1e-4, cosine schedule,
seed 2026 and device 0. Validation selects the checkpoint. `--dry-run` displays
the resolved training settings without loading weights or starting training.
Existing run directories are not overwritten. Training augmentation matches
the study settings; flips, MixUp and copy-paste are disabled.

For the exploratory Fold 3 controls, use `--family yolo` and the appropriate
YOLOv8-L or YOLO11-L initial checkpoint. The default YOLO batch is 4, not 2.

### 3. Cache held-out detections once per fold

```bash
python -m fetal_anatomy.predict --manifest data/cv5/fold1/test.csv --weights runs/rtdetr_fold1/weights/best.pt --fold 1 --out cache/fold1
```

Repeat for folds 2 through 5. Low-confidence predictions (>=0.001) are retained
for AP calculation. Cache metadata fingerprints the images, labels, supplied
metadata, weights and software. It is private and must not be pushed to GitHub.
All ablation arms reuse the same predictions; there is no retraining between arms.

### 4. Evaluate

```bash
python -m fetal_anatomy.evaluate --splits data/cv5 --cache cache --out outputs/evaluation
```

CSV outputs include five-fold ablation means/SD, per-fold and subgroup results,
per-structure box metrics, pooled image-structure metrics and raw confusion counts.
False-positive/negative counts per 100 images are computed within each fold
before averaging. Relative FP reduction is also computed per fold first.
No confidence intervals or significance tests are silently added.

### 5. Single-image use

```bash
python -m fetal_anatomy.infer --image data/example.jpg --view 3 --weights runs/rtdetr_fold1/weights/best.pt --output outputs/example.json
```

Output is a list of structure boxes, not a disease probability. Always provide
the correct known view. Do not infer a view from the filename or choose a view
that makes a prediction look more plausible.

### Native-validator supplementary comparison

```bash
python -m fetal_anatomy.native_compare --data data/cv5/fold3/data.yaml --rtdetr runs/rtdetr_fold3/weights/best.pt --yolov8 runs/yolov8_fold3/weights/best.pt --yolo11 runs/yolo11_fold3/weights/best.pt --out outputs/native_comparison
```

These native-validator values use a different operating-point/AP implementation
from the prior-ablation evaluator. Do not directly compare F1 between these outputs.

## Metric definitions and limitations

`metrics.py` retains the numerical function bodies from the study evaluator.
Box-level matching is score-ordered, same-class, one-to-one matching at IoU 0.5.
Precision, recall and F1 use confidence >=0.25. AP integrates the monotonically
interpolated precision-recall curve over confidence-ranked detections; it is not
the Ultralytics AP implementation. Classes without reference positives are
excluded from mAP averaging.

Image-structure metrics contribute one outcome per image/class. A correctly
localized prediction in a reference-positive image is a TP; otherwise it is an
FN. In reference-negative images, any detection is an FP and no detection is a TN.
This does not penalize every redundant/inaccurate box in reference-positive images,
and it is **not disease-diagnostic sensitivity or specificity**. High specificity
partly follows from the known-view whitelist. Zero rule violations is by construction.
Top-1 filtering can remove true detections, and priors cannot recover missed boxes.

The repository does not replace external validation, a clinical reader study,
or a medical-device evaluation. No clinical performance is claimed for unseen data.

## Data and code availability

Clinical data are not public because of privacy and institutional restrictions.
After publication, deidentified data requests may be considered by the corresponding
author, subject to reasonable request, institutional approval and applicable ethics.
Access is not guaranteed by this repository. Weights are not included in this release.

Before public release, the authors must confirm their code-distribution rights and
select a repository license. No license is assigned here on the authors' behalf.
Ultralytics is an external dependency with its own AGPL-3.0/enterprise licensing;
this package does not relicense it. See `THIRD_PARTY_NOTICES.md`.
Update the repository URL and paper citation after publication; no DOI is invented.
