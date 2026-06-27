# Task 3 — Pool-Ball Detection: YOLOv11 vs DETR

## Objective and setup
This work addresses the detection of balls in images of 8-ball pool tables and compares two
families of object detectors under identical conditions. The data is the Roboflow
**8-Ball Pool** set (247 images). All annotations are mapped to a single **`ball`** class,
so the task is framed as detection, localization, and counting. A deterministic
**70/15/15** partition (seed 42) yields **172 training, 37 validation, and 38 test** images.
Two architectures are fine-tuned from COCO-pretrained checkpoints and evaluated through a
shared protocol — COCO mean Average Precision (`pycocotools`) together with
precision, recall, and F1 (greedy IoU-0.5 matching) — on the held-out **test** split:

- **YOLOv11s** — a single-stage convolutional detector (9.4 M parameters).
- **DETR-R50** — a transformer-based set-prediction detector (≈41.5 M parameters).

## Methodology
The pipeline is organized into independent and reproducible stages.

- **Label preparation.** Each ball type (solid, striped, cue, black) is mapped to a single
  `ball` class, and the dataset's non-ball **`Dot`** category — the rail sight-markers — is
  excluded so that the target represents only genuine balls. Annotations are then converted
  to each framework's native format (normalized boxes for YOLO; COCO JSON for DETR).
- **Transfer learning.** Both detectors are initialized from COCO-pretrained weights and
  fine-tuned with a newly attached single-class head, a necessary choice given the limited
  amount of data. Training is configured for a 6 GB GPU through mixed precision, small
  batches, gradient accumulation (DETR), and reduced input resolution.
- **Unified evaluation.** A single harness scores every model on the same held-out test
  split, combining **threshold-independent ranking metrics** (mAP@0.50 and mAP@[.50:.95])
  with **operational metrics** computed at a fixed confidence (precision, recall, and F1).
  This combination characterizes both detection quality and the practical counting
  behavior, and ensures that the two architectures are directly comparable.

## Results and discussion

| Model      | mAP@0.50 | mAP@[.50:.95] | precision | recall |  F1   | params |
|------------|:--------:|:-------------:|:---------:|:------:|:-----:|:------:|
| **YOLOv11s** | **0.985** | **0.833**   | **0.989** | **0.980** | **0.984** | 9.4 M |
| DETR-R50   |  0.664   |     0.213     |   0.623   | 0.755  | 0.683 | ≈41.5 M |

**Strengths.** YOLOv11 attains near-saturated detection performance (mAP@0.50 = 0.985,
F1 = 0.984), and its recall of 0.980 indicates that the balls present are reliably found and
therefore correctly counted. The model also generalizes well: the validation scores observed
during training (0.995 / 0.818) are closely matched by the independent test set
(0.985 / 0.833), and the training and validation losses decrease in parallel, which is
consistent with a well-fitted rather than an overfitted model. These results are obtained
with roughly four times fewer parameters and a substantially lower training cost than DETR.

**Limitations.** Even for YOLOv11, accurate localization remains the more demanding regime:
mAP@[.50:.95] (0.833) is appreciably lower than mAP@0.50 (0.985), indicating that the
remaining errors correspond to slightly imprecise boxes, most plausibly on clustered or
partially occluded balls. DETR is the principal limitation of the comparison: it localizes
poorly (mAP@[.50:.95] = 0.213) and exhibits markedly lower precision (0.623) than recall
(0.755), meaning that it detects balls but also produces many false positives with imprecise
boxes. This behavior is attributable to data scale, as transformer detectors are data-hungry
and 172 training images are insufficient for DETR to converge; its training loss reached a
minimum early and did not improve thereafter.

## Conclusion
For this task and dataset, **YOLOv11s is the recommended model**, providing accurate and
well-generalizing detection at low computational cost, whereas DETR-R50 is constrained by the
limited training data. The principal enablers were transfer learning and clean single-class
labels. Several limitations should be acknowledged: the results derive from a single, small
test split (38 images) and therefore carry sampling variance; tight-box localization and
crowded scenes remain the most difficult cases; and a fair assessment of DETR would require
considerably more data, together with per-epoch validation logging.
