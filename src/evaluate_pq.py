import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model_resnet import build_model_from_state
from src.dataset import SolarFilamentDataset

DATA_DIR = "data/MAGFiLO_1.0_Kaggle_2026"
SPLIT_FILE = "data/train_val_split.json"

# Usage: python -m src.evaluate_pq <checkpoint> [min_areas] [views] [thresholds] [closing_radii] [image_limit]
# min_areas, thresholds, and closing_radii are comma-separated lists.
# views: 1 = normal, 2 = + horizontal flip, 4 = + both flips
# A closing radius of zero disables morphological closing.
MODEL_FILE = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/resnet34_v1_epoch30.pth"
MIN_AREAS = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
VIEWS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
PROBABILITY_THRESHOLDS = (
    [float(v) for v in sys.argv[4].split(",")]
    if len(sys.argv) > 4 else [0.5]
)
CLOSING_RADII = (
    [int(v) for v in sys.argv[5].split(",")]
    if len(sys.argv) > 5 else [0]
)
IMAGE_LIMIT = int(sys.argv[6]) if len(sys.argv) > 6 else None

if not MIN_AREAS or any(area < 0 for area in MIN_AREAS):
    sys.exit("min_areas must be a comma-separated list of nonnegative integers")
if VIEWS not in (1, 2, 4):
    sys.exit("views must be 1, 2 or 4")
if not PROBABILITY_THRESHOLDS or any(
    threshold <= 0.0 or threshold >= 1.0
    for threshold in PROBABILITY_THRESHOLDS
):
    sys.exit("thresholds must be comma-separated numbers between 0 and 1")
if not CLOSING_RADII or any(radius < 0 for radius in CLOSING_RADII):
    sys.exit("closing_radii must be a comma-separated list of nonnegative integers")

PROBABILITY_THRESHOLDS = list(dict.fromkeys(PROBABILITY_THRESHOLDS))
MIN_AREAS = list(dict.fromkeys(MIN_AREAS))
CLOSING_RADII = list(dict.fromkeys(CLOSING_RADII))
if IMAGE_LIMIT is not None and IMAGE_LIMIT <= 0:
    sys.exit("image_limit must be a positive integer")
if len(sys.argv) > 7:
    sys.exit("too many arguments")
if not Path(MODEL_FILE).is_file():
    sys.exit(f"Checkpoint does not exist: {MODEL_FILE}")
if not Path(DATA_DIR).is_dir():
    sys.exit(f"Dataset directory does not exist: {DATA_DIR}")
if not Path(SPLIT_FILE).is_file():
    sys.exit(f"Validation split does not exist: {SPLIT_FILE}")

IOU_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Checkpoint:", MODEL_FILE)
print("Minimum piece sizes to test:", MIN_AREAS)
print("Views averaged per image:", VIEWS)
print("Foreground probability thresholds:", PROBABILITY_THRESHOLDS)
print("Morphological closing radii:", CLOSING_RADII)
if IMAGE_LIMIT is not None:
    print("Validation image limit:", IMAGE_LIMIT)

with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

validation_ids = set(split["validation"])

dataset = SolarFilamentDataset(DATA_DIR)

checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state = checkpoint["model_state_dict"]
else:
    state = checkpoint

# Picks the small U-Net or the ResNet U-Net automatically
model = build_model_from_state(state)
print("Model type:", type(model).__name__)

model.to(DEVICE)
model.eval()


# Average class probabilities across the requested test-time flips.
def predict_foreground_probability(image):
    probabilities = torch.softmax(model(image), dim=1)

    if VIEWS >= 2:
        flipped = torch.softmax(model(image.flip(3)), dim=1)
        probabilities = probabilities + flipped.flip(3)

    if VIEWS >= 4:
        flipped = torch.softmax(model(image.flip(2)), dim=1)
        probabilities = probabilities + flipped.flip(2)

        flipped = torch.softmax(model(image.flip(2).flip(3)), dim=1)
        probabilities = probabilities + flipped.flip(3).flip(2)

    probabilities = probabilities / VIEWS
    return probabilities[:, 1:].sum(dim=1)[0].cpu().numpy()


# Track instance PQ for each connected-component cutoff and pixel threshold.
stats = {
    (min_area, threshold, radius): {
        "iou_sum": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "pieces": 0,
        "pred_pixels": 0,
        "per_image": []
    }
    for min_area in MIN_AREAS
    for threshold in PROBABILITY_THRESHOLDS
    for radius in CLOSING_RADII
}

total_gt = 0
actual_filament_pixels = 0
checked = 0

with torch.no_grad():

    for index, image_info in enumerate(dataset.images):

        if image_info["id"] not in validation_ids:
            continue

        sample = dataset[index]

        image = sample["image"].unsqueeze(0).to(DEVICE)

        foreground_probability = predict_foreground_probability(image)

        # True filaments: one mask per annotation
        gt_masks = sample["masks"].numpy() > 0
        num_gt = gt_masks.shape[0]

        total_gt += num_gt
        actual_filament_pixels += int(gt_masks.any(axis=0).sum())
        gt_areas = gt_masks.sum(axis=(1, 2), dtype=np.int64)

        for threshold in PROBABILITY_THRESHOLDS:
            thresholded = (foreground_probability >= threshold).astype(np.uint8)

            for radius in CLOSING_RADII:
                if radius:
                    kernel_size = 2 * radius + 1
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (kernel_size, kernel_size)
                    )
                    binary = cv2.morphologyEx(
                        thresholded,
                        cv2.MORPH_CLOSE,
                        kernel
                    )
                else:
                    binary = thresholded

                num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
                pred_areas = np.bincount(labels.ravel(), minlength=num_labels)

                inter_all = np.zeros((num_gt, num_labels), dtype=np.int64)
                for g in range(num_gt):
                    inter_all[g] = np.bincount(
                        labels[gt_masks[g]],
                        minlength=num_labels
                    )

                for min_area in MIN_AREAS:
                    pred_ids = np.nonzero(pred_areas >= min_area)[0]
                    pred_ids = pred_ids[pred_ids > 0]
                    num_pred = len(pred_ids)
                    inter = inter_all[:, pred_ids]
                    union = (
                        pred_areas[pred_ids][None, :]
                        + gt_areas[:, None]
                        - inter
                    )
                    iou_matrix = inter / np.maximum(union, 1)

                    # For IoU > 0.5, valid panoptic matches are one-to-one.
                    pairs = np.argwhere(iou_matrix > IOU_THRESHOLD)
                    pairs = sorted(
                        pairs.tolist(),
                        key=lambda pair: -iou_matrix[pair[0], pair[1]]
                    )
                    used_gt = set()
                    used_pred = set()
                    image_iou_sum = 0.0
                    image_tp = 0

                    for g, p in pairs:
                        if g in used_gt or p in used_pred:
                            continue
                        used_gt.add(g)
                        used_pred.add(p)
                        image_iou_sum += iou_matrix[g, p]
                        image_tp += 1

                    image_fp = num_pred - image_tp
                    image_fn = num_gt - image_tp
                    s = stats[(min_area, threshold, radius)]
                    s["iou_sum"] += image_iou_sum
                    s["tp"] += image_tp
                    s["fp"] += image_fp
                    s["fn"] += image_fn
                    s["pieces"] += num_pred
                    s["pred_pixels"] += int(pred_areas[pred_ids].sum())

                    image_denominator = image_tp + 0.5 * image_fp + 0.5 * image_fn
                    if image_denominator > 0:
                        s["per_image"].append(image_iou_sum / image_denominator)

        checked += 1

        if checked % 50 == 0:
            print(f"  ...{checked} images done")
        if IMAGE_LIMIT is not None and checked >= IMAGE_LIMIT:
            break

print()
print("Validation images checked:", checked)
print(f"True filaments (all images):  {total_gt}")
print(f"Actual filament pixels:       {actual_filament_pixels:,}")
print()

print(
    f"{'MinSize':>8} {'Threshold':>10} {'CloseR':>7} {'Pieces':>8} "
    f"{'TP':>6} {'FP':>8} {'FN':>6} {'PredPixels':>12} {'PQ(all)':>9} {'PQ(avg)':>9}"
)

for min_area in MIN_AREAS:
    for threshold in PROBABILITY_THRESHOLDS:
        for radius in CLOSING_RADII:
            s = stats[(min_area, threshold, radius)]
            denominator = s["tp"] + 0.5 * s["fp"] + 0.5 * s["fn"]
            pq_all = s["iou_sum"] / denominator if denominator > 0 else 0.0
            pq_avg = float(np.mean(s["per_image"])) if s["per_image"] else 0.0

            print(
                f"{min_area:>8} {threshold:>10.3f} {radius:>7} "
                f"{s['pieces']:>8} {s['tp']:>6} {s['fp']:>8} {s['fn']:>6} "
                f"{s['pred_pixels']:>12,} {pq_all:>9.4f} {pq_avg:>9.4f}"
            )