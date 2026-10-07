import sys
import json
import time
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.model_resnet import SolarResNetUNet
from src.dataset import SolarFilamentDataset
from src.targets import masks_to_target

# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = Path("data/MAGFiLO_1.0_Kaggle_2026")
SPLIT_FILE = Path("data/train_val_split.json")
OUTPUT_DIR = Path("checkpoints")

# PowerShell example:
# .\.venv\Scripts\python.exe -m src.train_resnet resnet34_finetune 30 0.00001 checkpoints\resnet34_v1_epoch30.pth 0 2 500 0.55
# val_images=0 uses the full held-out validation set for best-checkpoint selection.
# Usage: python -m src.train_resnet <run_name> <epochs> <learning_rate> <resume_from|none> [val_images] [val_every] [min_area] [threshold] [train_images]
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "resnet34_augpq"
EXTRA_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
LEARNING_RATE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.00003
RESUME_FROM = (
    Path(sys.argv[4])
    if len(sys.argv) > 4 and sys.argv[4].lower() != "none"
    else None
)
VALIDATION_IMAGE_LIMIT = int(sys.argv[5]) if len(sys.argv) > 5 else 0
VALIDATION_EVERY = int(sys.argv[6]) if len(sys.argv) > 6 else 2
VALIDATION_MIN_AREA = int(sys.argv[7]) if len(sys.argv) > 7 else 500
VALIDATION_THRESHOLD = float(sys.argv[8]) if len(sys.argv) > 8 else 0.55
TRAIN_IMAGE_LIMIT = int(sys.argv[9]) if len(sys.argv) > 9 else 0

if len(sys.argv) > 10:
    sys.exit("too many arguments")
if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME:
    sys.exit("run_name must be a plain file name, without directories")
if EXTRA_EPOCHS <= 0 or LEARNING_RATE <= 0:
    sys.exit("epochs and learning_rate must be positive")
if VALIDATION_IMAGE_LIMIT < 0 or VALIDATION_EVERY <= 0 or VALIDATION_MIN_AREA < 0:
    sys.exit("validation image limit/area must be nonnegative and val_every positive")
if not 0.0 < VALIDATION_THRESHOLD < 1.0:
    sys.exit("threshold must be between 0 and 1")
if TRAIN_IMAGE_LIMIT < 0:
    sys.exit("train_images must be nonnegative (0 means all training images)")

CROP_SIZE = 512
BATCH_SIZE = 4          # number of crops per training step
POSITIVE_CROP_PROBABILITY = 0.8
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def capture_rng_state():
    """Capture RNGs using tensor/basic types compatible with safe torch.load."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    """Restore saved RNGs after model construction and checkpoint loading."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "numpy" in state:
        numpy_state = state["numpy"]
        if isinstance(numpy_state, dict):
            numpy_state = (
                numpy_state["algorithm"],
                numpy_state["keys"].cpu().numpy().astype(np.uint32),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        np.random.set_state(numpy_state)
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda"]])


RUN_OUTPUTS = [
    OUTPUT_DIR / f"{RUN_NAME}_latest.pth",
    OUTPUT_DIR / f"{RUN_NAME}_best.pth",
    OUTPUT_DIR / f"{RUN_NAME}.pth",
]
if any(path.exists() for path in RUN_OUTPUTS):
    sys.exit(
        f"A run named '{RUN_NAME}' already has checkpoint files. "
        f"Choose a different run name."
    )
if RESUME_FROM is not None and not RESUME_FROM.is_file():
    sys.exit(f"Resume checkpoint does not exist: {RESUME_FROM}")
if not DATA_DIR.is_dir() or not SPLIT_FILE.is_file():
    sys.exit("Dataset or held-out split is missing; check DATA_DIR and SPLIT_FILE")

# 2 classes (0 = background, 1 = filament of any type)
NUM_CLASSES = 2

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Device
# -----------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)
print("Run name:", RUN_NAME)

# -----------------------------
# Dataset (training images only)
# -----------------------------

dataset = SolarFilamentDataset(DATA_DIR)

with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

validation_ids = set(split["validation"])

train_indices = [
    index for index, info in enumerate(dataset.images)
    if info["id"] not in validation_ids
]

if TRAIN_IMAGE_LIMIT:
    if TRAIN_IMAGE_LIMIT > len(train_indices):
        sys.exit(f"train_images exceeds available training images: {len(train_indices)}")
    train_indices = random.Random(SEED).sample(train_indices, TRAIN_IMAGE_LIMIT)

image_ids = {info["id"] for info in dataset.images}
missing_validation_ids = validation_ids - image_ids
if missing_validation_ids:
    sys.exit(f"Split contains {len(missing_validation_ids)} validation IDs missing from annotations")
validation_indices = [
    index for index, info in enumerate(dataset.images)
    if info["id"] in validation_ids
]
if VALIDATION_IMAGE_LIMIT and VALIDATION_IMAGE_LIMIT < len(validation_indices):
    validation_indices = random.Random(SEED).sample(
        validation_indices, VALIDATION_IMAGE_LIMIT
    )
if not validation_indices:
    sys.exit("No validation images found in the held-out split")

print("Training images:", len(train_indices))
print("Validation images held out:", len(validation_ids))
print("Validation images used for checkpoint selection:", len(validation_indices))
if TRAIN_IMAGE_LIMIT:
    print("SMOKE/limited run: training on a fixed subset of images")

train_dataset = Subset(dataset, train_indices)

# The loader gives one full image at a time;
# crops are collected into batches in the training loop below
loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

# -----------------------------
# Model
# -----------------------------

# Pretrained weights are only needed when starting from scratch
model = SolarResNetUNet(pretrained=(RESUME_FROM is None))
model = model.to(device)

# -----------------------------
# Loss and optimizer
# -----------------------------

class_weights = torch.tensor(
    [0.5, 2.0],
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

# -----------------------------
# Resume from an earlier checkpoint (optional)
# -----------------------------

start_epoch = 0

if RESUME_FROM is not None:
    resume = torch.load(RESUME_FROM, map_location=device)
    if isinstance(resume, dict) and "model_state_dict" in resume:
        model.load_state_dict(resume["model_state_dict"])
        if "optimizer_state_dict" in resume:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        if "scheduler_state_dict" in resume:
            scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_epoch = int(resume.get("epoch", 0))
        if "rng_state" in resume:
            restore_rng_state(resume["rng_state"])
            print("Restored saved random-number-generator state")
        else:
            print("Checkpoint has no RNG state; continuing with the seeded RNG")
    else:
        # Also accept plain model weights, such as a submission checkpoint.
        model.load_state_dict(resume)

    print("Resuming from:", RESUME_FROM)
    print("Already trained epochs:", start_epoch)

best_val_pq = float("-inf")

for group in optimizer.param_groups:
    group["lr"] = LEARNING_RATE

END_EPOCH = start_epoch + EXTRA_EPOCHS

print("Learning rate:", LEARNING_RATE)
print("Crop size:", CROP_SIZE, "| Crops per step:", BATCH_SIZE)
print(
    f"Validation: every {VALIDATION_EVERY} epoch(s), "
    f"{len(validation_indices)} image(s), min area {VALIDATION_MIN_AREA}, "
    f"probability threshold {VALIDATION_THRESHOLD:.2f}"
)
print(f"Training epochs {start_epoch + 1} to {END_EPOCH}")

# -----------------------------
# Dice loss
# -----------------------------

def dice_loss(outputs, targets, smooth=1.0):
    probabilities = torch.softmax(outputs, dim=1)

    targets_one_hot = F.one_hot(
        targets,
        num_classes=NUM_CLASSES
    ).permute(0, 3, 1, 2).float()

    intersection = (
        probabilities * targets_one_hot
    ).sum(dim=(2, 3))

    denominator = (
        probabilities + targets_one_hot
    ).sum(dim=(2, 3))

    dice = (
        (2.0 * intersection + smooth)
        / (denominator + smooth)
    )

    # Skips background (class 0), so this is Dice for filament only
    return 1.0 - dice[:, 1:].mean()

# -----------------------------
# Positive crop selection
# -----------------------------

def get_crop(image, target):
    height, width = target.shape

    crop_h = min(CROP_SIZE, height)
    crop_w = min(CROP_SIZE, width)

    if random.random() < POSITIVE_CROP_PROBABILITY:
        positive_pixels = torch.nonzero(target > 0)

        if len(positive_pixels) > 0:
            index = random.randrange(len(positive_pixels))

            y, x = positive_pixels[index].tolist()

            top = max(0, min(y - crop_h // 2, height - crop_h))
            left = max(0, min(x - crop_w // 2, width - crop_w))

        else:
            top = random.randint(0, height - crop_h)
            left = random.randint(0, width - crop_w)

    else:
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)

    image = image[:, top:top + crop_h, left:left + crop_w]
    target = target[top:top + crop_h, left:left + crop_w]

    # Pad small images up to the full crop size so crops can be stacked
    if crop_h < CROP_SIZE or crop_w < CROP_SIZE:
        padded_image = torch.zeros(1, CROP_SIZE, CROP_SIZE)
        padded_target = torch.zeros(CROP_SIZE, CROP_SIZE, dtype=torch.long)

        padded_image[:, :crop_h, :crop_w] = image
        padded_target[:crop_h, :crop_w] = target

        image, target = padded_image, padded_target

    return image, target


def augment_crop(image, target):
    """Apply label-safe dihedral transforms and mild solar intensity jitter."""
    rotations = random.randrange(4)
    image = torch.rot90(image, rotations, dims=(1, 2))
    target = torch.rot90(target, rotations, dims=(0, 1))
    if random.random() < 0.5:
        image = image.flip(2)
        target = target.flip(1)
    if random.random() < 0.5:
        image = image.flip(1)
        target = target.flip(0)

    contrast = random.uniform(0.9, 1.1)
    brightness = random.uniform(-0.06, 0.06)
    image = (image * contrast + brightness).clamp_(0.0, 1.0)
    return image.contiguous(), target.contiguous()


def evaluate_validation():
    """Compute connected-component PQ on the fixed held-out image subset."""
    model.eval()
    iou_sum = 0.0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.inference_mode():
        for index in validation_indices:
            sample = dataset[index]
            image = sample["image"].unsqueeze(0).to(device)
            foreground_probability = torch.softmax(model(image), dim=1)[0, 1]
            foreground_probability = foreground_probability.cpu().numpy()

            gt_masks = sample["masks"].numpy() > 0
            gt_areas = gt_masks.sum(axis=(1, 2), dtype=np.int64)
            num_gt = len(gt_areas)

            binary = (foreground_probability >= VALIDATION_THRESHOLD).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
            pred_areas = np.bincount(labels.ravel(), minlength=num_labels)
            pred_ids = np.flatnonzero(pred_areas >= VALIDATION_MIN_AREA)
            pred_ids = pred_ids[pred_ids > 0]
            intersections = np.zeros((num_gt, len(pred_ids)), dtype=np.int64)
            for gt_index in range(num_gt):
                counts = np.bincount(labels[gt_masks[gt_index]], minlength=num_labels)
                intersections[gt_index] = counts[pred_ids]

            unions = (
                gt_areas[:, None] + pred_areas[pred_ids][None, :] - intersections
            )
            ious = intersections / np.maximum(unions, 1)
            matches = np.argwhere(ious > 0.5)
            matches = sorted(
                matches.tolist(), key=lambda pair: -ious[pair[0], pair[1]]
            )
            used_gt = set()
            used_pred = set()
            for gt_index, pred_index in matches:
                if gt_index in used_gt or pred_index in used_pred:
                    continue
                used_gt.add(gt_index)
                used_pred.add(pred_index)
                iou_sum += float(ious[gt_index, pred_index])

            image_tp = len(used_gt)
            true_positives += image_tp
            false_positives += len(pred_ids) - image_tp
            false_negatives += num_gt - image_tp

    denominator = true_positives + 0.5 * false_positives + 0.5 * false_negatives
    return iou_sum / denominator if denominator else 0.0


def save_checkpoint(path, checkpoint):
    """Write checkpoints atomically and clean up temporary files on failures."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def make_checkpoint(epoch, loss, validation_pq):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "validation_pq": validation_pq,
        "best_val_pq": best_val_pq,
        "validation_images": len(validation_indices),
        "validation_min_area": VALIDATION_MIN_AREA,
        "validation_threshold": VALIDATION_THRESHOLD,
        "validation_image_ids": [dataset.images[index]["id"] for index in validation_indices],
        "training_images": len(train_indices),
        "crop_size": CROP_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "seed": SEED,
        "rng_state": capture_rng_state(),
    }


# Save a measured starting point too, so fine-tuning can never end without a
# usable best checkpoint if its first updates happen to reduce validation PQ.
baseline_val_pq = evaluate_validation()
scheduler.step(baseline_val_pq)
best_val_pq = baseline_val_pq
save_checkpoint(
    OUTPUT_DIR / f"{RUN_NAME}_best.pth",
    make_checkpoint(start_epoch, None, baseline_val_pq)
)
print(f"Starting validation PQ: {baseline_val_pq:.4f}")


# -----------------------------
# One training step on a batch of crops
# -----------------------------

def train_step(crops, targets):
    images = torch.stack(crops).to(device)
    labels = torch.stack(targets).to(device)

    optimizer.zero_grad(set_to_none=True)

    outputs = model(images)

    loss = (
        criterion(outputs, labels)
        + dice_loss(outputs, labels)
    )

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite training loss: {loss.item()}")

    loss.backward()
    optimizer.step()

    return loss.item()

# -----------------------------
# Training
# -----------------------------

for epoch in range(start_epoch, END_EPOCH):
    model.train()

    epoch_start = time.time()

    total_loss = 0.0
    steps = 0

    crops = []
    targets = []

    for batch in loader:
        image = batch["image"][0]
        masks = batch["masks"][0]
        categories = batch["categories"][0]

        target = masks_to_target(masks, categories)

        # Merge Left / Right / Unidentifiable into one filament class
        target = (target > 0).long()

        crop, target_crop = get_crop(image, target)
        crop, target_crop = augment_crop(crop, target_crop)

        crops.append(crop)
        targets.append(target_crop)

        if len(crops) == BATCH_SIZE:
            total_loss += train_step(crops, targets)
            steps += 1
            crops = []
            targets = []

    # Include all crops; dropping a one-item final batch wastes training data.
    if crops:
        total_loss += train_step(crops, targets)
        steps += 1

    average_loss = total_loss / max(steps, 1)
    should_validate = (
        (epoch + 1) % VALIDATION_EVERY == 0 or epoch + 1 == END_EPOCH
    )
    validation_pq = evaluate_validation() if should_validate else None
    if validation_pq is not None:
        scheduler.step(validation_pq)
        if validation_pq > best_val_pq:
            best_val_pq = validation_pq
            save_checkpoint(
                OUTPUT_DIR / f"{RUN_NAME}_best.pth",
                make_checkpoint(epoch + 1, average_loss, validation_pq)
            )
            print(f"  New best validation PQ: {validation_pq:.4f}")

    if validation_pq is None:
        validation_summary = "skipped"
    else:
        validation_summary = f"{validation_pq:.4f}"
    print(
        f"Epoch {epoch + 1}/{END_EPOCH} "
        f"- Loss: {average_loss:.4f} - Val PQ: {validation_summary}"
    )
    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e} - {time.time() - epoch_start:.0f}s")

    checkpoint = make_checkpoint(epoch + 1, average_loss, validation_pq)
    save_checkpoint(OUTPUT_DIR / f"{RUN_NAME}_latest.pth", checkpoint)
    save_checkpoint(OUTPUT_DIR / f"{RUN_NAME}.pth", model.state_dict())

print("Training complete!")
print("Best checkpoint:", OUTPUT_DIR / f"{RUN_NAME}_best.pth")
print("Latest checkpoint:", OUTPUT_DIR / f"{RUN_NAME}_latest.pth")
