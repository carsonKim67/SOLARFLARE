import sys
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.model import SolarUNet
from src.dataset import SolarFilamentDataset
from src.targets import masks_to_target

# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = Path("data/MAGFiLO_1.0_Kaggle_2026")
SPLIT_FILE = Path("data/train_val_split.json")
OUTPUT_DIR = Path("checkpoints")

# Usage: python -m src.train <run_name> <resume_from or none> <extra_epochs> <learning_rate>
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "binary_p08_long2"

if len(sys.argv) > 2:
    if sys.argv[2].lower() == "none":
        RESUME_FROM = None
    else:
        RESUME_FROM = Path(sys.argv[2])
else:
    RESUME_FROM = Path("checkpoints/binary_p08_long_latest.pth")

EXTRA_EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 30

BATCH_SIZE = 1

# CHANGED: learning rate can now be given on the command line
LEARNING_RATE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.001

CROP_SIZE = 256
POSITIVE_CROP_PROBABILITY = 0.8

# 2 classes (0 = background, 1 = filament of any type)
NUM_CLASSES = 2

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Never overwrite an existing run
if (OUTPUT_DIR / f"{RUN_NAME}_latest.pth").exists():
    sys.exit(
        f"A run named '{RUN_NAME}' already exists in checkpoints. "
        f"Choose a different run name."
    )

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

print("Training images:", len(train_indices))
print("Validation images held out:", len(validation_ids))

train_dataset = Subset(dataset, train_indices)

loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# -----------------------------
# Model
# -----------------------------

model = SolarUNet(num_classes=NUM_CLASSES)
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

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# -----------------------------
# Resume from an earlier checkpoint
# -----------------------------

start_epoch = 0

if RESUME_FROM is not None:
    resume = torch.load(RESUME_FROM, map_location=device)

    model.load_state_dict(resume["model_state_dict"])
    optimizer.load_state_dict(resume["optimizer_state_dict"])

    start_epoch = resume["epoch"]

    print("Resuming from:", RESUME_FROM)
    print("Already trained epochs:", start_epoch)

# CHANGED: loading the optimizer also restores its old learning rate,
# so set the requested learning rate again afterwards
for group in optimizer.param_groups:
    group["lr"] = LEARNING_RATE

print("Learning rate:", LEARNING_RATE)

END_EPOCH = start_epoch + EXTRA_EPOCHS

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

            top = max(
                0,
                min(
                    y - crop_h // 2,
                    height - crop_h
                )
            )

            left = max(
                0,
                min(
                    x - crop_w // 2,
                    width - crop_w
                )
            )

        else:
            top = random.randint(
                0,
                height - crop_h
            )

            left = random.randint(
                0,
                width - crop_w
            )

    else:
        top = random.randint(
            0,
            height - crop_h
        )

        left = random.randint(
            0,
            width - crop_w
        )

    image = image[
        :,
        top:top + crop_h,
        left:left + crop_w
    ]

    target = target[
        top:top + crop_h,
        left:left + crop_w
    ]

    return image, target

# -----------------------------
# Training
# -----------------------------

for epoch in range(start_epoch, END_EPOCH):
    model.train()

    total_loss = 0.0
    batches = 0

    for batch in loader:
        images = batch["image"]
        masks = batch["masks"]
        categories = batch["categories"]

        for i in range(images.shape[0]):
            image = images[i]

            target = masks_to_target(
                masks[i],
                categories[i]
            )

            # Merge Left / Right / Unidentifiable into one filament class
            target = (target > 0).long()

            image, target = get_crop(
                image,
                target
            )

            image = image.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)

            optimizer.zero_grad()

            outputs = model(image)

            loss = (
                criterion(outputs, target)
                + dice_loss(outputs, target)
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            batches += 1

    average_loss = (
        total_loss / max(batches, 1)
    )

    print(
        f"Epoch {epoch + 1}/{END_EPOCH} "
        f"- Loss: {average_loss:.4f}"
    )

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": average_loss
    }

    torch.save(
        checkpoint,
        OUTPUT_DIR / f"{RUN_NAME}_latest.pth"
    )

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / f"{RUN_NAME}.pth"
    )

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / f"{RUN_NAME}_epoch{epoch + 1:02d}.pth"
    )

print("Training complete!")