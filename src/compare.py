import sys
import json
import cv2
import numpy as np
import torch

from src.model import SolarUNet
from src.dataset import SolarFilamentDataset
from src.targets import masks_to_target

DATA_DIR = "data/MAGFiLO_1.0_Kaggle_2026"
SPLIT_FILE = "data/train_val_split.json"
OUTPUT_FILE = "compare.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Usage: python -m src.compare <checkpoint> [n]
# n = which validation image to show (0 = first one)
MODEL_FILE = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/weights_05_20_best_so_far.pth"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# Colors in BGR: Left = blue, Right = red, Unidentifiable = green, Ambiguous = yellow
COLORS = {
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (0, 255, 0),
    4: (0, 255, 255),
}

with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

validation_ids = set(split["validation"])

dataset = SolarFilamentDataset(DATA_DIR)

validation_indices = [
    index for index, info in enumerate(dataset.images)
    if info["id"] in validation_ids
]

index = validation_indices[N]

# Load model (works for plain weights and full checkpoints)
model = SolarUNet(num_classes=5)

checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

sample = dataset[index]

target = masks_to_target(
    sample["masks"],
    sample["categories"]
).numpy()

image = sample["image"].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(image)

prediction = torch.argmax(output, dim=1)[0].cpu().numpy()

gray = (sample["image"][0].numpy() * 255).astype(np.uint8)


def overlay(gray_image, mask):
    bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    for class_id, color in COLORS.items():
        bgr[mask == class_id] = color
    return bgr


def prepare(img, label):
    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
    cv2.putText(
        img, label, (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3
    )
    return img


panel_image = prepare(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Image")
panel_truth = prepare(overlay(gray, target), "Ground truth")
panel_prediction = prepare(overlay(gray, prediction), "Prediction")

combined = np.hstack([panel_image, panel_truth, panel_prediction])

cv2.imwrite(OUTPUT_FILE, combined)

print("Checkpoint:", MODEL_FILE)
print("Image:", sample["filename"])
print()

for class_id in range(1, 4):
    print(
        f"Class {class_id}: "
        f"Actual = {int((target == class_id).sum()):,}, "
        f"Predicted = {int((prediction == class_id).sum()):,}"
    )

print()
print("Saved:", OUTPUT_FILE)
print("Colors: Left = blue, Right = red, Unidentifiable = green")