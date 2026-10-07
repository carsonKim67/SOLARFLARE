import sys
import json
import torch
import numpy as np

from src.model import SolarUNet
from src.dataset import SolarFilamentDataset
from src.targets import masks_to_target

DATA_DIR = "data/MAGFiLO_1.0_Kaggle_2026"
SPLIT_FILE = "data/train_val_split.json"
DEFAULT_MODEL_FILE = "checkpoints/solar_unet_baseline.pth"

# Use the checkpoint given on the command line, or the default
if len(sys.argv) > 1:
    MODEL_FILE = sys.argv[1]
else:
    MODEL_FILE = DEFAULT_MODEL_FILE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Checkpoint:", MODEL_FILE)

# Load validation IDs
with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

validation_ids = set(split["validation"])

# Load dataset
dataset = SolarFilamentDataset(DATA_DIR)

# Load model (works for plain weights and full checkpoints)
model = SolarUNet(num_classes=5)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

# Store results
intersection = np.zeros(5)
union = np.zeros(5)
predicted_by_class = np.zeros(5)
actual_by_class = np.zeros(5)

# confusion[true_class, predicted_class] = number of pixels
confusion = np.zeros((5, 5), dtype=np.int64)

with torch.no_grad():

    for index, image_info in enumerate(dataset.images):

        if image_info["id"] not in validation_ids:
            continue

        sample = dataset[index]

        image = sample["image"].unsqueeze(0).to(DEVICE)

        target = masks_to_target(
            sample["masks"],
            sample["categories"]
        )

        output = model(image)

        prediction = torch.argmax(
            output,
            dim=1
        )[0].cpu().numpy()

        target = target.numpy()

        # Confusion matrix for this image
        pair_index = target.astype(np.int64) * 5 + prediction.astype(np.int64)
        confusion += np.bincount(
            pair_index.ravel(),
            minlength=25
        ).reshape(5, 5)

        # Count pixels for each class
        for class_id in range(5):

            pred_class = prediction == class_id
            target_class = target == class_id

            predicted_by_class[class_id] += pred_class.sum()
            actual_by_class[class_id] += target_class.sum()

            intersection[class_id] += np.logical_and(
                pred_class,
                target_class
            ).sum()

            union[class_id] += np.logical_or(
                pred_class,
                target_class
            ).sum()

# Display results
print()
print("Validation images checked:", len(validation_ids))
print()

print("Pixel Counts")
print("------------")

for class_id in range(5):
    print(
        f"Class {class_id}: "
        f"Predicted = {int(predicted_by_class[class_id]):,}, "
        f"Actual = {int(actual_by_class[class_id]):,}"
    )

print()
print("Validation IoU")
print("--------------")

ious = []

for class_id in range(5):

    if union[class_id] > 0:
        iou = intersection[class_id] / union[class_id]
    else:
        iou = 0.0

    ious.append(iou)

    print(f"Class {class_id} IoU: {iou:.4f}")

print()
print(f"Mean IoU (classes 1-3): {np.mean(ious[1:4]):.4f}")

# Confusion matrix as percentages of each TRUE class
print()
print("Confusion matrix (rows = true class, columns = predicted, % of true pixels)")
print("-------------------------------------------------------------------------")
print("            Pred0    Pred1    Pred2    Pred3    Pred4")

for true_class in range(4):
    row_total = confusion[true_class].sum()

    if row_total > 0:
        percents = confusion[true_class] / row_total * 100.0
    else:
        percents = np.zeros(5)

    print(
        f"True {true_class}   "
        + "  ".join(f"{p:6.2f}%" for p in percents)
    )