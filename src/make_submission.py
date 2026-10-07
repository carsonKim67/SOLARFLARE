import csv
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from src.model_resnet import build_model_from_state

TEST_DIR = Path("data/MAGFiLO_1.0_Kaggle_2026/test/test_images")
OUTPUT_DIR = Path("submissions")
IMAGE_SIZE = 2048

# Usage: python -m src.make_submission <checkpoint> [min_area] [views] [threshold] [closing_radius] [output_csv]
# views: 1 = normal, 2 = + horizontal flip, 4 = + both flips
MODEL_FILE = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/resnet34_v1_more_epoch43.pth"
MIN_AREA = int(sys.argv[2]) if len(sys.argv) > 2 else 500
VIEWS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
PROBABILITY_THRESHOLD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.55
CLOSING_RADIUS = int(sys.argv[5]) if len(sys.argv) > 5 else 0
REQUESTED_OUTPUT_FILE = Path(sys.argv[6]) if len(sys.argv) > 6 else None

if len(sys.argv) > 7:
    sys.exit("too many arguments")
if VIEWS not in (1, 2, 4):
    sys.exit("views must be 1, 2 or 4")
if MIN_AREA < 0:
    sys.exit("min_area must be nonnegative")
if not 0.0 < PROBABILITY_THRESHOLD < 1.0:
    sys.exit("threshold must be between 0 and 1")
if CLOSING_RADIUS < 0:
    sys.exit("closing_radius must be nonnegative")
if not Path(MODEL_FILE).is_file():
    sys.exit(f"Checkpoint does not exist: {MODEL_FILE}")
if not TEST_DIR.is_dir():
    sys.exit(f"Test image directory does not exist: {TEST_DIR}")
if not any(
    path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    for path in TEST_DIR.iterdir()
):
    sys.exit(f"No supported test images found in {TEST_DIR}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

threshold_tag = f"{PROBABILITY_THRESHOLD:.2f}".replace(".", "p")
output_file = REQUESTED_OUTPUT_FILE or (OUTPUT_DIR / (
    f"{Path(MODEL_FILE).stem}_min{MIN_AREA}_v{VIEWS}_"
    f"p{threshold_tag}_c{CLOSING_RADIUS}.csv"
))
if output_file.suffix.lower() != ".csv":
    sys.exit("output_csv must have a .csv extension")
if output_file.exists():
    sys.exit(f"Refusing to overwrite existing submission: {output_file}")
output_file.parent.mkdir(parents=True, exist_ok=True)

print("Checkpoint:", MODEL_FILE)
print("Minimum piece size (pixels):", MIN_AREA)
print("Views averaged per image:", VIEWS)
print("Foreground probability threshold:", PROBABILITY_THRESHOLD)
print("Morphological closing radius:", CLOSING_RADIUS)
print("Output file:", output_file)

# Load model (works for plain weights and full checkpoints)
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


image_paths = sorted(
    p for p in TEST_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

print("Test images found:", len(image_paths))

rows = []
images_without_pieces = 0
resized_images = 0
roundtrip_checked = False
seen_ids = set()

bad_characters = set(',"\'')

with torch.no_grad():

    for number, path in enumerate(image_paths, start=1):

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if gray is None:
            raise FileNotFoundError(f"Could not read {path}")

        image = torch.from_numpy(gray.astype(np.float32) / 255.0)
        image = image.unsqueeze(0).unsqueeze(0).to(DEVICE)

        foreground_probability = predict_foreground_probability(image)
        binary = (
            foreground_probability >= PROBABILITY_THRESHOLD
        ).astype(np.uint8)

        if CLOSING_RADIUS:
            kernel_size = 2 * CLOSING_RADIUS + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size)
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # The competition expects 2048 x 2048 masks
        if binary.shape != (IMAGE_SIZE, IMAGE_SIZE):
            resized_images += 1
            binary = cv2.resize(
                binary,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_NEAREST
            )

        # Split into separate pieces
        num_labels, labels = cv2.connectedComponents(
            binary,
            connectivity=8
        )

        areas = np.bincount(
            labels.ravel(),
            minlength=num_labels
        )

        piece_number = 0

        for label_id in range(1, num_labels):

            if areas[label_id] < MIN_AREA:
                continue

            piece_number += 1

            mask = np.asfortranarray(
                (labels == label_id).astype(np.uint8)
            )

            rle = mask_utils.encode(mask)
            counts = rle["counts"].decode("utf-8")

            if bad_characters & set(counts):
                raise ValueError(
                    f"Unexpected character in RLE for {path.stem}"
                )

            # Check once that the encoding decodes back to the same mask
            if not roundtrip_checked:
                decoded = mask_utils.decode({
                    "size": [IMAGE_SIZE, IMAGE_SIZE],
                    "counts": counts.encode("utf-8")
                })

                if not np.array_equal(decoded, mask):
                    raise ValueError("RLE round-trip check FAILED")

                roundtrip_checked = True
                print("RLE round-trip check passed")

            filament_id = f"{path.stem}_{piece_number}"
            if filament_id in seen_ids:
                raise ValueError(f"Duplicate filament ID generated: {filament_id}")
            if any(character in filament_id for character in ',"\'\r\n'):
                raise ValueError(f"Filename cannot be represented safely in CSV: {path.name}")
            seen_ids.add(filament_id)
            rows.append((filament_id, counts))

        if piece_number == 0:
            images_without_pieces += 1

        if number % 30 == 0:
            print(f"  ...{number} images done")

# Write beside the destination, then atomically publish the complete CSV. A
# crash while writing cannot leave a partial file that looks like a submission.
temporary_path = None
try:
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=output_file.parent,
        prefix=f".{output_file.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        temporary_path = Path(f.name)
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())

    # Hard-link publication is atomic and fails rather than overwriting if the
    # destination appeared after the initial existence check.
    os.link(temporary_path, output_file)
    temporary_path.unlink()
except BaseException:
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)
    raise

print()
print("Rows written (predicted filaments):", len(rows))
print("Images:", len(image_paths))
print("Average filaments per image:", round(len(rows) / max(len(image_paths), 1), 2))
print("Images with no filaments predicted:", images_without_pieces)
print("Images resized to 2048 x 2048:", resized_images)
print()
print("First rows of the file:")

with open(output_file, "r") as f:
    for _ in range(3):
        line = f.readline().strip()
        print(line[:100] + ("..." if len(line) > 100 else ""))

print()
print("Saved:", output_file)