import random
from pathlib import Path

from dataset import SolarFilamentDataset


# -----------------------------
# Settings
# -----------------------------

DATA_DIR = "data/MAGFiLO_1.0_Kaggle_2026"

VALIDATION_RATIO = 0.20

RANDOM_SEED = 42


# -----------------------------
# Load dataset
# -----------------------------

dataset = SolarFilamentDataset(DATA_DIR)

num_images = len(dataset)

print("Total images:", num_images)


# -----------------------------
# Create shuffled indices
# -----------------------------

indices = list(range(num_images))

random.seed(RANDOM_SEED)

random.shuffle(indices)


# -----------------------------
# Split
# -----------------------------

validation_size = int(
    num_images * VALIDATION_RATIO
)

validation_indices = indices[
    :validation_size
]

training_indices = indices[
    validation_size:
]


# -----------------------------
# Print results
# -----------------------------

print()
print("Training images:", len(training_indices))
print("Validation images:", len(validation_indices))


print()
print("First 10 training indices:")
print(training_indices[:10])


print()
print("First 10 validation indices:")
print(validation_indices[:10])