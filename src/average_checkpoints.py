import sys
from pathlib import Path

import torch

# Usage: python -m src.average_checkpoints <output_file> <checkpoint> <checkpoint> ...
# Averages the weights of several checkpoints from the same training run.

if len(sys.argv) < 4:
    sys.exit(
        "Usage: python -m src.average_checkpoints "
        "<output_file> <checkpoint1> <checkpoint2> ..."
    )

output_file = Path(sys.argv[1])
input_files = [Path(p) for p in sys.argv[2:]]

if output_file.suffix.lower() != ".pth":
    sys.exit("output_file must end with .pth")

if output_file.exists():
    sys.exit(f"Refusing to overwrite existing file: {output_file}")

for path in input_files:
    if not path.is_file():
        sys.exit(f"Checkpoint does not exist: {path}")


def load_state(path):
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    return checkpoint


print("Averaging", len(input_files), "checkpoints:")

for path in input_files:
    print("  ", path)

states = [load_state(path) for path in input_files]
keys = list(states[0].keys())

for state in states[1:]:
    if list(state.keys()) != keys:
        sys.exit("Checkpoints have different layers and cannot be averaged")

averaged = {}

for key in keys:
    first = states[0][key]

    if first.is_floating_point():
        stacked = torch.stack([state[key].float() for state in states])
        averaged[key] = stacked.mean(dim=0).to(first.dtype)
    else:
        # Counters (such as num_batches_tracked): keep the last checkpoint's value
        averaged[key] = states[-1][key]

output_file.parent.mkdir(parents=True, exist_ok=True)

torch.save(averaged, output_file)

print("Saved:", output_file)