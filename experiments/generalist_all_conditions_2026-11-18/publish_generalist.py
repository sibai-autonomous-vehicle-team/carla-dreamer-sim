"""Publish the single pooled CNN after checking its test deliverables."""

import hashlib
import json
from pathlib import Path
import shutil

import torch


experiment = Path(__file__).parent
source = experiment / "training" / "best.pth"
split_folder = experiment / "training" / "splits"
output = Path("generalist")
checkpoint = torch.load(source, map_location="cpu", weights_only=True)
conditions = sorted(path.stem for path in split_folder.glob("*.json"))
if len(conditions) != 9 or checkpoint["model"]["fc.weight"].shape != (1, 512):
    raise ValueError("The published model must be a nine-condition ResNet-18 gap checkpoint")
for condition in conditions:
    path = experiment / "evaluation" / condition / "test_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)

target = output / "cnn.pth"
sha = hashlib.sha256(source.read_bytes()).hexdigest()
output.mkdir(exist_ok=True)
if target.exists():
    if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
        raise FileExistsError(f"{target} has different weights")
else:
    temporary = target.with_suffix(".pth.tmp")
    shutil.copyfile(source, temporary)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != sha:
        raise IOError("The checkpoint copy failed its hash check")
    temporary.replace(target)

manifest = dict(
    architecture="ResNet-18 with one gap regression output",
    conditions=conditions,
    cnn_sha256=sha,
    cnn_source=str(source),
    seed=checkpoint["seed"],
    selected_epoch=checkpoint["epoch"],
    validation_mae_m=checkpoint["validation_mae"],
    archived_test_summary=str(experiment / "summary.json"),
)
temporary = output / "cnn.json.tmp"
temporary.write_text(json.dumps(manifest, indent=2) + "\n")
temporary.replace(output / "cnn.json")
print(f"Verified checkpoint saved to {target}")
