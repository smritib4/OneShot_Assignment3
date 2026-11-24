"""Run inference with the saved one-shot weights.

Class weights are loaded from the cache. This script does not look at training
images and does not rebuild the prototype.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from oneshot.adapter import accuracy, predict, score  # noqa: E402
from oneshot.data import IMAGE_EXTENSIONS, class_names, labeled_split, list_images  # noqa: E402
from oneshot.encoder import embed_images, load_siglip  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict scene classes with a saved one-shot cache.")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runs" / "official")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def flat_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")
    )


def main() -> None:
    args = parse_args()
    meta = json.loads((args.cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
    try:
        cache = torch.load(args.cache_dir / "cache.pt", map_location="cpu", weights_only=True)
    except TypeError:
        cache = torch.load(args.cache_dir / "cache.pt", map_location="cpu")
    weights = cache["weights"]
    logit_scale = float(cache["logit_scale"].item())
    names = meta["class_names"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, _scale = load_siglip(meta["model_id"], device)

    has_class_folders = False
    try:
        folder_names = class_names(args.image_root)
        has_class_folders = folder_names == names
    except (RuntimeError, FileNotFoundError):
        has_class_folders = False

    labels = None
    if has_class_folders:
        paths, label_ids, _found = labeled_split(args.image_root, expected_per_class=len(list_images(args.image_root / names[0])))
        labels = torch.tensor(label_ids, dtype=torch.long)
    else:
        paths = flat_images(args.image_root)
        if not paths:
            raise RuntimeError(f"No images under {args.image_root}")

    print(f"Embedding {len(paths)} images on {device}.")
    features = embed_images(model, processor, paths, device, args.batch_size)
    prediction = predict(score(features, weights, logit_scale))

    output = args.output or (args.cache_dir / "inference_predictions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "predicted_class"])
        for path, predicted in zip(paths, prediction.tolist()):
            writer.writerow([path.relative_to(args.image_root).as_posix(), names[int(predicted)]])
    print(f"Wrote {output}")

    if labels is not None:
        correct = int((prediction == labels).sum().item())
        print(f"Accuracy on this folder: {100.0 * accuracy(prediction, labels):.2f}% ({correct}/{len(labels)})")


if __name__ == "__main__":
    main()
