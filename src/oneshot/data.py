"""Folder layout and the one-image-per-class support set.

Listing a class folder to count files, or to draw one file name, does not open
those images. Training opens only the file named in the support manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_CLASSES = 16
TRAIN_PER_CLASS = 150
TEST_PER_CLASS = 25


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")
    )


def class_names(split_root: Path) -> list[str]:
    root = Path(split_root)
    names = sorted(
        path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if len(names) != EXPECTED_CLASSES:
        found = ", ".join(names) if names else "(none)"
        raise RuntimeError(f"Expected {EXPECTED_CLASSES} class folders in {root}, found {len(names)}: {found}")
    return names


def require_counts(split_root: Path, expected_per_class: int) -> None:
    for class_name in class_names(split_root):
        count = len(list_images(Path(split_root) / class_name))
        if count != expected_per_class:
            raise RuntimeError(
                f"{Path(split_root) / class_name} has {count} images; expected {expected_per_class}."
            )


def choose_support(train_root: Path, seed: int = 0) -> list[dict]:
    """Draw one file name per class. Does not open image pixels."""
    rng = np.random.default_rng(seed)
    items = []
    for class_name in class_names(train_root):
        files = list_images(Path(train_root) / class_name)
        if not files:
            raise RuntimeError(f"No images in {Path(train_root) / class_name}.")
        index = int(rng.integers(0, len(files)))
        items.append(
            {
                "class_name": class_name,
                "index": index,
                "file_name": files[index].name,
            }
        )
    return items


def load_support(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload["items"]
    if len(items) != EXPECTED_CLASSES:
        raise RuntimeError(f"Support manifest {path} has {len(items)} items, expected {EXPECTED_CLASSES}.")
    return items


def support_paths(train_root: Path, items: list[dict]) -> list[Path]:
    paths = []
    for item in items:
        path = Path(train_root) / item["class_name"] / item["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"Support image is missing: {path}")
        paths.append(path)
    return paths


def labeled_split(split_root: Path, expected_per_class: int) -> tuple[list[Path], list[int], list[str]]:
    names = class_names(split_root)
    paths: list[Path] = []
    labels: list[int] = []
    for label, class_name in enumerate(names):
        files = list_images(Path(split_root) / class_name)
        if len(files) != expected_per_class:
            raise RuntimeError(
                f"{split_root / class_name} has {len(files)} images; expected {expected_per_class}."
            )
        paths.extend(files)
        labels.extend([label] * len(files))
    return paths, labels, names
