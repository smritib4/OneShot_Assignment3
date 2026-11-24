"""Copy the scene dataset into data/train and data/test.

    python scripts/prepare_data.py --src C:/Users/you/SSL_Assignment2/data
    python scripts/prepare_data.py --zip dataset_scenes.zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oneshot.data import TEST_PER_CLASS, TRAIN_PER_CLASS, require_counts  # noqa: E402


def safe_extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            parts = Path(info.filename).parts
            if info.filename.startswith("/") or ".." in parts:
                raise RuntimeError(f"Unsafe path in zip: {info.filename}")
        archive.extractall(dest)


def find_split(root: Path, name: str) -> Path:
    matches = []
    for path in root.rglob(name):
        if path.is_dir() and any(child.is_dir() for child in path.iterdir()):
            matches.append(path)
    if not matches:
        raise RuntimeError(f"Could not find a {name} folder under {root}.")
    matches.sort(key=lambda path: len(path.parts))
    return matches[0]


def copy_split(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the scene dataset under data/.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--src", type=Path)
    source.add_argument("--zip", type=Path)
    parser.add_argument("--dest", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    if args.zip is not None:
        unpacked = ROOT / "data_download"
        if unpacked.exists():
            shutil.rmtree(unpacked)
        safe_extract(args.zip, unpacked)
        origin = unpacked
    else:
        origin = args.src

    train_src = find_split(origin, "train")
    test_src = find_split(origin, "test")
    require_counts(train_src, TRAIN_PER_CLASS)
    require_counts(test_src, TEST_PER_CLASS)
    args.dest.mkdir(parents=True, exist_ok=True)
    copy_split(train_src, args.dest / "train")
    copy_split(test_src, args.dest / "test")
    print(f"Installed training images in {args.dest / 'train'}")
    print(f"Installed test images in {args.dest / 'test'}")
    print("Train total should be 2400. Test total should be 400.")


if __name__ == "__main__":
    main()
