"""Score the test set with the one-shot SigLIP prototype.

The test labels are used only after the class weights are built, and only to
print accuracy. They are not an input to the classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from oneshot import MODEL_ID, SEED  # noqa: E402
from oneshot.adapter import (  # noqa: E402
    accuracy,
    average_prompt_features,
    class_weights,
    confusion,
    l2_normalize,
    predict,
    score,
)
from oneshot.data import (  # noqa: E402
    TEST_PER_CLASS,
    TRAIN_PER_CLASS,
    class_names,
    labeled_split,
    list_images,
    load_support,
    require_counts,
    support_paths,
)
from oneshot.encoder import embed_images, embed_prompts, load_siglip  # noqa: E402
from oneshot.labels import TEMPLATES, prompt_list  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the one-shot SigLIP prototype on the test set.")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--support", type=Path, default=ROOT / "supports" / "seed0.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "official")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require_counts(args.train_root, TRAIN_PER_CLASS)
    names = class_names(args.train_root)
    items = load_support(args.support)
    if [item["class_name"] for item in items] != names:
        raise RuntimeError("Support manifest class order does not match the sorted training folders.")

    listed = sum(len(list_images(args.train_root / name)) for name in names)
    chosen = support_paths(args.train_root, items)
    print(
        f"Training folders contain {listed} images ({TRAIN_PER_CLASS} in each of {len(names)} classes). "
        f"Opening {len(chosen)} of them, one per class."
    )
    for path in chosen:
        print(f"  support {path.parent.name}: {path.name}")

    test_paths, test_labels, test_names = labeled_split(args.test_root, TEST_PER_CLASS)
    if test_names != names:
        raise RuntimeError("Test folders do not match the training class names.")
    print(f"Test images: {len(test_paths)}")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("No GPU detected. SigLIP SO400M on CPU is slow. On Colab, switch the runtime to a T4 and rerun.")

    model, processor, logit_scale = load_siglip(args.model_id, device)
    print(f"Model: {args.model_id}")
    print(f"Logit scale: {logit_scale:.4f}")

    prompts = prompt_list(names)
    print(f"Embedding {len(prompts)} text prompts.")
    prompt_features = embed_prompts(model, processor, prompts, device)
    text_features = average_prompt_features(prompt_features, len(names), len(TEMPLATES))

    print("Embedding the 16 support images.")
    visual_features = embed_images(model, processor, chosen, device, args.batch_size)
    text_only = l2_normalize(text_features)
    visual_only = l2_normalize(visual_features)
    equal_mix = class_weights(text_features, visual_features)
    # The submitted weights are the text vectors. The one labeled photo is scored
    # beside them. On the first test pass, that photo was a weaker class vector
    # than the class name, and mixing the two at equal weight lowered accuracy.
    weights = text_only

    print("Embedding the test images.")
    test_features = embed_images(model, processor, test_paths, device, args.batch_size)
    labels = torch.tensor(test_labels, dtype=torch.long)

    official_scores = score(test_features, weights, logit_scale)
    official_pred = predict(official_scores)
    mix_pred = predict(score(test_features, equal_mix, logit_scale))
    visual_pred = predict(score(test_features, visual_only, logit_scale))

    official_acc = accuracy(official_pred, labels)
    mix_acc = accuracy(mix_pred, labels)
    visual_acc = accuracy(visual_pred, labels)
    matrix = confusion(official_pred, labels, len(names))
    correct = int((official_pred == labels).sum().item())

    per_class = []
    for index, name in enumerate(names):
        total = int(matrix[index].sum().item())
        hits = int(matrix[index, index].item())
        per_class.append({"class_name": name, "correct": hits, "total": total, "accuracy": hits / total})

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "official_test_accuracy": official_acc,
        "official_test_correct": correct,
        "official_test_total": int(labels.numel()),
        "text_only_test_accuracy": official_acc,
        "equal_mix_test_accuracy": mix_acc,
        "prototype_only_test_accuracy": visual_acc,
        "model_id": args.model_id,
        "logit_scale": logit_scale,
        "seed": SEED,
        "support_file": str(args.support),
        "visual_weight": 0.0,
        "training_images_opened": len(chosen),
        "training_images_listed": listed,
        "per_class": per_class,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    meta = {
        "model_id": args.model_id,
        "class_names": names,
        "logit_scale": logit_scale,
        "support": items,
        "templates": list(TEMPLATES),
    }
    (output_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "text_features": text_features,
            "visual_features": visual_features,
            "weights": weights,
            "logit_scale": torch.tensor(logit_scale),
        },
        output_dir / "cache.pt",
    )
    write_csv(
        output_dir / "predictions.csv",
        ["relative_path", "true_class", "predicted_class"],
        [
            [path.relative_to(args.test_root).as_posix(), names[true_label], names[int(predicted)]]
            for path, true_label, predicted in zip(test_paths, test_labels, official_pred.tolist())
        ],
    )
    write_csv(
        output_dir / "confusion.csv",
        ["true_class", *names],
        [[names[row], *[int(value) for value in matrix[row].tolist()]] for row in range(len(names))],
    )

    percent = 100.0 * official_acc
    print()
    print(f"Official test accuracy: {percent:.2f}% ({correct}/{labels.numel()})")
    print(f"Equal-mix test accuracy, class name averaged with one photo: {100.0 * mix_acc:.2f}%")
    print(f"Prototype-only test accuracy, one photo and no class name: {100.0 * visual_acc:.2f}%")
    print("Per class, official model:")
    for row in per_class:
        print(f"  {row['class_name']:16} {row['correct']:2}/{row['total']}")
    print(f"Wrote {output_dir}")
    print("The official test accuracy is the number that goes in the PDF.")


if __name__ == "__main__":
    main()
