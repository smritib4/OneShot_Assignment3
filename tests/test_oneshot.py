import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oneshot.adapter import (  # noqa: E402
    accuracy,
    average_prompt_features,
    class_weights,
    confusion,
    l2_normalize,
    predict,
    score,
)
from oneshot.data import choose_support, load_support  # noqa: E402
from oneshot.labels import TEMPLATES, prompt_list  # noqa: E402


class AdapterTest(unittest.TestCase):
    def test_fused_weight_matches_its_own_photo(self):
        text = l2_normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        visual = l2_normalize(torch.tensor([[0.9, 0.1], [0.1, 0.9]]))
        weights = class_weights(text, visual)
        image = visual[:1]
        self.assertEqual(int(predict(score(image, weights, 10.0))[0]), 0)

    def test_prompt_average_shape(self):
        raw = torch.eye(8)
        pooled = average_prompt_features(raw, n_classes=2, n_prompts=4)
        self.assertEqual(tuple(pooled.shape), (2, 8))
        self.assertTrue(torch.allclose(pooled.norm(dim=-1), torch.ones(2), atol=1e-5))

    def test_accuracy_and_confusion(self):
        prediction = torch.tensor([0, 1, 1])
        labels = torch.tensor([0, 1, 0])
        self.assertAlmostEqual(accuracy(prediction, labels), 2 / 3)
        matrix = confusion(prediction, labels, n_classes=2)
        self.assertEqual(int(matrix[0, 0]), 1)
        self.assertEqual(int(matrix[0, 1]), 1)
        self.assertEqual(int(matrix[1, 1]), 1)

    def test_prompts_cover_every_class_once_per_template(self):
        names = [
            "Bedroom",
            "Coast",
            "Flower",
            "Forest",
            "Highway",
            "Industrial",
            "InsideCity",
            "Kitchen",
            "LivingRoom",
            "Mountain",
            "Office",
            "OpenCountry",
            "Store",
            "Street",
            "Suburb",
            "TallBuilding",
        ]
        prompts = prompt_list(names)
        self.assertEqual(len(prompts), len(names) * len(TEMPLATES))
        self.assertIn("city scene surrounded by buildings", prompts[6 * len(TEMPLATES)])


class SupportTest(unittest.TestCase):
    def test_one_file_per_class_and_stable_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(16):
                folder = root / f"class_{index:02d}"
                folder.mkdir()
                for image_index in range(4):
                    (folder / f"image_{image_index:04d}.jpg").write_bytes(b"not-read")
            first = choose_support(root, seed=0)
            second = choose_support(root, seed=0)
            other = choose_support(root, seed=1)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 16)
            self.assertEqual(len({item["class_name"] for item in first}), 16)
            self.assertNotEqual(first, other)
            opened = list(root.rglob("*.jpg"))
            self.assertTrue(all(path.stat().st_size == len(b"not-read") for path in opened))

    def test_committed_manifest_matches_the_dataset_rule(self):
        train_root = Path.home() / "SSL_Assignment2" / "data" / "train"
        if not train_root.exists():
            self.skipTest("scene dataset is not on this machine")
        chosen = choose_support(train_root, seed=0)
        manifest = load_support(ROOT / "supports" / "seed0.json")
        self.assertEqual(
            [item["file_name"] for item in chosen],
            [item["file_name"] for item in manifest],
        )


if __name__ == "__main__":
    unittest.main()
