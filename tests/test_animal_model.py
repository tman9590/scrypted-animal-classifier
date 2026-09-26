import json
import unittest
from pathlib import Path

import torch

from animal_model import ANIMAL_GROUPS, AnimalClassifier, OUTPUT_LABELS, validate_groups


ROOT = Path(__file__).resolve().parents[1]


class AnimalModelTests(unittest.TestCase):
    def test_groups_are_disjoint_and_valid(self):
        validate_groups()
        indexes = [index for group in ANIMAL_GROUPS.values() for index in group]
        self.assertEqual(len(indexes), len(set(indexes)))
        self.assertTrue(all(0 <= index < 1000 for index in indexes))

    def test_expected_camera_animals_are_present(self):
        for label in ("bird", "cat", "dog", "fox", "bear", "rabbit_or_hare", "weasel_otter_skunk_or_badger"):
            self.assertIn(label, OUTPUT_LABELS)
        self.assertEqual(OUTPUT_LABELS[-1], "not_animal")

    def test_probability_aggregation_is_normalized(self):
        model = AnimalClassifier(weights=None).eval()
        with torch.no_grad():
            output = model(torch.zeros((2, 3, 224, 224)))
        self.assertEqual(tuple(output.shape), (2, len(OUTPUT_LABELS)))
        probabilities = torch.softmax(output, dim=1)
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-5))

    def test_backend_configs_match(self):
        configs = []
        for backend in ("onnx", "openvino", "coreml", "ncnn"):
            path = ROOT / "models" / backend / "config.json"
            if path.exists():
                configs.append((path.parent, json.loads(path.read_text())))
        self.assertGreaterEqual(len(configs), 1)
        expected = {str(index): label for index, label in enumerate(OUTPUT_LABELS)}
        for directory, config in configs:
            self.assertEqual(config["labels"], expected)
            self.assertEqual(config["input_shape"], [1, 3, 224, 224])
            for filename in config["files"]:
                self.assertTrue((directory / filename).is_file())


if __name__ == "__main__":
    unittest.main()
