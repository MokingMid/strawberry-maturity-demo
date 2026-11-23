import tempfile
import unittest
from collections import Counter
from pathlib import Path

from strawberry_cv.data.labels import StrawberryDataset
from strawberry_cv.data.scenes import SceneIndex
from strawberry_cv.data.splits import (
    load_split_assignments,
    make_group_split,
    write_split,
)
from tests.helpers import make_dataset, standard_specs


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.images, self.labels, self.manifest = make_dataset(
            root, standard_specs(), {"i01": "G01", "i02": "G01"}
        )
        self.dataset = StrawberryDataset(self.images, self.labels)
        self.scenes = SceneIndex.from_csv(self.manifest)

    def tearDown(self):
        self.tmp.cleanup()

    def test_group_split_constraints(self):
        result = make_group_split(self.dataset, self.scenes,
                                  ratios=(0.5, 0.25, 0.25), seed=7)
        self.assertEqual(sum(result.group_counts.values()), 12)
        self.assertEqual(set(result.assignments), set(self.dataset.stems))
        # 同组不跨 split
        pairs = {}
        for image, split in result.assignments.items():
            group = self.scenes.group_of(image)
            if group in pairs:
                self.assertEqual(pairs[group], split)
            pairs[group] = split
        # 三个 half 组尽量分散
        for split in ("train", "val", "test"):
            self.assertGreaterEqual(result.class_counts[split].get("half", 0), 1)

    def test_deterministic(self):
        a = make_group_split(self.dataset, self.scenes, ratios=(0.5, 0.25, 0.25), seed=7)
        b = make_group_split(self.dataset, self.scenes, ratios=(0.5, 0.25, 0.25), seed=7)
        self.assertEqual(a.assignments, b.assignments)

    def test_write_split(self):
        result = make_group_split(self.dataset, self.scenes,
                                  ratios=(0.5, 0.25, 0.25), seed=7)
        out = Path(self.tmp.name) / "splits"
        write_split(result, self.dataset, self.scenes, out)
        self.assertTrue((out / "split_manifest.csv").exists())
        loaded = load_split_assignments(out / "split_manifest.csv")
        self.assertEqual(loaded, result.assignments)
        for split in ("train", "val", "test"):
            lines = (out / f"{split}.txt").read_text(encoding="utf-8").split()
            self.assertTrue(lines)


if __name__ == "__main__":
    unittest.main()
