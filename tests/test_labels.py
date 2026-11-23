import json
import tempfile
import unittest
from pathlib import Path

from strawberry_cv.data.labels import LabelError, StrawberryDataset
from tests.helpers import make_dataset, standard_specs


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.images, self.labels, self.manifest = make_dataset(
            root, standard_specs(), {"i01": "G01", "i02": "G01"}
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_stats(self):
        dataset = StrawberryDataset(self.images, self.labels)
        stats = dataset.stats()
        self.assertEqual(stats["images"], 13)
        self.assertEqual(stats["boxes"], 16)
        self.assertEqual(stats["per_class"], {"ripe": 8, "half": 3, "unripe": 5})
        self.assertIn("i05", stats["mixed_images"])
        self.assertEqual(dataset.validate_all(), [])

    def test_normalized_box(self):
        dataset = StrawberryDataset(self.images, self.labels)
        box = dataset.load("i01")[0]
        cx, cy, w, h = box.normalized_cxcywh
        self.assertAlmostEqual(cx, 0.25)
        self.assertAlmostEqual(cy, 0.4)
        self.assertAlmostEqual(w, 0.3)
        self.assertAlmostEqual(h, 0.533333, places=4)

    def test_invalid_class_raises(self):
        dataset = StrawberryDataset(self.images, self.labels)
        path = self.labels / "i01.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["labels"][0]["name"] = "hslf"
        path.write_text(json.dumps(data), encoding="utf-8")
        fresh = StrawberryDataset(self.images, self.labels)
        with self.assertRaises(LabelError):
            fresh.load("i01")

    def test_missing_label_detected(self):
        (self.labels / "i13.json").unlink()
        dataset = StrawberryDataset(self.images, self.labels)
        issues = dataset.validate_all()
        self.assertTrue(any("缺少标签" in x for x in issues))


if __name__ == "__main__":
    unittest.main()
