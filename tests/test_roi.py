import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from strawberry_cv.data.labels import StrawberryDataset
from strawberry_cv.data.roi import crop_roi, export_roi_dataset
from strawberry_cv.data.scenes import SceneIndex
from strawberry_cv.data.splits import make_group_split
from tests.helpers import make_dataset, standard_specs


class RoiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images, self.labels, self.manifest = make_dataset(
            self.root, standard_specs(), {"i01": "G01", "i02": "G01"}
        )
        self.dataset = StrawberryDataset(self.images, self.labels)
        self.scenes = SceneIndex.from_csv(self.manifest)
        self.split = make_group_split(self.dataset, self.scenes,
                                      ratios=(0.5, 0.25, 0.25), seed=5)

    def tearDown(self):
        self.tmp.cleanup()

    def test_adaptive_padding_avoids_neighbor(self):
        image = np.full((150, 200, 3), 40, np.uint8)
        cv2.rectangle(image, (20, 30), (80, 110), (20, 30, 220), -1)
        cv2.rectangle(image, (90, 30), (150, 110), (20, 30, 220), -1)
        dataset = self.dataset
        box = dataset.load("i01")[0]
        neighbor = dataset.load("i02")[0]
        # 伪造邻居框：与 box 相邻
        from strawberry_cv.data.labels import Box
        neighbor = Box("i02", 1, "ripe", 85, 20, 145, 100, 200, 150)
        roi = crop_roi(image, box, pad_ratio=0.5, neighbor_boxes=[box, neighbor],
                       max_neighbor_iou=0.05)
        self.assertIsNotNone(roi.crop)
        self.assertLess(roi.pad_used, 0.5)

    def test_export_and_border_skip(self):
        out = self.root / "roi"
        report = export_roi_dataset(
            self.dataset, self.split, out, pad_ratio=0.12, skip_border=True, force=True
        )
        self.assertGreater(report["total_crops"], 0)
        manifest = out / "roi_manifest.csv"
        self.assertTrue(manifest.exists())
        rows = manifest.read_text(encoding="utf-8-sig").splitlines()
        self.assertGreater(len(rows), 1)
        # 每个写出的 crop 文件都存在
        import csv
        with manifest.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                self.assertTrue((out / row["crop_path"]).exists())

    def test_border_box_skipped(self):
        root = self.root / "border_case"
        specs = [{"stem": "b01", "width": 200, "height": 150,
                  "boxes": [("ripe", 0, 20, 70, 120)]}]
        images, labels, manifest = make_dataset(root, specs)
        dataset = StrawberryDataset(images, labels)
        scenes = SceneIndex.from_csv(manifest)
        split = SimpleNamespace(assignments={"b01": "train"})
        report = export_roi_dataset(dataset, split, root / "roi",
                                    skip_border=True, force=True)
        self.assertEqual(report["total_crops"], 0)
        self.assertEqual(report["skipped"].get("border"), 1)


if __name__ == "__main__":
    unittest.main()
