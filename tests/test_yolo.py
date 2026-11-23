import tempfile
import unittest
from pathlib import Path

from strawberry_cv.data.labels import StrawberryDataset
from strawberry_cv.data.scenes import SceneIndex
from strawberry_cv.data.splits import make_group_split
from strawberry_cv.data.yolo import export_yolo, inspect_yolo_dir
from tests.helpers import make_dataset, standard_specs


class YoloExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.images, self.labels, self.manifest = make_dataset(
            root, standard_specs(), {"i01": "G01", "i02": "G01"}
        )
        self.dataset = StrawberryDataset(self.images, self.labels)
        self.scenes = SceneIndex.from_csv(self.manifest)
        self.split = make_group_split(self.dataset, self.scenes,
                                      ratios=(0.5, 0.25, 0.25), seed=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_1cls(self):
        out = Path(self.tmp.name) / "yolo1"
        report = export_yolo(self.dataset, out, "1cls", self.split)
        self.assertEqual(report["images"], 13)
        self.assertEqual(report["boxes"], 16)
        self.assertEqual(report["nc"], 1)
        self.assertEqual(report["verify_issues"], [])
        text = (out / "labels" / "all" / "i01.txt").read_text(encoding="utf-8").strip()
        cls, cx, cy, w, h = text.split()
        self.assertEqual(cls, "0")
        self.assertAlmostEqual(float(cx), 0.25, places=5)
        self.assertAlmostEqual(float(cy), 0.4, places=5)
        self.assertTrue((out / "data.yaml").exists())
        self.assertTrue((out / "splits" / "train.txt").exists())
        info = inspect_yolo_dir(out)
        self.assertEqual(info["class_counts"], {0: 16})

    def test_export_3cls(self):
        out = Path(self.tmp.name) / "yolo3"
        report = export_yolo(self.dataset, out, "3cls", self.split)
        self.assertEqual(report["nc"], 3)
        self.assertEqual(report["verify_issues"], [])
        ids = set()
        for path in (out / "labels" / "all").glob("*.txt"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ids.add(int(line.split()[0]))
        self.assertEqual(ids, {0, 1, 2})

    def test_verify_catches_wrong_line(self):
        out = Path(self.tmp.name) / "yolo_bad"
        export_yolo(self.dataset, out, "1cls", self.split)
        path = out / "labels" / "all" / "i01.txt"
        path.write_text("0 1.2 0.4 0.3 0.5\n", encoding="utf-8")
        # 直接重新导出并校验会覆盖；这里调用内部校验
        from strawberry_cv.data.yolo import verify_export
        issues = verify_export(self.dataset, out, "1cls", self.split)
        self.assertTrue(any("归一化越界" in x for x in issues))


if __name__ == "__main__":
    unittest.main()
