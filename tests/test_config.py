import unittest
from pathlib import Path

from strawberry_cv.config import Config, resolve_device


class ConfigTests(unittest.TestCase):
    def test_load_and_validate(self):
        config = Config.load()
        self.assertEqual(config.class_names(), ("ripe", "half", "unripe"))
        self.assertEqual(config.validate(require_data=False), [])
        self.assertTrue(config.path("paths.images").exists())
        self.assertEqual(config.get("detector.num_classes"), 1)
        self.assertEqual(config.get("classifier.num_classes"), 3)

    def test_derived_paths(self):
        config = Config.load()
        work = config.path("paths.work_dir")
        self.assertEqual(config.derived("roi"), work / "roi_classification")
        self.assertEqual(config.derived("splits"), work / "splits")

    def test_device(self):
        self.assertIn(resolve_device("auto"), {"cpu", "0"})
        self.assertEqual(resolve_device("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
