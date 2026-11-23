import tempfile
import unittest
from pathlib import Path

from strawberry_cv.data.scenes import SceneIndex
from tests.helpers import make_dataset, standard_specs


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.images, self.labels, self.manifest = make_dataset(
            root, standard_specs(), {"i01": "G01", "i02": "G01"}
        )
        self.scenes = SceneIndex.from_csv(self.manifest)

    def tearDown(self):
        self.tmp.cleanup()

    def test_groups(self):
        self.assertEqual(len(self.scenes.groups), 12)
        self.assertEqual(self.scenes.group_of("i01"), "G01")
        self.assertEqual(self.scenes.group_of("i02"), "G01")
        self.assertEqual(self.scenes.stats()["duplicate_pairs"]["G01"], ["i01", "i02"])

    def test_validate(self):
        images = sorted(p.stem for p in self.images.glob("*.jpg"))
        self.assertEqual(self.scenes.validate(images), [])
        self.assertTrue(self.scenes.validate(images[:-1]))


if __name__ == "__main__":
    unittest.main()
