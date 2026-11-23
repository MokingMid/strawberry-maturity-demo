import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from strawberry_cv.inference.pipeline import TwoStagePipeline
from strawberry_cv.inference.tracker import IoUTracker
from strawberry_cv.utils.image_io import imwrite


class StubDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, frame):
        return [{"box": box, "score": 0.9} for box in self.boxes]


class StubClassifier:
    def __init__(self, state="ripe"):
        self.state = state

    def predict(self, crop):
        return self.state, {"ripe": 0.8, "half": 0.1, "unripe": 0.1}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((150, 200, 3), 40, np.uint8)
        cv2.rectangle(self.frame, (20, 20), (80, 100), (20, 30, 220), -1)
        cv2.rectangle(self.frame, (100, 20), (170, 100), (20, 30, 220), -1)
        self.boxes = [(20, 20, 80, 100), (100, 20, 170, 100)]

    def test_detector_pipeline(self):
        pipeline = TwoStagePipeline(
            detector=StubDetector(self.boxes), classifier=StubClassifier("ripe")
        )
        result = pipeline.process_frame(self.frame)
        self.assertEqual(len(result.fruits), 2)
        self.assertEqual(result.counts["ripe"], 2)
        self.assertEqual(result.annotated.shape, self.frame.shape)

    def test_gt_mode(self):
        gt = [{"box": box, "state": "half"} for box in self.boxes]
        pipeline = TwoStagePipeline(detector=None, classifier=StubClassifier("half"))
        result = pipeline.process_frame(self.frame, gt_fruits=gt)
        self.assertEqual(len(result.fruits), 2)
        self.assertEqual(result.counts["half"], 2)

    def test_tracking_ids(self):
        tracker = IoUTracker(iou_threshold=0.3, max_age=5)
        pipeline = TwoStagePipeline(
            detector=StubDetector(self.boxes), classifier=StubClassifier("ripe"),
            tracker=tracker,
        )
        r1 = pipeline.process_frame(self.frame, 0)
        self.assertTrue(all(f.track_id is not None for f in r1.fruits))
        moved = [(25, 20, 85, 100), (105, 20, 175, 100)]
        pipeline.detector = StubDetector(moved)
        r2 = pipeline.process_frame(self.frame, 1)
        self.assertTrue(all(f.track_id is not None for f in r2.fruits))
        self.assertTrue({f.track_id for f in r1.fruits} & {f.track_id for f in r2.fruits})

    def test_run_image_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.jpg"
            imwrite(image_path, self.frame)
            pipeline = TwoStagePipeline(
                detector=StubDetector(self.boxes), classifier=StubClassifier("ripe")
            )
            result = pipeline.run_image(image_path, Path(tmp) / "out")
            self.assertTrue((Path(tmp) / "out" / "frame_annotated.jpg").exists())
            self.assertTrue((Path(tmp) / "out" / "frame_result.json").exists())
            self.assertEqual(len(result["fruits"]), 2)


if __name__ == "__main__":
    unittest.main()
