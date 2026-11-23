import unittest

from strawberry_cv.evaluation.classification import classification_metrics
from strawberry_cv.evaluation.detection import average_precision, detection_metrics
from strawberry_cv.evaluation.end2end import end2end_metrics


class MetricTests(unittest.TestCase):
    def test_detection_metrics(self):
        gt = [(0, 0, 10, 10), (20, 20, 30, 30)]
        pred = [(0, 0, 10, 10), (50, 50, 60, 60)]
        metrics = detection_metrics(pred, [0.9, 0.8], gt, iou_thr=0.5)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_average_precision_perfect(self):
        gt = [(0, 0, 10, 10), (20, 20, 30, 30)]
        pred = [(0, 0, 10, 10), (20, 20, 30, 30)]
        ap = average_precision(pred, [0.9, 0.8], gt, iou_thr=0.5)
        self.assertAlmostEqual(ap, 1.0, places=5)

    def test_classification_metrics(self):
        y_true = ["ripe", "ripe", "half", "unripe"]
        y_pred = ["ripe", "half", "half", "unripe"]
        metrics = classification_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertEqual(len(metrics["confusion"]), 3)
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)

    def test_end2end_metrics(self):
        gts = {"img": [
            {"box": (0, 0, 10, 10), "state": "ripe"},
            {"box": (20, 20, 30, 30), "state": "half"},
        ]}
        preds = {"img": [
            {"box": (0, 0, 10, 10), "state": "ripe", "score": 0.9},
            {"box": (50, 50, 60, 60), "state": "unripe", "score": 0.8},
        ]}
        metrics = end2end_metrics(preds, gts, iou_thr=0.5)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["state_accuracy"], 1.0)
        self.assertEqual(metrics["count_error"], {"ripe": 0, "half": -1, "unripe": 1})
        self.assertEqual(metrics["total_count_error"], 0)
        self.assertAlmostEqual(metrics["end2end_correct_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
