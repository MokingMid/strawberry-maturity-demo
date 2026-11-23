import unittest

import numpy as np

from strawberry_cv.baselines.features import extract_features
from strawberry_cv.baselines.svm import evaluate_model, fit_svm_grid


TRADITIONAL = {
    "feature_size": 64,
    "color_bins": 16,
    "glcm_levels": 16,
    "glcm_distances": [1],
    "glcm_angles": [0, 90],
    "lbp_points": 8,
    "lbp_radius": 1,
}


class FeatureTests(unittest.TestCase):
    def test_feature_vector(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
        vector, names = extract_features(image, TRADITIONAL)
        self.assertEqual(len(vector), len(names))
        self.assertTrue(np.isfinite(vector).all())

    def test_svm_grid(self):
        rng = np.random.default_rng(1)
        n_per_class = 20
        X = np.vstack([
            rng.normal(0, 1, (n_per_class, 8)),
            rng.normal(2, 1, (n_per_class, 8)),
            rng.normal(-2, 1, (n_per_class, 8)),
        ]).astype(np.float32)
        y = np.array(["ripe"] * n_per_class + ["half"] * n_per_class
                     + ["unripe"] * n_per_class)
        groups = np.array([f"g{i}" for i in range(12)] * 5)
        model, best, results = fit_svm_grid(
            X, y, groups, C_values=[0.1, 1.0], gamma_values=["scale"], cv_folds=3
        )
        self.assertIn("C", best["params"])
        self.assertTrue(results)
        metrics = evaluate_model(model, X, y)
        self.assertGreater(metrics["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
