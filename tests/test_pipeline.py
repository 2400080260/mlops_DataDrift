import unittest

from src.data_preprocessing import FEATURES, TARGET, generate_customer_data, validate_dataset
from src.evaluate import calculate_metrics
from src.retrain import retrain_if_drifted


class PipelineTests(unittest.TestCase):
    def test_synthetic_data_is_reproducible_and_schema_consistent(self):
        first = generate_customer_data(80, seed=7)
        second = generate_customer_data(80, seed=7)
        self.assertTrue(first.equals(second))
        self.assertEqual(list(validate_dataset(first, require_target=True).columns), FEATURES + [TARGET])
        self.assertNotIn(TARGET, validate_dataset(first.drop(columns=[TARGET])).columns)

    def test_schema_validation_rejects_missing_features(self):
        frame = generate_customer_data(10).drop(columns=["monthly_charges"])
        with self.assertRaisesRegex(ValueError, "monthly_charges"):
            validate_dataset(frame, require_target=True)

    def test_metrics_include_all_required_measures_and_matrix(self):
        metrics = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(set(("accuracy", "precision", "recall", "f1", "confusion_matrix")) - metrics.keys(), set())
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])

    def test_retraining_requires_positive_drift(self):
        with self.assertRaisesRegex(ValueError, "drift-positive"):
            retrain_if_drifted({"drift_detected": False})


if __name__ == "__main__":
    unittest.main()