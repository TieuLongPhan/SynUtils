import unittest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from synutils.Uncertainty.conformal_predictor import ConformalPredictor


class TestConformalPredictor(unittest.TestCase):
    def setUp(self):
        # Creating a dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=12,
            n_redundant=4,
            n_repeated=4,
            n_classes=7,
            n_clusters_per_class=2,
            random_state=42,
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.predictor = ConformalPredictor(self.model, 0.05)

    def test_initialization(self):
        self.assertIsInstance(self.predictor, ConformalPredictor)
        self.assertEqual(self.predictor.alpha, 0.05)

    def test_probability_predictions(self):
        probas = ConformalPredictor.get_proba(self.model, self.X_test)
        self.assertEqual(probas.shape, (self.X_test.shape[0], 7))  # Assuming 7 classes

    def test_nonconformity_scores(self):
        probas = ConformalPredictor.get_proba(self.model, self.X_train)
        scores = ConformalPredictor.get_nonconformity_score(probas, self.y_train)
        self.assertEqual(scores.shape, self.y_train.shape)

    def test_threshold_calculation(self):
        probas = ConformalPredictor.get_proba(self.model, self.X_train)
        scores = ConformalPredictor.get_nonconformity_score(probas, self.y_train)
        threshold = ConformalPredictor.get_threshold(scores, self.predictor.alpha)
        self.assertIsInstance(threshold, float)

    def test_p_value_calculation(self):
        probas = ConformalPredictor.get_proba(self.model, self.X_test)
        probas_calib = ConformalPredictor.get_proba(self.model, self.X_train)
        scores = ConformalPredictor.get_nonconformity_score(probas_calib, self.y_train)
        p_values = ConformalPredictor.calculate_p_values(probas, scores)
        self.assertEqual(
            p_values.shape, (self.X_test.shape[0], 7)
        )  # Assuming 7 classes

    def test_prediction_labels(self):
        class_labels = [str(i) for i in range(7)]
        probas = ConformalPredictor.get_proba(self.model, self.X_test)
        probas_calib = ConformalPredictor.get_proba(self.model, self.X_train)
        scores = ConformalPredictor.get_nonconformity_score(probas_calib, self.y_train)
        threshold = ConformalPredictor.get_threshold(scores, self.predictor.alpha)
        p_values = ConformalPredictor.calculate_p_values(probas, scores)
        prediction_sets = ConformalPredictor.get_prediction_labels(
            p_values, threshold, class_labels
        )
        self.assertEqual(len(prediction_sets), self.X_test.shape[0])

    def test_calibrated_prediction(self):
        class_labels = list(range(7))
        prediction_sets = [set([0, 1]), set([1]), set([2, 3])]
        calibrated_labels = ConformalPredictor.get_calibrated_prediction(
            prediction_sets, class_labels
        )
        self.assertEqual(calibrated_labels.shape, (3,))


if __name__ == "__main__":
    unittest.main()
