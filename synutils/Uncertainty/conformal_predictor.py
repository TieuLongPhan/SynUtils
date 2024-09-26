from typing import Any, List, Set, Tuple, Union
import numpy as np


class ConformalPredictor:
    def __init__(self, model: Any, alpha: float):
        """
        Initialize the ConformalPredictor with a prediction model and a
        significance level.

        Parameters:
        - model (Any): A machine learning model that supports probability prediction.
        - alpha (float): The significance level used to compute the threshold
        for p-values.
        """
        self.model = model
        self.alpha = alpha

    @staticmethod
    def get_proba(model: Any, X: np.ndarray) -> np.ndarray:
        """
        Predict probability estimates for each class for each sample in X.

        Parameters:
        - model (Any): The prediction model.
        - X (np.ndarray): The input features for which to predict probabilities.

        Returns:
        - np.ndarray: The predicted probabilities.
        """
        return model.predict_proba(X)

    @staticmethod
    def get_nonconformity_score(
        probas_calib: np.ndarray, y_cal: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the nonconformity scores using the calibration data.

        Parameters:
        - probas_calib (np.ndarray): Probabilities obtained from the calibration dataset.
        - y_cal (np.ndarray): Actual labels of the calibration dataset.

        Returns:
        - np.ndarray: Nonconformity scores for the calibration dataset.
        """
        return 1 - probas_calib[np.arange(len(y_cal)), y_cal]

    @staticmethod
    def get_threshold(nonconformity_scores: np.ndarray, alpha: float) -> float:
        """
        Compute the threshold value for determining significant p-values based on
        the alpha level, with an adjustment for the finite sample size.

        Parameters:
        - nonconformity_scores (np.ndarray): Calculated nonconformity scores from the
        calibration data.
        - alpha (float): Significance level to determine the threshold.

        Returns:
        - float: The calculated threshold with a sample size adjustment.
        """
        number_of_samples = len(nonconformity_scores)
        qlevel = (1 - alpha) * ((number_of_samples + 1) / number_of_samples)
        return np.percentile(nonconformity_scores, qlevel * 100)

    @staticmethod
    def calculate_p_values(
        probs_test: np.ndarray, nonconformity_scores: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the p-values for the test dataset based on the nonconformity scores.

        Parameters:
        - probs_test (np.ndarray): Probability estimates from the test data.
        - nonconformity_scores (np.ndarray): Nonconformity scores from the
        calibration data.

        Returns:
        - np.ndarray: The calculated p-values for each test instance.
        """
        margins = 1 - probs_test
        nonconformity_scores = nonconformity_scores.reshape(1, -1)
        comparisons = margins[:, :, np.newaxis] <= nonconformity_scores
        p_values = comparisons.sum(axis=2) + 1
        p_values = p_values.astype(float)  # Ensure the array is of float type
        p_values /= len(nonconformity_scores[0]) + 1  # Safe division
        return p_values

    @staticmethod
    def get_prediction_labels(
        p_values: np.ndarray, threshold: float, class_labels: List[str]
    ) -> List[Set[str]]:
        """
        Generate prediction sets based on p-values and class labels, where predictions
        meet or exceed the threshold.

        Parameters:
        - p_values (np.ndarray): The p-values for each class of each test instance.
        - threshold (float): The threshold value to determine significant predictions.
        - class_labels (List[str]): The list of class labels.

        Returns:
        - List[Set[str]]: Sets of predicted labels for each test instance.
        """
        class_labels = np.array(class_labels)
        prediction_sets = p_values >= threshold
        prediction_set_labels = [
            set(class_labels[prediction_set]) for prediction_set in prediction_sets
        ]
        return prediction_set_labels

    @staticmethod
    def get_calibrated_prediction(
        prediction_set_labels: list[set], class_labels: list, novelty: bool = False
    ) -> np.ndarray:
        """
        Transform prediction sets into calibrated labels using numpy vectorization for efficiency.

        Parameters:
        - prediction_set_labels (list[set]): A list of sets, each containing predicted class labels.
        - class_labels (list): A list of all possible class labels.

        Returns:
        - np.ndarray: An array of calibrated labels, with a specific label for uncertainty.
        """
        # Determine the uncertainty class as one more than the maximum class label
        if novelty:
            uncertain_class = 1
        else:
            uncertain_class = max(class_labels) + 1

        # Initialize the output array with the uncertain class by default
        y_calibrated = np.full(len(prediction_set_labels), uncertain_class, dtype=int)

        # Iterate over prediction sets and assign calibrated labels
        for idx, values in enumerate(prediction_set_labels):
            if len(values) == 1:
                if novelty:
                    y_calibrated[idx] = 0
                else:
                    y_calibrated[idx] = next(iter(values))

        return y_calibrated

    def cal_non_cp(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the nonconformity scores for given input features and labels.

        Parameters:
        - X (np.ndarray): Input features.
        - y (np.ndarray): Corresponding labels.

        Returns:
        - np.ndarray: Nonconformity scores calculated using the model's probability
        predictions.
        """
        probas = self.get_proba(self.model, X)
        return self.get_nonconformity_score(probas, y)

    def fit(
        self,
        X: np.ndarray,
        class_labels: List[str],
        nonconformity_scores: np.ndarray,
        novelty: bool = False,
    ) -> Tuple[List[Set[str]], np.ndarray]:
        """
        Fit the conformal prediction model and generate prediction labels.

        Parameters:
        - X (np.ndarray): Test dataset features.
        - class_labels (List[str]): List of all class labels.
        - nonconformity_scores (np.ndarray): Pre-computed nonconformity scores from the
        calibration data.

        Returns:
        - Tuple[List[Set[str]], np.ndarray]: A tuple containing sets of predicted labels
        for each instance and the used nonconformity scores.
        """
        probas = self.get_proba(self.model, X)
        threshold = self.get_threshold(nonconformity_scores, self.alpha)
        p_values = self.calculate_p_values(probas, nonconformity_scores)
        prediction_set_labels = self.get_prediction_labels(
            p_values, threshold, class_labels
        )
        y_calibrated = self.get_calibrated_prediction(
            prediction_set_labels, class_labels, novelty
        )
        return prediction_set_labels, y_calibrated, threshold

    @staticmethod
    def get_certain_results(
        y: np.ndarray, y_calibrated: np.ndarray, uncertain_class: Union[int, float, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts and returns results from the given arrays where the calibrated results
        are certain.

        Parameters:
        -----------
        - y (np.ndarray): An array of original labels or predictions.
        - y_calibrated (np.ndarray): An array of calibrated labels or predictions
        corresponding to `y`.
        - uncertain_class (Union[int, float, str]):  The value representing uncertainty
        in the `y_calibrated` array. Any entry in `y_calibrated` that matches this value
        will be excluded from the results.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - `certain_y`: A numpy array of original labels or predictions from `y` that are
        certain.
        - `certain_y_calibrated`: A numpy array of calibrated labels or predictions
        from `y_calibrated` that are certain.

        Raises:
        -------
        TypeError
            If `y` or `y_calibrated` are not numpy arrays.

        ValueError
            If `y` and `y_calibrated` do not have the same shape.
        """

        if not isinstance(y, np.ndarray) or not isinstance(y_calibrated, np.ndarray):
            raise TypeError("Both `y` and `y_calibrated` must be numpy arrays.")

        if y.shape != y_calibrated.shape:
            raise ValueError("`y` and `y_calibrated` must have the same shape.")

        # Boolean mask where y_calibrated does not equal the uncertain_class
        certain_indices = y_calibrated != uncertain_class

        # Return the filtered results
        return y[certain_indices], y_calibrated[certain_indices]
