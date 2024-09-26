import numpy as np
from typing import Dict, Any
from sklearn.metrics import matthews_corrcoef, f1_score
from synutils.Uncertainty.conformal_predictor import ConformalPredictor
from synutils.utils import setup_logging

logger = setup_logging()


def cp_evaluate(
    model: Any,
    data_test: Any,
    data_cal: Any,
    class_column: str,
    class_mapping_column: str,
    remove_train: Any,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Evaluate a conformal predictor model using training, test, calibration, and novelty datasets.

    Parameters:
    -----------
    model : Any
        The underlying predictive model used by the conformal predictor.

    data_test : Any
        Test dataset including features and labels.

    data_cal : Any
        Calibration dataset including features and labels.

    remove_train : Any
        Dataset used for novelty detection, including features and labels.

    alpha : float, optional, default=0.05
        The significance level for the conformal predictor.

    Returns:
    --------
    Dict[str, float]
        A dictionary containing various evaluation metrics, including:
        - 'MCC_original': Matthews correlation coefficient for original predictions.
        - 'MCC_calibrated': Matthews correlation coefficient for calibrated predictions.
        - 'MCC_certainty': Matthews correlation coefficient for predictions classified with certainty.
        - 'F1_original_novelty': F1 score for novelty detection (original predictions).
        - 'F1_calibrated_novelty': F1 score for novelty detection (calibrated predictions).
    """

    X_test = data_test.drop([class_column, class_mapping_column], axis=1).values
    y_test = data_test[class_mapping_column].values

    X_cal = data_cal.drop([class_column, class_mapping_column], axis=1).values
    y_cal = data_cal[class_mapping_column].values

    X_novelty = remove_train.drop([class_column, class_mapping_column], axis=1).values
    y_novelty = remove_train[class_mapping_column].values

    class_labels = list(set(data_cal[class_mapping_column].unique()))

    # Initialize the conformal predictor
    cp = ConformalPredictor(model, alpha)

    # Calculate nonconformity scores using the calibration dataset
    nonconformity_scores = cp.cal_non_cp(X_cal, y_cal)

    # Fit the model to the test set and get calibrated predictions
    _, y_calibrated_test, _ = cp.fit(X_test, class_labels, nonconformity_scores)

    # Extract certain predictions
    y_certain, y_pre_certain = cp.get_certain_results(
        y_test, y_calibrated_test, uncertain_class=np.max(y_calibrated_test)
    )

    # Fit the model to the novelty set and get calibrated predictions
    _, y_calibrated_novelty, threshold = cp.fit(
        X_novelty, class_labels, nonconformity_scores, novelty=True
    )

    # Calculate metrics
    mcc_original = matthews_corrcoef(y_test, model.predict(X_test))
    logger.info(f"MCC_original: {mcc_original}")

    mcc_calibrated = matthews_corrcoef(y_test, y_calibrated_test)
    logger.info(f"MCC_calibrated: {mcc_calibrated}")

    mcc_certainty = matthews_corrcoef(y_certain, y_pre_certain)
    logger.info(f"MCC_certainty: {mcc_certainty}")

    f1_original_novelty = f1_score(y_novelty, model.predict(X_novelty), average="micro")
    logger.info(f"F1_original_novelty: {f1_original_novelty}")

    f1_calibrated_novelty = f1_score(
        np.ones(y_novelty.shape[0]), y_calibrated_novelty, average="micro"
    )
    logger.info(f"F1_calibrated_novelty: {f1_calibrated_novelty}")
    logger.info(f"Threshold: {threshold}")

    return {
        "Threshold": threshold,
        "MCC_original": mcc_original,
        "MCC_calibrated": mcc_calibrated,
        "MCC_certainty": mcc_certainty,
        "F1_original_novelty": f1_original_novelty,
        "F1_calibrated_novelty": f1_calibrated_novelty,
    }
