from sklearn.metrics import (
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
import numpy as np


class Metrics:
    def __init__(self, X_train, y_train, X_test, y_test, models, verbose=True):
        """
        Initializes the Metrics class.

        Parameters:
        - X_train (pd.DataFrame or np.array): Training data features.
        - y_train (pd.Series or np.array): Training data labels.
        - X_test (pd.DataFrame or np.array): Testing data features.
        - y_test (pd.Series or np.array): Testing data labels.
        - models (dict): Dictionary of models to evaluate. Keys are model names,
        values are the model objects.
        - verbose (bool): If True, shows progress bar using tqdm.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = models
        self.verbose = verbose
        self.metric_dict = {}

    def evaluate_model(self, model, name):
        """
        Evaluates a single model and computes multiple metrics.

        Parameters:
        - model: The model to evaluate.
        - name: The name of the model.

        Returns:
        - A dictionary with metrics for the model.
        """
        try:
            # Fit the model
            model.fit(self.X_train, self.y_train)

            # Predictions and probability estimates
            y_pred = model.predict(self.X_test)
            mcc = matthews_corrcoef(self.y_test, y_pred)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="weighted")
            precision = precision_score(self.y_test, y_pred, average="weighted")
            recall = recall_score(self.y_test, y_pred, average="weighted")

            # Default values for AP and AUC in case predict_proba is not available
            ap, auc = np.nan, np.nan

            # If the model has predict_proba, compute AP and AUC
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(self.X_test)
                num_classes = len(np.unique(self.y_test))
                pred_classes = y_proba.shape[1]

                if num_classes == pred_classes:
                    ap = average_precision_score(
                        self.y_test, y_proba, average="weighted"
                    )
                    auc = roc_auc_score(
                        self.y_test, y_proba, multi_class="ovo", average="weighted"
                    )
                else:
                    # Handle case where predicted classes don't match true classes
                    y_proba_pseudo = np.concatenate(
                        [y_proba, np.zeros(y_proba.shape[0]).reshape(-1, 1)], axis=1
                    )
                    ap = average_precision_score(
                        self.y_test, y_proba_pseudo, average="weighted"
                    )
                    auc = roc_auc_score(
                        self.y_test,
                        y_proba_pseudo,
                        multi_class="ovo",
                        average="weighted",
                    )

            return {
                "MCC": mcc,
                "Accuracy": accuracy,
                "F1-Score": f1,
                "Precision": precision,
                "Recall": recall,
                "Average_Precision": ap,
                "ROC-AUC": auc,
            }
        except Exception as e:
            print(f"Error evaluating model {name}: {str(e)}")
            return None

    def cal_metric_df(self):
        """
        Calculate metrics for all models and return a DataFrame containing the results.

        Returns:
        - A pandas DataFrame with models as rows and metrics as columns.
        """
        for name, model in tqdm(self.models.items(), disable=not self.verbose):
            metrics = self.evaluate_model(model, name)
            if metrics:
                self.metric_dict[name] = metrics

        return self.metric_dict

    def fit(self):
        """
        Fit all models, compute metrics, and return the metrics DataFrame.

        Returns:
        - A pandas DataFrame containing the metrics for each model.
        """
        _metrics = self.cal_metric_df()
        return _metrics
