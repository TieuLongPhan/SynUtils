from typing import Any, Dict, List, Optional
from joblib import Parallel, delayed
from synutils.Partition.data_partition import DataPartition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from synutils.Analysis.metrics import Metrics
import numpy as np


class SplitBenchmark:
    """Class to benchmark different data partitioning methods using various classifiers.

    Attributes:
    - data (Any): The dataset to be partitioned and used for model training.
    - test_size (float): The proportion of the dataset to include in the test split.
    - class_column (str): The name of the column to be used as the target for models.
    - random_state (int): Random state for reproducibility of the splits.
    - drop_class_ratio (float): The ratio of the class to drop during partitioning.
    """

    def __init__(
        self,
        data: Any,
        test_size: float,
        class_columns: str,
        random_state: int,
        drop_class_ratio: float,
    ) -> None:
        self.data = data
        self.test_size = test_size
        self.class_column = class_columns
        self.random_state = random_state
        self.drop_class_ratio = drop_class_ratio

    @staticmethod
    def seed_generator(
        number_of_seed: int, random_state: Optional[int] = None
    ) -> List[int]:
        """Generates a list of random seeds using NumPy.

        Parameters:
        - number_of_seed (int): The number of seeds to generate.
        - random_state (Optional[int]): The random state to ensure reproducibility.

        Returns:
        - List[int]: A list of generated seeds.
        """
        rng = np.random.default_rng(seed=random_state)
        seeds = rng.integers(low=0, high=2**32 - 1, size=number_of_seed).tolist()
        return seeds

    @staticmethod
    def setup_models(random_state: int) -> Dict[str, Any]:
        """Sets up a dictionary of different ML models configured with a common random state.

        Parametrs:
        - random_state (int): The random state for model reproducibility.

        Returns:
        - Dict[str, Any]: A dictionary of initialized models.
        """
        return {
            "kNN": KNeighborsClassifier(n_jobs=1),
            "MLP": MLPClassifier(random_state=random_state),
            "XGB": XGBClassifier(
                random_state=random_state, objective="multi:softmax", n_jobs=1
            ),
            "Logistic_regression": LogisticRegression(
                random_state=random_state, n_jobs=1
            ),
        }

    @staticmethod
    def process_method(
        seed: int,
        data: Any,
        test_size: float,
        class_column: str,
        method: str,
        random_state: int,
        drop_class_ratio: float,
    ) -> Dict[str, Any]:
        """Processes the data partitioning and model evaluation for a given method and seed.

        Parameters:
        - seed (int): Random seed for partitioning reproducibility.
        - data (Any): The dataset to be used.
        - test_size (float): Proportion of the dataset to include in the test split.
        - class_column (str): Target column name.
        - method (str): Partitioning method name.
        - random_state (int): Random state for model reproducibility.
        - drop_class_ratio (float): Ratio of the class to be dropped in partitioning.

        Returns:
            Dict[str, Any]: A dictionary containing the performance metrics of each model.
        """
        models = SplitBenchmark.setup_models(random_state)
        partitioner = DataPartition(
            data,
            test_size,
            class_column,
            method,
            seed,
            drop_class_ratio,
            keep_data=False,
        )
        data_train, data_test = partitioner.fit()
        if method == "stratified_class_reduction":
            X_train, y_train = (
                data_train.drop([class_column, "class_mapping"], axis=1),
                data_train[class_column],
            )
            X_test, y_test = (
                data_test.drop([class_column, "class_mapping"], axis=1),
                data_test[class_column],
            )
        else:
            X_train, y_train = (
                data_train.drop([class_column], axis=1),
                data_train[class_column],
            )
            X_test, y_test = (
                data_test.drop([class_column], axis=1),
                data_test[class_column],
            )

        return Metrics(X_train, y_train, X_test, y_test, models).fit()

    def fit(
        self,
        number_sample: int = 5,
        n_jobs: int = 3,
        verbose: int = 0,
        methods: List[str] = [
            "random",
            "stratified_target",
            "stratified_class_reduction",
        ],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Executes the benchmarking across different partition methods.

        Parameters:
        - number_sample (int): Number of samples to generate seeds for.
        - n_jobs (int): Number of parallel jobs to run.
        - verbose (int): Verbosity level.
        - methods (List[str]): List of partition methods to evaluate.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary with method names as keys and lists of results as values.
        """
        seed_list = self.seed_generator(number_sample, self.random_state)
        results = {}

        for method in methods:
            method_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(self.process_method)(
                    seed,
                    self.data,
                    self.test_size,
                    self.class_column,
                    method,
                    self.random_state,
                    self.drop_class_ratio,
                )
                for seed in seed_list
            )
            results[method] = method_results

        return results
