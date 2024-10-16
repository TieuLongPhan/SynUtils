import random
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from synutility.SynSplit.stratified_split import StratifiedSplit


class StratifiedReductionSplit:
    """
    A class for partitioning a dataset into training and test sets with the additional
    functionality of selectively reducing the presence of certain classes based
    on a specified drop ratio.

    Attributes:
    - data (pd.DataFrame): The dataset to partition.
    - test_size (float): The proportion of the dataset to include in the test split.
    - drop_class_ratio (float): The maximum cumulative proportion of classes
    to remove from the training data.
    - class_column (str): The name of the column in data that contains the class labels.
    - random_state (int): The seed for the random number generator.
    - keep_data (bool): return also data_remove or not
    """

    def __init__(
        self,
        data: pd.DataFrame,
        test_size: float,
        drop_class_ratio: float,
        class_column: str,
        random_state: int,
        keep_data: bool = True,
    ):
        """
        Initializes the StratifiedReductionPartition with the specified attributes.
        """
        self.data = data
        self.test_size = test_size
        self.drop_class_ratio = drop_class_ratio
        self.class_column = class_column
        self.random_state = random_state
        self.keep_data = keep_data
        random.seed(self.random_state)
        # Create the 'class_mapping' column and insert it at position 0
        self.data.insert(0, "class_mapping", self.data[self.class_column].copy())

    @staticmethod
    def random_select(data: dict, threshold: float, random_state: int = 42) -> list:
        """
        Selects keys from a dictionary randomly without exceeding a specified
        cumulative threshold.

        Parameters:
        - data (dict): A dictionary where keys are identifiers and values are weights.
        - threshold (float): The cumulative weight not to exceed.
        - random_state (int): Seed for the random number generator.

        Returns:
            list: A list of selected keys.
        """
        random.seed(random_state)
        remaining_keys = list(data.keys())
        selected_keys = []
        cumulative_sum = 0.0

        while remaining_keys and cumulative_sum < threshold:
            key = random.choice(remaining_keys)
            if cumulative_sum + data[key] > threshold:
                break
            cumulative_sum += data[key]
            selected_keys.append(key)
            remaining_keys.remove(key)

        return selected_keys

    def fit(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Partitions the dataset into training and testing sets while removing selected
        classes from the training set based on the drop_class_ratio.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Returns the training data with
        selected classes removed, the test data, and the subset of the training data that
        contains the removed classes.
        """
        splitter = StratifiedSplit(
            self.data, self.test_size, self.class_column, self.random_state
        )
        data_train, data_test = splitter.fit()

        class_counts = data_train[self.class_column].value_counts(normalize=True)

        removed_classes_list = self.random_select(
            class_counts.to_dict(), self.drop_class_ratio
        )

        data_train_remove = data_train[
            ~data_train[self.class_column].isin(removed_classes_list)
        ]
        remove_train = data_train[
            data_train[self.class_column].isin(removed_classes_list)
        ]

        data_test_remove = data_test[
            ~data_test[self.class_column].isin(removed_classes_list)
        ]
        remove_test = data_test[data_test[self.class_column].isin(removed_classes_list)]

        labelencoder = LabelEncoder()
        data_train_remove[self.class_column] = labelencoder.fit_transform(
            data_train_remove[self.class_column]
        )
        data_test_remove[self.class_column] = labelencoder.transform(
            data_test_remove[self.class_column]
        )

        removed_label = data_train_remove[self.class_column].max() + 1
        remove_train[self.class_column] = removed_label
        remove_test[self.class_column] = removed_label

        data_test = pd.concat([data_test_remove, remove_test], axis=0).sort_index()
        if self.keep_data:
            return data_train_remove, data_test, remove_train
        else:
            return data_train_remove, data_test
