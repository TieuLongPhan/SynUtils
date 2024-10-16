import numpy as np
from pandas import DataFrame
import random


class RCSplit:
    """
    Class for partitioning data into training and test sets based on a specified class column,
    ensuring that the cumulative percentage of data points in the test set does not exceed the specified test_size.

    Attributes:
        data (DataFrame): The dataset to partition.
        test_size (float): The maximum proportion of the dataset to include in the test split.
        class_column (str): The name of the column in `data` that represents the class labels.
        random_state (int): Controls the shuffling applied to the data before applying the split.
    """

    def __init__(
        self, data: DataFrame, test_size: float, class_column: str, random_state: int
    ) -> None:
        """
        Initializes the RCSplit with the given data and parameters.
        """
        self.data = data
        self.test_size = test_size
        self.class_column = class_column
        self.random_state = random_state
        random.seed(random_state)  # Set random seed for Python's random module
        np.random.seed(random_state)  # Set random seed for NumPy's random functions

    @staticmethod
    def random_select(data: dict, threshold: float, random_state: int = 42) -> list:
        """
        Selects keys from a dictionary randomly without exceeding a specified
        cumulative threshold. Continues to search for fitting classes even if one class exceeds the threshold.

        Parameters:
        - data (dict): A dictionary where keys are identifiers and values are weights.
        - threshold (float): The cumulative weight not to exceed.
        - random_state (int): Seed for the random number generator.

        Returns:
            list: A list of selected keys.
        """
        random.seed(random_state)
        keys = list(data.keys())
        random.shuffle(keys)  # Shuffle keys to ensure randomness in selection order
        selected_keys = []
        cumulative_sum = 0.0
        skipped_keys = []  # Temporarily hold keys that are too big at first glance

        for key in keys:
            if cumulative_sum + data[key] <= threshold:
                cumulative_sum += data[key]
                selected_keys.append(key)
            else:
                skipped_keys.append(key)

        # After initial pass, try adding any skipped keys that might fit now
        random.shuffle(
            skipped_keys
        )  # Reshuffle the skipped keys before trying to add them
        for key in skipped_keys:
            if cumulative_sum + data[key] <= threshold:
                cumulative_sum += data[key]
                selected_keys.append(key)

        return selected_keys

    def fit(self) -> tuple[DataFrame, DataFrame]:
        """
        Partitions the dataset into training and testing datasets.
        """
        class_counts = (
            self.data[self.class_column].value_counts(normalize=True).to_dict()
        )
        print(class_counts)
        test_class_list = RCSplit.random_select(
            class_counts, self.test_size, self.random_state
        )

        train_set = self.data[~self.data[self.class_column].isin(test_class_list)]
        test_set = self.data[self.data[self.class_column].isin(test_class_list)]

        return train_set, test_set
