from typing import Tuple
from pandas import DataFrame
from sklearn.model_selection import train_test_split


class StratifiedSplit:
    """
    Class for partitioning data into training and test sets using stratified sampling based on
    the specified class column to maintain the proportion of classes in each subset.

    Attributes:
        data (DataFrame): The dataset to partition.
        test_size (float): The proportion of the dataset to include in the test split.
        class_column (str): The name of the column in `data` that represents the class labels.
        random_state (int): Controls the shuffling applied to the data before applying the split.
    """

    def __init__(
        self, data: DataFrame, test_size: float, class_column: str, random_state: int
    ) -> None:
        """
        Initializes the StratifiedSplit instance with the given data and parameters.

        Parameters:
            data (DataFrame): The dataset to be partitioned.
            test_size (float): The proportion of the dataset to be included in the test set.
            class_column (str): The column that indicates the class label of the data.
            random_state (int): The seed used by the random number generator to ensure reproducibility.
        """
        if class_column not in data.columns:
            raise ValueError(
                f"The specified class column '{class_column}' does not exist in the DataFrame."
            )
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1.")

        self.data = data
        self.test_size = test_size
        self.class_column = class_column
        self.random_state = random_state

    def fit(self) -> Tuple[DataFrame, DataFrame]:
        """
        Partitions the dataset into training and test datasets using stratified sampling to
        maintain the percentage of samples for each class.

        Returns:
            tuple[DataFrame, DataFrame]: A tuple containing the training data as the first element
            and the test data as the second element.
        """
        # Perform the stratified split
        data_train, data_test = train_test_split(
            self.data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.data[self.class_column],
        )

        return data_train, data_test
