from pandas import DataFrame
from sklearn.model_selection import train_test_split


class RandomSplit:
    """
    Class for partitioning data into training and test sets based on specified class columns.

    Attributes:
    - data (DataFrame): The dataset to partition.
    - test_size (float): The proportion of the dataset to include in the test split.
    - class_column (str): The name of the column in `data` that represents the class labels.
    - random_state (int): Controls the shuffling applied to the data before applying the split.
    """

    def __init__(
        self, data: DataFrame, test_size: float, class_column: str, random_state: int
    ) -> None:
        """
        Initializes the RandomSplit with the given data and parameters.

        Parameters:
        - data (DataFrame): The dataset to be partitioned.
        - test_size (float): The proportion of the dataset to be included in the test set.
        - class_column (str): The column that indicates the class label of the data.
        - random_state (int): The seed used by the random number generator.
        """
        self.data = data
        self.test_size = test_size
        self.class_column = class_column
        self.random_state = random_state

    def fit(self) -> tuple[DataFrame, DataFrame]:
        """
        Partitions the dataset into training and testing datasets.

        Returns:
        - tuple[DataFrame, DataFrame]: A tuple containing the training data as the first element
                                         and the test data as the second element.
        """
        data_train, data_test = train_test_split(
            self.data, test_size=self.test_size, random_state=self.random_state
        )
        return data_train, data_test
