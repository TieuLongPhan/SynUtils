from typing import Tuple, Union
import pandas as pd
from synutility.SynSplit.random_split import RandomSplit
from synutility.SynSplit.stratified_split import StratifiedSplit
from synutility.SynSplit.stratified_reduction_partition import (
    StratifiedReductionSplit,
)
from synutility.SynSplit.rc_split import RCSplit
from synutility.SynIO.debug import setup_logging
from synutility.SynIO.data_type import load_database

logger = setup_logging()


class DataSplit:
    """
    Class for partitioning a dataset into training and testing sets using various
    partitioning methods.

    Attributes:
    - data (pd.DataFrame): The dataset to be partitioned.
    - test_size (float): The proportion of the dataset to include in the test split,
      between 0.0 and 1.0.
    - class_column (str): Column name in the dataset that contains class labels.
    - method (str): Method used for data partitioning; valid options include
      'random', 'stratified_target', and 'stratified_class_reduction'.
    - drop_class_ratio (float): The maximum cumulative proportion of classes to remove
      from the training data, used with 'stratified_class_reduction' method.
      Default is 0.1.
    - random_state (int): Seed for random number generation to ensure reproducibility.
    - keep_data (bool): Whether to return the data removed from the training set or not.
    """

    def __init__(
        self,
        data,
        test_size: float,
        class_column: str,
        method: str,
        random_state: int,
        drop_class_ratio: float = 0.1,
        keep_data: bool = True,
    ) -> None:
        """
        Initializes the DataSplit instance with the specified parameters.

        Parameters:
        - data: Dataset to be partitioned, can be a pd.DataFrame or a path to
        a json.gz file.
        - test_size (float): Proportion of the dataset to include in the test split.
        - class_column (str): Column name containing class labels.
        - method (str): Partitioning method ('random', 'stratified_target',
          'stratified_class_reduction').
        - random_state (int): Random seed for reproducibility.
        - drop_class_ratio (float, optional): Ratio of minority classes to remove
          (default 0.1).
        - keep_data (bool): Return also data removed or not.
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str) and data.endswith(".json.gz"):
            self.data = load_database(data)
        else:
            logger.error(
                "Unsupported data format. Please provide "
                + "a pandas DataFrame or a .json.gz file path."
            )
            raise ValueError(
                "Unsupported data format. Please provide "
                + "a pandas DataFrame or a .json.gz file path."
            )

        self.test_size = test_size
        self.class_column = class_column
        self.method = self._validate_method(method)
        self.drop_class_ratio = drop_class_ratio
        self.random_state = random_state
        self.keep_data = keep_data

    def _validate_method(self, method: str) -> str:
        """Validate the partitioning method."""
        valid_methods = [
            "random",
            "stratified_target",
            "stratified_class_reduction",
            "rc_split",
        ]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'."
                + " Valid options are: {', '.join(valid_methods)}"
            )
        return method

    def fit(
        self,
    ) -> Union[
        Tuple[pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ]:
        """
        Partitions the dataset into training and testing sets according to the specified
        partitioning method.

        Returns:
        - Tuple containing the training and testing datasets. If using
        'stratified_class_reduction', a third dataset containing the removed data
        is also returned.

        Raises:
            ValueError: If an invalid partitioning method is provided.
        """
        if self.method == "random":
            logger.info("Partition data using random approach")
            splitter = RandomSplit(
                data=self.data,
                test_size=self.test_size,
                class_column=self.class_column,
                random_state=self.random_state,
            )

            return splitter.fit()
        elif self.method == "stratified_target":
            logger.info("Partition data using stratify approach")
            splitter = StratifiedSplit(
                data=self.data,
                test_size=self.test_size,
                class_column=self.class_column,
                random_state=self.random_state,
            )
            return splitter.fit()
        elif self.method == "stratified_class_reduction":
            logger.info("Partition data using stratify reduction approach")
            splitter = StratifiedReductionSplit(
                data=self.data,
                test_size=self.test_size,
                class_column=self.class_column,
                drop_class_ratio=self.drop_class_ratio,
                random_state=self.random_state,
                keep_data=self.keep_data,
            )

            return splitter.fit()
        elif self.method == "rc_split":
            logger.info("Partition data using reaction center approach")
            splitter = RCSplit(
                data=self.data,
                test_size=self.test_size,
                class_column=self.class_column,
                random_state=self.random_state,
            )
            return splitter.fit()
        else:
            logger.error(
                f"Invalid method '{self.method}'. Use 'random',"
                + " 'stratified_target', or 'stratified_class_reduction'."
            )
            raise ValueError(
                f"Invalid method '{self.method}'. Use 'random',"
                + " 'stratified_target', or 'stratified_class_reduction'."
            )
