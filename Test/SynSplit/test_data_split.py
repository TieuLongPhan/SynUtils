import unittest
import pandas as pd
from synutility.SynSplit.data_split import DataSplit


class TestDataSplit(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset
        self.data = pd.DataFrame(
            {
                "class": [
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "B",
                    "B",
                    "B",
                    "B",
                    "B",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                ]
            }
        )
        self.test_size = 0.2
        self.class_column = "class"
        self.random_state = 42

    def test_random_partition(self):
        # Test random partition method with real RandomPartition class
        partitioner = DataSplit(
            data=self.data,
            test_size=self.test_size,
            class_column=self.class_column,
            method="random",
            random_state=self.random_state,
        )
        train_data, test_data = partitioner.fit()

        # Check that the partitioning happened correctly
        self.assertEqual(len(train_data) + len(test_data), len(self.data))
        self.assertAlmostEqual(
            len(test_data) / len(self.data), self.test_size, delta=0.05
        )

    def test_stratified_partition(self):
        # Test stratified partition method with real StratifiedPartition class
        partitioner = DataSplit(
            data=self.data,
            test_size=self.test_size,
            class_column=self.class_column,
            method="stratified_target",
            random_state=self.random_state,
        )
        train_data, test_data = partitioner.fit()

        # Check that the partitioning happened correctly
        self.assertEqual(len(train_data) + len(test_data), len(self.data))
        self.assertAlmostEqual(
            len(test_data) / len(self.data), self.test_size, delta=0.05
        )

        # Ensure stratified split respects the class distribution
        self.assertTrue(
            set(train_data[self.class_column].unique())
            == set(test_data[self.class_column].unique())
        )

    def test_stratified_class_reduction_partition(self):
        # Test stratified class reduction partition method with real StratifiedReductionPartition class
        partitioner = DataSplit(
            data=self.data,
            test_size=self.test_size,
            class_column=self.class_column,
            method="stratified_class_reduction",
            random_state=self.random_state,
            drop_class_ratio=0.4,
        )
        train_data, test_data, removed_data = partitioner.fit()

        # Check that the partitioning happened correctly
        self.assertEqual(
            len(train_data) + len(test_data) + len(removed_data), len(self.data)
        )
        self.assertAlmostEqual(
            len(test_data) / len(self.data), self.test_size, delta=0.05
        )

        # Ensure that some data was removed based on the reduction logic
        self.assertGreater(len(removed_data), 0)

    def test_reaction_center_split(self):
        # Test stratified class reduction partition method with real StratifiedReductionPartition class
        partitioner = DataSplit(
            data=self.data,
            test_size=self.test_size,
            class_column=self.class_column,
            method="stratified_class_reduction",
            random_state=self.random_state,
            drop_class_ratio=0.4,
        )
        train_data, test_data, removed_data = partitioner.fit()

        # Check that the partitioning happened correctly
        self.assertEqual(
            len(train_data) + len(test_data) + len(removed_data), len(self.data)
        )
        self.assertAlmostEqual(
            len(test_data) / len(self.data), self.test_size, delta=0.05
        )

        # Ensure that some data was removed based on the reduction logic
        self.assertGreater(len(removed_data), 0)


if __name__ == "__main__":
    unittest.main()
