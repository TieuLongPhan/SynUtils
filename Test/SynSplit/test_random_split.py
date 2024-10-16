import unittest
import pandas as pd

from synutility.SynSplit.random_split import RandomSplit


class TestRandomSplit(unittest.TestCase):
    def setUp(self):
        # Sample data setup
        self.data = pd.DataFrame(
            {"feature": [0.1, 0.2, 0.3, 0.4], "class_label": [1, 0, 1, 0]}
        )
        self.test_size = 0.5
        self.class_column = "class_label"
        self.random_state = 42

        # Instantiate RandomPartition
        self.random_partition = RandomSplit(
            data=self.data,
            test_size=self.test_size,
            class_column=self.class_column,
            random_state=self.random_state,
        )

    def test_constructor_and_attribute_assignment(self):
        # Test proper attribute assignments
        self.assertTrue(isinstance(self.random_partition.data, pd.DataFrame))
        self.assertEqual(self.random_partition.test_size, 0.5)
        self.assertEqual(self.random_partition.class_column, "class_label")
        self.assertEqual(self.random_partition.random_state, 42)

    def test_fit_returns_dataframes(self):
        train, test = self.random_partition.fit()
        # Test the types of the outputs
        self.assertTrue(isinstance(train, pd.DataFrame))
        self.assertTrue(isinstance(test, pd.DataFrame))

    def test_fit_partition_sizes(self):
        train, test = self.random_partition.fit()
        # Check the size of each partition
        self.assertEqual(len(train), 2)  # Expect 50% of 4 entries to be 2
        self.assertEqual(len(test), 2)


if __name__ == "__main__":
    unittest.main()
