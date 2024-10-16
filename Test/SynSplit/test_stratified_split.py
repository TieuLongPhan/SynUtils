import unittest
import pandas as pd
import numpy as np

from synutility.SynSplit.stratified_split import StratifiedSplit


class TestStratifiedSplit(unittest.TestCase):
    def setUp(self):
        # Sample data setup
        self.data = pd.DataFrame(
            {
                "feature": np.random.rand(100),
                "class_label": np.random.choice(
                    [0, 1], size=100, p=[0.7, 0.3]
                ),  # 70% 0s and 30% 1s
            }
        )
        self.test_size = 0.25
        self.class_column = "class_label"
        self.random_state = 42

        # Instantiate StratifiedPartition
        self.strat_partition = StratifiedSplit(
            data=self.data,
            test_size=self.test_size,
            class_column=self.class_column,
            random_state=self.random_state,
        )

    def test_constructor_and_attribute_assignment(self):
        # Test proper attribute assignments
        self.assertEqual(self.strat_partition.test_size, 0.25)
        self.assertEqual(self.strat_partition.class_column, "class_label")
        self.assertEqual(self.strat_partition.random_state, 42)

    def test_constructor_raises_for_invalid_class_column(self):
        with self.assertRaises(ValueError):
            StratifiedSplit(
                data=pd.DataFrame({"feature": [1, 2, 3]}),
                test_size=0.25,
                class_column="nonexistent",
                random_state=42,
            )

    def test_constructor_raises_for_invalid_test_size(self):
        with self.assertRaises(ValueError):
            StratifiedSplit(
                data=self.data,
                test_size=-0.1,
                class_column=self.class_column,
                random_state=42,
            )
        with self.assertRaises(ValueError):
            StratifiedSplit(
                data=self.data,
                test_size=1.5,
                class_column=self.class_column,
                random_state=42,
            )

    def test_fit_returns_dataframes(self):
        train, test = self.strat_partition.fit()
        # Test the types of the outputs
        self.assertTrue(isinstance(train, pd.DataFrame))
        self.assertTrue(isinstance(test, pd.DataFrame))

    def test_fit_stratified_proportions(self):
        train, test = self.strat_partition.fit()
        # Check the proportion of classes in the training and test sets
        train_proportion = train[self.class_column].value_counts(normalize=True)
        test_proportion = test[self.class_column].value_counts(normalize=True)
        np.testing.assert_almost_equal(train_proportion[1], 0.3, decimal=1)
        np.testing.assert_almost_equal(test_proportion[1], 0.3, decimal=1)


if __name__ == "__main__":
    unittest.main()
