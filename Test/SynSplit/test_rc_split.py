import unittest
import numpy as np
import pandas as pd
from synutility.SynSplit.rc_split import RCSplit


class TestRCSplit(unittest.TestCase):

    def setUp(self):
        """Set up a simple dataset for testing."""
        np.random.seed(42)
        # Create a DataFrame with 100 entries and a skewed distribution of classes
        data = {
            "class": np.random.choice(
                ["A", "B", "C", "D", "E"], p=[0.2, 0.3, 0.3, 0.1, 0.1], size=100
            ),
            "feature": np.random.rand(100),
        }
        self.df = pd.DataFrame(data)

        self.test_size = 0.2  # 20% of data as test set
        self.random_state = 42
        self.class_column = "class"

    def test_fit_function(self):
        """Test the fit function to ensure it divides the dataset correctly
        according to the test_size."""
        splitter = RCSplit(
            data=self.df,
            test_size=self.test_size,
            class_column=self.class_column,
            random_state=self.random_state,
        )
        train_set, test_set = splitter.fit()

        # Check the proportion of the test set
        actual_test_size = len(test_set) / len(self.df)
        self.assertAlmostEqual(
            actual_test_size,
            self.test_size,
            delta=0.05,
            msg="Test set size is not within the acceptable range.",
        )

        # Verify that test set does not exceed the test_size even slightly
        self.assertTrue(
            actual_test_size <= self.test_size,
            "Test set size exceeds the specified test_size.",
        )

        # Check if there's no overlap in the indices between train and test sets
        self.assertEqual(
            len(set(train_set.index).intersection(set(test_set.index))),
            0,
            "Train and test sets have overlapping indices.",
        )

        self.assertEqual(
            len(
                set(train_set[self.class_column]).intersection(
                    set(test_set[self.class_column])
                )
            ),
            0,
            "Train and test sets have overlapping indices.",
        )

    # def test_randomness_of_split(self):
    #     """Test if different random states produce different splits."""
    #     splitter1 = RCSplit(
    #         data=self.df,
    #         test_size=0.1,
    #         class_column=self.class_column,
    #         random_state=42,
    #     )
    #     splitter2 = RCSplit(
    #         data=self.df,
    #         test_size=0.1,
    #         class_column=self.class_column,
    #         random_state=1,
    #     )

    #     _, test_set1 = splitter1.fit()
    #     _, test_set2 = splitter2.fit()

    #     # Assert that the two test sets with different random seeds are not the same
    #     self.assertNotEqual(
    #         test_set1.index.tolist(),
    #         test_set2.index.tolist(),
    #         "Test sets are identical for different random states.",
    #     )


if __name__ == "__main__":
    unittest.main()
