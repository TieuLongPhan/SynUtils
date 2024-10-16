import unittest
import pandas as pd
from synutility.SynSplit.stratified_reduction_partition import (
    StratifiedReductionSplit,
)


class TestStratifiedReductionSplit(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        data = pd.DataFrame(
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
        self.srp = StratifiedReductionSplit(
            data=data,
            test_size=0.2,
            drop_class_ratio=0.1,
            class_column="class",
            random_state=42,
        )

    def test_initialization(self):
        self.assertIsInstance(self.srp.data, pd.DataFrame)
        self.assertEqual(self.srp.test_size, 0.2)
        self.assertEqual(self.srp.drop_class_ratio, 0.1)
        self.assertEqual(self.srp.class_column, "class")
        self.assertEqual(self.srp.random_state, 42)

    def test_random_select(self):
        test_dict = {"A": 0.5, "B": 0.3, "C": 0.2}
        selected = StratifiedReductionSplit.random_select(
            test_dict, 0.2, random_state=42
        )
        self.assertIn("C", selected)
        self.assertTrue(sum([test_dict[x] for x in selected]) <= 0.2)

    def test_fit(self):
        train, test, removed = self.srp.fit()
        # Check the correct DataFrame types
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)
        self.assertIsInstance(removed, pd.DataFrame)
        # Ensure proper partitioning
        self.assertLessEqual(len(test), int(0.2 * len(self.srp.data)))
        # Check if classes were reduced correctly in training data
        all_classes = set(self.srp.data["class"])
        train_classes = set(train["class_mapping"])
        removed_classes = set(removed["class_mapping"])
        self.assertTrue(train_classes.isdisjoint(removed_classes))
        # Ensure removed classes are not in the main training set
        self.assertTrue(all_classes.intersection(removed_classes).issubset(all_classes))


if __name__ == "__main__":
    unittest.main()
