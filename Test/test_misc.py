import unittest
from synutility.misc import stratified_random_sample, calculate_processing_time


class TestStratifiedRandomSample(unittest.TestCase):

    def setUp(self):
        self.data = [
            {"id": 1, "type": "A"},
            {"id": 2, "type": "A"},
            {"id": 3, "type": "B"},
            {"id": 4, "type": "B"},
            {"id": 5, "type": "B"},
            {"id": 6, "type": "C"},  # Adding a smaller group for bypass testing
        ]

    def test_valid_input(self):
        sampled = stratified_random_sample(self.data, "type", 1, seed=42)
        self.assertEqual(len(sampled), 3)  # A, B, C should each have 1 sample

    def test_insufficient_data_raise_error(self):
        with self.assertRaises(ValueError):
            stratified_random_sample(self.data, "type", 3)

    def test_no_such_key(self):
        sampled = stratified_random_sample(self.data, "category", 1, bypass=True)
        print(sampled)
        self.assertEqual(len(sampled), 0)  # Expects empty list if key does not exist

    def test_insufficient_data_bypass(self):
        sampled = stratified_random_sample(self.data, "type", 3, bypass=True)
        self.assertEqual(len(sampled), 3)  # Expects 3 from group B, skips others

    def test_bypass_all_groups(self):
        sampled = stratified_random_sample(self.data, "type", 10, bypass=True)
        self.assertEqual(len(sampled), 0)  # Expects all groups to be bypassed

    def test_seed_reproducibility(self):
        sample1 = stratified_random_sample(self.data, "type", 1, seed=42)
        sample2 = stratified_random_sample(self.data, "type", 1, seed=42)
        self.assertEqual(sample1, sample2)  # Same seed should give the same result


class TestCalculateProcessingTime(unittest.TestCase):

    def test_valid_input(self):
        result = calculate_processing_time(
            "2020-01-01 12:00:00,000", "2020-01-01 12:00:10,000"
        )
        self.assertEqual(result, 10.0)

    def test_invalid_format(self):
        with self.assertRaises(ValueError):
            calculate_processing_time("2020-01-01 12:00", "2020-01-01 12:01:00,000")

    def test_negative_duration(self):
        result = calculate_processing_time(
            "2020-01-01 12:01:00,000", "2020-01-01 12:00:00,000"
        )
        self.assertEqual(result, -60.0)


if __name__ == "__main__":
    unittest.main()
