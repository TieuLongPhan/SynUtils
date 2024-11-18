import unittest
from synutility.SynAAM.partial_expand import PartialExpand


class TestPartialExpand(unittest.TestCase):
    def test_expand(self):
        """
        Test the expand function of the PartialExpand class with a given RSMI.
        """
        # Input RSMI
        input_rsmi = "[CH3][CH:1]=[CH2:2].[H:3][H:4]>>[CH3][CH:1]([H:3])[CH2:2][H:4]"
        # Expected output
        expected_rsmi = (
            "[CH2:1]=[CH:2][CH3:3].[H:4][H:5]>>[CH2:1]([CH:2]([CH3:3])[H:5])[H:4]"
        )
        # Perform the expansion
        output_rsmi = PartialExpand.expand(input_rsmi)
        # Assert the result matches the expected output
        self.assertEqual(output_rsmi, expected_rsmi)

    def test_expand_2(self):
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
        output_rsmi = PartialExpand.expand(input_rsmi)
        expected_rsmi = (
            "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5][H:6]"
            + ">>[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4][H:6]"
        )
        self.assertEqual(output_rsmi, expected_rsmi)


if __name__ == "__main__":
    unittest.main()
