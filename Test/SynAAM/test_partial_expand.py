import unittest
from synutility.SynAAM.aam_validator import AAMValidator
from synutility.SynAAM.its_construction import ITSConstruction
from synutility.SynIO.Format.chemical_conversion import rsmi_to_graph

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
        output_rsmi = PartialExpand.expand_aam_with_transform(input_rsmi)
        # Assert the result matches the expected output
        self.assertTrue(AAMValidator.smiles_check(output_rsmi, expected_rsmi, "ITS"))

    def test_expand_2(self):
        input_rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
        output_rsmi = PartialExpand.expand_aam_with_transform(input_rsmi)
        expected_rsmi = (
            "[CH3:1][CH2:2][CH2:3][Cl:4].[NH2:5][H:6]"
            + ">>[CH3:1][CH2:2][CH2:3][NH2:5].[Cl:4][H:6]"
        )
        self.assertTrue(AAMValidator.smiles_check(output_rsmi, expected_rsmi, "ITS"))

    def test_graph_expand(self):
        input_rsmi = "BrCc1ccc(Br)cc1.COCCO>>Br.COCCOCc1ccc(Br)cc1"
        expect = (
            "[Br:1][CH2:2][C:3]1=[CH:4][CH:6]=[C:7]([Br:8])[CH:9]=[CH:5]1."
            + "[CH3:10][O:11][CH2:12][CH2:13][O:14][H:15]>>[Br:1][H:15]"
            + ".[CH2:2]([C:3]1=[CH:4][CH:6]=[C:7]([Br:8])[CH:9]=[CH:5]1)"
            + "[O:14][CH2:13][CH2:12][O:11][CH3:10]"
        )
        r, p = rsmi_to_graph(
            "[Br:1][CH3:2].[OH:3][H:4]>>[Br:1][H:4].[CH3:2][OH:3]", sanitize=False
        )
        its = ITSConstruction().ITSGraph(r, p)
        output = PartialExpand.graph_expand(its, input_rsmi)
        self.assertTrue(AAMValidator.smiles_check(output, expect))


if __name__ == "__main__":
    unittest.main()
