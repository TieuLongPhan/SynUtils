import unittest
from synutility.SynIO.Format.chemical_conversion import smart_to_gml
from synutility.SynAAM.inference import aam_infer


class TestAAMInference(unittest.TestCase):

    def setUp(self):

        self.rsmi = "BrCc1ccc(Br)cc1.COCCO>>Br.COCCOCc1ccc(Br)cc1"
        self.gml = smart_to_gml("[Br:1][CH3:2].[OH:3][H:4]>>[Br:1][H:4].[CH3:2][OH:3]")
        self.expect = (
            "[Br:1][CH2:2][C:3]1=[CH:4][CH:6]=[C:7]([Br:8])[CH:9]"
            + "=[CH:5]1.[CH3:10][O:11][CH2:12][CH2:13][O:14][H:15]>>"
            + "[Br:1][H:15].[CH2:2]([C:3]1=[CH:4][CH:6]=[C:7]([Br:8])"
            + "[CH:9]=[CH:5]1)[O:14][CH2:13][CH2:12][O:11][CH3:10]"
        )

    def test_aam_infer(self):
        result = aam_infer(self.rsmi, self.gml)
        self.assertEqual(result[0], self.expect)


if __name__ == "__main__":
    unittest.main()
