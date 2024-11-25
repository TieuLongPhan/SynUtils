import unittest
import networkx as nx

from synutility.SynChem.Reaction.standardize import Standardize
from synutility.SynIO.Format.chemical_conversion import (
    smiles_to_graph,
    rsmi_to_graph,
    graph_to_rsmi,
    smart_to_gml,
    gml_to_smart,
)

from synutility.SynGraph.Morphism.misc import rule_isomorphism


class TestChemicalConversions(unittest.TestCase):

    def setUp(self) -> None:
        self.rsmi = "[CH2:1]([H:4])[CH2:2][OH:3]>>[CH2:1]=[CH2:2].[H:4][OH:3]"
        self.gml = (
            "rule [\n"
            '   ruleID "rule"\n'
            "   left [\n"
            '      edge [ source 1 target 4 label "-" ]\n'
            '      edge [ source 1 target 2 label "-" ]\n'
            '      edge [ source 2 target 3 label "-" ]\n'
            "   ]\n"
            "   context [\n"
            '      node [ id 1 label "C" ]\n'
            '      node [ id 4 label "H" ]\n'
            '      node [ id 2 label "C" ]\n'
            '      node [ id 3 label "O" ]\n'
            "   ]\n"
            "   right [\n"
            '      edge [ source 1 target 2 label "=" ]\n'
            '      edge [ source 4 target 3 label "-" ]\n'
            "   ]\n"
            "]"
        )

        self.std = Standardize()

    def test_smiles_to_graph_valid(self):
        # Test converting a valid SMILES to a graph
        result = smiles_to_graph("[CH3:1][CH2:2][OH:3]", False, True, True)
        self.assertIsInstance(result, nx.Graph)
        self.assertEqual(result.number_of_nodes(), 3)

    def test_smiles_to_graph_invalid(self):
        # Test converting an invalid SMILES string to a graph
        result = smiles_to_graph("invalid_smiles", True, False, False)
        self.assertIsNone(result)

    def test_rsmi_to_graph_valid(self):
        # Test converting valid reaction SMILES to graphs for reactants and products
        reactants_graph, products_graph = rsmi_to_graph(self.rsmi, sanitize=True)
        self.assertIsInstance(reactants_graph, nx.Graph)
        self.assertEqual(reactants_graph.number_of_nodes(), 3)
        self.assertIsInstance(products_graph, nx.Graph)
        self.assertEqual(products_graph.number_of_nodes(), 3)

        reactants_graph, products_graph = rsmi_to_graph(self.rsmi, sanitize=False)
        self.assertIsInstance(reactants_graph, nx.Graph)
        self.assertEqual(reactants_graph.number_of_nodes(), 4)
        self.assertIsInstance(products_graph, nx.Graph)
        self.assertEqual(products_graph.number_of_nodes(), 4)

    def test_rsmi_to_graph_invalid(self):
        # Test handling of invalid RSMI format
        result = rsmi_to_graph("invalid_format")
        self.assertEqual((None, None), result)

    def test_graph_to_rsmi(self):
        r, p = rsmi_to_graph(self.rsmi, sanitize=False)
        rsmi = graph_to_rsmi(r, p)
        self.assertIsInstance(rsmi, str)
        self.assertEqual(self.std.fit(rsmi, False), self.std.fit(self.rsmi, False))

    def test_smart_to_gml(self):
        result = smart_to_gml(self.rsmi, core=False, sanitize=False, reindex=False)
        self.assertIsInstance(result, str)
        self.assertEqual(result, self.gml)

        result = smart_to_gml(self.rsmi, core=False, sanitize=False, reindex=True)
        self.assertTrue(rule_isomorphism(result, self.gml))

    def test_gml_to_smart(self):
        smarts, _ = gml_to_smart(self.gml)
        self.assertIsInstance(smarts, str)
        self.assertEqual(self.std.fit(smarts, False), self.std.fit(self.rsmi, False))


if __name__ == "__main__":
    unittest.main()
