import unittest
from synutility.SynIO.data_type import load_from_pickle
from synutility.SynGraph.Descriptor.graph_signature import GraphSignature


class TestGraphSignature(unittest.TestCase):

    def setUp(self):
        # Create a sample graph for testing
        data = load_from_pickle("Data/test.pkl.gz")
        self.rc = data[0]["GraphRules"][2]
        self.its = data[0]["ITSGraph"][2]

    def test_create_topology_signature(self):
        signature = GraphSignature(self.rc)
        self.assertEqual(
            signature.create_topology_signature(
                topo="Single Cyclic", cycle=[4], rstep=1
            ),
            "114",
        )

    def test_create_node_signature(self):
        signature = GraphSignature(self.rc)
        self.assertEqual(signature.create_node_signature(), "BrCHN")

    def test_create_node_signature_condensed(self):
        signature = GraphSignature(self.its)
        self.assertEqual(signature.create_node_signature(), "BrC{23}ClHN{3}O{5}S")

    def test_create_edge_signature(self):
        signature = GraphSignature(self.rc)
        self.assertEqual(
            signature.create_edge_signature(), "Br[-1]H/Br[1]C/C[-1]N/H[1]N"
        )

    def test_create_graph_signature(self):
        # Ensure the graph signature combines the results correctly
        signature = GraphSignature(self.rc)
        node_signature = "BrCHN"
        edge_signature = "Br[-1]H/Br[1]C/C[-1]N/H[1]N"
        topo_signature = "114"
        expected = f"{topo_signature}.{node_signature}.{edge_signature}"
        self.assertEqual(
            signature.create_graph_signature(topo="Single Cyclic", cycle=[4], rstep=1),
            expected,
        )


# Running the tests
if __name__ == "__main__":
    unittest.main()
