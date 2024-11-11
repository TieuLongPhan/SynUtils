import networkx as nx
from collections import Counter


class GraphSignature:
    """
    Provides methods to generate canonical signatures for graph nodes, edges, and complete graphs,
    useful for comparisons or identification in graph-based data structures.

    Attributes:
        graph (nx.Graph): The graph for which signatures will be generated.
    """

    def __init__(self, graph: nx.Graph):
        """
        Initializes the GraphSignature class with a specified graph.

        Parameters:
        - graph (nx.Graph): A NetworkX graph instance.
        """
        self.graph = graph

    def create_node_signature(self, condensed: bool = True) -> str:
        """
        Generates a canonical node signature. If `condensed` is True, it condenses
        consecutive occurrences of elements, formatting like 'Br{1}C{10}'.

        Parameters:
        - condensed (bool): If True, condenses elements with counts. If False, keeps the original format.

        Returns:
        - str: A concatenated string of sorted node elements, optionally with counts.
        """
        # Sort elements
        elements = sorted(data["element"] for _, data in self.graph.nodes(data=True))

        if condensed:
            # Count occurrences and format with counts
            element_counts = Counter(elements)
            signature_parts = []
            for element, count in element_counts.items():
                if count > 1:
                    signature_parts.append(f"{element}{{{count}}}")
                else:
                    signature_parts.append(element)
            return "".join(signature_parts)
        else:
            # Return the original, uncompressed format
            return "".join(elements)

    def create_edge_signature(self) -> str:
        """
        Generates a canonical edge signature by formatting each edge with sorted node elements and a bond order,
        separated by '/', with each edge represented as 'node1[standard_order]node2'.

        Returns:
        - str: A concatenated and sorted string of edge representations.
        """
        edge_signature_parts = []
        for u, v, data in self.graph.edges(data=True):
            standard_order = int(
                data.get("standard_order", 1)
            )  # Default to 1 if missing
            node1, node2 = sorted(
                [self.graph.nodes[u]["element"], self.graph.nodes[v]["element"]]
            )
            part = f"{node1}[{standard_order}]{node2}"
            edge_signature_parts.append(part)
        return "/".join(sorted(edge_signature_parts))

    def create_topology_signature(self, topo, cycle, rstep) -> str:
        """
        Generates a topology signature for the graph based on its cyclic properties and structure.
        The topology is classified and quantified by identifying cycles and other structural features.

        Returns:
        - str: A string representing the numerical and qualitative topology signature of the graph.
        """

        topo_mapping = {
            "Acyclic": 0,
            "Single Cyclic": 1,
            "Combinatorial Cyclic": 2,
            "Complex Cyclic": 3,
        }

        topo_code = topo_mapping.get(topo, 4)

        rstep = len(cycle)
        cycle_str = "".join(map(str, cycle))
        return f"{rstep}{topo_code}{cycle_str}"

    def create_graph_signature(
        self,
        condensed: bool = True,
        topology: bool = True,
        nodes: bool = True,
        edges: bool = True,
        topo: str = None,
        cycle: list = None,
        rstep: int = None,
    ) -> str:
        """
        Combines node, edge, and topology signatures into a single comprehensive graph signature.

        Returns:
        - str: A concatenated string representing the complete graph signature formatted as
          'topology_signature.node_signature.edge_signature'.
        """
        if topology:
            topo_signature = self.create_topology_signature(topo, cycle, rstep)
        else:
            topo_signature = ""
        if nodes:
            node_signature = self.create_node_signature(condensed)
        else:
            node_signature = ""
        if edges:
            edge_signature = self.create_edge_signature()
        else:
            edge_signature = ""
        return f"{topo_signature}.{node_signature}.{edge_signature}"
