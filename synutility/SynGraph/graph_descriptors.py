import networkx as nx
from typing import List, Dict, Any, Union
from collections import Counter, OrderedDict
from synutility.SynIO.debug import setup_logging

logger = setup_logging()


class GraphDescriptor:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    @staticmethod
    def is_graph_empty(graph: Union[nx.Graph, dict, list, Any]) -> bool:
        """
        Determine if a graph representation is empty.

        This function checks for emptiness in various types of graph representations,
        including NetworkX graphs, dictionaries (potentially adjacency lists),
        lists (adjacency matrices), and custom graph classes with an 'is_empty' method.

        Parameters:
        - graph (Union[nx.Graph, dict, list, Any]): A graph representation which can be
        a NetworkX graph, a dictionary, a list, or an object with a 'is_empty' method.

        Returns:
        - bool: Returns True if the graph is empty (no nodes/vertices), otherwise False.

        Raises:
        - TypeError: If the graph representation is not supported.
        """
        if isinstance(graph, nx.Graph):
            return graph.number_of_nodes() == 0
        elif isinstance(graph, dict):
            return len(graph) == 0
        elif isinstance(graph, list):
            return all(len(row) == 0 for row in graph)
        elif hasattr(graph, "is_empty"):
            return graph.is_empty()
        else:
            raise TypeError("Unsupported graph representation")

    @staticmethod
    def is_acyclic_graph(G: nx.Graph) -> bool:
        """
        Determines if the given graph is acyclic.

        Parameters:
        - G (nx.Graph): The graph to be checked.

        Returns:
        - bool: True if the graph is acyclic, False otherwise.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError("Input must be a networkx Graph object.")
        if GraphDescriptor.is_graph_empty(G):
            return False

        return nx.is_tree(G)

    @staticmethod
    def is_single_cyclic_graph(G: nx.Graph) -> bool:
        """
        Determines if the given graph is a single cyclic graph,
        which means the graph has exactly one cycle
        and all nodes in the graph are part of that cycle.

        Parameters:
        - G (nx.Graph): The graph to be checked.

        Returns:
        - bool: True if the graph is single cyclic, False otherwise.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError("Input must be a networkx Graph object.")

        if GraphDescriptor.is_graph_empty(G):
            return False

        if not nx.is_connected(G):
            return False

        cycles = nx.cycle_basis(G)

        if cycles:
            nodes_in_cycles = set(node for cycle in cycles for node in cycle)
            if (
                nodes_in_cycles == set(G.nodes())
                and G.number_of_edges() == G.number_of_nodes()
            ):
                return True

        return False

    @staticmethod
    def is_complex_cyclic_graph(G: nx.Graph) -> bool:
        """
        Determines if the given graph is a complex cyclic graph,
        which means all nodes are part of cycles,
        there are multiple cycles, and there are no acyclic parts.

        Parameters:
        - G (nx.Graph): The graph to be checked.

        Returns:
        - bool: True if the graph is complex cyclic, False otherwise.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError("Input must be a networkx Graph object.")

        if GraphDescriptor.is_graph_empty(G):
            return False

        # Check if the graph is connected and has at least one cycle
        if not nx.is_connected(G) or not any(nx.minimum_cycle_basis(G)):
            return False

        # Get a list of cycles that form a cycle basis for G
        cycles = nx.minimum_cycle_basis(G)

        # If there's only one cycle in the basis, it might not be a complex cyclic graph
        if len(cycles) <= 1:
            return False

        # Decompose cycles into a list of nodes, allowing for node overlap between cycles
        nodes_in_cycles = set(node for cycle in cycles for node in cycle)

        # Check if all nodes in G are covered by the nodes in cycles
        return nodes_in_cycles == set(G.nodes())

    @staticmethod
    def check_graph_type(G: nx.Graph) -> str:
        """
        Determines if the given graph is acyclic, single cyclic, or complex cyclic.

        Parameters:
        - G (nx.Graph): The graph to be checked.

        Returns:
        - str: A string indicating if the graph is "Acyclic",
                "Single Cyclic", or "Complex Cyclic".

        Raises:
        - TypeError: If the input G is not a networkx Graph.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError("Input must be a networkx Graph object.")

        if GraphDescriptor.is_graph_empty(G):
            return "Empty Graph"
        if GraphDescriptor.is_acyclic_graph(G):
            return "Acyclic"
        elif GraphDescriptor.is_single_cyclic_graph(G):
            return "Single Cyclic"
        elif GraphDescriptor.is_complex_cyclic_graph(G):
            return "Combinatorial Cyclic"
        else:
            return "Complex Cyclic"

    @staticmethod
    def get_cycle_member_rings(G: nx.Graph) -> List[int]:
        """
        Identifies all cycles in the given graph using cycle bases to ensure no overlap
        and returns a list of the sizes of these cycles (member rings),
        sorted in ascending order.

        Parameters:
        - G (nx.Graph): The NetworkX graph to be analyzed.

        Returns:
        - List[int]: A sorted list of cycle sizes (member rings) found in the graph.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError("Input must be a networkx Graph object.")

        # Find cycle basis for the graph which gives non-overlapping cycles
        cycles = nx.minimum_cycle_basis(G)
        # Determine the size of each cycle (member ring)
        member_rings = [len(cycle) for cycle in cycles]

        # Sort the sizes in ascending order
        member_rings.sort()

        return member_rings

    @staticmethod
    def get_element_count(graph: nx.Graph) -> Dict[str, int]:
        """
        Counts the occurrences of each chemical element in the graph nodes and returns
        a dictionary with these counts sorted alphabetically for easy comparison.

        Parameters:
        - graph (nx.Graph): A NetworkX graph object where each node has attributes including 'element'.

        Returns:
        - Dict[str, int]: An ordered dictionary with element symbols as keys and their counts as values,
        sorted alphabetically by element symbol.
        """
        # Use Counter to count occurrences of each element
        element_counts = Counter(
            [data["element"] for _, data in graph.nodes(data=True)]
        )

        # Create an ordered dictionary sorted alphabetically by element
        ordered_counts = OrderedDict(sorted(element_counts.items()))

        return ordered_counts

    @staticmethod
    def get_descriptors(data: List[Dict], reaction_centers: str = "RC") -> List[Dict]:
        """
        Enhance data with topology type and reaction type descriptors.

        Parameters:
        - data (List[Dict]): List of dictionaries containing reaction data.
        - reaction_centers (str): Key for accessing the reaction centers in the data dictionaries.

        Returns:
        - List[Dict]: Enhanced list of dictionaries with added descriptors.
        """
        for entry in data:
            rc_data = entry.get(reaction_centers)
            if isinstance(rc_data, list):
                try:
                    graph = rc_data[2]
                except IndexError:
                    logger.error(
                        f"No graph data available at index 2 for entry {entry}"
                    )
                    continue
            elif isinstance(rc_data, nx.Graph):
                graph = rc_data
            else:
                logger.error(
                    f"Unsupported data type for reaction centers in entry {entry}"
                )
                continue

            # Enhance the dictionary with additional descriptors
            entry["topo"] = GraphDescriptor.check_graph_type(graph)
            entry["cycle"] = GraphDescriptor.get_cycle_member_rings(graph)
            entry["atom_count"] = GraphDescriptor.get_element_count(graph)

            # Determine the reaction type based on the topology type
            if entry["topo"] in ["Single Cyclic", "Acyclic"]:
                entry["rtype"] = "Elementary"
            else:
                entry["rtype"] = "Complicated"

            # Adjust "Rings" and "Reaction Step" based on the topology type
            if entry["topo"] == "Acyclic":
                entry["cycle"] = [0]  # No rings in acyclic graphs
            elif entry["topo"] == "Complex Cyclic":
                entry["cycle"] = [0] + entry[
                    "cycle"
                ]  # Prepending zero might represent a base cycle count

            entry["rstep"] = len(entry["cycle"])  # Steps are based on cycle counts

        return data


def check_graph_connectivity(graph: nx.Graph) -> str:
    """
    Check the connectivity of a NetworkX graph.

    This function assesses whether all nodes in the graph are connected by some path,
    applicable to undirected graphs.

    Parameters:
    - graph (nx.Graph): A NetworkX graph object.

    Returns:
    - str: Returns 'Connected' if the graph is connected, otherwise 'Disconnected'.

    Raises:
    - NetworkXNotImplemented: If graph is directed and does not support is_connected.
    """
    if nx.is_connected(graph):
        return "Connected"
    else:
        return "Disconnected."
