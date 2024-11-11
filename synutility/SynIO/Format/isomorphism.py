import networkx as nx
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match
from operator import eq
from typing import List, Any


def isomorphism_check(
    graph_i: nx.Graph,
    graph_j: nx.Graph,
    node_label_names: List[str] = ["element", "charge"],
    node_label_default: List[Any] = ["*", 0],
    edge_attribute: str = "order",
    edge_default: Any = 1,
) -> bool:
    """
    Checks if two graphs are isomorphic based on specified node and edge attributes.

    Parameters:
    - graph_i (nx.Graph): First graph to be compared.
    - graph_j (nx.Graph): Second graph to be compared.
    - node_label_names (List[str]): List of node attribute names to consider for isomorphism.
    - node_label_default (List[Any]): Defaults for node attributes if not present.
    - edge_attribute (str): Edge attribute name to consider for isomorphism.
    - edge_default (Any): Default value for the edge attribute if not present.

    Returns:
    - bool: True if the graphs are isomorphic considering specified attributes, else False.
    """

    # Prepare the operator list for node matching
    node_label_operators = [eq for _ in node_label_names]

    # Create node and edge match functions using generic matchers
    node_match = generic_node_match(
        node_label_names,  # attribute names
        node_label_default,  # default values for each attribute
        node_label_operators,  # operators for each attribute comparison
    )
    edge_match = generic_edge_match(
        edge_attribute,  # The attribute name to compare for edges
        edge_default,  # Default value if the attribute is missing
        eq,  # Operator for comparing edge attributes
    )

    # Use the isomorphic check with node and edge match functions
    return nx.is_isomorphic(
        graph_i, graph_j, node_match=node_match, edge_match=edge_match
    )
