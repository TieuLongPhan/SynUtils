import re
import networkx as nx
from typing import Tuple
from synutility.SynAAM.its_construction import ITSConstruction


class GMLToNX:
    def __init__(self, gml_text: str):
        """
        Initializes a GMLToNX object that can parse GML-like text into separate
        NetworkX graphs representing different stages or components of
        a chemical reaction.

        Parameters:
        - gml_text (str): The GML-like text content that will be parsed into graphs.
        """
        self.gml_text = gml_text
        self.graphs = {"left": nx.Graph(), "context": nx.Graph(), "right": nx.Graph()}

    def _parse_element(self, line: str, current_section: str):
        """
        Parses a line of GML-like text to extract node or edge data and adds it to the
        current section's graph.

        Parameters:
        - line (str): A line from the GML-like text representing either a node or an edge.
        - current_section (str): The graph section (left, context, right) to which the
        node or edge belongs.
        """
        label_to_order = {"-": 1, ":": 1.5, "=": 2, "#": 3}
        tokens = line.split()
        if "node" in line:
            node_id = int(tokens[tokens.index("id") + 1])
            label = tokens[tokens.index("label") + 1].strip('"')
            element, charge = self._extract_element_and_charge(label)
            node_attributes = {
                "element": element,
                "charge": charge,
                "atom_map": node_id,
            }
            self.graphs[current_section].add_node(node_id, **node_attributes)
        elif "edge" in line:
            source = int(tokens[tokens.index("source") + 1])
            target = int(tokens[tokens.index("target") + 1])
            label = tokens[tokens.index("label") + 1].strip('"')
            order = label_to_order.get(label, 0)
            self.graphs[current_section].add_edge(source, target, order=float(order))

    def _synchronize_nodes_and_edges(self):
        """
        Ensures that all nodes and edges present in the 'context' graph are also
        present in the 'left' and 'right' graphs, maintaining consistency across graphs.
        """
        context_nodes = self.graphs["context"].nodes(data=True)
        context_edges = self.graphs["context"].edges(data=True)
        for graph_key in ["left", "right"]:
            for node, data in context_nodes:
                self.graphs[graph_key].add_node(node, **data)
            for source, target, data in context_edges:
                self.graphs[graph_key].add_edge(source, target, **data)

    def _extract_element_and_charge(self, label: str) -> Tuple[str, int]:
        """
        Extracts the chemical element and its charge from a node label using regex.

        Parameters:
        - label (str): The node label in formats like "Element", "Element+", "Element-",
        "Element2+", etc.

        Returns:
        - Tuple[str, int]: A tuple containing the chemical element (str) and
        its charge (int).
        """
        match = re.match(r"([A-Za-z*]+)(\d+)?([+-])?$", label)
        if not match:
            return ("X", 0)  # Safe fallback for unrecognized patterns

        element = match.group(1)
        charge_number = match.group(2)
        charge_sign = match.group(3)

        if charge_number and charge_sign:
            charge = int(charge_number) * (1 if charge_sign == "+" else -1)
        elif charge_sign:
            charge = 1 if charge_sign == "+" else -1
        else:
            charge = 0

        return element, charge

    def transform(self) -> Tuple[nx.Graph, nx.Graph, nx.Graph]:
        """
        Transforms the GML-like text into three NetworkX graphs representing different
        aspects of a chemical reaction: 'left' for reactants, 'context' for intermediates,
        and 'right' for products.

        Returns:
        - Tuple[nx.Graph, nx.Graph, nx.Graph]: A tuple containing the graphs for the
        reactants, products, and its graph, respectively.
        """
        current_section = None
        lines = self.gml_text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("rule") or line == "]":
                continue
            if any(section in line for section in ["left", "context", "right"]):
                current_section = line.split("[")[0].strip()
                continue
            if line.startswith("node") or line.startswith("edge"):
                self._parse_element(line, current_section)

        self._synchronize_nodes_and_edges()

        self.graphs["context"] = ITSConstruction.ITSGraph(
            self.graphs["left"], self.graphs["right"]
        )

        return (self.graphs["left"], self.graphs["right"], self.graphs["context"])
