import networkx as nx
import re
from typing import Tuple
from synutility.SynIO.Format.its_construction import ITSConstruction


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
        Private method to parse a line from the GML text, extracting nodes and edges
        to populate the specified graph section.

        Parameters:
        - line (str): The line of text from the GML data.
        - current_section (str): The key of the graph section
        ('left', 'context', 'right') being populated.
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

    def _synchronize_nodes(self):
        """
        Private method to ensure that all nodes present in the 'context' graph are also
        present in the 'left' and 'right' graphs.
        """
        context_nodes = self.graphs["context"].nodes(data=True)
        for graph_key in ["left", "right"]:
            for node, data in context_nodes:
                self.graphs[graph_key].add_node(node, **data)

    def _extract_element_and_charge(self, label: str) -> Tuple[str, int]:
        """
        Extracts the chemical element and its charge from a node label. This function is
        designed to handle labels formatted in several ways, including just an element
        symbol ("C"), an element with a charge and sign ("Na+"), or an element with a
        multi-digit charge and a sign ("Mg2+"). The function uses regular expressions
        to parse the label accurately and extract the needed information.

        Parameters:
        - label (str): The label from which to extract information. Expected to be in
        one of the following formats:
        - "Element" (e.g., "C")
        - "Element[charge][sign]" (e.g., "Na+", "Mg2+")
        - "Element[charge][sign]" where charge is optional and if present, must be
        followed by a sign (e.g., "Al3+", "K+"). The charge is not assumed if no sign
        is present.

        Returns:
        - Tuple[str, int]: A tuple where the first element is the chemical element
        symbol (str) extracted from the label, and the second element is an integer
        representing the charge. The charge defaults to 0 if no charge information
        is present in the label.

        Raises:
        - ValueError: If the label does not conform to the expected formats,
        which should not happen if labels are pre-validated.

        Note:
        - The function assumes that the input label is well-formed according to the
        described patterns. Labels without any recognizable pattern will default to
        returning "X" as the element with a charge of 0, though this behavior
        is conservative and primarily for error handling.
        """
        # Regex to separate the element symbols from the optional charge and sign
        match = re.match(r"([A-Za-z]+)(\d+)?([+-])?$", label)
        if not match:
            return (
                "X",
                0,
            )  # Default case if regex fails to match, unlikely but safe to handle

        element = match.group(1)
        charge_number = match.group(2)
        charge_sign = match.group(3)

        if charge_number and charge_sign:
            # If there's a number and a sign, combine them to form the charge
            charge = int(charge_number) * (1 if charge_sign == "+" else -1)
        elif charge_sign:
            # If there is no number but there's a sign, it means the charge is 1 or -1
            charge = 1 if charge_sign == "+" else -1
        else:
            # If no charge information is provided, assume a charge of 0
            charge = 0

        return element, charge

    def transform(self) -> Tuple[nx.Graph, nx.Graph, nx.Graph]:
        """
        Transforms the GML-like text into three distinct NetworkX graphs, each
        representing different aspects of the reaction: 'left' for reactants,
        'context' for ITS, and 'right' for products.

        Returns:
        - Tuple[nx.Graph, nx.Graph, nx.Graph]: A tuple containing the graphs for
        reactants, products, and ITS.
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

        self._synchronize_nodes()
        self.graphs["context"] = ITSConstruction.ITSGraph(
            self.graphs["left"], self.graphs["right"]
        )

        return (self.graphs["left"], self.graphs["right"], self.graphs["context"])
