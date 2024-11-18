from synutility.SynIO.Format.nx_to_gml import NXToGML
from synutility.SynIO.Format.smi_to_graph import rsmi_to_graph
from synutility.SynIO.Format.its_construction import ITSConstruction

from synutility.SynAAM.misc import its_decompose, get_rc
from synutility.SynAAM.normalize_aam import NormalizeAAM

from synutility.SynChem.Reaction.standardize import Standardize

from synutility.SynGraph.Transform.rule_apply import rule_apply, getReactionSmiles


class PartialExpand:
    """
    A class for partially expanding reaction SMILES (RSMI) by applying transformation
    rules based on the reaction center (RC) graph.

    Methods:
        expand(rsmi: str) -> str:
            Expands a reaction SMILES string and returns the transformed RSMI.
    """

    def __init__(self) -> None:
        """
        Initializes the PartialExpand class.
        """
        pass

    @staticmethod
    def expand(rsmi: str) -> str:
        """
        Expands a reaction SMILES string by identifying the reaction center (RC),
        applying transformation rules, and standardizing the atom mappings.

        Parameters:
        - rsmi (str): The input reaction SMILES string.

        Returns:
        - str: The transformed reaction SMILES string.
        """
        try:
            # Convert RSMI to reactant and product graphs
            r_graph, p_graph = rsmi_to_graph(rsmi, light_weight=True, sanitize=False)

            # Construct ITS (Intermediate Transition State) graph
            its = ITSConstruction().ITSGraph(r_graph, p_graph)

            # Extract the reaction center (RC) graph
            rc = get_rc(its)

            # Decompose the RC into reactant and product graphs
            r_graph, p_graph = its_decompose(rc)

            # Transform the graph to a GML rule
            rule = NXToGML().transform((r_graph, p_graph, rc))

            # Standardize the input reaction SMILES
            original_rsmi = Standardize().fit(rsmi)

            # Extract reactants from the standardized RSMI
            reactants = original_rsmi.split(">>")[0].split(".")

            # Apply the transformation rule to the reactants
            transformed_graph = rule_apply(reactants, rule)

            # Extract the transformed reaction SMILES
            transformed_rsmi = list(getReactionSmiles(transformed_graph).values())[0][0]

            # Normalize atom mappings in the transformed RSMI
            normalized_rsmi = NormalizeAAM().fit(transformed_rsmi)

            return normalized_rsmi

        except Exception as e:
            print(f"An error occurred during RSMI expansion: {e}")
            return rsmi


if __name__ == "__main__":
    rsmi = "[CH3][CH:1]=[CH2:2].[H:3][H:4]>>[CH3][CH:1]([H:3])[CH2:2][H:4]"
    rsmi = "CC[CH2:3][Cl:1].[NH2:2][H:4]>>CC[CH2:3][NH2:2].[Cl:1][H:4]"
    print(PartialExpand.expand(rsmi))
