import networkx as nx
from rdkit import Chem
from typing import Optional, Tuple

from synutility.SynIO.debug import setup_logging
from synutility.SynIO.Format.mol_to_graph import MolToGraph
from synutility.SynIO.Format.graph_to_mol import GraphToMol
from synutility.SynAAM.its_construction import ITSConstruction
from synutility.SynIO.Format.nx_to_gml import NXToGML
from synutility.SynIO.Format.gml_to_nx import GMLToNX
from synutility.SynAAM.misc import get_rc, its_decompose


logger = setup_logging()


def smiles_to_graph(
    smiles: str, drop_non_aam: bool, light_weight: bool, sanitize: bool
) -> Optional[nx.Graph]:
    """
    Helper function to convert SMILES string to a graph using MolToGraph class.

    Parameters:
    - smiles (str): SMILES representation of the molecule.
    - drop_non_aam (bool): Whether to drop nodes without atom mapping.
    - light_weight (bool): Whether to create a light-weight graph.
    - sanitize (bool): Whether to sanitize the molecule during conversion.

    Returns:
    - nx.Graph or None: The networkx graph representation of the molecule,
    or None if conversion fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize)
        if mol:
            return MolToGraph().mol_to_graph(mol, drop_non_aam, light_weight)
        else:
            logger.warning(f"Failed to parse SMILES: {smiles}")
    except Exception as e:
        logger.error(f"Error converting SMILES to graph: {smiles}, Error: {str(e)}")
    return None


def rsmi_to_graph(
    rsmi: str,
    drop_non_aam: bool = True,
    light_weight: bool = True,
    sanitize: bool = True,
) -> Tuple[Optional[nx.Graph], Optional[nx.Graph]]:
    """
    Converts reactant and product SMILES strings from a reaction SMILES (RSMI) format
    to graph representations.

    Parameters:
    - rsmi (str): Reaction SMILES string in "reactants>>products" format.
    - drop_non_aam (bool, optional): If True, nodes without atom mapping numbers
    will be dropped.
    - light_weight (bool, optional): If True, creates a light-weight graph.
    - sanitize (bool, optional): If True, sanitizes molecules during conversion.

    Returns:
    - Tuple[Optional[nx.Graph], Optional[nx.Graph]]: A tuple containing t
    he graph representations of the reactants and products.
    """
    try:
        reactants_smiles, products_smiles = rsmi.split(">>")
        r_graph = smiles_to_graph(
            reactants_smiles, drop_non_aam, light_weight, sanitize
        )
        p_graph = smiles_to_graph(products_smiles, drop_non_aam, light_weight, sanitize)
        return (r_graph, p_graph)
    except ValueError:
        logger.error(f"Invalid RSMI format: {rsmi}")
        return (None, None)


def graph_to_rsmi(r: nx.Graph, p: nx.Graph) -> str:
    """
    Converts graph representations of reactants and products to a reaction SMILES string.

    Parameters:
    - r (nx.Graph): Graph of the reactants.
    - p (nx.Graph): Graph of the products.

    Returns:
    - str: Reaction SMILES string.
    """
    r = GraphToMol().graph_to_mol(r)
    p = GraphToMol().graph_to_mol(p)
    return f"{Chem.MolToSmiles(r)}>>{Chem.MolToSmiles(p)}"


def smart_to_gml(
    smart: str,
    core: bool = True,
    sanitize: bool = False,
    rule_name: str = "rule",
    reindex: bool = False,
) -> str:
    """
    Converts a SMARTS string to GML format, optionally focusing on the reaction core.

    Parameters:
    - smart (str): The SMARTS string representing the reaction.
    - core (bool): Whether to extract and focus on the reaction core. Defaults to True.

    Returns:
    - str: The GML representation of the reaction.
    """
    r, p = rsmi_to_graph(smart, sanitize=sanitize)
    its = ITSConstruction.ITSGraph(r, p)
    if core:
        its = get_rc(its)
        r, p = its_decompose(its)
    gml = NXToGML().transform((r, p, its), reindex=reindex, rule_name=rule_name)
    return gml


def gml_to_smart(gml: str) -> str:
    """
    Converts a GML string back to a SMARTS string by interpreting the graph structures.

    Parameters:
    - gml (str): The GML string to convert.

    Returns:
    - str: The corresponding SMARTS string.
    """
    r, p, rc = GMLToNX(gml).transform()
    return graph_to_rsmi(r, p), rc
