import networkx as nx
from rdkit import Chem
from typing import Optional, Tuple

from synutility.SynIO.debug import setup_logging
from synutility.SynIO.Format.mol_to_graph import MolToGraph


logger = setup_logging


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
    - nx.Graph or None: The networkx graph representation of the molecule, or None if conversion fails.
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
