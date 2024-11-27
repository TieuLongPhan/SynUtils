from collections import Counter
from typing import List, Dict, Tuple
from synutility.SynChem.Reaction.standardize import Standardize
from synutility.SynGraph.Transform.core_engine import CoreEngine

std = Standardize()


def remove_reagent_from_smiles(rsmi: str) -> str:
    """
    Removes common molecules from the reactants and products in a SMILES reaction string.

    This function identifies the molecules that appear on both sides of the reaction
    (reactants and products) and removes one occurrence of each common molecule from
    both sides.

    Parameters:
    - rsmi (str): A SMILES string representing a chemical reaction in the form:
    'reactant1.reactant2...>>product1.product2...'

    Returns:
    - str: A new SMILES string with the common molecules removed, in the form:
    'reactant1.reactant2...>>product1.product2...'

    Example:
    >>> remove_reagent_from_smiles('CC=O.CC=O.CCC=O>>CC=CO.CC=O.CC=O')
    'CCC=O>>CC=CO'
    """

    # Split the input SMILES string into reactants and products
    reactants, products = rsmi.split(">>")

    # Split the reactants and products by '.' to separate molecules
    reactant_molecules = reactants.split(".")
    product_molecules = products.split(".")

    # Count the occurrences of each molecule in reactants and products
    reactant_count = Counter(reactant_molecules)
    product_count = Counter(product_molecules)

    # Find common molecules between reactants and products
    common_molecules = set(reactant_count) & set(product_count)

    # Remove common molecules by the minimum occurrences in both reactants and products
    for molecule in common_molecules:
        common_occurrences = min(reactant_count[molecule], product_count[molecule])

        # Decrease the count by the common occurrences
        reactant_count[molecule] -= common_occurrences
        product_count[molecule] -= common_occurrences

    # Rebuild the lists of reactant and product molecules after removal
    filtered_reactant_molecules = [
        molecule for molecule, count in reactant_count.items() for _ in range(count)
    ]
    filtered_product_molecules = [
        molecule for molecule, count in product_count.items() for _ in range(count)
    ]

    # Join the remaining molecules back into SMILES strings
    new_reactants = ".".join(filtered_reactant_molecules)
    new_products = ".".join(filtered_product_molecules)

    # Return the updated reaction string
    return f"{new_reactants}>>{new_products}"


def perform_multi_step_reaction(
    gml_list: List[str], order: List[int], rsmi: str
) -> Tuple[List[List[str]], Dict[str, List[str]]]:
    """
    Applies a sequence of multi-step reactions to a starting SMILES string. The function
    processes each reaction step in a specified order, and returns both the intermediate
    and final products, as well as a mapping of reactant SMILES to their
    corresponding products.

    Parameters:
    - gml_list (List[str]): A list of reaction rules (in GML format) to be applied.
    Each element corresponds to a reaction step.
    - order (List[int]): A list of integers that defines the order in which the
    reaction steps should be applied. Each integer is an index referring to the position
    of a reaction rule in the `gml_list`.
    - rsmi (str): The starting reaction SMILES string, representing the reactants for the
    first reaction.

    Returns:
    - Tuple[List[List[str]], Dict[str, List[str]]]:
        - A list of lists of SMILES strings, where each inner list contains the
        RSMI generated  at each reaction step.
        - A dictionary mapping each RSMI string to the resulting products after applying
          the reaction rules. The keys are the input RSMIs, and the values are the
          resulting product  SMILES strings.
    """

    # Initialize CoreEngine for reaction processing
    core = CoreEngine()
    # Initialize a dictionary to hold reaction results
    reaction_results = {}

    # List to store the results of each reaction step
    all_steps: List[List[str]] = []
    result: List[str] = [rsmi]  # Initial result is the input SMILES string

    # Loop over the reaction steps in the specified order
    for i, j in enumerate(order):
        # Get the reaction SMILES (RSMI) for the current step
        current_step_gml = gml_list[j]
        new_result: List[str] = []  # List to hold products for this step

        # Apply the reaction for each current reactant SMILES
        for current_rsmi in result:
            smi_lst = (
                current_rsmi.split(">>")[0].split(
                    "."
                )  # Split reactants at the first step
                if i == 0
                else current_rsmi.split(">>")[1].split(
                    "."
                )  # Split products for subsequent steps
            )

            # Perform the reaction using the CoreEngine
            o = core.perform_reaction(current_step_gml, smi_lst)

            # Apply standardization on the products
            o = [std.fit(i) for i in o]

            # Collect the new results (products) from this reaction step
            new_result.extend(o)

            # Record the reaction results in the dictionary, mapping input RSMI to output products
            if len(o) > 0:
                reaction_results[current_rsmi] = o

        # Update the result list for the next step
        result = new_result

        # Append the results of this step to the overall steps list
        all_steps.append(result)

    # Return the results: a list of all steps and a dictionary of reaction results
    return all_steps, reaction_results


def calculate_max_depth(reaction_tree, current_node=None, depth=0):
    """
    Calculate the maximum depth of a reaction tree.

    Parameters:
    - reaction_tree (dict): A dictionary where keys are reaction SMILES (RSMI)
    and values are lists of product reactions.
    - current_node (str): The current node in the tree being explored (reaction SMILES).
    - depth (int): The current depth of the tree.

    Returns:
    - int: The maximum depth of the tree.
    """
    # If current_node is None, start from the root node (first key in the reaction tree)
    if current_node is None:
        current_node = list(reaction_tree.keys())[0]

    # Get the products of the current node (reaction)
    products = reaction_tree.get(current_node, [])

    # If no products, we are at a leaf node, return the current depth
    if not products:
        return depth

    # Recursively calculate the depth for each product and return the maximum
    max_subtree_depth = max(
        calculate_max_depth(reaction_tree, product, depth + 1) for product in products
    )
    return max_subtree_depth


def find_all_paths(
    reaction_tree,
    target_products,
    current_node,
    target_depth,
    current_depth=0,
    path=None,
):
    """
    Recursively find all paths from the root to the maximum depth in the reaction tree.

    Parameters:
    - reaction_tree (dict): A dictionary of reaction SMILES with products.
    - current_node (str): The current node (reaction SMILES).
    - target_depth (int): The depth at which the product matches the root's product.
    - current_depth (int): The current depth of the search.
    - path (list): The current path in the tree.

    Returns:
    - List of all paths to the max depth.
    """
    if path is None:
        path = []

    # Add the current node (reaction SMILES) to the path
    path.append(current_node)

    # If we have reached the target depth, check the product
    if current_depth == target_depth:
        # Extract products of the current node
        products = sorted(current_node.split(">>")[1].split("."))
        return [path] if products == target_products else []

    # If we haven't reached the target depth, recurse on the products
    paths = []
    for product in reaction_tree.get(current_node, []):
        paths.extend(
            find_all_paths(
                reaction_tree,
                target_products,
                product,
                target_depth,
                current_depth + 1,
                path.copy(),
            )
        )

    return paths
