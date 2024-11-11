import os
from synutility.SynIO.debug import setup_logging
from mod import smiles, ruleGMLString, DG, config

logger = setup_logging()


def deduplicateGraphs(initial):
    """
    Removes duplicate graphs from a list based on graph isomorphism.

    Parameters:
    - initial (list): List of graph objects.

    Returns:
    - List of unique graph objects.
    """
    unique_graphs = []
    for candidate in initial:
        # Check if candidate is isomorphic to any graph already in unique_graphs
        if not any(candidate.isomorphism(existing) != 0 for existing in unique_graphs):
            unique_graphs.append(candidate)
    return unique_graphs


def rule_apply(smiles_list, rule, print_output=False):
    """
    Applies a reaction rule to a list of SMILES and optionally prints the output.

    Parameters:
    - smiles_list (list): List of SMILES strings.
    - rule (str): Reaction rule in GML string format.
    - print_output (bool): If True, output will be printed to a directory.

    Returns:
    - dg (DG): The derivation graph after applying the rule.
    """
    try:
        initial_molecules = [smiles(smile, add=False) for smile in smiles_list]
        initial_molecules = deduplicateGraphs(initial_molecules)
        initial_molecules = sorted(
            initial_molecules, key=lambda molecule: molecule.numVertices, reverse=False
        )

        reaction_rule = ruleGMLString(rule)

        dg = DG(graphDatabase=initial_molecules)
        config.dg.doRuleIsomorphismDuringBinding = False
        dg.build().apply(initial_molecules, reaction_rule, verbosity=8)

        # Optionally print the output
        if print_output:
            os.makedirs("out", exist_ok=True)
            dg.print()

        return dg
    except Exception as e:
        logger.error(f"An error occurred: {e}")
