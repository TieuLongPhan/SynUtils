from typing import List
from synutility.SynIO.data_type import load_gml_as_text
from rdkit import Chem
from copy import deepcopy
import torch
from mod import *


class CoreEngine:
    """
    The MØDModeling class encapsulates functionalities for reaction modeling using the MØD
    toolkit. It provides methods for forward and backward prediction based on templates
    library.
    """

    @staticmethod
    def generate_reaction_smiles(
        temp_results: List[str], base_smiles: str, is_forward: bool = True
    ) -> List[str]:
        """
        Constructs reaction SMILES strings from intermediate results using a base SMILES
        string, indicating whether the process is a forward or backward reaction. This
        function iterates over a list of intermediate SMILES strings, combines them with
        the base SMILES, and formats them into complete reaction SMILES strings.

        Parameters:
        - temp_results (List[str]): Intermediate SMILES strings resulting from partial
        reactions or combinations.
        - base_smiles (str): The SMILES string representing the starting point of the
        reaction, either as reactants or products, depending on the reaction direction.
        - is_forward (bool, optional): Flag to determine the direction of the reaction;
        'True' for forward reactions where 'base_smiles' are reactants, and 'False' for
        backward reactions where 'base_smiles' are products. Defaults to True.

        Returns:
        - List[str]: A list of complete reaction SMILES strings, formatted according to
        the specified reaction direction.
        """
        results = []
        for comb in temp_results:
            joined_smiles = ".".join(comb)
            reaction_smiles = (
                f"{base_smiles}>>{joined_smiles}"
                if is_forward
                else f"{joined_smiles}>>{base_smiles}"
            )
            results.append(reaction_smiles)
        return results

    @staticmethod
    def perform_reaction(
        rule_file_path: str,
        initial_smiles: List[str],
        prediction_type: str = "forward",
        print_results: bool = False,
        verbosity: int = 0,
    ) -> List[str]:
        """
        Applies a specified reaction rule, loaded from a GML file, to a set of initial
        molecules represented by SMILES strings. The reaction can be simulated in forward
        or backward direction and repeated multiple times.

        Parameters:
        - rule_file_path (str): Path to the GML file containing the reaction rule.
        - initial_smiles (List[str]): Initial molecules represented as SMILES strings.
        - type (str, optional): Direction of the reaction ('forward' for forward,
        'backward' for backward). Defaults to 'forward'.
        - print_results (bool): Print results in latex or not. Defaults to False.

        Returns:
        - List[str]: SMILES strings of the resulting molecules or reactions.
        """

        # Determine the rule inversion based on reaction type
        invert_rule = prediction_type == "backward"
        # Convert SMILES strings to molecule objects, avoiding duplicate conversions
        initial_molecules = [smiles(smile, add=False) for smile in (initial_smiles)]

        def deduplicateGraphs(initial):
            res = []
            for cand in initial:
                for a in res:
                    if cand.isomorphism(a) != 0:
                        res.append(a)  # the one we had already
                        break
                else:
                    # didn't find any isomorphic, use the new one
                    res.append(cand)
            return res

        initial_molecules = deduplicateGraphs(initial_molecules)

        initial_molecules = sorted(
            initial_molecules, key=lambda molecule: molecule.numVertices, reverse=False
        )
        # Load the reaction rule from the GML file
        gml_content = load_gml_as_text(rule_file_path)
        reaction_rule = ruleGMLString(gml_content, invert=invert_rule, add=False)
        # Initialize the derivation graph and execute the strategy
        dg = DG(graphDatabase=initial_molecules)
        config.dg.doRuleIsomorphismDuringBinding = False
        dg.build().apply(initial_molecules, reaction_rule, verbosity=verbosity)
        if print_results:
            dg.print()

        temp_results = []
        for e in dg.edges:
            productSmiles = [v.graph.smiles for v in e.targets]
            temp_results.append(productSmiles)

        if len(temp_results) == 0:
            dg = DG(graphDatabase=initial_molecules)
            # dg.build().execute(strategy, verbosity=8)
            config.dg.doRuleIsomorphismDuringBinding = False
            dg.build().apply(
                initial_molecules, reaction_rule, verbosity=verbosity, onlyProper=False
            )
            temp_results, small_educt = [], []
            for edge in dg.edges:
                temp_results.append([vertex.graph.smiles for vertex in edge.targets])
                small_educt.extend([vertex.graph.smiles for vertex in edge.sources])

            small_educt_set = [
                Chem.CanonSmiles(smile) for smile in small_educt if smile is not None
            ]

            reagent = deepcopy(initial_smiles)
            for value in small_educt_set:
                if value in reagent:
                    reagent.remove(value)

            # Update solutions with reagents and normalize SMILES
            for solution in temp_results:
                solution.extend(reagent)
                for i, smile in enumerate(solution):
                    try:
                        mol = Chem.MolFromSmiles(smile)
                        if mol:  # Only convert if mol creation was successful
                            solution[i] = Chem.MolToSmiles(mol)
                    except Exception as e:
                        print(f"Error processing SMILES {smile}: {str(e)}")

        reaction_processing_map = {
            "forward": lambda smiles: CoreEngine.generate_reaction_smiles(
                temp_results, ".".join(initial_smiles), is_forward=True
            ),
            "backward": lambda smiles: CoreEngine.generate_reaction_smiles(
                temp_results, ".".join(initial_smiles), is_forward=False
            ),
        }

        # Use the reaction type to select the appropriate processing function and apply it
        if prediction_type in reaction_processing_map:
            return reaction_processing_map[prediction_type](initial_smiles)
        else:
            return ""
