import re
from typing import Optional, Any
from rdkit import Chem
from rdkit.Chem import rdmolops
from synutility.misc import remove_explicit_hydrogen
from synutility.SynAAM.misc import get_rc
from synutility.SynIO.Format.gml_to_nx import GMLToNX
from synutility.SynIO.Format.graph_to_mol import GraphToMol
from synutility.SynIO.Format.chemical_conversion import graph_to_rsmi


def parse_smiles_and_check_valence(smiles: str) -> Optional[int]:
    """
    Parses a SMILES string to check for valence errors and identifies
    the problematic atom.

    Parameters:
    - smiles (str): The SMILES string of the molecule to be analyzed.

    Returns:
    - Optional[int]: Returns the 0-based index of the atom causing a valence issue,
    or the atom map number if available. Returns None if there is no valence issue
    or the molecule cannot be parsed.

    Raises:
    - Exception: Propagates exceptions that indicate types of chemical issues other
    than valence errors.
    """
    # First, parse without sanitization
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None

    # Attempt to sanitize the molecule
    try:
        rdmolops.SanitizeMol(mol)
    except Exception as e:
        msg = str(e)
        # Check if the exception message contains a valence error
        match = re.search(r"Explicit valence for atom # (\d+)", msg)
        if match:
            atom_idx = int(match.group(1))
            atom = mol.GetAtomWithIdx(atom_idx)
            return atom.GetAtomMapNum() if atom.GetAtomMapNum() else atom_idx
        else:
            raise

    return None


def resolve_bug_phospho(gml: Any) -> Optional[str]:
    """
    Attempts to resolve issues in phosphorous-containing molecules represented
    in GML format.

    Parameters:
    - gml (Any): The GML representation of the molecule.

    Returns:
    - Optional[str]: Returns the corrected SMILES string after modifying phosphorous
    atom properties, or None if no solution could be found.
    """
    r, p, its = GMLToNX(gml).transform()
    rc = get_rc(its)
    r, p = remove_explicit_hydrogen(r, rc.nodes()), remove_explicit_hydrogen(
        p, rc.nodes()
    )
    p_smi = Chem.MolToSmiles(
        GraphToMol().graph_to_mol(p, use_h_count=True, sanitize=False)
    )

    try:
        atom_issue_p = parse_smiles_and_check_valence(p_smi)
        if atom_issue_p is not None:
            for key, node in r.nodes(data=True):
                if node["element"] == "P":
                    node["charge"] += 1
                if key == atom_issue_p:
                    node["hcount"] -= 1
                    node["charge"] -= 1
            for key, node in p.nodes(data=True):
                if key == atom_issue_p:
                    node["hcount"] -= 1
            smart = graph_to_rsmi(r, p, its, True, False, True)
    except Exception:
        smart = None

    return smart
