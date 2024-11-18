from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from typing import Any, Dict, Optional
import random
from synutility.SynIO.debug import setup_logging

logger = setup_logging()


class MolToGraph:
    """
    A class for converting molecules from SMILES strings to graph representations using
    RDKit and NetworkX. It supports creating both lightweight and detailed
    graph representations with customizable atom and bond attributes,
    allowing for exclusion of atoms without atom mapping numbers.
    """

    def __init__(self) -> None:
        """
        Initialize the MolToGraph class.
        """
        pass

    @staticmethod
    def add_partial_charges(mol: Chem.Mol) -> None:
        """
        Computes and assigns Gasteiger partial charges to each atom in the given molecule.

        Parameters:
        - mol (Chem.Mol): An RDKit molecule object.
        """
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception as e:
            logger.error(f"Error computing Gasteiger charges: {e}")

    @staticmethod
    def get_stereochemistry(atom: Chem.Atom) -> str:
        """
        Determines the stereochemistry (R/S configuration) of a given atom.

        Parameters:
        - atom (Chem.Atom): An RDKit atom object.

        Returns:
        - str: The stereochemistry ('R', 'S', or 'N' for non-chiral).
        """
        chiral_tag = atom.GetChiralTag()
        return (
            "S"
            if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW
            else "R" if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW else "N"
        )

    @staticmethod
    def get_bond_stereochemistry(bond: Chem.Bond) -> str:
        """
        Determines the stereochemistry (E/Z configuration) of a given bond.

        Parameters:
        - bond (Chem.Bond): An RDKit bond object.

        Returns:
        - str: The stereochemistry ('E', 'Z', or 'N' for non-stereospecific
        or non-double bonds).
        """
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            return "N"
        stereo = bond.GetStereo()
        if stereo == Chem.BondStereo.STEREOE:
            return "E"
        elif stereo == Chem.BondStereo.STEREOZ:
            return "Z"
        return "N"

    @staticmethod
    def has_atom_mapping(mol: Chem.Mol) -> bool:
        """
        Check if the given molecule has any atom mapping numbers.

        Parameters:
        - mol (Chem.Mol): An RDKit molecule object.

        Returns:
        - bool: True if any atom in the molecule has a mapping number, False otherwise.
        """
        return any(atom.HasProp("molAtomMapNumber") for atom in mol.GetAtoms())

    @staticmethod
    def random_atom_mapping(mol: Chem.Mol) -> Chem.Mol:
        """
        Assigns a random atom mapping number to each atom in the given molecule.

        Parameters:
        - mol (Chem.Mol): An RDKit molecule object.

        Returns:
        - Chem.Mol: The RDKit molecule object with random atom mapping numbers assigned.
        """
        atom_indices = list(range(1, mol.GetNumAtoms() + 1))
        random.shuffle(atom_indices)
        for atom, idx in zip(mol.GetAtoms(), atom_indices):
            atom.SetProp("molAtomMapNumber", str(idx))
        return mol

    @classmethod
    def mol_to_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: Optional[bool] = False,
        light_weight: Optional[bool] = False,
    ) -> nx.Graph:
        """
        Converts an RDKit molecule object to a NetworkX graph with specified atom and bond
        attributes. Optionally excludes atoms without atom mapping numbers
        if drop_non_aam is True.

        Parameters:
        - mol (Chem.Mol): An RDKit molecule object.
        - drop_non_aam (bool, optional): If True, nodes without atom mapping numbers will
          be dropped.
        - light_weight (bool, optional): If True, creates a graph with minimal attributes.

        Returns:
        - nx.Graph: A NetworkX graph representing the molecule.
        """
        if light_weight:
            return cls._create_light_weight_graph(mol, drop_non_aam)
        else:
            return cls._create_detailed_graph(mol, drop_non_aam)

    @classmethod
    def _create_light_weight_graph(cls, mol: Chem.Mol, drop_non_aam: bool) -> nx.Graph:
        graph = nx.Graph()
        for atom in mol.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if drop_non_aam and atom_map == 0:
                continue
            graph.add_node(
                atom_map,
                element=atom.GetSymbol(),
                aromatic=atom.GetIsAromatic(),
                hcount=atom.GetTotalNumHs(),
                charge=atom.GetFormalCharge(),
                neighbors=[neighbor.GetSymbol() for neighbor in atom.GetNeighbors()],
                atom_map=atom_map,
            )
            for bond in atom.GetBonds():
                neighbor = bond.GetOtherAtom(atom)
                neighbor_map = neighbor.GetAtomMapNum()
                if not drop_non_aam or neighbor_map != 0:
                    graph.add_edge(
                        atom_map, neighbor_map, order=bond.GetBondTypeAsDouble()
                    )
        return graph

    @classmethod
    def _create_detailed_graph(cls, mol: Chem.Mol, drop_non_aam: bool) -> nx.Graph:
        cls.add_partial_charges(mol)  # Compute charges if not already present
        graph = nx.Graph()
        index_to_class = {}
        if not cls.has_atom_mapping(mol):
            mol = cls.random_atom_mapping(mol)

        for atom in mol.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if drop_non_aam and atom_map == 0:
                continue
            props = cls._gather_atom_properties(atom)
            index_to_class[atom.GetIdx()] = atom_map
            graph.add_node(atom_map, **props)

        for bond in mol.GetBonds():
            begin_atom_map = index_to_class.get(bond.GetBeginAtomIdx())
            end_atom_map = index_to_class.get(bond.GetEndAtomIdx())
            if begin_atom_map and end_atom_map:
                graph.add_edge(
                    begin_atom_map, end_atom_map, **cls._gather_bond_properties(bond)
                )

        return graph

    @staticmethod
    def _gather_atom_properties(atom: Chem.Atom) -> Dict[str, Any]:
        """Collect all relevant properties from an atom to use
        as graph node attributes."""
        gasteiger_charge = (
            round(float(atom.GetProp("_GasteigerCharge")), 3)
            if atom.HasProp("_GasteigerCharge")
            else 0.0
        )
        return {
            "charge": atom.GetFormalCharge(),
            "hcount": atom.GetTotalNumHs(),
            "aromatic": atom.GetIsAromatic(),
            "element": atom.GetSymbol(),
            "atom_map": atom.GetAtomMapNum(),
            "isomer": MolToGraph.get_stereochemistry(atom),
            "partial_charge": gasteiger_charge,
            "hybridization": str(atom.GetHybridization()),
            "in_ring": atom.IsInRing(),
            "implicit_hcount": atom.GetNumImplicitHs(),
            "neighbors": sorted(
                neighbor.GetSymbol() for neighbor in atom.GetNeighbors()
            ),
        }

    @staticmethod
    def _gather_bond_properties(bond: Chem.Bond) -> Dict[str, Any]:
        """Collect all relevant properties from a bond to use as graph edge attributes."""
        return {
            "order": bond.GetBondTypeAsDouble(),
            "ez_isomer": MolToGraph.get_bond_stereochemistry(bond),
            "bond_type": str(bond.GetBondType()),
            "conjugated": bond.GetIsConjugated(),
            "in_ring": bond.IsInRing(),
        }
