"""Molecular utility functions for graph conversion and property calculation."""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

# Atom features
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Bond features
BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ],
}


def one_hot_encoding(value, choices):
    """Create one-hot encoding for categorical features."""
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding


def get_atom_features(atom):
    """Extract atom features for GNN input."""
    features = []
    features += one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot_encoding(atom.GetDegree(), ATOM_FEATURES['degree'])
    features += one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs'])
    features += one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    features += [int(atom.GetIsAromatic())]
    return features


def get_bond_features(bond):
    """Extract bond features for GNN input."""
    features = []
    features += one_hot_encoding(bond.GetBondType(), BOND_FEATURES['bond_type'])
    features += one_hot_encoding(bond.GetStereo(), BOND_FEATURES['stereo'])
    features += [int(bond.GetIsConjugated())]
    features += [int(bond.IsInRing())]
    return features


def smiles_to_graph(smiles):
    """Convert SMILES string to PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Get edge indices and features
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Add both directions
        edge_indices.extend([[i, j], [j, i]])
        bond_feature = get_bond_features(bond)
        edge_features.extend([bond_feature, bond_feature])
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    
    return data


def graph_to_smiles(node_features, edge_index, edge_features=None):
    """Convert graph representation back to SMILES (simplified version)."""
    # This is a placeholder - in practice, you'd use more sophisticated methods
    # like junction tree VAE decoding or iterative refinement
    # For now, return None to indicate this needs proper implementation
    return None


def check_validity(smiles):
    """Check if a SMILES string represents a valid molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def standardize_smiles(smiles):
    """Standardize SMILES representation."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def calculate_properties(smiles):
    """Calculate molecular properties for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    properties = {
        'MW': Descriptors.ExactMolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'QED': QED.qed(mol),
        'SA': calculate_sa_score(mol),
        'NP': calculate_np_score(mol),
    }
    return properties


def calculate_sa_score(mol):
    """Calculate synthetic accessibility score."""
    # Simplified SA score calculation
    # In practice, use the full SA score implementation
    fp = AllChem.GetMorganFingerprint(mol, 2)
    bits = fp.GetNonzeroElements()
    score = 10.0 - (len(bits) / 100.0)  # Simplified scoring
    return max(1.0, min(10.0, score))


def calculate_np_score(mol):
    """Calculate natural product likeness score."""
    # Simplified NP score calculation
    # In practice, use the full NP score implementation
    num_rings = Lipinski.NumSaturatedRings(mol)
    num_heteroatoms = Lipinski.NumHeteroatoms(mol)
    score = (num_rings * 0.3 + num_heteroatoms * 0.1)
    return max(-5.0, min(5.0, score))


def get_atom_feature_dims():
    """Get dimensions of atom features for model initialization."""
    return [len(choices) for choices in ATOM_FEATURES.values()] + [1]  # +1 for aromaticity


def get_bond_feature_dims():
    """Get dimensions of bond features for model initialization."""
    return [len(choices) for choices in BOND_FEATURES.values()] + [2]  # +2 for conjugated and in_ring