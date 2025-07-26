"""Evaluation metrics for molecular generation."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from collections import Counter
import torch
from tqdm import tqdm

from .mol_utils import check_validity, calculate_properties, standardize_smiles


def calculate_validity(smiles_list):
    """Calculate the percentage of valid molecules."""
    valid_count = sum(1 for smiles in smiles_list if check_validity(smiles))
    return valid_count / len(smiles_list) if smiles_list else 0.0


def calculate_uniqueness(smiles_list):
    """Calculate the percentage of unique molecules."""
    valid_smiles = [standardize_smiles(s) for s in smiles_list if check_validity(s)]
    valid_smiles = [s for s in valid_smiles if s is not None]
    
    if not valid_smiles:
        return 0.0
    
    return len(set(valid_smiles)) / len(valid_smiles)


def calculate_novelty(generated_smiles, training_smiles):
    """Calculate the percentage of novel molecules not in training set."""
    # Standardize all SMILES
    gen_valid = set(standardize_smiles(s) for s in generated_smiles if check_validity(s))
    gen_valid = {s for s in gen_valid if s is not None}
    
    train_set = set(standardize_smiles(s) for s in training_smiles if check_validity(s))
    train_set = {s for s in train_set if s is not None}
    
    if not gen_valid:
        return 0.0
    
    novel = gen_valid - train_set
    return len(novel) / len(gen_valid)


def calculate_property_statistics(smiles_list, properties=['QED', 'LogP', 'MW']):
    """Calculate statistics for molecular properties."""
    results = {prop: [] for prop in properties}
    
    for smiles in tqdm(smiles_list, desc="Calculating properties"):
        if not check_validity(smiles):
            continue
        
        props = calculate_properties(smiles)
        if props is None:
            continue
        
        for prop in properties:
            if prop in props:
                results[prop].append(props[prop])
    
    # Calculate statistics
    stats = {}
    for prop, values in results.items():
        if values:
            stats[prop] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        else:
            stats[prop] = None
    
    return stats


def calculate_diversity(smiles_list, radius=2):
    """Calculate Tanimoto diversity of generated molecules."""
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    
    valid_mols = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_mols.append(mol)
    
    if len(valid_mols) < 2:
        return 0.0
    
    # Calculate fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius) for mol in valid_mols]
    
    # Calculate pairwise Tanimoto similarities
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
    
    # Diversity is 1 - average similarity
    avg_similarity = np.mean(similarities) if similarities else 0.0
    return 1.0 - avg_similarity


def evaluate_property_optimization(original_smiles, optimized_smiles, target_properties):
    """
    Evaluate how well the model optimized for target properties.
    
    Args:
        original_smiles: List of original SMILES
        optimized_smiles: List of optimized SMILES
        target_properties: Dict of property names and target directions (+1 for maximize, -1 for minimize)
    
    Returns:
        Dictionary of improvement statistics
    """
    results = {}
    
    for prop_name, direction in target_properties.items():
        original_values = []
        optimized_values = []
        
        # Calculate properties for original molecules
        for smiles in original_smiles:
            props = calculate_properties(smiles)
            if props and prop_name in props:
                original_values.append(props[prop_name])
        
        # Calculate properties for optimized molecules
        for smiles in optimized_smiles:
            if not check_validity(smiles):
                continue
            props = calculate_properties(smiles)
            if props and prop_name in props:
                optimized_values.append(props[prop_name])
        
        if original_values and optimized_values:
            orig_mean = np.mean(original_values)
            opt_mean = np.mean(optimized_values)
            
            # Calculate improvement
            if direction > 0:  # Maximize
                improvement = (opt_mean - orig_mean) / abs(orig_mean) * 100
            else:  # Minimize
                improvement = (orig_mean - opt_mean) / abs(orig_mean) * 100
            
            results[prop_name] = {
                'original_mean': orig_mean,
                'optimized_mean': opt_mean,
                'improvement_percent': improvement,
                'success_rate': sum(1 for v in optimized_values if 
                                  (v > orig_mean if direction > 0 else v < orig_mean)) / len(optimized_values)
            }
    
    return results


def calculate_reconstruction_accuracy(original_graphs, reconstructed_graphs, threshold=0.85):
    """
    Calculate reconstruction accuracy for VAE.
    
    Args:
        original_graphs: List of original molecular graphs
        reconstructed_graphs: List of reconstructed molecular graphs
        threshold: Similarity threshold for considering a reconstruction successful
    
    Returns:
        Reconstruction accuracy
    """
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    
    successful = 0
    total = 0
    
    for orig, recon in zip(original_graphs, reconstructed_graphs):
        if orig.smiles and recon.smiles:
            mol1 = Chem.MolFromSmiles(orig.smiles)
            mol2 = Chem.MolFromSmiles(recon.smiles)
            
            if mol1 is not None and mol2 is not None:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                
                if similarity >= threshold:
                    successful += 1
                total += 1
    
    return successful / total if total > 0 else 0.0


class MetricsTracker:
    """Track and log metrics during training."""
    
    def __init__(self, metric_names):
        self.metric_names = metric_names
        self.history = {name: [] for name in metric_names}
        self.current_epoch = 0
    
    def update(self, metrics_dict):
        """Update metrics for current epoch."""
        for name, value in metrics_dict.items():
            if name in self.history:
                self.history[name].append(value)
    
    def get_best(self, metric_name, mode='max'):
        """Get best value for a metric."""
        if metric_name not in self.history or not self.history[metric_name]:
            return None
        
        values = self.history[metric_name]
        if mode == 'max':
            return max(values)
        else:
            return min(values)
    
    def save(self, filepath):
        """Save metrics history."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)