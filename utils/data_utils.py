"""Data loading and processing utilities for molecular datasets."""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split

from .mol_utils import smiles_to_graph, calculate_properties, standardize_smiles


class MolecularDataset(Dataset):
    """PyTorch Dataset for molecular graphs."""
    
    def __init__(self, data_path, split='train', max_size=None, cache_dir='data/processed'):
        """
        Initialize molecular dataset.
        
        Args:
            data_path: Path to CSV file containing SMILES strings
            split: 'train', 'val', or 'test'
            max_size: Maximum number of molecules to load
            cache_dir: Directory to cache processed data
        """
        self.data_path = data_path
        self.split = split
        self.cache_dir = cache_dir
        self.max_size = max_size
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load and process data
        self.data = self._load_data()
        
    def _load_data(self):
        """Load and process molecular data."""
        cache_file = os.path.join(self.cache_dir, f'{self.split}_processed.pkl')
        
        # Try to load from cache
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Load raw data
        print(f"Processing {self.split} data...")
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
            smiles_list = df['smiles'].tolist()
        else:
            # Assume text file with one SMILES per line
            with open(self.data_path, 'r') as f:
                smiles_list = [line.strip() for line in f.readlines()]
        
        if self.max_size:
            smiles_list = smiles_list[:self.max_size]
        
        # Process molecules
        processed_data = []
        for smiles in tqdm(smiles_list, desc="Converting SMILES to graphs"):
            # Standardize SMILES
            std_smiles = standardize_smiles(smiles)
            if std_smiles is None:
                continue
            
            # Convert to graph
            graph = smiles_to_graph(std_smiles)
            if graph is None:
                continue
            
            # Calculate properties
            properties = calculate_properties(std_smiles)
            if properties is None:
                continue
            
            # Add properties to graph data
            graph.properties = properties
            processed_data.append(graph)
        
        print(f"Processed {len(processed_data)} valid molecules out of {len(smiles_list)}")
        
        # Cache processed data
        with open(cache_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_molecular_graphs(data_list):
    """Custom collate function for molecular graphs."""
    return Batch.from_data_list(data_list)


def get_dataloader(data_path, batch_size=32, split='train', shuffle=True, 
                   num_workers=4, max_size=None):
    """
    Create a DataLoader for molecular data.
    
    Args:
        data_path: Path to data file
        batch_size: Batch size
        split: 'train', 'val', or 'test'
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        max_size: Maximum dataset size
    
    Returns:
        DataLoader object
    """
    dataset = MolecularDataset(data_path, split=split, max_size=max_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_molecular_graphs
    )


def download_zinc_dataset(save_path='data/raw'):
    """Download ZINC dataset for training."""
    import urllib.request
    import gzip
    
    os.makedirs(save_path, exist_ok=True)
    
    # ZINC250k dataset URL
    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    
    save_file = os.path.join(save_path, 'zinc250k.csv')
    
    if not os.path.exists(save_file):
        print("Downloading ZINC250k dataset...")
        urllib.request.urlretrieve(url, save_file)
        print(f"Dataset saved to {save_file}")
    else:
        print(f"Dataset already exists at {save_file}")
    
    return save_file


def prepare_property_targets(graphs, property_names=['QED', 'LogP', 'SA']):
    """
    Extract property targets from graph data for training property predictors.
    
    Args:
        graphs: List of graph Data objects
        property_names: List of property names to extract
    
    Returns:
        Tensor of property values
    """
    targets = []
    for graph in graphs:
        props = [graph.properties[prop] for prop in property_names]
        targets.append(props)
    
    return torch.tensor(targets, dtype=torch.float)


def split_dataset(data_path, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """
    Split dataset into train/val/test sets.
    
    Args:
        data_path: Path to full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        random_state: Random seed
    
    Returns:
        Paths to train, val, test files
    """
    # Load full dataset
    df = pd.read_csv(data_path)
    
    # Split into train/val/test
    train_df, test_df = train_test_split(df, test_size=1-train_ratio, random_state=random_state)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=random_state)
    
    # Save splits
    base_dir = os.path.dirname(data_path)
    train_path = os.path.join(base_dir, 'train.csv')
    val_path = os.path.join(base_dir, 'val.csv')
    test_path = os.path.join(base_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_path, val_path, test_path