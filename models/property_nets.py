"""Property prediction networks for guided generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from .layers import GraphEncoder, Set2Set, MLPReadout, GraphAttentionLayer


class PropertyPredictor(nn.Module):
    """Graph neural network for molecular property prediction."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim=256, output_dim=1,
                 num_layers=4, conv_type='gat', pooling='set2set', 
                 dropout=0.2, residual=True):
        """
        Initialize property predictor.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension
            output_dim: Number of properties to predict
            num_layers: Number of GNN layers
            conv_type: Type of graph convolution ('gcn' or 'gat')
            pooling: Pooling method
            dropout: Dropout rate
            residual: Use residual connections
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.residual = residual
        
        # Graph encoder
        self.encoder = GraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            conv_type=conv_type,
            dropout=dropout
        )
        
        # Pooling
        if pooling == 'set2set':
            self.pool = Set2Set(hidden_dim, processing_steps=4)
            pool_dim = hidden_dim * 2
        else:
            self.pool = pooling
            pool_dim = hidden_dim
        
        # Property prediction head
        self.predictor = MLPReadout(
            pool_dim, hidden_dim, output_dim, 
            num_layers=3, dropout=dropout
        )
        
        # Optional: normalize outputs for stable training
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            batch: Batch assignment
        
        Returns:
            Property predictions
        """
        # Encode graph
        h = self.encoder(x, edge_index, edge_attr, batch)
        
        # Pool to graph level
        if isinstance(self.pool, Set2Set):
            h_graph = self.pool(h, batch)
        elif self.pool == 'mean':
            h_graph = global_mean_pool(h, batch)
        elif self.pool == 'max':
            h_graph = global_max_pool(h, batch)
        elif self.pool == 'sum':
            h_graph = global_add_pool(h, batch)
        
        # Predict properties
        out = self.predictor(h_graph)
        out = self.output_norm(out)
        
        return out


class LatentPropertyPredictor(nn.Module):
    """Predict properties directly from latent space."""
    
    def __init__(self, latent_dim, hidden_dim=256, output_dim=3, 
                 num_layers=4, dropout=0.2):
        """
        Initialize latent property predictor.
        
        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension
            output_dim: Number of properties
            num_layers: Number of MLP layers
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        in_dim = latent_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(nn.Linear(in_dim, output_dim))
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, z):
        """Predict properties from latent vectors."""
        return self.mlp(z)


class MultiTaskPropertyPredictor(PropertyPredictor):
    """Multi-task property predictor with task-specific heads."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim=256, 
                 property_dims=None, num_layers=4, **kwargs):
        """
        Initialize multi-task predictor.
        
        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension
            property_dims: Dictionary of property names to output dimensions
            num_layers: Number of GNN layers
            **kwargs: Additional arguments for base class
        """
        # Initialize without output head
        super().__init__(
            node_dim, edge_dim, hidden_dim, output_dim=1,
            num_layers=num_layers, **kwargs
        )
        
        # Remove single predictor
        del self.predictor
        del self.output_norm
        
        # Create task-specific heads
        self.property_dims = property_dims or {'qed': 1, 'logp': 1, 'sa': 1}
        pool_dim = hidden_dim * 2 if isinstance(self.pool, Set2Set) else hidden_dim
        
        self.task_heads = nn.ModuleDict({
            name: MLPReadout(pool_dim, hidden_dim, dim, num_layers=2)
            for name, dim in self.property_dims.items()
        })
    
    def forward(self, x, edge_index, edge_attr, batch, tasks=None):
        """
        Forward pass with optional task selection.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            batch: Batch assignment
            tasks: List of tasks to predict (None = all tasks)
        
        Returns:
            Dictionary of property predictions
        """
        # Encode graph
        h = self.encoder(x, edge_index, edge_attr, batch)
        
        # Pool to graph level
        if isinstance(self.pool, Set2Set):
            h_graph = self.pool(h, batch)
        elif self.pool == 'mean':
            h_graph = global_mean_pool(h, batch)
        elif self.pool == 'max':
            h_graph = global_max_pool(h, batch)
        elif self.pool == 'sum':
            h_graph = global_add_pool(h, batch)
        
        # Predict properties
        if tasks is None:
            tasks = list(self.property_dims.keys())
        
        predictions = {}
        for task in tasks:
            if task in self.task_heads:
                predictions[task] = self.task_heads[task](h_graph)
        
        return predictions


class EnsemblePropertyPredictor(nn.Module):
    """Ensemble of property predictors for uncertainty estimation."""
    
    def __init__(self, num_models=5, **predictor_kwargs):
        """
        Initialize ensemble.
        
        Args:
            num_models: Number of models in ensemble
            **predictor_kwargs: Arguments for individual predictors
        """
        super().__init__()
        self.num_models = num_models
        
        self.models = nn.ModuleList([
            PropertyPredictor(**predictor_kwargs)
            for _ in range(num_models)
        ])
    
    def forward(self, x, edge_index, edge_attr, batch, return_std=False):
        """
        Forward pass through ensemble.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            batch: Batch assignment
            return_std: Return standard deviation
        
        Returns:
            Mean predictions and optionally std
        """
        predictions = []
        
        for model in self.models:
            pred = model(x, edge_index, edge_attr, batch)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        
        if return_std:
            std_pred = predictions.std(dim=0)
            return mean_pred, std_pred
        
        return mean_pred


def create_property_predictor(predictor_type='single', **kwargs):
    """Factory function to create property predictors."""
    if predictor_type == 'single':
        return PropertyPredictor(**kwargs)
    elif predictor_type == 'latent':
        return LatentPropertyPredictor(**kwargs)
    elif predictor_type == 'multi':
        return MultiTaskPropertyPredictor(**kwargs)
    elif predictor_type == 'ensemble':
        return EnsemblePropertyPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")