"""Graph Variational Autoencoder for molecular encoding/decoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch

from .layers import GraphEncoder, Set2Set, MLPReadout
from ..utils.training import reparameterize


class GraphVAE(nn.Module):
    """Graph VAE for encoding molecular graphs to latent space."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim=256, latent_dim=128, 
                 num_layers=4, dropout=0.2, pooling='set2set'):
        """
        Initialize Graph VAE.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for GNN layers
            latent_dim: Dimension of latent space
            num_layers: Number of GNN layers
            dropout: Dropout rate
            pooling: Pooling method ('mean', 'max', 'sum', 'set2set')
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = GraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            conv_type='gcn',
            dropout=dropout
        )
        
        # Pooling layer
        if pooling == 'set2set':
            self.pool = Set2Set(hidden_dim, processing_steps=4)
            pool_dim = hidden_dim * 2
        else:
            self.pool = pooling
            pool_dim = hidden_dim
        
        # Latent projections
        self.fc_mu = nn.Linear(pool_dim, latent_dim)
        self.fc_logvar = nn.Linear(pool_dim, latent_dim)
        
        # Decoder (simplified for latent diffusion - full graph decoder not needed)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pool_dim)
        )
        
        # Output projections for reconstruction
        self.node_decoder = MLPReadout(hidden_dim, hidden_dim, node_dim, num_layers=2)
        self.edge_decoder = MLPReadout(hidden_dim * 2, hidden_dim, edge_dim, num_layers=2)
    
    def encode(self, x, edge_index, edge_attr, batch):
        """Encode graph to latent distribution parameters."""
        # Graph encoding
        h = self.encoder(x, edge_index, edge_attr, batch)
        
        # Pooling
        if isinstance(self.pool, Set2Set):
            h_graph = self.pool(h, batch)
        elif self.pool == 'mean':
            h_graph = global_mean_pool(h, batch)
        elif self.pool == 'max':
            h_graph = global_max_pool(h, batch)
        elif self.pool == 'sum':
            h_graph = global_add_pool(h, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pool}")
        
        # Get distribution parameters
        mu = self.fc_mu(h_graph)
        logvar = self.fc_logvar(h_graph)
        
        return mu, logvar, h
    
    def decode(self, z, batch_size=None):
        """Decode from latent space (simplified version)."""
        h = self.decoder(z)
        return h
    
    def forward(self, x, edge_index, edge_attr, batch):
        """Forward pass through VAE."""
        # Encode
        mu, logvar, node_embeddings = self.encode(x, edge_index, edge_attr, batch)
        
        # Reparameterization
        z = reparameterize(mu, logvar)
        
        # Decode (simplified)
        h_decoded = self.decode(z)
        
        return {
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'h_decoded': h_decoded,
            'node_embeddings': node_embeddings
        }
    
    def sample(self, num_samples, device='cuda'):
        """Sample from the latent space."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        h = self.decode(z)
        return z, h
    
    def reconstruct_nodes(self, node_embeddings):
        """Reconstruct node features from embeddings."""
        return self.node_decoder(node_embeddings)
    
    def reconstruct_edges(self, node_embeddings, edge_index):
        """Reconstruct edge features from node embeddings."""
        row, col = edge_index
        edge_embeddings = torch.cat([node_embeddings[row], node_embeddings[col]], dim=1)
        return self.edge_decoder(edge_embeddings)


class GraphVAELoss(nn.Module):
    """Loss function for Graph VAE."""
    
    def __init__(self, beta=1.0, node_recon_weight=1.0, edge_recon_weight=0.5):
        """
        Initialize VAE loss.
        
        Args:
            beta: Weight for KL divergence term (beta-VAE)
            node_recon_weight: Weight for node reconstruction loss
            edge_recon_weight: Weight for edge reconstruction loss
        """
        super().__init__()
        self.beta = beta
        self.node_recon_weight = node_recon_weight
        self.edge_recon_weight = edge_recon_weight
    
    def forward(self, outputs, batch_data):
        """
        Calculate VAE loss.
        
        Args:
            outputs: Dictionary of model outputs
            batch_data: Original batch data
        
        Returns:
            Dictionary of losses
        """
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
        ) / outputs['mu'].size(0)
        
        # Node reconstruction loss
        recon_nodes = outputs['vae'].reconstruct_nodes(outputs['node_embeddings'])
        node_recon_loss = F.mse_loss(recon_nodes, batch_data.x)
        
        # Edge reconstruction loss (optional)
        if batch_data.edge_attr is not None and batch_data.edge_attr.numel() > 0:
            recon_edges = outputs['vae'].reconstruct_edges(
                outputs['node_embeddings'], batch_data.edge_index
            )
            edge_recon_loss = F.mse_loss(recon_edges, batch_data.edge_attr)
        else:
            edge_recon_loss = torch.tensor(0.0).to(batch_data.x.device)
        
        # Total loss
        total_loss = (
            self.node_recon_weight * node_recon_loss + 
            self.edge_recon_weight * edge_recon_loss + 
            self.beta * kl_loss
        )
        
        return {
            'loss': total_loss,
            'recon_loss': node_recon_loss + edge_recon_loss,
            'kl_loss': kl_loss,
            'node_recon_loss': node_recon_loss,
            'edge_recon_loss': edge_recon_loss
        }


class ConditionalGraphVAE(GraphVAE):
    """Conditional Graph VAE that can condition on molecular properties."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim=256, latent_dim=128,
                 num_properties=3, num_layers=4, dropout=0.2, pooling='set2set'):
        """
        Initialize Conditional Graph VAE.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension
            latent_dim: Dimension of latent space
            num_properties: Number of properties to condition on
            num_layers: Number of GNN layers
            dropout: Dropout rate
            pooling: Pooling method
        """
        super().__init__(node_dim, edge_dim, hidden_dim, latent_dim, 
                        num_layers, dropout, pooling)
        
        self.num_properties = num_properties
        
        # Modify latent projections to include property conditioning
        pool_dim = hidden_dim * 2 if pooling == 'set2set' else hidden_dim
        self.fc_mu = nn.Linear(pool_dim + num_properties, latent_dim)
        self.fc_logvar = nn.Linear(pool_dim + num_properties, latent_dim)
        
        # Modify decoder to accept properties
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_properties, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pool_dim)
        )
    
    def encode(self, x, edge_index, edge_attr, batch, properties=None):
        """Encode graph with optional property conditioning."""
        # Get base encoding
        h = self.encoder(x, edge_index, edge_attr, batch)
        
        # Pooling
        if isinstance(self.pool, Set2Set):
            h_graph = self.pool(h, batch)
        elif self.pool == 'mean':
            h_graph = global_mean_pool(h, batch)
        elif self.pool == 'max':
            h_graph = global_max_pool(h, batch)
        elif self.pool == 'sum':
            h_graph = global_add_pool(h, batch)
        
        # Concatenate properties if provided
        if properties is not None:
            h_graph = torch.cat([h_graph, properties], dim=1)
        
        # Get distribution parameters
        mu = self.fc_mu(h_graph)
        logvar = self.fc_logvar(h_graph)
        
        return mu, logvar, h
    
    def decode(self, z, properties=None, batch_size=None):
        """Decode from latent space with optional property conditioning."""
        if properties is not None:
            z_cond = torch.cat([z, properties], dim=1)
        else:
            z_cond = z
        
        h = self.decoder(z_cond)
        return h