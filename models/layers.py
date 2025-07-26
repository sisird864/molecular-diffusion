"""Custom layers and modules for graph neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax
import math


class GraphConvLayer(MessagePassing):
    """Graph convolution layer with edge features."""
    
    def __init__(self, in_channels, out_channels, edge_dim=0, aggr='add'):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        # Linear transformations
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim, out_channels) if edge_dim > 0 else None
        
        # Update function
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_node.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr=None):
        # Transform node features
        x = self.lin_node(x)
        
        # Propagate
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        # Incorporate edge features if available
        if edge_attr is not None and self.lin_edge is not None:
            edge_feat = self.lin_edge(edge_attr)
            return x_j + edge_feat
        return x_j
    
    def update(self, aggr_out, x):
        # Combine aggregated messages with node features
        return self.mlp(torch.cat([x, aggr_out], dim=-1))


class GraphAttentionLayer(MessagePassing):
    """Graph attention layer with multi-head attention."""
    
    def __init__(self, in_channels, out_channels, heads=4, edge_dim=0, dropout=0.0):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        self.dropout = dropout
        
        # Multi-head projections
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        
        # Edge projection if edge features are used
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels) if edge_dim > 0 else None
        
        # Output projection
        self.lin_out = nn.Linear(heads * out_channels, out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_out.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr=None):
        H, C = self.heads, self.out_channels
        
        # Multi-head projections
        key = self.lin_key(x).view(-1, H, C)
        query = self.lin_query(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)
        
        # Propagate
        out = self.propagate(edge_index, query=query, key=key, value=value, 
                           edge_attr=edge_attr)
        
        # Concatenate heads and project
        out = out.view(-1, H * C)
        out = self.lin_out(out)
        
        return out
    
    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):
        # Calculate attention scores
        score = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        
        # Include edge features if available
        if edge_attr is not None and self.lin_edge is not None:
            edge_feat = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            score = score + (query_i * edge_feat).sum(dim=-1)
        
        # Apply softmax
        alpha = softmax(score, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention to values
        return alpha.unsqueeze(-1) * value_j


class Set2Set(nn.Module):
    """Set2Set global pooling layer."""
    
    def __init__(self, in_channels, processing_steps=4):
        super().__init__()
        self.in_channels = in_channels
        self.processing_steps = processing_steps
        
        # LSTM for iterative refinement
        self.lstm = nn.LSTM(in_channels * 2, in_channels, batch_first=True)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lstm.reset_parameters()
    
    def forward(self, x, batch):
        batch_size = batch.max().item() + 1
        h = x.new_zeros((batch_size, self.in_channels))
        c = x.new_zeros((batch_size, self.in_channels))
        q = x.new_zeros((batch_size, self.in_channels))
        
        for _ in range(self.processing_steps):
            # Attention
            e = torch.sum(x * q[batch], dim=1, keepdim=True)
            a = softmax(e, batch, num_nodes=x.size(0))
            r = global_mean_pool(a * x, batch)
            
            # Update with LSTM
            q, (h, c) = self.lstm(torch.cat([q, r], dim=1).unsqueeze(1), (h.unsqueeze(0), c.unsqueeze(0)))
            q = q.squeeze(1)
            h = h.squeeze(0)
            c = c.squeeze(0)
        
        return torch.cat([q, r], dim=1)


class GraphNorm(nn.Module):
    """Graph normalization layer."""
    
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, batch):
        # Calculate mean and variance per graph
        mean = global_mean_pool(x, batch)
        mean_x2 = global_mean_pool(x ** 2, batch)
        var = mean_x2 - mean ** 2
        
        # Normalize
        mean = mean[batch]
        var = var[batch]
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.weight * x_norm + self.bias


class MLPReadout(nn.Module):
    """MLP readout layer for graph-level predictions."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class GraphEncoder(nn.Module):
    """Generic graph encoder using multiple graph convolution layers."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3, 
                 conv_type='gcn', dropout=0.0):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial projection
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            if conv_type == 'gcn':
                conv = GraphConvLayer(hidden_dim, hidden_dim, edge_dim)
            elif conv_type == 'gat':
                conv = GraphAttentionLayer(hidden_dim, hidden_dim, edge_dim=edge_dim)
            else:
                raise ValueError(f"Unknown conv type: {conv_type}")
            
            self.convs.append(conv)
            self.norms.append(GraphNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Encode nodes
        x = self.node_encoder(x)
        
        # Apply graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x, batch)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_res  # Residual connection
        
        return x