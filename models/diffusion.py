"""Latent Diffusion Model for molecular generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time and property conditioning."""
    
    def __init__(self, in_dim, out_dim, time_dim, cond_dim=0, dropout=0.1):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU()
        )
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU()
        )
        
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, out_dim),
                nn.SiLU()
            )
        else:
            self.cond_mlp = None
        
        self.mlp2 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x, t, cond=None):
        h = self.mlp1(x)
        h = h + self.time_mlp(t)
        
        if cond is not None and self.cond_mlp is not None:
            h = h + self.cond_mlp(cond)
        
        h = self.mlp2(h)
        return h + self.skip(x)


class DiffusionUNet(nn.Module):
    """U-Net architecture for diffusion in latent space."""
    
    def __init__(self, latent_dim, hidden_dims=[256, 512, 1024], 
                 time_dim=256, cond_dim=0, num_res_blocks=2, dropout=0.1):
        """
        Initialize Diffusion U-Net.
        
        Args:
            latent_dim: Dimension of latent space
            hidden_dims: Hidden dimensions for each level
            time_dim: Dimension of time embeddings
            cond_dim: Dimension of conditioning vector (properties)
            num_res_blocks: Number of residual blocks per level
            dropout: Dropout rate
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Initial projection
        self.init_conv = nn.Linear(latent_dim, hidden_dims[0])
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        in_dim = hidden_dims[0]
        
        for i, out_dim in enumerate(hidden_dims):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_dim, out_dim, time_dim, cond_dim, dropout))
                in_dim = out_dim
            self.encoder_blocks.append(blocks)
        
        # Middle
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_dim, cond_dim, dropout)
            for _ in range(num_res_blocks)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        hidden_dims_rev = list(reversed(hidden_dims))
        
        for i, (in_dim, out_dim) in enumerate(zip(hidden_dims_rev[:-1], hidden_dims_rev[1:])):
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                # Account for skip connections
                dim = in_dim * 2 if j == 0 else in_dim
                blocks.append(ResidualBlock(dim, out_dim, time_dim, cond_dim, dropout))
                in_dim = out_dim
            self.decoder_blocks.append(blocks)
        
        # Final projection
        self.final_conv = nn.Sequential(
            nn.Linear(hidden_dims[0] * 2, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], latent_dim)
        )
    
    def forward(self, x, t, cond=None):
        """
        Forward pass through U-Net.
        
        Args:
            x: Latent vectors (batch_size, latent_dim)
            t: Timesteps (batch_size,)
            cond: Conditioning properties (batch_size, cond_dim)
        
        Returns:
            Predicted noise (batch_size, latent_dim)
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Initial conv
        h = self.init_conv(x)
        
        # Encoder
        encoder_outs = [h]
        for blocks in self.encoder_blocks:
            for block in blocks:
                h = block(h, t_emb, cond)
            encoder_outs.append(h)
        
        # Middle
        for block in self.middle_blocks:
            h = block(h, t_emb, cond)
        
        # Decoder with skip connections
        for i, blocks in enumerate(self.decoder_blocks):
            # Skip connection from encoder
            h = torch.cat([h, encoder_outs[-(i+2)]], dim=1)
            for block in blocks:
                h = block(h, t_emb, cond)
        
        # Final conv with skip from input
        h = torch.cat([h, encoder_outs[0]], dim=1)
        h = self.final_conv(h)
        
        return h


class LatentDiffusion(nn.Module):
    """Latent Diffusion Model for molecular generation."""
    
    def __init__(self, latent_dim, hidden_dims=[256, 512, 1024], 
                 num_timesteps=1000, beta_schedule='cosine', 
                 cond_dim=0, num_res_blocks=2, dropout=0.1):
        """
        Initialize Latent Diffusion Model.
        
        Args:
            latent_dim: Dimension of latent space
            hidden_dims: Hidden dimensions for U-Net
            num_timesteps: Number of diffusion timesteps
            beta_schedule: Beta schedule type ('linear' or 'cosine')
            cond_dim: Dimension of conditioning
            num_res_blocks: Number of residual blocks
            dropout: Dropout rate
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        self.cond_dim = cond_dim
        
        # Noise prediction network
        self.model = DiffusionUNet(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            cond_dim=cond_dim,
            num_res_blocks=num_res_blocks,
            dropout=dropout
        )
        
        # Set up beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Calculate posterior variance
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod))
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine beta schedule."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=0.0001, max=0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_noise(self, x_t, t, cond=None):
        """Predict noise using the model."""
        return self.model(x_t, t, cond)
    
    def p_mean_variance(self, x_t, t, cond=None, clip_denoised=True):
        """Calculate mean and variance for p(x_{t-1} | x_t)."""
        # Predict noise
        noise_pred = self.predict_noise(x_t, t, cond)
        
        # Predict x_0
        x_0_pred = (
            self.sqrt_recip_alphas_cumprod[t][:, None] * x_t -
            self.sqrt_recipm1_alphas_cumprod[t][:, None] * noise_pred
        )
        
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, min=-1, max=1)
        
        # Calculate posterior mean
        posterior_mean = (
            self.posterior_mean_coef1[t][:, None] * x_0_pred +
            self.posterior_mean_coef2[t][:, None] * x_t
        )
        
        posterior_variance = self.posterior_variance[t][:, None]
        posterior_log_variance = self.posterior_log_variance[t][:, None]
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_sample(self, x_t, t, cond=None):
        """Sample from p(x_{t-1} | x_t)."""
        mean, _, log_variance = self.p_mean_variance(x_t, t, cond)
        noise = torch.randn_like(x_t)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float()[:, None]
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
    
    @torch.no_grad()
    def sample(self, batch_size, cond=None, device='cuda', progress=True):
        """Generate samples from the model."""
        # Start from noise
        x = torch.randn(batch_size, self.latent_dim).to(device)
        
        # Denoising loop
        timesteps = reversed(range(self.num_timesteps))
        if progress:
            timesteps = tqdm(timesteps, desc='Sampling')
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, cond)
        
        return x
    
    def training_loss(self, x_start, cond=None):
        """Calculate training loss."""
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Get noisy latent
        x_t = self.q_sample(x_start, t, noise)
        
        # Predict noise
        noise_pred = self.predict_noise(x_t, t, cond)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss


class GuidedLatentDiffusion(LatentDiffusion):
    """Latent Diffusion with gradient-based guidance."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.property_predictors = None
    
    def set_property_predictors(self, predictors):
        """Set property prediction models for guidance."""
        self.property_predictors = predictors
    
    @torch.no_grad()
    def guided_sample(self, batch_size, property_targets, guidance_scale=1.0, 
                     device='cuda', progress=True):
        """
        Sample with property guidance.
        
        Args:
            batch_size: Number of samples
            property_targets: Target property values
            guidance_scale: Strength of guidance
            device: Device to use
            progress: Show progress bar
        
        Returns:
            Generated latent vectors
        """
        if self.property_predictors is None:
            raise ValueError("Property predictors not set")
        
        # Enable gradients for guidance
        with torch.enable_grad():
            # Start from noise
            x = torch.randn(batch_size, self.latent_dim, requires_grad=True).to(device)
            
            # Denoising loop
            timesteps = reversed(range(self.num_timesteps))
            if progress:
                timesteps = tqdm(timesteps, desc='Guided Sampling')
            
            for t in timesteps:
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # Get denoised sample
                x = x.detach().requires_grad_(True)
                mean, _, log_variance = self.p_mean_variance(x, t_batch)
                
                # Calculate property gradients
                if t > 0:  # No guidance at final step
                    # Predict properties from current latent
                    pred_props = self.property_predictors(mean)
                    
                    # Calculate loss for guidance
                    property_loss = F.mse_loss(pred_props, property_targets)
                    
                    # Get gradients
                    gradients = torch.autograd.grad(property_loss, x)[0]
                    
                    # Apply guidance
                    mean = mean - guidance_scale * gradients
                
                # Sample next step
                noise = torch.randn_like(x)
                nonzero_mask = (t != 0).float()[:, None]
                x = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
        
        return x.detach()