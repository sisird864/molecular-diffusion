"""Train Latent Diffusion Model."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import hydra
from omegaconf import DictConfig
import wandb
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from models import GraphVAE, LatentDiffusion
from utils import (
    get_dataloader, get_atom_feature_dims, get_bond_feature_dims,
    EarlyStopping, AverageMeter, set_seed,
    get_optimizer, get_scheduler, save_checkpoint, load_checkpoint,
    MetricsTracker
)


class DiffusionTrainer:
    """Trainer for Latent Diffusion Model."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        set_seed(config.seed)
        
        # Load pre-trained VAE
        self.setup_vae()
        
        # Extract latent codes
        self.setup_latent_data()
        
        # Initialize diffusion model
        self.setup_model()
        
        # Initialize training components
        self.setup_training()
        
        # Initialize logging
        if config.logging.use_wandb:
            wandb.init(
                project=config.logging.wandb_project,
                name="diffusion_training",
                config=dict(config)
            )
    
    def setup_vae(self):
        """Load pre-trained VAE."""
        # Get feature dimensions
        node_dims = get_atom_feature_dims()
        edge_dims = get_bond_feature_dims()
        node_dim = sum(node_dims)
        edge_dim = sum(edge_dims)
        
        # Create VAE
        self.vae = GraphVAE(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=self.config.model.vae.hidden_dim,
            latent_dim=self.config.model.vae.latent_dim,
            num_layers=self.config.model.vae.num_layers,
            dropout=self.config.model.vae.dropout,
            pooling=self.config.model.vae.pooling
        ).to(self.device)
        
        # Load checkpoint
        vae_path = os.path.join(
            self.config.logging.checkpoint_dir,
            'vae_best.pt'
        )
        
        if os.path.exists(vae_path):
            checkpoint = torch.load(vae_path, map_location=self.device)
            self.vae.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded VAE from {vae_path}")
        else:
            raise ValueError(f"VAE checkpoint not found at {vae_path}")
        
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def setup_latent_data(self):
        """Extract and save latent codes from training data."""
        print("Extracting latent codes...")
        
        # Setup data paths
        base_dir = os.path.dirname(self.config.data.data_path)
        train_path = os.path.join(base_dir, 'train.csv')
        val_path = os.path.join(base_dir, 'val.csv')
        
        # Check if latent codes already exist
        train_latent_path = os.path.join(base_dir, 'train_latents.pt')
        val_latent_path = os.path.join(base_dir, 'val_latents.pt')
        
        if os.path.exists(train_latent_path) and os.path.exists(val_latent_path):
            print("Loading existing latent codes...")
            train_data = torch.load(train_latent_path)
            val_data = torch.load(val_latent_path)
        else:
            # Extract latent codes
            train_data = self.extract_latents(train_path, 'train')
            val_data = self.extract_latents(val_path, 'val')
            
            # Save for future use
            torch.save(train_data, train_latent_path)
            torch.save(val_data, val_latent_path)
        
        # Create dataloaders
        self.train_latent_loader = DataLoader(
            TensorDataset(train_data['latents'], train_data['properties']),
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.val_latent_loader = DataLoader(
            TensorDataset(val_data['latents'], val_data['properties']),
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Train latents: {len(train_data['latents'])}")
        print(f"Val latents: {len(val_data['latents'])}")
    
    @torch.no_grad()
    def extract_latents(self, data_path, split):
        """Extract latent codes from molecular graphs."""
        loader = get_dataloader(
            data_path,
            batch_size=self.config.data.batch_size,
            split=split,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            max_size=self.config.data.max_molecules
        )
        
        all_latents = []
        all_properties = []
        
        for batch in tqdm(loader, desc=f'Extracting {split} latents'):
            batch = batch.to(self.device)
            
            # Encode to latent space
            mu, logvar, _ = self.vae.encode(
                batch.x, batch.edge_index,
                batch.edge_attr, batch.batch
            )
            
            # Use mean of distribution
            all_latents.append(mu.cpu())
            
            # Extract properties
            props = []
            for i in range(batch.num_graphs):
                graph_props = []
                for prop_name in self.config.model.property.properties:
                    # Get property value from batch
                    prop_val = batch[i].properties.get(prop_name, 0.0)
                    graph_props.append(prop_val)
                props.append(graph_props)
            all_properties.append(torch.tensor(props))
        
        latents = torch.cat(all_latents, dim=0)
        properties = torch.cat(all_properties, dim=0)
        
        return {'latents': latents, 'properties': properties}
    
    def setup_model(self):
        """Initialize diffusion model."""
        self.model = LatentDiffusion(
            latent_dim=self.config.model.vae.latent_dim,
            hidden_dims=self.config.model.diffusion.hidden_dims,
            num_timesteps=self.config.model.diffusion.num_timesteps,
            beta_schedule=self.config.model.diffusion.beta_schedule,
            cond_dim=len(self.config.model.property.properties),
            num_res_blocks=self.config.model.diffusion.num_res_blocks,
            dropout=self.config.model.diffusion.dropout
        ).to(self.device)
        
        print(f"Diffusion model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_training(self):
        """Setup training components."""
        # Optimizer
        self.optimizer = get_optimizer(
            self.model,
            lr=self.config.training.diffusion.lr,
            weight_decay=self.config.training.diffusion.weight_decay,
            optimizer_type='adamw'
        )
        
        # Scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=self.config.training.diffusion.scheduler,
            num_epochs=self.config.training.diffusion.epochs
        )
        
        # EMA model
        self.ema_model = None
        if self.config.training.diffusion.ema_decay > 0:
            self.ema_model = torch.optim.swa_utils.AveragedModel(
                self.model,
                avg_fn=lambda avg, model, num: self.config.training.diffusion.ema_decay * avg + 
                                               (1 - self.config.training.diffusion.ema_decay) * model
            )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=20,
            mode='min'
        )
        
        # Metrics tracker
        self.metrics = MetricsTracker(['train_loss', 'val_loss'])
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        losses = AverageMeter()
        
        pbar = tqdm(self.train_latent_loader, desc='Training')
        for latents, properties in pbar:
            latents = latents.to(self.device)
            properties = properties.to(self.device)
            
            # Calculate loss
            loss = self.model.training_loss(latents, cond=properties)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.diffusion.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.diffusion.gradient_clip
                )
            
            self.optimizer.step()
            
            # Update EMA
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
            
            # Update metrics
            losses.update(loss.item(), latents.size(0))
            pbar.set_postfix({'loss': losses.avg})
        
        return losses.avg
    
    @torch.no_grad()
    def validate(self):
        """Validate model."""
        self.model.eval()
        losses = AverageMeter()
        
        for latents, properties in tqdm(self.val_latent_loader, desc='Validation'):
            latents = latents.to(self.device)
            properties = properties.to(self.device)
            
            # Calculate loss
            loss = self.model.training_loss(latents, cond=properties)
            losses.update(loss.item(), latents.size(0))
        
        return losses.avg
    
    def train(self):
        """Main training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.diffusion.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.training.diffusion.epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            self.metrics.update(metrics)
            
            if self.config.logging.use_wandb:
                wandb.log(metrics, step=epoch)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(
                    self.config.logging.checkpoint_dir,
                    'diffusion_best.pt'
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Save both regular and EMA model
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, save_path
                )
                
                if self.ema_model is not None:
                    ema_path = save_path.replace('.pt', '_ema.pt')
                    torch.save(self.ema_model.state_dict(), ema_path)
                
                print(f"Saved best model with val loss: {val_loss:.4f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.logging.save_interval == 0:
                save_path = os.path.join(
                    self.config.logging.checkpoint_dir,
                    f'diffusion_epoch_{epoch+1}.pt'
                )
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, save_path
                )
            
            # Early stopping
            if self.early_stopping(val_loss):
                print("Early stopping triggered")
                break
        
        # Save final model
        save_path = os.path.join(
            self.config.logging.checkpoint_dir,
            'diffusion_final.pt'
        )
        save_checkpoint(
            self.model, self.optimizer, epoch, val_loss, save_path
        )
        
        if self.ema_model is not None:
            ema_path = save_path.replace('.pt', '_ema.pt')
            torch.save(self.ema_model.state_dict(), ema_path)
        
        if self.config.logging.use_wandb:
            wandb.finish()
        
        return best_val_loss


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    """Main training script."""
    print("Starting Diffusion training...")
    print(f"Config:\n{cfg}")
    
    trainer = DiffusionTrainer(cfg)
    best_loss = trainer.train()
    
    print(f"\nTraining completed! Best validation loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()