"""Train Graph VAE for molecular encoding."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import wandb
from tqdm import tqdm
import numpy as np

from models import GraphVAE
from models.graph_vae import GraphVAELoss
from utils import (
    get_dataloader, download_zinc_dataset, split_dataset,
    get_atom_feature_dims, get_bond_feature_dims,
    EarlyStopping, AverageMeter, set_seed,
    get_optimizer, get_scheduler, save_checkpoint,
    MetricsTracker, calculate_reconstruction_accuracy
)


class VAETrainer:
    """Trainer for Graph VAE."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        set_seed(config.seed)
        
        # Initialize model
        self.setup_model()
        
        # Initialize data
        self.setup_data()
        
        # Initialize training components
        self.setup_training()
        
        # Initialize logging
        if config.logging.use_wandb:
            wandb.init(
                project=config.logging.wandb_project,
                name="vae_training",
                config=dict(config)
            )
    
    def setup_model(self):
        """Initialize VAE model."""
        # Get feature dimensions
        node_dims = get_atom_feature_dims()
        edge_dims = get_bond_feature_dims()
        node_dim = sum(node_dims)
        edge_dim = sum(edge_dims)
        
        # Create model
        self.model = GraphVAE(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=self.config.model.vae.hidden_dim,
            latent_dim=self.config.model.vae.latent_dim,
            num_layers=self.config.model.vae.num_layers,
            dropout=self.config.model.vae.dropout,
            pooling=self.config.model.vae.pooling
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_data(self):
        """Setup data loaders."""
        # Download dataset if needed
        data_path = self.config.data.data_path
        if not os.path.exists(data_path):
            data_path = download_zinc_dataset()
        
        # Split dataset
        if not os.path.exists(os.path.join(os.path.dirname(data_path), 'train.csv')):
            train_path, val_path, test_path = split_dataset(
                data_path,
                train_ratio=self.config.data.train_split,
                val_ratio=self.config.data.val_split
            )
        else:
            base_dir = os.path.dirname(data_path)
            train_path = os.path.join(base_dir, 'train.csv')
            val_path = os.path.join(base_dir, 'val.csv')
            test_path = os.path.join(base_dir, 'test.csv')
        
        # Create data loaders
        self.train_loader = get_dataloader(
            train_path,
            batch_size=self.config.data.batch_size,
            split='train',
            shuffle=True,
            num_workers=self.config.data.num_workers,
            max_size=self.config.data.max_molecules
        )
        
        self.val_loader = get_dataloader(
            val_path,
            batch_size=self.config.data.batch_size,
            split='val',
            shuffle=False,
            num_workers=self.config.data.num_workers
        )
        
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
    
    def setup_training(self):
        """Setup training components."""
        # Loss function
        self.criterion = GraphVAELoss(
            beta=self.config.model.vae.beta,
            node_recon_weight=1.0,
            edge_recon_weight=0.5
        )
        
        # Optimizer
        self.optimizer = get_optimizer(
            self.model,
            lr=self.config.training.vae.lr,
            weight_decay=self.config.training.vae.weight_decay
        )
        
        # Scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=self.config.training.vae.scheduler,
            num_epochs=self.config.training.vae.epochs
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.training.vae.patience,
            mode='min'
        )
        
        # Metrics tracker
        self.metrics = MetricsTracker(['train_loss', 'val_loss', 'recon_acc'])
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        losses = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            outputs = self.model(
                batch.x, batch.edge_index, 
                batch.edge_attr, batch.batch
            )
            outputs['vae'] = self.model
            
            # Calculate loss
            loss_dict = self.criterion(outputs, batch)
            loss = loss_dict['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), batch.num_graphs)
            pbar.set_postfix({'loss': losses.avg})
        
        return losses.avg
    
    @torch.no_grad()
    def validate(self):
        """Validate model."""
        self.model.eval()
        losses = AverageMeter()
        recon_accs = []
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            batch = batch.to(self.device)
            
            # Forward pass
            outputs = self.model(
                batch.x, batch.edge_index,
                batch.edge_attr, batch.batch
            )
            outputs['vae'] = self.model
            
            # Calculate loss
            loss_dict = self.criterion(outputs, batch)
            losses.update(loss_dict['loss'].item(), batch.num_graphs)
            
            # Calculate reconstruction accuracy
            # Note: This is simplified - full implementation would decode to molecules
            recon_acc = torch.mean(
                (outputs['node_embeddings'] - batch.x).abs() < 0.1
            ).item()
            recon_accs.append(recon_acc)
        
        return losses.avg, np.mean(recon_accs)
    
    def train(self):
        """Main training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.vae.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.training.vae.epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, recon_acc = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'recon_acc': recon_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            self.metrics.update(metrics)
            
            if self.config.logging.use_wandb:
                wandb.log(metrics, step=epoch)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Recon Acc: {recon_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(
                    self.config.logging.checkpoint_dir,
                    'vae_best.pt'
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, save_path
                )
                print(f"Saved best model with val loss: {val_loss:.4f}")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print("Early stopping triggered")
                break
        
        # Save final model
        save_path = os.path.join(
            self.config.logging.checkpoint_dir,
            'vae_final.pt'
        )
        save_checkpoint(
            self.model, self.optimizer, epoch, val_loss, save_path
        )
        
        if self.config.logging.use_wandb:
            wandb.finish()
        
        return best_val_loss


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    """Main training script."""
    print("Starting VAE training...")
    print(f"Config:\n{cfg}")
    
    trainer = VAETrainer(cfg)
    best_loss = trainer.train()
    
    print(f"\nTraining completed! Best validation loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()