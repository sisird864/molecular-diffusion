"""Train property prediction networks."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from models import PropertyPredictor, LatentPropertyPredictor
from utils import (
    get_dataloader, get_atom_feature_dims, get_bond_feature_dims,
    prepare_property_targets, EarlyStopping, AverageMeter, set_seed,
    get_optimizer, get_scheduler, save_checkpoint,
    MetricsTracker
)


class PropertyPredictorTrainer:
    """Trainer for property prediction networks."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        set_seed(config.seed)
        
        # Setup data
        self.setup_data()
        
        # Initialize models
        self.setup_models()
        
        # Initialize training components
        self.setup_training()
        
        # Initialize logging
        if config.logging.use_wandb:
            wandb.init(
                project=config.logging.wandb_project,
                name="property_training",
                config=dict(config)
            )
    
    def setup_data(self):
        """Setup data loaders."""
        base_dir = os.path.dirname(self.config.data.data_path)
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
        
        self.test_loader = get_dataloader(
            test_path,
            batch_size=self.config.data.batch_size,
            split='test',
            shuffle=False,
            num_workers=self.config.data.num_workers
        )
        
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
        # Get property names
        self.property_names = self.config.model.property.properties
        self.num_properties = len(self.property_names)
    
    def setup_models(self):
        """Initialize property prediction models."""
        # Get feature dimensions
        node_dims = get_atom_feature_dims()
        edge_dims = get_bond_feature_dims()
        node_dim = sum(node_dims)
        edge_dim = sum(edge_dims)
        
        # Graph-based property predictor
        self.graph_predictor = PropertyPredictor(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=self.config.model.property.hidden_dim,
            output_dim=self.num_properties,
            num_layers=self.config.model.property.num_layers,
            conv_type=self.config.model.property.conv_type,
            pooling=self.config.model.property.pooling,
            dropout=self.config.model.property.dropout
        ).to(self.device)
        
        # Latent-based property predictor (for diffusion guidance)
        self.latent_predictor = LatentPropertyPredictor(
            latent_dim=self.config.model.vae.latent_dim,
            hidden_dim=self.config.model.property.hidden_dim,
            output_dim=self.num_properties,
            num_layers=4,
            dropout=self.config.model.property.dropout
        ).to(self.device)
        
        print(f"Graph predictor parameters: {sum(p.numel() for p in self.graph_predictor.parameters()):,}")
        print(f"Latent predictor parameters: {sum(p.numel() for p in self.latent_predictor.parameters()):,}")
    
    def setup_training(self):
        """Setup training components."""
        # Optimizers
        self.graph_optimizer = get_optimizer(
            self.graph_predictor,
            lr=self.config.training.property.lr,
            weight_decay=self.config.training.property.weight_decay
        )
        
        self.latent_optimizer = get_optimizer(
            self.latent_predictor,
            lr=self.config.training.property.lr,
            weight_decay=self.config.training.property.weight_decay
        )
        
        # Schedulers
        self.graph_scheduler = get_scheduler(
            self.graph_optimizer,
            scheduler_type=self.config.training.property.scheduler,
            num_epochs=self.config.training.property.epochs
        )
        
        self.latent_scheduler = get_scheduler(
            self.latent_optimizer,
            scheduler_type=self.config.training.property.scheduler,
            num_epochs=self.config.training.property.epochs
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.training.property.patience,
            mode='min'
        )
        
        # Metrics tracker
        self.metrics = MetricsTracker([
            'train_loss', 'val_loss', 'val_mae', 'val_r2'
        ])
        
        # Load VAE for latent training
        self.load_vae()
    
    def load_vae(self):
        """Load pre-trained VAE for extracting latents."""
        from models import GraphVAE
        
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
            print("Warning: VAE checkpoint not found. Latent predictor training will be skipped.")
            self.vae = None
        
        if self.vae is not None:
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
    
    def train_epoch(self):
        """Train for one epoch."""
        self.graph_predictor.train()
        self.latent_predictor.train()
        
        graph_losses = AverageMeter()
        latent_losses = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Prepare targets
            targets = prepare_property_targets([batch[i] for i in range(batch.num_graphs)], 
                                             self.property_names).to(self.device)
            
            # Train graph predictor
            self.graph_optimizer.zero_grad()
            graph_pred = self.graph_predictor(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            graph_loss = self.criterion(graph_pred, targets)
            graph_loss.backward()
            self.graph_optimizer.step()
            graph_losses.update(graph_loss.item(), batch.num_graphs)
            
            # Train latent predictor if VAE is available
            if self.vae is not None:
                self.latent_optimizer.zero_grad()
                
                # Get latent codes
                with torch.no_grad():
                    mu, _, _ = self.vae.encode(
                        batch.x, batch.edge_index, batch.edge_attr, batch.batch
                    )
                
                # Predict from latents
                latent_pred = self.latent_predictor(mu)
                latent_loss = self.criterion(latent_pred, targets)
                latent_loss.backward()
                self.latent_optimizer.step()
                latent_losses.update(latent_loss.item(), batch.num_graphs)
            
            pbar.set_postfix({
                'graph_loss': graph_losses.avg,
                'latent_loss': latent_losses.avg
            })
        
        return graph_losses.avg, latent_losses.avg
    
    @torch.no_grad()
    def validate(self):
        """Validate models."""
        self.graph_predictor.eval()
        self.latent_predictor.eval()
        
        all_targets = []
        all_graph_preds = []
        all_latent_preds = []
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            batch = batch.to(self.device)
            
            # Prepare targets
            targets = prepare_property_targets([batch[i] for i in range(batch.num_graphs)], 
                                             self.property_names)
            all_targets.append(targets)
            
            # Graph predictions
            graph_pred = self.graph_predictor(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            all_graph_preds.append(graph_pred.cpu())
            
            # Latent predictions
            if self.vae is not None:
                mu, _, _ = self.vae.encode(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                latent_pred = self.latent_predictor(mu)
                all_latent_preds.append(latent_pred.cpu())
        
        # Concatenate all predictions
        targets = torch.cat(all_targets, dim=0).numpy()
        graph_preds = torch.cat(all_graph_preds, dim=0).numpy()
        
        # Calculate metrics
        graph_loss = np.mean((graph_preds - targets) ** 2)
        graph_mae = mean_absolute_error(targets, graph_preds)
        graph_r2 = r2_score(targets, graph_preds)
        
        results = {
            'graph_loss': graph_loss,
            'graph_mae': graph_mae,
            'graph_r2': graph_r2
        }
        
        if all_latent_preds:
            latent_preds = torch.cat(all_latent_preds, dim=0).numpy()
            latent_loss = np.mean((latent_preds - targets) ** 2)
            latent_mae = mean_absolute_error(targets, latent_preds)
            latent_r2 = r2_score(targets, latent_preds)
            
            results.update({
                'latent_loss': latent_loss,
                'latent_mae': latent_mae,
                'latent_r2': latent_r2
            })
        
        return results
    
    def train(self):
        """Main training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.property.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.training.property.epochs}")
            
            # Train
            graph_loss, latent_loss = self.train_epoch()
            
            # Validate
            val_results = self.validate()
            
            # Update schedulers
            if isinstance(self.graph_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.graph_scheduler.step(val_results['graph_loss'])
                if self.vae is not None:
                    self.latent_scheduler.step(val_results.get('latent_loss', 0))
            else:
                self.graph_scheduler.step()
                if self.vae is not None:
                    self.latent_scheduler.step()
            
            # Log metrics
            metrics = {
                'train_graph_loss': graph_loss,
                'val_graph_loss': val_results['graph_loss'],
                'val_graph_mae': val_results['graph_mae'],
                'val_graph_r2': val_results['graph_r2'],
                'lr': self.graph_optimizer.param_groups[0]['lr']
            }
            
            if self.vae is not None:
                metrics.update({
                    'train_latent_loss': latent_loss,
                    'val_latent_loss': val_results['latent_loss'],
                    'val_latent_mae': val_results['latent_mae'],
                    'val_latent_r2': val_results['latent_r2']
                })
            
            self.metrics.update(metrics)
            
            if self.config.logging.use_wandb:
                wandb.log(metrics, step=epoch)
            
            # Print results
            print(f"Graph - Train Loss: {graph_loss:.4f}, Val Loss: {val_results['graph_loss']:.4f}")
            print(f"Graph - Val MAE: {val_results['graph_mae']:.4f}, Val R2: {val_results['graph_r2']:.4f}")
            
            if self.vae is not None:
                print(f"Latent - Train Loss: {latent_loss:.4f}, Val Loss: {val_results['latent_loss']:.4f}")
                print(f"Latent - Val MAE: {val_results['latent_mae']:.4f}, Val R2: {val_results['latent_r2']:.4f}")
            
            # Save best models
            current_val_loss = val_results['graph_loss']
            if self.vae is not None:
                current_val_loss = (val_results['graph_loss'] + val_results['latent_loss']) / 2
            
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                
                # Save graph predictor
                save_path = os.path.join(
                    self.config.logging.checkpoint_dir,
                    'graph_predictor_best.pt'
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_checkpoint(
                    self.graph_predictor, self.graph_optimizer, 
                    epoch, val_results['graph_loss'], save_path
                )
                
                # Save latent predictor
                if self.vae is not None:
                    save_path = os.path.join(
                        self.config.logging.checkpoint_dir,
                        'latent_predictor_best.pt'
                    )
                    save_checkpoint(
                        self.latent_predictor, self.latent_optimizer,
                        epoch, val_results['latent_loss'], save_path
                    )
                
                print(f"Saved best models with val loss: {best_val_loss:.4f}")
            
            # Early stopping
            if self.early_stopping(current_val_loss):
                print("Early stopping triggered")
                break
        
        # Test final models
        print("\nEvaluating on test set...")
        self.test()
        
        if self.config.logging.use_wandb:
            wandb.finish()
    
    @torch.no_grad()
    def test(self):
        """Evaluate on test set."""
        # Load best models
        graph_path = os.path.join(
            self.config.logging.checkpoint_dir,
            'graph_predictor_best.pt'
        )
        if os.path.exists(graph_path):
            checkpoint = torch.load(graph_path, map_location=self.device)
            self.graph_predictor.load_state_dict(checkpoint['model_state_dict'])
        
        if self.vae is not None:
            latent_path = os.path.join(
                self.config.logging.checkpoint_dir,
                'latent_predictor_best.pt'
            )
            if os.path.exists(latent_path):
                checkpoint = torch.load(latent_path, map_location=self.device)
                self.latent_predictor.load_state_dict(checkpoint['model_state_dict'])
        
        # Run evaluation
        test_results = self.validate()  # Uses val_loader, but same logic
        
        print("\nTest Results:")
        print(f"Graph - MAE: {test_results['graph_mae']:.4f}, R2: {test_results['graph_r2']:.4f}")
        
        if self.vae is not None and 'latent_mae' in test_results:
            print(f"Latent - MAE: {test_results['latent_mae']:.4f}, R2: {test_results['latent_r2']:.4f}")


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    """Main training script."""
    print("Starting Property Predictor training...")
    print(f"Config:\n{cfg}")
    
    trainer = PropertyPredictorTrainer(cfg)
    trainer.train()
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()