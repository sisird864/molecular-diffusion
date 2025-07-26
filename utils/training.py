"""Training utilities and helper functions."""

import torch
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, Optional


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
        elif self._is_improvement(value):
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, value):
        if self.mode == 'min':
            return value < self.best_value - self.min_delta
        else:
            return value > self.best_value + self.min_delta


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model, lr=1e-3, weight_decay=0, optimizer_type='adam'):
    """Get optimizer for model."""
    if optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_scheduler(optimizer, scheduler_type='cosine', num_epochs=100, warmup_epochs=5):
    """Get learning rate scheduler."""
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    elif scheduler_type == 'linear':
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 1.0 - (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return None


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath, device='cuda'):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def count_parameters(model):
    """Count number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class WandbLogger:
    """Weights & Biases logger wrapper."""
    
    def __init__(self, project_name, config, enabled=True):
        self.enabled = enabled
        if self.enabled:
            wandb.init(project=project_name, config=config)
    
    def log(self, metrics: Dict, step: Optional[int] = None):
        if self.enabled:
            wandb.log(metrics, step=step)
    
    def finish(self):
        if self.enabled:
            wandb.finish()


def gradient_penalty(real_data, fake_data, discriminator, device='cuda'):
    """Calculate gradient penalty for WGAN-GP."""
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1).to(device)
    alpha = alpha.expand_as(real_data)
    
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    prob_interpolated = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


def kl_divergence(mu, logvar):
    """Calculate KL divergence for VAE."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def reparameterize(mu, logvar):
    """Reparameterization trick for VAE."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class GradientAccumulator:
    """Handle gradient accumulation for large batch training."""
    
    def __init__(self, accumulation_steps=1):
        self.accumulation_steps = accumulation_steps
        self.steps = 0
    
    def should_step(self):
        self.steps += 1
        return self.steps % self.accumulation_steps == 0
    
    def reset(self):
        self.steps = 0


def clip_gradients(model, max_norm=1.0):
    """Clip gradients to prevent explosion."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def warmup_lr(optimizer, current_step, warmup_steps, initial_lr, target_lr):
    """Linear warmup of learning rate."""
    if current_step < warmup_steps:
        lr = initial_lr + (target_lr - initial_lr) * current_step / warmup_steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr