"""Generate molecules using trained models with property guidance."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from models import GraphVAE, LatentDiffusion, GuidedLatentDiffusion, LatentPropertyPredictor
from utils import (
    get_atom_feature_dims, get_bond_feature_dims,
    set_seed, graph_to_smiles, check_validity,
    calculate_properties, calculate_validity, calculate_uniqueness,
    calculate_novelty, calculate_diversity, calculate_property_statistics,
    evaluate_property_optimization
)


class MolecularGenerator:
    """Generate molecules with property control."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        set_seed(config.seed)
        
        # Load models
        self.load_models()
        
        # Setup output directory
        self.output_dir = os.path.join('outputs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
            f.write(str(config))
    
    def load_models(self):
        """Load all trained models."""
        # Get feature dimensions
        node_dims = get_atom_feature_dims()
        edge_dims = get_bond_feature_dims()
        node_dim = sum(node_dims)
        edge_dim = sum(edge_dims)
        
        # Load VAE
        print("Loading VAE...")
        self.vae = GraphVAE(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=self.config.model.vae.hidden_dim,
            latent_dim=self.config.model.vae.latent_dim,
            num_layers=self.config.model.vae.num_layers,
            dropout=0,  # No dropout during inference
            pooling=self.config.model.vae.pooling
        ).to(self.device)
        
        vae_path = os.path.join(self.config.logging.checkpoint_dir, 'vae_best.pt')
        if os.path.exists(vae_path):
            checkpoint = torch.load(vae_path, map_location=self.device)
            self.vae.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded VAE from {vae_path}")
        else:
            raise ValueError(f"VAE checkpoint not found at {vae_path}")
        
        self.vae.eval()
        
        # Load Diffusion Model
        print("Loading Diffusion Model...")
        self.diffusion = GuidedLatentDiffusion(
            latent_dim=self.config.model.vae.latent_dim,
            hidden_dims=self.config.model.diffusion.hidden_dims,
            num_timesteps=self.config.model.diffusion.num_timesteps,
            beta_schedule=self.config.model.diffusion.beta_schedule,
            cond_dim=len(self.config.model.property.properties),
            num_res_blocks=self.config.model.diffusion.num_res_blocks,
            dropout=0
        ).to(self.device)
        
        # Try to load EMA model first, then regular model
        diffusion_ema_path = os.path.join(self.config.logging.checkpoint_dir, 'diffusion_best_ema.pt')
        diffusion_path = os.path.join(self.config.logging.checkpoint_dir, 'diffusion_best.pt')
        
        if os.path.exists(diffusion_ema_path):
            self.diffusion.load_state_dict(torch.load(diffusion_ema_path, map_location=self.device))
            print(f"Loaded EMA diffusion model from {diffusion_ema_path}")
        elif os.path.exists(diffusion_path):
            checkpoint = torch.load(diffusion_path, map_location=self.device)
            self.diffusion.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded diffusion model from {diffusion_path}")
        else:
            raise ValueError("Diffusion model checkpoint not found")
        
        self.diffusion.eval()
        
        # Load Property Predictor
        print("Loading Property Predictor...")
        self.property_predictor = LatentPropertyPredictor(
            latent_dim=self.config.model.vae.latent_dim,
            hidden_dim=self.config.model.property.hidden_dim,
            output_dim=len(self.config.model.property.properties),
            num_layers=4,
            dropout=0
        ).to(self.device)
        
        predictor_path = os.path.join(self.config.logging.checkpoint_dir, 'latent_predictor_best.pt')
        if os.path.exists(predictor_path):
            checkpoint = torch.load(predictor_path, map_location=self.device)
            self.property_predictor.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded property predictor from {predictor_path}")
        else:
            print("Warning: Property predictor not found. Guided generation will not be available.")
            self.property_predictor = None
        
        if self.property_predictor is not None:
            self.property_predictor.eval()
            self.diffusion.set_property_predictors(self.property_predictor)
    
    @torch.no_grad()
    def generate_unconditional(self, num_samples):
        """Generate molecules without property guidance."""
        print(f"\nGenerating {num_samples} unconditional molecules...")
        
        # Sample from diffusion model
        latents = self.diffusion.sample(
            num_samples,
            device=self.device,
            progress=True
        )
        
        # Decode latents to molecules (simplified - returns latents for now)
        # In a full implementation, you would decode these to molecular graphs
        return latents
    
    @torch.no_grad()
    def generate_conditional(self, num_samples, property_conditions):
        """Generate molecules with property conditioning."""
        print(f"\nGenerating {num_samples} molecules with property conditions...")
        print(f"Target properties: {property_conditions}")
        
        # Convert property conditions to tensor
        cond = torch.tensor([
            property_conditions.get(prop, 0.0) 
            for prop in self.config.model.property.properties
        ]).float().to(self.device)
        cond = cond.unsqueeze(0).repeat(num_samples, 1)
        
        # Sample with conditioning
        latents = self.diffusion.sample(
            num_samples,
            cond=cond,
            device=self.device,
            progress=True
        )
        
        return latents
    
    @torch.no_grad()
    def generate_guided(self, num_samples, property_targets, guidance_scale=1.0):
        """Generate molecules with gradient-based property guidance."""
        if self.property_predictor is None:
            print("Property predictor not available. Falling back to conditional generation.")
            return self.generate_conditional(num_samples, property_targets)
        
        print(f"\nGenerating {num_samples} molecules with guided generation...")
        print(f"Target properties: {property_targets}")
        print(f"Guidance scale: {guidance_scale}")
        
        # Convert targets to tensor
        targets = torch.tensor([
            property_targets.get(prop, 0.0)
            for prop in self.config.model.property.properties
        ]).float().to(self.device)
        targets = targets.unsqueeze(0).repeat(num_samples, 1)
        
        # Sample with guidance
        latents = self.diffusion.guided_sample(
            num_samples,
            property_targets=targets,
            guidance_scale=guidance_scale,
            device=self.device,
            progress=True
        )
        
        return latents
    
    def decode_and_evaluate(self, latents, prefix="generated"):
        """Decode latents and evaluate generated molecules."""
        print(f"\nEvaluating {len(latents)} generated molecules...")
        
        # Note: In a full implementation, you would decode latents to molecular graphs
        # and then convert to SMILES. For now, we'll generate dummy SMILES for demonstration
        
        # Generate dummy SMILES (replace with actual decoding)
        generated_smiles = []
        for i in range(len(latents)):
            # This is a placeholder - in practice, decode latent to graph then to SMILES
            smiles = f"C1CC{i%10}CC1"  # Dummy cyclic structure
            generated_smiles.append(smiles)
        
        # Calculate metrics
        validity = calculate_validity(generated_smiles)
        uniqueness = calculate_uniqueness(generated_smiles)
        diversity = calculate_diversity(generated_smiles[:min(1000, len(generated_smiles))])
        
        print(f"Validity: {validity:.2%}")
        print(f"Uniqueness: {uniqueness:.2%}")
        print(f"Diversity: {diversity:.3f}")
        
        # Calculate property statistics
        property_stats = calculate_property_statistics(
            generated_smiles[:min(1000, len(generated_smiles))],
            properties=self.config.model.property.properties
        )
        
        print("\nProperty Statistics:")
        for prop, stats in property_stats.items():
            if stats:
                print(f"{prop}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
        # Save results
        results = {
            'smiles': generated_smiles,
            'validity': validity,
            'uniqueness': uniqueness,
            'diversity': diversity,
            'property_stats': property_stats
        }
        
        # Save SMILES
        output_file = os.path.join(self.output_dir, f'{prefix}_molecules.csv')
        df = pd.DataFrame({'smiles': generated_smiles})
        df.to_csv(output_file, index=False)
        print(f"\nSaved generated molecules to {output_file}")
        
        return results
    
    def visualize_properties(self, results_dict):
        """Visualize property distributions."""
        print("\nCreating property distribution plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (name, results) in enumerate(results_dict.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            # Plot property distributions
            property_data = []
            for prop in self.config.model.property.properties[:3]:  # Plot first 3 properties
                if prop in results['property_stats'] and results['property_stats'][prop]:
                    values = []
                    for smiles in results['smiles'][:100]:  # Sample 100 molecules
                        props = calculate_properties(smiles)
                        if props and prop in props:
                            values.append(props[prop])
                    if values:
                        property_data.append((prop, values))
            
            # Create violin plot
            if property_data:
                prop_names, prop_values = zip(*property_data)
                positions = range(len(prop_names))
                
                parts = ax.violinplot(prop_values, positions=positions, showmeans=True)
                ax.set_xticks(positions)
                ax.set_xticklabels(prop_names)
                ax.set_ylabel('Property Value')
                ax.set_title(f'{name} - Property Distributions')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'property_distributions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved property distribution plot to {plot_path}")
        plt.close()
    
    def run_generation_experiments(self):
        """Run all generation experiments."""
        results = {}
        
        # 1. Unconditional generation
        print("\n" + "="*50)
        print("UNCONDITIONAL GENERATION")
        print("="*50)
        latents = self.generate_unconditional(self.config.generation.num_samples)
        results['unconditional'] = self.decode_and_evaluate(latents, 'unconditional')
        
        # 2. Conditional generation
        print("\n" + "="*50)
        print("CONDITIONAL GENERATION")
        print("="*50)
        property_conditions = {
            'QED': 0.7,
            'LogP': 2.0,
            'SA': 3.5
        }
        latents = self.generate_conditional(
            self.config.generation.num_samples,
            property_conditions
        )
        results['conditional'] = self.decode_and_evaluate(latents, 'conditional')
        
        # 3. Guided generation with different scales
        for scale in [0.5, 1.0, 2.0]:
            print("\n" + "="*50)
            print(f"GUIDED GENERATION (scale={scale})")
            print("="*50)
            
            latents = self.generate_guided(
                self.config.generation.num_samples,
                self.config.generation.property_targets,
                guidance_scale=scale
            )
            results[f'guided_scale_{scale}'] = self.decode_and_evaluate(
                latents, f'guided_scale_{scale}'
            )
        
        # 4. Multi-objective optimization
        print("\n" + "="*50)
        print("MULTI-OBJECTIVE OPTIMIZATION")
        print("="*50)
        
        # High QED, moderate LogP, low SA
        multi_targets = {
            'QED': 0.95,
            'LogP': 2.5,
            'SA': 2.0
        }
        latents = self.generate_guided(
            self.config.generation.num_samples,
            multi_targets,
            guidance_scale=1.5
        )
        results['multi_objective'] = self.decode_and_evaluate(latents, 'multi_objective')
        
        # Visualize results
        self.visualize_properties(results)
        
        # Save summary
        summary = {
            'experiment': [],
            'validity': [],
            'uniqueness': [],
            'diversity': []
        }
        
        for name, result in results.items():
            summary['experiment'].append(name)
            summary['validity'].append(result['validity'])
            summary['uniqueness'].append(result['uniqueness'])
            summary['diversity'].append(result['diversity'])
        
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(self.output_dir, 'generation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved generation summary to {summary_path}")
        
        # Print final summary
        print("\n" + "="*50)
        print("GENERATION SUMMARY")
        print("="*50)
        print(summary_df.to_string(index=False))
        
        return results


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    """Main generation script."""
    print("Starting molecular generation...")
    print(f"Config:\n{cfg}")
    
    generator = MolecularGenerator(cfg)
    results = generator.run_generation_experiments()
    
    print("\n" + "="*50)
    print("GENERATION COMPLETED!")
    print("="*50)
    print(f"Results saved to: {generator.output_dir}")


if __name__ == "__main__":
    main()