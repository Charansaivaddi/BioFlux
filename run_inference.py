#!/usr/bin/env python3
"""
BioFlux Inference SOP (Standard Operating Procedure)
====================================================

This script demonstrates how to run inference using trained BioFlux models.
It serves as the standard procedure for loading models and running predictions.

Usage:
    python run_inference.py [OPTIONS]

Options:
    --algorithm     Algorithm to use (ppo, maddpg, lotka_volterra, epsilon_greedy)
    --episode       Episode number to load (default: latest)
    --steps         Number of simulation steps (default: 500)
    --output        Output directory (default: ./output)
    --visualize     Generate visualizations (default: True)

Examples:
    # Run inference with PPO models (latest)
    python run_inference.py --algorithm ppo
    
    # Run inference with specific episode
    python run_inference.py --algorithm maddpg --episode 1000
    
    # Run longer simulation with visualizations
    python run_inference.py --algorithm ppo --steps 1000 --visualize
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

# Import BioFlux modules
from bioflux.core.environment import Environment
from bioflux.core.agents import Predator, Prey, Plant
from bioflux.training import (
    TrainingConfig, 
    LotkaVolterraAgent, 
    EpsilonGreedyAgent,
    PPOAgent,
    MADDPGWrapper
)
from bioflux.visualization.plots import plot_population_dynamics, plot_agent_trajectories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BioFluxInference:
    """
    Standard Operating Procedure for BioFlux Model Inference
    
    This class encapsulates all steps required to:
    1. Load trained models
    2. Initialize simulation environment
    3. Run inference with trained policies
    4. Collect and analyze results
    5. Generate visualizations and reports
    """
    
    def __init__(
        self,
        algorithm: str = 'ppo',
        models_dir: str = './models',
        output_dir: str = './output',
        device: str = 'cpu'
    ):
        """
        Initialize inference system.
        
        Args:
            algorithm: RL algorithm to use
            models_dir: Directory containing trained models
            output_dir: Directory for output files
            device: Computing device ('cpu' or 'cuda')
        """
        self.algorithm = algorithm
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.config = TrainingConfig(device=device)
        self.env = None
        self.predator_agent = None
        self.prey_agent = None
        
        logger.info(f"Initialized BioFlux Inference System")
        logger.info(f"  Algorithm: {algorithm}")
        logger.info(f"  Models: {models_dir}")
        logger.info(f"  Output: {output_dir}")
    
    def step1_load_models(self, episode: Optional[int] = None) -> None:
        """
        STEP 1: Load Trained Models
        
        Loads the trained predator and prey models from the specified algorithm.
        If episode is None, loads the latest available models.
        
        Args:
            episode: Specific episode to load, or None for latest
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Loading Trained Models")
        logger.info("=" * 60)
        
        algorithm_dir = self.models_dir / self.algorithm
        if not algorithm_dir.exists():
            raise FileNotFoundError(
                f"Algorithm directory not found: {algorithm_dir}\n"
                f"Available algorithms: {[d.name for d in self.models_dir.iterdir() if d.is_dir()]}"
            )
        
        # Find model files
        if episode is None:
            predator_models = sorted(algorithm_dir.glob("predator_episode_*.pth"))
            prey_models = sorted(algorithm_dir.glob("prey_episode_*.pth"))
            
            if not predator_models or not prey_models:
                raise FileNotFoundError(
                    f"No trained models found in {algorithm_dir}\n"
                    f"Please train models first using examples/full_training.py"
                )
            
            predator_path = predator_models[-1]
            prey_path = prey_models[-1]
            episode = int(predator_path.stem.split('_')[-1])
        else:
            predator_path = algorithm_dir / f"predator_episode_{episode}.pth"
            prey_path = algorithm_dir / f"prey_episode_{episode}.pth"
            
            if not predator_path.exists() or not prey_path.exists():
                raise FileNotFoundError(
                    f"Models for episode {episode} not found in {algorithm_dir}"
                )
        
        logger.info(f"Loading models from episode {episode}")
        logger.info(f"  Predator: {predator_path.name}")
        logger.info(f"  Prey: {prey_path.name}")
        
        # Load models based on algorithm
        state_dim = 8
        action_dim = 8
        
        if self.algorithm == 'lotka_volterra':
            self.predator_agent = LotkaVolterraAgent('predator', self.config)
            self.prey_agent = LotkaVolterraAgent('prey', self.config)
            logger.info("✓ Loaded Lotka-Volterra agents (mathematical model)")
            
        elif self.algorithm == 'epsilon_greedy':
            import torch
            self.predator_agent = EpsilonGreedyAgent('predator', self.config, state_dim, action_dim)
            self.prey_agent = EpsilonGreedyAgent('prey', self.config, state_dim, action_dim)
            
            # Load predator
            checkpoint = torch.load(predator_path, map_location=self.device)
            self.predator_agent.q_network.load_state_dict(checkpoint['q_network'])
            self.predator_agent.epsilon = 0.0  # No exploration during inference
            
            # Load prey
            checkpoint = torch.load(prey_path, map_location=self.device)
            self.prey_agent.q_network.load_state_dict(checkpoint['q_network'])
            self.prey_agent.epsilon = 0.0  # No exploration during inference
            
            logger.info("✓ Loaded Epsilon-Greedy Q-Networks")
            
        elif self.algorithm == 'ppo':
            import torch
            self.predator_agent = PPOAgent('predator', self.config, state_dim, action_dim)
            self.prey_agent = PPOAgent('prey', self.config, state_dim, action_dim)
            
            # Load predator
            checkpoint = torch.load(predator_path, map_location=self.device)
            self.predator_agent.actor.load_state_dict(checkpoint['actor'])
            self.predator_agent.critic.load_state_dict(checkpoint['critic'])
            
            # Load prey
            checkpoint = torch.load(prey_path, map_location=self.device)
            self.prey_agent.actor.load_state_dict(checkpoint['actor'])
            self.prey_agent.critic.load_state_dict(checkpoint['critic'])
            
            logger.info("✓ Loaded PPO Actor-Critic Networks")
            
        elif self.algorithm == 'maddpg':
            self.predator_agent = MADDPGWrapper('predator', self.config)
            self.prey_agent = MADDPGWrapper('prey', self.config)
            
            # Load MADDPG agents
            for i in range(self.predator_agent.num_agents):
                predator_agent_path = str(predator_path).replace('.pth', f'_agent_{i}.pth')
                prey_agent_path = str(prey_path).replace('.pth', f'_agent_{i}.pth')
                
                if Path(predator_agent_path).exists():
                    self.predator_agent.agents[i].load(predator_agent_path)
                if Path(prey_agent_path).exists():
                    self.prey_agent.agents[i].load(prey_agent_path)
            
            logger.info("✓ Loaded MADDPG Multi-Agent Networks")
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        logger.info("✓ Models loaded successfully\n")
    
    def step2_initialize_environment(
        self,
        grid_size: int = 50,
        n_predators: int = 10,
        n_prey: int = 30,
        n_plants: int = 100
    ) -> None:
        """
        STEP 2: Initialize Simulation Environment
        
        Creates a fresh environment with specified parameters.
        
        Args:
            grid_size: Size of the simulation grid
            n_predators: Number of predator agents
            n_prey: Number of prey agents
            n_plants: Number of plant resources
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Initializing Simulation Environment")
        logger.info("=" * 60)
        
        self.env = Environment(
            grid_size=grid_size,
            n_predators=n_predators,
            n_prey=n_prey,
            n_plants=n_plants
        )
        
        logger.info(f"Environment created:")
        logger.info(f"  Grid size: {grid_size}x{grid_size}")
        logger.info(f"  Predators: {n_predators}")
        logger.info(f"  Prey: {n_prey}")
        logger.info(f"  Plants: {n_plants}")
        logger.info("✓ Environment initialized\n")
    
    def step3_run_inference(self, num_steps: int = 500) -> Dict[str, Any]:
        """
        STEP 3: Run Inference Simulation
        
        Executes the simulation using trained policies to make decisions.
        
        Args:
            num_steps: Number of simulation steps to run
            
        Returns:
            Dictionary containing simulation results and statistics
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Running Inference Simulation")
        logger.info("=" * 60)
        logger.info(f"Simulating {num_steps} time steps...")
        
        # Initialize tracking
        results = {
            'predator_count': [],
            'prey_count': [],
            'plant_count': [],
            'predator_positions': [],
            'prey_positions': [],
            'step': []
        }
        
        # Run simulation
        for step in range(num_steps):
            # Get observations for all agents
            observations = {}
            
            # Predator actions
            for predator in self.env.predators:
                obs = self.env.get_observation(predator)
                observations[predator.agent_id] = obs
                action = self.predator_agent.select_action(obs)
                predator.move(action, self.env.grid_size)
            
            # Prey actions
            for prey in self.env.prey:
                obs = self.env.get_observation(prey)
                observations[prey.agent_id] = obs
                action = self.prey_agent.select_action(obs)
                prey.move(action, self.env.grid_size)
            
            # Update environment
            self.env.step()
            
            # Record statistics
            results['step'].append(step)
            results['predator_count'].append(len(self.env.predators))
            results['prey_count'].append(len(self.env.prey))
            results['plant_count'].append(len(self.env.plants))
            results['predator_positions'].append([
                (p.x, p.y) for p in self.env.predators
            ])
            results['prey_positions'].append([
                (p.x, p.y) for p in self.env.prey
            ])
            
            # Progress logging
            if (step + 1) % 100 == 0:
                logger.info(
                    f"  Step {step + 1}/{num_steps} | "
                    f"Predators: {len(self.env.predators)} | "
                    f"Prey: {len(self.env.prey)} | "
                    f"Plants: {len(self.env.plants)}"
                )
        
        # Calculate summary statistics
        results['summary'] = {
            'total_steps': num_steps,
            'avg_predators': np.mean(results['predator_count']),
            'avg_prey': np.mean(results['prey_count']),
            'avg_plants': np.mean(results['plant_count']),
            'final_predators': results['predator_count'][-1],
            'final_prey': results['prey_count'][-1],
            'final_plants': results['plant_count'][-1],
            'predator_survival_rate': results['predator_count'][-1] / results['predator_count'][0],
            'prey_survival_rate': results['prey_count'][-1] / results['prey_count'][0]
        }
        
        logger.info("✓ Simulation complete")
        logger.info("\nSummary Statistics:")
        for key, value in results['summary'].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
        logger.info("")
        
        return results
    
    def step4_save_results(self, results: Dict[str, Any]) -> str:
        """
        STEP 4: Save Results
        
        Saves simulation results to JSON file for later analysis.
        
        Args:
            results: Simulation results dictionary
            
        Returns:
            Path to saved results file
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Saving Results")
        logger.info("=" * 60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"inference_{self.algorithm}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Prepare data for JSON serialization (remove positions as they're large)
        save_data = {
            'algorithm': self.algorithm,
            'timestamp': timestamp,
            'step': results['step'],
            'predator_count': results['predator_count'],
            'prey_count': results['prey_count'],
            'plant_count': results['plant_count'],
            'summary': results['summary']
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"✓ Results saved to: {filepath}")
        logger.info("")
        
        return str(filepath)
    
    def step5_generate_visualizations(self, results: Dict[str, Any]) -> List[str]:
        """
        STEP 5: Generate Visualizations
        
        Creates plots and visualizations of the simulation results.
        
        Args:
            results: Simulation results dictionary
            
        Returns:
            List of paths to generated visualization files
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Generating Visualizations")
        logger.info("=" * 60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_files = []
        
        # 1. Population Dynamics Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(results['step'], results['predator_count'], 'r-', label='Predators', linewidth=2)
        ax.plot(results['step'], results['prey_count'], 'b-', label='Prey', linewidth=2)
        ax.plot(results['step'], results['plant_count'], 'g-', label='Plants', linewidth=2)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Population', fontsize=12)
        ax.set_title(f'Population Dynamics - {self.algorithm.upper()} Inference', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        pop_file = self.output_dir / f"population_dynamics_{self.algorithm}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(pop_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        viz_files.append(str(pop_file))
        logger.info(f"✓ Population dynamics plot: {pop_file.name}")
        
        # 2. Summary Statistics Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Inference Summary - {self.algorithm.upper()}', fontsize=16, fontweight='bold')
        
        # Subplot 1: Average populations
        categories = ['Predators', 'Prey', 'Plants']
        averages = [
            results['summary']['avg_predators'],
            results['summary']['avg_prey'],
            results['summary']['avg_plants']
        ]
        axes[0, 0].bar(categories, averages, color=['red', 'blue', 'green'], alpha=0.7)
        axes[0, 0].set_title('Average Populations')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: Survival rates
        survival_data = [
            results['summary']['predator_survival_rate'] * 100,
            results['summary']['prey_survival_rate'] * 100
        ]
        axes[0, 1].bar(['Predators', 'Prey'], survival_data, color=['red', 'blue'], alpha=0.7)
        axes[0, 1].set_title('Survival Rates')
        axes[0, 1].set_ylabel('Percentage (%)')
        axes[0, 1].set_ylim([0, 100])
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Subplot 3: Population over time (zoomed)
        mid_point = len(results['step']) // 2
        axes[1, 0].plot(results['step'][:mid_point], results['predator_count'][:mid_point], 'r-', label='Predators')
        axes[1, 0].plot(results['step'][:mid_point], results['prey_count'][:mid_point], 'b-', label='Prey')
        axes[1, 0].set_title('Early Simulation Phase')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Population')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Subplot 4: Final populations
        final_data = [
            results['summary']['final_predators'],
            results['summary']['final_prey'],
            results['summary']['final_plants']
        ]
        axes[1, 1].bar(categories, final_data, color=['red', 'blue', 'green'], alpha=0.7)
        axes[1, 1].set_title('Final Populations')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        summary_file = self.output_dir / f"summary_{self.algorithm}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        viz_files.append(str(summary_file))
        logger.info(f"✓ Summary plot: {summary_file.name}")
        
        logger.info(f"✓ Generated {len(viz_files)} visualizations")
        logger.info("")
        
        return viz_files
    
    def run_complete_inference(
        self,
        episode: Optional[int] = None,
        num_steps: int = 500,
        visualize: bool = True
    ) -> Dict[str, Any]:
        """
        Run Complete Inference Pipeline
        
        Executes all steps in sequence:
        1. Load models
        2. Initialize environment
        3. Run simulation
        4. Save results
        5. Generate visualizations
        
        Args:
            episode: Model episode to load (None for latest)
            num_steps: Number of simulation steps
            visualize: Whether to generate visualizations
            
        Returns:
            Complete results dictionary
        """
        logger.info("\n" + "=" * 60)
        logger.info("BIOFLUX INFERENCE - STANDARD OPERATING PROCEDURE")
        logger.info("=" * 60)
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        try:
            # Execute SOP steps
            self.step1_load_models(episode)
            self.step2_initialize_environment()
            results = self.step3_run_inference(num_steps)
            results_file = self.step4_save_results(results)
            
            if visualize:
                viz_files = self.step5_generate_visualizations(results)
                results['visualization_files'] = viz_files
            
            results['results_file'] = results_file
            
            logger.info("=" * 60)
            logger.info("INFERENCE COMPLETE - ALL STEPS SUCCESSFUL")
            logger.info("=" * 60)
            logger.info(f"Results saved to: {results_file}")
            if visualize and 'visualization_files' in results:
                logger.info(f"Visualizations: {len(results['visualization_files'])} files")
            logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {str(e)}")
            raise


def main():
    """Main entry point for the inference SOP script."""
    parser = argparse.ArgumentParser(
        description='BioFlux Model Inference - Standard Operating Procedure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        default='ppo',
        choices=['ppo', 'maddpg', 'lotka_volterra', 'epsilon_greedy'],
        help='RL algorithm to use (default: ppo)'
    )
    parser.add_argument(
        '--episode',
        type=int,
        default=None,
        help='Episode number to load (default: latest)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=500,
        help='Number of simulation steps (default: 500)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory (default: ./output)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Generate visualizations (default: True)'
    )
    parser.add_argument(
        '--no-visualize',
        dest='visualize',
        action='store_false',
        help='Skip visualization generation'
    )
    
    args = parser.parse_args()
    
    # Initialize and run inference
    inference = BioFluxInference(
        algorithm=args.algorithm,
        models_dir='./models',
        output_dir=args.output
    )
    
    results = inference.run_complete_inference(
        episode=args.episode,
        num_steps=args.steps,
        visualize=args.visualize
    )
    
    print("\n" + "=" * 60)
    print("INFERENCE SOP COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nResults file: {results['results_file']}")
    if 'visualization_files' in results:
        print(f"Visualizations: {len(results['visualization_files'])} files created")
    print("\nTo run again:")
    print(f"  python run_inference.py --algorithm {args.algorithm} --steps {args.steps}")
    print("")


if __name__ == '__main__':
    main()
