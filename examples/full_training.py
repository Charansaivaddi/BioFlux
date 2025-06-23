#!/usr/bin/env python3
"""
BioFlux RL Training Loop - Full Training Implementation

This script implements a comprehensive training loop for all RL algorithms
and prepares models for inference use.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Any, Tuple

# Add the parent directory to the path to import bioflux
sys.path.append(str(Path(__file__).parent.parent))

from bioflux.training import (
    TrainingConfig, LotkaVolterraAgent, EpsilonGreedyAgent, 
    PPOAgent, MADDPGWrapper, create_training_environment, ReplayBuffer
)
from bioflux.core.environment import Environment

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FullTrainingLoop:
    """Complete training loop implementation for all RL algorithms."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Training metrics
        self.training_metrics = {
            'lotka_volterra': {'rewards': [], 'losses': [], 'episode_lengths': []},
            'epsilon_greedy': {'rewards': [], 'losses': [], 'episode_lengths': []},
            'ppo': {'rewards': [], 'losses': [], 'episode_lengths': []},
            'maddpg': {'rewards': [], 'losses': [], 'episode_lengths': []}
        }
        
        # Model save directory
        self.save_dir = Path("models")
        self.save_dir.mkdir(exist_ok=True)
        
        logger.info(f"Training configuration: {config.num_episodes} episodes, device: {config.device}")
    
    def create_agents(self, algorithm: str) -> Dict[str, Any]:
        """Create agents for specific algorithm."""
        state_dim = 8
        action_dim = 8
        
        if algorithm == 'lotka_volterra':
            return {
                'predator': LotkaVolterraAgent('predator', self.config),
                'prey': LotkaVolterraAgent('prey', self.config)
            }
        elif algorithm == 'epsilon_greedy':
            return {
                'predator': EpsilonGreedyAgent('predator', self.config, state_dim, action_dim),
                'prey': EpsilonGreedyAgent('prey', self.config, state_dim, action_dim)
            }
        elif algorithm == 'ppo':
            return {
                'predator': PPOAgent('predator', self.config, state_dim, action_dim),
                'prey': PPOAgent('prey', self.config, state_dim, action_dim)
            }
        elif algorithm == 'maddpg':
            return {
                'predator': MADDPGWrapper('predator', self.config),
                'prey': MADDPGWrapper('prey', self.config)
            }
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def get_agent_lists(self, env: Environment) -> Tuple[List, List]:
        """Get lists of predators and prey from environment."""
        predators = [a for a in env.predators if a.is_alive]
        prey = [a for a in env.prey if a.is_alive]
        return predators, prey
    
    def get_agent_state(self, agent, env: Environment) -> Dict[str, Any]:
        """Get standardized state observation for an agent."""
        # Get nearby agents
        nearby_prey = []
        nearby_predators = []
        nearby_food = []
        
        all_agents = env.predators + env.prey + env.plants
        
        for other in all_agents:
            if other != agent and hasattr(other, 'pos_x') and hasattr(other, 'pos_y'):
                distance = np.sqrt((other.pos_x - agent.pos_x)**2 + (other.pos_y - agent.pos_y)**2)
                if distance < 15:  # Detection range
                    if hasattr(other, 'agent_type'):
                        if other.agent_type == 'prey':
                            nearby_prey.append({'pos': (other.pos_x, other.pos_y), 'distance': distance})
                        elif other.agent_type == 'predator':
                            nearby_predators.append({'pos': (other.pos_x, other.pos_y), 'distance': distance})
                        elif other.agent_type == 'plant':
                            nearby_food.append({'pos': (other.pos_x, other.pos_y), 'distance': distance})
        
        # Get environmental conditions
        temp = env.temperature_map[min(agent.pos_y, env.height-1), min(agent.pos_x, env.width-1)]
        vegetation = env.vegetation_map[min(agent.pos_y, env.height-1), min(agent.pos_x, env.width-1)]
        
        return {
            'energy': agent.energy,
            'age': getattr(agent, 'age', 1),
            'nearby_prey': nearby_prey,
            'nearby_predators': nearby_predators,
            'nearby_food': nearby_food,
            'temperature': temp,
            'vegetation': vegetation,
            'is_hungry': agent.energy < 30,
            'pos_x': agent.pos_x,
            'pos_y': agent.pos_y,
            'environment_bounds': {'width': env.width, 'height': env.height}
        }
    
    def calculate_reward(self, agent, action: Dict[str, Any], env: Environment) -> float:
        """Calculate reward for an agent's action."""
        reward = 0.0
        action_type = action.get('action', 'stay')
        
        if agent.agent_type == 'predator':
            if action_type == 'hunt':
                # Check for successful hunt
                for prey_agent in env.prey:
                    if prey_agent.is_alive:
                        distance = np.sqrt((prey_agent.pos_x - agent.pos_x)**2 + 
                                         (prey_agent.pos_y - agent.pos_y)**2)
                        if distance < 3:
                            reward += 20.0  # Successful hunt
                            prey_agent.energy -= 30  # Damage prey
                            agent.energy += 15  # Gain energy
                            break
                else:
                    reward -= 2.0  # Failed hunt
            
            elif action_type == 'move' or action_type == 'explore':
                # Small penalty for moving (energy cost)
                reward -= 0.5
                agent.energy -= 1
            
            # Survival bonus
            if agent.energy > 0:
                reward += 1.0
        
        elif agent.agent_type == 'prey':
            if action_type == 'flee':
                # Check if successfully avoided predator
                nearby_predators = 0
                for pred_agent in env.predators:
                    if pred_agent.is_alive:
                        distance = np.sqrt((pred_agent.pos_x - agent.pos_x)**2 + 
                                         (pred_agent.pos_y - agent.pos_y)**2)
                        if distance < 8:
                            nearby_predators += 1
                
                if nearby_predators > 0:
                    reward += 5.0  # Successful escape
                else:
                    reward -= 1.0  # Unnecessary flee
            
            elif action_type == 'forage':
                # Check for food
                for plant in env.plants:
                    distance = np.sqrt((plant.pos_x - agent.pos_x)**2 + 
                                     (plant.pos_y - agent.pos_y)**2)
                    if distance < 2:
                        reward += 8.0  # Successful foraging
                        agent.energy += 10
                        plant.energy -= 5
                        break
                else:
                    reward -= 1.0  # Failed foraging
            
            elif action_type == 'move' or action_type == 'explore':
                reward -= 0.3  # Small movement cost
                agent.energy -= 0.5
            
            # Survival bonus
            if agent.energy > 0:
                reward += 0.5
        
        # Death penalty
        if agent.energy <= 0:
            agent.is_alive = False
            reward -= 50.0
        
        return reward
    
    def apply_action(self, agent, action: Dict[str, Any], env: Environment):
        """Apply action to agent in environment."""
        if 'x' in action and 'y' in action:
            # Bound positions to environment
            new_x = max(0, min(env.width - 1, int(action['x'])))
            new_y = max(0, min(env.height - 1, int(action['y'])))
            
            # Update position
            agent.pos_x = new_x
            agent.pos_y = new_y
        
        # Age the agent
        if hasattr(agent, 'age'):
            agent.age += 1
        
        # Natural energy decay
        agent.energy -= 0.5
    
    def train_algorithm(self, algorithm: str) -> Dict[str, List]:
        """Train a specific algorithm."""
        logger.info(f"Training {algorithm.upper()} algorithm...")
        
        # Create agents
        agents = self.create_agents(algorithm)
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        losses = []
        
        for episode in range(self.config.num_episodes):
            # Create fresh environment
            env = create_training_environment()
            
            # Episode tracking
            episode_reward = {'predator': 0, 'prey': 0}
            episode_steps = 0
            trajectories = {'predator': [], 'prey': []}
            
            for step in range(self.config.max_steps_per_episode):
                episode_steps += 1
                
                # Get active agents
                predators, prey = self.get_agent_lists(env)
                
                if len(predators) == 0 or len(prey) == 0:
                    break  # Episode ends when one species is extinct
                
                # Process predators
                for predator in predators:
                    state = self.get_agent_state(predator, env)
                    
                    if algorithm == 'maddpg':
                        action = agents['predator'].select_action(state, agent_id=0)
                    else:
                        action = agents['predator'].select_action(state)
                    
                    # Apply action and calculate reward
                    self.apply_action(predator, action, env)
                    reward = self.calculate_reward(predator, action, env)
                    
                    episode_reward['predator'] += reward
                    
                    # Store trajectory
                    trajectories['predator'].append({
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'done': not predator.is_alive
                    })
                
                # Process prey
                for prey_agent in prey:
                    state = self.get_agent_state(prey_agent, env)
                    
                    if algorithm == 'maddpg':
                        action = agents['prey'].select_action(state, agent_id=0)
                    else:
                        action = agents['prey'].select_action(state)
                    
                    # Apply action and calculate reward
                    self.apply_action(prey_agent, action, env)
                    reward = self.calculate_reward(prey_agent, action, env)
                    
                    episode_reward['prey'] += reward
                    
                    # Store trajectory
                    trajectories['prey'].append({
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'done': not prey_agent.is_alive
                    })
                
                # Update environment
                env.step()
            
            # Update agents with collected trajectories
            for agent_type in ['predator', 'prey']:
                if trajectories[agent_type]:
                    agents[agent_type].update(trajectories[agent_type])
            
            # Record metrics
            avg_reward = np.mean(list(episode_reward.values()))
            episode_rewards.append(avg_reward)
            episode_lengths.append(episode_steps)
            
            # Log progress
            if episode % 50 == 0:
                recent_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else avg_reward
                logger.info(f"{algorithm.upper()} Episode {episode:4d}: "
                           f"Reward = {recent_reward:8.2f}, "
                           f"Steps = {episode_steps:3d}, "
                           f"Predators = {len(predators):2d}, "
                           f"Prey = {len(prey):2d}")
            
            # Save models periodically
            if episode % 200 == 0 and episode > 0:
                self.save_models(algorithm, agents, episode)
        
        # Final model save
        self.save_models(algorithm, agents, self.config.num_episodes)
        
        # Store metrics
        self.training_metrics[algorithm]['rewards'] = episode_rewards
        self.training_metrics[algorithm]['episode_lengths'] = episode_lengths
        self.training_metrics[algorithm]['losses'] = losses
        
        logger.info(f"{algorithm.upper()} training completed. "
                   f"Final reward: {np.mean(episode_rewards[-10:]):.2f}")
        
        return {
            'rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'losses': losses
        }
    
    def save_models(self, algorithm: str, agents: Dict[str, Any], episode: int):
        """Save trained models."""
        algo_dir = self.save_dir / algorithm
        algo_dir.mkdir(exist_ok=True)
        
        for agent_type, agent in agents.items():
            if hasattr(agent, 'save'):
                model_path = algo_dir / f"{agent_type}_episode_{episode}.pth"
                agent.save(str(model_path))
                logger.info(f"Saved {algorithm} {agent_type} model to {model_path}")
    
    def train_all_algorithms(self) -> Dict[str, Dict]:
        """Train all algorithms sequentially."""
        results = {}
        
        algorithms = ['lotka_volterra', 'epsilon_greedy', 'ppo', 'maddpg']
        
        for algorithm in algorithms:
            try:
                start_time = time.time()
                results[algorithm] = self.train_algorithm(algorithm)
                elapsed = time.time() - start_time
                logger.info(f"{algorithm.upper()} training completed in {elapsed:.1f} seconds")
            except Exception as e:
                logger.error(f"Error training {algorithm}: {e}")
                results[algorithm] = {'error': str(e)}
        
        return results
    
    def create_training_plots(self, results: Dict[str, Dict]):
        """Create comprehensive training plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BioFlux RL Training Results', fontsize=16, fontweight='bold')
        
        # Reward curves
        ax1 = axes[0, 0]
        for algorithm, metrics in results.items():
            if 'rewards' in metrics and metrics['rewards']:
                rewards = metrics['rewards']
                # Smooth rewards with moving average
                window = min(50, len(rewards) // 10)
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    episodes = range(window-1, len(rewards))
                    ax1.plot(episodes, smoothed, label=algorithm.replace('_', '-').upper(), linewidth=2)
                else:
                    ax1.plot(rewards, label=algorithm.replace('_', '-').upper(), linewidth=2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Learning Progress (Smoothed)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode lengths
        ax2 = axes[0, 1]
        for algorithm, metrics in results.items():
            if 'episode_lengths' in metrics and metrics['episode_lengths']:
                lengths = metrics['episode_lengths']
                # Moving average for episode lengths
                window = min(50, len(lengths) // 10)
                if window > 1:
                    smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
                    episodes = range(window-1, len(lengths))
                    ax2.plot(episodes, smoothed, label=algorithm.replace('_', '-').upper(), linewidth=2)
                else:
                    ax2.plot(lengths, label=algorithm.replace('_', '-').upper(), linewidth=2)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Duration Trends')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Final performance comparison
        ax3 = axes[1, 0]
        algorithms = []
        final_rewards = []
        
        for algorithm, metrics in results.items():
            if 'rewards' in metrics and metrics['rewards']:
                algorithms.append(algorithm.replace('_', '-').upper())
                # Average of last 50 episodes
                final_rewards.append(np.mean(metrics['rewards'][-50:]))
        
        if algorithms:
            bars = ax3.bar(algorithms, final_rewards, alpha=0.8)
            ax3.set_ylabel('Final Average Reward')
            ax3.set_title('Final Performance Comparison')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, reward in zip(bars, final_rewards):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{reward:.1f}', ha='center', va='bottom')
        
        # Training stability (reward variance)
        ax4 = axes[1, 1]
        for algorithm, metrics in results.items():
            if 'rewards' in metrics and metrics['rewards']:
                rewards = metrics['rewards']
                # Calculate rolling standard deviation
                window = 100
                if len(rewards) > window:
                    rolling_std = []
                    for i in range(window, len(rewards)):
                        rolling_std.append(np.std(rewards[i-window:i]))
                    
                    episodes = range(window, len(rewards))
                    ax4.plot(episodes, rolling_std, label=algorithm.replace('_', '-').upper(), linewidth=2)
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Reward Standard Deviation')
        ax4.set_title('Training Stability (Lower is Better)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path("output") / "training_results.png"
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training plots saved to {plot_path}")
        
        plt.show()
        
        return plot_path
    
    def save_training_results(self, results: Dict[str, Dict]):
        """Save complete training results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        results_path = Path("output") / f"training_results_{timestamp}.json"
        results_path.parent.mkdir(exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for algorithm, metrics in results.items():
            json_results[algorithm] = {}
            for metric_name, metric_values in metrics.items():
                if isinstance(metric_values, np.ndarray):
                    json_results[algorithm][metric_name] = metric_values.tolist()
                elif isinstance(metric_values, list):
                    json_results[algorithm][metric_name] = metric_values
                else:
                    json_results[algorithm][metric_name] = metric_values
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Training results saved to {results_path}")
        return results_path

def main():
    """Main training function."""
    print("🚀 BioFlux RL Full Training Loop")
    print("=" * 50)
    
    # Configuration
    config = TrainingConfig(
        num_episodes=1000,  # Full training
        max_steps_per_episode=300,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon=0.3,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        batch_size=64,
        memory_size=10000,
        target_update_frequency=100,
        save_frequency=200,
        log_frequency=50,
        hidden_dim=256,
        device="cpu"  # Use CPU for stability
    )
    
    print(f"Training configuration:")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Max steps per episode: {config.max_steps_per_episode}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Device: {config.device}")
    print()
    
    # Create trainer
    trainer = FullTrainingLoop(config)
    
    # Train all algorithms
    print("🎯 Starting comprehensive training...")
    start_time = time.time()
    
    results = trainer.train_all_algorithms()
    
    total_time = time.time() - start_time
    print(f"\n✅ All training completed in {total_time:.1f} seconds")
    
    # Generate plots and save results
    print("📊 Generating training plots...")
    trainer.create_training_plots(results)
    
    print("💾 Saving training results...")
    trainer.save_training_results(results)
    
    # Summary
    print("\n🏆 Training Summary:")
    for algorithm, metrics in results.items():
        if 'rewards' in metrics and metrics['rewards']:
            final_reward = np.mean(metrics['rewards'][-50:])
            print(f"  {algorithm.replace('_', '-').upper():15s}: Final reward = {final_reward:8.2f}")
        elif 'error' in metrics:
            print(f"  {algorithm.replace('_', '-').upper():15s}: Training failed - {metrics['error']}")
    
    print(f"\n📁 Models saved in: {trainer.save_dir}")
    print("🚀 Models are ready for inference!")

if __name__ == "__main__":
    main()
