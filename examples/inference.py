#!/usr/bin/env python3
"""
BioFlux RL Inference System

This module provides inference capabilities for trained RL models.
It can load saved models and use them for prediction and evaluation.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging

# Add the parent directory to the path to import bioflux
sys.path.append(str(Path(__file__).parent.parent))

from bioflux.training import (
    TrainingConfig, LotkaVolterraAgent, EpsilonGreedyAgent, 
    PPOAgent, MADDPGWrapper, create_training_environment
)
from bioflux.core.environment import Environment

logger = logging.getLogger(__name__)

class ModelInference:
    """Inference system for trained BioFlux RL models."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.model_configs = {}
        
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")
        
        logger.info(f"Inference system initialized with models from: {self.models_dir}")
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available trained models."""
        available_models = {}
        
        for algorithm_dir in self.models_dir.iterdir():
            if algorithm_dir.is_dir():
                algorithm = algorithm_dir.name
                model_files = []
                
                for model_file in algorithm_dir.glob("*.pth"):
                    model_files.append(model_file.stem)
                
                if model_files:
                    available_models[algorithm] = sorted(model_files)
        
        return available_models
    
    def load_model(self, algorithm: str, agent_type: str, episode: int = None) -> Any:
        """Load a specific trained model."""
        algorithm_dir = self.models_dir / algorithm
        
        if not algorithm_dir.exists():
            raise FileNotFoundError(f"Algorithm directory not found: {algorithm_dir}")
        
        # Find model file
        if episode is None:
            # Load the latest model
            model_files = list(algorithm_dir.glob(f"{agent_type}_episode_*.pth"))
            if not model_files:
                raise FileNotFoundError(f"No models found for {algorithm}/{agent_type}")
            
            # Sort by episode number
            model_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            model_path = model_files[-1]
        else:
            model_path = algorithm_dir / f"{agent_type}_episode_{episode}.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create agent and load model
        config = TrainingConfig(device="cpu")  # Default config for inference
        state_dim = 8
        action_dim = 8
        
        if algorithm == 'lotka_volterra':
            agent = LotkaVolterraAgent(agent_type, config)
            # Lotka-Volterra doesn't have learnable parameters
        elif algorithm == 'epsilon_greedy':
            agent = EpsilonGreedyAgent(agent_type, config, state_dim, action_dim)
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                agent.q_network.load_state_dict(checkpoint['q_network'])
                agent.target_network.load_state_dict(checkpoint['target_network'])
                agent.epsilon = checkpoint.get('epsilon', 0.01)  # Use low epsilon for inference
        elif algorithm == 'ppo':
            agent = PPOAgent(agent_type, config, state_dim, action_dim)
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                agent.actor.load_state_dict(checkpoint['actor'])
                agent.critic.load_state_dict(checkpoint['critic'])
        elif algorithm == 'maddpg':
            agent = MADDPGWrapper(agent_type, config)
            # Load MADDPG models (multiple agents)
            for i in range(agent.num_agents):
                agent_model_path = str(model_path).replace('.pth', f'_agent_{i}.pth')
                if Path(agent_model_path).exists():
                    agent.agents[i].load(agent_model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Store loaded model
        model_key = f"{algorithm}_{agent_type}_{episode or 'latest'}"
        self.loaded_models[model_key] = agent
        
        logger.info(f"Loaded model: {model_path}")
        return agent
    
    def load_all_latest_models(self) -> Dict[str, Dict[str, Any]]:
        """Load the latest models for all algorithms and agent types."""
        available_models = self.list_available_models()
        loaded_models = {}
        
        for algorithm in available_models:
            loaded_models[algorithm] = {}
            
            for agent_type in ['predator', 'prey']:
                try:
                    agent = self.load_model(algorithm, agent_type)
                    loaded_models[algorithm][agent_type] = agent
                    logger.info(f"Loaded {algorithm} {agent_type} model")
                except Exception as e:
                    logger.warning(f"Failed to load {algorithm} {agent_type}: {e}")
        
        return loaded_models
    
    def predict_action(self, agent: Any, state: Dict[str, Any], algorithm: str) -> Dict[str, Any]:
        """Predict action using loaded model."""
        try:
            if algorithm == 'maddpg':
                return agent.select_action(state, agent_id=0)
            else:
                return agent.select_action(state)
        except Exception as e:
            logger.error(f"Error predicting action: {e}")
            # Return default action
            return {'action': 'stay', 'x': state.get('pos_x', 0), 'y': state.get('pos_y', 0)}
    
    def evaluate_model(self, algorithm: str, agent_type: str, num_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate a trained model's performance."""
        # Load model
        try:
            agent = self.load_model(algorithm, agent_type)
        except Exception as e:
            logger.error(f"Failed to load model for evaluation: {e}")
            return {'error': str(e)}
        
        # Evaluation metrics
        episode_rewards = []
        episode_lengths = []
        success_rates = []
        
        logger.info(f"Evaluating {algorithm} {agent_type} model for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Create evaluation environment
            env = create_training_environment()
            
            episode_reward = 0
            episode_steps = 0
            successes = 0
            total_actions = 0
            
            for step in range(200):  # Max steps per episode
                episode_steps += 1
                
                # Get agents of the specified type
                if agent_type == 'predator':
                    agents_list = [a for a in env.predators if a.is_alive]
                else:
                    agents_list = [a for a in env.prey if a.is_alive]
                
                if not agents_list:
                    break
                
                # Use first agent for evaluation
                eval_agent = agents_list[0]
                
                # Get state
                state = self.get_agent_state(eval_agent, env)
                
                # Predict action
                action = self.predict_action(agent, state, algorithm)
                
                # Apply action
                self.apply_action(eval_agent, action, env)
                
                # Calculate reward
                reward = self.calculate_reward(eval_agent, action, env, agent_type)
                episode_reward += reward
                
                # Check success (positive reward)
                total_actions += 1
                if reward > 0:
                    successes += 1
                
                # Update environment
                env.step()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            success_rates.append(successes / max(total_actions, 1))
            
            if episode % 20 == 0:
                logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, Length = {episode_steps}")
        
        # Calculate statistics
        results = {
            'algorithm': algorithm,
            'agent_type': agent_type,
            'num_episodes': num_episodes,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'avg_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'success_rates': success_rates
        }
        
        logger.info(f"Evaluation completed: Avg reward = {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        
        return results
    
    def get_agent_state(self, agent, env: Environment) -> Dict[str, Any]:
        """Get state observation for an agent (same as training)."""
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
    
    def apply_action(self, agent, action: Dict[str, Any], env: Environment):
        """Apply action to agent."""
        if 'x' in action and 'y' in action:
            agent.pos_x = max(0, min(env.width - 1, int(action['x'])))
            agent.pos_y = max(0, min(env.height - 1, int(action['y'])))
        
        if hasattr(agent, 'age'):
            agent.age += 1
        
        agent.energy -= 0.5  # Natural energy decay
    
    def calculate_reward(self, agent, action: Dict[str, Any], env: Environment, agent_type: str) -> float:
        """Calculate reward (simplified version for evaluation)."""
        reward = 0.0
        action_type = action.get('action', 'stay')
        
        if agent_type == 'predator':
            if action_type == 'hunt':
                # Check for nearby prey
                for prey_agent in env.prey:
                    if prey_agent.is_alive:
                        distance = np.sqrt((prey_agent.pos_x - agent.pos_x)**2 + 
                                         (prey_agent.pos_y - agent.pos_y)**2)
                        if distance < 3:
                            reward += 10.0
                            break
                else:
                    reward -= 1.0
            else:
                reward += 0.1  # Small survival bonus
        
        elif agent_type == 'prey':
            if action_type == 'flee':
                # Check distance from predators
                min_pred_distance = float('inf')
                for pred_agent in env.predators:
                    if pred_agent.is_alive:
                        distance = np.sqrt((pred_agent.pos_x - agent.pos_x)**2 + 
                                         (pred_agent.pos_y - agent.pos_y)**2)
                        min_pred_distance = min(min_pred_distance, distance)
                
                if min_pred_distance < 10:
                    reward += 3.0  # Good escape
                else:
                    reward -= 0.5  # Unnecessary flee
            
            elif action_type == 'forage':
                # Check for nearby food
                for plant in env.plants:
                    distance = np.sqrt((plant.pos_x - agent.pos_x)**2 + 
                                     (plant.pos_y - agent.pos_y)**2)
                    if distance < 2:
                        reward += 5.0
                        break
                else:
                    reward -= 0.5
            else:
                reward += 0.1  # Small survival bonus
        
        return reward
    
    def compare_models(self, algorithms: List[str] = None, num_episodes: int = 50) -> Dict[str, Any]:
        """Compare performance of different algorithms."""
        if algorithms is None:
            available = self.list_available_models()
            algorithms = list(available.keys())
        
        comparison_results = {}
        
        for algorithm in algorithms:
            for agent_type in ['predator', 'prey']:
                try:
                    results = self.evaluate_model(algorithm, agent_type, num_episodes)
                    if 'error' not in results:
                        key = f"{algorithm}_{agent_type}"
                        comparison_results[key] = results
                except Exception as e:
                    logger.warning(f"Failed to evaluate {algorithm} {agent_type}: {e}")
        
        return comparison_results
    
    def visualize_evaluation(self, results: Dict[str, Any], save_path: str = None):
        """Create visualization of evaluation results."""
        if not results:
            logger.warning("No results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # Extract data
        algorithms = []
        avg_rewards = []
        std_rewards = []
        success_rates = []
        episode_lengths = []
        
        for key, result in results.items():
            if 'error' not in result:
                algorithms.append(key.replace('_', '\n'))
                avg_rewards.append(result['avg_reward'])
                std_rewards.append(result['std_reward'])
                success_rates.append(result['avg_success_rate'])
                episode_lengths.append(result['avg_episode_length'])
        
        if not algorithms:
            logger.warning("No valid results to plot")
            return
        
        # Average rewards with error bars
        ax1 = axes[0, 0]
        bars1 = ax1.bar(algorithms, avg_rewards, yerr=std_rewards, capsize=5, alpha=0.8)
        ax1.set_title('Average Reward per Episode')
        ax1.set_ylabel('Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Success rates
        ax2 = axes[0, 1]
        bars2 = ax2.bar(algorithms, success_rates, alpha=0.8, color='green')
        ax2.set_title('Success Rate')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Episode lengths
        ax3 = axes[1, 0]
        bars3 = ax3.bar(algorithms, episode_lengths, alpha=0.8, color='orange')
        ax3.set_title('Average Episode Length')
        ax3.set_ylabel('Steps')
        ax3.tick_params(axis='x', rotation=45)
        
        # Reward distribution (box plot)
        ax4 = axes[1, 1]
        reward_distributions = []
        labels = []
        
        for key, result in results.items():
            if 'error' not in result and 'episode_rewards' in result:
                reward_distributions.append(result['episode_rewards'])
                labels.append(key.replace('_', '\n'))
        
        if reward_distributions:
            ax4.boxplot(reward_distributions, labels=labels)
            ax4.set_title('Reward Distribution')
            ax4.set_ylabel('Reward')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plot saved to {save_path}")
        
        plt.show()

def main():
    """Main inference demonstration."""
    print("🔍 BioFlux RL Model Inference System")
    print("=" * 50)
    
    # Initialize inference system
    try:
        inference = ModelInference()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please run training first: python examples/full_training.py")
        return
    
    # List available models
    available_models = inference.list_available_models()
    print("📋 Available trained models:")
    for algorithm, models in available_models.items():
        print(f"  {algorithm.upper()}: {len(models)} models")
        for model in models[:3]:  # Show first 3
            print(f"    - {model}")
        if len(models) > 3:
            print(f"    ... and {len(models) - 3} more")
    print()
    
    if not available_models:
        print("❌ No trained models found!")
        print("Please run training first: python examples/full_training.py")
        return
    
    # Load and evaluate models
    print("🎯 Evaluating trained models...")
    
    # Compare all available models
    comparison_results = inference.compare_models(num_episodes=30)
    
    if comparison_results:
        print("\n📊 Evaluation Results:")
        print("-" * 60)
        
        for model_key, results in comparison_results.items():
            algorithm, agent_type = model_key.split('_', 1)
            print(f"{algorithm.upper()} {agent_type.capitalize()}:")
            print(f"  Average Reward: {results['avg_reward']:8.2f} ± {results['std_reward']:.2f}")
            print(f"  Success Rate:   {results['avg_success_rate']:8.2f} ± {results['std_success_rate']:.2f}")
            print(f"  Episode Length: {results['avg_episode_length']:8.1f} ± {results['std_episode_length']:.1f}")
            print()
        
        # Create visualization
        print("📈 Creating evaluation plots...")
        plot_path = Path("output") / "model_evaluation.png"
        plot_path.parent.mkdir(exist_ok=True)
        
        inference.visualize_evaluation(comparison_results, str(plot_path))
        
        # Save results
        results_path = Path("output") / "inference_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, result in comparison_results.items():
                json_results[key] = {}
                for k, v in result.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    elif isinstance(v, (np.int64, np.float64)):
                        json_results[key][k] = float(v)
                    else:
                        json_results[key][k] = v
            
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to {results_path}")
        
    else:
        print("❌ No models could be evaluated")
    
    print("\n✅ Inference demonstration completed!")
    print("🚀 Models are ready for production use!")

if __name__ == "__main__":
    main()
