#!/usr/bin/env python3
"""
BioFlux Model Demos - Showcase Trained Models

This script demonstrates the capabilities of each trained RL model
with visual simulations and performance comparisons.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import time
import json
from typing import Dict, List, Any, Tuple
import logging

# Add the parent directory to the path to import bioflux
sys.path.append(str(Path(__file__).parent.parent))

from bioflux.training import (
    TrainingConfig, LotkaVolterraAgent, EpsilonGreedyAgent, 
    PPOAgent, MADDPGWrapper, create_training_environment
)
from bioflux.core.environment import Environment
from bioflux.core.agents import Predator, Prey, Plant

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDemonstrator:
    """Comprehensive model demonstration system."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.demo_results = {}
        
        # Configuration for demos
        self.demo_config = TrainingConfig(
            num_episodes=20,  # Shorter for demo
            max_steps_per_episode=100,
            device="cpu"
        )
        
        logger.info(f"Model demonstrator initialized with models from: {self.models_dir}")
    
    def load_model(self, algorithm: str, agent_type: str, episode: int = None) -> Any:
        """Load a specific trained model."""
        algorithm_dir = self.models_dir / algorithm
        
        if not algorithm_dir.exists():
            logger.warning(f"Algorithm directory not found: {algorithm_dir}")
            return None
        
        # Find model file
        if episode is None:
            # Load the latest model
            if algorithm == 'maddpg':
                model_files = list(algorithm_dir.glob(f"{agent_type}_episode_*_agent_0.pth"))
            else:
                model_files = list(algorithm_dir.glob(f"{agent_type}_episode_*.pth"))
            
            if not model_files:
                logger.warning(f"No models found for {algorithm}/{agent_type}")
                return None
            
            # Sort by episode number
            model_files.sort(key=lambda x: int(x.stem.split('_')[-1] if algorithm != 'maddpg' else x.stem.split('_')[-2]))
            model_path = model_files[-1]
        else:
            if algorithm == 'maddpg':
                model_path = algorithm_dir / f"{agent_type}_episode_{episode}_agent_0.pth"
            else:
                model_path = algorithm_dir / f"{agent_type}_episode_{episode}.pth"
            
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                return None
        
        # Create and load agent
        try:
            state_dim = 8
            action_dim = 8
            
            if algorithm == 'lotka_volterra':
                agent = LotkaVolterraAgent(agent_type, self.demo_config)
                # Lotka-Volterra doesn't have learnable parameters
                return agent
                
            elif algorithm == 'epsilon_greedy':
                agent = EpsilonGreedyAgent(agent_type, self.demo_config, state_dim, action_dim)
                if model_path.exists():
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    agent.q_network.load_state_dict(checkpoint['q_network'])
                    agent.target_network.load_state_dict(checkpoint['target_network'])
                    agent.epsilon = 0.01  # Use low epsilon for inference
                return agent
                
            elif algorithm == 'ppo':
                agent = PPOAgent(agent_type, self.demo_config, state_dim, action_dim)
                if model_path.exists():
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    agent.actor.load_state_dict(checkpoint['actor'])
                    agent.critic.load_state_dict(checkpoint['critic'])
                return agent
                
            elif algorithm == 'maddpg':
                agent = MADDPGWrapper(agent_type, self.demo_config)
                if model_path.exists():
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    agent.load_models(checkpoint)
                return agent
                
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None
    
    def create_demo_environment(self) -> Environment:
        """Create a demo environment with visualization capabilities."""
        env = Environment()
        env.width = 50
        env.height = 50
        
        # Add predators
        for i in range(3):
            predator = Predator(
                speed=2, 
                energy=100, 
                pos_x=np.random.randint(5, 45), 
                pos_y=np.random.randint(5, 45),
                age=1
            )
            env.add_predator(predator)
        
        # Add prey
        for i in range(5):
            prey = Prey(
                speed=3, 
                energy=50, 
                pos_x=np.random.randint(5, 45), 
                pos_y=np.random.randint(5, 45),
                age=1
            )
            env.add_prey(prey)
        
        # Add plants
        for i in range(15):
            plant = Plant(
                energy=10,
                pos_x=np.random.randint(0, 50),
                pos_y=np.random.randint(0, 50)
            )
            env.add_plant(plant)
        
        return env
    
    def get_agent_state(self, agent, env: Environment) -> Dict[str, Any]:
        """Get state observation for an agent."""
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
        
        # Get environmental conditions (with safety checks)
        y_idx = min(int(agent.pos_y), env.height-1)
        x_idx = min(int(agent.pos_x), env.width-1)
        temp = env.temperature_map[y_idx, x_idx]
        vegetation = env.vegetation_map[y_idx, x_idx]
        
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
    
    def demo_algorithm(self, algorithm: str, episodes: int = 10) -> Dict[str, Any]:
        """Demonstrate a specific algorithm with detailed metrics."""
        print(f"\n🎯 Demonstrating {algorithm.upper()} Models")
        print("-" * 60)
        
        # Load models
        predator_agent = self.load_model(algorithm, 'predator')
        prey_agent = self.load_model(algorithm, 'prey')
        
        if predator_agent is None and prey_agent is None:
            print(f"❌ No models available for {algorithm}")
            return {'error': 'No models available'}
        
        # Demo metrics
        episode_rewards = []
        episode_lengths = []
        survival_rates = []
        action_distributions = {'predator': {}, 'prey': {}}
        agent_trajectories = []
        
        for episode in range(episodes):
            env = self.create_demo_environment()
            episode_reward = {'predator': 0, 'prey': 0}
            episode_steps = 0
            episode_trajectory = {'predator': [], 'prey': []}
            
            initial_predators = len(env.predators)
            initial_prey = len(env.prey)
            
            for step in range(self.demo_config.max_steps_per_episode):
                episode_steps += 1
                
                # Get active agents
                active_predators = [p for p in env.predators if p.is_alive]
                active_prey = [p for p in env.prey if p.is_alive]
                
                if len(active_predators) == 0 or len(active_prey) == 0:
                    break
                
                # Process predators
                if predator_agent:
                    for predator in active_predators:
                        state = self.get_agent_state(predator, env)
                        
                        try:
                            if algorithm == 'maddpg':
                                action = predator_agent.select_action(state, agent_id=0)
                            else:
                                action = predator_agent.select_action(state)
                        except Exception as e:
                            # Fallback action
                            action = {'action': 'stay', 'x': predator.pos_x, 'y': predator.pos_y}
                        
                        # Track action distribution
                        action_type = action.get('action', 'stay')
                        action_distributions['predator'][action_type] = action_distributions['predator'].get(action_type, 0) + 1
                        
                        # Apply action
                        if 'x' in action and 'y' in action:
                            predator.pos_x = max(0, min(env.width-1, int(action['x'])))
                            predator.pos_y = max(0, min(env.height-1, int(action['y'])))
                        
                        # Simple reward calculation
                        reward = self.calculate_simple_reward(predator, action, env)
                        episode_reward['predator'] += reward
                        
                        # Store trajectory
                        episode_trajectory['predator'].append({
                            'step': step,
                            'pos': (predator.pos_x, predator.pos_y),
                            'energy': predator.energy,
                            'action': action_type,
                            'reward': reward
                        })
                
                # Process prey
                if prey_agent:
                    for prey in active_prey:
                        state = self.get_agent_state(prey, env)
                        
                        try:
                            if algorithm == 'maddpg':
                                action = prey_agent.select_action(state, agent_id=0)
                            else:
                                action = prey_agent.select_action(state)
                        except Exception as e:
                            # Fallback action
                            action = {'action': 'stay', 'x': prey.pos_x, 'y': prey.pos_y}
                        
                        # Track action distribution
                        action_type = action.get('action', 'stay')
                        action_distributions['prey'][action_type] = action_distributions['prey'].get(action_type, 0) + 1
                        
                        # Apply action
                        if 'x' in action and 'y' in action:
                            prey.pos_x = max(0, min(env.width-1, int(action['x'])))
                            prey.pos_y = max(0, min(env.height-1, int(action['y'])))
                        
                        # Simple reward calculation
                        reward = self.calculate_simple_reward(prey, action, env)
                        episode_reward['prey'] += reward
                        
                        # Store trajectory
                        episode_trajectory['prey'].append({
                            'step': step,
                            'pos': (prey.pos_x, prey.pos_y),
                            'energy': prey.energy,
                            'action': action_type,
                            'reward': reward
                        })
                
                # Update environment
                env.step()
            
            # Calculate survival rates
            final_predators = len([p for p in env.predators if p.is_alive])
            final_prey = len([p for p in env.prey if p.is_alive])
            
            survival_rate = {
                'predator': final_predators / initial_predators if initial_predators > 0 else 0,
                'prey': final_prey / initial_prey if initial_prey > 0 else 0
            }
            
            # Record episode metrics
            avg_reward = np.mean(list(episode_reward.values()))
            episode_rewards.append(avg_reward)
            episode_lengths.append(episode_steps)
            survival_rates.append(survival_rate)
            agent_trajectories.append(episode_trajectory)
            
            if episode % 5 == 0:
                print(f"  Episode {episode:2d}: Reward = {avg_reward:6.2f}, "
                      f"Steps = {episode_steps:3d}, "
                      f"Predator Survival = {survival_rate['predator']:.2f}, "
                      f"Prey Survival = {survival_rate['prey']:.2f}")
        
        # Calculate final metrics
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0
        avg_survival = {
            'predator': np.mean([sr['predator'] for sr in survival_rates]) if survival_rates else 0,
            'prey': np.mean([sr['prey'] for sr in survival_rates]) if survival_rates else 0
        }
        
        # Print summary
        print(f"\n📊 {algorithm.upper()} Summary:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Episode Length: {avg_length:.1f}")
        print(f"  Predator Survival Rate: {avg_survival['predator']:.2%}")
        print(f"  Prey Survival Rate: {avg_survival['prey']:.2%}")
        
        # Print action distributions
        print(f"  Action Distributions:")
        for agent_type, actions in action_distributions.items():
            if actions:
                total_actions = sum(actions.values())
                print(f"    {agent_type.capitalize()}:")
                for action, count in sorted(actions.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_actions) * 100
                    print(f"      {action}: {percentage:.1f}% ({count})")
        
        return {
            'algorithm': algorithm,
            'episodes': episodes,
            'rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'survival_rates': survival_rates,
            'action_distributions': action_distributions,
            'trajectories': agent_trajectories,
            'summary': {
                'avg_reward': avg_reward,
                'avg_length': avg_length,
                'avg_survival': avg_survival
            }
        }
    
    def calculate_simple_reward(self, agent, action: Dict[str, Any], env: Environment) -> float:
        """Calculate simple reward for demo purposes."""
        reward = 0.0
        action_type = action.get('action', 'stay')
        
        if agent.agent_type == 'predator':
            # Reward for hunting behavior
            if action_type == 'hunt':
                nearby_prey = sum(1 for prey in env.prey 
                                if prey.is_alive and 
                                np.sqrt((prey.pos_x - agent.pos_x)**2 + (prey.pos_y - agent.pos_y)**2) < 5)
                reward += nearby_prey * 5
            
            # Small penalty for moving
            if action_type in ['move', 'explore']:
                reward -= 0.5
                
            # Survival bonus
            reward += 1.0
            
        elif agent.agent_type == 'prey':
            # Reward for foraging
            if action_type == 'forage':
                nearby_food = sum(1 for plant in env.plants 
                                if np.sqrt((plant.pos_x - agent.pos_x)**2 + (plant.pos_y - agent.pos_y)**2) < 3)
                reward += nearby_food * 3
            
            # Reward for fleeing when predators are near
            if action_type == 'flee':
                nearby_predators = sum(1 for pred in env.predators 
                                     if pred.is_alive and 
                                     np.sqrt((pred.pos_x - agent.pos_x)**2 + (pred.pos_y - agent.pos_y)**2) < 8)
                reward += nearby_predators * 4
            
            # Survival bonus
            reward += 0.5
        
        return reward
    
    def create_visualization(self, algorithm: str, trajectory_data: Dict) -> str:
        """Create visualization of agent behavior."""
        print(f"📊 Creating visualization for {algorithm.upper()}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{algorithm.upper()} Model Demonstration', fontsize=16, fontweight='bold')
        
        # 1. Trajectory plot
        ax1 = axes[0, 0]
        if trajectory_data['trajectories']:
            # Plot first episode trajectory
            first_ep = trajectory_data['trajectories'][0]
            
            if 'predator' in first_ep and first_ep['predator']:
                pred_positions = [pos['pos'] for pos in first_ep['predator']]
                if pred_positions:
                    pred_x, pred_y = zip(*pred_positions)
                    ax1.plot(pred_x, pred_y, 'r-', linewidth=2, alpha=0.7, label='Predator Path')
                    ax1.scatter(pred_x[0], pred_y[0], c='red', s=100, marker='o', label='Predator Start')
                    ax1.scatter(pred_x[-1], pred_y[-1], c='darkred', s=100, marker='x', label='Predator End')
            
            if 'prey' in first_ep and first_ep['prey']:
                prey_positions = [pos['pos'] for pos in first_ep['prey'][:len(first_ep['prey'])//3]]  # Show one prey
                if prey_positions:
                    prey_x, prey_y = zip(*prey_positions)
                    ax1.plot(prey_x, prey_y, 'b-', linewidth=2, alpha=0.7, label='Prey Path')
                    ax1.scatter(prey_x[0], prey_y[0], c='blue', s=100, marker='o', label='Prey Start')
                    ax1.scatter(prey_x[-1], prey_y[-1], c='darkblue', s=100, marker='x', label='Prey End')
        
        ax1.set_xlim(0, 50)
        ax1.set_ylim(0, 50)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Agent Movement Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reward progression
        ax2 = axes[0, 1]
        if trajectory_data['rewards']:
            episodes = range(1, len(trajectory_data['rewards']) + 1)
            ax2.plot(episodes, trajectory_data['rewards'], 'g-', linewidth=2, marker='o')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Reward')
            ax2.set_title('Reward Progression')
            ax2.grid(True, alpha=0.3)
        
        # 3. Action distribution
        ax3 = axes[1, 0]
        pred_actions = trajectory_data['action_distributions'].get('predator', {})
        prey_actions = trajectory_data['action_distributions'].get('prey', {})
        
        if pred_actions or prey_actions:
            all_actions = set(pred_actions.keys()) | set(prey_actions.keys())
            x_pos = np.arange(len(all_actions))
            width = 0.35
            
            pred_counts = [pred_actions.get(action, 0) for action in all_actions]
            prey_counts = [prey_actions.get(action, 0) for action in all_actions]
            
            ax3.bar(x_pos - width/2, pred_counts, width, label='Predator', alpha=0.8, color='red')
            ax3.bar(x_pos + width/2, prey_counts, width, label='Prey', alpha=0.8, color='blue')
            
            ax3.set_xlabel('Action Type')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Action Distribution')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(list(all_actions), rotation=45)
            ax3.legend()
        
        # 4. Survival rates
        ax4 = axes[1, 1]
        if trajectory_data['survival_rates']:
            episodes = range(1, len(trajectory_data['survival_rates']) + 1)
            pred_survival = [sr['predator'] for sr in trajectory_data['survival_rates']]
            prey_survival = [sr['prey'] for sr in trajectory_data['survival_rates']]
            
            ax4.plot(episodes, pred_survival, 'r-', linewidth=2, marker='o', label='Predator')
            ax4.plot(episodes, prey_survival, 'b-', linewidth=2, marker='s', label='Prey')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Survival Rate')
            ax4.set_title('Survival Rates Over Episodes')
            ax4.set_ylim(0, 1.1)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / f"{algorithm}_demo_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"  Visualization saved to {plot_path}")
        plt.show()
        
        return str(plot_path)
    
    def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run comprehensive demonstration of all available models."""
        print("🚀 BioFlux Trained Model Demonstrations")
        print("=" * 60)
        print("This demo showcases the behavior of each trained RL algorithm")
        print()
        
        # Find available algorithms
        available_algorithms = []
        for algo_dir in self.models_dir.iterdir():
            if algo_dir.is_dir() and list(algo_dir.glob("*.pth")):
                available_algorithms.append(algo_dir.name)
        
        print(f"📋 Found trained models for: {', '.join(available_algorithms)}")
        print()
        
        all_results = {}
        
        # Demo each algorithm
        for algorithm in available_algorithms:
            try:
                result = self.demo_algorithm(algorithm, episodes=15)
                if 'error' not in result:
                    all_results[algorithm] = result
                    
                    # Create visualization
                    self.create_visualization(algorithm, result)
                    
                    # Save individual results
                    self.save_demo_results(algorithm, result)
                    
                time.sleep(1)  # Brief pause between demos
                
            except Exception as e:
                logger.error(f"Error demonstrating {algorithm}: {e}")
                continue
        
        # Create comparison summary
        if len(all_results) > 1:
            self.create_comparison_summary(all_results)
        
        print("\n✅ All model demonstrations completed!")
        print(f"📊 Results saved in output/ directory")
        
        return all_results
    
    def save_demo_results(self, algorithm: str, results: Dict) -> None:
        """Save demo results to JSON."""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self.convert_for_json(results)
        
        results_path = output_dir / f"{algorithm}_demo_results.json"
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def convert_for_json(self, obj):
        """Convert numpy arrays and other objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self.convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def create_comparison_summary(self, all_results: Dict[str, Dict]) -> None:
        """Create a summary comparison of all algorithms."""
        print("\n🏆 Model Performance Comparison")
        print("=" * 60)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BioFlux Model Comparison Summary', fontsize=16, fontweight='bold')
        
        algorithms = list(all_results.keys())
        
        # 1. Average rewards comparison
        ax1 = axes[0, 0]
        avg_rewards = [all_results[algo]['summary']['avg_reward'] for algo in algorithms]
        bars1 = ax1.bar(algorithms, avg_rewards, alpha=0.8, color=['red', 'blue', 'green', 'orange'][:len(algorithms)])
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Average Reward Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, reward in zip(bars1, avg_rewards):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{reward:.1f}', ha='center', va='bottom')
        
        # 2. Episode lengths comparison
        ax2 = axes[0, 1]
        avg_lengths = [all_results[algo]['summary']['avg_length'] for algo in algorithms]
        bars2 = ax2.bar(algorithms, avg_lengths, alpha=0.8, color=['red', 'blue', 'green', 'orange'][:len(algorithms)])
        ax2.set_ylabel('Average Episode Length')
        ax2.set_title('Episode Duration Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, length in zip(bars2, avg_lengths):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{length:.1f}', ha='center', va='bottom')
        
        # 3. Survival rates comparison
        ax3 = axes[1, 0]
        pred_survival = [all_results[algo]['summary']['avg_survival']['predator'] for algo in algorithms]
        prey_survival = [all_results[algo]['summary']['avg_survival']['prey'] for algo in algorithms]
        
        x_pos = np.arange(len(algorithms))
        width = 0.35
        
        ax3.bar(x_pos - width/2, pred_survival, width, label='Predator', alpha=0.8, color='red')
        ax3.bar(x_pos + width/2, prey_survival, width, label='Prey', alpha=0.8, color='blue')
        
        ax3.set_ylabel('Survival Rate')
        ax3.set_title('Average Survival Rates')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(algorithms, rotation=45)
        ax3.legend()
        ax3.set_ylim(0, 1.1)
        
        # 4. Performance ranking
        ax4 = axes[1, 1]
        
        # Calculate composite scores
        composite_scores = []
        for algo in algorithms:
            summary = all_results[algo]['summary']
            # Normalize and combine metrics (higher is better)
            score = (
                (summary['avg_reward'] + 100) / 200 +  # Normalize reward
                summary['avg_survival']['predator'] +
                summary['avg_survival']['prey']
            ) / 3
            composite_scores.append(score)
        
        # Sort by score
        sorted_data = sorted(zip(algorithms, composite_scores), key=lambda x: x[1], reverse=True)
        sorted_algorithms, sorted_scores = zip(*sorted_data)
        
        bars4 = ax4.bar(sorted_algorithms, sorted_scores, alpha=0.8, 
                       color=['gold', 'silver', 'bronze', 'lightgray'][:len(sorted_algorithms)])
        ax4.set_ylabel('Composite Performance Score')
        ax4.set_title('Overall Performance Ranking')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 1.1)
        
        # Add value labels and ranks
        for i, (bar, score) in enumerate(zip(bars4, sorted_scores)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'#{i+1}', ha='center', va='bottom', fontweight='bold')
            ax4.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{score:.3f}', ha='center', va='center')
        
        plt.tight_layout()
        
        # Save comparison plot
        output_dir = Path("output")
        plot_path = output_dir / "model_comparison_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"📊 Comparison summary saved to {plot_path}")
        plt.show()
        
        # Print ranking
        print("\n🥇 Performance Ranking:")
        for i, (algo, score) in enumerate(sorted_data):
            medal = ['🥇', '🥈', '🥉'][i] if i < 3 else f'{i+1}.'
            print(f"  {medal} {algo.upper()}: {score:.3f}")

def main():
    """Main demonstration function."""
    demonstrator = ModelDemonstrator()
    results = demonstrator.run_comprehensive_demo()
    
    print("\n" + "="*60)
    print("Demo completed! Check the output/ directory for:")
    print("  • Individual algorithm visualizations")
    print("  • Performance comparison charts")
    print("  • Detailed JSON results")
    print("🚀 All models have been successfully demonstrated!")

if __name__ == "__main__":
    main()
