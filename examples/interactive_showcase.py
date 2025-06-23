#!/usr/bin/env python3
"""
BioFlux Interactive Model Showcase

This script provides an interactive demonstration of trained models
with real-time visualization and detailed behavioral analysis.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
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

class InteractiveModelShowcase:
    """Interactive demonstration of trained models."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.config = TrainingConfig(device="cpu")
        
    def create_showcase_environment(self) -> Environment:
        """Create an environment optimized for visualization."""
        env = Environment()
        env.width = 30
        env.height = 30
        
        # Strategic placement for better visualization
        # Predators in corners
        predator1 = Predator(speed=2, energy=100, pos_x=5, pos_y=5, age=1)
        predator2 = Predator(speed=2, energy=100, pos_x=25, pos_y=25, age=1)
        env.add_predator(predator1)
        env.add_predator(predator2)
        
        # Prey in middle areas
        for i in range(4):
            x = np.random.randint(10, 20)
            y = np.random.randint(10, 20)
            prey = Prey(speed=3, energy=50, pos_x=x, pos_y=y, age=1)
            env.add_prey(prey)
        
        # Plants scattered around
        for i in range(12):
            x = np.random.randint(2, 28)
            y = np.random.randint(2, 28)
            plant = Plant(energy=10, pos_x=x, pos_y=y)
            env.add_plant(plant)
        
        return env
    
    def load_working_model(self, algorithm: str, agent_type: str) -> Any:
        """Load a working model with proper error handling."""
        try:
            if algorithm == 'lotka_volterra':
                return LotkaVolterraAgent(agent_type, self.config)
            
            elif algorithm == 'ppo':
                # Load PPO model (early training version)
                agent = PPOAgent(agent_type, self.config, 8, 8)
                model_path = self.models_dir / algorithm / f"{agent_type}_episode_0.pth"
                if model_path.exists():
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    if 'actor' in checkpoint:
                        agent.actor.load_state_dict(checkpoint['actor'])
                    if 'critic' in checkpoint:
                        agent.critic.load_state_dict(checkpoint['critic'])
                return agent
            
            # Add more algorithms as they become available
            return None
            
        except Exception as e:
            logger.error(f"Failed to load {algorithm} {agent_type}: {e}")
            return None
    
    def get_agent_state(self, agent, env: Environment) -> Dict[str, Any]:
        """Get standardized state for agent."""
        nearby_prey = []
        nearby_predators = []
        nearby_food = []
        
        all_agents = env.predators + env.prey + env.plants
        
        for other in all_agents:
            if other != agent and hasattr(other, 'pos_x') and hasattr(other, 'pos_y'):
                distance = np.sqrt((other.pos_x - agent.pos_x)**2 + (other.pos_y - agent.pos_y)**2)
                if distance < 10:  # Detection range
                    if hasattr(other, 'agent_type'):
                        if other.agent_type == 'prey':
                            nearby_prey.append({'pos': (other.pos_x, other.pos_y), 'distance': distance})
                        elif other.agent_type == 'predator':
                            nearby_predators.append({'pos': (other.pos_x, other.pos_y), 'distance': distance})
                        elif other.agent_type == 'plant':
                            nearby_food.append({'pos': (other.pos_x, other.pos_y), 'distance': distance})
        
        # Safe environmental data access
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
    
    def showcase_algorithm(self, algorithm: str, steps: int = 50) -> Dict[str, Any]:
        """Showcase a specific algorithm with detailed tracking."""
        print(f"\n🎭 {algorithm.upper()} Model Showcase")
        print("-" * 50)
        
        # Load agents
        predator_agent = self.load_working_model(algorithm, 'predator')
        prey_agent = self.load_working_model(algorithm, 'prey')
        
        if not predator_agent and not prey_agent:
            print(f"❌ No working models for {algorithm}")
            return {'error': 'No working models'}
        
        # Create environment
        env = self.create_showcase_environment()
        
        # Tracking data
        step_data = []
        action_log = {'predator': [], 'prey': []}
        energy_log = {'predator': [], 'prey': []}
        position_log = {'predator': [], 'prey': []}
        
        print(f"🎬 Running {steps}-step simulation...")
        
        for step in range(steps):
            step_info = {
                'step': step,
                'predators': [],
                'prey': [],
                'plants': [],
                'actions': {},
                'rewards': {}
            }
            
            # Get current state
            active_predators = [p for p in env.predators if p.is_alive]
            active_prey = [p for p in env.prey if p.is_alive]
            
            if not active_predators and not active_prey:
                break
            
            total_reward = {'predator': 0, 'prey': 0}
            
            # Process predators
            if predator_agent and active_predators:
                for i, predator in enumerate(active_predators):
                    state = self.get_agent_state(predator, env)
                    
                    try:
                        action = predator_agent.select_action(state)
                    except:
                        action = {'action': 'stay', 'x': predator.pos_x, 'y': predator.pos_y}
                    
                    # Apply action
                    if 'x' in action and 'y' in action:
                        new_x = max(0, min(env.width-1, int(action['x'])))
                        new_y = max(0, min(env.height-1, int(action['y'])))
                        predator.pos_x = new_x
                        predator.pos_y = new_y
                    
                    # Calculate reward
                    reward = self.calculate_showcase_reward(predator, action, env)
                    total_reward['predator'] += reward
                    
                    # Log data
                    action_log['predator'].append(action.get('action', 'stay'))
                    energy_log['predator'].append(predator.energy)
                    position_log['predator'].append((predator.pos_x, predator.pos_y))
                    
                    step_info['predators'].append({
                        'id': i,
                        'pos': (predator.pos_x, predator.pos_y),
                        'energy': predator.energy,
                        'action': action.get('action', 'stay'),
                        'reward': reward
                    })
            
            # Process prey
            if prey_agent and active_prey:
                for i, prey in enumerate(active_prey):
                    state = self.get_agent_state(prey, env)
                    
                    try:
                        action = prey_agent.select_action(state)
                    except:
                        action = {'action': 'stay', 'x': prey.pos_x, 'y': prey.pos_y}
                    
                    # Apply action
                    if 'x' in action and 'y' in action:
                        new_x = max(0, min(env.width-1, int(action['x'])))
                        new_y = max(0, min(env.height-1, int(action['y'])))
                        prey.pos_x = new_x
                        prey.pos_y = new_y
                    
                    # Calculate reward
                    reward = self.calculate_showcase_reward(prey, action, env)
                    total_reward['prey'] += reward
                    
                    # Log data
                    action_log['prey'].append(action.get('action', 'stay'))
                    energy_log['prey'].append(prey.energy)
                    position_log['prey'].append((prey.pos_x, prey.pos_y))
                    
                    step_info['prey'].append({
                        'id': i,
                        'pos': (prey.pos_x, prey.pos_y),
                        'energy': prey.energy,
                        'action': action.get('action', 'stay'),
                        'reward': reward
                    })
            
            # Log plants
            for i, plant in enumerate(env.plants):
                step_info['plants'].append({
                    'id': i,
                    'pos': (plant.pos_x, plant.pos_y),
                    'energy': plant.energy
                })
            
            step_info['actions'] = {
                'predator': action_log['predator'][-len(active_predators):] if active_predators else [],
                'prey': action_log['prey'][-len(active_prey):] if active_prey else []
            }
            step_info['rewards'] = total_reward
            
            step_data.append(step_info)
            
            # Update environment
            env.step()
            
            # Progress indicator
            if step % 10 == 0:
                print(f"  Step {step:2d}: Predators={len(active_predators)}, Prey={len(active_prey)}, Total Reward={sum(total_reward.values()):.1f}")
        
        # Analyze results
        analysis = self.analyze_showcase_results(step_data, action_log, energy_log, position_log)
        
        print(f"\n📊 {algorithm.upper()} Analysis:")
        print(f"  Simulation Length: {len(step_data)} steps")
        print(f"  Average Predator Energy: {analysis['avg_predator_energy']:.1f}")
        print(f"  Average Prey Energy: {analysis['avg_prey_energy']:.1f}")
        print(f"  Most Common Predator Action: {analysis['top_predator_action']}")
        print(f"  Most Common Prey Action: {analysis['top_prey_action']}")
        print(f"  Final Survival: {analysis['final_survival']} predators, {analysis['final_prey']} prey")
        
        return {
            'algorithm': algorithm,
            'step_data': step_data,
            'action_log': action_log,
            'energy_log': energy_log,
            'position_log': position_log,
            'analysis': analysis
        }\n    \n    def calculate_showcase_reward(self, agent, action: Dict[str, Any], env: Environment) -> float:\n        """Calculate reward for showcase purposes."""\n        reward = 0.0\n        action_type = action.get('action', 'stay')\n        \n        if agent.agent_type == 'predator':\n            if action_type == 'hunt':\n                # Check for nearby prey\n                nearby_prey_count = sum(1 for prey in env.prey \n                                      if prey.is_alive and \n                                      np.sqrt((prey.pos_x - agent.pos_x)**2 + (prey.pos_y - agent.pos_y)**2) < 4)\n                reward += nearby_prey_count * 10\n            \n            elif action_type == 'explore':\n                reward += 2  # Exploration bonus\n            \n            elif action_type == 'move':\n                reward += 1  # Movement bonus\n            \n            # Survival bonus\n            reward += 3\n            \n        elif agent.agent_type == 'prey':\n            if action_type == 'flee':\n                # Check if fleeing from predators\n                nearby_predators = sum(1 for pred in env.predators \n                                     if pred.is_alive and \n                                     np.sqrt((pred.pos_x - agent.pos_x)**2 + (pred.pos_y - agent.pos_y)**2) < 6)\n                reward += nearby_predators * 8\n            \n            elif action_type == 'forage':\n                # Check for nearby food\n                nearby_food = sum(1 for plant in env.plants \n                                if np.sqrt((plant.pos_x - agent.pos_x)**2 + (plant.pos_y - agent.pos_y)**2) < 3)\n                reward += nearby_food * 5\n            \n            elif action_type == 'explore':\n                reward += 1\n            \n            # Survival bonus\n            reward += 2\n        \n        return reward\n    \n    def analyze_showcase_results(self, step_data: List, action_log: Dict, energy_log: Dict, position_log: Dict) -> Dict:\n        """Analyze showcase results."""\n        analysis = {}\n        \n        # Energy analysis\n        if energy_log['predator']:\n            analysis['avg_predator_energy'] = np.mean(energy_log['predator'])\n        else:\n            analysis['avg_predator_energy'] = 0\n            \n        if energy_log['prey']:\n            analysis['avg_prey_energy'] = np.mean(energy_log['prey'])\n        else:\n            analysis['avg_prey_energy'] = 0\n        \n        # Action analysis\n        if action_log['predator']:\n            pred_actions = action_log['predator']\n            analysis['top_predator_action'] = max(set(pred_actions), key=pred_actions.count)\n        else:\n            analysis['top_predator_action'] = 'none'\n            \n        if action_log['prey']:\n            prey_actions = action_log['prey']\n            analysis['top_prey_action'] = max(set(prey_actions), key=prey_actions.count)\n        else:\n            analysis['top_prey_action'] = 'none'\n        \n        # Final state\n        if step_data:\n            final_step = step_data[-1]\n            analysis['final_survival'] = len(final_step['predators'])\n            analysis['final_prey'] = len(final_step['prey'])\n        else:\n            analysis['final_survival'] = 0\n            analysis['final_prey'] = 0\n        \n        return analysis\n    \n    def create_showcase_visualization(self, algorithm: str, showcase_data: Dict) -> str:\n        """Create comprehensive visualization of the showcase."""\n        print(f"🎨 Creating visualization for {algorithm.upper()}...")\n        \n        fig = plt.figure(figsize=(16, 12))\n        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])\n        \n        # Main simulation view\n        ax_main = fig.add_subplot(gs[0, 0])\n        self.plot_simulation_overview(ax_main, showcase_data)\n        \n        # Energy progression\n        ax_energy = fig.add_subplot(gs[0, 1])\n        self.plot_energy_progression(ax_energy, showcase_data)\n        \n        # Action distribution\n        ax_actions = fig.add_subplot(gs[0, 2])\n        self.plot_action_distribution(ax_actions, showcase_data)\n        \n        # Movement heatmap\n        ax_heatmap = fig.add_subplot(gs[1, :])\n        self.plot_movement_heatmap(ax_heatmap, showcase_data)\n        \n        # Performance metrics\n        ax_metrics = fig.add_subplot(gs[2, :])\n        self.plot_performance_metrics(ax_metrics, showcase_data)\n        \n        fig.suptitle(f'{algorithm.upper()} Model Showcase Analysis', fontsize=16, fontweight='bold')\n        plt.tight_layout()\n        \n        # Save visualization\n        output_dir = Path("output")\n        output_dir.mkdir(exist_ok=True)\n        plot_path = output_dir / f"{algorithm}_showcase_analysis.png"\n        plt.savefig(plot_path, dpi=300, bbox_inches='tight')\n        \n        print(f"  Visualization saved to {plot_path}")\n        plt.show()\n        \n        return str(plot_path)\n    \n    def plot_simulation_overview(self, ax, showcase_data):\n        """Plot simulation overview."""\n        if not showcase_data['step_data']:\n            return\n            \n        ax.set_xlim(0, 30)\n        ax.set_ylim(0, 30)\n        ax.set_title('Final Simulation State')\n        ax.set_xlabel('X Position')\n        ax.set_ylabel('Y Position')\n        \n        # Plot final positions\n        final_step = showcase_data['step_data'][-1]\n        \n        # Predators\n        for pred in final_step['predators']:\n            ax.scatter(pred['pos'][0], pred['pos'][1], c='red', s=100, marker='^', \n                      alpha=0.8, edgecolors='black', label='Predator' if pred['id'] == 0 else "")\n            ax.annotate(f"E:{pred['energy']:.0f}", pred['pos'], xytext=(5, 5), \n                       textcoords='offset points', fontsize=8)\n        \n        # Prey\n        for prey in final_step['prey']:\n            ax.scatter(prey['pos'][0], prey['pos'][1], c='blue', s=80, marker='o', \n                      alpha=0.8, edgecolors='black', label='Prey' if prey['id'] == 0 else "")\n            ax.annotate(f"E:{prey['energy']:.0f}", prey['pos'], xytext=(5, 5), \n                       textcoords='offset points', fontsize=8)\n        \n        # Plants\n        for plant in final_step['plants']:\n            ax.scatter(plant['pos'][0], plant['pos'][1], c='green', s=30, marker='s', \n                      alpha=0.6, label='Plant' if plant['id'] == 0 else "")\n        \n        ax.legend()\n        ax.grid(True, alpha=0.3)\n    \n    def plot_energy_progression(self, ax, showcase_data):\n        """Plot energy progression over time."""\n        energy_log = showcase_data['energy_log']\n        \n        if energy_log['predator']:\n            ax.plot(energy_log['predator'], 'r-', linewidth=2, alpha=0.7, label='Predator')\n        \n        if energy_log['prey']:\n            ax.plot(energy_log['prey'], 'b-', linewidth=2, alpha=0.7, label='Prey')\n        \n        ax.set_xlabel('Step')\n        ax.set_ylabel('Energy')\n        ax.set_title('Energy Progression')\n        ax.legend()\n        ax.grid(True, alpha=0.3)\n    \n    def plot_action_distribution(self, ax, showcase_data):\n        """Plot action distribution."""\n        action_log = showcase_data['action_log']\n        \n        all_actions = set()\n        pred_counts = {}\n        prey_counts = {}\n        \n        if action_log['predator']:\n            for action in action_log['predator']:\n                all_actions.add(action)\n                pred_counts[action] = pred_counts.get(action, 0) + 1\n        \n        if action_log['prey']:\n            for action in action_log['prey']:\n                all_actions.add(action)\n                prey_counts[action] = prey_counts.get(action, 0) + 1\n        \n        if all_actions:\n            actions = list(all_actions)\n            x_pos = np.arange(len(actions))\n            width = 0.35\n            \n            pred_vals = [pred_counts.get(action, 0) for action in actions]\n            prey_vals = [prey_counts.get(action, 0) for action in actions]\n            \n            ax.bar(x_pos - width/2, pred_vals, width, label='Predator', alpha=0.8, color='red')\n            ax.bar(x_pos + width/2, prey_vals, width, label='Prey', alpha=0.8, color='blue')\n            \n            ax.set_xlabel('Action Type')\n            ax.set_ylabel('Frequency')\n            ax.set_title('Action Distribution')\n            ax.set_xticks(x_pos)\n            ax.set_xticklabels(actions, rotation=45)\n            ax.legend()\n    \n    def plot_movement_heatmap(self, ax, showcase_data):\n        """Plot movement heatmap."""\n        position_log = showcase_data['position_log']\n        \n        # Create heatmap grid\n        heatmap = np.zeros((30, 30))\n        \n        for pos_list in position_log.values():\n            for x, y in pos_list:\n                if 0 <= int(x) < 30 and 0 <= int(y) < 30:\n                    heatmap[int(y), int(x)] += 1\n        \n        im = ax.imshow(heatmap, cmap='YlOrRd', alpha=0.7, origin='lower')\n        ax.set_title('Movement Heatmap')\n        ax.set_xlabel('X Position')\n        ax.set_ylabel('Y Position')\n        plt.colorbar(im, ax=ax, label='Visit Frequency')\n    \n    def plot_performance_metrics(self, ax, showcase_data):\n        """Plot performance metrics summary."""\n        analysis = showcase_data['analysis']\n        \n        metrics = ['Avg Predator\\nEnergy', 'Avg Prey\\nEnergy', 'Final\\nPredators', 'Final\\nPrey']\n        values = [analysis['avg_predator_energy'], analysis['avg_prey_energy'], \n                 analysis['final_survival'], analysis['final_prey']]\n        \n        bars = ax.bar(metrics, values, alpha=0.8, color=['red', 'blue', 'darkred', 'darkblue'])\n        ax.set_ylabel('Value')\n        ax.set_title('Performance Summary')\n        \n        # Add value labels\n        for bar, value in zip(bars, values):\n            height = bar.get_height()\n            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,\n                   f'{value:.1f}', ha='center', va='bottom')\n    \n    def run_interactive_showcase(self) -> None:\n        """Run the interactive showcase for all available models."""\n        print("🎭 BioFlux Interactive Model Showcase")\n        print("=" * 60)\n        print("Showcasing trained models with detailed behavioral analysis")\n        print()\n        \n        # Find working algorithms\n        working_algorithms = []\n        \n        # Test Lotka-Volterra (always works)\n        if (self.models_dir / 'lotka_volterra').exists():\n            working_algorithms.append('lotka_volterra')\n        \n        # Test PPO\n        if (self.models_dir / 'ppo' / 'predator_episode_0.pth').exists():\n            working_algorithms.append('ppo')\n        \n        print(f"📋 Available algorithms: {', '.join(working_algorithms)}")\n        print()\n        \n        showcase_results = {}\n        \n        for algorithm in working_algorithms:\n            try:\n                result = self.showcase_algorithm(algorithm, steps=30)\n                if 'error' not in result:\n                    showcase_results[algorithm] = result\n                    \n                    # Create visualization\n                    self.create_showcase_visualization(algorithm, result)\n                    \n                    # Save results\n                    self.save_showcase_results(algorithm, result)\n                    \n                print(f"✅ {algorithm.upper()} showcase completed")\n                time.sleep(1)\n                \n            except Exception as e:\n                logger.error(f"Error showcasing {algorithm}: {e}")\n                continue\n        \n        # Create final comparison if multiple algorithms\n        if len(showcase_results) > 1:\n            self.create_algorithm_comparison(showcase_results)\n        \n        print("\\n🎉 Interactive showcase completed!")\n        print("📊 Check the output/ directory for detailed visualizations")\n        \n        return showcase_results\n    \n    def save_showcase_results(self, algorithm: str, results: Dict) -> None:\n        """Save showcase results."""\n        output_dir = Path("output")\n        output_dir.mkdir(exist_ok=True)\n        \n        # Convert to JSON-serializable format\n        json_results = self.convert_to_json(results)\n        \n        results_path = output_dir / f"{algorithm}_showcase_results.json"\n        with open(results_path, 'w') as f:\n            json.dump(json_results, f, indent=2)\n    \n    def convert_to_json(self, obj):\n        """Convert object to JSON-serializable format."""\n        if isinstance(obj, dict):\n            return {k: self.convert_to_json(v) for k, v in obj.items()}\n        elif isinstance(obj, list):\n            return [self.convert_to_json(v) for v in obj]\n        elif isinstance(obj, (np.integer, np.floating)):\n            return obj.item()\n        elif isinstance(obj, np.ndarray):\n            return obj.tolist()\n        else:\n            return obj\n    \n    def create_algorithm_comparison(self, showcase_results: Dict) -> None:\n        """Create comparison between algorithms."""\n        print("\\n🔍 Algorithm Comparison Analysis")\n        print("-" * 40)\n        \n        fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')\n        \n        algorithms = list(showcase_results.keys())\n        \n        # Energy comparison\n        ax1 = axes[0, 0]\n        pred_energies = [showcase_results[algo]['analysis']['avg_predator_energy'] for algo in algorithms]\n        prey_energies = [showcase_results[algo]['analysis']['avg_prey_energy'] for algo in algorithms]\n        \n        x = np.arange(len(algorithms))\n        width = 0.35\n        \n        ax1.bar(x - width/2, pred_energies, width, label='Predator', alpha=0.8, color='red')\n        ax1.bar(x + width/2, prey_energies, width, label='Prey', alpha=0.8, color='blue')\n        ax1.set_ylabel('Average Energy')\n        ax1.set_title('Energy Management')\n        ax1.set_xticks(x)\n        ax1.set_xticklabels([a.upper() for a in algorithms])\n        ax1.legend()\n        \n        # Survival comparison\n        ax2 = axes[0, 1]\n        pred_survival = [showcase_results[algo]['analysis']['final_survival'] for algo in algorithms]\n        prey_survival = [showcase_results[algo]['analysis']['final_prey'] for algo in algorithms]\n        \n        ax2.bar(x - width/2, pred_survival, width, label='Predator', alpha=0.8, color='red')\n        ax2.bar(x + width/2, prey_survival, width, label='Prey', alpha=0.8, color='blue')\n        ax2.set_ylabel('Final Count')\n        ax2.set_title('Survival Rates')\n        ax2.set_xticks(x)\n        ax2.set_xticklabels([a.upper() for a in algorithms])\n        ax2.legend()\n        \n        # Action diversity (placeholder)\n        ax3 = axes[1, 0]\n        action_diversity = []\n        for algo in algorithms:\n            pred_actions = set(showcase_results[algo]['action_log']['predator'])\n            prey_actions = set(showcase_results[algo]['action_log']['prey'])\n            diversity = len(pred_actions) + len(prey_actions)\n            action_diversity.append(diversity)\n        \n        ax3.bar(algorithms, action_diversity, alpha=0.8, color='green')\n        ax3.set_ylabel('Unique Actions Used')\n        ax3.set_title('Behavioral Diversity')\n        ax3.set_xticklabels([a.upper() for a in algorithms])\n        \n        # Overall performance score\n        ax4 = axes[1, 1]\n        performance_scores = []\n        for algo in algorithms:\n            analysis = showcase_results[algo]['analysis']\n            # Composite score: energy management + survival + diversity\n            score = (\n                (analysis['avg_predator_energy'] + analysis['avg_prey_energy']) / 200 +\n                (analysis['final_survival'] + analysis['final_prey']) / 10 +\n                len(set(showcase_results[algo]['action_log']['predator'] + showcase_results[algo]['action_log']['prey'])) / 10\n            )\n            performance_scores.append(score)\n        \n        bars = ax4.bar(algorithms, performance_scores, alpha=0.8, color='purple')\n        ax4.set_ylabel('Performance Score')\n        ax4.set_title('Overall Performance')\n        ax4.set_xticklabels([a.upper() for a in algorithms])\n        \n        # Add value labels\n        for bar, score in zip(bars, performance_scores):\n            height = bar.get_height()\n            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,\n                    f'{score:.2f}', ha='center', va='bottom')\n        \n        plt.tight_layout()\n        \n        # Save comparison\n        output_dir = Path("output")\n        plot_path = output_dir / "algorithm_comparison.png"\n        plt.savefig(plot_path, dpi=300, bbox_inches='tight')\n        \n        print(f"📊 Comparison saved to {plot_path}")\n        plt.show()\n        \n        # Print ranking\n        sorted_algos = sorted(zip(algorithms, performance_scores), key=lambda x: x[1], reverse=True)\n        print("\\n🏆 Performance Ranking:")\n        for i, (algo, score) in enumerate(sorted_algos):\n            medal = ['🥇', '🥈', '🥉'][i] if i < 3 else f'{i+1}.'\n            print(f"  {medal} {algo.upper()}: {score:.3f}")\n\ndef main():\n    \"\"\"Main showcase function.\"\"\"\n    showcase = InteractiveModelShowcase()\n    results = showcase.run_interactive_showcase()\n    \n    print(\"\\n\" + \"=\"*60)\n    print("🎭 Interactive Model Showcase Complete!")\n    print("📊 Generated visualizations and analysis reports")\n    print("🚀 Your trained models are ready for deployment!")\n\nif __name__ == "__main__":\n    main()
