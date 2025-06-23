#!/usr/bin/env python3
"""
BioFlux Real-Time Simulation Demo
=================================

An interactive real-time simulation that demonstrates the trained models
in action with live visualization of the ecosystem dynamics.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import torch
import time
from pathlib import Path
import json
from typing import Dict, List, Any, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from bioflux.training import (
    TrainingConfig, LotkaVolterraAgent, PPOAgent, 
    create_training_environment
)
from bioflux.core.environment import Environment
from bioflux.core.agents import Predator, Prey, Plant

class RealTimeSimulationDemo:
    """Real-time simulation demo with live visualization."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.models_dir = self.base_dir / "models"
        self.output_dir = self.base_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.config = TrainingConfig(
            num_episodes=1,
            max_steps_per_episode=200,
            device="cpu"
        )
        
        # Simulation parameters
        self.env_width = 40
        self.env_height = 40
        self.fps = 10  # Animation frames per second
        
        # Initialize environment and models
        self.env = None
        self.models = {}
        self.animation_data = []
        
        print("🎬 Real-Time Simulation Demo Initialized")
        print(f"Environment Size: {self.env_width}x{self.env_height}")
        print(f"Animation FPS: {self.fps}")
    
    def setup_models(self):
        """Load and setup all available models."""
        print("\n🤖 Loading Models...")
        
        # Lotka-Volterra (always available)
        self.models['Lotka-Volterra'] = {
            'predator': LotkaVolterraAgent('predator', self.config),
            'prey': LotkaVolterraAgent('prey', self.config),
            'color': {'predator': 'red', 'prey': 'blue'},
            'loaded': True
        }
        print("✅ Lotka-Volterra models loaded")
        
        # PPO
        try:
            ppo_predator = PPOAgent('predator', self.config, state_dim=8, action_dim=8)
            ppo_prey = PPOAgent('prey', self.config, state_dim=8, action_dim=8)
            
            # Try to load trained models
            pred_path = self.models_dir / "ppo" / "predator_episode_1000.pth"
            prey_path = self.models_dir / "ppo" / "prey_episode_1000.pth"
            
            if pred_path.exists():
                checkpoint = torch.load(pred_path, map_location='cpu', weights_only=False)
                ppo_predator.actor.load_state_dict(checkpoint['actor'])
                ppo_predator.critic.load_state_dict(checkpoint['critic'])
            
            if prey_path.exists():
                checkpoint = torch.load(prey_path, map_location='cpu', weights_only=False)
                ppo_prey.actor.load_state_dict(checkpoint['actor'])
                ppo_prey.critic.load_state_dict(checkpoint['critic'])
            
            self.models['PPO'] = {
                'predator': ppo_predator,
                'prey': ppo_prey,
                'color': {'predator': 'darkred', 'prey': 'darkblue'},
                'loaded': True
            }
            print("✅ PPO models loaded")
            
        except Exception as e:
            print(f"⚠️ PPO models not available: {e}")
        
        print(f"📋 Available models: {list(self.models.keys())}")
    
    def create_simulation_environment(self, scenario='balanced'):
        """Create environment for simulation."""
        env = Environment()
        env.width = self.env_width
        env.height = self.env_height
        
        # Different scenarios
        if scenario == 'balanced':
            num_predators, num_prey, num_plants = 3, 5, 20
        elif scenario == 'predator_advantage':
            num_predators, num_prey, num_plants = 6, 3, 15
        elif scenario == 'prey_advantage':
            num_predators, num_prey, num_plants = 2, 8, 25
        else:
            num_predators, num_prey, num_plants = 3, 5, 20
        
        # Add predators
        for i in range(num_predators):
            predator = Predator(
                speed=2,
                energy=100,
                pos_x=np.random.randint(5, self.env_width - 5),
                pos_y=np.random.randint(5, self.env_height - 5),
                age=1
            )
            predator.id = f"pred_{i}"
            env.add_predator(predator)
        
        # Add prey
        for i in range(num_prey):
            prey = Prey(
                speed=3,
                energy=60,
                pos_x=np.random.randint(5, self.env_width - 5),
                pos_y=np.random.randint(5, self.env_height - 5),
                age=1
            )
            prey.id = f"prey_{i}"
            env.add_prey(prey)
        
        # Add plants
        for i in range(num_plants):
            plant = Plant(
                energy=15,
                pos_x=np.random.randint(0, self.env_width),
                pos_y=np.random.randint(0, self.env_height)
            )
            plant.id = f"plant_{i}"
            env.add_plant(plant)
        
        return env
    
    def get_agent_state(self, agent, env):
        """Get state observation for agent."""
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
        
        return {
            'energy': agent.energy,
            'age': getattr(agent, 'age', 1),
            'nearby_prey': nearby_prey,
            'nearby_predators': nearby_predators,
            'nearby_food': nearby_food,
            'pos_x': agent.pos_x,
            'pos_y': agent.pos_y,
            'environment_bounds': {'width': env.width, 'height': env.height}
        }
    
    def apply_action(self, agent, action, env):
        """Apply action to agent."""
        if 'x' in action and 'y' in action:
            # Move agent
            new_x = max(0, min(env.width - 1, int(action['x'])))
            new_y = max(0, min(env.height - 1, int(action['y'])))
            agent.pos_x = new_x
            agent.pos_y = new_y
        
        # Energy management
        action_type = action.get('action', 'stay')
        
        if action_type in ['move', 'hunt', 'explore']:
            agent.energy -= 1  # Movement cost
        elif action_type == 'forage':
            # Check for nearby plants
            for plant in env.plants:
                if plant.is_alive:
                    distance = np.sqrt((plant.pos_x - agent.pos_x)**2 + (plant.pos_y - agent.pos_y)**2)
                    if distance < 2:
                        agent.energy += 5
                        plant.energy -= 5
                        if plant.energy <= 0:
                            plant.is_alive = False
                        break
        
        # Natural energy decay
        agent.energy -= 0.1
        
        # Death check
        if agent.energy <= 0:
            agent.is_alive = False
    
    def move_prey_randomly(self, env):
        """Add some random movement to prey for dynamic environment."""
        for prey in env.prey:
            if prey.is_alive and np.random.random() < 0.3:  # 30% chance of random movement
                dx = np.random.randint(-1, 2)
                dy = np.random.randint(-1, 2)
                
                new_x = max(0, min(env.width - 1, prey.pos_x + dx))
                new_y = max(0, min(env.height - 1, prey.pos_y + dy))
                
                prey.pos_x = new_x
                prey.pos_y = new_y
    
    def run_simulation_step(self, algorithm, step):
        """Run one simulation step."""
        if algorithm not in self.models:
            return
        
        model = self.models[algorithm]
        
        # Get active agents
        active_predators = [p for p in self.env.predators if p.is_alive]
        active_prey = [p for p in self.env.prey if p.is_alive]
        
        step_data = {
            'step': step,
            'algorithm': algorithm,
            'predators': [],
            'prey': [],
            'plants': [],
            'actions': []
        }
        
        # Process predators
        for predator in active_predators:
            state = self.get_agent_state(predator, self.env)
            
            try:
                action = model['predator'].select_action(state)
            except Exception as e:
                # Fallback action
                action = {'action': 'stay', 'x': predator.pos_x, 'y': predator.pos_y}
            
            # Apply action
            self.apply_action(predator, action, self.env)
            
            # Record data
            step_data['predators'].append({
                'id': predator.id,
                'pos': (predator.pos_x, predator.pos_y),
                'energy': predator.energy,
                'action': action.get('action', 'stay'),
                'alive': predator.is_alive
            })
            
            step_data['actions'].append({
                'agent_id': predator.id,
                'agent_type': 'predator',
                'action': action.get('action', 'stay'),
                'pos': (predator.pos_x, predator.pos_y)
            })
        
        # Process prey
        for prey in active_prey:
            state = self.get_agent_state(prey, self.env)
            
            try:
                action = model['prey'].select_action(state)
            except Exception as e:
                # Fallback action
                action = {'action': 'stay', 'x': prey.pos_x, 'y': prey.pos_y}
            
            # Apply action
            self.apply_action(prey, action, self.env)
            
            # Record data
            step_data['prey'].append({
                'id': prey.id,
                'pos': (prey.pos_x, prey.pos_y),
                'energy': prey.energy,
                'action': action.get('action', 'stay'),
                'alive': prey.is_alive
            })
            
            step_data['actions'].append({
                'agent_id': prey.id,
                'agent_type': 'prey',
                'action': action.get('action', 'stay'),
                'pos': (prey.pos_x, prey.pos_y)
            })
        
        # Record plants
        for plant in self.env.plants:
            if plant.is_alive:
                step_data['plants'].append({
                    'id': plant.id,
                    'pos': (plant.pos_x, plant.pos_y),
                    'energy': plant.energy
                })
        
        # Add some random movement
        self.move_prey_randomly(self.env)
        
        return step_data
    
    def create_simulation_plot(self, algorithm, max_steps=100):
        """Create and run a simulation with live plotting."""
        print(f"\n🎬 Running Real-Time Simulation: {algorithm}")
        print("-" * 50)
        
        # Setup environment
        self.env = self.create_simulation_environment('balanced')
        
        # Setup plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'BioFlux Real-Time Simulation - {algorithm}', fontsize=16, fontweight='bold')
        
        # Initialize data storage
        simulation_data = []
        energy_history = {'predator': [], 'prey': []}
        population_history = {'predator': [], 'prey': []}
        
        # Animation function
        def update_plot(frame):
            if frame >= max_steps:
                return
            
            # Run simulation step
            step_data = self.run_simulation_step(algorithm, frame)
            if step_data:
                simulation_data.append(step_data)
            
            # Clear plots
            ax1.clear()
            ax2.clear()
            
            # Plot 1: Environment State
            ax1.set_xlim(0, self.env_width)
            ax1.set_ylim(0, self.env_height)
            ax1.set_title(f'Ecosystem State - Step {frame}')
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.grid(True, alpha=0.3)
            
            # Draw plants
            for plant_data in step_data['plants'] if step_data else []:
                pos = plant_data['pos']
                ax1.scatter(pos[0], pos[1], c='green', s=30, marker='s', alpha=0.6, label='Plants' if plant_data['id'] == 'plant_0' else "")
            
            # Draw prey
            for prey_data in step_data['prey'] if step_data else []:
                if prey_data['alive']:
                    pos = prey_data['pos']
                    energy = prey_data['energy']
                    color_intensity = max(0.3, min(1.0, energy / 60))
                    ax1.scatter(pos[0], pos[1], c='blue', s=80, marker='o', alpha=color_intensity, 
                               edgecolors='darkblue', linewidth=1, label='Prey' if prey_data['id'] == 'prey_0' else "")
                    # Energy label
                    ax1.annotate(f'{energy:.0f}', pos, xytext=(3, 3), textcoords='offset points', fontsize=8)
            
            # Draw predators
            for pred_data in step_data['predators'] if step_data else []:
                if pred_data['alive']:
                    pos = pred_data['pos']
                    energy = pred_data['energy']
                    color_intensity = max(0.3, min(1.0, energy / 100))
                    ax1.scatter(pos[0], pos[1], c='red', s=100, marker='^', alpha=color_intensity,
                               edgecolors='darkred', linewidth=1, label='Predators' if pred_data['id'] == 'pred_0' else "")
                    # Energy label
                    ax1.annotate(f'{energy:.0f}', pos, xytext=(3, 3), textcoords='offset points', fontsize=8)
            
            # Add legend
            if frame == 0:
                ax1.legend(loc='upper right')
            
            # Plot 2: Population and Energy Dynamics
            if simulation_data:
                # Update history
                current_pred_energy = np.mean([p['energy'] for p in step_data['predators'] if p['alive']]) if step_data['predators'] else 0
                current_prey_energy = np.mean([p['energy'] for p in step_data['prey'] if p['alive']]) if step_data['prey'] else 0
                current_pred_pop = len([p for p in step_data['predators'] if p['alive']]) if step_data['predators'] else 0
                current_prey_pop = len([p for p in step_data['prey'] if p['alive']]) if step_data['prey'] else 0
                
                energy_history['predator'].append(current_pred_energy)
                energy_history['prey'].append(current_prey_energy)
                population_history['predator'].append(current_pred_pop)
                population_history['prey'].append(current_prey_pop)
                
                # Plot energy over time
                steps = range(len(energy_history['predator']))
                ax2.plot(steps, energy_history['predator'], 'r-', linewidth=2, alpha=0.8, label='Predator Energy')
                ax2.plot(steps, energy_history['prey'], 'b-', linewidth=2, alpha=0.8, label='Prey Energy')
                
                # Plot population on secondary axis
                ax2_twin = ax2.twinx()
                ax2_twin.plot(steps, population_history['predator'], 'r--', linewidth=2, alpha=0.6, label='Predator Count')
                ax2_twin.plot(steps, population_history['prey'], 'b--', linewidth=2, alpha=0.6, label='Prey Count')
                
                ax2.set_xlabel('Simulation Step')
                ax2.set_ylabel('Average Energy', color='black')
                ax2_twin.set_ylabel('Population Count', color='gray')
                ax2.set_title('Population & Energy Dynamics')
                ax2.legend(loc='upper left')
                ax2_twin.legend(loc='upper right')
                ax2.grid(True, alpha=0.3)
            
            # Status info
            alive_predators = len([p for p in step_data['predators'] if p['alive']]) if step_data and step_data['predators'] else 0
            alive_prey = len([p for p in step_data['prey'] if p['alive']]) if step_data and step_data['prey'] else 0
            alive_plants = len(step_data['plants']) if step_data and step_data['plants'] else 0
            
            fig.suptitle(f'{algorithm} Simulation - Step {frame} | Predators: {alive_predators} | Prey: {alive_prey} | Plants: {alive_plants}', 
                        fontsize=14, fontweight='bold')
        
        # Create and run animation
        print(f"🎥 Starting {max_steps}-step simulation...")
        anim = animation.FuncAnimation(fig, update_plot, frames=max_steps, interval=1000//self.fps, repeat=False)
        
        # Save animation data
        self.animation_data = simulation_data
        
        # Show plot
        plt.tight_layout()
        plt.show()
        
        # Save final frame
        plot_path = self.output_dir / f"{algorithm.lower()}_simulation_demo.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Final simulation frame saved to {plot_path}")
        
        return simulation_data
    
    def save_simulation_data(self, algorithm, simulation_data):
        """Save simulation data to file."""
        if not simulation_data:
            return
        
        # Calculate summary statistics
        summary = {
            'algorithm': algorithm,
            'total_steps': len(simulation_data),
            'final_populations': {
                'predators': len([p for p in simulation_data[-1]['predators'] if p['alive']]),
                'prey': len([p for p in simulation_data[-1]['prey'] if p['alive']]),
                'plants': len(simulation_data[-1]['plants'])
            },
            'average_energies': {
                'predator': np.mean([np.mean([p['energy'] for p in step['predators'] if p['alive']]) 
                                   for step in simulation_data if step['predators']]),
                'prey': np.mean([np.mean([p['energy'] for p in step['prey'] if p['alive']]) 
                               for step in simulation_data if step['prey']])
            },
            'action_counts': {}
        }
        
        # Count actions
        all_actions = []
        for step in simulation_data:
            for action in step['actions']:
                all_actions.append(action['action'])
        
        for action in set(all_actions):
            summary['action_counts'][action] = all_actions.count(action)
        
        # Save data
        data_path = self.output_dir / f"{algorithm.lower()}_simulation_data.json"
        full_data = {
            'summary': summary,
            'simulation_steps': simulation_data
        }
        
        with open(data_path, 'w') as f:
            json.dump(full_data, f, indent=2, default=str)
        
        print(f"💾 Simulation data saved to {data_path}")
        return summary
    
    def run_comprehensive_simulation_demo(self):
        """Run simulation demos for all available models."""
        print("🎬 BioFlux Real-Time Simulation Demo")
        print("=" * 50)
        
        # Setup models
        self.setup_models()
        
        if not self.models:
            print("❌ No models available for simulation")
            return
        
        results = {}
        
        # Run simulations for each model
        for algorithm in self.models.keys():
            try:
                print(f"\n{'='*60}")
                print(f"🎯 Simulating {algorithm} Model")
                print(f"{'='*60}")
                
                # Run simulation
                simulation_data = self.create_simulation_plot(algorithm, max_steps=50)
                
                # Save results
                summary = self.save_simulation_data(algorithm, simulation_data)
                results[algorithm] = summary
                
                print(f"✅ {algorithm} simulation completed successfully!")
                
                # Brief pause between simulations
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error running {algorithm} simulation: {e}")
                continue
        
        # Final summary
        self.print_simulation_summary(results)
        
        return results
    
    def print_simulation_summary(self, results):
        """Print comprehensive simulation summary."""
        print("\n" + "="*60)
        print("🏆 SIMULATION DEMO RESULTS")
        print("="*60)
        
        if not results:
            print("❌ No simulations completed successfully")
            return
        
        print(f"📊 Completed {len(results)} successful simulations")
        print()
        
        for algorithm, summary in results.items():
            print(f"🤖 {algorithm}:")
            print(f"   Steps Simulated: {summary['total_steps']}")
            print(f"   Final Populations:")
            print(f"     Predators: {summary['final_populations']['predators']}")
            print(f"     Prey: {summary['final_populations']['prey']}")
            print(f"     Plants: {summary['final_populations']['plants']}")
            print(f"   Average Energies:")
            print(f"     Predator: {summary['average_energies']['predator']:.1f}")
            print(f"     Prey: {summary['average_energies']['prey']:.1f}")
            print(f"   Top Actions: {list(summary['action_counts'].keys())[:3]}")
            print()
        
        print("📁 Generated Files:")
        print(f"   • Simulation plots: output/*_simulation_demo.png")
        print(f"   • Simulation data: output/*_simulation_data.json")
        print()
        print("🎉 Real-Time Simulation Demo Complete!")
        print("🚀 Your models have been successfully demonstrated in action!")

def main():
    """Main simulation demo function."""
    demo = RealTimeSimulationDemo()
    results = demo.run_comprehensive_simulation_demo()
    
    print("\n" + "="*60)
    print("🎭 BioFlux Real-Time Simulation Demo Complete!")
    print("📊 All available models have been simulated and visualized")
    print("🚀 Your ecosystem AI is ready for real-world applications!")
    print("="*60)

if __name__ == "__main__":
    main()
