#!/usr/bin/env python3
"""
Simple RL Demo for BioFlux - Showcases basic functionality of different algorithms.

This script demonstrates the four different agent types:
1. Lotka-Volterra (classical ecological model)
2. Epsilon-Greedy Q-learning  
3. PPO (Proximal Policy Optimization)
4. MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

# Add the parent directory to the path to import bioflux
sys.path.append(str(Path(__file__).parent.parent))

from bioflux.training import (
    TrainingConfig, LotkaVolterraAgent, EpsilonGreedyAgent, 
    PPOAgent, MADDPGWrapper
)
from bioflux.core.environment import Environment
from bioflux.core.agents import Predator, Prey, Plant

def create_demo_environment():
    """Create a simple demo environment."""
    env = Environment()
    env.width = 50
    env.height = 50
    
    # Add a few agents for demonstration
    predator = Predator(speed=2, energy=100, pos_x=25, pos_y=25, age=1)
    prey1 = Prey(speed=3, energy=50, pos_x=10, pos_y=10, age=1)
    prey2 = Prey(speed=3, energy=50, pos_x=40, pos_y=40, age=1)
    
    env.add_predator(predator)
    env.add_prey(prey1)
    env.add_prey(prey2)
    
    # Add some plants
    for _ in range(10):
        plant = Plant(
            energy=10,
            pos_x=np.random.randint(0, 50),
            pos_y=np.random.randint(0, 50)
        )
        env.add_plant(plant)
    
    return env

def get_agent_state(agent, env):
    """Get state observation for an agent."""
    # Always use fallback state creation for demo simplicity
    nearby_prey = []
    nearby_predators = []
    nearby_food = []
    
    for other in env.agents:
        if other != agent:
            distance = np.sqrt((other.pos_x - agent.pos_x)**2 + (other.pos_y - agent.pos_y)**2)
            if distance < 10:  # Within detection range
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
        'temperature': 20,
        'vegetation': 0.5,
        'is_hungry': agent.energy < 30,
        'pos_x': agent.pos_x,
        'pos_y': agent.pos_y,
        'environment_bounds': {'width': env.width, 'height': env.height}
    }

def demo_algorithm(algorithm_name, agent_class, agent_type='predator'):
    """Demonstrate a specific algorithm."""
    print(f"\n🎯 Demonstrating {algorithm_name}")
    print("-" * 40)
    
    # Create environment and config
    env = create_demo_environment()
    config = TrainingConfig(
        algorithm=algorithm_name.lower().replace('-', '_'),
        num_episodes=10,
        max_steps_per_episode=50,
        learning_rate=0.001,
        device="cpu"
    )
    
    # Create agent
    if algorithm_name == "LOTKA-VOLTERRA":
        agent = agent_class(agent_type, config)
    elif algorithm_name == "MADDPG":
        agent = agent_class(agent_type, config)
    else:
        state_dim = 8
        action_dim = 8
        agent = agent_class(agent_type, config, state_dim, action_dim)
    
    # Run simulation
    episode_rewards = []
    
    for episode in range(config.num_episodes):
        env = create_demo_environment()  # Reset environment
        episode_reward = 0
        trajectory = []
        
        for step in range(config.max_steps_per_episode):
            # Find our agent (predator)
            our_agent = None
            for a in env.agents:
                if a.agent_type == agent_type:
                    our_agent = a
                    break
            
            if our_agent is None:
                break
            
            # Get state and action
            state = get_agent_state(our_agent, env)
            
            if algorithm_name == "MADDPG":
                action = agent.select_action(state, agent_id=0)
            else:
                action = agent.select_action(state)
            
            # Apply action
            if 'x' in action and 'y' in action:
                our_agent.pos_x = max(0, min(env.width-1, action['x']))
                our_agent.pos_y = max(0, min(env.height-1, action['y']))
            
            # Calculate simple reward
            reward = 0
            if action.get('action') == 'hunt' and agent_type == 'predator':
                # Check if there's prey nearby
                for other in env.agents:
                    if other.agent_type == 'prey':
                        distance = np.sqrt((other.pos_x - our_agent.pos_x)**2 + (other.pos_y - our_agent.pos_y)**2)
                        if distance < 3:
                            reward += 10
                            break
                else:
                    reward -= 1
            elif action.get('action') == 'forage' and agent_type == 'prey':
                reward += 2
            
            episode_reward += reward
            
            # Store trajectory
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'done': False
            })
        
        # Update agent
        if hasattr(agent, 'update') and trajectory:
            agent.update(trajectory)
        
        episode_rewards.append(episode_reward)
        
        if episode % 2 == 0:
            print(f"  Episode {episode:2d}: Reward = {episode_reward:6.2f}")
    
    avg_reward = np.mean(episode_rewards[-5:])  # Average of last 5 episodes
    print(f"  Final 5-episode average: {avg_reward:.2f}")
    
    return episode_rewards

def run_demo():
    """Run demonstration of all algorithms."""
    print("🚀 BioFlux RL Algorithms Demo")
    print("=" * 50)
    print("This demo showcases four different approaches to agent behavior:")
    print("1. Lotka-Volterra: Classical ecological dynamics")
    print("2. Epsilon-Greedy: Q-learning with exploration")  
    print("3. PPO: Proximal Policy Optimization")
    print("4. MADDPG: Multi-Agent Deep Deterministic Policy Gradient")
    
    # Demo each algorithm
    results = {}
    
    try:
        # Lotka-Volterra
        results['Lotka-Volterra'] = demo_algorithm(
            "LOTKA-VOLTERRA", LotkaVolterraAgent, 'predator'
        )
        
        # Epsilon-Greedy
        results['Epsilon-Greedy'] = demo_algorithm(
            "EPSILON-GREEDY", EpsilonGreedyAgent, 'predator'
        )
        
        # PPO
        results['PPO'] = demo_algorithm(
            "PPO", PPOAgent, 'predator'
        )
        
        # MADDPG
        results['MADDPG'] = demo_algorithm(
            "MADDPG", MADDPGWrapper, 'predator'
        )
        
    except Exception as e:
        print(f"\n⚠️ Some algorithms encountered issues: {e}")
        print("This is normal for a demo - the full training script handles these cases better.")
    
    # Create simple comparison plot
    if len(results) > 1:
        create_demo_plot(results)
    
    print("\n✅ Demo completed!")
    print("\n💡 For a full comparative study, run:")
    print("   python examples/train_comparative.py")

def create_demo_plot(results):
    """Create a simple comparison plot."""
    try:
        plt.figure(figsize=(12, 6))
        
        for algorithm, rewards in results.items():
            plt.plot(rewards, label=algorithm, linewidth=2, alpha=0.8)
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('BioFlux RL Algorithms Demo - Learning Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / "demo_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        print(f"\n📊 Demo plot saved to {plot_path}")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not create plot: {e}")

if __name__ == "__main__":
    run_demo()
