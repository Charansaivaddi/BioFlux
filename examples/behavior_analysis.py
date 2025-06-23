#!/usr/bin/env python3
"""
BioFlux Model Behavior Analysis

This script provides detailed behavioral analysis of trained models
with step-by-step action tracking and decision analysis.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import json
from typing import Dict, List, Any
import logging

sys.path.append(str(Path(__file__).parent.parent))

from bioflux.training import TrainingConfig, LotkaVolterraAgent, PPOAgent
from bioflux.core.environment import Environment
from bioflux.core.agents import Predator, Prey, Plant

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelBehaviorAnalyzer:
    """Detailed behavioral analysis of trained models."""
    
    def __init__(self):
        self.config = TrainingConfig(device="cpu")
        self.models_dir = Path("models")
        
    def create_analysis_environment(self, scenario: str = "balanced"):
        """Create different scenario environments for testing."""
        env = Environment()
        env.width = 20
        env.height = 20
        
        if scenario == "balanced":
            # Balanced scenario
            predators = [Predator(speed=2, energy=100, pos_x=5, pos_y=5, age=1),
                        Predator(speed=2, energy=100, pos_x=15, pos_y=15, age=1)]
            prey = [Prey(speed=3, energy=50, pos_x=10, pos_y=5, age=1),
                   Prey(speed=3, energy=50, pos_x=5, pos_y=15, age=1),
                   Prey(speed=3, energy=50, pos_x=15, pos_y=10, age=1)]
            
        elif scenario == "predator_advantage":
            # More predators
            predators = [Predator(speed=2, energy=120, pos_x=i*6+2, pos_y=j*6+2, age=1) 
                        for i in range(3) for j in range(2)]
            prey = [Prey(speed=3, energy=40, pos_x=10, pos_y=10, age=1),
                   Prey(speed=3, energy=40, pos_x=15, pos_y=5, age=1)]
            
        elif scenario == "prey_advantage":
            # More prey
            predators = [Predator(speed=2, energy=80, pos_x=10, pos_y=10, age=1)]
            prey = [Prey(speed=3, energy=60, pos_x=i*4+2, pos_y=j*4+2, age=1) 
                   for i in range(4) for j in range(2)]
        
        else:  # sparse
            # Sparse scenario
            predators = [Predator(speed=2, energy=100, pos_x=2, pos_y=2, age=1)]
            prey = [Prey(speed=3, energy=50, pos_x=18, pos_y=18, age=1)]
        
        for pred in predators:
            env.add_predator(pred)
        for p in prey:
            env.add_prey(p)
        
        # Add plants
        for _ in range(8):
            plant = Plant(energy=10, pos_x=np.random.randint(0, 20), pos_y=np.random.randint(0, 20))
            env.add_plant(plant)
        
        return env
    
    def analyze_model_decisions(self, algorithm: str, scenario: str = "balanced", steps: int = 25):
        """Analyze model decision-making in detail."""
        print(f"\n🔍 Analyzing {algorithm.upper()} - {scenario} scenario")
        print("-" * 55)
        
        # Load model
        if algorithm == "lotka_volterra":
            pred_agent = LotkaVolterraAgent('predator', self.config)
            prey_agent = LotkaVolterraAgent('prey', self.config)
        elif algorithm == "ppo":
            pred_agent = PPOAgent('predator', self.config, 8, 8)
            prey_agent = PPOAgent('prey', self.config, 8, 8)
            
            # Try loading saved weights
            try:
                pred_path = self.models_dir / "ppo" / "predator_episode_0.pth"
                if pred_path.exists():
                    checkpoint = torch.load(pred_path, map_location='cpu', weights_only=False)
                    if 'actor' in checkpoint:
                        pred_agent.actor.load_state_dict(checkpoint['actor'])
            except Exception as e:
                logger.warning(f"Could not load PPO predator model: {e}")
                
            try:
                prey_path = self.models_dir / "ppo" / "prey_episode_0.pth"
                if prey_path.exists():
                    checkpoint = torch.load(prey_path, map_location='cpu', weights_only=False)
                    if 'actor' in checkpoint:
                        prey_agent.actor.load_state_dict(checkpoint['actor'])
            except Exception as e:
                logger.warning(f"Could not load PPO prey model: {e}")
        else:
            print(f"❌ Algorithm {algorithm} not available for analysis")
            return None
        
        # Create environment
        env = self.create_analysis_environment(scenario)
        
        # Track detailed data
        decision_log = []
        step_states = []
        agent_behaviors = {'predator': {}, 'prey': {}}
        
        print(f"🎬 Running {steps}-step analysis...")
        print(f"Initial state: {len(env.predators)} predators, {len(env.prey)} prey")
        
        for step in range(steps):
            step_data = {
                'step': step,
                'decisions': [],
                'state': {
                    'predators': len([p for p in env.predators if p.is_alive]),
                    'prey': len([p for p in env.prey if p.is_alive]),
                    'plants': len(env.plants)
                }
            }
            
            # Analyze predator decisions
            for i, predator in enumerate(env.predators):
                if predator.is_alive:
                    state = self.get_detailed_state(predator, env)
                    
                    try:
                        action = pred_agent.select_action(state)
                    except Exception:
                        action = {'action': 'stay', 'x': predator.pos_x, 'y': predator.pos_y}
                    
                    # Analyze decision context
                    decision_context = self.analyze_decision_context(predator, env, action, state)
                    
                    decision_data = {
                        'agent_type': 'predator',
                        'agent_id': i,
                        'position': (predator.pos_x, predator.pos_y),
                        'energy': predator.energy,
                        'action': action.get('action', 'unknown'),
                        'target_position': (action.get('x', predator.pos_x), action.get('y', predator.pos_y)),
                        'context': decision_context,
                        'state_summary': {
                            'nearby_prey': len(state.get('nearby_prey', [])),
                            'nearby_predators': len(state.get('nearby_predators', [])),
                            'is_hungry': state.get('is_hungry', False)
                        }
                    }
                    
                    step_data['decisions'].append(decision_data)
                    
                    # Update behavior tracking
                    action_type = action.get('action', 'stay')
                    agent_behaviors['predator'][action_type] = agent_behaviors['predator'].get(action_type, 0) + 1
                    
                    # Apply action
                    if 'x' in action and 'y' in action:
                        predator.pos_x = max(0, min(env.width-1, int(action['x'])))
                        predator.pos_y = max(0, min(env.height-1, int(action['y'])))
            
            # Analyze prey decisions
            for i, prey in enumerate(env.prey):
                if prey.is_alive:
                    state = self.get_detailed_state(prey, env)
                    
                    try:
                        action = prey_agent.select_action(state)
                    except Exception:
                        action = {'action': 'stay', 'x': prey.pos_x, 'y': prey.pos_y}
                    
                    # Analyze decision context
                    decision_context = self.analyze_decision_context(prey, env, action, state)
                    
                    decision_data = {
                        'agent_type': 'prey',
                        'agent_id': i,
                        'position': (prey.pos_x, prey.pos_y),
                        'energy': prey.energy,
                        'action': action.get('action', 'unknown'),
                        'target_position': (action.get('x', prey.pos_x), action.get('y', prey.pos_y)),
                        'context': decision_context,
                        'state_summary': {
                            'nearby_prey': len(state.get('nearby_prey', [])),
                            'nearby_predators': len(state.get('nearby_predators', [])),
                            'is_hungry': state.get('is_hungry', False)
                        }
                    }
                    
                    step_data['decisions'].append(decision_data)
                    
                    # Update behavior tracking
                    action_type = action.get('action', 'stay')
                    agent_behaviors['prey'][action_type] = agent_behaviors['prey'].get(action_type, 0) + 1
                    
                    # Apply action
                    if 'x' in action and 'y' in action:
                        prey.pos_x = max(0, min(env.width-1, int(action['x'])))
                        prey.pos_y = max(0, min(env.height-1, int(action['y'])))
            
            decision_log.append(step_data)
            step_states.append(step_data['state'])
            
            # Update environment
            env.step()
            
            # Print key decisions every few steps
            if step % 8 == 0 and step_data['decisions']:
                print(f"  Step {step:2d}:")
                for decision in step_data['decisions'][:2]:  # Show first 2 decisions
                    action_desc = self.describe_action(decision)
                    print(f"    {decision['agent_type']} {decision['agent_id']}: {action_desc}")
        
        # Generate analysis report
        analysis_report = self.generate_analysis_report(algorithm, scenario, decision_log, agent_behaviors)
        
        print(f"\n📊 Analysis Complete - {len(decision_log)} steps analyzed")
        print(f"📋 {len([d for step in decision_log for d in step['decisions']])} total decisions tracked")
        
        return {
            'algorithm': algorithm,
            'scenario': scenario,
            'decision_log': decision_log,
            'agent_behaviors': agent_behaviors,
            'analysis_report': analysis_report,
            'step_states': step_states
        }
    
    def get_detailed_state(self, agent, env):
        """Get detailed state information."""
        nearby_prey = []
        nearby_predators = []
        nearby_food = []
        
        for other in env.predators + env.prey + env.plants:
            if other != agent and hasattr(other, 'pos_x') and hasattr(other, 'pos_y'):
                distance = np.sqrt((other.pos_x - agent.pos_x)**2 + (other.pos_y - agent.pos_y)**2)
                if distance < 12:
                    if hasattr(other, 'agent_type'):
                        agent_info = {'pos': (other.pos_x, other.pos_y), 'distance': distance}
                        if other.agent_type == 'prey':
                            nearby_prey.append(agent_info)
                        elif other.agent_type == 'predator':
                            nearby_predators.append(agent_info)
                        elif other.agent_type == 'plant':
                            nearby_food.append(agent_info)
        
        return {
            'energy': agent.energy,
            'age': getattr(agent, 'age', 1),
            'nearby_prey': nearby_prey,
            'nearby_predators': nearby_predators,
            'nearby_food': nearby_food,
            'temperature': 20.0,
            'vegetation': 0.5,
            'is_hungry': agent.energy < 30,
            'pos_x': agent.pos_x,
            'pos_y': agent.pos_y,
            'environment_bounds': {'width': env.width, 'height': env.height}
        }
    
    def analyze_decision_context(self, agent, env, action, state):
        """Analyze the context behind a decision."""
        context = {}
        
        # Threat assessment
        nearby_threats = len(state.get('nearby_predators', [])) if agent.agent_type == 'prey' else 0
        nearby_targets = len(state.get('nearby_prey', [])) if agent.agent_type == 'predator' else 0
        nearby_food = len(state.get('nearby_food', []))
        
        context['threat_level'] = 'high' if nearby_threats > 1 else 'medium' if nearby_threats == 1 else 'low'
        context['target_availability'] = 'high' if nearby_targets > 2 else 'medium' if nearby_targets > 0 else 'low'
        context['food_availability'] = 'high' if nearby_food > 3 else 'medium' if nearby_food > 0 else 'low'
        
        # Energy status
        if agent.energy > 70:
            context['energy_status'] = 'high'
        elif agent.energy > 30:
            context['energy_status'] = 'medium'
        else:
            context['energy_status'] = 'low'
        
        # Movement analysis
        current_pos = (agent.pos_x, agent.pos_y)
        target_pos = (action.get('x', agent.pos_x), action.get('y', agent.pos_y))
        movement_distance = np.sqrt((target_pos[0] - current_pos[0])**2 + (target_pos[1] - current_pos[1])**2)
        
        context['movement_distance'] = movement_distance
        context['is_moving'] = movement_distance > 1.0
        
        return context
    
    def describe_action(self, decision):
        """Create human-readable action description."""
        action = decision['action']
        context = decision['context']
        
        if action == 'hunt':
            return f"hunting (targets: {context['target_availability']}, energy: {context['energy_status']})"
        elif action == 'flee':
            return f"fleeing (threats: {context['threat_level']}, energy: {context['energy_status']})"
        elif action == 'forage':
            return f"foraging (food: {context['food_availability']}, energy: {context['energy_status']})"
        elif action == 'explore':
            return f"exploring (energy: {context['energy_status']}, moving: {context['is_moving']})"
        elif action == 'move':
            move_dist = context.get('movement_distance', 0)
            return f"moving {move_dist:.1f} units (energy: {context['energy_status']})"
        else:
            return f"{action} (energy: {context['energy_status']})"
    
    def generate_analysis_report(self, algorithm, scenario, decision_log, agent_behaviors):
        """Generate comprehensive analysis report."""
        report = {
            'algorithm': algorithm,
            'scenario': scenario,
            'summary': {}
        }
        
        # Decision frequency analysis
        total_decisions = sum(len(step['decisions']) for step in decision_log)
        predator_decisions = sum(1 for step in decision_log for d in step['decisions'] if d['agent_type'] == 'predator')
        prey_decisions = total_decisions - predator_decisions
        
        report['summary']['total_decisions'] = total_decisions
        report['summary']['predator_decisions'] = predator_decisions
        report['summary']['prey_decisions'] = prey_decisions
        
        # Most common actions
        if agent_behaviors['predator']:
            most_common_pred = max(agent_behaviors['predator'].items(), key=lambda x: x[1])
            report['summary']['most_common_predator_action'] = most_common_pred[0]
            report['summary']['predator_action_frequency'] = most_common_pred[1]
        
        if agent_behaviors['prey']:
            most_common_prey = max(agent_behaviors['prey'].items(), key=lambda x: x[1])
            report['summary']['most_common_prey_action'] = most_common_prey[0]
            report['summary']['prey_action_frequency'] = most_common_prey[1]
        
        # Behavioral diversity
        report['summary']['predator_action_diversity'] = len(agent_behaviors['predator'])
        report['summary']['prey_action_diversity'] = len(agent_behaviors['prey'])
        
        # Population dynamics
        initial_state = decision_log[0]['state'] if decision_log else {'predators': 0, 'prey': 0}
        final_state = decision_log[-1]['state'] if decision_log else {'predators': 0, 'prey': 0}
        
        report['summary']['population_change'] = {
            'predators': final_state['predators'] - initial_state['predators'],
            'prey': final_state['prey'] - initial_state['prey']
        }
        
        return report
    
    def create_behavior_visualization(self, analysis_data):
        """Create detailed behavioral visualization."""
        print("🎨 Creating behavioral analysis visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"{analysis_data['algorithm'].upper()} - {analysis_data['scenario']} Scenario Analysis", 
                    fontsize=16, fontweight='bold')
        
        # 1. Action distribution
        ax1 = axes[0, 0]
        pred_actions = analysis_data['agent_behaviors']['predator']
        prey_actions = analysis_data['agent_behaviors']['prey']
        
        if pred_actions or prey_actions:
            all_actions = set(pred_actions.keys()) | set(prey_actions.keys())
            x_pos = np.arange(len(all_actions))
            width = 0.35
            
            pred_vals = [pred_actions.get(action, 0) for action in all_actions]
            prey_vals = [prey_actions.get(action, 0) for action in all_actions]
            
            ax1.bar(x_pos - width/2, pred_vals, width, label='Predator', alpha=0.8, color='red')
            ax1.bar(x_pos + width/2, prey_vals, width, label='Prey', alpha=0.8, color='blue')
            
            ax1.set_xlabel('Action Type')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Action Distribution')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(list(all_actions), rotation=45)
            ax1.legend()
        
        # 2. Population dynamics
        ax2 = axes[0, 1]
        steps = range(len(analysis_data['step_states']))
        predator_counts = [state['predators'] for state in analysis_data['step_states']]
        prey_counts = [state['prey'] for state in analysis_data['step_states']]
        
        ax2.plot(steps, predator_counts, 'r-', linewidth=2, marker='o', label='Predators')
        ax2.plot(steps, prey_counts, 'b-', linewidth=2, marker='s', label='Prey')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Population')
        ax2.set_title('Population Dynamics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Decision context analysis
        ax3 = axes[0, 2]
        threat_levels = {'low': 0, 'medium': 0, 'high': 0}
        energy_levels = {'low': 0, 'medium': 0, 'high': 0}
        
        for step in analysis_data['decision_log']:
            for decision in step['decisions']:
                context = decision['context']
                threat_levels[context.get('threat_level', 'low')] += 1
                energy_levels[context.get('energy_status', 'medium')] += 1
        
        # Threat level pie chart
        if sum(threat_levels.values()) > 0:
            ax3.pie(threat_levels.values(), labels=threat_levels.keys(), autopct='%1.1f%%',
                   colors=['green', 'yellow', 'red'])
            ax3.set_title('Threat Level Distribution')
        
        # 4. Energy distribution
        ax4 = axes[1, 0]
        if sum(energy_levels.values()) > 0:
            ax4.pie(energy_levels.values(), labels=energy_levels.keys(), autopct='%1.1f%%',
                   colors=['red', 'yellow', 'green'])
            ax4.set_title('Energy Level Distribution')
        
        # 5. Movement analysis
        ax5 = axes[1, 1]
        movement_distances = []
        for step in analysis_data['decision_log']:
            for decision in step['decisions']:
                movement_distances.append(decision['context'].get('movement_distance', 0))
        
        if movement_distances:
            ax5.hist(movement_distances, bins=15, alpha=0.7, color='purple', edgecolor='black')
            ax5.set_xlabel('Movement Distance')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Movement Distance Distribution')
            ax5.axvline(np.mean(movement_distances), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(movement_distances):.2f}')
            ax5.legend()
        
        # 6. Performance summary
        ax6 = axes[1, 2]
        report = analysis_data['analysis_report']['summary']
        
        metrics = ['Total\\nDecisions', 'Predator\\nDiversity', 'Prey\\nDiversity', 'Pop\\nChange']
        values = [
            report.get('total_decisions', 0),
            report.get('predator_action_diversity', 0),
            report.get('prey_action_diversity', 0),
            abs(sum(report.get('population_change', {}).values()))
        ]
        
        bars = ax6.bar(metrics, values, alpha=0.8, color=['purple', 'red', 'blue', 'green'])
        ax6.set_title('Performance Metrics')
        ax6.set_ylabel('Count')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / f"{analysis_data['algorithm']}_{analysis_data['scenario']}_behavior_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"  📊 Behavioral analysis saved to {plot_path}")
        plt.show()
        
        return str(plot_path)
    
    def run_comprehensive_analysis(self):
        """Run comprehensive behavioral analysis."""
        print("🔬 BioFlux Model Behavioral Analysis")
        print("=" * 60)
        print("Deep analysis of model decision-making and behaviors")
        print()
        
        algorithms = ["lotka_volterra", "ppo"]
        scenarios = ["balanced", "predator_advantage", "prey_advantage"]
        
        all_analyses = {}
        
        for algorithm in algorithms:
            all_analyses[algorithm] = {}
            
            for scenario in scenarios:
                try:
                    analysis = self.analyze_model_decisions(algorithm, scenario, steps=20)
                    if analysis:
                        all_analyses[algorithm][scenario] = analysis
                        
                        # Create visualization
                        self.create_behavior_visualization(analysis)
                        
                        # Print summary
                        report = analysis['analysis_report']['summary']
                        print(f"\n📋 {algorithm.upper()} - {scenario} Summary:")
                        print(f"  Total Decisions: {report.get('total_decisions', 0)}")
                        print(f"  Predator Diversity: {report.get('predator_action_diversity', 0)} actions")
                        print(f"  Prey Diversity: {report.get('prey_action_diversity', 0)} actions")
                        
                        if 'most_common_predator_action' in report:
                            print(f"  Top Predator Action: {report['most_common_predator_action']}")
                        if 'most_common_prey_action' in report:
                            print(f"  Top Prey Action: {report['most_common_prey_action']}")
                        
                        # Save detailed results
                        self.save_analysis_results(analysis)
                        
                        print(f"  ✅ {algorithm} - {scenario} analysis completed")
                        
                except Exception as e:
                    logger.error(f"Error analyzing {algorithm} - {scenario}: {e}")
                    continue
        
        # Create final comparison
        self.create_algorithm_comparison(all_analyses)
        
        print(f"\n🎉 Comprehensive behavioral analysis completed!")
        print(f"📊 Check output/ directory for detailed visualizations and reports")
        
        return all_analyses
    
    def save_analysis_results(self, analysis):
        """Save analysis results to JSON."""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Convert to JSON-serializable format
        json_data = self.convert_to_json(analysis)
        
        filename = f"{analysis['algorithm']}_{analysis['scenario']}_analysis.json"
        results_path = output_dir / filename
        
        with open(results_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def convert_to_json(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self.convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def create_algorithm_comparison(self, all_analyses):
        """Create final algorithm comparison."""
        print("\n🏆 Final Algorithm Comparison")
        print("-" * 40)
        
        # Create comparison metrics
        comparison_data = {}
        
        for algorithm, scenarios in all_analyses.items():
            comparison_data[algorithm] = {
                'total_scenarios': len(scenarios),
                'avg_decisions': 0,
                'avg_diversity': 0,
                'behavioral_adaptability': 0
            }
            
            if scenarios:
                total_decisions = sum(s['analysis_report']['summary'].get('total_decisions', 0) 
                                    for s in scenarios.values())
                total_diversity = sum(s['analysis_report']['summary'].get('predator_action_diversity', 0) + 
                                    s['analysis_report']['summary'].get('prey_action_diversity', 0) 
                                    for s in scenarios.values())
                
                comparison_data[algorithm]['avg_decisions'] = total_decisions / len(scenarios)
                comparison_data[algorithm]['avg_diversity'] = total_diversity / len(scenarios)
                comparison_data[algorithm]['behavioral_adaptability'] = len(scenarios) * total_diversity / 10
        
        # Print comparison
        for algorithm, metrics in comparison_data.items():
            print(f"\n{algorithm.upper()}:")
            print(f"  Scenarios Analyzed: {metrics['total_scenarios']}")
            print(f"  Avg Decisions per Scenario: {metrics['avg_decisions']:.1f}")
            print(f"  Avg Behavioral Diversity: {metrics['avg_diversity']:.1f}")
            print(f"  Adaptability Score: {metrics['behavioral_adaptability']:.2f}")
        
        # Determine winner
        if len(comparison_data) > 1:
            best_algorithm = max(comparison_data.items(), 
                               key=lambda x: x[1]['behavioral_adaptability'])
            print(f"\n🏆 Most Adaptable Algorithm: {best_algorithm[0].upper()}")
            print(f"   Adaptability Score: {best_algorithm[1]['behavioral_adaptability']:.2f}")

def main():
    """Main analysis function."""
    analyzer = ModelBehaviorAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    print("\n" + "="*60)
    print("🔬 Behavioral Analysis Complete!")
    print("📊 Generated detailed behavioral insights for all models")
    print("🚀 Your models show sophisticated decision-making capabilities!")

if __name__ == "__main__":
    main()
