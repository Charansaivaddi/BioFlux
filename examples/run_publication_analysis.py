#!/usr/bin/env python3
"""
Publication Quality Analysis Script
===================================

Generate comprehensive, statistically robust results for research paper publication.
This script runs extended evaluations with proper statistical analysis.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import time
from pathlib import Path
from datetime import datetime
from scipy import stats
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from bioflux.training import (
    TrainingConfig, LotkaVolterraAgent, PPOAgent,
    create_training_environment
)
from bioflux.core.environment import Environment

class PublicationAnalysis:
    """Generate publication-quality analysis and statistics."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.models_dir = self.base_dir / "models"
        self.output_dir = self.base_dir / "output" / "publication_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistical analysis configuration
        self.n_runs = 50  # For statistical significance
        self.n_episodes_per_run = 20
        self.max_steps_per_episode = 100
        
        # Results storage
        self.results = {}
        
        print("📊 Publication Quality Analysis System")
        print(f"Statistical Runs: {self.n_runs}")
        print(f"Episodes per Run: {self.n_episodes_per_run}")
        print(f"Output Directory: {self.output_dir}")
    
    def setup_config(self):
        """Setup training configuration."""
        return TrainingConfig(
            num_episodes=self.n_episodes_per_run,
            max_steps_per_episode=self.max_steps_per_episode,
            device="cpu"
        )
    
    def create_test_environment(self, scenario='balanced'):
        """Create standardized test environment."""
        env = Environment()
        env.width = 30
        env.height = 30
        
        # Standardized scenarios for research
        scenarios = {
            'balanced': {'predators': 3, 'prey': 5, 'plants': 15},
            'predator_advantage': {'predators': 6, 'prey': 3, 'plants': 12},
            'prey_advantage': {'predators': 2, 'prey': 8, 'plants': 20}
        }
        
        config = scenarios.get(scenario, scenarios['balanced'])
        
        # Add agents with controlled initialization
        from bioflux.core.agents import Predator, Prey, Plant
        
        # Add predators
        for i in range(config['predators']):
            predator = Predator(
                speed=2,
                energy=100,
                pos_x=np.random.randint(5, 25),
                pos_y=np.random.randint(5, 25),
                age=1
            )
            env.add_predator(predator)
        
        # Add prey
        for i in range(config['prey']):
            prey = Prey(
                speed=3,
                energy=60,
                pos_x=np.random.randint(5, 25),
                pos_y=np.random.randint(5, 25),
                age=1
            )
            env.add_prey(prey)
        
        # Add plants
        for i in range(config['plants']):
            plant = Plant(
                energy=15,
                pos_x=np.random.randint(0, 30),
                pos_y=np.random.randint(0, 30)
            )
            env.add_plant(plant)
        
        return env
    
    def get_agent_state(self, agent, env):
        """Get standardized agent state."""
        nearby_prey = []
        nearby_predators = []
        nearby_food = []
        
        all_agents = env.predators + env.prey + env.plants
        
        for other in all_agents:
            if other != agent and hasattr(other, 'pos_x') and hasattr(other, 'pos_y'):
                distance = np.sqrt((other.pos_x - agent.pos_x)**2 + (other.pos_y - agent.pos_y)**2)
                if distance < 10:
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
    
    def calculate_reward(self, agent, action, env):
        """Calculate standardized reward."""
        reward = 1.0  # Base survival reward
        
        # Reward for being near appropriate targets
        if agent.agent_type == 'predator':
            for prey in env.prey:
                if prey.is_alive:
                    distance = np.sqrt((prey.pos_x - agent.pos_x)**2 + (prey.pos_y - agent.pos_y)**2)
                    if distance < 5:
                        reward += 10.0 / (distance + 1)
        elif agent.agent_type == 'prey':
            for plant in env.plants:
                if plant.is_alive:
                    distance = np.sqrt((plant.pos_x - agent.pos_x)**2 + (plant.pos_y - agent.pos_y)**2)
                    if distance < 3:
                        reward += 5.0 / (distance + 1)
        
        return reward
    
    def apply_action(self, agent, action, env):
        """Apply action with realistic consequences."""
        # Move agent
        if 'x' in action and 'y' in action:
            new_x = max(0, min(env.width - 1, int(action['x'])))
            new_y = max(0, min(env.height - 1, int(action['y'])))
            agent.pos_x = new_x
            agent.pos_y = new_y
        
        # Energy costs
        action_type = action.get('action', 'stay')
        if action_type in ['move', 'hunt', 'explore']:
            agent.energy -= 0.5
        
        # Energy decay
        agent.energy -= 0.1
        
        # Death check
        if agent.energy <= 0:
            agent.is_alive = False
    
    def run_single_evaluation(self, algorithm, scenario='balanced'):
        """Run a single evaluation episode."""
        config = self.setup_config()
        
        # Create agent
        if algorithm == 'lotka_volterra':
            predator_agent = LotkaVolterraAgent('predator', config)
            prey_agent = LotkaVolterraAgent('prey', config)
        elif algorithm == 'ppo':
            predator_agent = PPOAgent('predator', config, state_dim=8, action_dim=8)
            prey_agent = PPOAgent('prey', config, state_dim=8, action_dim=8)
            
            # Try to load trained models
            try:
                pred_path = self.models_dir / "ppo" / "predator_episode_1000.pth"
                if pred_path.exists():
                    checkpoint = torch.load(pred_path, map_location='cpu', weights_only=False)
                    predator_agent.actor.load_state_dict(checkpoint['actor'])
                    predator_agent.critic.load_state_dict(checkpoint['critic'])
            except Exception as e:
                print(f"⚠️ Could not load PPO model: {e}")
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        survival_rates = []
        
        for episode in range(self.n_episodes_per_run):
            env = self.create_test_environment(scenario)
            total_reward = 0
            episode_length = 0
            
            initial_predators = len(env.predators)
            initial_prey = len(env.prey)
            
            for step in range(self.max_steps_per_episode):
                episode_length += 1
                
                # Get active agents
                active_predators = [p for p in env.predators if p.is_alive]
                active_prey = [p for p in env.prey if p.is_alive]
                
                if len(active_predators) == 0 or len(active_prey) == 0:
                    break
                
                # Process predators
                for predator in active_predators:
                    state = self.get_agent_state(predator, env)
                    action = predator_agent.select_action(state)
                    self.apply_action(predator, action, env)
                    reward = self.calculate_reward(predator, action, env)
                    total_reward += reward
                
                # Process prey
                for prey in active_prey:
                    state = self.get_agent_state(prey, env)
                    action = prey_agent.select_action(state)
                    self.apply_action(prey, action, env)
                    reward = self.calculate_reward(prey, action, env)
                    total_reward += reward
            
            # Calculate survival rate
            final_predators = len([p for p in env.predators if p.is_alive])
            final_prey = len([p for p in env.prey if p.is_alive])
            survival_rate = (final_predators + final_prey) / (initial_predators + initial_prey)
            
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            survival_rates.append(survival_rate)
        
        return {
            'rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'survival_rates': survival_rates,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_episode_length': np.mean(episode_lengths),
            'avg_survival_rate': np.mean(survival_rates)
        }
    
    def run_comprehensive_analysis(self):
        """Run comprehensive statistical analysis."""
        print("\\n📊 Starting Comprehensive Publication Analysis")
        print("=" * 60)
        
        algorithms = ['lotka_volterra', 'ppo']
        scenarios = ['balanced', 'predator_advantage', 'prey_advantage']
        
        all_results = {}
        
        for algorithm in algorithms:
            print(f"\\n🔬 Analyzing {algorithm.upper()}")
            print("-" * 40)
            
            algorithm_results = {}
            
            for scenario in scenarios:
                print(f"\\n📋 Scenario: {scenario}")
                
                scenario_results = []
                
                # Run multiple statistical runs
                for run in range(self.n_runs):
                    if run % 10 == 0:
                        print(f"  Run {run + 1}/{self.n_runs}")
                    
                    try:
                        result = self.run_single_evaluation(algorithm, scenario)
                        scenario_results.append(result)
                    except Exception as e:
                        print(f"  ⚠️ Error in run {run + 1}: {e}")
                        continue
                
                # Aggregate results
                if scenario_results:
                    aggregated = self.aggregate_results(scenario_results)
                    algorithm_results[scenario] = aggregated
                    
                    print(f"  ✅ {scenario}: {aggregated['overall_avg_reward']:.2f} ± {aggregated['overall_std_reward']:.2f}")
            
            all_results[algorithm] = algorithm_results
        
        # Perform statistical tests
        statistical_analysis = self.perform_statistical_analysis(all_results)
        
        # Save results
        self.save_publication_results(all_results, statistical_analysis)
        
        # Generate publication plots
        self.generate_publication_plots(all_results)
        
        return all_results, statistical_analysis
    
    def aggregate_results(self, scenario_results):
        """Aggregate results across runs."""
        all_rewards = []
        all_lengths = []
        all_survival_rates = []
        
        for result in scenario_results:
            all_rewards.extend(result['rewards'])
            all_lengths.extend(result['episode_lengths'])
            all_survival_rates.extend(result['survival_rates'])
        
        return {
            'n_runs': len(scenario_results),
            'n_episodes': len(all_rewards),
            'overall_avg_reward': np.mean(all_rewards),
            'overall_std_reward': np.std(all_rewards),
            'overall_avg_length': np.mean(all_lengths),
            'overall_std_length': np.std(all_lengths),
            'overall_avg_survival': np.mean(all_survival_rates),
            'overall_std_survival': np.std(all_survival_rates),
            'reward_95_ci': stats.t.interval(0.95, len(all_rewards)-1, 
                                           loc=np.mean(all_rewards), 
                                           scale=stats.sem(all_rewards)),
            'all_rewards': all_rewards,
            'all_lengths': all_lengths,
            'all_survival_rates': all_survival_rates
        }
    
    def perform_statistical_analysis(self, all_results):
        """Perform statistical significance tests."""
        print("\\n📈 Performing Statistical Analysis")
        print("-" * 40)
        
        statistical_results = {}
        
        # Compare algorithms across scenarios
        algorithms = list(all_results.keys())
        if len(algorithms) >= 2:
            algo1, algo2 = algorithms[0], algorithms[1]
            
            for scenario in ['balanced', 'predator_advantage', 'prey_advantage']:
                if scenario in all_results[algo1] and scenario in all_results[algo2]:
                    rewards1 = all_results[algo1][scenario]['all_rewards']
                    rewards2 = all_results[algo2][scenario]['all_rewards']
                    
                    # T-test
                    t_stat, p_value = stats.ttest_ind(rewards1, rewards2)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(rewards1) - 1) * np.var(rewards1, ddof=1) + 
                                        (len(rewards2) - 1) * np.var(rewards2, ddof=1)) / 
                                       (len(rewards1) + len(rewards2) - 2))
                    cohens_d = (np.mean(rewards1) - np.mean(rewards2)) / pooled_std
                    
                    statistical_results[f"{algo1}_vs_{algo2}_{scenario}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05,
                        'effect_size_interpretation': self.interpret_effect_size(cohens_d)
                    }
                    
                    print(f"  {scenario}: t={t_stat:.3f}, p={p_value:.3f}, d={cohens_d:.3f}")
        
        return statistical_results
    
    def interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def save_publication_results(self, results, statistical_analysis):
        """Save results in publication format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results
        results_file = self.output_dir / f"publication_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'results': results,
                'statistical_analysis': statistical_analysis,
                'metadata': {
                    'n_statistical_runs': self.n_runs,
                    'n_episodes_per_run': self.n_episodes_per_run,
                    'max_steps_per_episode': self.max_steps_per_episode,
                    'timestamp': timestamp
                }
            }, f, indent=2, default=str)
        
        # Save summary table for paper
        self.create_summary_table(results, statistical_analysis)
        
        print(f"\\n💾 Results saved to: {results_file}")
    
    def create_summary_table(self, results, statistical_analysis):
        """Create publication-ready summary table."""
        table_data = []
        
        for algorithm, algo_results in results.items():
            for scenario, scenario_results in algo_results.items():
                table_data.append({
                    'Algorithm': algorithm.replace('_', '-').upper(),
                    'Scenario': scenario.replace('_', ' ').title(),
                    'Avg Reward': f"{scenario_results['overall_avg_reward']:.2f}",
                    'Std Reward': f"{scenario_results['overall_std_reward']:.2f}",
                    'Avg Episode Length': f"{scenario_results['overall_avg_length']:.1f}",
                    'Avg Survival Rate': f"{scenario_results['overall_avg_survival']:.3f}",
                    'N Episodes': scenario_results['n_episodes']
                })
        
        df = pd.DataFrame(table_data)
        
        # Save as CSV for easy paper inclusion
        table_file = self.output_dir / "publication_summary_table.csv"
        df.to_csv(table_file, index=False)
        
        # Save as LaTeX table
        latex_file = self.output_dir / "publication_table.tex"
        with open(latex_file, 'w') as f:
            f.write(df.to_latex(index=False, float_format="%.2f"))
        
        print(f"📋 Summary table saved to: {table_file}")
        print(f"📋 LaTeX table saved to: {latex_file}")
    
    def generate_publication_plots(self, results):
        """Generate publication-quality plots."""
        print("\\n🎨 Generating Publication Plots")
        print("-" * 30)
        
        # Create performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Algorithm Performance Comparison Across Scenarios', fontsize=14, fontweight='bold')
        
        scenarios = ['balanced', 'predator_advantage', 'prey_advantage']
        algorithms = list(results.keys())
        
        # Plot 1: Average Rewards
        ax1 = axes[0, 0]
        x = np.arange(len(scenarios))
        width = 0.35
        
        for i, algo in enumerate(algorithms):
            rewards = [results[algo][scenario]['overall_avg_reward'] for scenario in scenarios]
            errors = [results[algo][scenario]['overall_std_reward'] for scenario in scenarios]
            ax1.bar(x + i*width, rewards, width, label=algo.replace('_', '-').upper(), 
                   yerr=errors, capsize=5, alpha=0.8)
        
        ax1.set_xlabel('Scenario')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Reward Performance')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        ax2 = axes[0, 1]
        for i, algo in enumerate(algorithms):
            lengths = [results[algo][scenario]['overall_avg_length'] for scenario in scenarios]
            errors = [results[algo][scenario]['overall_std_length'] for scenario in scenarios]
            ax2.bar(x + i*width, lengths, width, label=algo.replace('_', '-').upper(),
                   yerr=errors, capsize=5, alpha=0.8)
        
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Average Episode Length')
        ax2.set_title('Episode Duration')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Survival Rates
        ax3 = axes[1, 0]
        for i, algo in enumerate(algorithms):
            survival = [results[algo][scenario]['overall_avg_survival'] for scenario in scenarios]
            errors = [results[algo][scenario]['overall_std_survival'] for scenario in scenarios]
            ax3.bar(x + i*width, survival, width, label=algo.replace('_', '-').upper(),
                   yerr=errors, capsize=5, alpha=0.8)
        
        ax3.set_xlabel('Scenario')
        ax3.set_ylabel('Average Survival Rate')
        ax3.set_title('Agent Survival')
        ax3.set_xticks(x + width/2)
        ax3.set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Statistical Significance
        ax4 = axes[1, 1]
        ax4.text(0.1, 0.8, 'Statistical Analysis Summary:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f'Sample Size: {self.n_runs} runs × {self.n_episodes_per_run} episodes', transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f'Total Episodes: {self.n_runs * self.n_episodes_per_run} per scenario', transform=ax4.transAxes)
        ax4.text(0.1, 0.5, 'Confidence Intervals: 95%', transform=ax4.transAxes)
        ax4.text(0.1, 0.4, 'Statistical Tests: Independent t-tests', transform=ax4.transAxes)
        ax4.text(0.1, 0.3, 'Effect Size: Cohen\'s d', transform=ax4.transAxes)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "publication_performance_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"🎨 Publication plot saved to: {plot_file}")

def main():
    """Main publication analysis function."""
    analyzer = PublicationAnalysis()
    results, statistical_analysis = analyzer.run_comprehensive_analysis()
    
    print("\\n" + "=" * 60)
    print("📊 PUBLICATION ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\\n📋 Generated Files:")
    print(f"  • Publication results JSON")
    print(f"  • Summary table (CSV & LaTeX)")
    print(f"  • Performance comparison plots")
    print(f"  • Statistical analysis report")
    print("\\n🚀 Results are ready for research paper inclusion!")

if __name__ == "__main__":
    main()
