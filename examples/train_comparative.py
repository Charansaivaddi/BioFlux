#!/usr/bin/env python3
"""
Comprehensive RL Training and Comparison Script for BioFlux.

This script runs a comparative study of different RL algorithms:
- Lotka-Volterra (baseline ecological model)
- Epsilon-Greedy Q-learning
- PPO (Proximal Policy Optimization)
- MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import json
import argparse

# Add the parent directory to the path to import bioflux
sys.path.append(str(Path(__file__).parent.parent))

from bioflux.training import (
    TrainingConfig, TrainingRunner, create_training_environment
)

def setup_training_config(algorithm: str = 'all') -> TrainingConfig:
    """Setup training configuration."""
    return TrainingConfig(
        algorithm=algorithm,
        num_episodes=500,  # Reduced for demo
        max_steps_per_episode=200,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=32,
        memory_size=5000,
        target_update_frequency=50,
        save_frequency=50,
        log_frequency=25,
        hidden_dim=128,
        device="cpu"  # Use CPU for compatibility
    )

def run_training(config: TrainingConfig):
    """Run the comparative training study."""
    print("🚀 Starting BioFlux RL Comparative Study")
    print("=" * 50)
    
    # Create training environment
    env = create_training_environment()
    print(f"✅ Environment created with {len(env.agents)} agents")
    
    # Create training runner
    runner = TrainingRunner(env, config)
    print("🏃 Training runner initialized")
    
    # Run comparative study
    print("\n🎯 Beginning comparative training...")
    results = runner.run_comparative_study()
    
    return results, runner

def create_comparison_plots(results: dict, runner):
    """Create comprehensive comparison plots."""
    print("\n📊 Generating comparison plots...")
    
    # Create output directory
    output_dir = Path("output/training_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Reward comparison over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('BioFlux RL Algorithms Comparison', fontsize=16, fontweight='bold')
    
    # Reward trends
    ax1 = axes[0, 0]
    for algorithm in ['lotka_volterra', 'epsilon_greedy', 'ppo', 'maddpg']:
        if algorithm in runner.results and runner.results[algorithm]['rewards']:
            rewards = runner.results[algorithm]['rewards']
            episodes = range(0, len(rewards) * 25, 25)  # Based on log_frequency
            ax1.plot(episodes[:len(rewards)], rewards, label=algorithm.replace('_', '-').upper(), 
                    linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Learning Progress: Average Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode length trends
    ax2 = axes[0, 1]
    for algorithm in ['lotka_volterra', 'epsilon_greedy', 'ppo', 'maddpg']:
        if algorithm in runner.results and runner.results[algorithm]['episode_lengths']:
            lengths = runner.results[algorithm]['episode_lengths']
            episodes = range(0, len(lengths) * 25, 25)
            ax2.plot(episodes[:len(lengths)], lengths, label=algorithm.replace('_', '-').upper(),
                    linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Duration Trends')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Survival rate trends
    ax3 = axes[1, 0]
    for algorithm in ['lotka_volterra', 'epsilon_greedy', 'ppo', 'maddpg']:
        if algorithm in runner.results and runner.results[algorithm]['survival_rates']:
            survival = runner.results[algorithm]['survival_rates']
            episodes = range(0, len(survival) * 25, 25)
            ax3.plot(episodes[:len(survival)], survival, label=algorithm.replace('_', '-').upper(),
                    linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Survival Rate')
    ax3.set_title('Ecosystem Stability: Survival Rates')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Performance summary (bar chart)
    ax4 = axes[1, 1]
    algorithms = []
    avg_rewards = []
    avg_survival = []
    
    for algorithm in ['lotka_volterra', 'epsilon_greedy', 'ppo', 'maddpg']:
        if algorithm in results['summary']:
            algorithms.append(algorithm.replace('_', '-').upper())
            avg_rewards.append(results['summary'][algorithm]['avg_reward'])
            avg_survival.append(results['summary'][algorithm]['avg_survival_rate'])
    
    if algorithms:
        x_pos = np.arange(len(algorithms))
        width = 0.35
        
        bars1 = ax4.bar(x_pos - width/2, avg_rewards, width, label='Avg Reward', alpha=0.8)
        bars2 = ax4.bar(x_pos + width/2, [s*10 for s in avg_survival], width, 
                       label='Survival Rate (×10)', alpha=0.8)
        
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel('Performance Metric')
        ax4.set_title('Final Performance Comparison')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(algorithms, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height/10:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plots
    plot_path = output_dir / "rl_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Comparison plots saved to {plot_path}")
    
    plt.show()
    
    return plot_path

def generate_detailed_report(results: dict, runner):
    """Generate detailed text report."""
    print("\n📝 Generating detailed report...")
    
    output_dir = Path("output/training_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"training_report_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write("# BioFlux RL Algorithms Comparative Study Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report presents a comparative analysis of four different approaches ")
        f.write("to ecosystem agent behavior in the BioFlux simulation:\n\n")
        
        f.write("1. **Lotka-Volterra**: Classical ecological dynamics (baseline)\n")
        f.write("2. **Epsilon-Greedy**: Q-learning with exploration\n")
        f.write("3. **PPO**: Proximal Policy Optimization\n")
        f.write("4. **MADDPG**: Multi-Agent Deep Deterministic Policy Gradient\n\n")
        
        f.write("## Performance Summary\n\n")
        
        if results.get('summary'):
            f.write("| Algorithm | Avg Reward | Std Reward | Avg Episode Length | Avg Survival Rate |\n")
            f.write("|-----------|------------|------------|-------------------|------------------|\n")
            
            for algorithm, metrics in results['summary'].items():
                f.write(f"| {algorithm.replace('_', '-').upper()} | ")
                f.write(f"{metrics['avg_reward']:.3f} | ")
                f.write(f"{metrics['std_reward']:.3f} | ")  
                f.write(f"{metrics['avg_episode_length']:.1f} | ")
                f.write(f"{metrics['avg_survival_rate']:.3f} |\n")
        
        f.write("\n## Detailed Analysis\n\n")
        
        for algorithm in ['lotka_volterra', 'epsilon_greedy', 'ppo', 'maddpg']:
            if algorithm in results.get('summary', {}):
                f.write(f"### {algorithm.replace('_', '-').upper()}\n\n")
                metrics = results['summary'][algorithm]
                
                f.write(f"- **Average Reward**: {metrics['avg_reward']:.3f} ± {metrics['std_reward']:.3f}\n")
                f.write(f"- **Episode Length**: {metrics['avg_episode_length']:.1f} steps\n")
                f.write(f"- **Survival Rate**: {metrics['avg_survival_rate']:.1%}\n")
                
                # Add algorithm-specific insights
                if algorithm == 'lotka_volterra':
                    f.write("- **Analysis**: Baseline ecological model providing reference behavior\n")
                elif algorithm == 'epsilon_greedy':
                    f.write("- **Analysis**: Simple RL approach with exploration-exploitation trade-off\n")
                elif algorithm == 'ppo':
                    f.write("- **Analysis**: Advanced policy gradient method with stable learning\n")
                elif algorithm == 'maddpg':
                    f.write("- **Analysis**: Multi-agent approach considering agent interactions\n")
                
                f.write("\n")
        
        f.write("## Recommendations\n\n")
        
        if results.get('recommendations'):
            for i, rec in enumerate(results['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        f.write("\n## Methodology\n\n")
        f.write("- **Environment**: 100x100 grid world\n")
        f.write("- **Agents**: 5 predators, 20 prey, 100 plants\n")
        f.write("- **Episodes**: 500 per algorithm\n")
        f.write("- **Max Steps**: 200 per episode\n")
        f.write("- **Evaluation**: Average over final 100 episodes\n")
        
        f.write("\n## Technical Configuration\n\n")
        config = runner.config
        f.write(f"- **Learning Rate**: {config.learning_rate}\n")
        f.write(f"- **Discount Factor**: {config.gamma}\n")
        f.write(f"- **Batch Size**: {config.batch_size}\n")
        f.write(f"- **Memory Size**: {config.memory_size}\n")
        f.write(f"- **Hidden Dimensions**: {config.hidden_dim}\n")
    
    print(f"📋 Detailed report saved to {report_path}")
    return report_path

def save_results_json(results: dict, runner):
    """Save results in JSON format for further analysis."""
    output_dir = Path("output/training_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"results_{timestamp}.json"
    
    # Save complete results
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to {json_path}")
    return json_path

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='BioFlux RL Comparative Study')
    parser.add_argument('--episodes', type=int, default=500, 
                       help='Number of training episodes per algorithm')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip generating plots')
    parser.add_argument('--algorithm', choices=['all', 'lotka_volterra', 'epsilon_greedy', 'ppo', 'maddpg'],
                       default='all', help='Run specific algorithm only')
    
    args = parser.parse_args()
    
    try:
        # Setup configuration
        config = setup_training_config(args.algorithm)
        config.num_episodes = args.episodes
        
        print(f"🔧 Configuration: {args.episodes} episodes, algorithm: {args.algorithm}")
        
        # Run training
        results, runner = run_training(config)
        
        # Generate outputs
        if not args.no_plots:
            create_comparison_plots(results, runner)
        
        generate_detailed_report(results, runner)
        save_results_json(results, runner)
        
        print("\n✅ Comparative study completed successfully!")
        print("📊 Check the output/training_results/ directory for detailed results")
        
        # Print quick summary
        if results.get('summary'):
            print("\n🏆 Quick Performance Summary:")
            for algorithm, metrics in results['summary'].items():
                print(f"  {algorithm.replace('_', '-').upper()}: "
                      f"Reward={metrics['avg_reward']:.2f}, "
                      f"Survival={metrics['avg_survival_rate']:.1%}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
