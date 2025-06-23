# BioFlux RL Training Framework

This document describes the reinforcement learning training framework implemented in BioFlux for comparative studies of different algorithms in ecosystem simulation.

## Overview

The training framework implements and compares four different approaches to agent behavior in the BioFlux ecosystem:

1. **Lotka-Volterra**: Classical ecological dynamics model (baseline)
2. **Epsilon-Greedy**: Q-learning with ε-greedy exploration
3. **PPO**: Proximal Policy Optimization
4. **MADDPG**: Multi-Agent Deep Deterministic Policy Gradient

## Architecture

### Core Components

```
bioflux/training/
├── __init__.py          # Main training classes and utilities
├── maddpg.py           # MADDPG implementation
└── README.md           # This documentation
```

### Key Classes

#### `TrainingConfig`
Configuration dataclass containing all hyperparameters:
- `algorithm`: Algorithm to use ('lotka_volterra', 'epsilon_greedy', 'ppo', 'maddpg')
- `num_episodes`: Number of training episodes (default: 1000)
- `max_steps_per_episode`: Maximum steps per episode (default: 200)
- `learning_rate`: Learning rate for neural networks (default: 3e-4)
- `gamma`: Discount factor (default: 0.99)
- And many more...

#### `TrainingRunner`
Main orchestrator class that:
- Creates and manages agents for all algorithms
- Runs comparative training studies
- Collects and analyzes results
- Generates performance reports

### Agent Implementations

#### 1. Lotka-Volterra Agent
```python
class LotkaVolterraAgent:
    def __init__(self, agent_type: str, config: TrainingConfig)
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]
    def update(self, trajectory: List[Dict[str, Any]]) -> None  # No-op for baseline
```

**Behavior**: Uses classical predator-prey dynamics with fixed parameters:
- Predators hunt when energy is high and prey are nearby
- Prey flee from predators and forage when safe
- No learning - provides baseline performance

#### 2. Epsilon-Greedy Agent
```python
class EpsilonGreedyAgent:
    def __init__(self, agent_type: str, config: TrainingConfig, state_dim: int, action_dim: int)
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]
    def update(self, trajectory: List[Dict[str, Any]]) -> None
```

**Features**:
- Deep Q-Network (DQN) with target network
- Experience replay buffer
- ε-greedy exploration with decay
- 8-dimensional state space, 8 discrete actions

#### 3. PPO Agent
```python
class PPOAgent:
    def __init__(self, agent_type: str, config: TrainingConfig, state_dim: int, action_dim: int)
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]
    def update(self, trajectory: List[Dict[str, Any]]) -> None
```

**Features**:
- Actor-Critic architecture
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Entropy regularization

#### 4. MADDPG Agent
```python
class MADDPGWrapper:
    def __init__(self, agent_type: str, config: TrainingConfig, num_agents: int = 4)
    def select_action(self, state: Dict[str, Any], agent_id: int = 0) -> Dict[str, Any]
    def update(self, trajectory: List[Dict[str, Any]]) -> None
```

**Features**:
- Multi-agent deep deterministic policy gradient
- Centralized training, decentralized execution
- Ornstein-Uhlenbeck noise for exploration
- Actor-Critic with continuous actions

## State Representation

All agents use a standardized 8-dimensional state vector:

```python
features = [
    energy / 100.0,                           # Normalized energy level
    age / 20.0,                              # Normalized age
    len(nearby_prey) / 10.0,                 # Nearby prey count
    len(nearby_predators) / 10.0,            # Nearby predator count  
    len(nearby_food) / 10.0,                 # Nearby food sources
    temperature / 40.0,                      # Environmental temperature
    vegetation,                              # Vegetation density (0-1)
    1.0 if is_hungry else 0.0               # Hunger status
]
```

## Action Space

### Discrete Actions (Epsilon-Greedy, PPO)
- `stay`: Remain in current position
- `north`, `south`, `east`, `west`: Move in cardinal directions
- `hunt`: Hunt nearby prey (predators only)
- `flee`: Flee from nearby predators (prey only)  
- `forage`: Forage for food (prey only)

### Continuous Actions (MADDPG)
- `[x_move, y_move, action_type]`: 3D continuous vector mapped to discrete actions

## Usage

### Running the Demo
```bash
python examples/rl_demo.py
```

### Full Comparative Study
```bash
python examples/train_comparative.py --episodes 1000
```

### Specific Algorithm
```bash
python examples/train_comparative.py --algorithm ppo --episodes 500
```

### Command Line Options
- `--episodes N`: Number of training episodes (default: 500)
- `--algorithm ALGO`: Train specific algorithm only
- `--no-plots`: Skip generating plots

## Training Process

### 1. Environment Setup
- 100×100 grid world
- 5 predators, 20 prey, 100 plants
- Fixed initial conditions for fair comparison

### 2. Training Loop
For each algorithm and episode:
1. Reset environment to initial state
2. Run simulation for max_steps_per_episode
3. Collect state-action-reward trajectories
4. Update agent using algorithm-specific method
5. Log performance metrics

### 3. Evaluation Metrics
- **Average Reward**: Cumulative reward per episode
- **Episode Length**: Number of steps before termination
- **Survival Rate**: Fraction of agents surviving

## Reward Structure

### Predators
- `+10`: Successful hunt (close to prey)
- `-1`: Failed hunt attempt
- `0`: Other actions

### Prey  
- `+5`: Successful escape from predator
- `+2-5`: Foraging reward (based on vegetation)
- `0`: Safe exploration

## Output Files

Training generates several output files in `output/training_results/`:

- `rl_comparison.png`: Comparative performance plots
- `training_report_TIMESTAMP.md`: Detailed analysis report
- `results_TIMESTAMP.json`: Raw numerical results
- `models/ALGORITHM/`: Saved model checkpoints

## Performance Analysis

The framework automatically generates:

### Plots
1. **Learning Progress**: Reward trends over episodes
2. **Episode Duration**: Length trends showing stability
3. **Survival Rates**: Ecosystem stability metrics
4. **Performance Summary**: Final comparison bar charts

### Report Sections
- **Executive Summary**: High-level findings
- **Performance Summary**: Quantitative comparison table
- **Detailed Analysis**: Algorithm-specific insights
- **Recommendations**: Suggested next steps
- **Methodology**: Experimental setup details

## Extending the Framework

### Adding New Algorithms
1. Create agent class implementing the standard interface:
   ```python
   def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]
   def update(self, trajectory: List[Dict[str, Any]]) -> None
   def save(self, filepath: str) -> None
   ```

2. Add to `TrainingRunner._create_agents()`
3. Update algorithm list in training scripts

### Custom Environments
Replace `create_training_environment()` with your own:
```python
def create_custom_environment():
    env = EcosystemEnvironment(width=200, height=200)
    # Add custom agents and configuration
    return env
```

### Custom Rewards
Override reward calculation methods in `TrainingRunner`:
- `_calculate_hunt_reward()`
- `_calculate_forage_reward()`
- `_calculate_flee_reward()`

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**CUDA Errors**: Set device to CPU in config:
```python
config.device = "cpu"
```

**Memory Issues**: Reduce batch size or buffer size:
```python
config.batch_size = 16
config.memory_size = 1000
```

**Training Slow**: Reduce episodes or use CPU:
```python
config.num_episodes = 100
config.device = "cpu"
```

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check agent states and actions:
```python
print(f"State: {state}")
print(f"Action: {action}")
print(f"Reward: {reward}")
```

## Research Applications

This framework supports various research directions:

### Comparative Studies
- Algorithm performance in different environments
- Hyperparameter sensitivity analysis
- Scalability with agent count

### Ecosystem Modeling
- Emergent behavior analysis
- Population dynamics studies
- Environmental impact assessment

### Multi-Agent Learning
- Cooperation vs competition dynamics
- Communication protocols
- Hierarchical learning structures

## References

1. **PPO**: Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017)
2. **MADDPG**: Lowe, R., et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." NIPS 2017
3. **DQN**: Mnih, V., et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015)
4. **Lotka-Volterra**: Lotka, A.J. "Elements of Physical Biology." Williams & Wilkins (1925)

## License

This training framework is part of the BioFlux project and follows the same licensing terms.
