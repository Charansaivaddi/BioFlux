#!/usr/bin/env python3
"""
RL Training Framework for BioFlux Ecosystem Agents.

This module implements and compares different RL algorithms:
- Lotka-Volterra (baseline ecological model)
- Epsilon-Greedy (simple Q-learning)
- PPO (Proximal Policy Optimization)
- MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import random
from collections import deque, namedtuple
import logging
from dataclasses import dataclass
import pickle
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    algorithm: str = "ppo"  # lotka_volterra, epsilon_greedy, ppo, maddpg
    num_episodes: int = 1000
    max_steps_per_episode: int = 200
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    batch_size: int = 64
    memory_size: int = 10000
    target_update_frequency: int = 100
    save_frequency: int = 100
    log_frequency: int = 10
    hidden_dim: int = 128
    device: str = "cpu"

class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class LotkaVolterraAgent:
    """Baseline agent using classical Lotka-Volterra dynamics."""
    
    def __init__(self, agent_type: str, config: TrainingConfig):
        self.agent_type = agent_type  # 'predator' or 'prey'
        self.config = config
        
        # Lotka-Volterra parameters
        if agent_type == 'predator':
            self.alpha = 0.1  # Predation rate
            self.beta = 0.075  # Predator efficiency
            self.delta = 0.05  # Predator death rate
        else:  # prey
            self.alpha = 1.0  # Prey birth rate
            self.beta = 0.5  # Predation impact
    
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select action based on Lotka-Volterra dynamics."""
        if self.agent_type == 'predator':
            return self._predator_action(state)
        else:
            return self._prey_action(state)
    
    def _predator_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Predator behavior based on prey density."""
        nearby_prey = state.get('nearby_prey', [])
        energy = state.get('energy', 50)
        
        if nearby_prey and energy > 20:
            # Hunt closest prey
            closest_prey = min(nearby_prey, key=lambda p: p['distance'])
            return {
                'action': 'hunt',
                'target': closest_prey['pos'],
                'x': closest_prey['pos'][0],
                'y': closest_prey['pos'][1]
            }
        elif energy < 30:
            # Rest to conserve energy
            return {'action': 'rest', 'x': state.get('pos_x', 0), 'y': state.get('pos_y', 0)}
        else:
            # Patrol for prey
            return self._random_move(state)
    
    def _prey_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prey behavior based on predator presence and food availability."""
        nearby_predators = state.get('nearby_predators', [])
        nearby_food = state.get('nearby_food', [])
        energy = state.get('energy', 30)
        
        if nearby_predators:
            # Flee from predators
            closest_predator = min(nearby_predators, key=lambda p: p['distance'])
            pred_x, pred_y = closest_predator['pos']
            current_x = state.get('pos_x', 0)
            current_y = state.get('pos_y', 0)
            
            # Move away from predator
            flee_x = current_x + (current_x - pred_x)
            flee_y = current_y + (current_y - pred_y)
            
            bounds = state.get('environment_bounds', {'width': 100, 'height': 100})
            flee_x = max(0, min(bounds['width'] - 1, flee_x))
            flee_y = max(0, min(bounds['height'] - 1, flee_y))
            
            return {'action': 'flee', 'x': flee_x, 'y': flee_y}
        
        elif nearby_food and energy < 50:
            # Forage for food
            closest_food = min(nearby_food, key=lambda f: f['distance'])
            return {
                'action': 'forage',
                'target': closest_food['pos'],
                'x': closest_food['pos'][0],
                'y': closest_food['pos'][1]
            }
        else:
            # Explore
            return self._random_move(state)
    
    def _random_move(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Random movement for exploration."""
        current_x = state.get('pos_x', 0)
        current_y = state.get('pos_y', 0)
        bounds = state.get('environment_bounds', {'width': 100, 'height': 100})
        
        new_x = max(0, min(bounds['width'] - 1, current_x + random.randint(-2, 2)))
        new_y = max(0, min(bounds['height'] - 1, current_y + random.randint(-2, 2)))
        
        return {'action': 'explore', 'x': new_x, 'y': new_y}
    
    def update(self, trajectory: List[Dict[str, Any]]) -> None:
        """No learning for Lotka-Volterra (baseline)."""
        pass
    
    def save(self, filepath: str) -> None:
        """Save agent parameters."""
        params = {
            'agent_type': self.agent_type,
            'alpha': self.alpha,
            'beta': self.beta if hasattr(self, 'beta') else None,
            'delta': self.delta if hasattr(self, 'delta') else None
        }
        with open(filepath, 'w') as f:
            json.dump(params, f)

class EpsilonGreedyAgent:
    """Q-learning agent with epsilon-greedy exploration."""
    
    def __init__(self, agent_type: str, config: TrainingConfig, state_dim: int, action_dim: int):
        self.agent_type = agent_type
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-table approximation using neural network
        self.device = torch.device(config.device)
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Exploration parameters
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        
        # Experience replay
        self.memory = ReplayBuffer(config.memory_size)
        self.update_counter = 0
        
    def _build_network(self) -> nn.Module:
        """Build Q-network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.action_dim)
        )
    
    def _state_to_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        """Convert state dict to tensor."""
        features = [
            state.get('energy', 0) / 100.0,  # Normalize energy
            state.get('age', 0) / 20.0,  # Normalize age
            len(state.get('nearby_prey', [])) / 10.0,  # Normalize counts
            len(state.get('nearby_predators', [])) / 10.0,
            len(state.get('nearby_food', [])) / 10.0,
            state.get('temperature', 20) / 40.0,  # Normalize temperature
            state.get('vegetation', 0.5),  # Already normalized
            1.0 if state.get('is_hungry', False) else 0.0
        ]
        return torch.FloatTensor(features).to(self.device)
    
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Random action
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # Greedy action
            state_tensor = self._state_to_tensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        
        return self._action_idx_to_dict(action_idx, state)
    
    def _action_idx_to_dict(self, action_idx: int, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert action index to action dictionary."""
        actions = ['stay', 'north', 'south', 'east', 'west', 'hunt', 'flee', 'forage']
        action_name = actions[action_idx % len(actions)]
        
        current_x = state.get('pos_x', 0)
        current_y = state.get('pos_y', 0)
        bounds = state.get('environment_bounds', {'width': 100, 'height': 100})
        
        if action_name == 'stay':
            return {'action': 'stay', 'x': current_x, 'y': current_y}
        elif action_name == 'north':
            return {'action': 'move', 'x': current_x, 'y': max(0, current_y - 1)}
        elif action_name == 'south':
            return {'action': 'move', 'x': current_x, 'y': min(bounds['height'] - 1, current_y + 1)}
        elif action_name == 'east':
            return {'action': 'move', 'x': min(bounds['width'] - 1, current_x + 1), 'y': current_y}
        elif action_name == 'west':
            return {'action': 'move', 'x': max(0, current_x - 1), 'y': current_y}
        elif action_name == 'hunt' and self.agent_type == 'predator':
            nearby_prey = state.get('nearby_prey', [])
            if nearby_prey:
                target = min(nearby_prey, key=lambda p: p['distance'])
                return {'action': 'hunt', 'x': target['pos'][0], 'y': target['pos'][1]}
        elif action_name == 'flee' and self.agent_type == 'prey':
            nearby_predators = state.get('nearby_predators', [])
            if nearby_predators:
                closest_pred = min(nearby_predators, key=lambda p: p['distance'])
                flee_x = current_x + (current_x - closest_pred['pos'][0])
                flee_y = current_y + (current_y - closest_pred['pos'][1])
                flee_x = max(0, min(bounds['width'] - 1, flee_x))
                flee_y = max(0, min(bounds['height'] - 1, flee_y))
                return {'action': 'flee', 'x': flee_x, 'y': flee_y}
        elif action_name == 'forage' and self.agent_type == 'prey':
            nearby_food = state.get('nearby_food', [])
            if nearby_food:
                target = min(nearby_food, key=lambda f: f['distance'])
                return {'action': 'forage', 'x': target['pos'][0], 'y': target['pos'][1]}
        
        # Default: random move
        new_x = max(0, min(bounds['width'] - 1, current_x + random.randint(-1, 1)))
        new_y = max(0, min(bounds['height'] - 1, current_y + random.randint(-1, 1)))
        return {'action': 'explore', 'x': new_x, 'y': new_y}
    
    def update(self, trajectory: List[Dict[str, Any]]) -> None:
        """Update Q-network based on experience."""
        # Add experiences to replay buffer
        for i in range(len(trajectory) - 1):
            state = trajectory[i]['state']
            action = trajectory[i]['action']
            reward = trajectory[i].get('reward', 0)
            next_state = trajectory[i + 1]['state']
            done = trajectory[i].get('done', False)
            
            self.memory.push(state, action, reward, next_state, done)
        
        # Train if enough experiences
        if len(self.memory) >= self.config.batch_size:
            self._train_step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.config.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _train_step(self):
        """Perform one training step."""
        batch = self.memory.sample(self.config.batch_size)
        
        states = torch.stack([self._state_to_tensor(exp.state) for exp in batch])
        actions = torch.LongTensor([self._action_dict_to_idx(exp.action) for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_states = torch.stack([self._state_to_tensor(exp.next_state) for exp in batch])
        dones = torch.BoolTensor([exp.done for exp in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # Loss and optimization
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _action_dict_to_idx(self, action: Dict[str, Any]) -> int:
        """Convert action dictionary to index."""
        action_name = action.get('action', 'stay')
        actions = ['stay', 'north', 'south', 'east', 'west', 'hunt', 'flee', 'forage']
        try:
            return actions.index(action_name)
        except ValueError:
            return 0  # Default to 'stay'
    
    def save(self, filepath: str) -> None:
        """Save agent."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config
        }, filepath)

class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self, agent_type: str, config: TrainingConfig, state_dim: int, action_dim: int):
        self.agent_type = agent_type
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.device = torch.device(config.device)
        
        # Actor-Critic networks
        self.actor = self._build_actor().to(self.device)
        self.critic = self._build_critic().to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.ppo_epochs = 4
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        
        # Experience storage
        self.trajectory_buffer = []
    
    def _build_actor(self) -> nn.Module:
        """Build actor network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.action_dim),
            nn.Softmax(dim=-1)
        )
    
    def _build_critic(self) -> nn.Module:
        """Build critic network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1)
        )
    
    def _state_to_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        """Convert state dict to tensor."""
        features = [
            state.get('energy', 0) / 100.0,
            state.get('age', 0) / 20.0,
            len(state.get('nearby_prey', [])) / 10.0,
            len(state.get('nearby_predators', [])) / 10.0,
            len(state.get('nearby_food', [])) / 10.0,
            state.get('temperature', 20) / 40.0,
            state.get('vegetation', 0.5),
            1.0 if state.get('is_hungry', False) else 0.0
        ]
        return torch.FloatTensor(features).to(self.device)
    
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select action using policy network."""
        state_tensor = self._state_to_tensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample()
        
        return self._action_idx_to_dict(action_idx.item(), state)
    
    def _action_idx_to_dict(self, action_idx: int, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert action index to action dictionary."""
        # Same implementation as EpsilonGreedyAgent
        actions = ['stay', 'north', 'south', 'east', 'west', 'hunt', 'flee', 'forage']
        action_name = actions[action_idx % len(actions)]
        
        current_x = state.get('pos_x', 0)
        current_y = state.get('pos_y', 0)
        bounds = state.get('environment_bounds', {'width': 100, 'height': 100})
        
        if action_name == 'stay':
            return {'action': 'stay', 'x': current_x, 'y': current_y}
        elif action_name == 'north':
            return {'action': 'move', 'x': current_x, 'y': max(0, current_y - 1)}
        elif action_name == 'south':
            return {'action': 'move', 'x': current_x, 'y': min(bounds['height'] - 1, current_y + 1)}
        elif action_name == 'east':
            return {'action': 'move', 'x': min(bounds['width'] - 1, current_x + 1), 'y': current_y}
        elif action_name == 'west':
            return {'action': 'move', 'x': max(0, current_x - 1), 'y': current_y}
        elif action_name == 'hunt' and self.agent_type == 'predator':
            nearby_prey = state.get('nearby_prey', [])
            if nearby_prey:
                target = min(nearby_prey, key=lambda p: p['distance'])
                return {'action': 'hunt', 'x': target['pos'][0], 'y': target['pos'][1]}
        elif action_name == 'flee' and self.agent_type == 'prey':
            nearby_predators = state.get('nearby_predators', [])
            if nearby_predators:
                closest_pred = min(nearby_predators, key=lambda p: p['distance'])
                flee_x = current_x + (current_x - closest_pred['pos'][0])
                flee_y = current_y + (current_y - closest_pred['pos'][1])
                flee_x = max(0, min(bounds['width'] - 1, flee_x))
                flee_y = max(0, min(bounds['height'] - 1, flee_y))
                return {'action': 'flee', 'x': flee_x, 'y': flee_y}
        elif action_name == 'forage' and self.agent_type == 'prey':
            nearby_food = state.get('nearby_food', [])
            if nearby_food:
                target = min(nearby_food, key=lambda f: f['distance'])
                return {'action': 'forage', 'x': target['pos'][0], 'y': target['pos'][1]}
        
        # Default: random move
        new_x = max(0, min(bounds['width'] - 1, current_x + random.randint(-1, 1)))
        new_y = max(0, min(bounds['height'] - 1, current_y + random.randint(-1, 1)))
        return {'action': 'explore', 'x': new_x, 'y': new_y}
    
    def update(self, trajectory: List[Dict[str, Any]]) -> None:
        """Update policy using PPO."""
        # Store trajectory for batch update
        self.trajectory_buffer.extend(trajectory)
        
        # Update when we have enough data
        if len(self.trajectory_buffer) >= self.config.batch_size:
            self._ppo_update()
            self.trajectory_buffer.clear()
    
    def _ppo_update(self):
        """Perform PPO update."""
        # Convert trajectory to tensors
        states = torch.stack([self._state_to_tensor(exp['state']) for exp in self.trajectory_buffer])
        actions = torch.LongTensor([self._action_dict_to_idx(exp['action']) for exp in self.trajectory_buffer]).to(self.device)
        rewards = torch.FloatTensor([exp.get('reward', 0) for exp in self.trajectory_buffer]).to(self.device)
        
        # Calculate advantages and returns
        with torch.no_grad():
            values = self.critic(states).squeeze()
            advantages, returns = self._compute_gae(rewards, values)
            
            # Old log probabilities
            old_action_probs = self.actor(states)
            old_log_probs = torch.log(old_action_probs.gather(1, actions.unsqueeze(1))).squeeze()
        
        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Current policy
            action_probs = self.actor(states)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
            
            # Policy ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
            
            # Value loss
            current_values = self.critic(states).squeeze()
            value_loss = F.mse_loss(current_values, returns)
            
            # Total loss
            total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Update actor
            self.actor_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.config.gamma * values[t + 1] - values[t]
            gae = delta + self.config.gamma * gae_lambda * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        return advantages[:-1], returns
    
    def _action_dict_to_idx(self, action: Dict[str, Any]) -> int:
        """Convert action dictionary to index."""
        action_name = action.get('action', 'stay')
        actions = ['stay', 'north', 'south', 'east', 'west', 'hunt', 'flee', 'forage']
        try:
            return actions.index(action_name)
        except ValueError:
            return 0
    
    def save(self, filepath: str) -> None:
        """Save agent."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config
        }, filepath)

from .maddpg import MADDPGWrapper

class TrainingRunner:
    """Main training runner for comparative study."""
    
    def __init__(self, environment, config: TrainingConfig):
        self.environment = environment
        self.config = config
        self.results = {
            'lotka_volterra': {'rewards': [], 'episode_lengths': [], 'survival_rates': []},
            'epsilon_greedy': {'rewards': [], 'episode_lengths': [], 'survival_rates': []},
            'ppo': {'rewards': [], 'episode_lengths': [], 'survival_rates': []},
            'maddpg': {'rewards': [], 'episode_lengths': [], 'survival_rates': []}
        }
        
        # Create agents
        self.agents = self._create_agents()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_agents(self) -> Dict[str, Any]:
        """Create agents for each algorithm."""
        state_dim = 8  # Standardized state dimension
        action_dim = 8  # Number of discrete actions
        
        agents = {}
        
        # Lotka-Volterra agents
        agents['lotka_volterra'] = {
            'predator': LotkaVolterraAgent('predator', self.config),
            'prey': LotkaVolterraAgent('prey', self.config)
        }
        
        # Epsilon-Greedy Q-learning agents
        agents['epsilon_greedy'] = {
            'predator': EpsilonGreedyAgent('predator', self.config, state_dim, action_dim),
            'prey': EpsilonGreedyAgent('prey', self.config, state_dim, action_dim)
        }
        
        # PPO agents
        agents['ppo'] = {
            'predator': PPOAgent('predator', self.config, state_dim, action_dim),
            'prey': PPOAgent('prey', self.config, state_dim, action_dim)
        }
        
        # MADDPG agents
        agents['maddpg'] = {
            'predator': MADDPGWrapper('predator', self.config),
            'prey': MADDPGWrapper('prey', self.config)
        }
        
        return agents
    
    def run_comparative_study(self) -> Dict[str, Any]:
        """Run comparative study of all algorithms."""
        self.logger.info("Starting comparative study of RL algorithms...")
        
        for algorithm in ['lotka_volterra', 'epsilon_greedy', 'ppo', 'maddpg']:
            self.logger.info(f"Training {algorithm.upper()} agents...")
            self._train_algorithm(algorithm)
        
        # Generate comparison report
        return self._generate_report()
    
    def _train_algorithm(self, algorithm: str):
        """Train agents for a specific algorithm."""
        agents = self.agents[algorithm]
        
        for episode in range(self.config.num_episodes):
            # Reset environment
            self.environment.reset()
            
            # Episode tracking
            episode_rewards = {'predator': 0, 'prey': 0}
            episode_steps = 0
            trajectories = {'predator': [], 'prey': []}
            
            for step in range(self.config.max_steps_per_episode):
                # Get states for all agents
                states = self._get_agent_states()
                
                # Select actions
                actions = {}
                for agent_type in ['predator', 'prey']:
                    action = agents[agent_type].select_action(states[agent_type])
                    actions[agent_type] = action
                
                # Execute actions in environment
                rewards, done = self._execute_actions(actions)
                
                # Store trajectories
                for agent_type in ['predator', 'prey']:
                    trajectories[agent_type].append({
                        'state': states[agent_type],
                        'action': actions[agent_type],
                        'reward': rewards[agent_type],
                        'done': done
                    })
                    episode_rewards[agent_type] += rewards[agent_type]
                
                episode_steps += 1
                
                if done:
                    break
            
            # Update agents
            for agent_type in ['predator', 'prey']:
                agents[agent_type].update(trajectories[agent_type])
            
            # Log progress
            if episode % self.config.log_frequency == 0:
                avg_reward = np.mean(list(episode_rewards.values()))
                survival_rate = self._calculate_survival_rate()
                
                self.logger.info(
                    f"{algorithm.upper()} Episode {episode}: "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"Steps: {episode_steps}, "
                    f"Survival Rate: {survival_rate:.2f}"
                )
                
                # Store results
                self.results[algorithm]['rewards'].append(avg_reward)
                self.results[algorithm]['episode_lengths'].append(episode_steps)
                self.results[algorithm]['survival_rates'].append(survival_rate)
            
            # Save models periodically
            if episode % self.config.save_frequency == 0:
                self._save_models(algorithm, episode)
    
    def _get_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """Get state observations for all agents."""
        states = {}
        
        for agent in self.environment.agents:
            if hasattr(agent, 'get_observation'):
                state = agent.get_observation()
                agent_type = 'predator' if agent.agent_type == 'predator' else 'prey'
                states[agent_type] = state
        
        # Ensure we have states for both types
        if 'predator' not in states:
            states['predator'] = self._default_state()
        if 'prey' not in states:
            states['prey'] = self._default_state()
        
        return states
    
    def _default_state(self) -> Dict[str, Any]:
        """Default state when no agents of a type are present."""
        return {
            'energy': 0,
            'age': 0,
            'nearby_prey': [],
            'nearby_predators': [],
            'nearby_food': [],
            'temperature': 20,
            'vegetation': 0.5,
            'is_hungry': False,
            'pos_x': 0,
            'pos_y': 0,
            'environment_bounds': {'width': 100, 'height': 100}
        }
    
    def _execute_actions(self, actions: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, float], bool]:
        """Execute actions in the environment and return rewards."""
        rewards = {'predator': 0, 'prey': 0}
        
        # Apply actions to agents
        for agent in self.environment.agents:
            agent_type = 'predator' if agent.agent_type == 'predator' else 'prey'
            if agent_type in actions:
                action = actions[agent_type]
                
                # Move agent
                if 'x' in action and 'y' in action:
                    agent.x = action['x']
                    agent.y = action['y']
                
                # Calculate reward based on action outcome
                if action.get('action') == 'hunt' and agent_type == 'predator':
                    rewards[agent_type] += self._calculate_hunt_reward(agent)
                elif action.get('action') == 'forage' and agent_type == 'prey':
                    rewards[agent_type] += self._calculate_forage_reward(agent)
                elif action.get('action') == 'flee' and agent_type == 'prey':
                    rewards[agent_type] += self._calculate_flee_reward(agent)
        
        # Step environment
        self.environment.step()
        
        # Check if episode is done
        done = self._check_episode_done()
        
        return rewards, done
    
    def _calculate_hunt_reward(self, predator) -> float:
        """Calculate reward for hunting action."""
        # Reward based on proximity to prey and energy gain
        nearby_prey = [agent for agent in self.environment.agents 
                      if agent.agent_type == 'prey' and 
                      np.sqrt((agent.x - predator.x)**2 + (agent.y - predator.y)**2) < 5]
        
        if nearby_prey:
            return 10.0  # Successful hunt
        else:
            return -1.0  # Unsuccessful hunt
    
    def _calculate_forage_reward(self, prey) -> float:
        """Calculate reward for foraging action."""
        # Reward based on vegetation density
        vegetation = self.environment.vegetation_map.get((prey.x, prey.y), 0.5)
        return vegetation * 5.0
    
    def _calculate_flee_reward(self, prey) -> float:
        """Calculate reward for fleeing action."""
        # Reward based on distance from predators
        nearby_predators = [agent for agent in self.environment.agents 
                           if agent.agent_type == 'predator' and 
                           np.sqrt((agent.x - prey.x)**2 + (agent.y - prey.y)**2) < 10]
        
        if nearby_predators:
            return 5.0  # Successful escape
        else:
            return 0.0  # No immediate threat
    
    def _check_episode_done(self) -> bool:
        """Check if episode should end."""
        # End if all agents of one type are dead or max steps reached
        predators = [a for a in self.environment.agents if a.agent_type == 'predator']
        prey = [a for a in self.environment.agents if a.agent_type == 'prey']
        
        return len(predators) == 0 or len(prey) == 0
    
    def _calculate_survival_rate(self) -> float:
        """Calculate current survival rate."""
        total_agents = len(self.environment.agents)
        if total_agents == 0:
            return 0.0
        
        alive_agents = sum(1 for agent in self.environment.agents if agent.energy > 0)
        return alive_agents / total_agents
    
    def _save_models(self, algorithm: str, episode: int):
        """Save trained models."""
        save_dir = Path(f"models/{algorithm}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for agent_type, agent in self.agents[algorithm].items():
            if hasattr(agent, 'save'):
                filepath = save_dir / f"{agent_type}_episode_{episode}.pth"
                agent.save(str(filepath))
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comparative analysis report."""
        report = {
            'summary': {},
            'detailed_results': self.results,
            'recommendations': []
        }
        
        # Calculate summary statistics
        for algorithm in self.results:
            if self.results[algorithm]['rewards']:
                report['summary'][algorithm] = {
                    'avg_reward': np.mean(self.results[algorithm]['rewards']),
                    'std_reward': np.std(self.results[algorithm]['rewards']),
                    'avg_episode_length': np.mean(self.results[algorithm]['episode_lengths']),
                    'avg_survival_rate': np.mean(self.results[algorithm]['survival_rates'])
                }
        
        # Generate recommendations
        if report['summary']:
            best_reward_alg = max(report['summary'], 
                                key=lambda x: report['summary'][x]['avg_reward'])
            best_survival_alg = max(report['summary'], 
                                  key=lambda x: report['summary'][x]['avg_survival_rate'])
            
            report['recommendations'] = [
                f"Best performing algorithm by reward: {best_reward_alg.upper()}",
                f"Best performing algorithm by survival: {best_survival_alg.upper()}",
                "Consider ensemble methods combining multiple algorithms",
                "Fine-tune hyperparameters for the best performing algorithms"
            ]
        
        return report
    
    def save_results(self, filepath: str):
        """Save training results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")

def create_training_environment():
    """Create a standardized training environment."""
    from bioflux.core.environment import Environment
    from bioflux.core.agents import Predator, Prey, Plant
    
    # Create environment with fixed settings for fair comparison
    env = Environment()
    env.width = 100
    env.height = 100
    
    # Add initial agents
    for _ in range(5):  # 5 predators
        predator = Predator(
            speed=2,
            energy=100,
            pos_x=np.random.randint(0, 100),
            pos_y=np.random.randint(0, 100),
            age=1
        )
        env.add_predator(predator)
    
    for _ in range(20):  # 20 prey
        prey = Prey(
            speed=3,
            energy=50,
            pos_x=np.random.randint(0, 100),
            pos_y=np.random.randint(0, 100),
            age=1
        )
        env.add_prey(prey)
    
    for _ in range(100):  # 100 plants
        plant = Plant(
            energy=10,
            pos_x=np.random.randint(0, 100),
            pos_y=np.random.randint(0, 100)
        )
        env.add_plant(plant)
    
    return env

# Export all classes and functions
__all__ = [
    'TrainingConfig',
    'LotkaVolterraAgent',
    'EpsilonGreedyAgent', 
    'PPOAgent',
    'MADDPGWrapper',
    'TrainingRunner',
    'create_training_environment',
    'ReplayBuffer'
]
