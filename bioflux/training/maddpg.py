#!/usr/bin/env python3
"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation for BioFlux.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import random
from collections import deque
import copy

class Actor(nn.Module):
    """Actor network for MADDPG."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output actions in [-1, 1]
        )
    
    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    """Critic network for MADDPG (takes all agents' states and actions)."""
    
    def __init__(self, total_state_dim: int, total_action_dim: int, hidden_dim: int = 128):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.network(x)

class OUNoise:
    """Ornstein-Uhlenbeck process for action exploration."""
    
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def noise(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state

class MADDPGAgent:
    """MADDPG agent implementation."""
    
    def __init__(self, agent_id: int, agent_type: str, state_dim: int, action_dim: int, 
                 total_state_dim: int, total_action_dim: int, config):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        self.device = torch.device(config.device)
        
        # Networks
        self.actor = Actor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = Critic(total_state_dim, total_action_dim, config.hidden_dim).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # Exploration noise
        self.ou_noise = OUNoise(action_dim)
        
        # Hyperparameters
        self.tau = 0.01  # Soft update parameter
        self.exploration_noise = 0.1
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Select action using actor network."""
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()
        
        if training:
            # Add exploration noise
            noise = torch.FloatTensor(self.ou_noise.noise()).to(self.device)
            action = action + self.exploration_noise * noise
            action = torch.clamp(action, -1, 1)
        
        return action
    
    def update(self, replay_buffer, agents_list, batch_size: int):
        """Update actor and critic networks."""
        if len(replay_buffer) < batch_size:
            return
        
        # Sample batch
        batch = replay_buffer.sample(batch_size)
        
        # Extract batch data
        states = torch.stack([exp.state for exp in batch]).to(self.device)
        actions = torch.stack([exp.action for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_states = torch.stack([exp.next_state for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in batch]).to(self.device)
        
        # Get agent-specific data
        agent_states = states[:, self.agent_id]
        agent_next_states = next_states[:, self.agent_id]
        agent_rewards = rewards[:, self.agent_id]
        
        # Critic update
        with torch.no_grad():
            # Get next actions from all target actors
            next_actions = []
            for i, agent in enumerate(agents_list):
                next_action = agent.target_actor(next_states[:, i])
                next_actions.append(next_action)
            next_actions = torch.stack(next_actions, dim=1)
            
            # Flatten for critic input
            next_states_flat = next_states.view(batch_size, -1)
            next_actions_flat = next_actions.view(batch_size, -1)
            
            target_q = self.target_critic(next_states_flat, next_actions_flat).squeeze()
            target_q = agent_rewards + (self.config.gamma * target_q * ~dones[:, self.agent_id])
        
        # Current Q value
        states_flat = states.view(batch_size, -1)
        actions_flat = actions.view(batch_size, -1)
        current_q = self.critic(states_flat, actions_flat).squeeze()
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        # Get current actions from all actors
        current_actions = []
        for i, agent in enumerate(agents_list):
            if i == self.agent_id:
                current_action = self.actor(states[:, i])
            else:
                current_action = agent.actor(states[:, i]).detach()
            current_actions.append(current_action)
        current_actions = torch.stack(current_actions, dim=1)
        current_actions_flat = current_actions.view(batch_size, -1)
        
        # Actor loss (maximize Q value)
        actor_loss = -self.critic(states_flat, current_actions_flat).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)
    
    def soft_update(self, target_net, source_net):
        """Soft update target network."""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )
    
    def save(self, filepath: str):
        """Save agent networks."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent networks."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

class MADDPGWrapper:
    """Wrapper to make MADDPG compatible with BioFlux interface."""
    
    def __init__(self, agent_type: str, config, num_agents: int = 4):
        self.agent_type = agent_type
        self.config = config
        self.num_agents = num_agents
        
        # State and action dimensions
        self.state_dim = 8  # Same as other agents
        self.action_dim = 3  # [x_move, y_move, action_type]
        self.total_state_dim = self.state_dim * num_agents
        self.total_action_dim = self.action_dim * num_agents
        
        # Create MADDPG agents
        self.agents = []
        for i in range(num_agents):
            agent = MADDPGAgent(
                agent_id=i,
                agent_type=agent_type,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                total_state_dim=self.total_state_dim,
                total_action_dim=self.total_action_dim,
                config=config
            )
            self.agents.append(agent)
        
        # Shared replay buffer
        from bioflux.training import ReplayBuffer, Experience
        self.replay_buffer = ReplayBuffer(config.memory_size)
        
        self.training_step = 0
    
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
        return torch.FloatTensor(features)
    
    def select_action(self, state: Dict[str, Any], agent_id: int = 0) -> Dict[str, Any]:
        """Select action for specific agent."""
        state_tensor = self._state_to_tensor(state).unsqueeze(0)
        
        # For demonstration, we'll use the first agent
        action_tensor = self.agents[agent_id].select_action(state_tensor)
        
        return self._tensor_to_action_dict(action_tensor.cpu().numpy()[0], state)
    
    def _tensor_to_action_dict(self, action: np.ndarray, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert action tensor to action dictionary."""
        x_move, y_move, action_type = action
        
        current_x = state.get('pos_x', 0)
        current_y = state.get('pos_y', 0)
        bounds = state.get('environment_bounds', {'width': 100, 'height': 100})
        
        # Convert continuous action to discrete movement
        new_x = current_x + int(np.round(x_move * 2))  # Scale to [-2, 2]
        new_y = current_y + int(np.round(y_move * 2))
        
        new_x = max(0, min(bounds['width'] - 1, new_x))
        new_y = max(0, min(bounds['height'] - 1, new_y))
        
        # Determine action type based on continuous output
        if action_type > 0.5 and self.agent_type == 'predator':
            nearby_prey = state.get('nearby_prey', [])
            if nearby_prey:
                return {'action': 'hunt', 'x': new_x, 'y': new_y}
        elif action_type > 0.5 and self.agent_type == 'prey':
            nearby_predators = state.get('nearby_predators', [])
            if nearby_predators:
                return {'action': 'flee', 'x': new_x, 'y': new_y}
            nearby_food = state.get('nearby_food', [])
            if nearby_food:
                return {'action': 'forage', 'x': new_x, 'y': new_y}
        
        return {'action': 'move', 'x': new_x, 'y': new_y}
    
    def update(self, trajectory: List[Dict[str, Any]]) -> None:
        """Update MADDPG agents."""
        # For simplicity, we'll update using the first agent's experience
        # In a full MADDPG implementation, you'd need coordinated multi-agent experiences
        
        for i in range(len(trajectory) - 1):
            state = trajectory[i]['state']
            action = trajectory[i]['action']
            reward = trajectory[i].get('reward', 0)
            next_state = trajectory[i + 1]['state']
            done = trajectory[i].get('done', False)
            
            # Convert to tensors for storage
            state_tensor = self._state_to_tensor(state)
            next_state_tensor = self._state_to_tensor(next_state)
            action_tensor = torch.FloatTensor([0.0, 0.0, 0.0])  # Simplified
            
            # Create multi-agent experience (simplified)
            multi_state = torch.stack([state_tensor] * self.num_agents)
            multi_action = torch.stack([action_tensor] * self.num_agents)
            multi_next_state = torch.stack([next_state_tensor] * self.num_agents)
            multi_reward = [reward] * self.num_agents
            multi_done = [done] * self.num_agents
            
            from bioflux.training import Experience
            self.replay_buffer.push(
                multi_state, multi_action, multi_reward, multi_next_state, multi_done
            )
        
        # Update agents
        if len(self.replay_buffer) >= self.config.batch_size:
            for agent in self.agents:
                agent.update(self.replay_buffer, self.agents, self.config.batch_size)
        
        self.training_step += 1
    
    def save(self, filepath: str) -> None:
        """Save all MADDPG agents."""
        for i, agent in enumerate(self.agents):
            agent_path = filepath.replace('.pth', f'_agent_{i}.pth')
            agent.save(agent_path)
