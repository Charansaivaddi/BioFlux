"""
BioFlux Agent Classes - Refactored and Optimized

This module contains the core agent classes for the BioFlux ecosystem simulation.
All agents are RL-enabled with improved architecture and cleaner separation of concerns.

Key improvements:
- Better separation of concerns
- Reduced code duplication
- Improved performance
- Enhanced configurability
- Cleaner interfaces
"""

import random
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging


# Configure logging
logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Enumeration of possible agent actions."""
    STAY = "stay"
    MOVE = "move"
    HUNT = "hunt"
    FLEE = "flee"
    FORAGE = "forage"
    EXPLORE = "explore"
    REPRODUCE = "reproduce"
    COMPETE = "compete"
    DORMANT = "dormant"


@dataclass
class AgentState:
    """Data class for agent state information."""
    energy: float
    pos_x: float
    pos_y: float
    age: int
    is_alive: bool = True
    is_hungry: bool = True
    is_moving: bool = False
    is_eating: bool = False


@dataclass
class Environment_Info:
    """Data class for environmental information."""
    width: int
    height: int
    temperature: float = 20.0
    humidity: float = 60.0
    vegetation: float = 0.5
    light: float = 0.8
    nutrients: float = 0.5


@dataclass
class Action:
    """Data class for agent actions."""
    action_type: ActionType
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    energy_cost: float = 1.0
    terrain_difficulty: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the BioFlux ecosystem.
    
    Provides common functionality and enforces interface contracts.
    """
    
    def __init__(self, 
                 agent_id: str,
                 agent_type: str,
                 initial_energy: float,
                 pos_x: float,
                 pos_y: float,
                 age: int = 0,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (predator, prey, plant)
            initial_energy: Starting energy level
            pos_x, pos_y: Initial position
            age: Initial age
            config: Configuration parameters
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        
        # Core state
        self.state = AgentState(
            energy=initial_energy,
            pos_x=pos_x,
            pos_y=pos_y,
            age=age
        )
        
        # RL components
        self.experience_buffer: List[Dict[str, Any]] = []
        self.policy_params: Dict[str, Any] = {}
        
        # Performance tracking
        self.metrics = {
            'total_energy_consumed': 0.0,
            'total_distance_traveled': 0.0,
            'actions_taken': 0,
            'survival_time': 0
        }
        
        # Initialize agent-specific parameters
        self._initialize_agent_params()
    
    @abstractmethod
    def _initialize_agent_params(self) -> None:
        """Initialize agent-specific parameters."""
        pass
    
    @abstractmethod
    def get_observation(self, environment: Any) -> Dict[str, Any]:
        """Get environmental observation for decision making."""
        pass
    
    @abstractmethod
    def select_action(self, observation: Dict[str, Any]) -> Action:
        """Select action based on current observation."""
        pass
    
    @abstractmethod
    def _calculate_reward(self, action: Action, environment: Any) -> float:
        """Calculate reward for the taken action."""
        pass
    
    def execute_action(self, action: Action, environment: Any) -> bool:
        """
        Execute the selected action and update agent state.
        
        Args:
            action: Action to execute
            environment: Environment context
            
        Returns:
            bool: True if action was successful
        """
        if not self.state.is_alive:
            return False
        
        success = False
        
        try:
            # Execute action based on type
            if action.action_type == ActionType.MOVE:
                success = self._execute_move(action, environment)
            elif action.action_type == ActionType.STAY:
                success = self._execute_stay(action)
            else:
                success = self._execute_specific_action(action, environment)
            
            # Update metrics
            if success:
                self.metrics['actions_taken'] += 1
                self._update_state_after_action(action)
            
            # Calculate and store reward
            reward = self._calculate_reward(action, environment)
            self._add_experience(action, reward, environment)
            
        except Exception as e:
            logger.error(f"Error executing action for {self.agent_id}: {e}")
            success = False
        
        return success
    
    def _execute_move(self, action: Action, environment: Any) -> bool:
        """Execute movement action."""
        if action.target_x is None or action.target_y is None:
            return False
        
        # Validate movement bounds
        new_x = max(0, min(environment.width - 1, action.target_x))
        new_y = max(0, min(environment.height - 1, action.target_y))
        
        # Calculate energy cost
        distance = self._calculate_distance(
            self.state.pos_x, self.state.pos_y, new_x, new_y
        )
        energy_cost = distance * action.terrain_difficulty * self._get_movement_efficiency()
        
        # Check if agent has enough energy
        if self.state.energy < energy_cost:
            return False
        
        # Update position and energy
        self.state.pos_x = new_x
        self.state.pos_y = new_y
        self.state.energy -= energy_cost
        self.state.is_moving = True
        
        # Update metrics
        self.metrics['total_distance_traveled'] += distance
        self.metrics['total_energy_consumed'] += energy_cost
        
        return True
    
    def _execute_stay(self, action: Action) -> bool:
        """Execute stay action."""
        self.state.is_moving = False
        # Small energy cost for staying alive
        self.state.energy -= 0.1
        return True
    
    @abstractmethod
    def _execute_specific_action(self, action: Action, environment: Any) -> bool:
        """Execute agent-specific actions."""
        pass
    
    def _update_state_after_action(self, action: Action) -> None:
        """Update agent state after action execution."""
        # Age increment
        self.state.age += 1
        self.metrics['survival_time'] += 1
        
        # Check for death conditions
        if self.state.energy <= 0 or self.state.age >= self._get_max_age():
            self._die()
    
    def _add_experience(self, action: Action, reward: float, environment: Any) -> None:
        """Add experience to buffer for learning."""
        experience = {
            'state': self.get_observation(environment),
            'action': action.__dict__,
            'reward': reward,
            'timestamp': self.state.age
        }
        
        self.experience_buffer.append(experience)
        
        # Limit buffer size
        max_buffer_size = self.config.get('max_experience_buffer', 1000)
        if len(self.experience_buffer) > max_buffer_size:
            self.experience_buffer.pop(0)
    
    def update_policy(self, trajectory: List[Dict[str, Any]]) -> None:
        """Update agent policy based on experience trajectory."""
        if not trajectory:
            return
        
        # Simple reward-based policy update
        total_reward = sum(exp.get('reward', 0) for exp in trajectory)
        
        # Update policy parameters based on performance
        self._update_policy_params(total_reward, trajectory)
    
    @abstractmethod
    def _update_policy_params(self, total_reward: float, trajectory: List[Dict[str, Any]]) -> None:
        """Update agent-specific policy parameters."""
        pass
    
    def _die(self) -> None:
        """Handle agent death."""
        self.state.is_alive = False
        self.state.energy = 0
        self.state.is_moving = False
        self.state.is_eating = False
        logger.info(f"Agent {self.agent_id} died at age {self.state.age}")
    
    # Utility methods
    @staticmethod
    def _calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    @abstractmethod
    def _get_movement_efficiency(self) -> float:
        """Get movement efficiency multiplier."""
        pass
    
    @abstractmethod
    def _get_max_age(self) -> int:
        """Get maximum age for this agent type."""
        pass
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get agent state as dictionary."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'state': self.state.__dict__,
            'metrics': self.metrics.copy(),
            'policy_params': self.policy_params.copy()
        }
    
    def __str__(self) -> str:
        return (f"{self.agent_type.title()}({self.agent_id}: "
                f"energy={self.state.energy:.1f}, "
                f"pos=({self.state.pos_x:.1f},{self.state.pos_y:.1f}), "
                f"age={self.state.age}, alive={self.state.is_alive})")
    
    def __repr__(self) -> str:
        return self.__str__()


class PredatorAgent(BaseAgent):
    """
    Predator agent with hunting and survival behaviors.
    
    Predators hunt prey, compete with other predators, and adapt their
    hunting strategies based on success/failure.
    """
    
    def _initialize_agent_params(self) -> None:
        """Initialize predator-specific parameters."""
        self.policy_params.update({
            'hunting_aggression': self.config.get('hunting_aggression', 1.0),
            'hunt_success_rate': 0.0,
            'energy_efficiency': self.config.get('energy_efficiency', 1.0),
            'detection_radius': self.config.get('detection_radius', 5),
            'hunt_threshold': self.config.get('hunt_threshold', 3)
        })
        
        # Predator-specific metrics
        self.metrics.update({
            'kills': 0,
            'hunt_attempts': 0,
            'prey_consumed': 0.0
        })
    
    def get_observation(self, environment: Any) -> Dict[str, Any]:
        """Get environmental observation for predator decision making."""
        # Detect nearby prey
        nearby_prey = self._detect_nearby_agents(
            environment.prey, self.policy_params['detection_radius']
        )
        
        # Detect nearby predators (competition)
        nearby_predators = self._detect_nearby_agents(
            environment.predators, self.policy_params['detection_radius']
        )
        
        # Get environmental conditions
        local_env = environment.get_local_environment(
            int(self.state.pos_x), int(self.state.pos_y)
        )
        
        return {
            'agent_state': self.state.__dict__,
            'nearby_prey': nearby_prey,
            'nearby_predators': len([p for p in nearby_predators if p != self]),
            'environment': local_env,
            'bounds': {'width': environment.width, 'height': environment.height}
        }
    
    def select_action(self, observation: Dict[str, Any]) -> Action:
        """Select action using predator policy."""
        if not self.state.is_alive:
            return Action(ActionType.STAY)
        
        nearby_prey = observation.get('nearby_prey', [])
        environment_bounds = observation.get('bounds', {'width': 50, 'height': 50})
        
        # Hunt if prey is nearby and energy is sufficient
        if nearby_prey and self.state.energy > 20:
            return self._plan_hunt_action(nearby_prey, environment_bounds)
        
        # Explore if energy is high enough
        if self.state.energy > 30:
            return self._plan_explore_action(environment_bounds)
        
        # Stay and conserve energy if low
        return Action(ActionType.STAY)
    
    def _plan_hunt_action(self, nearby_prey: List[Dict], bounds: Dict[str, int]) -> Action:
        """Plan hunting action targeting nearest prey."""
        # Select nearest prey
        target_prey = min(nearby_prey, 
                         key=lambda p: self._calculate_distance(
                             self.state.pos_x, self.state.pos_y, 
                             p['pos'][0], p['pos'][1]
                         ))
        
        target_x, target_y = target_prey['pos']
        
        # Move towards prey
        dx = 1 if target_x > self.state.pos_x else -1 if target_x < self.state.pos_x else 0
        dy = 1 if target_y > self.state.pos_y else -1 if target_y < self.state.pos_y else 0
        
        new_x = max(0, min(bounds['width'] - 1, self.state.pos_x + dx))
        new_y = max(0, min(bounds['height'] - 1, self.state.pos_y + dy))
        
        # Check if close enough to hunt
        distance = self._calculate_distance(self.state.pos_x, self.state.pos_y, target_x, target_y)
        action_type = ActionType.HUNT if distance < self.policy_params['hunt_threshold'] else ActionType.MOVE
        
        return Action(
            action_type=action_type,
            target_x=new_x,
            target_y=new_y,
            energy_cost=2.0 * self.policy_params['hunting_aggression'],
            metadata={'target_prey': target_prey}
        )
    
    def _plan_explore_action(self, bounds: Dict[str, int]) -> Action:
        """Plan exploration action."""
        # Biased random walk
        dx = random.choice([-2, -1, 0, 1, 2])
        dy = random.choice([-2, -1, 0, 1, 2])
        
        new_x = max(0, min(bounds['width'] - 1, self.state.pos_x + dx))
        new_y = max(0, min(bounds['height'] - 1, self.state.pos_y + dy))
        
        return Action(
            action_type=ActionType.EXPLORE,
            target_x=new_x,
            target_y=new_y,
            energy_cost=1.5
        )
    
    def _execute_specific_action(self, action: Action, environment: Any) -> bool:
        """Execute predator-specific actions."""
        if action.action_type == ActionType.HUNT:
            return self._execute_hunt(action, environment)
        elif action.action_type == ActionType.EXPLORE:
            return self._execute_move(action, environment)
        return False
    
    def _execute_hunt(self, action: Action, environment: Any) -> bool:
        """Execute hunting action."""
        self.metrics['hunt_attempts'] += 1
        
        # Find prey in hunting range
        nearby_prey = self._detect_nearby_agents(environment.prey, 2)
        
        if nearby_prey:
            # Attempt to catch prey
            success_rate = self.policy_params['hunting_aggression'] * 0.3
            if random.random() < success_rate:
                # Successful hunt
                prey = nearby_prey[0]  # Catch first prey
                energy_gained = min(30, prey.state.energy)
                
                self.state.energy += energy_gained
                self.metrics['kills'] += 1
                self.metrics['prey_consumed'] += energy_gained
                
                # Damage or kill prey
                prey.state.energy -= 40
                if prey.state.energy <= 0:
                    prey._die()
                
                return True
        
        # Failed hunt - still costs energy
        self.state.energy -= action.energy_cost
        return False
    
    def _detect_nearby_agents(self, agent_list: List, radius: float) -> List:
        """Detect agents within specified radius."""
        nearby = []
        for agent in agent_list:
            if agent.state.is_alive and agent != self:
                distance = self._calculate_distance(
                    self.state.pos_x, self.state.pos_y,
                    agent.state.pos_x, agent.state.pos_y
                )
                if distance <= radius:
                    nearby.append({
                        'agent': agent,
                        'distance': distance,
                        'pos': (agent.state.pos_x, agent.state.pos_y),
                        'energy': agent.state.energy
                    })
        return nearby
    
    def _calculate_reward(self, action: Action, environment: Any) -> float:
        """Calculate reward for predator actions."""
        reward = 0.0
        
        # Base survival reward
        reward += 1.0
        
        # Action-specific rewards
        if action.action_type == ActionType.HUNT:
            if self.metrics['hunt_attempts'] > 0:
                recent_success_rate = self.metrics['kills'] / self.metrics['hunt_attempts']
                reward += recent_success_rate * 20.0
            else:
                reward -= 5.0  # Failed hunt penalty
        
        elif action.action_type == ActionType.EXPLORE:
            reward += 2.0  # Exploration bonus
        
        # Energy level reward
        if self.state.energy > 80:
            reward += 5.0
        elif self.state.energy < 20:
            reward -= 10.0
        
        return reward
    
    def _update_policy_params(self, total_reward: float, trajectory: List[Dict[str, Any]]) -> None:
        """Update predator policy parameters."""
        # Update hunting success rate
        recent_kills = sum(1 for exp in trajectory[-10:] 
                          if exp.get('action', {}).get('action_type') == 'hunt' 
                          and exp.get('reward', 0) > 15)
        recent_hunts = sum(1 for exp in trajectory[-10:] 
                          if exp.get('action', {}).get('action_type') == 'hunt')
        
        if recent_hunts > 0:
            self.policy_params['hunt_success_rate'] = recent_kills / recent_hunts
        
        # Adjust hunting aggression based on success
        if total_reward > 0:
            self.policy_params['hunting_aggression'] = min(2.0, 
                self.policy_params['hunting_aggression'] + 0.1)
        else:
            self.policy_params['hunting_aggression'] = max(0.5, 
                self.policy_params['hunting_aggression'] - 0.05)
    
    def _get_movement_efficiency(self) -> float:
        """Predators are moderately efficient movers."""
        return 1.0 * self.policy_params['energy_efficiency']
    
    def _get_max_age(self) -> int:
        """Maximum age for predators."""
        return self.config.get('max_age', 25)


class PreyAgent(BaseAgent):
    """
    Prey agent with foraging and escape behaviors.
    
    Prey forage for food, avoid predators, and adapt their escape
    strategies based on survival success.
    """
    
    def _initialize_agent_params(self) -> None:
        """Initialize prey-specific parameters."""
        self.policy_params.update({
            'escape_distance': self.config.get('escape_distance', 4),
            'foraging_efficiency': self.config.get('foraging_efficiency', 1.0),
            'caution_level': self.config.get('caution_level', 0.7),
            'detection_radius': self.config.get('detection_radius', 6),
            'hunger_threshold': self.config.get('hunger_threshold', 30)
        })
        
        # Prey-specific metrics
        self.metrics.update({
            'escapes': 0,
            'food_consumed': 0.0,
            'predator_encounters': 0
        })
    
    def get_observation(self, environment: Any) -> Dict[str, Any]:
        """Get environmental observation for prey decision making."""
        # Detect nearby predators (danger)
        nearby_predators = self._detect_nearby_agents(
            environment.predators, self.policy_params['detection_radius']
        )
        
        # Detect nearby food sources
        nearby_food = self._detect_nearby_agents(
            environment.plants, 4
        )
        
        # Detect other prey (herd behavior)
        nearby_prey = self._detect_nearby_agents(
            environment.prey, 3
        )
        
        # Get environmental conditions
        local_env = environment.get_local_environment(
            int(self.state.pos_x), int(self.state.pos_y)
        )
        
        return {
            'agent_state': self.state.__dict__,
            'nearby_predators': nearby_predators,
            'nearby_food': nearby_food,
            'nearby_prey': len([p for p in nearby_prey if p != self]),
            'environment': local_env,
            'bounds': {'width': environment.width, 'height': environment.height}
        }
    
    def select_action(self, observation: Dict[str, Any]) -> Action:
        """Select action using prey policy."""
        if not self.state.is_alive:
            return Action(ActionType.STAY)
        
        nearby_predators = observation.get('nearby_predators', [])
        nearby_food = observation.get('nearby_food', [])
        environment_bounds = observation.get('bounds', {'width': 50, 'height': 50})
        
        # Escape from predators (highest priority)
        if nearby_predators:
            closest_distance = min(p['distance'] for p in nearby_predators)
            if closest_distance <= self.policy_params['escape_distance']:
                self.metrics['predator_encounters'] += 1
                return self._plan_escape_action(nearby_predators, environment_bounds)
        
        # Forage for food if hungry
        if self.state.energy < self.policy_params['hunger_threshold'] and nearby_food:
            return self._plan_forage_action(nearby_food, environment_bounds)
        
        # Explore for food or better position
        return self._plan_explore_action(environment_bounds)
    
    def _plan_escape_action(self, nearby_predators: List[Dict], bounds: Dict[str, int]) -> Action:
        """Plan escape action from predators."""
        # Find average predator position to escape from
        avg_pred_x = sum(p['pos'][0] for p in nearby_predators) / len(nearby_predators)
        avg_pred_y = sum(p['pos'][1] for p in nearby_predators) / len(nearby_predators)
        
        # Escape in opposite direction
        escape_dx = self.state.pos_x - avg_pred_x
        escape_dy = self.state.pos_y - avg_pred_y
        
        # Normalize and amplify escape direction
        length = max(1, np.sqrt(escape_dx**2 + escape_dy**2))
        escape_dx = (escape_dx / length) * 3  # Move 3 units away
        escape_dy = (escape_dy / length) * 3
        
        new_x = max(0, min(bounds['width'] - 1, self.state.pos_x + escape_dx))
        new_y = max(0, min(bounds['height'] - 1, self.state.pos_y + escape_dy))
        
        return Action(
            action_type=ActionType.FLEE,
            target_x=new_x,
            target_y=new_y,
            energy_cost=2.5,  # Escaping is energy-intensive
            terrain_difficulty=0.8  # Adrenaline makes movement easier
        )
    
    def _plan_forage_action(self, nearby_food: List[Dict], bounds: Dict[str, int]) -> Action:
        """Plan foraging action."""
        # Select nearest food source
        target_food = min(nearby_food, 
                         key=lambda f: f['distance'])
        
        target_x, target_y = target_food['pos']
        
        # Move towards food
        dx = 1 if target_x > self.state.pos_x else -1 if target_x < self.state.pos_x else 0
        dy = 1 if target_y > self.state.pos_y else -1 if target_y < self.state.pos_y else 0
        
        new_x = max(0, min(bounds['width'] - 1, self.state.pos_x + dx))
        new_y = max(0, min(bounds['height'] - 1, self.state.pos_y + dy))
        
        return Action(
            action_type=ActionType.FORAGE,
            target_x=new_x,
            target_y=new_y,
            energy_cost=1.0,
            metadata={'target_food': target_food}
        )
    
    def _plan_explore_action(self, bounds: Dict[str, int]) -> Action:
        """Plan exploration action."""
        # Conservative random walk
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        
        new_x = max(0, min(bounds['width'] - 1, self.state.pos_x + dx))
        new_y = max(0, min(bounds['height'] - 1, self.state.pos_y + dy))
        
        return Action(
            action_type=ActionType.EXPLORE,
            target_x=new_x,
            target_y=new_y,
            energy_cost=0.8
        )
    
    def _execute_specific_action(self, action: Action, environment: Any) -> bool:
        """Execute prey-specific actions."""
        if action.action_type == ActionType.FLEE:
            success = self._execute_move(action, environment)
            if success:
                self.metrics['escapes'] += 1
            return success
        
        elif action.action_type == ActionType.FORAGE:
            return self._execute_forage(action, environment)
        
        elif action.action_type == ActionType.EXPLORE:
            return self._execute_move(action, environment)
        
        return False
    
    def _execute_forage(self, action: Action, environment: Any) -> bool:
        """Execute foraging action."""
        # Find food in foraging range
        nearby_food = self._detect_nearby_agents(environment.plants, 2)
        
        if nearby_food:
            # Consume food
            plant = nearby_food[0]['agent']  # Consume from first plant
            energy_gained = min(15, plant.state.energy)
            
            self.state.energy += energy_gained * self.policy_params['foraging_efficiency']
            self.metrics['food_consumed'] += energy_gained
            
            # Reduce plant energy
            plant.state.energy -= energy_gained
            if plant.state.energy <= 0:
                plant._die()
            
            self.state.is_eating = True
            return True
        
        # No food found - still costs energy to search
        self.state.energy -= action.energy_cost
        return False
    
    def _detect_nearby_agents(self, agent_list: List, radius: float) -> List:
        """Detect agents within specified radius."""
        nearby = []
        for agent in agent_list:
            if agent.state.is_alive and agent != self:
                distance = self._calculate_distance(
                    self.state.pos_x, self.state.pos_y,
                    agent.state.pos_x, agent.state.pos_y
                )
                if distance <= radius:
                    nearby.append({
                        'agent': agent,
                        'distance': distance,
                        'pos': (agent.state.pos_x, agent.state.pos_y),
                        'energy': agent.state.energy
                    })
        return nearby
    
    def _calculate_reward(self, action: Action, environment: Any) -> float:
        """Calculate reward for prey actions."""
        reward = 0.0
        
        # Base survival reward
        reward += 1.0
        
        # Action-specific rewards
        if action.action_type == ActionType.FLEE:
            reward += 8.0  # High reward for escaping
        
        elif action.action_type == ActionType.FORAGE:
            if self.state.is_eating:
                reward += 10.0  # Successful foraging
            else:
                reward -= 2.0  # Failed foraging
        
        elif action.action_type == ActionType.EXPLORE:
            reward += 1.0  # Small exploration bonus
        
        # Energy level reward
        if self.state.energy > 40:
            reward += 3.0
        elif self.state.energy < 15:
            reward -= 8.0
        
        # Proximity to predators penalty
        nearby_predators = self._detect_nearby_agents(environment.predators, 5)
        if nearby_predators:
            min_distance = min(p['distance'] for p in nearby_predators)
            reward -= max(0, 10 - min_distance)  # Penalty decreases with distance
        
        return reward
    
    def _update_policy_params(self, total_reward: float, trajectory: List[Dict[str, Any]]) -> None:
        """Update prey policy parameters."""
        # Adjust escape distance based on survival success
        predator_encounters = sum(1 for exp in trajectory[-10:] 
                                 if exp.get('action', {}).get('action_type') == 'flee')
        
        if predator_encounters > 0 and total_reward > 0:
            # Successful escapes - current strategy is working
            pass
        elif predator_encounters > 0 and total_reward < 0:
            # Failed escapes - increase caution
            self.policy_params['escape_distance'] = min(8, 
                self.policy_params['escape_distance'] + 0.5)
            self.policy_params['caution_level'] = min(1.0, 
                self.policy_params['caution_level'] + 0.1)
        
        # Adjust foraging efficiency based on food success
        foraging_attempts = sum(1 for exp in trajectory[-10:] 
                               if exp.get('action', {}).get('action_type') == 'forage')
        successful_foraging = sum(1 for exp in trajectory[-10:] 
                                 if exp.get('action', {}).get('action_type') == 'forage' 
                                 and exp.get('reward', 0) > 5)
        
        if foraging_attempts > 0:
            success_rate = successful_foraging / foraging_attempts
            if success_rate > 0.7:
                self.policy_params['foraging_efficiency'] = min(1.5, 
                    self.policy_params['foraging_efficiency'] + 0.05)
    
    def _get_movement_efficiency(self) -> float:
        """Prey are efficient movers."""
        return 0.8 * self.policy_params['foraging_efficiency']
    
    def _get_max_age(self) -> int:
        """Maximum age for prey."""
        return self.config.get('max_age', 20)


class PlantAgent(BaseAgent):
    """
    Plant agent with growth, reproduction, and resource competition behaviors.
    
    Plants grow based on environmental conditions, compete for resources,
    and reproduce when conditions are favorable.
    """
    
    def _initialize_agent_params(self) -> None:
        """Initialize plant-specific parameters."""
        self.policy_params.update({
            'growth_rate': self.config.get('growth_rate', 0.5),
            'reproduction_threshold': self.config.get('reproduction_threshold', 50.0),
            'stress_tolerance': self.config.get('stress_tolerance', 0.6),
            'resource_competition': self.config.get('resource_competition', 0.8),
            'maturity_age': self.config.get('maturity_age', 10)
        })
        
        # Plant-specific state
        self.state.is_mature = self.state.age >= self.policy_params['maturity_age']
        
        # Plant-specific metrics
        self.metrics.update({
            'growth_cycles': 0,
            'reproductions': 0,
            'resource_competition_events': 0
        })
    
    def get_observation(self, environment: Any) -> Dict[str, Any]:
        """Get environmental observation for plant decision making."""
        # Get local environmental conditions
        local_env = environment.get_local_environment(
            int(self.state.pos_x), int(self.state.pos_y)
        )
        
        # Detect nearby competing plants
        nearby_plants = self._detect_nearby_agents(environment.plants, 3)
        
        # Calculate resource availability
        light_availability = self._calculate_light_availability(local_env, nearby_plants)
        water_availability = local_env.get('humidity', 60) / 100.0
        nutrient_availability = local_env.get('vegetation', 0.5)
        
        return {
            'agent_state': self.state.__dict__,
            'light': light_availability,
            'water': water_availability,
            'nutrients': nutrient_availability,
            'competition_level': len(nearby_plants) / 10.0,
            'environment': local_env,
            'can_reproduce': (self.state.energy > self.policy_params['reproduction_threshold'] 
                            and self.state.is_mature),
            'stress_level': self._calculate_stress_level(local_env)
        }
    
    def select_action(self, observation: Dict[str, Any]) -> Action:
        """Select action using plant policy."""
        if not self.state.is_alive:
            return Action(ActionType.STAY)
        
        stress_level = observation.get('stress_level', 0)
        can_reproduce = observation.get('can_reproduce', False)
        competition_level = observation.get('competition_level', 0)
        
        # Go dormant under high stress
        if stress_level > 0.8:
            return Action(
                action_type=ActionType.DORMANT,
                energy_cost=0.1,
                metadata={'conservation_factor': 0.9}
            )
        
        # Reproduce if conditions are favorable
        if can_reproduce and stress_level < 0.3:
            reproduction_prob = self.policy_params.get('reproduction_rate', 0.1)
            if random.random() < reproduction_prob:
                return Action(
                    action_type=ActionType.REPRODUCE,
                    energy_cost=20.0,
                    metadata={'offspring_count': 1}
                )
        
        # Compete for resources if crowded
        if competition_level > 0.5:
            return Action(
                action_type=ActionType.COMPETE,
                energy_cost=0.5,
                metadata={'competition_intensity': competition_level}
            )
        
        # Default: grow
        light = observation.get('light', 0.5)
        water = observation.get('water', 0.5)
        nutrients = observation.get('nutrients', 0.5)
        
        growth_potential = (light + water + nutrients) / 3.0
        
        return Action(
            action_type=ActionType.MOVE,  # Growth is represented as "movement"
            energy_cost=0.3,
            metadata={'growth_potential': growth_potential}
        )
    
    def _execute_specific_action(self, action: Action, environment: Any) -> bool:
        """Execute plant-specific actions."""
        if action.action_type == ActionType.REPRODUCE:
            return self._execute_reproduce(action, environment)
        
        elif action.action_type == ActionType.COMPETE:
            return self._execute_compete(action, environment)
        
        elif action.action_type == ActionType.DORMANT:
            return self._execute_dormant(action)
        
        elif action.action_type == ActionType.MOVE:  # Growth
            return self._execute_grow(action, environment)
        
        return False
    
    def _execute_grow(self, action: Action, environment: Any) -> bool:
        """Execute growth action."""
        growth_potential = action.metadata.get('growth_potential', 0.5)
        
        # Calculate energy gain from photosynthesis
        base_growth = self.policy_params['growth_rate']
        energy_gain = base_growth * growth_potential * 2.0
        
        # Age factor reduces growth efficiency
        age_factor = max(0.3, 1.0 - (self.state.age / self._get_max_age()))
        energy_gain *= age_factor
        
        self.state.energy += energy_gain
        self.metrics['growth_cycles'] += 1
        
        # Check for maturity
        if self.state.age >= self.policy_params['maturity_age']:
            self.state.is_mature = True
        
        return True
    
    def _execute_reproduce(self, action: Action, environment: Any) -> bool:
        """Execute reproduction action."""
        energy_cost = action.energy_cost
        
        if self.state.energy < energy_cost or not self.state.is_mature:
            return False
        
        self.state.energy -= energy_cost
        self.metrics['reproductions'] += 1
        
        # Note: Actual offspring creation would be handled by the environment
        return True
    
    def _execute_compete(self, action: Action, environment: Any) -> bool:
        """Execute resource competition action."""
        competition_intensity = action.metadata.get('competition_intensity', 0.5)
        
        # Competitive resource uptake
        resource_gain = self.policy_params['resource_competition'] * competition_intensity * 1.5
        energy_cost = competition_intensity * 0.8  # Competition is costly
        
        net_energy = resource_gain - energy_cost
        self.state.energy += max(0, net_energy)
        self.metrics['resource_competition_events'] += 1
        
        return True
    
    def _execute_dormant(self, action: Action) -> bool:
        """Execute dormancy action."""
        conservation_factor = action.metadata.get('conservation_factor', 0.9)
        
        # Minimal energy loss during dormancy
        energy_loss = 0.1 * (1 - conservation_factor)
        self.state.energy -= energy_loss
        
        return True
    
    def _detect_nearby_agents(self, agent_list: List, radius: float) -> List:
        """Detect agents within specified radius."""
        nearby = []
        for agent in agent_list:
            if agent.state.is_alive and agent != self:
                distance = self._calculate_distance(
                    self.state.pos_x, self.state.pos_y,
                    agent.state.pos_x, agent.state.pos_y
                )
                if distance <= radius:
                    nearby.append({
                        'agent': agent,
                        'distance': distance,
                        'pos': (agent.state.pos_x, agent.state.pos_y),
                        'energy': agent.state.energy
                    })
        return nearby
    
    def _calculate_light_availability(self, local_env: Dict, nearby_plants: List) -> float:
        """Calculate available light based on competition."""
        base_light = 0.8  # Assume good light availability
        
        # Reduce light based on nearby plant competition
        competition_factor = 1.0 - (len(nearby_plants) * 0.05)
        
        # Environmental factors
        vegetation_density = local_env.get('vegetation', 0.5)
        light_reduction = vegetation_density * 0.2
        
        return max(0.1, base_light * competition_factor - light_reduction)
    
    def _calculate_stress_level(self, local_env: Dict[str, Any]) -> float:
        """Calculate environmental stress level."""
        temperature = local_env.get('temperature', 20)
        humidity = local_env.get('humidity', 60)
        
        # Temperature stress (optimal around 22°C)
        temp_stress = abs(temperature - 22) / 25.0
        
        # Water stress (stress when humidity < 40%)
        water_stress = max(0, (40 - humidity) / 40.0)
        
        # Combined stress
        return min(1.0, temp_stress + water_stress)
    
    def _calculate_reward(self, action: Action, environment: Any) -> float:
        """Calculate reward for plant actions."""
        reward = 0.0
        
        # Base survival reward
        reward += 0.5
        
        # Action-specific rewards
        if action.action_type == ActionType.MOVE:  # Growth
            growth_potential = action.metadata.get('growth_potential', 0.5)
            reward += growth_potential * 5.0
        
        elif action.action_type == ActionType.REPRODUCE:
            reward += 15.0  # High reward for successful reproduction
        
        elif action.action_type == ActionType.COMPETE:
            reward += 3.0  # Moderate reward for competing
        
        elif action.action_type == ActionType.DORMANT:
            reward += 1.0  # Small reward for surviving stress
        
        # Energy level reward
        if self.state.energy > 60:
            reward += 3.0
        elif self.state.energy < 10:
            reward -= 5.0
        
        return reward
    
    def _update_policy_params(self, total_reward: float, trajectory: List[Dict[str, Any]]) -> None:
        """Update plant policy parameters."""
        # Adjust growth rate based on success
        if total_reward > 0:
            self.policy_params['growth_rate'] = min(1.0, 
                self.policy_params['growth_rate'] + 0.02)
        else:
            # Poor performance - increase stress tolerance and competition
            self.policy_params['stress_tolerance'] = min(1.0, 
                self.policy_params['stress_tolerance'] + 0.05)
            self.policy_params['resource_competition'] = min(1.0, 
                self.policy_params['resource_competition'] + 0.05)
    
    def _get_movement_efficiency(self) -> float:
        """Plants don't move, but this affects growth efficiency."""
        return 1.0
    
    def _get_max_age(self) -> int:
        """Maximum age for plants."""
        return self.config.get('max_age', 200)


# Factory function for creating agents
def create_agent(agent_type: str, agent_id: str, **kwargs) -> BaseAgent:
    """
    Factory function to create agents of different types.
    
    Args:
        agent_type: Type of agent ('predator', 'prey', 'plant')
        agent_id: Unique identifier for the agent
        **kwargs: Additional parameters for agent initialization
        
    Returns:
        BaseAgent: Instance of the appropriate agent subclass
    """
    agent_classes = {
        'predator': PredatorAgent,
        'prey': PreyAgent,
        'plant': PlantAgent
    }
    
    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent_class = agent_classes[agent_type]
    return agent_class(agent_id=agent_id, agent_type=agent_type, **kwargs)


# Utility functions for agent management
def get_agent_stats(agents: List[BaseAgent]) -> Dict[str, Any]:
    """Get statistics for a list of agents."""
    alive_agents = [a for a in agents if a.state.is_alive]
    
    if not alive_agents:
        return {'count': 0, 'avg_energy': 0, 'avg_age': 0}
    
    return {
        'count': len(alive_agents),
        'total_count': len(agents),
        'avg_energy': np.mean([a.state.energy for a in alive_agents]),
        'avg_age': np.mean([a.state.age for a in alive_agents]),
        'energy_distribution': [a.state.energy for a in alive_agents],
        'age_distribution': [a.state.age for a in alive_agents]
    }


def update_all_agents(agents: List[BaseAgent], environment: Any) -> None:
    """Update all agents for one time step."""
    for agent in agents:
        if agent.state.is_alive:
            try:
                # Get observation
                observation = agent.get_observation(environment)
                
                # Select and execute action
                action = agent.select_action(observation)
                agent.execute_action(action, environment)
                
            except Exception as e:
                logger.error(f"Error updating agent {agent.agent_id}: {e}")
