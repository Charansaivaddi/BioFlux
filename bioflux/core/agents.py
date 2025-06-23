"""
BioFlux Agent Classes - Refactored

This module contains the core agent classes for the BioFlux ecosystem simulation.
All agents are RL-enabled and can use different policies and learning algorithms.
"""

import random
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from abc import ABC, abstractmethod


class RLAgent(ABC):
    """Base class for all reinforcement learning agents in the ecosystem."""
    
    def __init__(self, agent_id: str, init_params: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.params = init_params or {}
        self.experience_buffer = []
        
    @abstractmethod
    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Select an action based on current observation."""
        pass
    
    @abstractmethod
    def update(self, trajectory: List[Dict[str, Any]]) -> None:
        """Update agent parameters based on experience trajectory."""
        pass
    
    @abstractmethod
    def act(self, action: Dict[str, Any]) -> None:
        """Execute the selected action."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current agent parameters."""
        return self.params.copy()
    
    def set_parameters(self, new_params: Dict[str, Any]) -> None:
        """Set new agent parameters."""
        self.params.update(new_params)
    
    def add_experience(self, state: Dict[str, Any], action: Dict[str, Any], 
                      reward: float, next_state: Dict[str, Any]) -> None:
        """Add experience to the agent's buffer."""
        self.experience_buffer.append({
            'state': state,
            'action': action, 
            'reward': reward,
            'next_state': next_state
        })


class Predator(RLAgent):
    """Predator agent with hunting and survival behaviors."""
    
    def __init__(self, speed: int, energy: float, pos_x: int, pos_y: int, 
                 age: int, agent_id: Optional[str] = None, 
                 init_params: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id or f"predator_{random.randint(1000, 9999)}", init_params)
        
        # Agent type
        self.agent_type = 'predator'
        
        # Physical attributes
        self.speed = speed
        self.energy = energy
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.age = age
        
        # State flags
        self.is_alive = True
        self.is_hungry = True
        self.is_moving = False
        self.is_eating = False
        
        # Performance tracking
        self.kills = 0
        self.distance_traveled = 0
        self.energy_consumed = 0
    
    def __str__(self) -> str:
        return (f"Predator({self.agent_id}: speed={self.speed}, energy={self.energy:.1f}, "
                f"pos=({self.pos_x},{self.pos_y}), age={self.age}, alive={self.is_alive})")
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def move(self, new_x: int, new_y: int, terrain_difficulty: float = 1.0) -> None:
        """Move to new position with terrain-adjusted energy cost."""
        if not self.is_alive:
            return
            
        # Calculate distance and energy cost
        distance = abs(new_x - self.pos_x) + abs(new_y - self.pos_y)
        energy_cost = distance * terrain_difficulty
        
        self.pos_x = new_x
        self.pos_y = new_y
        self.energy -= energy_cost
        self.distance_traveled += distance
        self.energy_consumed += energy_cost
        
        self.is_moving = True
        self.is_eating = False
        
        if self.energy <= 0:
            self.die()
    
    def eat(self, prey_energy: float) -> bool:
        """Attempt to eat prey and gain energy."""
        if not self.is_alive or not self.is_hungry:
            return False
            
        if self.energy < 100:  # Don't overeat
            energy_gained = min(prey_energy, 100 - self.energy)
            self.energy += energy_gained
            self.kills += 1
            self.is_eating = True
            self.is_hungry = self.energy < 50  # Hungry threshold
            return True
        return False
    
    def die(self) -> None:
        """Handle predator death."""
        self.is_alive = False
        self.is_moving = False
        self.is_eating = False
        self.energy = 0
    
    def age_step(self) -> None:
        """Age the predator and check for natural death."""
        if not self.is_alive:
            return
            
        self.age += 1
        
        # Natural death conditions
        if self.age > 20 or self.energy <= 0:
            self.die()
        
        # Hunger increases with age
        if self.age > 10:
            self.energy -= 0.5  # Older predators lose energy faster
    
    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Select action using current policy."""
        policy_fn = self.params.get('policy_fn')
        if policy_fn:
            return policy_fn(observation, self)
        return self._default_policy(observation)
    
    def _default_policy(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced hunting policy using environmental data."""
        if not self.is_alive:
            return {'action': 'stay', 'x': self.pos_x, 'y': self.pos_y}
        
        # Get environmental information
        prey_positions = observation.get('prey', [])
        geospatial = observation.get('geospatial', {})
        environment_bounds = observation.get('bounds', {'width': 50, 'height': 50})
        
        if prey_positions:
            # Hunt nearest prey
            closest_prey = min(prey_positions, 
                             key=lambda p: abs(p[0] - self.pos_x) + abs(p[1] - self.pos_y))
            target_x, target_y = closest_prey
            
            # Calculate movement direction
            dx = 1 if target_x > self.pos_x else -1 if target_x < self.pos_x else 0
            dy = 1 if target_y > self.pos_y else -1 if target_y < self.pos_y else 0
            
            new_x = max(0, min(environment_bounds['width'] - 1, self.pos_x + dx))
            new_y = max(0, min(environment_bounds['height'] - 1, self.pos_y + dy))
            
            # Consider terrain difficulty
            terrain_difficulty = 1.0
            if geospatial:
                vegetation_density = geospatial.get('vegetation_density', 0)
                elevation = geospatial.get('elevation', 500)
                
                # Avoid very dense vegetation or steep terrain
                if vegetation_density > 80 or elevation > 1000:
                    terrain_difficulty = 2.0
                    # Try alternative direction
                    new_x = max(0, min(environment_bounds['width'] - 1, 
                                     self.pos_x + random.choice([-1, 0, 1])))
                    new_y = max(0, min(environment_bounds['height'] - 1,
                                     self.pos_y + random.choice([-1, 0, 1])))
            
            return {
                'action': 'move',
                'x': new_x,
                'y': new_y,
                'terrain_difficulty': terrain_difficulty
            }
        else:
            # Explore when no prey visible
            return self._explore_action(environment_bounds)
    
    def _explore_action(self, bounds: Dict[str, int]) -> Dict[str, Any]:
        """Generate exploration action when no prey is visible."""
        # Random walk with boundary checking
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        
        new_x = max(0, min(bounds['width'] - 1, self.pos_x + dx))
        new_y = max(0, min(bounds['height'] - 1, self.pos_y + dy))
        
        return {
            'action': 'move',
            'x': new_x,
            'y': new_y,
            'terrain_difficulty': 1.0
        }
    
    def act(self, action: Dict[str, Any]) -> None:
        """Execute the selected action."""
        action_type = action.get('action', 'stay')
        
        if action_type == 'move':
            new_x = action.get('x', self.pos_x)
            new_y = action.get('y', self.pos_y)
            terrain_difficulty = action.get('terrain_difficulty', 1.0)
            self.move(new_x, new_y, terrain_difficulty)
        elif action_type == 'hunt':
            # Hunting action would be handled by the environment
            pass
    
    def update(self, trajectory: List[Dict[str, Any]]) -> None:
        """Update predator parameters based on experience."""
        if not trajectory:
            return
        
        # Simple reward-based parameter update
        total_reward = sum(exp.get('reward', 0) for exp in trajectory)
        
        # Adjust hunting aggressiveness based on success
        if total_reward > 0:
            # Successful hunt - become more aggressive
            current_aggression = self.params.get('hunting_aggression', 1.0)
            self.params['hunting_aggression'] = min(2.0, current_aggression + 0.1)
        else:
            # Failed hunt - become more cautious
            current_aggression = self.params.get('hunting_aggression', 1.0)
            self.params['hunting_aggression'] = max(0.5, current_aggression - 0.05)
    
    def get_observation(self, environment) -> Dict[str, Any]:
        """Get environmental observation for decision making."""
        # Find nearby prey
        nearby_prey = []
        for prey in environment.prey:
            if prey.is_alive:
                distance = abs(prey.pos_x - self.pos_x) + abs(prey.pos_y - self.pos_y)
                if distance <= 5:  # Detection radius
                    nearby_prey.append({
                        'distance': distance,
                        'energy': prey.energy,
                        'pos': (prey.pos_x, prey.pos_y)
                    })
        
        # Get local environment conditions
        local_env = environment.get_local_environment(int(self.pos_x), int(self.pos_y))
        
        # Find nearby predators (competition)
        nearby_predators = []
        for predator in environment.predators:
            if predator != self and predator.is_alive:
                distance = abs(predator.pos_x - self.pos_x) + abs(predator.pos_y - self.pos_y)
                if distance <= 3:
                    nearby_predators.append(distance)
        
        return {
            'energy': self.energy,
            'age': self.age,
            'is_hungry': self.is_hungry,
            'nearby_prey': nearby_prey,
            'nearby_predators': len(nearby_predators),
            'temperature': local_env.get('temperature', 20),
            'vegetation': local_env.get('vegetation', 0.5),
            'terrain_difficulty': 1.0,  # Could be calculated from elevation
            'environment_bounds': {
                'width': environment.width,
                'height': environment.height
            }
        }


class Prey(RLAgent):
    """Prey agent with foraging and escape behaviors."""
    
    def __init__(self, speed: int, energy: float, pos_x: int, pos_y: int,
                 age: int, agent_id: Optional[str] = None,
                 init_params: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id or f"prey_{random.randint(1000, 9999)}", init_params)
        
        # Agent type
        self.agent_type = 'prey'
        
        # Physical attributes
        self.speed = speed
        self.energy = energy
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.age = age
        
        # State flags
        self.is_alive = True
        self.is_hungry = True
        self.is_moving = False
        self.is_eating = False
        
        # Performance tracking
        self.food_consumed = 0
        self.distance_traveled = 0
        self.escapes = 0
    
    def __str__(self) -> str:
        return (f"Prey({self.agent_id}: speed={self.speed}, energy={self.energy:.1f}, "
                f"pos=({self.pos_x},{self.pos_y}), age={self.age}, alive={self.is_alive})")
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def move(self, new_x: int, new_y: int, terrain_difficulty: float = 1.0) -> None:
        """Move to new position with terrain-adjusted energy cost."""
        if not self.is_alive:
            return
        
        # Calculate distance and energy cost
        distance = abs(new_x - self.pos_x) + abs(new_y - self.pos_y)
        energy_cost = distance * terrain_difficulty * 0.8  # Prey are more efficient
        
        self.pos_x = new_x
        self.pos_y = new_y
        self.energy -= energy_cost
        self.distance_traveled += distance
        
        self.is_moving = True
        self.is_eating = False
        
        if self.energy <= 0:
            self.die()
    
    def eat(self, food_amount: float) -> bool:
        """Consume food and gain energy."""
        if not self.is_alive:
            return False
        
        if self.energy < 50:  # Don't overeat
            energy_gained = min(food_amount, 50 - self.energy)
            self.energy += energy_gained
            self.food_consumed += energy_gained
            self.is_eating = True
            self.is_hungry = self.energy < 30  # Hunger threshold
            return True
        return False
    
    def die(self) -> None:
        """Handle prey death."""
        self.is_alive = False
        self.is_moving = False
        self.is_eating = False
        self.energy = 0
    
    def age_step(self) -> None:
        """Age the prey and check for natural death."""
        if not self.is_alive:
            return
        
        self.age += 1
        
        # Natural death conditions
        if self.age > 15 or self.energy <= 0:
            self.die()
        
        # Gradual energy loss
        self.energy -= 0.3
    
    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Select action using current policy."""
        policy_fn = self.params.get('policy_fn')
        if policy_fn:
            return policy_fn(observation, self)
        return self._default_policy(observation)
    
    def get_observation(self, environment) -> Dict[str, Any]:
        """Get environmental observation for decision making."""
        # Find nearby predators (danger)
        nearby_predators = []
        for predator in environment.predators:
            if predator.is_alive:
                distance = abs(predator.pos_x - self.pos_x) + abs(predator.pos_y - self.pos_y)
                if distance <= 6:  # Detection radius
                    nearby_predators.append({
                        'distance': distance,
                        'energy': predator.energy,
                        'pos': (predator.pos_x, predator.pos_y)
                    })
        
        # Find nearby food (plants)
        nearby_food = []
        for plant in environment.plants:
            if plant.is_alive:
                distance = abs(plant.pos_x - self.pos_x) + abs(plant.pos_y - self.pos_y)
                if distance <= 4:  # Foraging radius
                    nearby_food.append({
                        'distance': distance,
                        'energy': plant.energy,
                        'pos': (plant.pos_x, plant.pos_y)
                    })
        
        # Get local environment conditions
        local_env = environment.get_local_environment(int(self.pos_x), int(self.pos_y))
        
        # Find other prey (herd behavior)
        nearby_prey = []
        for prey in environment.prey:
            if prey != self and prey.is_alive:
                distance = abs(prey.pos_x - self.pos_x) + abs(prey.pos_y - self.pos_y)
                if distance <= 3:
                    nearby_prey.append(distance)
        
        return {
            'energy': self.energy,
            'age': self.age,
            'is_hungry': self.is_hungry,
            'nearby_predators': nearby_predators,
            'nearby_food': nearby_food,
            'nearby_prey': len(nearby_prey),
            'temperature': local_env.get('temperature', 20),
            'vegetation': local_env.get('vegetation', 0.5),
            'terrain_difficulty': 1.0,
            'environment_bounds': {
                'width': environment.width,
                'height': environment.height
            }
        }
    
    def _default_policy(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced foraging and escape policy."""
        if not self.is_alive:
            return {'action': 'stay', 'x': self.pos_x, 'y': self.pos_y}
        
        # Get environmental information
        predator_positions = observation.get('predators', [])
        food_positions = observation.get('food', [])
        geospatial = observation.get('geospatial', {})
        environment_bounds = observation.get('bounds', {'width': 50, 'height': 50})
        
        # Escape from predators (highest priority)
        if predator_positions:
            nearest_predator = min(predator_positions,
                                 key=lambda p: abs(p[0] - self.pos_x) + abs(p[1] - self.pos_y))
            pred_x, pred_y = nearest_predator
            
            # Check if predator is close
            distance_to_predator = abs(pred_x - self.pos_x) + abs(pred_y - self.pos_y)
            if distance_to_predator <= 3:  # Danger zone
                # Escape in opposite direction
                escape_x = self.pos_x - (pred_x - self.pos_x)
                escape_y = self.pos_y - (pred_y - self.pos_y)
                
                # Bound check
                escape_x = max(0, min(environment_bounds['width'] - 1, escape_x))
                escape_y = max(0, min(environment_bounds['height'] - 1, escape_y))
                
                self.escapes += 1
                return {
                    'action': 'flee',
                    'x': escape_x,
                    'y': escape_y,
                    'terrain_difficulty': 0.8  # Adrenaline makes movement easier
                }
        
        # Forage for food
        if food_positions and self.is_hungry:
            nearest_food = min(food_positions,
                             key=lambda f: abs(f[0] - self.pos_x) + abs(f[1] - self.pos_y))
            food_x, food_y = nearest_food
            
            # Move toward food
            dx = 1 if food_x > self.pos_x else -1 if food_x < self.pos_x else 0
            dy = 1 if food_y > self.pos_y else -1 if food_y < self.pos_y else 0
            
            new_x = max(0, min(environment_bounds['width'] - 1, self.pos_x + dx))
            new_y = max(0, min(environment_bounds['height'] - 1, self.pos_y + dy))
            
            # Prefer areas with high NDVI (more food)
            terrain_difficulty = 1.0
            if geospatial:
                ndvi = geospatial.get('ndvi', 0.5)
                if ndvi > 0.7:
                    terrain_difficulty = 0.9  # Easy movement in good vegetation
                elif ndvi < 0.3:
                    terrain_difficulty = 1.2  # Harder to move in sparse areas
            
            return {
                'action': 'forage',
                'x': new_x,
                'y': new_y,
                'terrain_difficulty': terrain_difficulty
            }
        
        # Explore when no immediate needs
        return self._explore_action(environment_bounds)
    
    def _explore_action(self, bounds: Dict[str, int]) -> Dict[str, Any]:
        """Generate exploration action when no immediate needs."""
        # Biased random walk (prefer areas with potential food)
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        
        new_x = max(0, min(bounds['width'] - 1, self.pos_x + dx))
        new_y = max(0, min(bounds['height'] - 1, self.pos_y + dy))
        
        return {
            'action': 'explore',
            'x': new_x,
            'y': new_y,
            'terrain_difficulty': 1.0
        }
    
    def act(self, action: Dict[str, Any]) -> None:
        """Execute the selected action."""
        action_type = action.get('action', 'stay')
        
        if action_type in ['move', 'flee', 'forage', 'explore']:
            new_x = action.get('x', self.pos_x)
            new_y = action.get('y', self.pos_y)
            terrain_difficulty = action.get('terrain_difficulty', 1.0)
            self.move(new_x, new_y, terrain_difficulty)
    
    def update(self, trajectory: List[Dict[str, Any]]) -> None:
        """Update prey parameters based on experience."""
        if not trajectory:
            return
        
        # Simple reward-based parameter update
        total_reward = sum(exp.get('reward', 0) for exp in trajectory)
        
        # Adjust escape behavior based on survival
        if total_reward > 0:
            # Successful survival - maintain current strategy
            current_caution = self.params.get('escape_distance', 3)
            self.params['escape_distance'] = current_caution
        else:
            # Danger encountered - become more cautious
            current_caution = self.params.get('escape_distance', 3)
            self.params['escape_distance'] = min(5, current_caution + 0.5)


class Plant(RLAgent):
    """
    Plant agent that grows, reproduces, and responds to environmental conditions.
    
    Plants have adaptive growth strategies based on light, water, and nutrient availability.
    They can compete for resources and have basic reproduction mechanisms.
    """
    
    def __init__(self, energy: float = 20.0, pos_x: float = 0.0, pos_y: float = 0.0,
                 age: int = 0, max_age: int = 200, init_params: Optional[Dict[str, Any]] = None):
        """
        Initialize a plant agent.
        
        Args:
            energy: Initial energy/biomass
            pos_x: X position 
            pos_y: Y position
            age: Initial age
            max_age: Maximum lifespan
            init_params: Additional initialization parameters
        """
        super().__init__(f"plant_{random.randint(1000, 9999)}", init_params)
        
        # Agent type  
        self.agent_type = 'plant'
        
        # Physical attributes
        self.energy = energy
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.age = age
        self.max_age = max_age
        
        # State variables
        self.is_alive = True
        self.is_mature = age > 10  # Can reproduce after age 10
        self.growth_rate = 0.5
        self.reproduction_threshold = 50.0
        self.water_need = 0.3
        self.light_need = 0.4
        self.nutrient_need = 0.2
        
        # RL parameters - plants learn optimal growth strategies
        self.params.update({
            'growth_efficiency': 1.0,
            'reproduction_rate': 0.1,
            'resource_competition': 0.8,
            'environmental_adaptation': 0.5,
            'stress_tolerance': 0.6
        })
    
    def __str__(self):
        return (f"Plant(energy={self.energy:.1f}, pos=({self.pos_x:.1f},{self.pos_y:.1f}), "
                f"age={self.age}, alive={self.is_alive}, mature={self.is_mature})")
    
    def __repr__(self):
        return self.__str__()
    
    def get_observation(self, environment) -> Dict[str, Any]:
        """Get environmental observation for decision making."""
        local_env = environment.get_local_environment(int(self.pos_x), int(self.pos_y))
        
        # Calculate resource availability
        light_availability = self._calculate_light_availability(environment)
        water_availability = local_env.get('humidity', 50) / 100.0
        nutrient_availability = local_env.get('vegetation', 0.5)
        
        # Detect nearby competition
        nearby_plants = self._detect_nearby_plants(environment, radius=3)
        competition_level = len(nearby_plants) / 10.0  # Normalize
        
        return {
            'energy': self.energy,
            'age': self.age,
            'light': light_availability,
            'water': water_availability,
            'nutrients': nutrient_availability,
            'competition': competition_level,
            'temperature': local_env.get('temperature', 20),
            'can_reproduce': self.energy > self.reproduction_threshold and self.is_mature,
            'stress_level': self._calculate_stress_level(local_env)
        }
    
    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Select plant action based on environmental conditions."""
        # Plants have several strategies: grow, reproduce, compete, or go dormant
        
        stress_level = observation.get('stress_level', 0)
        energy = observation.get('energy', 0)
        can_reproduce = observation.get('can_reproduce', False)
        
        # High stress - go dormant to conserve energy
        if stress_level > 0.8:
            return {
                'action': 'dormant',
                'energy_conservation': 0.9,
                'growth_rate': 0.1
            }
        
        # Can reproduce and enough energy
        if can_reproduce and energy > self.reproduction_threshold:
            reproduction_prob = self.params.get('reproduction_rate', 0.1)
            if random.random() < reproduction_prob:
                return {
                    'action': 'reproduce',
                    'energy_cost': 20.0,
                    'offspring_count': 1
                }
        
        # Optimal growing conditions
        light = observation.get('light', 0.5)
        water = observation.get('water', 0.5)
        nutrients = observation.get('nutrients', 0.5)
        
        if light > 0.6 and water > 0.4 and nutrients > 0.3:
            growth_efficiency = self.params.get('growth_efficiency', 1.0)
            return {
                'action': 'grow',
                'growth_rate': self.growth_rate * growth_efficiency,
                'resource_uptake': 0.8
            }
        
        # Competition for resources
        competition = observation.get('competition', 0)
        if competition > 0.5:
            competition_response = self.params.get('resource_competition', 0.8)
            return {
                'action': 'compete',
                'resource_uptake': 0.9 * competition_response,
                'growth_rate': self.growth_rate * 0.7  # Slower growth when competing
            }
        
        # Default: moderate growth
        return {
            'action': 'grow',
            'growth_rate': self.growth_rate * 0.8,
            'resource_uptake': 0.6
        }
    
    def act(self, action: Dict[str, Any]) -> None:
        """Execute the selected action."""
        action_type = action.get('action', 'grow')
        
        if action_type == 'grow':
            growth_rate = action.get('growth_rate', self.growth_rate)
            resource_uptake = action.get('resource_uptake', 0.6)
            self.grow(growth_rate, resource_uptake)
        
        elif action_type == 'reproduce':
            energy_cost = action.get('energy_cost', 20.0)
            self.reproduce(energy_cost)
        
        elif action_type == 'compete':
            resource_uptake = action.get('resource_uptake', 0.9)
            growth_rate = action.get('growth_rate', self.growth_rate * 0.7)
            self.compete_for_resources(resource_uptake, growth_rate)
        
        elif action_type == 'dormant':
            conservation = action.get('energy_conservation', 0.9)
            self.go_dormant(conservation)
    
    def grow(self, growth_rate: float, resource_uptake: float):
        """Grow based on available resources."""
        if not self.is_alive:
            return
        
        # Energy gain from photosynthesis and resource uptake
        energy_gain = growth_rate * resource_uptake * 2.0
        
        # Growth efficiency decreases with age
        age_factor = max(0.3, 1.0 - (self.age / self.max_age))
        energy_gain *= age_factor
        
        self.energy += energy_gain
        
        # Become mature
        if self.age > 10:
            self.is_mature = True
    
    def reproduce(self, energy_cost: float) -> bool:
        """Attempt reproduction."""
        if not self.is_alive or not self.is_mature or self.energy < energy_cost:
            return False
        
        self.energy -= energy_cost
        # Note: Actual offspring creation would be handled by the environment
        return True
    
    def compete_for_resources(self, resource_uptake: float, growth_rate: float):
        """Compete with nearby plants for resources."""
        if not self.is_alive:
            return
        
        # Competitive growth - higher uptake but more energy cost
        energy_gain = growth_rate * resource_uptake * 1.5
        energy_cost = resource_uptake * 0.5  # Competition is costly
        
        net_energy = energy_gain - energy_cost
        self.energy += max(0, net_energy)
    
    def go_dormant(self, conservation_factor: float):
        """Enter dormant state to conserve energy."""
        if not self.is_alive:
            return
        
        # Minimal energy loss during dormancy
        self.energy -= 0.1 * (1 - conservation_factor)
        self.energy = max(0, self.energy)
    
    def age_step(self):
        """Age the plant by one time step."""
        if not self.is_alive:
            return
        
        self.age += 1
        
        # Natural energy decay
        decay_rate = 0.05 + (self.age / self.max_age) * 0.1
        self.energy -= decay_rate
        
        # Die from old age or lack of energy
        if self.age >= self.max_age or self.energy <= 0:
            self.die()
    
    def die(self):
        """Plant dies."""
        self.is_alive = False
        self.energy = 0
    
    def update(self, trajectory: List[Dict[str, Any]]) -> None:
        """Update plant parameters based on experience."""
        if not trajectory:
            return
        
        # Learn from growth success/failure
        total_reward = sum(exp.get('reward', 0) for exp in trajectory)
        
        if total_reward > 0:
            # Successful growth - reinforce current strategy
            current_efficiency = self.params.get('growth_efficiency', 1.0)
            self.params['growth_efficiency'] = min(1.5, current_efficiency + 0.05)
        else:
            # Poor growth - try different strategy
            self.params['resource_competition'] += 0.1
            self.params['stress_tolerance'] += 0.05
    
    def _calculate_light_availability(self, environment) -> float:
        """Calculate available light based on plant density and position."""
        # Mock calculation - in reality this would consider canopy coverage
        base_light = 0.8  # Assume good light availability
        
        # Reduce light based on vegetation density
        local_env = environment.get_local_environment(int(self.pos_x), int(self.pos_y))
        vegetation_density = local_env.get('vegetation', 0.5)
        
        # Higher vegetation density means more competition for light
        light_competition_factor = 1.0 - (vegetation_density * 0.3)
        
        return base_light * light_competition_factor
    
    def _detect_nearby_plants(self, environment, radius: float = 3) -> List['Plant']:
        """Detect nearby competing plants."""
        nearby_plants = []
        
        for plant in environment.plants:
            if plant != self and plant.is_alive:
                distance = np.sqrt((plant.pos_x - self.pos_x)**2 + (plant.pos_y - self.pos_y)**2)
                if distance <= radius:
                    nearby_plants.append(plant)
        
        return nearby_plants
    
    def _calculate_stress_level(self, local_env: Dict[str, Any]) -> float:
        """Calculate environmental stress level."""
        temperature = local_env.get('temperature', 20)
        humidity = local_env.get('humidity', 60)
        
        # Temperature stress
        optimal_temp = 22
        temp_stress = abs(temperature - optimal_temp) / 20.0
        
        # Water stress
        water_stress = max(0, (40 - humidity) / 40.0)  # Stress when humidity < 40%
        
        # Combined stress
        total_stress = min(1.0, temp_stress + water_stress)
        
        return total_stress
