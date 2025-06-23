import random
import numpy as np
try:
    from geospatial import VegetationLayer, LogisticGrowthModel
except ImportError:
    # Fallback if geospatial module not available
    VegetationLayer = None
    LogisticGrowthModel = None

class RLAgent:
    def __init__(self, init_params=None):
        if init_params is None:
            init_params = {}
        self.params = init_params

    def select_action(self, observation):
        """Return an action or parameter update given an observation."""
        raise NotImplementedError

    def update(self, trajectory):
        """Update internal parameters from experience."""
        raise NotImplementedError

    def get_parameters(self):
        return self.params

    def set_parameters(self, new_params):
        self.params = new_params
    
    def act(self, action):
        """Execute a chosen action."""
        raise NotImplementedError

class Predator(RLAgent):
    def __init__(self, speed, energy, pos_x, pos_y, age, init_params=None):
        super().__init__(init_params)
        self.speed = speed
        self.energy = energy
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.age = age
        self.is_alive = True
        self.is_hungry = True
        self.is_moving = False
        self.is_eating = False

    def __str__(self):
        return f"Predator(speed={self.speed}, energy={self.energy}, pos=({self.pos_x},{self.pos_y}), age={self.age}, alive={self.is_alive})"

    def __repr__(self):
        return self.__str__()

    def move(self, new_x, new_y):
        self.pos_x = new_x
        self.pos_y = new_y
        self.is_moving = True
        self.is_eating = False
        self.energy -= 1  # movement costs energy

    def eat(self, prey_energy):
        if self.is_hungry and self.energy < 100:
            self.energy += prey_energy
            self.is_eating = True
            self.is_hungry = False
            print(f"Predator ate and now has {self.energy} energy")

    def die(self):
        self.is_alive = False
        self.is_moving = False
        self.is_eating = False
        print("Predator has died")

    def age_predator(self):
        self.age += 1
        if self.age > 20 or self.energy <= 0:
            self.die()
    
    def select_action(self, observation):
        policy_fn = self.params.get('policy_fn')
        return policy_fn(observation, self) if policy_fn else self.default_policy(observation)
    
    def default_policy(self, observation):
        """Enhanced chase-prey policy using geospatial data."""
        if observation.get('prey'):
            closest_prey = min(observation['prey'], key=lambda p: abs(p[0]) + abs(p[1]))
            dx, dy = closest_prey[0], closest_prey[1]
            
            # Consider terrain difficulty in movement decision
            geospatial = observation.get('geospatial', {})
            if geospatial:
                # Avoid very dense vegetation or steep terrain
                if geospatial.get('vegetation_density', 0) > 80 or geospatial.get('elevation', 500) > 1000:
                    # Move to easier terrain instead
                    new_x = self.pos_x + random.choice([-1, 0, 1])
                    new_y = self.pos_y + random.choice([-1, 0, 1])
                else:
                    new_x = self.pos_x + (1 if dx > 0 else -1 if dx < 0 else 0)
                    new_y = self.pos_y + (1 if dy > 0 else -1 if dy < 0 else 0)
            else:
                new_x = self.pos_x + (1 if dx > 0 else -1 if dx < 0 else 0)
                new_y = self.pos_y + (1 if dy > 0 else -1 if dy < 0 else 0)
                
            return {'type': 'move', 'x': new_x, 'y': new_y}
        else:
            new_x = self.pos_x + random.randint(-1, 1)
            new_y = self.pos_y + random.randint(-1, 1)
            return {'type': 'move', 'x': new_x, 'y': new_y}
    
    def update(self, trajectory):
        updater = self.params.get('updater')
        if updater:
            self.params = updater(self.params, trajectory)
    
    def act(self, action):
        if not action:
            return
        t = action.get('type')
        if t == 'eat':
            self.eat(action.get('energy', 0))
        elif t == 'age':
            self.age_predator()

class Prey(RLAgent):
    def __init__(self, speed, energy, pos_x, pos_y, age, init_params=None):
        super().__init__(init_params)
        self.speed = speed
        self.energy = energy
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.age = age
        self.is_alive = True
        self.is_hungry = True
        self.is_moving = False
        self.is_eating = False

    def __str__(self):
        return f"Prey(speed={self.speed}, energy={self.energy}, pos=({self.pos_x},{self.pos_y}), age={self.age}, alive={self.is_alive})"

    def __repr__(self):
        return self.__str__()

    def move(self, new_x, new_y):
        self.pos_x = new_x
        self.pos_y = new_y
        self.is_moving = True
        self.is_eating = False
        self.energy -= 0.5

    def eat(self, food_energy):
        if self.is_hungry and self.energy < 100:
            self.energy += food_energy
            self.is_eating = True
            self.is_hungry = False
            print(f"Prey ate and now has {self.energy} energy")

    def die(self):
        self.is_alive = False
        self.is_moving = False
        self.is_eating = False
        print("Prey has died")

    def age_prey(self):
        self.age += 1
        if self.age > 15 or self.energy <= 0:
            self.die()
    
    def select_action(self, observation):
        policy_fn = self.params.get('policy_fn')
        return policy_fn(observation, self) if policy_fn else self.default_policy(observation)
    
    def default_policy(self, observation):
        """Enhanced avoid-predator, seek-food policy using geospatial data."""
        geospatial = observation.get('geospatial', {})
        
        # Avoid predators first (with terrain advantage)
        if observation.get('predators'):
            closest_predator = min(observation['predators'], key=lambda p: abs(p[0]) + abs(p[1]))
            dx, dy = closest_predator[0], closest_predator[1]
            
            # Use dense vegetation for cover
            if geospatial.get('vegetation_density', 0) > 60:
                # Stay in dense vegetation for protection
                new_x = self.pos_x + random.choice([-1, 0, 1])
                new_y = self.pos_y + random.choice([-1, 0, 1])
            else:
                # Move away from predator toward denser vegetation if possible
                new_x = self.pos_x - (1 if dx > 0 else -1 if dx < 0 else 0)
                new_y = self.pos_y - (1 if dy > 0 else -1 if dy < 0 else 0)
            
            return {'type': 'move', 'x': new_x, 'y': new_y}
            
        # Seek food (prefer areas with good vegetation)
        elif observation.get('food'):
            closest_food = min(observation['food'], key=lambda f: abs(f[0]) + abs(f[1]))
            dx, dy = closest_food[0], closest_food[1]
            new_x = self.pos_x + (1 if dx > 0 else -1 if dx < 0 else 0)
            new_y = self.pos_y + (1 if dy > 0 else -1 if dy < 0 else 0)
            return {'type': 'move', 'x': new_x, 'y': new_y}
            
        # Random movement, but prefer areas with higher NDVI
        else:
            if geospatial.get('ndvi', 0.5) < 0.3:
                # Try to move toward better vegetation
                new_x = self.pos_x + random.choice([-1, 1])  # Avoid staying in place
                new_y = self.pos_y + random.choice([-1, 1])
            else:
                new_x = self.pos_x + random.randint(-1, 1)
                new_y = self.pos_y + random.randint(-1, 1)
            return {'type': 'move', 'x': new_x, 'y': new_y}
    
    def update(self, trajectory):
        updater = self.params.get('updater')
        if updater:
            self.params = updater(self.params, trajectory)
    
    def act(self, action):
        if not action:
            return
        t = action.get('type')
        if t == 'eat':
            self.eat(action.get('energy', 0))
        elif t == 'age':
            self.age_prey()

class Environment(RLAgent):
    def __init__(self, width=50, height=50, predators=None, prey=None, init_params=None, bbox=None):
        super().__init__(init_params)
        self.width = width
        self.height = height
        self.grid = [[[] for _ in range(width)] for _ in range(height)]
        self.predators = predators or []
        self.prey = prey or []
        
        # Climate-driven environmental variables (IBM Data Simulation compatible)
        self.temperature = 20.0
        self.precipitation = 0.5
        self.food_growth_rate = 1.0
        self.terrain_accessibility = 1.0
        self.season = 0
        self.timestep = 0
        
        # Geospatial integration
        self.bbox = bbox or (-122.5, 37.7, -122.3, 37.9)  # Default SF Bay Area
        self.vegetation_layer = None
        self.elevation_map = None
        self.ndvi_map = None
        
        # Initialize geospatial layers if available
        if VegetationLayer is not None:
            self.vegetation_layer = VegetationLayer(width, height, self.bbox)
            self.elevation_map = self.vegetation_layer.elevation
            self.ndvi_map = self.vegetation_layer.ndvi
        
        # Resource management
        self.food_patches = {}
        self.weather_zones = {}
        self.initialize_environment()
        
        # Place agents on grid
        for agent in self.predators + self.prey:
            self._grid_add(agent, agent.pos_x, agent.pos_y)

    def initialize_environment(self):
        """Initialize food patches and weather zones using geospatial data."""
        if self.vegetation_layer is not None:
            # Use real vegetation data for food initialization
            for y in range(self.height):
                for x in range(self.width):
                    food_amount = self.vegetation_layer.get_food_availability(x, y)
                    if food_amount > 1:  # Only create patches with significant food
                        self.food_patches[(x, y)] = food_amount
        else:
            # Fallback to random initialization
            for _ in range(self.width * self.height // 10):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                self.food_patches[(x, y)] = random.uniform(10, 50)

    def add_predator(self, predator):
        self.predators.append(predator)
        self._grid_add(predator, predator.pos_x, predator.pos_y)

    def add_prey(self, prey):
        self.prey.append(prey)
        self._grid_add(prey, prey.pos_x, prey.pos_y)

    def _grid_add(self, agent, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x].append(agent)

    def _grid_remove(self, agent, x, y):
        if 0 <= x < self.width and 0 <= y < self.height and agent in self.grid[y][x]:
            self.grid[y][x].remove(agent)

    def _grid_move(self, agent, new_x, new_y):
        self._grid_remove(agent, agent.pos_x, agent.pos_y)
        agent.pos_x = max(0, min(self.width - 1, new_x))
        agent.pos_y = max(0, min(self.height - 1, new_y))
        self._grid_add(agent, agent.pos_x, agent.pos_y)

    def get_observation(self, agent, view_radius=2):
        obs = {
            'predators': [],
            'prey': [],
            'food': [],
            'climate': {
                'temperature': self.temperature,
                'precipitation': self.precipitation,
                'terrain': self.terrain_accessibility,
                'season': self.season
            },
            'geospatial': {}
        }
        
        # Add geospatial information if available
        if self.elevation_map is not None and self.ndvi_map is not None:
            obs['geospatial'] = {
                'elevation': self.elevation_map[agent.pos_y, agent.pos_x],
                'ndvi': self.ndvi_map[agent.pos_y, agent.pos_x],
                'vegetation_density': self.vegetation_layer.biomass[agent.pos_y, agent.pos_x] if self.vegetation_layer else 0
            }
        
        for dy in range(-view_radius, view_radius + 1):
            for dx in range(-view_radius, view_radius + 1):
                x, y = agent.pos_x + dx, agent.pos_y + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    for other in self.grid[y][x]:
                        if other != agent:
                            if isinstance(other, Predator):
                                obs['predators'].append((dx, dy))
                            elif isinstance(other, Prey):
                                obs['prey'].append((dx, dy))
                    
                    if (x, y) in self.food_patches:
                        obs['food'].append((dx, dy, self.food_patches[(x, y)]))
        
        return obs

    def step(self):
        """Execute one simulation timestep."""
        self.timestep += 1
        self.update_climate()
        
        # Update vegetation using logistic growth model
        if self.vegetation_layer is not None:
            self.vegetation_layer.update(self.temperature, self.precipitation)
            self.update_food_from_vegetation()
        
        env_action = self.select_action(None)
        self.act(env_action)
        self.update_food_growth()
        
        agent_actions = []
        for agent in self.predators + self.prey:
            if agent.is_alive:
                obs = self.get_observation(agent)
                action = agent.select_action(obs)
                agent_actions.append((agent, action))
        
        for agent, action in agent_actions:
            if action and action.get('type') == 'move':
                new_x = action.get('x', agent.pos_x)
                new_y = action.get('y', agent.pos_y)
                self._grid_move(agent, new_x, new_y)
            agent.act(action or {})
        
        self.handle_interactions()
        self.handle_food_consumption()
        self.age_agents()
        self.cleanup_dead_agents()

    def update_climate(self):
        """Update climate variables (IBM Data Simulation integration point)."""
        self.season = (self.timestep // 100) % 4
        seasonal_temp = [15, 25, 15, 5][self.season]
        self.temperature = seasonal_temp + random.uniform(-5, 5)
        seasonal_precip = [0.7, 0.3, 0.6, 0.8][self.season]
        self.precipitation = max(0, min(1, seasonal_precip + random.uniform(-0.2, 0.2)))
        self.food_growth_rate = 0.5 + self.precipitation * (1 - abs(self.temperature - 20) / 40)

    def handle_interactions(self):
        """Handle predator-prey encounters with terrain effects."""
        for predator in list(self.predators):
            if not predator.is_alive:
                continue
            cell_agents = self.grid[predator.pos_y][predator.pos_x]
            for other in list(cell_agents):
                if isinstance(other, Prey) and other.is_alive and predator.is_hungry:
                    # Calculate success rate based on terrain and environment
                    base_success_rate = 0.7
                    terrain_difficulty = self.get_terrain_difficulty(predator.pos_x, predator.pos_y)
                    
                    # Terrain affects predator success differently than prey escape
                    predator_success_modifier = 1.0 / terrain_difficulty  # Harder terrain = lower success
                    success_rate = base_success_rate * predator_success_modifier * self.terrain_accessibility
                    
                    if random.random() < success_rate:
                        prey_energy = other.energy
                        other.die()
                        predator.eat(prey_energy)

    def handle_food_consumption(self):
        """Handle prey eating food patches with vegetation consumption."""
        for prey in self.prey:
            if not prey.is_alive or not prey.is_hungry:
                continue
            pos = (prey.pos_x, prey.pos_y)
            if pos in self.food_patches:
                food_amount = min(self.food_patches[pos], 20)
                prey.eat(food_amount)
                self.food_patches[pos] -= food_amount
                
                # Consume vegetation if vegetation layer exists
                if self.vegetation_layer is not None:
                    self.vegetation_layer.consume_vegetation(prey.pos_x, prey.pos_y, food_amount * 2)
                
                if self.food_patches[pos] <= 0:
                    del self.food_patches[pos]

    def age_agents(self):
        """Age all agents."""
        for agent in self.predators + self.prey:
            if agent.is_alive:
                if isinstance(agent, Predator):
                    agent.age_predator()
                elif isinstance(agent, Prey):
                    agent.age_prey()

    def update_food_growth(self):
        """Update food patches based on climate."""
        growth_factor = self.food_growth_rate
        for pos in list(self.food_patches.keys()):
            self.food_patches[pos] *= (1 + growth_factor * 0.1)
            if self.food_patches[pos] > 100:
                self.food_patches[pos] = 100
        
        if random.random() < growth_factor * 0.05:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.food_patches[(x, y)] = random.uniform(5, 25)

    def cleanup_dead_agents(self):
        """Remove dead agents."""
        self.predators = [p for p in self.predators if p.is_alive]
        self.prey = [p for p in self.prey if p.is_alive]
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = [agent for agent in self.grid[y][x] if agent.is_alive]

    def select_action(self, observation):
        env_policy = self.params.get('env_policy')
        return env_policy(self, observation) if env_policy else {}

    def update(self, feedback):
        env_updater = self.params.get('env_updater')
        if env_updater:
            self.params = env_updater(self.params, feedback)

    def act(self, action):
        if not action:
            return
        if 'temperature_delta' in action:
            self.temperature = max(-20, min(50, self.temperature + action['temperature_delta']))
        if 'precipitation_delta' in action:
            self.precipitation = max(0, min(1, self.precipitation + action['precipitation_delta']))

    def get_stats(self):
        return {
            'timestep': self.timestep,
            'predator_count': len(self.predators),
            'prey_count': len(self.prey),
            'food_patches': len(self.food_patches),
            'temperature': self.temperature,
            'precipitation': self.precipitation,
            'season': ['Spring', 'Summer', 'Fall', 'Winter'][self.season]
        }

    def simulate(self):
        self.step()

    def update_food_from_vegetation(self):
        """Update food patches based on vegetation layer biomass."""
        if self.vegetation_layer is None:
            return
        
        # Clear existing food patches and regenerate from vegetation
        self.food_patches.clear()
        
        for y in range(self.height):
            for x in range(self.width):
                food_amount = self.vegetation_layer.get_food_availability(x, y)
                if food_amount > 1:  # Only create patches with significant food
                    self.food_patches[(x, y)] = food_amount

    def get_terrain_difficulty(self, x: int, y: int) -> float:
        """Get terrain difficulty for movement based on elevation and NDVI."""
        if self.elevation_map is None:
            return 1.0
        
        elevation = self.elevation_map[y, x]
        ndvi = self.ndvi_map[y, x] if self.ndvi_map is not None else 0.5
        
        # Higher elevation = more difficult
        elevation_factor = 1 + (elevation - 500) / 1000  # Normalize around 500m
        
        # Dense vegetation (high NDVI) = more difficult for predators, easier for prey
        vegetation_factor = 1 + (ndvi - 0.5) * 0.5
        
        return max(0.1, min(2.0, elevation_factor * vegetation_factor))