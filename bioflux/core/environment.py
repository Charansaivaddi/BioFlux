#!/usr/bin/env python3
"""
Environment class for BioFlux ecosystem simulation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import random
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnvironmentConfig:
    """Configuration for the simulation environment."""
    width: int = 100
    height: int = 100
    max_predators: int = 20
    max_prey: int = 100
    max_plants: int = 200
    energy_decay_rate: float = 0.01
    reproduction_threshold: float = 80.0
    mutation_rate: float = 0.1
    use_real_data: bool = False
    bbox: Optional[Tuple[float, float, float, float]] = None

class Environment:
    """
    Ecosystem simulation environment with climate and geospatial data integration.
    
    This class manages the simulation space, agents, and environmental factors
    including temperature, humidity, vegetation, and terrain data.
    """
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """
        Initialize the environment.
        
        Args:
            config: Environment configuration. If None, uses default values.
        """
        self.config = config or EnvironmentConfig()
        
        # Environment dimensions
        self.width = self.config.width
        self.height = self.config.height
        
        # Agent containers
        self.predators = []
        self.prey = []
        self.plants = []
        
        # Environmental layers
        self.temperature_map = np.ones((self.height, self.width)) * 20.0  # Celsius
        self.humidity_map = np.ones((self.height, self.width)) * 60.0     # Percentage
        self.vegetation_map = np.ones((self.height, self.width)) * 0.5    # Density [0-1]
        self.elevation_map = np.ones((self.height, self.width)) * 100.0   # Meters
        
        # Simulation state
        self.step_count = 0
        self.total_energy = 0.0
        self.stats_history = []
        
        # Real data integration
        self.real_data_loader = None
        self.weather_data = None
        
        # Initialize environment
        self._initialize_environment()
        
    def _initialize_environment(self):
        """Initialize the environment with base conditions."""
        logger.info(f"Initializing environment ({self.width}x{self.height})")
        
        # Initialize with realistic patterns
        self._initialize_temperature()
        self._initialize_humidity()
        self._initialize_vegetation()
        self._initialize_elevation()
        
    def _initialize_temperature(self):
        """Initialize temperature map with realistic patterns."""
        # Create temperature gradient with some randomness
        y_coords = np.linspace(0, 1, self.height)
        temp_gradient = 25 - 10 * y_coords  # Warmer at top, cooler at bottom
        
        for i in range(self.height):
            self.temperature_map[i, :] = temp_gradient[i] + np.random.normal(0, 2, self.width)
        
        # Smooth the temperature map
        from scipy import ndimage
        self.temperature_map = ndimage.gaussian_filter(self.temperature_map, sigma=2)
        
    def _initialize_humidity(self):
        """Initialize humidity map with realistic patterns."""
        # Higher humidity near water bodies and vegetation
        humidity_base = 60 + np.random.normal(0, 10, (self.height, self.width))
        
        # Add water bodies (higher humidity)
        water_centers = [(20, 30), (80, 70)]
        for wx, wy in water_centers:
            y, x = np.ogrid[:self.height, :self.width]
            mask = (x - wx)**2 + (y - wy)**2 < 200
            humidity_base[mask] += 20
        
        self.humidity_map = np.clip(humidity_base, 0, 100)
        
    def _initialize_vegetation(self):
        """Initialize vegetation map with realistic patterns."""
        # Base vegetation density
        veg_base = np.random.beta(2, 2, (self.height, self.width))
        
        # Add forest patches
        forest_centers = [(40, 60), (75, 25)]
        for fx, fy in forest_centers:
            y, x = np.ogrid[:self.height, :self.width]
            mask = (x - fx)**2 + (y - fy)**2 < 300
            veg_base[mask] = np.maximum(veg_base[mask], 0.8)
        
        # Reduce vegetation in "urban" areas
        urban_centers = [(10, 10), (90, 90)]
        for ux, uy in urban_centers:
            y, x = np.ogrid[:self.height, :self.width]
            mask = (x - ux)**2 + (y - uy)**2 < 100
            veg_base[mask] = np.minimum(veg_base[mask], 0.2)
        
        self.vegetation_map = veg_base
        
    def _initialize_elevation(self):
        """Initialize elevation map with realistic terrain."""
        # Create hills and valleys
        x = np.linspace(0, 4 * np.pi, self.width)
        y = np.linspace(0, 4 * np.pi, self.height)
        X, Y = np.meshgrid(x, y)
        
        elevation = (
            100 * np.sin(X * 0.5) * np.cos(Y * 0.5) +
            50 * np.sin(X * 1.2) +
            30 * np.cos(Y * 1.5) +
            np.random.normal(0, 10, (self.height, self.width))
        )
        
        self.elevation_map = elevation + 200  # Base elevation
        
    def add_predator(self, predator):
        """Add a predator to the environment."""
        if len(self.predators) < self.config.max_predators:
            self.predators.append(predator)
            return True
        return False
        
    def add_prey(self, prey):
        """Add prey to the environment."""
        if len(self.prey) < self.config.max_prey:
            self.prey.append(prey)
            return True
        return False
        
    def add_plant(self, plant):
        """Add a plant to the environment."""
        if len(self.plants) < self.config.max_plants:
            self.plants.append(plant)
            return True
        return False
    
    @property
    def agents(self):
        """Get all agents in the environment."""
        return self.predators + self.prey + self.plants
    
    def add_agent(self, agent):
        """Add an agent to the environment."""
        if agent.agent_type == 'predator':
            return self.add_predator(agent)
        elif agent.agent_type == 'prey':
            return self.add_prey(agent)
        elif agent.agent_type == 'plant':
            return self.add_plant(agent)
        else:
            logger.warning(f"Unknown agent type: {agent.agent_type}")
            return False
    
    def get_local_environment(self, x: int, y: int) -> Dict[str, float]:
        """
        Get environmental conditions at a specific location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Dictionary of environmental conditions
        """
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        
        return {
            'temperature': self.temperature_map[y, x],
            'humidity': self.humidity_map[y, x],
            'vegetation': self.vegetation_map[y, x],
            'elevation': self.elevation_map[y, x]
        }
    
    def update_environment(self, weather_data: Optional[Dict] = None):
        """
        Update environmental conditions based on weather data.
        
        Args:
            weather_data: Real weather data from APIs
        """
        if weather_data:
            # Update temperature map based on real weather
            temp_adjustment = weather_data.get('temperature', 20) - np.mean(self.temperature_map)
            self.temperature_map += temp_adjustment * 0.1  # Gradual adjustment
            
            # Update humidity map
            humidity_adjustment = weather_data.get('humidity', 60) - np.mean(self.humidity_map)
            self.humidity_map += humidity_adjustment * 0.1
            
            # Update vegetation based on precipitation
            if 'precipitation' in weather_data:
                precipitation = weather_data['precipitation']
                vegetation_growth = np.clip(precipitation / 100, 0, 0.1)
                self.vegetation_map = np.clip(self.vegetation_map + vegetation_growth, 0, 1)
    
    def get_carrying_capacity(self, x: int, y: int) -> float:
        """
        Calculate carrying capacity at a location based on environmental factors.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Carrying capacity value
        """
        env_data = self.get_local_environment(x, y)
        
        # Optimal temperature range for life
        temp_factor = 1.0 - abs(env_data['temperature'] - 22) / 30
        temp_factor = max(0.1, temp_factor)
        
        # Humidity factor
        humidity_factor = 1.0 - abs(env_data['humidity'] - 70) / 100
        humidity_factor = max(0.1, humidity_factor)
        
        # Vegetation factor
        vegetation_factor = env_data['vegetation']
        
        return temp_factor * humidity_factor * vegetation_factor * 100
    
    def step(self):
        """Execute one simulation step."""
        self.step_count += 1
        
        # Update agents
        self._update_agents()
        
        # Update environment
        self.update_environment()
        
        # Record statistics
        self._record_stats()
        
    def _update_agents(self):
        """Update all agents in the environment."""
        # Update predators
        for predator in self.predators[:]:  # Copy list to avoid modification during iteration
            if predator.is_alive:
                # Get observation and select action
                observation = predator.get_observation(self)
                action = predator.select_action(observation)
                predator.act(action)
                
                # Age and check survival
                predator.age_step()
                if not predator.is_alive:
                    self.predators.remove(predator)
        
        # Update prey
        for prey_agent in self.prey[:]:
            if prey_agent.is_alive:
                # Get observation and select action
                observation = prey_agent.get_observation(self)
                action = prey_agent.select_action(observation)
                prey_agent.act(action)
                
                # Age and check survival
                prey_agent.age_step()
                if not prey_agent.is_alive:
                    self.prey.remove(prey_agent)
        
        # Update plants
        for plant in self.plants[:]:
            if plant.is_alive:
                # Get observation and select action
                observation = plant.get_observation(self)
                action = plant.select_action(observation)
                plant.act(action)
                
                # Age and check survival
                plant.age_step()
                if not plant.is_alive:
                    self.plants.remove(plant)
    
    def _record_stats(self):
        """Record current simulation statistics."""
        stats = {
            'step': self.step_count,
            'predators': len(self.predators),
            'prey': len(self.prey),
            'plants': len(self.plants),
            'avg_temperature': np.mean(self.temperature_map),
            'avg_humidity': np.mean(self.humidity_map),
            'avg_vegetation': np.mean(self.vegetation_map),
            'total_energy': sum(agent.energy for agent in self.predators + self.prey)
        }
        self.stats_history.append(stats)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current environment statistics."""
        if not self.stats_history:
            return {}
        return self.stats_history[-1]
    
    def reset(self):
        """Reset the environment to initial state."""
        self.predators.clear()
        self.prey.clear()
        self.plants.clear()
        self.step_count = 0
        self.total_energy = 0.0
        self.stats_history.clear()
        self._initialize_environment()
        
    def __str__(self):
        return (f"Environment({self.width}x{self.height}, "
                f"predators={len(self.predators)}, "
                f"prey={len(self.prey)}, "
                f"plants={len(self.plants)}, "
                f"step={self.step_count})")
