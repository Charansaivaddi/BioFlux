#!/usr/bin/env python3
"""
BioFlux - Climate-driven ecosystem simulation with real-time data integration.

A sophisticated ecosystem simulation that models predator-prey-plant dynamics
with real-time weather and geospatial data integration.
"""

# Core components
from .core.agents import (
    RLAgent,
    Predator, 
    Prey,
    Plant
)

from .core.environment import (
    Environment,
    EnvironmentConfig
)

# Data integration
from .data import (
    GeospatialDataLoader,
    LogisticGrowthModel,
    VegetationLayer,
    TerrainAnalyzer,
    WeatherData,
    WeatherDataManager,
    create_mock_weather_data
)

# Visualization
from .visualization import (
    SimulationVisualizer,
    InteractiveVisualizer,
    create_summary_report
)

# Configuration
from .config import (
    BioFluxConfig,
    get_config,
    reload_config
)

__version__ = "0.3.0"
__author__ = "BioFlux Team"
__description__ = "Climate-driven ecosystem simulation with real-time data integration"

__all__ = [
    # Core
    'RLAgent',
    'Predator',
    'Prey', 
    'Plant',
    'Environment',
    'EnvironmentConfig',
    
    # Data
    'GeospatialDataLoader',
    'LogisticGrowthModel',
    'VegetationLayer',
    'TerrainAnalyzer',
    'WeatherData',
    'WeatherDataManager',
    'create_mock_weather_data',
    
    # Visualization
    'SimulationVisualizer',
    'InteractiveVisualizer',
    'create_summary_report',
    
    # Configuration
    'BioFluxConfig',
    'get_config',
    'reload_config'
]
