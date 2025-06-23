# 🧬 BioFlux: Climate-Driven Ecosystem Simulation

A sophisticated **Python package** for ecosystem simulation that models predator-prey-plant dynamics with **real-time weather and geospatial data integration**. Built for research, education, and environmental modeling.

## � What's New in v0.3.0

🚀 **Complete Package Restructure**: BioFlux is now a professional Python package with modular architecture:
- **Modular Design**: Clean separation of core logic, data integration, visualization, and configuration
- **Professional APIs**: Well-documented interfaces with type hints and docstrings  
- **Easy Installation**: Standard Python package with pip/uv installation
- **Extensible**: Plugin architecture for adding new data sources and agent behaviors
- **Production Ready**: Comprehensive error handling, logging, and configuration management

## 🎯 Core Features

### � **Real-Time Data Integration**
- **Live Weather Data**: OpenWeatherMap API integration for real temperature, humidity, precipitation
- **Satellite Imagery**: Sentinel Hub API for NDVI vegetation data from Sentinel-2
- **Elevation Data**: Free USGS elevation service integration
- **Climate-Driven Dynamics**: Environmental conditions directly affect agent behavior and survival

### 🤖 **Advanced Agent System**
- **Reinforcement Learning Agents**: All agents (predators, prey, plants) use RL for decision-making
- **Adaptive Behavior**: Agents learn and adapt strategies based on environmental feedback
- **Realistic Ecology**: Predator-prey-plant food web with realistic energy flows
- **Individual-Based Modeling**: Each agent has unique properties, age, energy, and learning history

### 📊 **Rich Visualization Suite**
- **Real-Time Plots**: Live updating population dynamics, environmental maps, and agent positions
- **Interactive Dashboards**: Plotly-based interactive visualizations
- **3D Environment Visualization**: Terrain-aware 3D plots with agent movements
- **Export Capabilities**: Save plots, animations, and comprehensive HTML reports

### ⚙️ **Flexible Configuration**
- **API Key Management**: Secure handling of weather and satellite API keys
- **Environment Variables**: Easy configuration via `.env` files
- **Simulation Parameters**: Customizable population sizes, environment dimensions, and behavior parameters
- **Mock Data Mode**: Works offline with realistic simulated data when APIs unavailable

## 🚀 Quick Start

### Installation
```bash
# Install with uv (recommended)
uv add bioflux

# Or with pip
pip install -e .

# Install dependencies
uv sync
```

### Basic Usage
```python
import bioflux

# Create simulation environment
config = bioflux.EnvironmentConfig(width=50, height=50)
env = bioflux.Environment(config)

# Add agents
predator = bioflux.Predator(speed=2.0, energy=50.0, pos_x=10.0, pos_y=10.0, age=1)
prey = bioflux.Prey(speed=3.0, energy=30.0, pos_x=20.0, pos_y=20.0, age=1)
plant = bioflux.Plant(energy=20.0, pos_x=15.0, pos_y=15.0)

env.add_predator(predator)
env.add_prey(prey)  
env.add_plant(plant)

# Run simulation
for step in range(100):
    env.step()
    if step % 10 == 0:
        stats = env.get_stats()
        print(f"Step {step}: {stats}")
```

### Configuration Setup
```python
# Load configuration
config = bioflux.get_config()
print(f"Live data available: {config.has_live_data()}")

# Check API status
apis = config.get_available_apis()
print(f"Weather APIs: {apis['weather']}")
print(f"Satellite APIs: {apis['satellite']}")
```

### Run Examples
```bash
# Test package structure
python examples/test_structure.py

# Test configuration  
python examples/config_test.py

# Run main demonstration
python examples/demo.py

# Or use legacy scripts (still supported)
python demo.py
python config.py
```

## 📁 Package Structure

```
bioflux/
├── __init__.py                 # Package interface
├── core/                       # Core simulation engine
│   ├── agents.py              # RL-enabled agent classes
│   └── environment.py         # Simulation environment
├── data/                      # Data integration
│   ├── geospatial.py          # NDVI, elevation, terrain
│   └── weather.py             # Weather APIs and climate data
├── visualization/             # Plotting and visualization
│   └── plots.py               # Advanced plotting suite
└── config/                    # Configuration management
    └── settings.py            # API keys and settings

examples/                      # Example scripts and demos
tests/                         # Unit tests (coming soon)
docs/                          # Documentation (expanding)
```

## 🔧 Configuration

### API Keys Setup
Create a `.env` file with your API keys:
```env
# Weather Data (choose one or more)
OPENWEATHER_API_KEY=your_openweather_key_here
WEATHER_UNDERGROUND_API_KEY=your_wu_key_here

# Satellite Data (optional)
SENTINELHUB_CLIENT_ID=your_client_id
SENTINELHUB_CLIENT_SECRET=your_client_secret
SENTINELHUB_INSTANCE_ID=your_instance_id

# NASA EarthData (optional)  
NASA_EARTHDATA_USERNAME=your_username
NASA_EARTHDATA_PASSWORD=your_password

# Simulation Settings (optional)
SIMULATION_WIDTH=100
SIMULATION_HEIGHT=100
MAX_PREDATORS=20
MAX_PREY=100
MAX_PLANTS=200
```

### Mock Data Mode
BioFlux works great even without API keys! When no live data is available, it automatically uses realistic mock data that simulates:
- Weather patterns based on geographic location
- NDVI vegetation data with seasonal variations
- Elevation data with realistic terrain features
    stats = env.get_stats()
    print(f"Step {step}: Predators={stats['predator_count']}, Prey={stats['prey_count']}")
```

## 🧠 Advanced RL Integration

### Custom Agent Policies
```python
def smart_predator_policy(observation, agent):
    """Advanced predator policy using geospatial data."""
    geospatial = observation.get('geospatial', {})
    
    # Use elevation and vegetation for hunting strategy
    if geospatial.get('elevation', 500) > 800:
        return {'type': 'move', 'x': agent.pos_x, 'y': agent.pos_y - 1}  # Move downhill
    
    # Standard chase behavior in good terrain
    if observation.get('prey'):
        closest_prey = min(observation['prey'], key=lambda p: abs(p[0]) + abs(p[1]))
        dx, dy = closest_prey[0], closest_prey[1]
        return {'type': 'move', 'x': agent.pos_x + dx, 'y': agent.pos_y + dy}
    
    return {'type': 'move', 'x': agent.pos_x, 'y': agent.pos_y}  # Stay

# Inject custom policy
predator = Predator(speed=2, energy=50, pos_x=25, pos_y=25, age=1,
                   init_params={'policy_fn': smart_predator_policy})
```

### Environment as RL Agent
```python
def climate_adaptation_policy(env, observation):
    """Environment policy for climate management."""
    if env.timestep % 50 == 0:  # Every 50 steps
        return {
            'temperature_delta': random.uniform(-2, 2),
            'precipitation_delta': random.uniform(-0.1, 0.1)
        }
    return {}

env = Environment(width=50, height=50, 
                 init_params={'env_policy': climate_adaptation_policy})
```

## 📊 Geospatial Data Integration

### NDVI and Elevation
```python
# Access real-time geospatial data
ndvi_value = env.ndvi_map[y, x]           # Vegetation index [0-1]
elevation = env.elevation_map[y, x]       # Meters above sea level
biomass = env.vegetation_layer.biomass[y, x]  # Current vegetation biomass

# Terrain difficulty affects movement
difficulty = env.get_terrain_difficulty(x, y)
```

### Vegetation Dynamics
```python
# Logistic growth with environmental factors
from geospatial import LogisticGrowthModel

model = LogisticGrowthModel(carrying_capacity=100, growth_rate=0.1)
new_biomass = model.growth(
    population=current_biomass,
    environmental_factor=model.environmental_factor(
        ndvi=0.8, elevation=600, temperature=22, precipitation=0.6
    )
)
```

## 🔬 Research Applications

### 1. **Climate Impact Studies**
- Study population stability under different climate scenarios
- Analyze extinction risk during droughts or temperature extremes
- Model species migration patterns with changing vegetation

### 2. **AI Training Environments**
- Train RL agents in realistic, data-driven environments
- Develop adaptive policies for uncertain environmental conditions
- Study emergent behaviors in multi-agent systems

### 3. **Conservation Biology**
- Model habitat fragmentation effects
- Optimize wildlife corridor placement
- Predict species response to land use changes

### 4. **Ecosystem Management**
- Test intervention strategies (reforestation, grazing management)
- Model carrying capacity under different management scenarios
- Study predator reintroduction programs

## 📈 Evaluation Metrics

The framework tracks comprehensive metrics for research:

```python
stats = env.get_stats()
# Returns:
{
    'timestep': 150,
    'predator_count': 3,
    'prey_count': 8, 
    'food_patches': 45,
    'temperature': 23.5,
    'precipitation': 0.7,
    'season': 'Summer'
}
```

### Custom Metrics
- **Population Stability**: Coefficient of variation in population sizes
- **Biodiversity Index**: Shannon diversity of agent behaviors
- **Resource Utilization**: Efficiency of vegetation consumption
- **Habitat Quality**: NDVI-weighted carrying capacity
- **Climate Resilience**: Population recovery after disturbances

## 🛠️ Extension Points

### Adding New Data Sources
```python
# Integrate with NASA EarthData API
def fetch_real_ndvi(bbox, date):
    # Your NASA API integration
    return ndvi_array

# Hook into climate updates
def update_climate_from_ibm(self):
    # IBM Watson environmental data
    self.temperature = get_ibm_temperature(self.bbox, self.timestep)
    self.precipitation = get_ibm_precipitation(self.bbox, self.timestep)
```

### Multi-Species Extensions
```python
class Herbivore(RLAgent):
    """New agent type that only eats vegetation"""
    def default_policy(self, observation):
        # Seek high-NDVI areas
        return seek_vegetation_policy(observation)

class Carnivore(RLAgent):
    """Specialist predator with different hunting patterns"""
    def default_policy(self, observation):
        # Hunt specific prey types
        return specialist_hunting_policy(observation)
```

## 📚 Scientific Background

This simulation implements:

1. **Lotka-Volterra Extensions** with environmental stochasticity
2. **Logistic Growth Models** for vegetation dynamics  
3. **Spatial Ecology** principles with realistic terrain effects
4. **Climate Change Biology** with data-driven environmental variation
5. **Agent-Based Modeling** best practices for ecological systems

## 🤝 Contributing

We welcome contributions! Areas of interest:
- New data source integrations (Sentinel, Landsat, MODIS)
- Advanced RL algorithms (PPO, SAC, multi-agent)
- Visualization improvements (interactive maps, 3D terrain)
- Performance optimizations (GPU acceleration, parallelization)

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **NASA** for SRTM elevation and Landsat NDVI data
- **IBM** for Environmental Intelligence Suite integration
- **Google Earth Engine** for geospatial processing examples
- **OpenAI Gym** for RL environment design patterns

---

**Ready to explore climate-driven ecosystem dynamics? Start with `python demo.py`!** 🚀
