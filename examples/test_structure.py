#!/usr/bin/env python3
"""
Configuration test script for BioFlux package structure.
"""

import sys
from pathlib import Path

# Add parent directory to path to import bioflux
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all bioflux imports."""
    logger.info("🧪 Testing BioFlux package imports...")
    
    try:
        import bioflux
        logger.info(f"✅ bioflux v{bioflux.__version__}")
    except Exception as e:
        logger.error(f"❌ Failed to import bioflux: {e}")
        return False
    
    # Test core imports
    try:
        from bioflux import Environment, EnvironmentConfig
        logger.info("✅ Core environment classes")
    except Exception as e:
        logger.error(f"❌ Failed to import environment: {e}")
        return False
    
    try:
        from bioflux import Predator, Prey, Plant, RLAgent
        logger.info("✅ Agent classes")
    except Exception as e:
        logger.error(f"❌ Failed to import agents: {e}")
        return False
    
    # Test data imports
    try:
        from bioflux import GeospatialDataLoader, WeatherDataManager
        logger.info("✅ Data integration classes")
    except Exception as e:
        logger.error(f"❌ Failed to import data classes: {e}")
        return False
    
    # Test visualization imports
    try:
        from bioflux import SimulationVisualizer
        logger.info("✅ Visualization classes")
    except Exception as e:
        logger.error(f"❌ Failed to import visualization: {e}")
        return False
    
    # Test configuration imports
    try:
        from bioflux import get_config, BioFluxConfig
        logger.info("✅ Configuration classes")
    except Exception as e:
        logger.error(f"❌ Failed to import configuration: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading."""
    logger.info("\n🔧 Testing configuration...")
    
    try:
        from bioflux import get_config
        config = get_config()
        
        logger.info(f"✅ Configuration loaded: {config}")
        logger.info(f"🌍 Live data available: {config.has_live_data()}")
        
        # Test configuration access
        sim_config = config.get_simulation_config()
        logger.info(f"📐 Simulation size: {sim_config['width']}x{sim_config['height']}")
        
        apis = config.get_available_apis()
        logger.info(f"🔌 Available APIs: {apis}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_simulation():
    """Test basic simulation creation."""
    logger.info("\n🎮 Testing basic simulation...")
    
    try:
        from bioflux import Environment, EnvironmentConfig, Predator, Prey
        
        # Create simple environment
        config = EnvironmentConfig(width=10, height=10, max_predators=2, max_prey=5)
        env = Environment(config)
        
        # Create agents
        predator = Predator(speed=1.0, energy=50.0, pos_x=5.0, pos_y=5.0, age=1)
        prey = Prey(speed=2.0, energy=30.0, pos_x=3.0, pos_y=3.0, age=1)
        
        # Add to environment
        env.add_predator(predator)
        env.add_prey(prey)
        
        # Test simulation step
        initial_stats = env.get_stats()
        env.step()
        final_stats = env.get_stats()
        
        logger.info(f"✅ Simulation test passed")
        logger.info(f"   Initial: {initial_stats}")
        logger.info(f"   After step: {final_stats}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("🧬 BioFlux Package Structure Test")
    logger.info("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test configuration
    if not test_configuration():
        all_passed = False
    
    # Test basic simulation
    if not test_basic_simulation():
        all_passed = False
    
    # Final result
    logger.info("\n" + "=" * 50)
    if all_passed:
        logger.info("🎉 All tests passed! Package structure is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
