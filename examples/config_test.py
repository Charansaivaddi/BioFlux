#!/usr/bin/env python3
"""
Configuration script using the new BioFlux package structure.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bioflux import get_config, reload_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test and display configuration."""
    logger.info("🔧 BioFlux Configuration Test")
    logger.info("=" * 50)
    
    try:
        # Load configuration
        config = get_config()
        
        # Display configuration status
        logger.info("📊 Configuration Status:")
        logger.info(f"  Live data available: {config.has_live_data()}")
        
        # Show available APIs
        apis = config.get_available_apis()
        logger.info(f"  Weather APIs: {apis['weather']}")
        logger.info(f"  Satellite APIs: {apis['satellite']}")
        
        # Show simulation settings
        sim_config = config.get_simulation_config()
        logger.info(f"  Simulation size: {sim_config['width']}x{sim_config['height']}")
        logger.info(f"  Max agents: {sim_config['max_predators']} predators, {sim_config['max_prey']} prey")
        
        # Validate configuration
        validation = config.validate_config()
        if validation['errors']:
            logger.error("Configuration errors:")
            for error in validation['errors']:
                logger.error(f"  - {error}")
        
        if validation['warnings']:
            logger.warning("Configuration warnings:")
            for warning in validation['warnings']:
                logger.warning(f"  - {warning}")
        
        if not validation['errors']:
            logger.info("✅ Configuration is valid!")
        
        return 0 if not validation['errors'] else 1
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
