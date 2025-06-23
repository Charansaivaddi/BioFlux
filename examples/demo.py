#!/usr/bin/env python3
"""
Demo script for Climate-Driven Predator-Prey Simulation with Real Geospatial Data
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import bioflux
sys.path.insert(0, str(Path(__file__).parent.parent))

import bioflux
from bioflux import (
    Environment, 
    EnvironmentConfig,
    Predator, 
    Prey, 
    Plant,
    SimulationVisualizer,
    get_config
)
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_sample_simulation():
    """Create a sample simulation with geospatial data integration."""
    logger.info("🌍 Initializing Climate-Driven Predator-Prey Simulation...")
    
    # Load configuration
    config = get_config()
    logger.info(f"Configuration: {config}")
    
    # Create environment configuration
    env_config = EnvironmentConfig(
        width=30,
        height=30,
        max_predators=10,
        max_prey=50,
        max_plants=100,
        use_real_data=config.has_live_data(),
        bbox=(-122.5, 37.7, -122.3, 37.9)  # SF Bay Area
    )
    
    # Create environment
    env = Environment(env_config)
    
    logger.info(f"📍 Using bounding box: {env_config.bbox}")
    logger.info(f"🌱 Environment size: {env_config.width}x{env_config.height}")
    logger.info(f"🌍 Live data: {'✓ Active' if config.has_live_data() else '✗ Mock mode'}")
    
    # Create initial population
    logger.info("\n🦁 Creating predators...")
    for i in range(3):
        x, y = random.randint(5, 25), random.randint(5, 25)
        predator = Predator(
            speed=2.0, 
            energy=50.0, 
            pos_x=float(x), 
            pos_y=float(y), 
            age=random.randint(1, 5)
        )
        env.add_predator(predator)
        logger.info(f"  Created predator at ({x}, {y})")
    
    logger.info("🐰 Creating prey...")
    for i in range(8):
        x, y = random.randint(5, 25), random.randint(5, 25)
        prey = Prey(
            speed=3.0, 
            energy=30.0, 
            pos_x=float(x), 
            pos_y=float(y), 
            age=random.randint(1, 3)
        )
        env.add_prey(prey)
        logger.info(f"  Created prey at ({x}, {y})")
    
    logger.info("🌿 Creating plants...")
    for i in range(15):
        x, y = random.randint(0, 29), random.randint(0, 29)
        plant = Plant(
            energy=20.0,
            pos_x=float(x),
            pos_y=float(y)
        )
        env.add_plant(plant)
        logger.info(f"  Created plant at ({x}, {y})")
    
    return env

def run_simulation(env, steps=100):
    """Run the simulation for specified number of steps."""
    logger.info(f"\n🚀 Starting simulation for {steps} steps...")
    
    # Get configuration
    config = get_config()
    
    # Set up visualization
    visualizer = SimulationVisualizer(env)
    fig, axes = visualizer.setup_live_plot()
    
    stats_history = []
    
    for step in range(steps):
        # Step the environment
        env.step()
        
        # Get current stats
        current_stats = env.get_stats()
        stats_history.append(current_stats)
        
        # Log progress
        if step % 10 == 0:
            logger.info(f"Step {step}: Predators={current_stats.get('predators', 0)}, "
                       f"Prey={current_stats.get('prey', 0)}, "
                       f"Plants={current_stats.get('plants', 0)}")
        
        # Update visualization every 10 steps
        if step % 10 == 0 and fig is not None:
            visualizer.update_live_plot(stats_history)
            try:
                import matplotlib.pyplot as plt
                plt.pause(0.1)
            except ImportError:
                pass
    
    logger.info("✅ Simulation complete!")
    
    # Final statistics
    final_stats = stats_history[-1] if stats_history else {}
    logger.info(f"\n📊 Final Statistics:")
    logger.info(f"  Predators: {final_stats.get('predators', 0)}")
    logger.info(f"  Prey: {final_stats.get('prey', 0)}")
    logger.info(f"  Plants: {final_stats.get('plants', 0)}")
    logger.info(f"  Avg Temperature: {final_stats.get('avg_temperature', 0):.1f}°C")
    logger.info(f"  Avg Humidity: {final_stats.get('avg_humidity', 0):.1f}%")
    
    # Save plots if configured
    output_config = config.get_output_config()
    if output_config['save_plots'] and fig is not None:
        output_dir = Path(output_config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        plot_file = output_dir / "simulation_demo.png"
        visualizer.save_plots(str(plot_file))
    
    return stats_history

def main():
    """Main demo function."""
    try:
        # Create and run simulation
        env = create_sample_simulation()
        stats_history = run_simulation(env, steps=50)
        
        # Create summary report
        from bioflux.visualization import create_summary_report
        report_html = create_summary_report(stats_history, "demo_report.html")
        logger.info("📄 Summary report created: demo_report.html")
        
        # Keep plot window open
        try:
            import matplotlib.pyplot as plt
            if plt.get_fignums():  # If there are active figures
                logger.info("🖼️  Close the plot window to exit...")
                plt.show()
        except ImportError:
            logger.info("Matplotlib not available for interactive plotting")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
