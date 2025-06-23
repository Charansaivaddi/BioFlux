#!/usr/bin/env python3
"""
Advanced visualization suite for BioFlux ecosystem simulation.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import ListedColormap
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    
    # Set style for better-looking plots
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    sns.set_palette("husl")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    animation = None
    ListedColormap = None
    sns = None
    logger.warning("Matplotlib/Seaborn not available. Visualization features disabled.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None
    logger.warning("Plotly not available. Interactive features disabled.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    logger.warning("Pandas not available. Some data processing features disabled.")

class SimulationVisualizer:
    """Advanced visualization suite for the predator-prey simulation."""
    
    def __init__(self, env, figsize=(20, 15)):
        """
        Initialize the simulation visualizer.
        
        Args:
            env: Environment object
            figsize: Figure size for matplotlib plots
        """
        self.env = env
        self.figsize = figsize
        self.fig = None
        self.axes = None
        self.stats_history = []
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Visualization features limited.")
        
    def setup_live_plot(self):
        """Set up matplotlib figure for live plotting."""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for live plotting")
            return None, None
        
        self.fig, self.axes = plt.subplots(3, 4, figsize=self.figsize)
        self.fig.suptitle('🧬 BioFlux: Real-Time Climate-Driven Ecosystem Simulation', 
                         fontsize=16, fontweight='bold')
        
        # Initialize empty plots
        self.population_line = None
        self.climate_line = None
        self.food_line = None
        
        plt.tight_layout()
        return self.fig, self.axes
    
    def update_live_plot(self, stats_history: List[Dict]):
        """
        Update live plots with new data.
        
        Args:
            stats_history: List of simulation statistics
        """
        if not MATPLOTLIB_AVAILABLE or not self.fig:
            return
        
        self.stats_history = stats_history
        
        if not stats_history:
            return
        
        # Extract data for plotting
        steps = [s['step'] for s in stats_history]
        predators = [s.get('predators', 0) for s in stats_history]
        prey = [s.get('prey', 0) for s in stats_history]
        plants = [s.get('plants', 0) for s in stats_history]
        
        # Clear previous plots
        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()
        
        # Population dynamics plot
        ax = self.axes[0, 0]
        ax.plot(steps, predators, 'r-', label='Predators', linewidth=2)
        ax.plot(steps, prey, 'b-', label='Prey', linewidth=2)
        ax.plot(steps, plants, 'g-', label='Plants', linewidth=2)
        ax.set_title('Population Dynamics Over Time', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Population')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Environmental conditions
        if 'avg_temperature' in stats_history[-1]:
            ax = self.axes[0, 1]
            temps = [s.get('avg_temperature', 20) for s in stats_history]
            humidity = [s.get('avg_humidity', 60) for s in stats_history]
            ax.plot(steps, temps, 'orange', label='Temperature (°C)', linewidth=2)
            ax2 = ax.twinx()
            ax2.plot(steps, humidity, 'cyan', label='Humidity (%)', linewidth=2)
            ax.set_title('Environmental Conditions', fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Temperature (°C)', color='orange')
            ax2.set_ylabel('Humidity (%)', color='cyan')
            ax.grid(True, alpha=0.3)
        
        # Vegetation coverage
        if 'avg_vegetation' in stats_history[-1]:
            ax = self.axes[0, 2]
            vegetation = [s.get('avg_vegetation', 0.5) for s in stats_history]
            ax.plot(steps, vegetation, 'green', linewidth=2)
            ax.set_title('Average Vegetation Density', fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Vegetation Density')
            ax.grid(True, alpha=0.3)
        
        # Energy distribution
        if 'total_energy' in stats_history[-1]:
            ax = self.axes[0, 3]
            energy = [s.get('total_energy', 0) for s in stats_history]
            ax.plot(steps, energy, 'purple', linewidth=2)
            ax.set_title('Total System Energy', fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Energy')
            ax.grid(True, alpha=0.3)
        
        # Current environment heatmaps
        self._plot_environment_heatmaps()
        
        plt.tight_layout()
        plt.draw()
    
    def _plot_environment_heatmaps(self):
        """Plot environmental condition heatmaps."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Temperature map
        ax = self.axes[1, 0]
        im = ax.imshow(self.env.temperature_map, cmap='coolwarm', aspect='auto')
        ax.set_title('Temperature Map (°C)', fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Humidity map
        ax = self.axes[1, 1]
        im = ax.imshow(self.env.humidity_map, cmap='Blues', aspect='auto')
        ax.set_title('Humidity Map (%)', fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Vegetation map
        ax = self.axes[1, 2]
        im = ax.imshow(self.env.vegetation_map, cmap='Greens', aspect='auto')
        ax.set_title('Vegetation Density', fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Elevation map
        ax = self.axes[1, 3]
        im = ax.imshow(self.env.elevation_map, cmap='terrain', aspect='auto')
        ax.set_title('Elevation Map (m)', fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Agent positions
        self._plot_agent_positions()
    
    def _plot_agent_positions(self):
        """Plot current agent positions."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        ax = self.axes[2, 0]
        
        # Plot agents
        if self.env.predators:
            pred_x = [p.pos_x for p in self.env.predators if p.is_alive]
            pred_y = [p.pos_y for p in self.env.predators if p.is_alive]
            ax.scatter(pred_x, pred_y, c='red', s=50, marker='^', 
                      label=f'Predators ({len(pred_x)})', alpha=0.7)
        
        if self.env.prey:
            prey_x = [p.pos_x for p in self.env.prey if p.is_alive]
            prey_y = [p.pos_y for p in self.env.prey if p.is_alive]
            ax.scatter(prey_x, prey_y, c='blue', s=30, marker='o', 
                      label=f'Prey ({len(prey_x)})', alpha=0.7)
        
        if self.env.plants:
            plant_x = [p.pos_x for p in self.env.plants if p.is_alive]
            plant_y = [p.pos_y for p in self.env.plants if p.is_alive]
            ax.scatter(plant_x, plant_y, c='green', s=20, marker='s', 
                      label=f'Plants ({len(plant_x)})', alpha=0.5)
        
        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_title('Agent Positions', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Phase space plot
        self._plot_phase_space()
    
    def _plot_phase_space(self):
        """Plot predator-prey phase space."""
        if not MATPLOTLIB_AVAILABLE or len(self.stats_history) < 2:
            return
        
        ax = self.axes[2, 1]
        
        predators = [s.get('predators', 0) for s in self.stats_history]
        prey = [s.get('prey', 0) for s in self.stats_history]
        
        if predators and prey:
            ax.plot(prey, predators, 'b-', alpha=0.7, linewidth=2)
            ax.scatter(prey[-1], predators[-1], c='red', s=100, marker='o')
            ax.set_xlabel('Prey Population')
            ax.set_ylabel('Predator Population')
            ax.set_title('Predator-Prey Phase Space', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def save_plots(self, filename: str):
        """
        Save current plots to file.
        
        Args:
            filename: Output filename
        """
        if MATPLOTLIB_AVAILABLE and self.fig:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {filename}")
    
    def create_animation(self, stats_history: List[Dict], 
                        filename: str = "simulation_animation.gif",
                        interval: int = 200) -> Optional[Any]:
        """
        Create animation of the simulation.
        
        Args:
            stats_history: Complete simulation history
            filename: Output filename
            interval: Animation interval in milliseconds
            
        Returns:
            Animation object or None
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for animation")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('BioFlux Simulation Animation', fontsize=14, fontweight='bold')
        
        def animate(frame):
            # Clear axes
            for ax_row in axes:
                for ax in ax_row:
                    ax.clear()
            
            # Get data up to current frame
            current_data = stats_history[:frame+1]
            if not current_data:
                return
            
            steps = [s['step'] for s in current_data]
            predators = [s.get('predators', 0) for s in current_data]
            prey = [s.get('prey', 0) for s in current_data]
            plants = [s.get('plants', 0) for s in current_data]
            
            # Population plot
            ax = axes[0, 0]
            ax.plot(steps, predators, 'r-', label='Predators', linewidth=2)
            ax.plot(steps, prey, 'b-', label='Prey', linewidth=2)
            ax.plot(steps, plants, 'g-', label='Plants', linewidth=2)
            ax.set_title(f'Population Dynamics (Step {frame})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add other plots...
            
        frames = len(stats_history)
        anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                     interval=interval, blit=False)
        
        # Save animation
        try:
            anim.save(filename, writer='pillow', fps=5)
            logger.info(f"Animation saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save animation: {e}")
        
        return anim

class InteractiveVisualizer:
    """Interactive visualization using Plotly."""
    
    def __init__(self, env):
        """
        Initialize interactive visualizer.
        
        Args:
            env: Environment object
        """
        self.env = env
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Interactive features disabled.")
    
    def create_dashboard(self, stats_history: List[Dict]) -> Optional[Any]:
        """
        Create interactive dashboard.
        
        Args:
            stats_history: Simulation statistics history
            
        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for interactive dashboard")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Population Dynamics', 'Environmental Conditions',
                          'Agent Distribution', 'Phase Space'),
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        if not stats_history:
            return fig
        
        steps = [s['step'] for s in stats_history]
        predators = [s.get('predators', 0) for s in stats_history]
        prey = [s.get('prey', 0) for s in stats_history]
        plants = [s.get('plants', 0) for s in stats_history]
        
        # Population dynamics
        fig.add_trace(go.Scatter(x=steps, y=predators, name='Predators', 
                               line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=steps, y=prey, name='Prey', 
                               line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=steps, y=plants, name='Plants', 
                               line=dict(color='green')), row=1, col=1)
        
        # Environmental conditions
        if 'avg_temperature' in stats_history[-1]:
            temps = [s.get('avg_temperature', 20) for s in stats_history]
            humidity = [s.get('avg_humidity', 60) for s in stats_history]
            
            fig.add_trace(go.Scatter(x=steps, y=temps, name='Temperature', 
                                   line=dict(color='orange')), row=1, col=2)
            fig.add_trace(go.Scatter(x=steps, y=humidity, name='Humidity', 
                                   line=dict(color='cyan')), row=1, col=2)
        
        # Phase space
        if len(predators) > 1 and len(prey) > 1:
            fig.add_trace(go.Scatter(x=prey, y=predators, mode='lines+markers',
                                   name='Phase Space', line=dict(color='purple')), 
                         row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title='BioFlux Interactive Dashboard',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_environment_3d(self) -> Optional[Any]:
        """
        Create 3D visualization of environment.
        
        Returns:
            Plotly 3D figure or None
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for 3D visualization")
            return None
        
        # Create meshgrid for surface plot
        x = np.arange(self.env.width)
        y = np.arange(self.env.height)
        X, Y = np.meshgrid(x, y)
        
        fig = go.Figure(data=[
            go.Surface(
                x=X, y=Y, z=self.env.elevation_map,
                colorscale='terrain',
                name='Elevation'
            )
        ])
        
        # Add agent positions as 3D scatter
        if self.env.predators:
            pred_x = [p.pos_x for p in self.env.predators if p.is_alive]
            pred_y = [p.pos_y for p in self.env.predators if p.is_alive]
            pred_z = [self.env.elevation_map[int(p.pos_y), int(p.pos_x)] + 10 
                     for p in self.env.predators if p.is_alive]
            
            fig.add_trace(go.Scatter3d(
                x=pred_x, y=pred_y, z=pred_z,
                mode='markers',
                marker=dict(size=8, color='red', symbol='diamond'),
                name='Predators'
            ))
        
        fig.update_layout(
            title='3D Environment with Agents',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Elevation (m)'
            )
        )
        
        return fig

def create_summary_report(stats_history: List[Dict], 
                         filename: str = "simulation_report.html") -> str:
    """
    Create HTML summary report of simulation.
    
    Args:
        stats_history: Complete simulation history
        filename: Output filename
        
    Returns:
        HTML content as string
    """
    if not stats_history:
        return "<html><body><h1>No simulation data available</h1></body></html>"
    
    final_stats = stats_history[-1]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BioFlux Simulation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
            .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .stat-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .label {{ font-size: 14px; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧬 BioFlux Simulation Report</h1>
            <p>Ecosystem simulation completed after {len(stats_history)} steps</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <div class="metric">{final_stats.get('predators', 0)}</div>
                <div class="label">Final Predators</div>
            </div>
            <div class="stat-box">
                <div class="metric">{final_stats.get('prey', 0)}</div>
                <div class="label">Final Prey</div>
            </div>
            <div class="stat-box">
                <div class="metric">{final_stats.get('plants', 0)}</div>
                <div class="label">Final Plants</div>
            </div>
            <div class="stat-box">
                <div class="metric">{final_stats.get('avg_temperature', 0):.1f}°C</div>
                <div class="label">Avg Temperature</div>
            </div>
        </div>
        
        <h2>Simulation Summary</h2>
        <ul>
            <li>Duration: {len(stats_history)} time steps</li>
            <li>Peak predator population: {max(s.get('predators', 0) for s in stats_history)}</li>
            <li>Peak prey population: {max(s.get('prey', 0) for s in stats_history)}</li>
            <li>Peak plant population: {max(s.get('plants', 0) for s in stats_history)}</li>
        </ul>
    </body>
    </html>
    """
    
    # Save to file
    try:
        with open(filename, 'w') as f:
            f.write(html_content)
        logger.info(f"Report saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
    
    return html_content
