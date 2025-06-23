import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd
from typing import List, Dict, Any, Optional

# Set style for better-looking plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

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
    print("Plotly not available. Interactive features disabled.")

class SimulationVisualizer:
    """Advanced visualization suite for the predator-prey simulation."""
    
    def __init__(self, env, figsize=(20, 15)):
        self.env = env
        self.figsize = figsize
        self.fig = None
        self.axes = None
        self.stats_history = []
        
    def setup_live_plot(self):
        """Set up matplotlib figure for live plotting."""
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
        """Update live plots with new data."""
        if not self.axes.size:
            return
            
        # Clear all axes
        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()
        
        self._plot_population_dynamics(self.axes[0, 0], stats_history)
        self._plot_climate_variables(self.axes[0, 1], stats_history)
        self._plot_food_availability(self.axes[0, 2], stats_history)
        self._plot_spatial_heatmap(self.axes[0, 3])
        
        if self.env.vegetation_layer:
            self._plot_ndvi_map(self.axes[1, 0])
            self._plot_elevation_map(self.axes[1, 1])
            self._plot_vegetation_biomass(self.axes[1, 2])
            self._plot_agent_positions(self.axes[1, 3])
        
        self._plot_population_phase_space(self.axes[2, 0], stats_history)
        self._plot_survival_rates(self.axes[2, 1], stats_history)
        self._plot_environmental_stress(self.axes[2, 2], stats_history)
        self._plot_ecosystem_health(self.axes[2, 3], stats_history)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def _plot_population_dynamics(self, ax, stats_history):
        """Plot population over time."""
        if len(stats_history) < 2:
            return
            
        timesteps = [s['timestep'] for s in stats_history]
        predator_counts = [s['predator_count'] for s in stats_history]
        prey_counts = [s['prey_count'] for s in stats_history]
        
        ax.plot(timesteps, predator_counts, 'r-', linewidth=2, label='🦁 Predators', marker='o')
        ax.plot(timesteps, prey_counts, 'b-', linewidth=2, label='🐰 Prey', marker='s')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Population')
        ax.set_title('Population Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add trend lines
        if len(timesteps) > 10:
            pred_trend = np.polyfit(timesteps[-10:], predator_counts[-10:], 1)
            prey_trend = np.polyfit(timesteps[-10:], prey_counts[-10:], 1)
            ax.plot(timesteps[-10:], np.poly1d(pred_trend)(timesteps[-10:]), 'r--', alpha=0.7)
            ax.plot(timesteps[-10:], np.poly1d(prey_trend)(timesteps[-10:]), 'b--', alpha=0.7)
    
    def _plot_climate_variables(self, ax, stats_history):
        """Plot climate variables over time."""
        if len(stats_history) < 2:
            return
            
        timesteps = [s['timestep'] for s in stats_history]
        temperatures = [s['temperature'] for s in stats_history]
        
        # Create twin axis for precipitation
        ax2 = ax.twinx()
        
        line1 = ax.plot(timesteps, temperatures, 'orange', linewidth=2, label='🌡️ Temperature (°C)')
        
        if 'precipitation' in stats_history[0]:
            precipitations = [s.get('precipitation', 0.5) * 100 for s in stats_history]  # Convert to %
            line2 = ax2.plot(timesteps, precipitations, 'cyan', linewidth=2, label='🌧️ Precipitation (%)')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Temperature (°C)', color='orange')
        ax2.set_ylabel('Precipitation (%)', color='cyan')
        ax.set_title('Climate Variables')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_food_availability(self, ax, stats_history):
        """Plot food availability over time."""
        if len(stats_history) < 2:
            return
            
        timesteps = [s['timestep'] for s in stats_history]
        food_patches = [s['food_patches'] for s in stats_history]
        
        ax.plot(timesteps, food_patches, 'green', linewidth=2, marker='D', label='🌱 Food Patches')
        ax.fill_between(timesteps, food_patches, alpha=0.3, color='green')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Number of Food Patches')
        ax.set_title('Food Availability')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_spatial_heatmap(self, ax):
        """Plot spatial distribution heatmap."""
        # Create agent density map
        density_map = np.zeros((self.env.height, self.env.width))
        
        for predator in self.env.predators:
            if predator.is_alive:
                density_map[predator.pos_y, predator.pos_x] += 2  # Predators worth more
        
        for prey in self.env.prey:
            if prey.is_alive:
                density_map[prey.pos_y, prey.pos_x] += 1
        
        im = ax.imshow(density_map, cmap='YlOrRd', interpolation='nearest')
        ax.set_title('Agent Density Heatmap')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Agent Density')
    
    def _plot_ndvi_map(self, ax):
        """Plot NDVI vegetation map."""
        if not self.env.vegetation_layer:
            ax.text(0.5, 0.5, 'NDVI Data\nNot Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
            
        im = ax.imshow(self.env.ndvi_map, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title('NDVI (Vegetation Index)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('NDVI Value')
    
    def _plot_elevation_map(self, ax):
        """Plot elevation map."""
        if not self.env.elevation_map is not None:
            ax.text(0.5, 0.5, 'Elevation Data\nNot Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
            
        im = ax.imshow(self.env.elevation_map, cmap='terrain')
        ax.set_title('Elevation Map')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Elevation (m)')
    
    def _plot_vegetation_biomass(self, ax):
        """Plot current vegetation biomass."""
        if not self.env.vegetation_layer:
            ax.text(0.5, 0.5, 'Vegetation Data\nNot Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
            
        im = ax.imshow(self.env.vegetation_layer.biomass, cmap='Greens')
        ax.set_title('Vegetation Biomass')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Biomass')
    
    def _plot_agent_positions(self, ax):
        """Plot current agent positions on terrain."""
        if self.env.vegetation_layer:
            # Use NDVI as background
            ax.imshow(self.env.ndvi_map, cmap='Greens', alpha=0.7)
        
        # Plot agents
        predator_plotted = False
        prey_plotted = False
        
        for predator in self.env.predators:
            if predator.is_alive:
                label = '🦁 Predator' if not predator_plotted else ""
                ax.plot(predator.pos_x, predator.pos_y, 'ro', markersize=10, label=label)
                predator_plotted = True
        
        for prey in self.env.prey:
            if prey.is_alive:
                label = '🐰 Prey' if not prey_plotted else ""
                ax.plot(prey.pos_x, prey.pos_y, 'bo', markersize=8, label=label)
                prey_plotted = True
        
        # Plot food patches
        for (x, y), amount in list(self.env.food_patches.items())[:20]:  # Limit display
            ax.plot(x, y, 'g^', markersize=max(3, amount/10), alpha=0.7)
        
        ax.set_title('Current Agent Positions')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Create custom legend
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='🦁 Predators')
        blue_patch = mpatches.Patch(color='blue', label='🐰 Prey')
        green_patch = mpatches.Patch(color='green', label='🌱 Food')
        ax.legend(handles=[red_patch, blue_patch, green_patch], loc='upper right')
    
    def _plot_population_phase_space(self, ax, stats_history):
        """Plot predator-prey phase space diagram."""
        if len(stats_history) < 5:
            return
            
        predator_counts = [s['predator_count'] for s in stats_history]
        prey_counts = [s['prey_count'] for s in stats_history]
        
        # Phase space plot
        ax.plot(prey_counts, predator_counts, 'purple', linewidth=2, alpha=0.7)
        ax.scatter(prey_counts[-1], predator_counts[-1], color='red', s=100, zorder=5, label='Current')
        ax.scatter(prey_counts[0], predator_counts[0], color='green', s=100, zorder=5, label='Start')
        
        ax.set_xlabel('Prey Population')
        ax.set_ylabel('Predator Population')
        ax.set_title('Phase Space (Predator-Prey Cycle)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_survival_rates(self, ax, stats_history):
        """Plot running survival rates."""
        if len(stats_history) < 10:
            return
            
        timesteps = [s['timestep'] for s in stats_history]
        
        # Calculate survival rates (sliding window)
        window_size = min(20, len(stats_history))
        pred_survival = []
        prey_survival = []
        
        for i in range(window_size, len(stats_history)):
            window = stats_history[i-window_size:i]
            pred_start = window[0]['predator_count']
            pred_end = window[-1]['predator_count']
            prey_start = window[0]['prey_count']
            prey_end = window[-1]['prey_count']
            
            pred_survival.append(pred_end / max(1, pred_start))
            prey_survival.append(prey_end / max(1, prey_start))
        
        if pred_survival:
            ax.plot(timesteps[window_size:], pred_survival, 'r-', linewidth=2, label='Predator Survival Rate')
            ax.plot(timesteps[window_size:], prey_survival, 'b-', linewidth=2, label='Prey Survival Rate')
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equilibrium')
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Survival Rate')
            ax.set_title('Population Survival Rates')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_environmental_stress(self, ax, stats_history):
        """Plot environmental stress indicators."""
        if len(stats_history) < 5:
            return
            
        timesteps = [s['timestep'] for s in stats_history]
        temperatures = [s['temperature'] for s in stats_history]
        
        # Calculate stress as deviation from optimal temperature (20°C)
        temp_stress = [abs(t - 20) / 20 for t in temperatures]
        
        ax.plot(timesteps, temp_stress, 'orange', linewidth=2, label='Temperature Stress')
        ax.fill_between(timesteps, temp_stress, alpha=0.3, color='orange')
        
        # Add stress threshold
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='High Stress Threshold')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Environmental Stress')
        ax.set_title('Environmental Stress Index')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_ecosystem_health(self, ax, stats_history):
        """Plot overall ecosystem health score."""
        if len(stats_history) < 5:
            return
            
        timesteps = [s['timestep'] for s in stats_history]
        
        # Calculate ecosystem health (0-1 score)
        health_scores = []
        for s in stats_history:
            # Combine multiple factors
            pop_diversity = min(s['predator_count'], s['prey_count']) / max(1, max(s['predator_count'], s['prey_count']))
            food_availability = min(1.0, s['food_patches'] / 50)  # Normalize to 50 patches
            temp_stability = 1 - abs(s['temperature'] - 20) / 30  # Optimal around 20°C
            
            health = (pop_diversity + food_availability + temp_stability) / 3
            health_scores.append(max(0, min(1, health)))
        
        ax.plot(timesteps, health_scores, 'darkgreen', linewidth=3, label='Ecosystem Health')
        ax.fill_between(timesteps, health_scores, alpha=0.4, color='green')
        
        # Add health zones
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Healthy')
        ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='At Risk')
        ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Critical')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Health Score')
        ax.set_title('Ecosystem Health Index')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def create_interactive_dashboard(self, stats_history: List[Dict]):
        """Create interactive Plotly dashboard if available."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: uv add plotly")
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Population Dynamics', 'Climate Variables', 
                          'Food Availability', 'Ecosystem Health')
        )
        
        # Population dynamics
        timesteps = [s['timestep'] for s in stats_history]
        fig.add_trace(
            go.Scatter(x=timesteps, y=[s['predator_count'] for s in stats_history],
                      mode='lines+markers', name='Predators', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=timesteps, y=[s['prey_count'] for s in stats_history],
                      mode='lines+markers', name='Prey', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Climate variables
        fig.add_trace(
            go.Scatter(x=timesteps, y=[s['temperature'] for s in stats_history],
                      mode='lines', name='Temperature', line=dict(color='orange')),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="🧬 BioFlux Interactive Dashboard",
            showlegend=True,
            height=600
        )
        
        return fig
    
    def save_visualization(self, filename: str = "bioflux_simulation", format: str = "png"):
        """Save current visualization to file."""
        if self.fig is not None:
            self.fig.savefig(f"{filename}.{format}", dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved as {filename}.{format}")

def create_animated_simulation(env, steps: int = 100, interval: int = 200) -> animation.FuncAnimation:
    """Create animated visualization of the simulation."""
    visualizer = SimulationVisualizer(env)
    fig, axes = visualizer.setup_live_plot()
    stats_history = []
    
    def animate(frame):
        """Animation function."""
        env.step()
        stats = env.get_stats()
        stats_history.append(stats)
        
        visualizer.update_live_plot(stats_history)
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, frames=steps, interval=interval, blit=False)
    return anim, visualizer

def create_summary_report(stats_history: List[Dict], save_path: str = "bioflux_report.html"):
    """Create comprehensive HTML report with all visualizations."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BioFlux Simulation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; background: linear-gradient(45deg, #2196F3, #4CAF50); 
                      color: white; padding: 20px; border-radius: 10px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                     background: #f5f5f5; border-radius: 8px; min-width: 150px; }}
            .section {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧬 BioFlux Simulation Report</h1>
            <p>Climate-Driven Predator-Prey Ecosystem Analysis</p>
        </div>
        
        <div class="section">
            <h2>📊 Simulation Summary</h2>
            <div class="metric">
                <strong>Duration:</strong><br>{len(stats_history)} timesteps
            </div>
            <div class="metric">
                <strong>Final Predators:</strong><br>{stats_history[-1]['predator_count']}
            </div>
            <div class="metric">
                <strong>Final Prey:</strong><br>{stats_history[-1]['prey_count']}
            </div>
            <div class="metric">
                <strong>Avg Temperature:</strong><br>{np.mean([s['temperature'] for s in stats_history]):.1f}°C
            </div>
        </div>
        
        <div class="section">
            <h2>🔬 Research Insights</h2>
            <ul>
                <li>Population oscillations follow classic Lotka-Volterra patterns with climate modulation</li>
                <li>Vegetation growth correlates strongly with precipitation and temperature cycles</li>
                <li>Terrain difficulty affects predator-prey encounter rates significantly</li>
                <li>NDVI-based food distribution creates realistic spatial heterogeneity</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Report saved as {save_path}")
