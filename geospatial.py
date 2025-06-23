import numpy as np
import requests
import rasterio
from rasterio.transform import from_bounds
from scipy import ndimage
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os
import tempfile

class GeospatialDataLoader:
    """Load NDVI and elevation data from open sources."""
    
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def download_landsat_ndvi(self, bbox: Tuple[float, float, float, float], 
                             width: int = 50, height: int = 50) -> np.ndarray:
        """
        Download NDVI data from NASA Landsat (mock implementation - replace with real API).
        bbox: (min_lon, min_lat, max_lon, max_lat)
        Returns normalized NDVI array [0-1]
        """
        # Mock NDVI data with realistic patterns
        # In production, use NASA EarthData API or Google Earth Engine
        np.random.seed(42)  # Reproducible for testing
        
        # Create realistic NDVI pattern
        ndvi = np.random.normal(0.6, 0.2, (height, width))
        
        # Add spatial correlation (vegetation clusters)
        ndvi = ndimage.gaussian_filter(ndvi, sigma=2)
        
        # Add water bodies (low NDVI)
        water_mask = np.random.random((height, width)) < 0.05
        ndvi[water_mask] = np.random.uniform(-0.1, 0.1, np.sum(water_mask))
        
        # Add urban areas (medium NDVI)
        urban_mask = np.random.random((height, width)) < 0.1
        ndvi[urban_mask] = np.random.uniform(0.1, 0.3, np.sum(urban_mask))
        
        # Normalize to [0, 1] range
        ndvi = np.clip((ndvi + 1) / 2, 0, 1)
        
        return ndvi
    
    def download_srtm_elevation(self, bbox: Tuple[float, float, float, float],
                               width: int = 50, height: int = 50) -> np.ndarray:
        """
        Download elevation data from SRTM (mock implementation).
        Returns elevation in meters.
        """
        # Mock elevation data with realistic topography
        np.random.seed(123)
        
        # Create base elevation
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        X, Y = np.meshgrid(x, y)
        
        # Add multiple terrain features
        elevation = (
            200 * np.sin(X * 0.5) * np.cos(Y * 0.3) +  # Rolling hills
            100 * np.sin(X * 1.2) +                     # Ridge
            50 * np.random.normal(0, 1, (height, width)) # Noise
        )
        
        # Smooth and add base elevation
        elevation = ndimage.gaussian_filter(elevation, sigma=1)
        elevation += 500  # Base elevation 500m
        elevation = np.maximum(elevation, 0)  # No negative elevation
        
        return elevation
    
    def get_real_ndvi_from_nasa(self, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        Example of how to fetch real NDVI from NASA (requires API key).
        This is a template - implement with actual NASA EarthData API.
        """
        # Example NASA MODIS NDVI endpoint (simplified)
        # In practice, use NASA EarthData, Google Earth Engine, or Sentinel Hub
        try:
            # Mock implementation - replace with real API call
            print("Real NASA API would be called here with bbox:", bbox)
            return None
        except Exception as e:
            print(f"Failed to fetch real NDVI data: {e}")
            return None

class LogisticGrowthModel:
    """Implements logistic growth model for vegetation."""
    
    def __init__(self, carrying_capacity: float = 100.0, growth_rate: float = 0.1):
        self.K = carrying_capacity  # Carrying capacity
        self.r = growth_rate       # Intrinsic growth rate
    
    def growth(self, population: float, dt: float = 1.0, 
               environmental_factor: float = 1.0) -> float:
        """
        Logistic growth equation: dP/dt = r * P * (1 - P/K) * environmental_factor
        """
        if population <= 0:
            return 0
        
        # Logistic growth with environmental modulation
        growth_rate = self.r * environmental_factor
        dp_dt = growth_rate * population * (1 - population / self.K)
        
        # Update population
        new_population = population + dp_dt * dt
        return max(0, min(self.K, new_population))
    
    def environmental_factor(self, ndvi: float, elevation: float, 
                           temperature: float, precipitation: float) -> float:
        """
        Calculate environmental factor based on multiple variables.
        Returns value between 0 and 2 (multiplicative factor).
        """
        # NDVI factor (higher NDVI = better growing conditions)
        ndvi_factor = np.clip(ndvi * 2, 0, 1)
        
        # Elevation factor (optimal range 200-800m)
        elev_factor = 1.0
        if elevation < 200:
            elev_factor = elevation / 200
        elif elevation > 800:
            elev_factor = max(0.2, 1 - (elevation - 800) / 1000)
        
        # Temperature factor (optimal 15-25°C)
        temp_factor = 1.0
        if temperature < 15:
            temp_factor = max(0.1, temperature / 15)
        elif temperature > 25:
            temp_factor = max(0.1, 1 - (temperature - 25) / 20)
        
        # Precipitation factor (0.3-0.8 optimal)
        precip_factor = np.clip(precipitation / 0.6, 0.2, 1.5)
        
        # Combine all factors
        env_factor = ndvi_factor * elev_factor * temp_factor * precip_factor
        return np.clip(env_factor, 0.1, 2.0)

class VegetationLayer:
    """Manages vegetation growth using logistic model and real geospatial data."""
    
    def __init__(self, width: int, height: int, bbox: Optional[Tuple[float, float, float, float]] = None):
        self.width = width
        self.height = height
        self.bbox = bbox or (-122.5, 37.7, -122.3, 37.9)  # Default SF Bay Area
        
        # Load geospatial data
        self.data_loader = GeospatialDataLoader()
        self.ndvi = self.data_loader.download_landsat_ndvi(self.bbox, width, height)
        self.elevation = self.data_loader.download_srtm_elevation(self.bbox, width, height)
        
        # Initialize vegetation biomass
        self.biomass = self.ndvi * 50  # Initial biomass based on NDVI
        
        # Growth models for different vegetation types
        self.grass_model = LogisticGrowthModel(carrying_capacity=30, growth_rate=0.2)
        self.shrub_model = LogisticGrowthModel(carrying_capacity=60, growth_rate=0.1)
        self.tree_model = LogisticGrowthModel(carrying_capacity=100, growth_rate=0.05)
        
        # Vegetation type map (0=grass, 1=shrub, 2=tree)
        self.veg_type = self._classify_vegetation()
    
    def _classify_vegetation(self) -> np.ndarray:
        """Classify vegetation types based on NDVI and elevation."""
        veg_type = np.zeros((self.height, self.width), dtype=int)
        
        # Trees in high NDVI, moderate elevation areas
        tree_mask = (self.ndvi > 0.7) & (self.elevation > 300) & (self.elevation < 800)
        veg_type[tree_mask] = 2
        
        # Shrubs in medium NDVI areas
        shrub_mask = (self.ndvi > 0.4) & (self.ndvi <= 0.7) & ~tree_mask
        veg_type[shrub_mask] = 1
        
        # Grass elsewhere (default)
        # veg_type remains 0 for grass
        
        return veg_type
    
    def update(self, temperature: float, precipitation: float, dt: float = 1.0):
        """Update vegetation biomass using logistic growth model."""
        for i in range(self.height):
            for j in range(self.width):
                # Get environmental conditions for this cell
                ndvi = self.ndvi[i, j]
                elevation = self.elevation[i, j]
                veg_type = self.veg_type[i, j]
                current_biomass = self.biomass[i, j]
                
                # Select appropriate growth model
                if veg_type == 0:  # Grass
                    model = self.grass_model
                elif veg_type == 1:  # Shrub
                    model = self.shrub_model
                else:  # Tree
                    model = self.tree_model
                
                # Calculate environmental factor
                env_factor = model.environmental_factor(ndvi, elevation, temperature, precipitation)
                
                # Update biomass
                self.biomass[i, j] = model.growth(current_biomass, dt, env_factor)
                
                # Update NDVI based on new biomass (simplified)
                self.ndvi[i, j] = np.clip(self.biomass[i, j] / 100, 0, 1)
    
    def get_food_availability(self, x: int, y: int) -> float:
        """Get food availability at grid position (for prey)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.biomass[y, x] * 0.1  # Food is 10% of biomass
        return 0.0
    
    def consume_vegetation(self, x: int, y: int, amount: float):
        """Consume vegetation at position (herbivore grazing)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.biomass[y, x] = max(0, self.biomass[y, x] - amount)
            # Update NDVI
            self.ndvi[y, x] = np.clip(self.biomass[y, x] / 100, 0, 1)
    
    def visualize(self, save_path: Optional[str] = None):
        """Visualize vegetation layers."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # NDVI
        im1 = axes[0, 0].imshow(self.ndvi, cmap='RdYlGn', vmin=0, vmax=1)
        axes[0, 0].set_title('NDVI')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Elevation
        im2 = axes[0, 1].imshow(self.elevation, cmap='terrain')
        axes[0, 1].set_title('Elevation (m)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Biomass
        im3 = axes[1, 0].imshow(self.biomass, cmap='Greens')
        axes[1, 0].set_title('Vegetation Biomass')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Vegetation types
        im4 = axes[1, 1].imshow(self.veg_type, cmap='viridis')
        axes[1, 1].set_title('Vegetation Types\n(0=Grass, 1=Shrub, 2=Tree)')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
