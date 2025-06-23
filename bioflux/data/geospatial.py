#!/usr/bin/env python3
"""
Geospatial data loading and processing for BioFlux.
"""

import numpy as np
import requests
from typing import Tuple, Optional, Dict, Any
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logger.warning("Rasterio not available. Some geospatial features may be limited.")

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Spatial filtering will be limited.")

class GeospatialDataLoader:
    """Load NDVI and elevation data from open sources."""
    
    def __init__(self, cache_dir: str = "./data_cache"):
        """
        Initialize the geospatial data loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"Geospatial data cache: {self.cache_dir}")
    
    def download_landsat_ndvi(self, bbox: Tuple[float, float, float, float], 
                             width: int = 50, height: int = 50) -> np.ndarray:
        """
        Download NDVI data from NASA Landsat (mock implementation - replace with real API).
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            width: Output width in pixels
            height: Output height in pixels
            
        Returns:
            Normalized NDVI array [0-1]
        """
        logger.info(f"Generating NDVI data for bbox: {bbox}")
        
        # Mock NDVI data with realistic patterns
        # In production, use NASA EarthData API or Google Earth Engine
        np.random.seed(42)  # Reproducible for testing
        
        # Create realistic NDVI pattern
        ndvi = np.random.normal(0.6, 0.2, (height, width))
        
        # Add spatial correlation (vegetation clusters)
        if SCIPY_AVAILABLE:
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
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            width: Output width in pixels
            height: Output height in pixels
            
        Returns:
            Elevation in meters
        """
        logger.info(f"Generating elevation data for bbox: {bbox}")
        
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
        if SCIPY_AVAILABLE:
            elevation = ndimage.gaussian_filter(elevation, sigma=1)
        elevation += 500  # Base elevation 500m
        elevation = np.maximum(elevation, 0)  # No negative elevation
        
        return elevation
    
    def get_real_ndvi_from_nasa(self, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        Example of how to fetch real NDVI from NASA (requires API key).
        This is a template - implement with actual NASA EarthData API.
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            NDVI array or None if failed
        """
        # Example NASA MODIS NDVI endpoint (simplified)
        # In practice, use NASA EarthData, Google Earth Engine, or Sentinel Hub
        try:
            logger.info("Real NASA API would be called here with bbox: %s", bbox)
            return None
        except Exception as e:
            logger.error("Failed to fetch real NDVI data: %s", e)
            return None

class LogisticGrowthModel:
    """Implements logistic growth model for vegetation."""
    
    def __init__(self, carrying_capacity: float = 100.0, growth_rate: float = 0.1):
        """
        Initialize the logistic growth model.
        
        Args:
            carrying_capacity: Maximum sustainable population
            growth_rate: Intrinsic growth rate
        """
        self.K = carrying_capacity  # Carrying capacity
        self.r = growth_rate       # Intrinsic growth rate
    
    def growth(self, population: float, dt: float = 1.0, 
               environmental_factor: float = 1.0) -> float:
        """
        Calculate population growth using logistic model.
        
        Args:
            population: Current population
            dt: Time step
            environmental_factor: Environmental modifier (0-1)
            
        Returns:
            Population change
        """
        if population <= 0:
            return 0
        
        # Logistic growth equation: dP/dt = rP(1 - P/K)
        growth_rate = self.r * environmental_factor
        change = growth_rate * population * (1 - population / self.K) * dt
        
        return change
    
    def step(self, population: float, dt: float = 1.0, 
             environmental_factor: float = 1.0) -> float:
        """
        Advance population by one time step.
        
        Args:
            population: Current population
            dt: Time step
            environmental_factor: Environmental modifier (0-1)
            
        Returns:
            New population
        """
        change = self.growth(population, dt, environmental_factor)
        new_population = max(0, population + change)
        return new_population

class VegetationLayer:
    """Manages vegetation data and growth dynamics."""
    
    def __init__(self, width: int, height: int, initial_density: float = 0.5):
        """
        Initialize the vegetation layer.
        
        Args:
            width: Grid width
            height: Grid height
            initial_density: Initial vegetation density [0-1]
        """
        self.width = width
        self.height = height
        self.density = np.full((height, width), initial_density, dtype=np.float32)
        self.growth_model = LogisticGrowthModel(carrying_capacity=1.0, growth_rate=0.05)
        
        # Environmental factors affecting growth
        self.temperature_optimum = 22.0  # Celsius
        self.humidity_optimum = 70.0     # Percentage
        
    def update(self, temperature_map: np.ndarray, humidity_map: np.ndarray, 
               precipitation: float = 0.0, dt: float = 1.0):
        """
        Update vegetation density based on environmental conditions.
        
        Args:
            temperature_map: Temperature array
            humidity_map: Humidity array
            precipitation: Precipitation amount
            dt: Time step
        """
        # Calculate environmental factors
        temp_factor = self._calculate_temperature_factor(temperature_map)
        humidity_factor = self._calculate_humidity_factor(humidity_map)
        precip_factor = self._calculate_precipitation_factor(precipitation)
        
        # Combined environmental factor
        env_factor = temp_factor * humidity_factor * precip_factor
        
        # Update vegetation density for each cell
        for i in range(self.height):
            for j in range(self.width):
                current_density = self.density[i, j]
                local_env_factor = env_factor[i, j] if hasattr(env_factor, 'shape') else env_factor
                
                # Apply logistic growth
                new_density = self.growth_model.step(
                    current_density, dt, local_env_factor
                )
                self.density[i, j] = np.clip(new_density, 0.0, 1.0)
    
    def _calculate_temperature_factor(self, temperature_map: np.ndarray) -> np.ndarray:
        """Calculate temperature growth factor."""
        # Optimal temperature range
        temp_diff = np.abs(temperature_map - self.temperature_optimum)
        temp_factor = np.exp(-temp_diff / 10.0)  # Gaussian-like response
        return np.clip(temp_factor, 0.1, 1.0)
    
    def _calculate_humidity_factor(self, humidity_map: np.ndarray) -> np.ndarray:
        """Calculate humidity growth factor."""
        # Optimal humidity range
        humidity_diff = np.abs(humidity_map - self.humidity_optimum)
        humidity_factor = np.exp(-humidity_diff / 20.0)
        return np.clip(humidity_factor, 0.1, 1.0)
    
    def _calculate_precipitation_factor(self, precipitation: float) -> float:
        """Calculate precipitation growth factor."""
        # More precipitation generally helps vegetation (up to a point)
        if precipitation < 0:
            return 0.5  # Drought conditions
        elif precipitation < 10:
            return 0.8 + precipitation * 0.02  # Light rain helps
        elif precipitation < 50:
            return 1.0  # Optimal precipitation
        else:
            return max(0.5, 1.0 - (precipitation - 50) * 0.01)  # Too much rain
    
    def get_density_at(self, x: int, y: int) -> float:
        """Get vegetation density at a specific location."""
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        return self.density[y, x]
    
    def consume_vegetation(self, x: int, y: int, amount: float) -> float:
        """
        Consume vegetation at a location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            amount: Amount to consume
            
        Returns:
            Amount actually consumed
        """
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        
        available = self.density[y, x]
        consumed = min(amount, available)
        self.density[y, x] = max(0, available - consumed)
        
        return consumed
    
    def get_stats(self) -> Dict[str, float]:
        """Get vegetation statistics."""
        return {
            'total_vegetation': float(np.sum(self.density)),
            'average_density': float(np.mean(self.density)),
            'max_density': float(np.max(self.density)),
            'min_density': float(np.min(self.density)),
            'coverage_percent': float((self.density > 0.1).sum() / self.density.size * 100)
        }

class TerrainAnalyzer:
    """Analyzes terrain data for ecological modeling."""
    
    @staticmethod
    def calculate_slope(elevation_map: np.ndarray, pixel_size: float = 1.0) -> np.ndarray:
        """
        Calculate slope from elevation data.
        
        Args:
            elevation_map: 2D elevation array
            pixel_size: Size of each pixel in meters
            
        Returns:
            Slope array in degrees
        """
        # Calculate gradients
        dy, dx = np.gradient(elevation_map)
        
        # Calculate slope in radians then convert to degrees
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2) / pixel_size)
        slope_deg = np.degrees(slope_rad)
        
        return slope_deg
    
    @staticmethod
    def calculate_aspect(elevation_map: np.ndarray) -> np.ndarray:
        """
        Calculate aspect (direction of slope) from elevation data.
        
        Args:
            elevation_map: 2D elevation array
            
        Returns:
            Aspect array in degrees (0-360)
        """
        dy, dx = np.gradient(elevation_map)
        
        # Calculate aspect in radians then convert to degrees
        aspect_rad = np.arctan2(-dy, dx)
        aspect_deg = np.degrees(aspect_rad)
        
        # Convert to 0-360 range
        aspect_deg = (aspect_deg + 360) % 360
        
        return aspect_deg
    
    @staticmethod
    def find_water_bodies(elevation_map: np.ndarray, 
                         threshold_percentile: float = 10.0) -> np.ndarray:
        """
        Identify potential water bodies based on elevation.
        
        Args:
            elevation_map: 2D elevation array
            threshold_percentile: Percentile below which areas are considered water
            
        Returns:
            Binary mask where True indicates water
        """
        threshold = np.percentile(elevation_map, threshold_percentile)
        return elevation_map <= threshold
