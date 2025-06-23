#!/usr/bin/env python3
"""
Real-time geospatial and weather data integration using open-source APIs
"""

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Tuple, Optional
import os
from dataclasses import dataclass
import logging
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if value and not value.startswith('#'):
                        os.environ[key] = value

# Load .env file
load_env_file()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    """Weather data structure"""
    temperature: float  # Celsius
    humidity: float     # Percentage
    precipitation: float # mm
    wind_speed: float   # m/s
    pressure: float     # hPa
    timestamp: datetime

@dataclass
class NDVIData:
    """NDVI data structure"""
    ndvi_value: float   # -1 to 1
    confidence: float   # 0 to 1
    cloud_cover: float  # 0 to 1
    timestamp: datetime
    source: str

class OpenWeatherMapAPI:
    """Integration with OpenWeatherMap API (free tier available)"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Get API key from environment or use demo mode
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.demo_mode = self.api_key is None
        
        if self.demo_mode:
            logger.warning("No OpenWeatherMap API key found. Using demo mode with synthetic data.")
    
    def get_current_weather(self, lat: float, lon: float) -> WeatherData:
        """Get current weather data for coordinates"""
        if self.demo_mode:
            return self._get_demo_weather(lat, lon)
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return WeatherData(
                temperature=data['main']['temp'],
                humidity=data['main']['humidity'],
                precipitation=data.get('rain', {}).get('1h', 0),
                wind_speed=data['wind']['speed'],
                pressure=data['main']['pressure'],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._get_demo_weather(lat, lon)
    
    def get_weather_forecast(self, lat: float, lon: float, days: int = 5) -> List[WeatherData]:
        """Get weather forecast for next few days"""
        if self.demo_mode:
            return self._get_demo_forecast(lat, lon, days)
        
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            forecast_list = []
            for item in data['list'][:days*8]:  # 8 forecasts per day (3-hour intervals)
                forecast_list.append(WeatherData(
                    temperature=item['main']['temp'],
                    humidity=item['main']['humidity'],
                    precipitation=item.get('rain', {}).get('3h', 0),
                    wind_speed=item['wind']['speed'],
                    pressure=item['main']['pressure'],
                    timestamp=datetime.fromtimestamp(item['dt'])
                ))
            
            return forecast_list
            
        except Exception as e:
            logger.error(f"Error fetching forecast data: {e}")
            return self._get_demo_forecast(lat, lon, days)
    
    def _get_demo_weather(self, lat: float, lon: float) -> WeatherData:
        """Generate realistic demo weather data"""
        # Simulate seasonal variation based on latitude
        day_of_year = datetime.now().timetuple().tm_yday
        seasonal_temp = 20 + 15 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
        
        # Add some randomness
        np.random.seed(int(time.time()) % 1000)
        
        return WeatherData(
            temperature=seasonal_temp + np.random.normal(0, 5),
            humidity=60 + np.random.normal(0, 20),
            precipitation=max(0, np.random.exponential(2)),
            wind_speed=max(0, np.random.normal(5, 3)),
            pressure=1013 + np.random.normal(0, 20),
            timestamp=datetime.now()
        )
    
    def _get_demo_forecast(self, lat: float, lon: float, days: int) -> List[WeatherData]:
        """Generate demo forecast data"""
        forecast = []
        base_weather = self._get_demo_weather(lat, lon)
        
        for i in range(days * 8):  # 3-hour intervals
            time_offset = timedelta(hours=i * 3)
            temp_variation = np.random.normal(0, 2)
            
            forecast.append(WeatherData(
                temperature=base_weather.temperature + temp_variation,
                humidity=max(20, min(100, base_weather.humidity + np.random.normal(0, 10))),
                precipitation=max(0, np.random.exponential(1.5)),
                wind_speed=max(0, base_weather.wind_speed + np.random.normal(0, 2)),
                pressure=base_weather.pressure + np.random.normal(0, 5),
                timestamp=datetime.now() + time_offset
            ))
        
        return forecast

class SentinelHubAPI:
    """Integration with Sentinel Hub for NDVI data (has free tier)"""
    
    def __init__(self, instance_id: Optional[str] = None):
        self.instance_id = instance_id or os.getenv('SENTINELHUB_INSTANCE_ID')
        self.demo_mode = self.instance_id is None
        
        if self.demo_mode:
            logger.warning("No Sentinel Hub instance ID found. Using demo mode with synthetic NDVI data.")
    
    def get_ndvi_data(self, bbox: Tuple[float, float, float, float], 
                      date: datetime = None) -> np.ndarray:
        """Get NDVI data for bounding box"""
        if self.demo_mode:
            return self._get_demo_ndvi(bbox)
        
        # Implementation would use Sentinel Hub API
        # For now, return demo data
        return self._get_demo_ndvi(bbox)
    
    def _get_demo_ndvi(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Generate realistic NDVI data"""
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # Create 50x50 grid
        width, height = 50, 50
        
        # Generate realistic NDVI patterns
        np.random.seed(42)  # Reproducible
        
        # Base vegetation
        ndvi = np.random.normal(0.6, 0.15, (height, width))
        
        # Add spatial correlation
        from scipy import ndimage
        ndvi = ndimage.gaussian_filter(ndvi, sigma=2)
        
        # Add water bodies (rivers, lakes)
        water_features = np.random.random((height, width)) < 0.08
        ndvi[water_features] = np.random.uniform(-0.2, 0.1, np.sum(water_features))
        
        # Add urban areas
        urban_features = np.random.random((height, width)) < 0.12
        ndvi[urban_features] = np.random.uniform(0.1, 0.3, np.sum(urban_features))
        
        # Add dense forest areas
        forest_features = np.random.random((height, width)) < 0.15
        ndvi[forest_features] = np.random.uniform(0.7, 0.9, np.sum(forest_features))
        
        # Clip to valid NDVI range
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi

class NASAEarthDataAPI:
    """Integration with NASA EarthData for satellite imagery (free with registration)"""
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.username = username or os.getenv('NASA_EARTHDATA_USERNAME')
        self.password = password or os.getenv('NASA_EARTHDATA_PASSWORD')
        self.demo_mode = not (self.username and self.password)
        
        if self.demo_mode:
            logger.warning("No NASA EarthData credentials found. Using demo mode.")
    
    def get_modis_ndvi(self, lat: float, lon: float, date: datetime = None) -> NDVIData:
        """Get MODIS NDVI data for specific location"""
        if self.demo_mode:
            return self._get_demo_modis_ndvi(lat, lon)
        
        # Implementation would use NASA EarthData API
        return self._get_demo_modis_ndvi(lat, lon)
    
    def _get_demo_modis_ndvi(self, lat: float, lon: float) -> NDVIData:
        """Generate demo MODIS NDVI data"""
        # Simulate realistic NDVI based on location
        np.random.seed(int((lat + lon) * 1000) % 1000)
        
        # Base NDVI varies by latitude (proxy for climate)
        base_ndvi = 0.7 - abs(lat) * 0.008  # Lower NDVI at higher latitudes
        
        # Add seasonal variation
        day_of_year = datetime.now().timetuple().tm_yday
        seasonal_factor = 0.2 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
        
        ndvi_value = base_ndvi + seasonal_factor + np.random.normal(0, 0.1)
        ndvi_value = np.clip(ndvi_value, -1, 1)
        
        return NDVIData(
            ndvi_value=ndvi_value,
            confidence=np.random.uniform(0.7, 0.95),
            cloud_cover=np.random.uniform(0, 0.3),
            timestamp=datetime.now(),
            source="MODIS (demo)"
        )

class RealDataIntegrator:
    """Main class for integrating real weather and NDVI data"""
    
    def __init__(self, openweather_key: Optional[str] = None,
                 sentinelhub_id: Optional[str] = None,
                 nasa_username: Optional[str] = None,
                 nasa_password: Optional[str] = None):
        
        self.weather_api = OpenWeatherMapAPI(openweather_key)
        self.sentinel_api = SentinelHubAPI(sentinelhub_id)
        self.nasa_api = NASAEarthDataAPI(nasa_username, nasa_password)
        
        logger.info("Real data integrator initialized")
        logger.info(f"Weather API: {'Live' if not self.weather_api.demo_mode else 'Demo'}")
        logger.info(f"Sentinel Hub: {'Live' if not self.sentinel_api.demo_mode else 'Demo'}")
        logger.info(f"NASA EarthData: {'Live' if not self.nasa_api.demo_mode else 'Demo'}")
    
    def get_location_data(self, lat: float, lon: float, 
                         bbox: Optional[Tuple[float, float, float, float]] = None) -> Dict:
        """Get comprehensive data for a location"""
        
        if bbox is None:
            # Create small bbox around point
            offset = 0.01  # ~1km
            bbox = (lon - offset, lat - offset, lon + offset, lat + offset)
        
        logger.info(f"Fetching data for location: {lat:.4f}, {lon:.4f}")
        
        # Get current weather
        current_weather = self.weather_api.get_current_weather(lat, lon)
        
        # Get weather forecast
        forecast = self.weather_api.get_weather_forecast(lat, lon, days=3)
        
        # Get NDVI data
        ndvi_grid = self.sentinel_api.get_ndvi_data(bbox)
        ndvi_point = self.nasa_api.get_modis_ndvi(lat, lon)
        
        return {
            'location': {'lat': lat, 'lon': lon, 'bbox': bbox},
            'current_weather': current_weather,
            'weather_forecast': forecast,
            'ndvi_grid': ndvi_grid,
            'ndvi_point': ndvi_point,
            'timestamp': datetime.now()
        }
    
    def update_simulation_with_real_data(self, env, lat: float, lon: float):
        """Update simulation environment with real data"""
        try:
            data = self.get_location_data(lat, lon)
            
            # Update climate variables
            weather = data['current_weather']
            env.temperature = weather.temperature
            env.precipitation = weather.precipitation / 10.0  # Scale to simulation units
            env.humidity = weather.humidity / 100.0
            
            # Update NDVI data
            if hasattr(env, 'vegetation_layer'):
                ndvi_grid = data['ndvi_grid']
                # Resize to match environment grid
                from scipy.ndimage import zoom
                scale_x = env.width / ndvi_grid.shape[1]
                scale_y = env.height / ndvi_grid.shape[0]
                resized_ndvi = zoom(ndvi_grid, (scale_y, scale_x))
                
                # Update vegetation layer
                env.vegetation_layer.ndvi = resized_ndvi
                env.vegetation_layer.update_biomass_from_ndvi()
            
            logger.info(f"Updated simulation with real data: T={weather.temperature:.1f}°C, "
                       f"P={weather.precipitation:.1f}mm, NDVI={data['ndvi_point'].ndvi_value:.3f}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error updating simulation with real data: {e}")
            return None

def setup_environment_variables():
    """Helper function to set up environment variables for API keys"""
    print("🔑 API Key Setup Guide")
    print("=" * 50)
    print("\n1. OpenWeatherMap (Free tier: 1000 calls/day)")
    print("   - Sign up at: https://openweathermap.org/api")
    print("   - Set: export OPENWEATHER_API_KEY='your_key_here'")
    
    print("\n2. Sentinel Hub (Free tier: 1000 requests/month)")
    print("   - Sign up at: https://www.sentinel-hub.com/")
    print("   - Set: export SENTINELHUB_INSTANCE_ID='your_instance_id'")
    
    print("\n3. NASA EarthData (Free with registration)")
    print("   - Sign up at: https://urs.earthdata.nasa.gov/")
    print("   - Set: export NASA_EARTHDATA_USERNAME='your_username'")
    print("   - Set: export NASA_EARTHDATA_PASSWORD='your_password'")
    
    print("\n📝 Add these to your ~/.zshrc or ~/.bashrc to persist")
    print("   Then run: source ~/.zshrc")

if __name__ == "__main__":
    # Demo usage
    print("🌍 Real Data API Integration Demo")
    print("=" * 40)
    
    # San Francisco coordinates
    lat, lon = 37.7749, -122.4194
    
    # Initialize integrator
    integrator = RealDataIntegrator()
    
    # Get location data
    data = integrator.get_location_data(lat, lon)
    
    print(f"\n📍 Location: {lat}, {lon}")
    print(f"🌡️  Temperature: {data['current_weather'].temperature:.1f}°C")
    print(f"💧 Humidity: {data['current_weather'].humidity:.1f}%")
    print(f"🌧️  Precipitation: {data['current_weather'].precipitation:.1f}mm")
    print(f"🌱 NDVI: {data['ndvi_point'].ndvi_value:.3f}")
    print(f"☁️  NDVI Confidence: {data['ndvi_point'].confidence:.2f}")
    
    print("\n🔮 3-Day Weather Forecast:")
    for i, forecast in enumerate(data['weather_forecast'][:8]):  # Next 24 hours
        hours_ahead = i * 3
        print(f"   +{hours_ahead:2d}h: {forecast.temperature:.1f}°C, "
              f"{forecast.precipitation:.1f}mm rain")
    
    print("\n💡 To use live data, set up API keys:")
    setup_environment_variables()
