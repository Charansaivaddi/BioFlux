#!/usr/bin/env python3
"""
Real-time weather and climate data integration using open-source APIs.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import os
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("Requests library not available. Weather data integration disabled.")

@dataclass
class WeatherData:
    """Weather data structure."""
    temperature: float  # Celsius
    humidity: float     # Percentage
    precipitation: float # mm
    wind_speed: float   # m/s
    pressure: float     # hPa
    timestamp: datetime
    location: Optional[str] = None

@dataclass
class SatelliteData:
    """Satellite data structure."""
    ndvi: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    moisture: Optional[np.ndarray] = None
    timestamp: Optional[datetime] = None
    bbox: Optional[Tuple[float, float, float, float]] = None

class WeatherAPI:
    """Interface for weather data APIs."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize weather API client.
        
        Args:
            api_key: API key for the weather service
        """
        self.api_key = api_key
        self.session = None
        if REQUESTS_AVAILABLE:
            self.session = requests.Session()
    
    def get_current_weather(self, lat: float, lon: float) -> Optional[WeatherData]:
        """Get current weather data for a location."""
        raise NotImplementedError
    
    def get_forecast(self, lat: float, lon: float, days: int = 5) -> List[WeatherData]:
        """Get weather forecast for a location."""
        raise NotImplementedError

class OpenWeatherMapAPI(WeatherAPI):
    """OpenWeatherMap API integration."""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    def __init__(self, api_key: str):
        """
        Initialize OpenWeatherMap API client.
        
        Args:
            api_key: OpenWeatherMap API key
        """
        super().__init__(api_key)
        self.base_url = self.BASE_URL
    
    def get_current_weather(self, lat: float, lon: float) -> Optional[WeatherData]:
        """
        Get current weather data from OpenWeatherMap.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            WeatherData object or None if failed
        """
        if not REQUESTS_AVAILABLE or not self.api_key:
            return None
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return WeatherData(
                temperature=data['main']['temp'],
                humidity=data['main']['humidity'],
                precipitation=data.get('rain', {}).get('1h', 0.0),
                wind_speed=data.get('wind', {}).get('speed', 0.0),
                pressure=data['main']['pressure'],
                timestamp=datetime.now(),
                location=data.get('name', 'Unknown')
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch weather data: {e}")
            return None
    
    def get_forecast(self, lat: float, lon: float, days: int = 5) -> List[WeatherData]:
        """
        Get weather forecast from OpenWeatherMap.
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of days to forecast
            
        Returns:
            List of WeatherData objects
        """
        if not REQUESTS_AVAILABLE or not self.api_key:
            return []
        
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': min(days * 8, 40)  # 8 forecasts per day, max 40
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            for item in data['list']:
                forecast = WeatherData(
                    temperature=item['main']['temp'],
                    humidity=item['main']['humidity'],
                    precipitation=item.get('rain', {}).get('3h', 0.0),
                    wind_speed=item.get('wind', {}).get('speed', 0.0),
                    pressure=item['main']['pressure'],
                    timestamp=datetime.fromtimestamp(item['dt']),
                    location=data['city']['name']
                )
                forecasts.append(forecast)
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Failed to fetch forecast data: {e}")
            return []

class SentinelHubAPI:
    """Sentinel Hub API for satellite data."""
    
    BASE_URL = "https://services.sentinel-hub.com"
    
    def __init__(self, client_id: str, client_secret: str, instance_id: str):
        """
        Initialize Sentinel Hub API client.
        
        Args:
            client_id: Sentinel Hub client ID
            client_secret: Sentinel Hub client secret
            instance_id: Sentinel Hub instance ID
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.instance_id = instance_id
        self.access_token = None
        self.token_expires = None
        self.session = None
        
        if REQUESTS_AVAILABLE:
            self.session = requests.Session()
    
    def _get_access_token(self) -> bool:
        """Get OAuth access token."""
        if not REQUESTS_AVAILABLE:
            return False
        
        try:
            url = f"{self.BASE_URL}/oauth/token"
            data = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            
            response = self.session.post(url, data=data, timeout=10)
            response.raise_for_status()
            token_data = response.json()
            
            self.access_token = token_data['access_token']
            expires_in = token_data.get('expires_in', 3600)
            self.token_expires = datetime.now() + timedelta(seconds=expires_in - 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            return False
    
    def _ensure_valid_token(self) -> bool:
        """Ensure we have a valid access token."""
        if not self.access_token or (self.token_expires and datetime.now() >= self.token_expires):
            return self._get_access_token()
        return True
    
    def get_ndvi_data(self, bbox: Tuple[float, float, float, float], 
                      width: int = 512, height: int = 512,
                      date_from: Optional[str] = None,
                      date_to: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get NDVI data from Sentinel-2.
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            width: Image width in pixels
            height: Image height in pixels
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            
        Returns:
            NDVI array or None if failed
        """
        if not REQUESTS_AVAILABLE or not self._ensure_valid_token():
            return None
        
        try:
            # Default date range (last month)
            if not date_to:
                date_to = datetime.now().strftime('%Y-%m-%d')
            if not date_from:
                date_from = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Sentinel Hub process API payload
            payload = {
                "input": {
                    "bounds": {
                        "bbox": list(bbox),
                        "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                    },
                    "data": [{
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": f"{date_from}T00:00:00Z",
                                "to": f"{date_to}T23:59:59Z"
                            }
                        },
                        "processing": {
                            "atmosphericCorrection": "NONE"
                        }
                    }]
                },
                "output": {
                    "width": width,
                    "height": height,
                    "responses": [{
                        "identifier": "default",
                        "format": {"type": "image/png"}
                    }]
                },
                "evalscript": """
                //VERSION=3
                function setup() {
                    return {
                        input: [{
                            bands: ["B04", "B08"]
                        }],
                        output: {
                            bands: 1,
                            sampleType: "FLOAT32"
                        }
                    };
                }
                
                function evaluatePixel(sample) {
                    let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
                    return [ndvi];
                }
                """
            }
            
            url = f"{self.BASE_URL}/api/v1/process"
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            response = self.session.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Convert response to numpy array
            # This is simplified - in reality you'd need to handle the image format
            logger.info("Successfully retrieved NDVI data from Sentinel Hub")
            return None  # Placeholder - implement image processing
            
        except Exception as e:
            logger.error(f"Failed to fetch NDVI data: {e}")
            return None

class WeatherDataManager:
    """Manages weather and satellite data integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize weather data manager.
        
        Args:
            config: Configuration dictionary with API keys
        """
        self.config = config
        self.weather_apis = {}
        self.satellite_apis = {}
        
        # Initialize weather APIs
        if config.get('openweather_api_key'):
            self.weather_apis['openweathermap'] = OpenWeatherMapAPI(
                config['openweather_api_key']
            )
        
        # Initialize satellite APIs
        if all(config.get(key) for key in ['sentinelhub_client_id', 'sentinelhub_client_secret', 'sentinelhub_instance_id']):
            self.satellite_apis['sentinelhub'] = SentinelHubAPI(
                config['sentinelhub_client_id'],
                config['sentinelhub_client_secret'],
                config['sentinelhub_instance_id']
            )
    
    def get_weather_data(self, lat: float, lon: float, 
                        provider: str = 'openweathermap') -> Optional[WeatherData]:
        """
        Get weather data from specified provider.
        
        Args:
            lat: Latitude
            lon: Longitude
            provider: Weather data provider
            
        Returns:
            WeatherData object or None
        """
        api = self.weather_apis.get(provider)
        if not api:
            logger.warning(f"Weather provider '{provider}' not available")
            return None
        
        return api.get_current_weather(lat, lon)
    
    def get_satellite_data(self, bbox: Tuple[float, float, float, float],
                          provider: str = 'sentinelhub',
                          data_type: str = 'ndvi') -> Optional[SatelliteData]:
        """
        Get satellite data from specified provider.
        
        Args:
            bbox: Bounding box
            provider: Satellite data provider
            data_type: Type of data to retrieve
            
        Returns:
            SatelliteData object or None
        """
        api = self.satellite_apis.get(provider)
        if not api:
            logger.warning(f"Satellite provider '{provider}' not available")
            return None
        
        if data_type == 'ndvi':
            ndvi_data = api.get_ndvi_data(bbox)
            if ndvi_data is not None:
                return SatelliteData(
                    ndvi=ndvi_data,
                    timestamp=datetime.now(),
                    bbox=bbox
                )
        
        return None
    
    def has_live_data(self) -> bool:
        """Check if any live data sources are available."""
        return len(self.weather_apis) > 0 or len(self.satellite_apis) > 0
    
    def get_available_providers(self) -> Dict[str, List[str]]:
        """Get list of available data providers."""
        return {
            'weather': list(self.weather_apis.keys()),
            'satellite': list(self.satellite_apis.keys())
        }

def create_mock_weather_data(lat: float, lon: float) -> WeatherData:
    """
    Create mock weather data for testing.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Mock WeatherData object
    """
    # Generate realistic but fake weather data
    np.random.seed(int(lat * 1000 + lon * 1000) % 2**32)
    
    base_temp = 20 - (abs(lat) / 90) * 30  # Temperature decreases with latitude
    temperature = base_temp + np.random.normal(0, 5)
    
    humidity = np.random.normal(60, 20)
    humidity = np.clip(humidity, 0, 100)
    
    precipitation = np.random.exponential(2) if np.random.random() < 0.3 else 0
    wind_speed = np.random.gamma(2, 2)
    pressure = np.random.normal(1013, 20)
    
    return WeatherData(
        temperature=temperature,
        humidity=humidity,
        precipitation=precipitation,
        wind_speed=wind_speed,
        pressure=pressure,
        timestamp=datetime.now(),
        location=f"Location({lat:.2f}, {lon:.2f})"
    )
