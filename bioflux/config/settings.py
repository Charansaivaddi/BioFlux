#!/usr/bin/env python3
"""
Configuration management for BioFlux API keys and settings.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BioFluxConfig:
    """Configuration manager for BioFlux API keys and settings."""
    
    def __init__(self, env_file: str = ".env"):
        """
        Initialize configuration manager.
        
        Args:
            env_file: Path to environment file
        """
        self.env_file = env_file
        self.config = {}
        self.load_environment()
    
    def load_environment(self):
        """Load environment variables from .env file and system environment."""
        
        # Try to load from .env file
        env_path = Path(self.env_file)
        if env_path.exists():
            logger.info(f"Loading environment from {self.env_file}")
            self.load_env_file(env_path)
        else:
            logger.warning(f"No {self.env_file} file found. Using system environment variables only.")
        
        # Load configuration
        self.config = {
            # Weather APIs
            'openweather_api_key': os.getenv('OPENWEATHER_API_KEY'),
            'weather_underground_api_key': os.getenv('WEATHER_UNDERGROUND_API_KEY'),
            
            # Satellite Data APIs
            'sentinelhub_instance_id': os.getenv('SENTINELHUB_INSTANCE_ID'),
            'sentinelhub_client_id': os.getenv('SENTINELHUB_CLIENT_ID'),
            'sentinelhub_client_secret': os.getenv('SENTINELHUB_CLIENT_SECRET'),
            'nasa_earthdata_username': os.getenv('NASA_EARTHDATA_USERNAME'),
            'nasa_earthdata_password': os.getenv('NASA_EARTHDATA_PASSWORD'),
            'google_earth_engine_key': os.getenv('GOOGLE_EARTH_ENGINE_SERVICE_ACCOUNT_KEY'),
            
            # Elevation APIs (Free services only)
            'usgs_elevation_enabled': True,  # Free USGS service
            
            # Default coordinates (can be overridden)
            'default_latitude': float(os.getenv('DEFAULT_LATITUDE', '40.7128')),  # NYC
            'default_longitude': float(os.getenv('DEFAULT_LONGITUDE', '-74.0060')),
            
            # Simulation settings
            'simulation_width': int(os.getenv('SIMULATION_WIDTH', '100')),
            'simulation_height': int(os.getenv('SIMULATION_HEIGHT', '100')),
            'max_predators': int(os.getenv('MAX_PREDATORS', '20')),
            'max_prey': int(os.getenv('MAX_PREY', '100')),
            'max_plants': int(os.getenv('MAX_PLANTS', '200')),
            
            # Data cache settings
            'cache_dir': os.getenv('CACHE_DIR', './data_cache'),
            'cache_enabled': os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
            
            # Visualization settings
            'save_plots': os.getenv('SAVE_PLOTS', 'true').lower() == 'true',
            'plot_interval': int(os.getenv('PLOT_INTERVAL', '10')),
            'output_dir': os.getenv('OUTPUT_DIR', './output'),
        }
        
        # Log configuration status
        self.log_config_status()
    
    def load_env_file(self, env_path: Path):
        """
        Load environment variables from .env file.
        
        Args:
            env_path: Path to .env file
        """
        try:
            with open(env_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        # Set environment variable if not empty
                        if value and not value.startswith('#'):
                            os.environ[key] = value
                    else:
                        logger.warning(f"Invalid line {line_num} in {env_path}: {line}")
                        
        except Exception as e:
            logger.error(f"Error loading {env_path}: {e}")
    
    def log_config_status(self):
        """Log the status of API configuration."""
        logger.info("=== BioFlux Configuration Status ===")
        
        # Weather APIs
        weather_apis = []
        if self.config['openweather_api_key']:
            weather_apis.append("OpenWeatherMap")
        if self.config['weather_underground_api_key']:
            weather_apis.append("Weather Underground")
        
        if weather_apis:
            logger.info(f"✅ Weather APIs: {', '.join(weather_apis)}")
        else:
            logger.warning("⚠️  No weather APIs configured - using mock data")
        
        # Satellite APIs
        satellite_apis = []
        if all([self.config['sentinelhub_client_id'], 
                self.config['sentinelhub_client_secret'], 
                self.config['sentinelhub_instance_id']]):
            satellite_apis.append("Sentinel Hub")
        if all([self.config['nasa_earthdata_username'], 
                self.config['nasa_earthdata_password']]):
            satellite_apis.append("NASA EarthData")
        if self.config['google_earth_engine_key']:
            satellite_apis.append("Google Earth Engine")
        
        if satellite_apis:
            logger.info(f"✅ Satellite APIs: {', '.join(satellite_apis)}")
        else:
            logger.warning("⚠️  No satellite APIs configured - using mock data")
        
        # Overall status
        if weather_apis or satellite_apis:
            logger.info("🌍 LIVE DATA mode enabled")
        else:
            logger.info("🧪 MOCK DATA mode - all data will be simulated")
        
        logger.info("=====================================")
    
    def has_live_data(self) -> bool:
        """Check if any live data sources are configured."""
        weather_available = bool(
            self.config['openweather_api_key'] or 
            self.config['weather_underground_api_key']
        )
        
        satellite_available = bool(
            (self.config['sentinelhub_client_id'] and 
             self.config['sentinelhub_client_secret'] and 
             self.config['sentinelhub_instance_id']) or
            (self.config['nasa_earthdata_username'] and 
             self.config['nasa_earthdata_password']) or
            self.config['google_earth_engine_key']
        )
        
        return weather_available or satellite_available
    
    def get_weather_config(self) -> Dict[str, Any]:
        """Get weather API configuration."""
        return {
            'openweather_api_key': self.config['openweather_api_key'],
            'weather_underground_api_key': self.config['weather_underground_api_key'],
            'default_lat': self.config['default_latitude'],
            'default_lon': self.config['default_longitude']
        }
    
    def get_satellite_config(self) -> Dict[str, Any]:
        """Get satellite API configuration."""
        return {
            'sentinelhub_instance_id': self.config['sentinelhub_instance_id'],
            'sentinelhub_client_id': self.config['sentinelhub_client_id'],
            'sentinelhub_client_secret': self.config['sentinelhub_client_secret'],
            'nasa_earthdata_username': self.config['nasa_earthdata_username'],
            'nasa_earthdata_password': self.config['nasa_earthdata_password'],
            'google_earth_engine_key': self.config['google_earth_engine_key']
        }
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """Get simulation configuration."""
        return {
            'width': self.config['simulation_width'],
            'height': self.config['simulation_height'],
            'max_predators': self.config['max_predators'],
            'max_prey': self.config['max_prey'],
            'max_plants': self.config['max_plants'],
            'default_lat': self.config['default_latitude'],
            'default_lon': self.config['default_longitude']
        }
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output and caching configuration."""
        return {
            'cache_dir': self.config['cache_dir'],
            'cache_enabled': self.config['cache_enabled'],
            'save_plots': self.config['save_plots'],
            'plot_interval': self.config['plot_interval'],
            'output_dir': self.config['output_dir']
        }
    
    def get_available_apis(self) -> Dict[str, List[str]]:
        """Get list of available APIs."""
        weather_apis = []
        if self.config['openweather_api_key']:
            weather_apis.append('openweathermap')
        if self.config['weather_underground_api_key']:
            weather_apis.append('weather_underground')
        
        satellite_apis = []
        if all([self.config['sentinelhub_client_id'], 
                self.config['sentinelhub_client_secret'], 
                self.config['sentinelhub_instance_id']]):
            satellite_apis.append('sentinelhub')
        if all([self.config['nasa_earthdata_username'], 
                self.config['nasa_earthdata_password']]):
            satellite_apis.append('nasa_earthdata')
        if self.config['google_earth_engine_key']:
            satellite_apis.append('google_earth_engine')
        
        return {
            'weather': weather_apis,
            'satellite': satellite_apis
        }
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self.config.update(updates)
        logger.info(f"Configuration updated: {list(updates.keys())}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        logger.debug(f"Configuration set: {key} = {value}")
    
    def validate_config(self) -> Dict[str, List[str]]:
        """
        Validate configuration and return issues.
        
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        errors = []
        warnings = []
        
        # Check for required paths
        try:
            cache_dir = Path(self.config['cache_dir'])
            cache_dir.mkdir(exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create cache directory: {e}")
        
        try:
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")
        
        # Check coordinate validity
        lat = self.config['default_latitude']
        lon = self.config['default_longitude']
        if not (-90 <= lat <= 90):
            errors.append(f"Invalid latitude: {lat} (must be -90 to 90)")
        if not (-180 <= lon <= 180):
            errors.append(f"Invalid longitude: {lon} (must be -180 to 180)")
        
        # Check simulation parameters
        if self.config['simulation_width'] <= 0:
            errors.append("Simulation width must be positive")
        if self.config['simulation_height'] <= 0:
            errors.append("Simulation height must be positive")
        
        # Warnings for missing APIs
        if not self.has_live_data():
            warnings.append("No live data APIs configured - simulation will use mock data")
        
        return {'errors': errors, 'warnings': warnings}
    
    def __str__(self) -> str:
        """String representation of configuration."""
        available_apis = self.get_available_apis()
        weather_count = len(available_apis['weather'])
        satellite_count = len(available_apis['satellite'])
        
        return (f"BioFluxConfig(weather_APIs={weather_count}, "
                f"satellite_APIs={satellite_count}, "
                f"live_data={self.has_live_data()})")

# Global configuration instance
_global_config = None

def get_config(env_file: str = ".env") -> BioFluxConfig:
    """
    Get global configuration instance.
    
    Args:
        env_file: Path to environment file
        
    Returns:
        BioFluxConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = BioFluxConfig(env_file)
    return _global_config

def reload_config(env_file: str = ".env") -> BioFluxConfig:
    """
    Reload global configuration.
    
    Args:
        env_file: Path to environment file
        
    Returns:
        New BioFluxConfig instance
    """
    global _global_config
    _global_config = BioFluxConfig(env_file)
    return _global_config
