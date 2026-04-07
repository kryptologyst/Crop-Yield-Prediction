"""Data generation and preprocessing for crop yield prediction."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CropYieldDataGenerator:
    """Generate synthetic agricultural data for crop yield prediction."""
    
    def __init__(self, seed: int = 42) -> None:
        """Initialize the data generator with a random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
    def generate_features(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic agricultural features.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with agricultural features
        """
        logger.info(f"Generating {n_samples} agricultural samples")
        
        # Soil quality (1-10 scale, higher is better)
        soil_quality = np.random.normal(7.0, 1.5, n_samples)
        soil_quality = np.clip(soil_quality, 1.0, 10.0)
        
        # Rainfall (mm/month)
        rainfall = np.random.normal(300.0, 50.0, n_samples)
        rainfall = np.clip(rainfall, 50.0, 600.0)
        
        # Temperature (°C)
        temperature = np.random.normal(25.0, 3.0, n_samples)
        temperature = np.clip(temperature, 10.0, 40.0)
        
        # Fertilizer use (kg/hectare)
        fertilizer_use = np.random.normal(150.0, 30.0, n_samples)
        fertilizer_use = np.clip(fertilizer_use, 0.0, 300.0)
        
        # Pesticide use (liters/hectare)
        pesticide_use = np.random.normal(1.2, 0.3, n_samples)
        pesticide_use = np.clip(pesticide_use, 0.0, 3.0)
        
        # Additional features for more realistic modeling
        # Irrigation (mm/month)
        irrigation = np.random.normal(100.0, 25.0, n_samples)
        irrigation = np.clip(irrigation, 0.0, 200.0)
        
        # Planting density (plants/hectare)
        planting_density = np.random.normal(50000.0, 10000.0, n_samples)
        planting_density = np.clip(planting_density, 20000.0, 80000.0)
        
        # Field slope (degrees)
        field_slope = np.random.exponential(2.0, n_samples)
        field_slope = np.clip(field_slope, 0.0, 15.0)
        
        # Days since last harvest
        days_since_harvest = np.random.exponential(30.0, n_samples)
        days_since_harvest = np.clip(days_since_harvest, 0.0, 365.0)
        
        # Create DataFrame
        features = pd.DataFrame({
            'soil_quality': soil_quality,
            'rainfall': rainfall,
            'temperature': temperature,
            'fertilizer_use': fertilizer_use,
            'pesticide_use': pesticide_use,
            'irrigation': irrigation,
            'planting_density': planting_density,
            'field_slope': field_slope,
            'days_since_harvest': days_since_harvest
        })
        
        return features
    
    def generate_yield(self, features: pd.DataFrame) -> pd.Series:
        """Generate crop yield based on features using a realistic model.
        
        Args:
            features: DataFrame with agricultural features
            
        Returns:
            Series with crop yield predictions (tons/hectare)
        """
        logger.info("Generating crop yield based on features")
        
        # Base yield calculation with realistic relationships
        yield_base = (
            0.4 * features['soil_quality'] +
            0.02 * features['rainfall'] +
            0.01 * features['irrigation'] -
            0.1 * np.abs(features['temperature'] - 25) +
            0.05 * features['fertilizer_use'] -
            0.2 * features['pesticide_use'] +
            0.0001 * features['planting_density'] -
            0.05 * features['field_slope'] -
            0.001 * features['days_since_harvest']
        )
        
        # Add interaction effects
        yield_interactions = (
            0.01 * features['soil_quality'] * features['rainfall'] / 100 +
            0.02 * features['fertilizer_use'] * features['soil_quality'] / 10 -
            0.005 * features['pesticide_use'] * features['temperature']
        )
        
        # Add noise
        noise = np.random.normal(0, 0.5, len(features))
        
        # Combine all effects
        yield_total = yield_base + yield_interactions + noise
        
        # Ensure realistic yield range (0.5 to 15 tons/hectare)
        yield_total = np.clip(yield_total, 0.5, 15.0)
        
        return pd.Series(yield_total, name='crop_yield')
    
    def generate_dataset(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate complete dataset with features and target.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (features_df, target_series)
        """
        features = self.generate_features(n_samples)
        target = self.generate_yield(features)
        
        logger.info(f"Generated dataset with {len(features)} samples and {len(features.columns)} features")
        
        return features, target
    
    def add_spatial_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add spatial features to the dataset.
        
        Args:
            features: Original features DataFrame
            
        Returns:
            DataFrame with additional spatial features
        """
        logger.info("Adding spatial features")
        
        # Generate random coordinates (simulating field locations)
        n_samples = len(features)
        latitude = np.random.uniform(30.0, 50.0, n_samples)  # US agricultural regions
        longitude = np.random.uniform(-120.0, -70.0, n_samples)
        
        # Add spatial features
        features_spatial = features.copy()
        features_spatial['latitude'] = latitude
        features_spatial['longitude'] = longitude
        
        # Add regional climate zones based on coordinates
        features_spatial['climate_zone'] = self._assign_climate_zones(latitude, longitude)
        
        # Add elevation (simplified)
        features_spatial['elevation'] = np.random.normal(200.0, 100.0, n_samples)
        features_spatial['elevation'] = np.clip(features_spatial['elevation'], 0.0, 1000.0)
        
        return features_spatial
    
    def _assign_climate_zones(self, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        """Assign climate zones based on coordinates.
        
        Args:
            latitude: Latitude coordinates
            longitude: Longitude coordinates
            
        Returns:
            Array of climate zone labels
        """
        zones = np.zeros(len(latitude), dtype=int)
        
        # Simple climate zone assignment based on latitude
        zones[latitude < 35] = 1  # Subtropical
        zones[(latitude >= 35) & (latitude < 42)] = 2  # Temperate
        zones[latitude >= 42] = 3  # Cool temperate
        
        return zones
