"""Feature engineering for crop yield prediction."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for agricultural data."""
    
    def __init__(self, scaler_type: str = 'standard') -> None:
        """Initialize feature engineer.
        
        Args:
            scaler_type: Type of scaler to use ('standard' or 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else RobustScaler()
        self.feature_selector = None
        self.feature_names = None
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional interaction features
        """
        logger.info("Creating interaction features")
        
        df_enhanced = df.copy()
        
        # Soil-fertilizer interaction
        df_enhanced['soil_fertilizer'] = df['soil_quality'] * df['fertilizer_use']
        
        # Temperature-rainfall interaction
        df_enhanced['temp_rainfall'] = df['temperature'] * df['rainfall']
        
        # Irrigation efficiency (irrigation per unit rainfall)
        df_enhanced['irrigation_efficiency'] = df['irrigation'] / (df['rainfall'] + 1e-6)
        
        # Planting density normalized by soil quality
        df_enhanced['density_soil_ratio'] = df['planting_density'] / (df['soil_quality'] + 1e-6)
        
        # Field slope impact on water retention
        df_enhanced['slope_water_factor'] = df['field_slope'] * (df['rainfall'] + df['irrigation'])
        
        # Time-based features
        df_enhanced['harvest_recovery'] = np.exp(-df['days_since_harvest'] / 30)
        
        return df_enhanced
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for non-linear relationships.
        
        Args:
            df: Input DataFrame
            degree: Degree of polynomial features
            
        Returns:
            DataFrame with polynomial features
        """
        logger.info(f"Creating polynomial features of degree {degree}")
        
        df_poly = df.copy()
        
        # Create polynomial features for key variables
        key_vars = ['soil_quality', 'temperature', 'rainfall', 'fertilizer_use']
        
        for var in key_vars:
            if var in df.columns:
                for d in range(2, degree + 1):
                    df_poly[f'{var}_pow_{d}'] = df[var] ** d
        
        return df_poly
    
    def create_rolling_features(self, df: pd.DataFrame, window_sizes: List[int] = [3, 7, 14]) -> pd.DataFrame:
        """Create rolling statistical features.
        
        Args:
            df: Input DataFrame
            window_sizes: List of window sizes for rolling features
            
        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Creating rolling features with windows {window_sizes}")
        
        df_rolling = df.copy()
        
        # For time-series like features, create rolling statistics
        time_like_vars = ['temperature', 'rainfall', 'irrigation']
        
        for var in time_like_vars:
            if var in df.columns:
                for window in window_sizes:
                    # Rolling mean
                    df_rolling[f'{var}_rolling_mean_{window}'] = df[var].rolling(window=window, min_periods=1).mean()
                    
                    # Rolling std
                    df_rolling[f'{var}_rolling_std_{window}'] = df[var].rolling(window=window, min_periods=1).std()
                    
                    # Rolling max/min
                    df_rolling[f'{var}_rolling_max_{window}'] = df[var].rolling(window=window, min_periods=1).max()
                    df_rolling[f'{var}_rolling_min_{window}'] = df[var].rolling(window=window, min_periods=1).min()
        
        return df_rolling
    
    def create_climate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create climate-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with climate features
        """
        logger.info("Creating climate features")
        
        df_climate = df.copy()
        
        # Temperature stress indicators
        df_climate['temp_stress_high'] = np.maximum(0, df['temperature'] - 30)
        df_climate['temp_stress_low'] = np.maximum(0, 15 - df['temperature'])
        
        # Optimal temperature range (25°C ± 5°C)
        df_climate['temp_optimal'] = np.where(
            (df['temperature'] >= 20) & (df['temperature'] <= 30), 1, 0
        )
        
        # Water stress indicators
        df_climate['water_stress'] = np.maximum(0, 200 - (df['rainfall'] + df['irrigation']))
        
        # Drought indicator
        df_climate['drought_indicator'] = np.where(
            (df['rainfall'] < 100) & (df['irrigation'] < 50), 1, 0
        )
        
        # Excess water indicator
        df_climate['excess_water'] = np.where(
            (df['rainfall'] + df['irrigation']) > 400, 1, 0
        )
        
        return df_climate
    
    def create_soil_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create soil-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with soil features
        """
        logger.info("Creating soil features")
        
        df_soil = df.copy()
        
        # Soil quality categories
        df_soil['soil_poor'] = np.where(df['soil_quality'] < 4, 1, 0)
        df_soil['soil_fair'] = np.where((df['soil_quality'] >= 4) & (df['soil_quality'] < 7), 1, 0)
        df_soil['soil_good'] = np.where((df['soil_quality'] >= 7) & (df['soil_quality'] < 9), 1, 0)
        df_soil['soil_excellent'] = np.where(df['soil_quality'] >= 9, 1, 0)
        
        # Fertilizer efficiency (fertilizer per soil quality)
        df_soil['fertilizer_efficiency'] = df['fertilizer_use'] / (df['soil_quality'] + 1e-6)
        
        # Pesticide necessity (higher for poor soil)
        df_soil['pesticide_necessity'] = df['pesticide_use'] * (10 - df['soil_quality'])
        
        return df_soil
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit the feature engineering pipeline and transform data.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            Transformed features
        """
        logger.info("Fitting and transforming features")
        
        # Apply all feature engineering steps
        X_processed = X.copy()
        
        # Create different types of features
        X_processed = self.create_interaction_features(X_processed)
        X_processed = self.create_polynomial_features(X_processed, degree=2)
        X_processed = self.create_climate_features(X_processed)
        X_processed = self.create_soil_features(X_processed)
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Store feature names
        self.feature_names = X_processed.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X_processed.index)
        
        # Feature selection if target is provided
        if y is not None:
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(50, len(self.feature_names)))
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            
            # Get selected feature names
            selected_features = [self.feature_names[i] for i in self.feature_selector.get_support(indices=True)]
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X_scaled.index)
            
            logger.info(f"Selected {len(selected_features)} features out of {len(self.feature_names)}")
            return X_selected
        
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        logger.info("Transforming features")
        
        if self.feature_names is None:
            raise ValueError("Feature engineer must be fitted first")
        
        # Apply same feature engineering steps
        X_processed = X.copy()
        X_processed = self.create_interaction_features(X_processed)
        X_processed = self.create_polynomial_features(X_processed, degree=2)
        X_processed = self.create_climate_features(X_processed)
        X_processed = self.create_soil_features(X_processed)
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Scale features
        X_scaled = self.scaler.transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X_processed.index)
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
            selected_features = [self.feature_names[i] for i in self.feature_selector.get_support(indices=True)]
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X_scaled.index)
            return X_selected
        
        return X_scaled
