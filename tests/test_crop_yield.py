# Crop Yield Prediction - Unit Tests

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.generator import CropYieldDataGenerator
from features.engineering import FeatureEngineer
from models.regression import get_default_models, LinearRegressionModel, RandomForestModel
from eval.metrics import ModelEvaluator


class TestDataGenerator:
    """Test cases for data generation."""
    
    def test_data_generator_init(self):
        """Test data generator initialization."""
        generator = CropYieldDataGenerator(seed=42)
        assert generator.seed == 42
    
    def test_generate_features(self):
        """Test feature generation."""
        generator = CropYieldDataGenerator(seed=42)
        features = generator.generate_features(n_samples=100)
        
        assert len(features) == 100
        assert len(features.columns) == 9  # 9 base features
        
        # Check feature ranges
        assert features['soil_quality'].min() >= 1.0
        assert features['soil_quality'].max() <= 10.0
        assert features['rainfall'].min() >= 50.0
        assert features['temperature'].min() >= 10.0
    
    def test_generate_yield(self):
        """Test yield generation."""
        generator = CropYieldDataGenerator(seed=42)
        features = generator.generate_features(n_samples=100)
        yield_values = generator.generate_yield(features)
        
        assert len(yield_values) == 100
        assert yield_values.min() >= 0.5
        assert yield_values.max() <= 15.0
    
    def test_generate_dataset(self):
        """Test complete dataset generation."""
        generator = CropYieldDataGenerator(seed=42)
        X, y = generator.generate_dataset(n_samples=100)
        
        assert len(X) == 100
        assert len(y) == 100
        assert len(X.columns) == 9
    
    def test_add_spatial_features(self):
        """Test spatial feature addition."""
        generator = CropYieldDataGenerator(seed=42)
        features = generator.generate_features(n_samples=100)
        spatial_features = generator.add_spatial_features(features)
        
        assert 'latitude' in spatial_features.columns
        assert 'longitude' in spatial_features.columns
        assert 'climate_zone' in spatial_features.columns
        assert 'elevation' in spatial_features.columns
        
        assert spatial_features['latitude'].min() >= 30.0
        assert spatial_features['latitude'].max() <= 50.0


class TestFeatureEngineer:
    """Test cases for feature engineering."""
    
    def test_feature_engineer_init(self):
        """Test feature engineer initialization."""
        engineer = FeatureEngineer(scaler_type='standard')
        assert engineer.scaler_type == 'standard'
    
    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'soil_quality': [5, 7, 9],
            'fertilizer_use': [100, 150, 200],
            'temperature': [20, 25, 30],
            'rainfall': [200, 300, 400]
        })
        
        enhanced_df = engineer.create_interaction_features(df)
        
        assert 'soil_fertilizer' in enhanced_df.columns
        assert 'temp_rainfall' in enhanced_df.columns
        assert len(enhanced_df.columns) > len(df.columns)
    
    def test_create_polynomial_features(self):
        """Test polynomial feature creation."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'soil_quality': [5, 7, 9],
            'temperature': [20, 25, 30]
        })
        
        poly_df = engineer.create_polynomial_features(df, degree=2)
        
        assert 'soil_quality_pow_2' in poly_df.columns
        assert 'temperature_pow_2' in poly_df.columns
    
    def test_fit_transform(self):
        """Test fit and transform pipeline."""
        engineer = FeatureEngineer()
        
        # Generate test data
        generator = CropYieldDataGenerator(seed=42)
        X, y = generator.generate_dataset(n_samples=100)
        
        X_processed = engineer.fit_transform(X, y)
        
        assert X_processed.shape[0] == 100
        assert X_processed.shape[1] > X.shape[1]  # More features after engineering
        assert not X_processed.isnull().any().any()  # No missing values


class TestModels:
    """Test cases for model implementations."""
    
    def test_linear_regression_model(self):
        """Test linear regression model."""
        model = LinearRegressionModel()
        
        # Generate test data
        generator = CropYieldDataGenerator(seed=42)
        X, y = generator.generate_dataset(n_samples=100)
        
        # Fit and predict
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == 100
        assert not np.isnan(predictions).any()
        assert model.is_fitted
    
    def test_random_forest_model(self):
        """Test random forest model."""
        model = RandomForestModel(n_estimators=10)  # Small for testing
        
        # Generate test data
        generator = CropYieldDataGenerator(seed=42)
        X, y = generator.generate_dataset(n_samples=100)
        
        # Fit and predict
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == 100
        assert not np.isnan(predictions).any()
        assert model.is_fitted
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == len(X.columns)
    
    def test_get_default_models(self):
        """Test default model list."""
        models = get_default_models()
        
        assert len(models) >= 5  # Should have multiple models
        assert all(hasattr(model, 'name') for model in models)
        assert all(hasattr(model, 'fit') for model in models)
        assert all(hasattr(model, 'predict') for model in models)


class TestModelEvaluator:
    """Test cases for model evaluation."""
    
    def test_evaluator_init(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator(cv_folds=3)
        assert evaluator.cv_folds == 3
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        assert 'smape' in metrics
        assert 'bias' in metrics
        assert 'correlation' in metrics
        
        # Check metric values are reasonable
        assert metrics['r2'] > 0.9  # Good fit
        assert metrics['mape'] < 10  # Low error percentage
    
    def test_evaluate_model(self):
        """Test single model evaluation."""
        evaluator = ModelEvaluator(cv_folds=3)
        
        # Generate test data
        generator = CropYieldDataGenerator(seed=42)
        X, y = generator.generate_dataset(n_samples=100)
        
        # Test with linear regression
        model = LinearRegressionModel()
        results = evaluator.evaluate_model(model, X, y)
        
        assert 'model_name' in results
        assert 'metrics' in results
        assert 'predictions' in results
        assert 'model' in results
        
        assert results['model_name'] == 'Linear Regression'
        assert len(results['predictions']) == 100
    
    def test_compare_models(self):
        """Test model comparison."""
        evaluator = ModelEvaluator(cv_folds=3)
        
        # Generate test data
        generator = CropYieldDataGenerator(seed=42)
        X, y = generator.generate_dataset(n_samples=100)
        
        # Test with subset of models for speed
        models = [LinearRegressionModel(), RandomForestModel(n_estimators=10)]
        
        leaderboard = evaluator.compare_models(models, X, y)
        
        assert len(leaderboard) == 2
        assert 'Model' in leaderboard.columns
        assert 'RMSE' in leaderboard.columns
        assert 'MAE' in leaderboard.columns
        assert 'R²' in leaderboard.columns
        
        # Check that results are sorted by RMSE
        assert leaderboard['RMSE'].iloc[0] <= leaderboard['RMSE'].iloc[1]


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_complete_pipeline(self):
        """Test the complete training and evaluation pipeline."""
        # Generate data
        generator = CropYieldDataGenerator(seed=42)
        X, y = generator.generate_dataset(n_samples=200)
        
        # Feature engineering
        engineer = FeatureEngineer()
        X_processed = engineer.fit_transform(X, y)
        
        # Train models
        models = [LinearRegressionModel(), RandomForestModel(n_estimators=10)]
        
        # Evaluate
        evaluator = ModelEvaluator(cv_folds=3)
        leaderboard = evaluator.compare_models(models, X_processed, y)
        
        # Check results
        assert len(leaderboard) == 2
        assert leaderboard['RMSE'].min() > 0
        assert leaderboard['R²'].max() > 0
        
        # Test best model
        best_model_name = leaderboard.iloc[0]['Model']
        best_model = evaluator.results[best_model_name]['model']
        
        assert best_model.is_fitted
        predictions = best_model.predict(X_processed)
        assert len(predictions) == 200


if __name__ == "__main__":
    pytest.main([__file__])
