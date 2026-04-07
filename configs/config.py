"""Configuration files for crop yield prediction project."""

# Data configuration
data_config = {
    'n_samples': 2000,
    'seed': 42,
    'test_size': 0.2,
    'validation_size': 0.1,
    'features': {
        'soil_quality': {'min': 1.0, 'max': 10.0, 'mean': 7.0, 'std': 1.5},
        'rainfall': {'min': 50.0, 'max': 600.0, 'mean': 300.0, 'std': 50.0},
        'temperature': {'min': 10.0, 'max': 40.0, 'mean': 25.0, 'std': 3.0},
        'fertilizer_use': {'min': 0.0, 'max': 300.0, 'mean': 150.0, 'std': 30.0},
        'pesticide_use': {'min': 0.0, 'max': 3.0, 'mean': 1.2, 'std': 0.3},
        'irrigation': {'min': 0.0, 'max': 200.0, 'mean': 100.0, 'std': 25.0},
        'planting_density': {'min': 20000.0, 'max': 80000.0, 'mean': 50000.0, 'std': 10000.0},
        'field_slope': {'min': 0.0, 'max': 15.0, 'mean': 2.0, 'std': 2.0},
        'days_since_harvest': {'min': 0.0, 'max': 365.0, 'mean': 30.0, 'std': 30.0}
    }
}

# Model configuration
model_config = {
    'linear_regression': {
        'fit_intercept': True,
        'normalize': False
    },
    'ridge_regression': {
        'alpha': 1.0,
        'fit_intercept': True
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42,
        'verbose': -1
    },
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    },
    'neural_network': {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'max_iter': 1000,
        'random_state': 42
    }
}

# Feature engineering configuration
feature_config = {
    'scaler_type': 'standard',  # 'standard' or 'robust'
    'polynomial_degree': 2,
    'rolling_windows': [3, 7, 14],
    'feature_selection': {
        'k_best': 50,
        'score_func': 'f_regression'
    },
    'interaction_features': True,
    'climate_features': True,
    'soil_features': True,
    'spatial_features': True
}

# Evaluation configuration
eval_config = {
    'cv_folds': 5,
    'metrics': ['mse', 'rmse', 'mae', 'r2', 'mape', 'smape'],
    'cross_validation': True,
    'feature_importance': True,
    'plots': {
        'predictions': True,
        'residuals': True,
        'feature_importance': True,
        'model_comparison': True
    }
}

# Visualization configuration
viz_config = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
    'save_plots': True,
    'plot_format': 'png'
}

# Geospatial configuration
geo_config = {
    'crs': 'EPSG:4326',
    'bounds': {
        'latitude': {'min': 30.0, 'max': 50.0},
        'longitude': {'min': -120.0, 'max': -70.0}
    },
    'climate_zones': {
        1: 'Subtropical',
        2: 'Temperate', 
        3: 'Cool Temperate'
    },
    'map_style': 'OpenStreetMap',
    'tile_provider': 'OpenStreetMap'
}

# Demo configuration
demo_config = {
    'app_title': 'Crop Yield Prediction Dashboard',
    'app_description': 'Interactive dashboard for agricultural yield prediction',
    'default_model': 'XGBoost',
    'sample_size': 100,
    'map_center': [40.0, -95.0],
    'map_zoom': 4,
    'update_interval': 1000  # milliseconds
}
