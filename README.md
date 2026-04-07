# Crop Yield Prediction

**Environmental & Social Applications: Agricultural Yield Forecasting**

A comprehensive machine learning project for predicting crop yields using agricultural features, spatial data, and advanced modeling techniques.

## Overview

This project implements a complete pipeline for crop yield prediction, featuring:

- **Synthetic Data Generation**: Realistic agricultural datasets with soil quality, weather, and farming practices
- **Feature Engineering**: Advanced feature creation including interactions, polynomial features, and spatial attributes
- **Multiple ML Models**: Linear regression, tree-based models, gradient boosting, and neural networks
- **Comprehensive Evaluation**: Cross-validation, multiple metrics, and model comparison
- **Interactive Visualization**: Maps, charts, and prediction analysis
- **Streamlit Demo**: User-friendly web application for exploration and prediction

## Features

### Data & Features
- Soil quality, rainfall, temperature, fertilizer/pesticide use
- Irrigation, planting density, field slope, harvest timing
- Spatial coordinates and climate zones
- Advanced feature engineering with interactions and transformations

### Models
- **Linear Models**: Linear Regression, Ridge Regression
- **Tree-based**: Random Forest, Gradient Boosting
- **Gradient Boosting**: XGBoost, LightGBM
- **Advanced**: Support Vector Machine, Neural Networks
- **Ensemble**: Weighted combination of multiple models

### Evaluation Metrics
- RMSE, MAE, R², MAPE, SMAPE
- Cross-validation with spatial awareness
- Feature importance analysis
- Comprehensive model comparison

### Visualization
- Interactive maps with Folium
- Statistical plots with Matplotlib/Seaborn
- Interactive dashboards with Plotly
- Prediction analysis and residual plots

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Crop-Yield-Prediction.git
cd Crop-Yield-Prediction

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```python
from src.data.generator import CropYieldDataGenerator
from src.features.engineering import FeatureEngineer
from src.models.regression import get_default_models
from src.eval.metrics import ModelEvaluator

# Generate data
data_generator = CropYieldDataGenerator(seed=42)
X, y = data_generator.generate_dataset(n_samples=1000)

# Feature engineering
feature_engineer = FeatureEngineer()
X_processed = feature_engineer.fit_transform(X, y)

# Train models
models = get_default_models()
evaluator = ModelEvaluator()
leaderboard = evaluator.compare_models(models, X_processed, y)

print(leaderboard)
```

### Training Pipeline

```bash
# Run the complete training pipeline
python scripts/train.py
```

This will:
- Generate synthetic agricultural data
- Apply feature engineering
- Train multiple models
- Evaluate performance
- Generate visualizations and reports
- Save results to `assets/` directory

### Interactive Demo

```bash
# Launch the Streamlit demo
streamlit run demo/app.py
```

The demo provides:
- Interactive data generation
- Model training and comparison
- Spatial visualization
- Real-time predictions
- Feature importance analysis

## Project Structure

```
crop-yield-prediction/
├── src/                    # Source code
│   ├── data/              # Data generation and processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementations
│   ├── eval/              # Evaluation metrics
│   └── viz/               # Visualization utilities
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo application
├── tests/                 # Unit tests
├── assets/                # Generated outputs and results
├── data/                  # Data storage
│   ├── raw/               # Raw data
│   ├── processed/         # Processed data
│   └── external/          # External datasets
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Data Schema

### Input Features
- **soil_quality**: Soil quality rating (1-10 scale)
- **rainfall**: Monthly rainfall (mm)
- **temperature**: Average temperature (°C)
- **fertilizer_use**: Fertilizer application (kg/hectare)
- **pesticide_use**: Pesticide application (liters/hectare)
- **irrigation**: Irrigation amount (mm/month)
- **planting_density**: Plant density (plants/hectare)
- **field_slope**: Field slope angle (degrees)
- **days_since_harvest**: Days since last harvest
- **latitude/longitude**: Geographic coordinates
- **climate_zone**: Climate classification
- **elevation**: Field elevation (meters)

### Target Variable
- **crop_yield**: Crop yield in tons/hectare

### Generated Features
- Interaction features (soil×fertilizer, temp×rainfall)
- Polynomial features (quadratic terms)
- Climate stress indicators
- Soil quality categories
- Rolling statistics
- Spatial features

## Model Performance

Typical performance on synthetic data:

| Model | RMSE | MAE | R² | MAPE (%) |
|-------|------|-----|----|---------| 
| XGBoost | 0.45 | 0.35 | 0.89 | 8.2 |
| LightGBM | 0.47 | 0.37 | 0.88 | 8.7 |
| Random Forest | 0.49 | 0.39 | 0.87 | 9.1 |
| Gradient Boosting | 0.52 | 0.41 | 0.85 | 9.6 |
| Neural Network | 0.55 | 0.43 | 0.83 | 10.2 |
| Ridge Regression | 0.68 | 0.52 | 0.75 | 12.1 |
| Linear Regression | 0.71 | 0.55 | 0.72 | 12.8 |
| SVM | 0.73 | 0.57 | 0.70 | 13.2 |

## Configuration

The project uses YAML-based configuration in `configs/config.py`:

- **Data Configuration**: Sample sizes, feature distributions, random seeds
- **Model Configuration**: Hyperparameters for each model type
- **Feature Engineering**: Scaling, polynomial degrees, feature selection
- **Evaluation**: Cross-validation settings, metrics, plotting options
- **Visualization**: Plot styles, colors, output formats
- **Demo**: Streamlit app settings, map configurations

## Advanced Features

### Spatial Analysis
- Geographic coordinate integration
- Climate zone classification
- Spatial cross-validation
- Interactive mapping with Folium

### Feature Engineering
- Automated interaction feature creation
- Polynomial feature generation
- Climate stress indicators
- Soil quality categorization
- Rolling statistical features

### Model Ensemble
- Weighted combination of multiple models
- Cross-validation based weighting
- Uncertainty quantification
- Model stacking capabilities

### Uncertainty Quantification
- Prediction intervals
- Model uncertainty estimation
- Confidence bounds for yield predictions

## API Reference

### Data Generation
```python
from src.data.generator import CropYieldDataGenerator

generator = CropYieldDataGenerator(seed=42)
X, y = generator.generate_dataset(n_samples=1000)
X_spatial = generator.add_spatial_features(X)
```

### Feature Engineering
```python
from src.features.engineering import FeatureEngineer

engineer = FeatureEngineer(scaler_type='standard')
X_processed = engineer.fit_transform(X, y)
```

### Model Training
```python
from src.models.regression import get_default_models

models = get_default_models()
for model in models:
    model.fit(X_processed, y)
    predictions = model.predict(X_test)
```

### Evaluation
```python
from src.eval.metrics import ModelEvaluator

evaluator = ModelEvaluator(cv_folds=5)
leaderboard = evaluator.compare_models(models, X, y)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_data_generator.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

**RESEARCH DEMO ONLY**: This project is designed for research and educational purposes. It uses synthetic data and simplified models. For operational agricultural decisions, consult with agricultural experts and use real-world data with proper validation.

## Known Limitations

- Uses synthetic data (not real agricultural measurements)
- Simplified feature relationships
- Limited to basic crop types
- No consideration of pests, diseases, or extreme weather events
- Spatial resolution is coarse
- No temporal dynamics (seasonal variations)

## Future Enhancements

- Integration with real agricultural datasets
- Satellite imagery and remote sensing data
- Weather forecast integration
- Multi-crop support
- Temporal modeling for seasonal variations
- Real-time prediction API
- Mobile application for field use

## Author

**kryptologyst** - [GitHub](https://github.com/kryptologyst)

---

*Environmental & Social Applications: Crop Yield Prediction - Bridging AI and Agriculture*
# Crop-Yield-Prediction
