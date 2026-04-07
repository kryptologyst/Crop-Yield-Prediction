"""Package initialization for crop yield prediction."""

__version__ = "1.0.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"
__description__ = "Environmental & Social Applications: Crop Yield Prediction"

from .data.generator import CropYieldDataGenerator
from .features.engineering import FeatureEngineer
from .models.regression import get_default_models
from .eval.metrics import ModelEvaluator
from .viz.plots import CropYieldVisualizer

__all__ = [
    "CropYieldDataGenerator",
    "FeatureEngineer", 
    "get_default_models",
    "ModelEvaluator",
    "CropYieldVisualizer"
]
