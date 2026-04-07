"""Main training and evaluation script for crop yield prediction."""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.generator import CropYieldDataGenerator
from features.engineering import FeatureEngineer
from models.regression import get_default_models
from eval.metrics import ModelEvaluator
from configs.config import data_config, model_config, feature_config, eval_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training and evaluation pipeline."""
    logger.info("Starting crop yield prediction pipeline")
    
    # Set random seeds for reproducibility
    np.random.seed(data_config['seed'])
    
    # Generate synthetic data
    logger.info("Generating synthetic agricultural data")
    data_generator = CropYieldDataGenerator(seed=data_config['seed'])
    X, y = data_generator.generate_dataset(n_samples=data_config['n_samples'])
    
    # Add spatial features
    X_spatial = data_generator.add_spatial_features(X)
    
    logger.info(f"Generated dataset: {X_spatial.shape[0]} samples, {X_spatial.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_spatial, y, test_size=data_config['test_size'], random_state=data_config['seed']
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Feature engineering
    logger.info("Applying feature engineering")
    feature_engineer = FeatureEngineer(scaler_type=feature_config['scaler_type'])
    X_train_processed = feature_engineer.fit_transform(X_train, y_train)
    X_test_processed = feature_engineer.transform(X_test)
    
    logger.info(f"Processed features: {X_train_processed.shape[1]} features")
    
    # Initialize models
    logger.info("Initializing models")
    models = get_default_models()
    
    # Evaluate models
    logger.info("Evaluating models")
    evaluator = ModelEvaluator(cv_folds=eval_config['cv_folds'])
    leaderboard = evaluator.compare_models(models, X_train_processed, y_train)
    
    # Print results
    print("\n" + "="*60)
    print("CROP YIELD PREDICTION - MODEL COMPARISON")
    print("="*60)
    print(leaderboard.to_string(index=False, float_format='%.4f'))
    
    # Get best model
    best_model_name = leaderboard.iloc[0]['Model']
    best_model = evaluator.results[best_model_name]['model']
    
    # Evaluate on test set
    logger.info(f"Evaluating best model ({best_model_name}) on test set")
    test_predictions = best_model.predict(X_test_processed)
    test_metrics = evaluator.calculate_metrics(y_test.values, test_predictions)
    
    print(f"\nBest Model ({best_model_name}) Test Set Performance:")
    print("-" * 50)
    for metric, value in test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Generate plots
    logger.info("Generating evaluation plots")
    
    # Create assets directory
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Plot predictions for best model
    evaluator.plot_predictions(
        best_model_name, 
        y_test.values, 
        test_predictions,
        save_path=assets_dir / f"{best_model_name.lower().replace(' ', '_')}_predictions.png"
    )
    
    # Plot model comparison
    evaluator.plot_model_comparison(
        leaderboard,
        metric='RMSE',
        save_path=assets_dir / "model_comparison.png"
    )
    
    # Plot feature importance if available
    if hasattr(best_model, 'get_feature_importance'):
        try:
            feature_importance = best_model.get_feature_importance()
            evaluator.plot_feature_importance(
                best_model_name,
                feature_importance.index.tolist(),
                feature_importance.values,
                save_path=assets_dir / f"{best_model_name.lower().replace(' ', '_')}_feature_importance.png"
            )
        except Exception as e:
            logger.warning(f"Could not plot feature importance: {e}")
    
    # Generate comprehensive report
    logger.info("Generating evaluation report")
    report = evaluator.generate_report(leaderboard, output_path=assets_dir / "evaluation_report.txt")
    print("\n" + report)
    
    # Save results
    logger.info("Saving results")
    
    # Save leaderboard
    leaderboard.to_csv(assets_dir / "model_leaderboard.csv", index=False)
    
    # Save test predictions
    test_results = pd.DataFrame({
        'actual': y_test.values,
        'predicted': test_predictions,
        'residuals': y_test.values - test_predictions
    })
    test_results.to_csv(assets_dir / "test_predictions.csv", index=False)
    
    # Save processed data
    X_train_processed.to_csv(assets_dir / "train_features.csv")
    X_test_processed.to_csv(assets_dir / "test_features.csv")
    y_train.to_csv(assets_dir / "train_target.csv", index=False)
    y_test.to_csv(assets_dir / "test_target.csv", index=False)
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Results saved to {assets_dir}")


if __name__ == "__main__":
    main()
