"""Evaluation metrics and model comparison for crop yield prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for crop yield prediction."""
    
    def __init__(self, cv_folds: int = 5) -> None:
        """Initialize evaluator.
        
        Args:
            cv_folds: Number of cross-validation folds
        """
        self.cv_folds = cv_folds
        self.results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100,
            'smape': np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100,
            'bias': np.mean(y_pred - y_true),
            'correlation': np.corrcoef(y_true, y_pred)[0, 1]
        }
        
        return metrics
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                      model_name: str = None) -> Dict[str, Any]:
        """Evaluate a single model with cross-validation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            model_name: Name of the model
            
        Returns:
            Evaluation results
        """
        if model_name is None:
            model_name = getattr(model, 'name', 'Unknown Model')
        
        logger.info(f"Evaluating {model_name}")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model.model, X, y, cv=self.cv_folds, 
                                   scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        # Fit model and get predictions
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y, y_pred)
        
        # Add cross-validation results
        metrics.update({
            'cv_rmse_mean': np.mean(cv_rmse),
            'cv_rmse_std': np.std(cv_rmse),
            'cv_r2_mean': np.mean(cross_val_score(model.model, X, y, cv=self.cv_folds, scoring='r2')),
            'cv_r2_std': np.std(cross_val_score(model.model, X, y, cv=self.cv_folds, scoring='r2'))
        })
        
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'predictions': y_pred,
            'model': model
        }
        
        self.results[model_name] = results
        return results
    
    def compare_models(self, models: List[Any], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compare multiple models and return leaderboard.
        
        Args:
            models: List of models to compare
            X: Features
            y: Target
            
        Returns:
            DataFrame with model comparison results
        """
        logger.info(f"Comparing {len(models)} models")
        
        comparison_results = []
        
        for model in models:
            results = self.evaluate_model(model, X, y)
            comparison_results.append({
                'Model': results['model_name'],
                'RMSE': results['metrics']['rmse'],
                'MAE': results['metrics']['mae'],
                'R²': results['metrics']['r2'],
                'MAPE (%)': results['metrics']['mape'],
                'SMAPE (%)': results['metrics']['smape'],
                'CV RMSE Mean': results['metrics']['cv_rmse_mean'],
                'CV RMSE Std': results['metrics']['cv_rmse_std'],
                'CV R² Mean': results['metrics']['cv_r2_mean'],
                'CV R² Std': results['metrics']['cv_r2_std'],
                'Bias': results['metrics']['bias'],
                'Correlation': results['metrics']['correlation']
            })
        
        leaderboard = pd.DataFrame(comparison_results)
        leaderboard = leaderboard.sort_values('RMSE').reset_index(drop=True)
        
        return leaderboard
    
    def plot_predictions(self, model_name: str, y_true: np.ndarray, 
                        y_pred: np.ndarray, save_path: str = None) -> None:
        """Plot actual vs predicted values.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Yield (tons/hectare)')
        plt.ylabel('Predicted Yield (tons/hectare)')
        plt.title(f'{model_name}: Actual vs Predicted')
        
        # Residuals plot
        plt.subplot(2, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Yield (tons/hectare)')
        plt.ylabel('Residuals')
        plt.title(f'{model_name}: Residuals Plot')
        
        # Distribution of residuals
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title(f'{model_name}: Residuals Distribution')
        
        # Q-Q plot
        plt.subplot(2, 2, 4)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'{model_name}: Q-Q Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, model_name: str, feature_names: List[str], 
                               importance_values: np.ndarray, top_n: int = 20,
                               save_path: str = None) -> None:
        """Plot feature importance for tree-based models.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            importance_values: Feature importance values
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        # Get top features
        top_indices = np.argsort(importance_values)[-top_n:]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance_values[top_indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name}: Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, leaderboard: pd.DataFrame, 
                            metric: str = 'RMSE', save_path: str = None) -> None:
        """Plot model comparison chart.
        
        Args:
            leaderboard: Model comparison DataFrame
            metric: Metric to plot
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Sort by metric
        sorted_df = leaderboard.sort_values(metric)
        
        plt.bar(range(len(sorted_df)), sorted_df[metric])
        plt.xticks(range(len(sorted_df)), sorted_df['Model'], rotation=45, ha='right')
        plt.ylabel(metric)
        plt.title(f'Model Comparison: {metric}')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(sorted_df[metric]):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, leaderboard: pd.DataFrame, 
                       output_path: str = None) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            leaderboard: Model comparison DataFrame
            output_path: Path to save the report
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("CROP YIELD PREDICTION - MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Best model
        best_model = leaderboard.iloc[0]
        report.append(f"BEST MODEL: {best_model['Model']}")
        report.append(f"RMSE: {best_model['RMSE']:.4f}")
        report.append(f"R²: {best_model['R²']:.4f}")
        report.append(f"MAE: {best_model['MAE']:.4f}")
        report.append("")
        
        # Full leaderboard
        report.append("FULL LEADERBOARD:")
        report.append("-" * 40)
        report.append(leaderboard.to_string(index=False, float_format='%.4f'))
        report.append("")
        
        # Model insights
        report.append("MODEL INSIGHTS:")
        report.append("-" * 40)
        
        # Best performing models by different metrics
        best_rmse = leaderboard.loc[leaderboard['RMSE'].idxmin()]
        best_r2 = leaderboard.loc[leaderboard['R²'].idxmax()]
        best_mae = leaderboard.loc[leaderboard['MAE'].idxmin()]
        
        report.append(f"• Best RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")
        report.append(f"• Best R²: {best_r2['Model']} ({best_r2['R²']:.4f})")
        report.append(f"• Best MAE: {best_mae['Model']} ({best_mae['MAE']:.4f})")
        report.append("")
        
        # Performance ranges
        report.append("PERFORMANCE RANGES:")
        report.append(f"• RMSE: {leaderboard['RMSE'].min():.4f} - {leaderboard['RMSE'].max():.4f}")
        report.append(f"• R²: {leaderboard['R²'].min():.4f} - {leaderboard['R²'].max():.4f}")
        report.append(f"• MAE: {leaderboard['MAE'].min():.4f} - {leaderboard['MAE'].max():.4f}")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text
