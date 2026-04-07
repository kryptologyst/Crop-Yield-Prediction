"""Visualization utilities for crop yield prediction."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CropYieldVisualizer:
    """Comprehensive visualization tools for crop yield prediction."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)) -> None:
        """Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        
    def plot_feature_distributions(self, df: pd.DataFrame, save_path: str = None) -> None:
        """Plot distributions of all features.
        
        Args:
            df: DataFrame with features
            save_path: Path to save the plot
        """
        logger.info("Plotting feature distributions")
        
        n_features = len(df.columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], n_rows * 3))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(df.columns):
            if i < len(axes):
                axes[i].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{col}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save_path: str = None) -> None:
        """Plot correlation matrix heatmap.
        
        Args:
            df: DataFrame with features
            save_path: Path to save the plot
        """
        logger.info("Plotting correlation matrix")
        
        plt.figure(figsize=(self.figsize[0], self.figsize[0]))
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8}
        )
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_yield_vs_features(self, df: pd.DataFrame, target_col: str = 'crop_yield', 
                              save_path: str = None) -> None:
        """Plot yield vs individual features.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            save_path: Path to save the plot
        """
        logger.info("Plotting yield vs features")
        
        feature_cols = [col for col in df.columns if col != target_col]
        n_features = len(feature_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], n_rows * 3))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(feature_cols):
            if i < len(axes):
                axes[i].scatter(df[col], df[target_col], alpha=0.6)
                axes[i].set_xlabel(col)
                axes[i].set_ylabel(target_col)
                axes[i].set_title(f'{target_col} vs {col}')
                axes[i].grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(df[col], df[target_col], 1)
                p = np.poly1d(z)
                axes[i].plot(df[col], p(df[col]), "r--", alpha=0.8)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_map(self, df: pd.DataFrame, target_col: str = 'crop_yield',
                             center_lat: float = 40.0, center_lon: float = -95.0,
                             zoom: int = 4) -> folium.Map:
        """Create interactive map with yield data.
        
        Args:
            df: DataFrame with spatial data
            target_col: Name of target column
            center_lat: Map center latitude
            center_lon: Map center longitude
            zoom: Map zoom level
            
        Returns:
            Folium map object
        """
        logger.info("Creating interactive map")
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='OpenStreetMap'
        )
        
        # Add yield data as circles
        for idx, row in df.iterrows():
            if 'latitude' in row and 'longitude' in row:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    popup=f"Yield: {row[target_col]:.2f} tons/hectare",
                    color='blue',
                    fill=True,
                    fillOpacity=0.6
                ).add_to(m)
        
        return m
    
    def plot_prediction_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = "Model", save_path: str = None) -> None:
        """Comprehensive prediction analysis plots.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save the plot
        """
        logger.info(f"Creating prediction analysis for {model_name}")
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Yield')
        axes[0, 0].set_ylabel('Predicted Yield')
        axes[0, 0].set_title(f'{model_name}: Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Yield')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title(f'{model_name}: Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'{model_name}: Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'{model_name}: Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], importance_values: np.ndarray,
                               model_name: str = "Model", top_n: int = 20, 
                               save_path: str = None) -> None:
        """Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importance_values: Feature importance values
            model_name: Name of the model
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        logger.info(f"Plotting feature importance for {model_name}")
        
        # Get top features
        top_indices = np.argsort(importance_values)[-top_n:]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance_values[top_indices]
        
        plt.figure(figsize=(self.figsize[0], max(6, top_n * 0.3)))
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name}: Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, df: pd.DataFrame, predictions: np.ndarray = None,
                                   model_name: str = "Model") -> go.Figure:
        """Create interactive Plotly dashboard.
        
        Args:
            df: DataFrame with features
            predictions: Model predictions (optional)
            model_name: Name of the model
            
        Returns:
            Plotly figure
        """
        logger.info(f"Creating interactive dashboard for {model_name}")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Yield Distribution', 'Feature Correlations', 
                          'Yield vs Soil Quality', 'Yield vs Temperature'),
            specs=[[{"type": "histogram"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Yield distribution
        fig.add_trace(
            go.Histogram(x=df['crop_yield'], name='Yield Distribution'),
            row=1, col=1
        )
        
        # Feature correlations
        corr_matrix = df[['soil_quality', 'rainfall', 'temperature', 'fertilizer_use', 'crop_yield']].corr()
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, 
                      x=corr_matrix.columns, 
                      y=corr_matrix.columns,
                      colorscale='RdBu'),
            row=1, col=2
        )
        
        # Yield vs Soil Quality
        fig.add_trace(
            go.Scatter(x=df['soil_quality'], y=df['crop_yield'], 
                      mode='markers', name='Yield vs Soil Quality'),
            row=2, col=1
        )
        
        # Yield vs Temperature
        fig.add_trace(
            go.Scatter(x=df['temperature'], y=df['crop_yield'], 
                      mode='markers', name='Yield vs Temperature'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'{model_name} - Crop Yield Analysis Dashboard',
            showlegend=False,
            height=800
        )
        
        return fig
    
    def plot_model_comparison(self, leaderboard: pd.DataFrame, metric: str = 'RMSE',
                             save_path: str = None) -> None:
        """Plot model comparison chart.
        
        Args:
            leaderboard: Model comparison DataFrame
            metric: Metric to plot
            save_path: Path to save the plot
        """
        logger.info(f"Plotting model comparison for {metric}")
        
        plt.figure(figsize=self.figsize)
        
        # Sort by metric
        sorted_df = leaderboard.sort_values(metric)
        
        bars = plt.bar(range(len(sorted_df)), sorted_df[metric])
        plt.xticks(range(len(sorted_df)), sorted_df['Model'], rotation=45, ha='right')
        plt.ylabel(metric)
        plt.title(f'Model Comparison: {metric}')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(sorted_df[metric]):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Color bars by performance (better = darker)
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_time_series_analysis(self, df: pd.DataFrame, time_col: str = 'days_since_harvest',
                                 target_col: str = 'crop_yield', save_path: str = None) -> None:
        """Plot time series analysis.
        
        Args:
            df: DataFrame with time series data
            time_col: Name of time column
            target_col: Name of target column
            save_path: Path to save the plot
        """
        logger.info("Plotting time series analysis")
        
        plt.figure(figsize=self.figsize)
        
        # Sort by time
        df_sorted = df.sort_values(time_col)
        
        plt.scatter(df_sorted[time_col], df_sorted[target_col], alpha=0.6)
        plt.xlabel(f'{time_col} (days)')
        plt.ylabel(f'{target_col} (tons/hectare)')
        plt.title(f'{target_col} vs {time_col}')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df_sorted[time_col], df_sorted[target_col], 1)
        p = np.poly1d(z)
        plt.plot(df_sorted[time_col], p(df_sorted[time_col]), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
