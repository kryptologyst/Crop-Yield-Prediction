"""Interactive Streamlit demo for crop yield prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.generator import CropYieldDataGenerator
from features.engineering import FeatureEngineer
from models.regression import get_default_models
from eval.metrics import ModelEvaluator
from viz.plots import CropYieldVisualizer
from configs.config import demo_config, data_config, model_config

# Page configuration
st.set_page_config(
    page_title=demo_config['app_title'],
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🌾 Crop Yield Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown(f"**{demo_config['app_description']}**")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Data generation controls
        st.subheader("📊 Data Generation")
        n_samples = st.slider("Number of samples", 100, 2000, 1000)
        seed = st.number_input("Random seed", value=42, min_value=1, max_value=1000)
        
        if st.button("🔄 Generate Data", key="generate_data"):
            with st.spinner("Generating agricultural data..."):
                data_generator = CropYieldDataGenerator(seed=seed)
                X, y = data_generator.generate_dataset(n_samples=n_samples)
                X_spatial = data_generator.add_spatial_features(X)
                
                st.session_state.X = X_spatial
                st.session_state.y = y
                st.session_state.data_generated = True
                st.success(f"Generated {n_samples} samples!")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        model_names = [model.name for model in get_default_models()]
        selected_model = st.selectbox("Choose model", model_names, 
                                    index=model_names.index(demo_config['default_model']))
        
        if st.button("🚀 Train Model", key="train_model"):
            if st.session_state.data_generated:
                with st.spinner("Training model..."):
                    # Feature engineering
                    feature_engineer = FeatureEngineer()
                    X_processed = feature_engineer.fit_transform(st.session_state.X, st.session_state.y)
                    
                    # Get selected model
                    models = get_default_models()
                    selected_model_obj = next(model for model in models if model.name == selected_model)
                    
                    # Train model
                    selected_model_obj.fit(X_processed, st.session_state.y)
                    
                    # Make predictions
                    predictions = selected_model_obj.predict(X_processed)
                    
                    st.session_state.model = selected_model_obj
                    st.session_state.predictions = predictions
                    st.session_state.feature_engineer = feature_engineer
                    st.session_state.X_processed = X_processed
                    st.session_state.models_trained = True
                    st.success(f"{selected_model} trained successfully!")
            else:
                st.error("Please generate data first!")
    
    # Main content
    if st.session_state.data_generated:
        
        # Data overview
        st.header("📈 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(st.session_state.X))
        
        with col2:
            st.metric("Features", len(st.session_state.X.columns))
        
        with col3:
            st.metric("Avg Yield", f"{st.session_state.y.mean():.2f} tons/hectare")
        
        with col4:
            st.metric("Yield Range", f"{st.session_state.y.min():.1f} - {st.session_state.y.max():.1f}")
        
        # Data visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "🗺️ Spatial Map", "🤖 Model Results", "🔮 Predictions"])
        
        with tab1:
            st.subheader("Feature Distributions")
            
            # Feature distributions
            fig_dist = px.histogram(
                st.session_state.X, 
                x=st.session_state.X.columns[:6],  # Show first 6 features
                facet_col_wrap=3,
                title="Feature Distributions"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Correlation matrix
            st.subheader("Feature Correlations")
            corr_matrix = st.session_state.X.corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Yield vs key features
            st.subheader("Yield vs Key Features")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_soil = px.scatter(
                    st.session_state.X, 
                    x='soil_quality', 
                    y=st.session_state.y,
                    title="Yield vs Soil Quality",
                    labels={'soil_quality': 'Soil Quality', 'y': 'Yield (tons/hectare)'}
                )
                st.plotly_chart(fig_soil, use_container_width=True)
            
            with col2:
                fig_temp = px.scatter(
                    st.session_state.X, 
                    x='temperature', 
                    y=st.session_state.y,
                    title="Yield vs Temperature",
                    labels={'temperature': 'Temperature (°C)', 'y': 'Yield (tons/hectare)'}
                )
                st.plotly_chart(fig_temp, use_container_width=True)
        
        with tab2:
            st.subheader("Spatial Distribution of Crop Yields")
            
            # Create map
            m = folium.Map(
                location=[demo_config['map_center'][0], demo_config['map_center'][1]],
                zoom_start=demo_config['map_zoom'],
                tiles='OpenStreetMap'
            )
            
            # Add yield data
            for idx, row in st.session_state.X.iterrows():
                if idx < 100:  # Limit to 100 points for performance
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5,
                        popup=f"Yield: {st.session_state.y.iloc[idx]:.2f} tons/hectare",
                        color='blue',
                        fill=True,
                        fillOpacity=0.6
                    ).add_to(m)
            
            # Display map
            st_folium(m, width=700, height=500)
        
        with tab3:
            if st.session_state.models_trained:
                st.subheader(f"Model Performance: {selected_model}")
                
                # Calculate metrics
                evaluator = ModelEvaluator()
                metrics = evaluator.calculate_metrics(st.session_state.y.values, st.session_state.predictions)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("RMSE", f"{metrics['rmse']:.3f}")
                
                with col2:
                    st.metric("MAE", f"{metrics['mae']:.3f}")
                
                with col3:
                    st.metric("R²", f"{metrics['r2']:.3f}")
                
                with col4:
                    st.metric("MAPE", f"{metrics['mape']:.1f}%")
                
                # Prediction plots
                st.subheader("Prediction Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Actual vs Predicted
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=st.session_state.y.values,
                        y=st.session_state.predictions,
                        mode='markers',
                        name='Predictions',
                        opacity=0.6
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=[st.session_state.y.min(), st.session_state.y.max()],
                        y=[st.session_state.y.min(), st.session_state.y.max()],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red')
                    ))
                    fig_pred.update_layout(
                        title="Actual vs Predicted Yield",
                        xaxis_title="Actual Yield (tons/hectare)",
                        yaxis_title="Predicted Yield (tons/hectare)"
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                with col2:
                    # Residuals
                    residuals = st.session_state.y.values - st.session_state.predictions
                    fig_res = px.scatter(
                        x=st.session_state.predictions,
                        y=residuals,
                        title="Residuals Plot",
                        labels={'x': 'Predicted Yield', 'y': 'Residuals'}
                    )
                    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_res, use_container_width=True)
                
                # Feature importance (if available)
                if hasattr(st.session_state.model, 'get_feature_importance'):
                    try:
                        st.subheader("Feature Importance")
                        feature_importance = st.session_state.model.get_feature_importance()
                        
                        fig_imp = px.bar(
                            x=feature_importance.values,
                            y=feature_importance.index,
                            orientation='h',
                            title="Feature Importance",
                            labels={'x': 'Importance', 'y': 'Features'}
                        )
                        fig_imp.update_layout(height=600)
                        st.plotly_chart(fig_imp, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display feature importance: {e}")
            else:
                st.info("Please train a model to see results.")
        
        with tab4:
            st.subheader("🔮 Interactive Predictions")
            
            st.markdown("Adjust the parameters below to see predicted crop yield:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                soil_quality = st.slider("Soil Quality (1-10)", 1.0, 10.0, 7.0)
                rainfall = st.slider("Rainfall (mm/month)", 50.0, 600.0, 300.0)
                temperature = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)
                fertilizer_use = st.slider("Fertilizer Use (kg/hectare)", 0.0, 300.0, 150.0)
            
            with col2:
                pesticide_use = st.slider("Pesticide Use (liters/hectare)", 0.0, 3.0, 1.2)
                irrigation = st.slider("Irrigation (mm/month)", 0.0, 200.0, 100.0)
                planting_density = st.slider("Planting Density (plants/hectare)", 20000.0, 80000.0, 50000.0)
                field_slope = st.slider("Field Slope (degrees)", 0.0, 15.0, 2.0)
            
            if st.button("🔮 Predict Yield"):
                if st.session_state.models_trained:
                    # Create input data
                    input_data = pd.DataFrame({
                        'soil_quality': [soil_quality],
                        'rainfall': [rainfall],
                        'temperature': [temperature],
                        'fertilizer_use': [fertilizer_use],
                        'pesticide_use': [pesticide_use],
                        'irrigation': [irrigation],
                        'planting_density': [planting_density],
                        'field_slope': [field_slope],
                        'days_since_harvest': [30.0],  # Default value
                        'latitude': [40.0],  # Default value
                        'longitude': [-95.0],  # Default value
                        'climate_zone': [2],  # Default value
                        'elevation': [200.0]  # Default value
                    })
                    
                    # Process features
                    input_processed = st.session_state.feature_engineer.transform(input_data)
                    
                    # Make prediction
                    prediction = st.session_state.model.predict(input_processed)[0]
                    
                    # Display result
                    st.success(f"**Predicted Crop Yield: {prediction:.2f} tons/hectare**")
                    
                    # Add context
                    avg_yield = st.session_state.y.mean()
                    if prediction > avg_yield:
                        st.info(f"📈 Above average yield (+{prediction - avg_yield:.2f} tons/hectare)")
                    else:
                        st.warning(f"📉 Below average yield ({prediction - avg_yield:.2f} tons/hectare)")
                else:
                    st.error("Please train a model first!")
    
    else:
        st.info("👈 Please generate data using the sidebar controls to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌾 Crop Yield Prediction Dashboard | Environmental & Social Applications</p>
        <p>Author: <a href='https://github.com/kryptologyst' target='_blank'>kryptologyst</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
