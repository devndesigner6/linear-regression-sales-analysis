"""
Streamlit Web Application for Linear Regression Analysis
========================================================

A web-based interface for the Sales vs Advertising Linear Regression model.
This can be deployed to Streamlit Cloud, Heroku, or other platforms.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Linear Regression Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data(n_samples=100, noise_level=8, seed=42):
    """Generate sample Sales vs Advertising data."""
    np.random.seed(seed)
    
    # Generate advertising spend data (in thousands)
    advertising = np.random.uniform(10, 100, n_samples)
    
    # Generate sales data with linear relationship + noise
    sales = 2.5 * advertising + 15 + np.random.normal(0, noise_level, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Advertising': advertising,
        'Sales': sales
    })
    
    return data

def train_model(data, test_size=0.2, random_state=42):
    """Train the linear regression model."""
    X = data[['Advertising']]
    y = data['Sales']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return mse, rmse, mae, r2

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Linear Regression: Sales vs Advertising</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This interactive web application demonstrates Simple Linear Regression analysis 
    for studying the relationship between advertising spend and sales for a dietary weight control product.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("🔧 Model Parameters")
    
    n_samples = st.sidebar.slider("Number of Samples", 50, 500, 100, 10)
    noise_level = st.sidebar.slider("Noise Level", 1, 20, 8, 1)
    test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20, 5) / 100
    random_seed = st.sidebar.number_input("Random Seed", 1, 1000, 42)
    
    # Generate data
    data = generate_sample_data(n_samples, noise_level, random_seed)
    
    # Train model
    model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = train_model(
        data, test_size, random_seed
    )
    
    # Calculate metrics
    train_mse, train_rmse, train_mae, train_r2 = calculate_metrics(y_train, y_pred_train)
    test_mse, test_rmse, test_mae, test_r2 = calculate_metrics(y_test, y_pred_test)
    
    # Display results in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Data Visualization")
        
        # Create scatter plot with regression line
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=data['Advertising'], 
            y=data['Sales'],
            mode='markers',
            name='Data Points',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))
        
        # Add regression line
        x_line = np.linspace(data['Advertising'].min(), data['Advertising'].max(), 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        
        fig.add_trace(go.Scatter(
            x=x_line, 
            y=y_line,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Sales vs Advertising with Regression Line",
            xaxis_title="Advertising Spend (thousands)",
            yaxis_title="Sales (thousands)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data summary
        st.subheader("📊 Data Summary")
        st.dataframe(data.describe())
        
    with col2:
        st.subheader("🎯 Model Performance")
        
        # Performance metrics
        col_train, col_test = st.columns(2)
        
        with col_train:
            st.markdown("**Training Set**")
            st.metric("R² Score", f"{train_r2:.4f}")
            st.metric("RMSE", f"{train_rmse:.2f}")
            st.metric("MAE", f"{train_mae:.2f}")
            
        with col_test:
            st.markdown("**Test Set**")
            st.metric("R² Score", f"{test_r2:.4f}")
            st.metric("RMSE", f"{test_rmse:.2f}")
            st.metric("MAE", f"{test_mae:.2f}")
        
        # Model coefficients
        st.subheader("🔍 Model Details")
        st.write(f"**Slope (β₁):** {model.coef_[0]:.4f}")
        st.write(f"**Intercept (β₀):** {model.intercept_:.4f}")
        st.write(f"**Equation:** Sales = {model.coef_[0]:.4f} × Advertising + {model.intercept_:.4f}")
        
        # Business insights
        st.subheader("💡 Business Insights")
        if model.coef_[0] > 0:
            st.success(f"✅ For every $1,000 increase in advertising spend, sales increase by ${model.coef_[0]*1000:.0f}")
        
        if test_r2 > 0.7:
            st.success(f"✅ Strong relationship (R² = {test_r2:.3f})")
        elif test_r2 > 0.5:
            st.warning(f"⚠️ Moderate relationship (R² = {test_r2:.3f})")
        else:
            st.error(f"❌ Weak relationship (R² = {test_r2:.3f})")
    
    # Additional visualizations
    st.subheader("📈 Additional Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Residuals plot
        residuals = y_test - y_pred_test
        fig_residuals = px.scatter(
            x=y_pred_test, 
            y=residuals,
            title="Residuals vs Predicted Values",
            labels={'x': 'Predicted Sales', 'y': 'Residuals'}
        )
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    with col4:
        # Distribution comparison
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=y_test, name="Actual Sales", opacity=0.7))
        fig_dist.add_trace(go.Histogram(x=y_pred_test, name="Predicted Sales", opacity=0.7))
        fig_dist.update_layout(
            title="Distribution: Actual vs Predicted Sales",
            xaxis_title="Sales (thousands)",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Data table
    if st.checkbox("Show Raw Data"):
        st.subheader("📋 Raw Data")
        st.dataframe(data)

if __name__ == "__main__":
    main()
