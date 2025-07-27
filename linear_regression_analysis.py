"""
Simple Linear Regression Analysis: Sales vs Advertising
=======================================================

This script implements a Simple Linear Regression model to study the linear 
relationship between Sales and Advertising for a dietary weight control product.

Implementation Steps:
1. Import required libraries
2. Load and explore the dataset
3. Split the dataset into training and testing sets
4. Train the model
5. Make predictions
6. Evaluate the model

Author: Your Name
Date: July 26, 2025
"""

# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_samples=100):
    """
    Generate sample Sales vs Advertising data for a dietary weight control product.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing Advertising and Sales data
    """
    # Generate advertising spend data (in thousands)
    advertising = np.random.uniform(10, 100, n_samples)
    
    # Generate sales data with linear relationship + noise
    # Assuming: Sales = 2.5 * Advertising + 15 + noise
    sales = 2.5 * advertising + 15 + np.random.normal(0, 8, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Advertising': advertising,
        'Sales': sales
    })
    
    return data

def explore_data(data):
    """
    Explore and visualize the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to explore
    """
    print("=== Dataset Exploration ===")
    print(f"Dataset shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())
    
    print("\nDataset summary statistics:")
    print(data.describe())
    
    print("\nDataset info:")
    print(data.info())
    
    print("\nChecking for missing values:")
    print(data.isnull().sum())
    
    # Correlation analysis
    correlation = data.corr()
    print(f"\nCorrelation between Advertising and Sales: {correlation.iloc[0,1]:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot
    axes[0, 0].scatter(data['Advertising'], data['Sales'], alpha=0.6, color='blue')
    axes[0, 0].set_xlabel('Advertising Spend (in thousands)')
    axes[0, 0].set_ylabel('Sales (in thousands)')
    axes[0, 0].set_title('Sales vs Advertising Spend')
    axes[0, 0].grid(True)
    
    # Distribution plots
    axes[0, 1].hist(data['Advertising'], bins=20, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Advertising Spend')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Advertising Spend')
    axes[0, 1].grid(True)
    
    axes[1, 0].hist(data['Sales'], bins=20, alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Sales')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Sales')
    axes[1, 0].grid(True)
    
    # Correlation heatmap
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.show()

def train_linear_regression(X_train, y_train):
    """
    Train a Simple Linear Regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
        
    Returns:
    --------
    LinearRegression
        Trained linear regression model
    """
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("=== Model Training Complete ===")
    print(f"Coefficient (slope): {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Linear equation: Sales = {model.coef_[0]:.4f} * Advertising + {model.intercept_:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """
    Evaluate the linear regression model.
    
    Parameters:
    -----------
    model : LinearRegression
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test target values
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print("=== Model Evaluation ===")
    print("Training Set Performance:")
    print(f"  Mean Squared Error: {train_mse:.4f}")
    print(f"  Root Mean Squared Error: {np.sqrt(train_mse):.4f}")
    print(f"  Mean Absolute Error: {train_mae:.4f}")
    print(f"  R² Score: {train_r2:.4f}")
    
    print("\nTest Set Performance:")
    print(f"  Mean Squared Error: {test_mse:.4f}")
    print(f"  Root Mean Squared Error: {np.sqrt(test_mse):.4f}")
    print(f"  Mean Absolute Error: {test_mae:.4f}")
    print(f"  R² Score: {test_r2:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Regression line plot
    X_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
    y_range_pred = model.predict(X_range)
    
    axes[0].scatter(X_test, y_test, alpha=0.6, color='blue', label='Actual')
    axes[0].plot(X_range, y_range_pred, color='red', linewidth=2, label='Regression Line')
    axes[0].set_xlabel('Advertising Spend')
    axes[0].set_ylabel('Sales')
    axes[0].set_title('Linear Regression: Test Data')
    axes[0].legend()
    axes[0].grid(True)
    
    # Predicted vs Actual
    axes[1].scatter(y_test, y_test_pred, alpha=0.6)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    axes[1].set_xlabel('Actual Sales')
    axes[1].set_ylabel('Predicted Sales')
    axes[1].set_title('Predicted vs Actual Sales')
    axes[1].grid(True)
    
    # Residuals plot
    residuals = y_test - y_test_pred
    axes[2].scatter(y_test_pred, residuals, alpha=0.6)
    axes[2].axhline(y=0, color='red', linestyle='--')
    axes[2].set_xlabel('Predicted Sales')
    axes[2].set_ylabel('Residuals')
    axes[2].set_title('Residual Plot')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return y_test_pred

def main():
    """
    Main function to run the complete linear regression analysis.
    """
    print("Simple Linear Regression: Sales vs Advertising Analysis")
    print("=" * 60)
    
    # Step 2: Load and explore the dataset
    print("\nStep 1: Generating sample data...")
    data = generate_sample_data(n_samples=100)
    
    print("\nStep 2: Exploring the dataset...")
    explore_data(data)
    
    # Step 3: Split the dataset into training and testing sets
    print("\nStep 3: Splitting data into train and test sets...")
    X = data[['Advertising']]  # Features (must be 2D for sklearn)
    y = data['Sales']          # Target variable
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    # Step 4: Train the model
    print("\nStep 4: Training the Linear Regression model...")
    model = train_linear_regression(X_train, y_train)
    
    # Step 5: Make predictions and Step 6: Evaluate the model
    print("\nStep 5 & 6: Making predictions and evaluating the model...")
    predictions = evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Additional analysis
    print("\n=== Business Insights ===")
    coef = model.coef_[0]
    intercept = model.intercept_
    
    print(f"For every $1,000 increase in advertising spend, sales increase by approximately ${coef*1000:.0f}")
    print(f"Base sales (with zero advertising) would be approximately ${intercept*1000:.0f}")
    
    # Example prediction
    example_ad_spend = 50  # $50,000
    example_prediction = model.predict([[example_ad_spend]])[0]
    print(f"\nExample: With ${example_ad_spend*1000:.0f} advertising spend, predicted sales: ${example_prediction*1000:.0f}")

if __name__ == "__main__":
    main()
