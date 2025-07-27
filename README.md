# 📊 Simple Linear Regression: Sales vs Advertising Analysis

![Python](https://img.shields.io/badge/python-v3.8+-blu## 🎓 Learning Objectives

This project helps you understand:vg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive Python implementation of Simple Linear Regression to study the linear relationship between Sales and Advertising data for a dietary weight control product using Scikit-learn.

## 🎯 Project Overview

This project demonstrates the complete machine learning workflow for linear regression analysis, including:
- Data generation and exploration
- Model training and evaluation
- Visualization of results
- Business insights extraction

## 🔧 Technologies Used

- **Python 3.8+**
- **Scikit-learn**: Machine learning library for linear regression
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization

## ✨ Features

- 📈 **Complete ML Pipeline**: From data generation to model evaluation
- 🔍 **Data Exploration**: Comprehensive statistical analysis and visualization
- 🎯 **Model Training**: Simple Linear Regression implementation
- 📊 **Performance Metrics**: MSE, RMSE, MAE, and R² evaluation
- 📋 **Business Insights**: Practical interpretation of results
- 🎨 **Rich Visualizations**: Multiple plots for data understanding
- 📝 **Well-documented Code**: Clear comments and explanations

## 📁 Project Structure

```
ML/
├── .vscode/
│   └── tasks.json
├── linear_regression_analysis.py    # Main implementation
├── linear_regression_workspace_setup.ipynb  # Interactive notebook
├── requirements.txt                 # Project dependencies
├── LICENSE                          # MIT License
└── README.md                       # Project documentation
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Installation

1. **Clone or download the project** (if not already done)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**:
   ```bash
   python linear_regression_analysis.py
   ```

## 📊 What the Script Does

### 1. Import Required Libraries
- Imports all necessary libraries for data analysis and machine learning

### 2. Load and Explore the Dataset
- Generates synthetic sales and advertising data
- Displays dataset statistics and information
- Visualizes data distributions and relationships

### 3. Split the Dataset
- Divides data into training (80%) and testing (20%) sets
- Ensures reproducible results with random seed

### 4. Train the Model
- Creates and trains a Simple Linear Regression model
- Displays the linear equation coefficients

### 5. Make Predictions
- Uses the trained model to predict sales on test data

### 6. Evaluate the Model
- Calculates performance metrics (MSE, RMSE, MAE, R²)
- Visualizes regression line, predictions vs actual, and residuals
- Provides business insights

## 📈 Expected Output

The script will display:
- Dataset exploration statistics
- Model coefficients and linear equation
- Performance metrics for both training and test sets
- Multiple visualization plots
- Business insights and example predictions

## 🔍 Key Features

- **Comprehensive Analysis**: Complete ML workflow from data to insights
- **Visualization**: Multiple plots for data understanding
- **Model Evaluation**: Various metrics to assess model performance
- **Business Context**: Practical insights for decision-making
- **Clean Code**: Well-documented and modular implementation

## 📊 Sample Outputs

The model will provide insights such as:
- Linear relationship equation: `Sales = 2.5 * Advertising + 15`
- R² score indicating model fit quality
- Predictions for new advertising spend amounts
- Business recommendations based on the analysis

## � Learning Objectives

This project helps you understand:
- Linear regression fundamentals
- Scikit-learn implementation
- Data preprocessing and exploration
- Model evaluation techniques
- Python data science workflow

## 🔧 Customization

You can modify the script to:
- Use real dataset by replacing the `generate_sample_data()` function
- Adjust model parameters
- Add more sophisticated visualizations
- Include additional evaluation metrics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

If you have any questions or suggestions, feel free to reach out or open an issue!

---
⭐ **Don't forget to star this repo if you found it helpful!** ⭐
