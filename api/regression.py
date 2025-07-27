"""
Vercel Serverless Function for Linear Regression Analysis
=========================================================

This function provides API endpoints for the linear regression model
that can be deployed to Vercel as serverless functions.
"""

from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # Parse query parameters
        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        
        # Get parameters with defaults
        n_samples = int(query_params.get('samples', [100])[0])
        noise_level = float(query_params.get('noise', [8])[0])
        test_size = float(query_params.get('test_size', [0.2])[0])
        seed = int(query_params.get('seed', [42])[0])
        
        try:
            # Generate data and train model
            result = self.analyze_regression(n_samples, noise_level, test_size, seed)
            
            # Send response
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            error_response = {"error": str(e)}
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_POST(self):
        """Handle POST requests with custom data"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            # Read POST data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Extract data
            advertising = np.array(data['advertising'])
            sales = np.array(data['sales'])
            test_size = data.get('test_size', 0.2)
            
            # Analyze custom data
            result = self.analyze_custom_data(advertising, sales, test_size)
            
            # Send response
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            error_response = {"error": str(e)}
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def analyze_regression(self, n_samples, noise_level, test_size, seed):
        """Perform regression analysis with generated data"""
        np.random.seed(seed)
        
        # Generate data
        advertising = np.random.uniform(10, 100, n_samples)
        sales = 2.5 * advertising + 15 + np.random.normal(0, noise_level, n_samples)
        
        return self.analyze_custom_data(advertising, sales, test_size)
    
    def analyze_custom_data(self, advertising, sales, test_size):
        """Analyze regression with custom data"""
        # Prepare data
        X = advertising.reshape(-1, 1)
        y = sales
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = {
            'mse': float(mean_squared_error(y_train, y_pred_train)),
            'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            'mae': float(mean_absolute_error(y_train, y_pred_train)),
            'r2': float(r2_score(y_train, y_pred_train))
        }
        
        test_metrics = {
            'mse': float(mean_squared_error(y_test, y_pred_test)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            'mae': float(mean_absolute_error(y_test, y_pred_test)),
            'r2': float(r2_score(y_test, y_pred_test))
        }
        
        # Model parameters
        model_params = {
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'equation': f"Sales = {model.coef_[0]:.4f} * Advertising + {model.intercept_:.4f}"
        }
        
        # Prepare data for visualization
        data_points = {
            'advertising': advertising.tolist(),
            'sales': sales.tolist(),
            'train_indices': list(range(len(X_train))),
            'test_indices': list(range(len(X_train), len(X_train) + len(X_test)))
        }
        
        # Generate regression line points
        x_line = np.linspace(advertising.min(), advertising.max(), 50)
        y_line = model.predict(x_line.reshape(-1, 1))
        
        regression_line = {
            'x': x_line.tolist(),
            'y': y_line.tolist()
        }
        
        return {
            'success': True,
            'model_params': model_params,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'data_points': data_points,
            'regression_line': regression_line,
            'predictions': {
                'train': y_pred_train.tolist(),
                'test': y_pred_test.tolist()
            }
        }
