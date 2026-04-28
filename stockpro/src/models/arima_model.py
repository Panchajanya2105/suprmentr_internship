"""
ARIMA/SARIMA Model for Stock Price Prediction
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pmdarima as pm
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ARIMAPredictor:
    """ARIMA/SARIMA-based stock price predictor"""
    
    def __init__(self, seasonal=False):
        """
        Initialize ARIMA predictor
        
        Args:
            seasonal: Whether to use SARIMA (True) or ARIMA (False)
        """
        self.seasonal = seasonal
        self.model = None
        self.order = None
        self.seasonal_order = None
        self.trained = False
        
    def check_stationarity(self, data):
        """Check if data is stationary using ADF test"""
        result = adfuller(data)
        return {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4],
            'Is Stationary': result[1] < 0.05
        }
    
    def auto_arima(self, data, seasonal=False, max_p=5, max_q=5, max_P=2, max_Q=2):
        """Automatically find best ARIMA parameters"""
        model = pm.auto_arima(
            data,
            start_p=1, start_q=1,
            max_p=max_p, max_q=max_q,
            max_P=max_P, max_Q=max_Q,
            m=12 if seasonal else 1,
            seasonal=seasonal,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        self.order = model.order
        if seasonal:
            self.seasonal_order = model.seasonal_order
        
        return model.order, model.seasonal_order if seasonal else None
    
    def train(self, data, order=None, seasonal_order=None):
        """
        Train ARIMA/SARIMA model
        
        Args:
            data: Time series data (pandas Series or array)
            order: (p, d, q) tuple for ARIMA
            seasonal_order: (P, D, Q, s) tuple for SARIMA
        """
        # If order not provided, find best automatically
        if order is None:
            if self.seasonal:
                order, seasonal_order = self.auto_arima(data, seasonal=True)
            else:
                order, _ = self.auto_arima(data, seasonal=False)
        
        self.order = order
        self.seasonal_order = seasonal_order
        
        # Train model
        if self.seasonal and seasonal_order:
            self.model = SARIMAX(
                data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            self.model = ARIMA(data, order=order)
        
        self.model = self.model.fit()
        self.trained = True
    
    def predict(self, data, days_ahead=30):
        """
        Make predictions
        
        Args:
            data: Historical data
            days_ahead: Number of days to predict
        """
        if not self.trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get predictions
        forecast = self.model.forecast(steps=days_ahead)
        
        # Get confidence intervals
        forecast_result = self.model.get_forecast(steps=days_ahead)
        confidence_intervals = forecast_result.conf_int()
        
        return forecast, confidence_intervals
    
    def evaluate(self, data, test_size=0.2):
        """Evaluate model performance"""
        if not self.trained:
            raise ValueError("Model not trained yet.")
        
        # Split data
        split_idx = int(len(data) * (1 - test_size))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        # Train on training data only
        if self.seasonal:
            eval_model = SARIMAX(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            eval_model = ARIMA(train_data, order=self.order)
        
        eval_model = eval_model.fit()
        
        # Predict test period
        predictions = eval_model.forecast(steps=len(test_data))
        
        # Calculate metrics
        mse = mean_squared_error(test_data, predictions)
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mse)
        
        # Handle R2 score
        try:
            r2 = r2_score(test_data, predictions)
        except:
            r2 = 0
        
        # MAPE
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def get_model_summary(self):
        """Get model summary statistics"""
        if not self.trained:
            raise ValueError("Model not trained yet.")
        
        return self.model.summary()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.trained:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'seasonal': self.seasonal
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.order = saved_data['order']
        self.seasonal_order = saved_data['seasonal_order']
        self.seasonal = saved_data['seasonal']
        self.trained = True