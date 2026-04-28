"""
Prophet Model for Stock Price Prediction
Handles holidays compatibility issues
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Patch holidays before importing prophet
import sys
import importlib

# Block problematic holidays imports
class DummyHolidays:
    """Dummy holidays class to avoid import errors"""
    def __init__(self, *args, **kwargs):
        pass
    def get(self, *args, **kwargs):
        return {}

# Patch the module
if 'holidays' not in sys.modules:
    sys.modules['holidays'] = type(sys)('holidays')
    sys.modules['holidays'].__dict__['country_holidays'] = DummyHolidays

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception as e:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except Exception as e2:
        PROPHET_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class ProphetPredictor:
    """Facebook Prophet-based stock price predictor"""
    
    def __init__(self):
        self.model = None
        self.trained = False
        
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Run: pip install prophet")
    
    def prepare_data(self, data):
        """Prepare data in Prophet format (ds, y)"""
        df = data.reset_index()
        
        # Handle different index names
        if 'Date' in df.columns:
            df['ds'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        else:
            date_col = df.columns[0]
            df['ds'] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        
        df['y'] = df['Close'].astype(float)
        df = df[['ds', 'y']].dropna()
        
        return df
    
    def train(self, data, **kwargs):
        """Train Prophet model"""
        train_df = self.prepare_data(data)
        
        if len(train_df) < 10:
            raise ValueError("Need at least 10 data points")
        
        # Simple parameters to avoid issues
        params = {
            'yearly_seasonality': False,
            'weekly_seasonality': False,
            'daily_seasonality': False,
        }
        params.update(kwargs)
        
        try:
            self.model = Prophet(**params)
            self.model.fit(train_df)
            self.trained = True
        except Exception as e:
            raise Exception(f"Training failed: {e}")
    
    def predict(self, data, days_ahead=30, include_history=False):
        """Make predictions"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        future = self.model.make_future_dataframe(
            periods=days_ahead,
            freq='D',
            include_history=include_history
        )
        
        forecast = self.model.predict(future)
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_ahead)
        
        return predictions, forecast, {}
    
    def save_model(self, filepath):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load model"""
        self.model = joblib.load(filepath)
        self.trained = True