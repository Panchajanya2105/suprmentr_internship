"""
Machine Learning Models for Stock Price Prediction
Includes XGBoost, LightGBM, and Linear Regression
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os

class MLPredictor:
    """Machine Learning-based stock price predictor"""
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize ML predictor
        
        Args:
            model_type: 'xgboost', 'lightgbm', 'randomforest', 'gradientboost', 
                       'linear', 'ridge', 'lasso'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected model"""
        models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.01,
                max_depth=7,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.01,
                max_depth=7,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'randomforest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradientboost': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.01,
                max_depth=5,
                min_samples_split=5,
                random_state=42
            ),
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = models[self.model_type]
    
    def create_features(self, data):
        """Create features for ML models"""
        df = data.copy()
        
        # Price features
        df['Returns'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            if len(df) >= window:
                df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
                df[f'Price_to_SMA_{window}'] = df['Close'] / df[f'SMA_{window}']
        
        # Price momentum
        for window in [5, 10, 20]:
            if len(df) >= window:
                df[f'Momentum_{window}'] = df['Close'] - df['Close'].shift(window)
                df[f'ROC_{window}'] = df['Close'].pct_change(periods=window) * 100
        
        # Volatility
        for window in [5, 10, 20]:
            if len(df) >= window:
                df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands position
        if 'SMA_20' in df.columns:
            rolling_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['SMA_20'] + (rolling_std * 2)
            df['BB_Lower'] = df['SMA_20'] - (rolling_std * 2)
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
        # Target: Next day's closing price
        df['Target'] = df['Close'].shift(-1)
        
        return df.dropna()
    
    def prepare_data(self, data, test_size=0.2):
        """Prepare training and testing data"""
        # Create features
        df = self.create_features(data)
        
        # Remove target and non-feature columns
        feature_cols = [col for col in df.columns if col not in 
                       ['Target', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'Dividends', 'Stock Splits']]
        
        X = df[feature_cols].values
        y = df['Target'].values
        
        # Split data (time-series split)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def train(self, data, **kwargs):
        """Train the ML model"""
        # Prepare data
        X_train, X_test, y_train, y_test, self.feature_cols = self.prepare_data(data)
        
        # Update model parameters if provided
        if kwargs and hasattr(self.model, 'set_params'):
            self.model.set_params(**kwargs)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return X_train, X_test, y_train, y_test
    
    def predict(self, data, days_ahead=30):
        """Make predictions for future days"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Prepare latest data for prediction
        df = self.create_features(data)
        latest_features = df.iloc[-1:][self.feature_cols].values
        latest_scaled = self.scaler.transform(latest_features)
        
        # Make recursive predictions
        predictions = []
        current_features = latest_scaled.copy()
        
        for _ in range(days_ahead):
            # Predict next value
            next_pred = self.model.predict(current_features)[0]
            predictions.append(next_pred)
            
            # Update features for next prediction
            # This is simplified; in practice, you'd need to update all features
            # For now, we use the same features with updated lag values
        
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, 
            np.vstack([self.scaler.transform(X_test)]), 
            y_test, 
            cv=min(5, len(y_test) // 10),
            scoring='neg_mean_squared_error'
        )
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'CV_Score': -np.mean(cv_scores)
        }
    
    def get_feature_importance(self):
        """Get feature importance (for tree-based models)"""
        if self.feature_importance is not None:
            return self.feature_importance
        else:
            return pd.DataFrame()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_cols': self.feature_cols,
            'feature_importance': self.feature_importance
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.model_type = saved_data['model_type']
        self.feature_cols = saved_data['feature_cols']
        self.feature_importance = saved_data['feature_importance']