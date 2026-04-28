"""
LSTM/GRU Neural Network Model for Stock Price Prediction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os

class LSTMPredictor:
    """LSTM-based stock price predictor"""
    
    def __init__(self, sequence_length=60, model_type='lstm'):
        """
        Initialize LSTM predictor
        
        Args:
            sequence_length: Number of time steps to look back
            model_type: 'lstm' or 'gru'
        """
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        
    def prepare_data(self, data, target_col='Close'):
        """Prepare data for LSTM model"""
        # Create features
        df = data.copy()
        
        # Add technical indicators as features
        df['Returns'] = df[target_col].pct_change()
        df['MA5'] = df[target_col].rolling(window=5).mean()
        df['MA20'] = df[target_col].rolling(window=20).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        # Prepare features and target
        feature_cols = [target_col, 'Volume', 'Returns', 'MA5', 'MA20', 
                       'Volatility', 'Volume_Change', 'Volume_MA5']
        
        features = df[feature_cols].values
        target = df[target_col].values.reshape(-1, 1)
        
        # Scale the data
        scaled_features = self.feature_scaler.fit_transform(features)
        scaled_target = self.scaler.fit_transform(target)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_target[i-1])  # Predict next day's closing price
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        return X, y, df.index[self.sequence_length:]
    
    def build_model(self, input_shape):
        """Build LSTM/GRU model architecture"""
        model = keras.Sequential()
        
        if self.model_type.lower() == 'lstm':
            # LSTM layers
            model.add(layers.LSTM(100, return_sequences=True, input_shape=input_shape))
            model.add(layers.Dropout(0.2))
            model.add(layers.LSTM(100, return_sequences=True))
            model.add(layers.Dropout(0.2))
            model.add(layers.LSTM(50, return_sequences=False))
            model.add(layers.Dropout(0.2))
        else:
            # GRU layers
            model.add(layers.GRU(100, return_sequences=True, input_shape=input_shape))
            model.add(layers.Dropout(0.2))
            model.add(layers.GRU(100, return_sequences=True))
            model.add(layers.Dropout(0.2))
            model.add(layers.GRU(50, return_sequences=False))
            model.add(layers.Dropout(0.2))
        
        # Dense layers
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(25, activation='relu'))
        model.add(layers.Dense(1))
        
        # Compile model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                     loss='mean_squared_error',
                     metrics=['mae'])
        
        return model
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2, verbose=1):
        """Train the LSTM model"""
        X, y, _ = self.prepare_data(data)
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        
        return history
    
    def predict(self, data, days_ahead=30):
        """Make predictions for future days"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Prepare recent data for prediction
        df = data.copy()
        
        # Create features
        df['Returns'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df = df.dropna()
        
        feature_cols = ['Close', 'Volume', 'Returns', 'MA5', 'MA20', 
                       'Volatility', 'Volume_Change', 'Volume_MA5']
        
        # Get the last sequence
        recent_data = df[feature_cols].values[-self.sequence_length:]
        scaled_recent = self.feature_scaler.transform(recent_data)
        
        # Make predictions
        predictions = []
        current_sequence = scaled_recent.copy()
        
        for _ in range(days_ahead):
            # Reshape for prediction
            current_sequence_reshaped = current_sequence.reshape(1, self.sequence_length, -1)
            
            # Predict next value
            next_pred = self.model.predict(current_sequence_reshaped, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            # This is a simplified approach; in practice, you'd need all features
            new_row = current_sequence[-1].copy()
            new_row[0] = next_pred[0, 0]  # Update close price
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Inverse transform
        y_test_inv = self.scaler.inverse_transform(y_test)
        y_pred_inv = self.scaler.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save Keras model
        self.model.save(filepath)
        
        # Save scalers
        joblib.dump(self.scaler, filepath.replace('.h5', '_scaler.pkl'))
        joblib.dump(self.feature_scaler, filepath.replace('.h5', '_feature_scaler.pkl'))
    
    def load_model(self, filepath):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load Keras model
        self.model = keras.models.load_model(filepath)
        
        # Load scalers
        self.scaler = joblib.load(filepath.replace('.h5', '_scaler.pkl'))
        self.feature_scaler = joblib.load(filepath.replace('.h5', '_feature_scaler.pkl'))