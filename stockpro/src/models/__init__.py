"""
StockPro Models Module
Contains all prediction models
"""

# Import each model with error handling
LSTM_AVAILABLE = False
PROPHET_AVAILABLE = False
ARIMA_AVAILABLE = False
ML_AVAILABLE = False

# Try importing LSTM
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from .lstm_model import LSTMPredictor
    LSTM_AVAILABLE = True
except Exception as e:
    LSTMPredictor = None

# Try importing Prophet
try:
    from .prophet_model import ProphetPredictor
    PROPHET_AVAILABLE = True
except Exception as e:
    ProphetPredictor = None

# Try importing ARIMA
try:
    from .arima_model import ARIMAPredictor
    ARIMA_AVAILABLE = True
except Exception as e:
    ARIMAPredictor = None

# Try importing ML Models
try:
    from .ml_models import MLPredictor
    ML_AVAILABLE = True
except Exception as e:
    MLPredictor = None

__all__ = [
    'LSTMPredictor',
    'ProphetPredictor', 
    'ARIMAPredictor',
    'MLPredictor',
    'LSTM_AVAILABLE',
    'PROPHET_AVAILABLE',
    'ARIMA_AVAILABLE',
    'ML_AVAILABLE'
]