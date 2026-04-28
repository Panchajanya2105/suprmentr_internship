#!/usr/bin/env python3
"""
StockPro - Model Availability Test Script
Run this to check which models are available on your system
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_import(module_name, import_statement):
    """Test if a module can be imported"""
    try:
        exec(import_statement)
        return True, "✅ Available"
    except Exception as e:
        return False, f"❌ Not Available: {str(e)[:80]}"

def test_model_functionality(model_name):
    """Test if model actually works with sample data"""
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    print(f"\n  Testing {model_name} with sample data...")
    
    try:
        if model_name == 'LSTM':
            from src.models.lstm_model import LSTMPredictor
            model = LSTMPredictor(sequence_length=20)
            # Quick test without full training
            print("    ✅ LSTM model can be instantiated")
            return True
            
        elif model_name == 'Prophet':
            from src.models.prophet_model import ProphetPredictor
            model = ProphetPredictor()
            model.train(sample_data)
            predictions, _, _ = model.predict(sample_data, days_ahead=5)
            print(f"    ✅ Prophet trained and predicted {len(predictions)} values")
            return True
            
        elif model_name == 'ARIMA':
            from src.models.arima_model import ARIMAPredictor
            model = ARIMAPredictor()
            model.train(sample_data['Close'].values)
            predictions, _ = model.predict(sample_data['Close'].values, days_ahead=5)
            print(f"    ✅ ARIMA trained and predicted {len(predictions)} values")
            return True
            
        elif model_name == 'XGBoost':
            from src.models.ml_models import MLPredictor
            model = MLPredictor(model_type='xgboost')
            model.train(sample_data)
            predictions = model.predict(sample_data, days_ahead=5)
            print(f"    ✅ XGBoost trained and predicted {len(predictions)} values")
            return True
            
        elif model_name == 'LightGBM':
            from src.models.ml_models import MLPredictor
            model = MLPredictor(model_type='lightgbm')
            model.train(sample_data)
            predictions = model.predict(sample_data, days_ahead=5)
            print(f"    ✅ LightGBM trained and predicted {len(predictions)} values")
            return True
            
        elif model_name == 'Linear Regression':
            from src.models.ml_models import MLPredictor
            model = MLPredictor(model_type='linear')
            model.train(sample_data)
            predictions = model.predict(sample_data, days_ahead=5)
            print(f"    ✅ Linear Regression trained and predicted {len(predictions)} values")
            return True
            
    except Exception as e:
        print(f"    ❌ Functionality test failed: {str(e)[:100]}")
        return False

def main():
    """Main test function"""
    
    print_header("StockPro Model Availability Test")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Working Directory: {os.getcwd()}")
    
    # ============================================
    # Test basic imports
    # ============================================
    print_header("1. Basic Dependencies")
    
    basic_imports = [
        ("pandas", "import pandas as pd"),
        ("numpy", "import numpy as np"),
        ("yfinance", "import yfinance as yf"),
        ("plotly", "import plotly.graph_objects as go"),
        ("streamlit", "import streamlit as st"),
        ("scikit-learn", "import sklearn"),
        ("ta (technical analysis)", "import ta"),
    ]
    
    basic_results = {}
    for name, imp in basic_imports:
        success, msg = test_import(name, imp)
        basic_results[name] = success
        print(f"  {msg}")
    
    # ============================================
    # Test ML/DL libraries
    # ============================================
    print_header("2. Machine Learning Libraries")
    
    ml_imports = [
        ("TensorFlow", "import tensorflow as tf"),
        ("XGBoost", "import xgboost as xgb"),
        ("LightGBM", "import lightgbm as lgb"),
        ("Prophet", "from prophet import Prophet"),
        ("statsmodels", "import statsmodels.api as sm"),
        ("pmdarima", "import pmdarima as pm"),
    ]
    
    ml_results = {}
    for name, imp in ml_imports:
        success, msg = test_import(name, imp)
        ml_results[name] = success
        print(f"  {msg}")
    
    # ============================================
    # Test custom modules
    # ============================================
    print_header("3. Custom Modules")
    
    custom_imports = [
        ("Data Fetcher", "from src.data_fetcher import StockDataFetcher"),
        ("Technical Analyzer", "from src.technical_analysis import TechnicalAnalyzer"),
        ("Visualizer", "from src.visualization import StockVisualizer"),
        ("Utils", "from src.utils import calculate_volatility, format_number"),
    ]
    
    for name, imp in custom_imports:
        success, msg = test_import(name, imp)
        print(f"  {msg}")
    
    # ============================================
    # Test prediction models
    # ============================================
    print_header("4. Prediction Models - Import Test")
    
    model_imports = [
        ("LSTM", "from src.models.lstm_model import LSTMPredictor"),
        ("Prophet", "from src.models.prophet_model import ProphetPredictor"),
        ("ARIMA", "from src.models.arima_model import ARIMAPredictor"),
        ("ML Models (XGBoost/LGBM/LR)", "from src.models.ml_models import MLPredictor"),
    ]
    
    model_status = {}
    for name, imp in model_imports:
        success, msg = test_import(name, imp)
        model_status[name] = success
        print(f"  {msg}")
    
    # ============================================
    # Test model functionality
    # ============================================
    print_header("5. Model Functionality Test (with sample data)")
    
    functionality_results = {}
    
    if model_status.get('LSTM'):
        functionality_results['LSTM'] = test_model_functionality('LSTM')
    
    if model_status.get('Prophet'):
        functionality_results['Prophet'] = test_model_functionality('Prophet')
    
    if model_status.get('ARIMA'):
        functionality_results['ARIMA'] = test_model_functionality('ARIMA')
    
    if model_status.get('ML Models (XGBoost/LGBM/LR)'):
        functionality_results['XGBoost'] = test_model_functionality('XGBoost')
        functionality_results['LightGBM'] = test_model_functionality('LightGBM')
        functionality_results['Linear Regression'] = test_model_functionality('Linear Regression')
    
    # ============================================
    # Installation suggestions
    # ============================================
    print_header("6. Installation Suggestions")
    
    suggestions = []
    
    if not ml_results.get('TensorFlow'):
        suggestions.append("pip install tensorflow")
    
    if not ml_results.get('XGBoost'):
        suggestions.append("pip install xgboost")
    
    if not ml_results.get('LightGBM'):
        suggestions.append("pip install lightgbm")
    
    if not ml_results.get('Prophet'):
        suggestions.append("pip install prophet")
    
    if not ml_results.get('statsmodels'):
        suggestions.append("pip install statsmodels")
    
    if not ml_results.get('pmdarima'):
        suggestions.append("pip install pmdarima")
    
    if suggestions:
        print("\n  Run these commands to install missing packages:")
        for cmd in suggestions:
            print(f"    {cmd}")
    else:
        print("\n  ✅ All packages are installed!")
    
    # ============================================
    # Summary
    # ============================================
    print_header("SUMMARY")
    
    total_tests = len(model_status)
    available = sum(model_status.values())
    
    print(f"\n  Models Available: {available}/{total_tests}")
    print("\n  Model Status:")
    for model, status in model_status.items():
        icon = "✅" if status else "❌"
        func_status = ""
        if model in functionality_results:
            func_status = " (Working)" if functionality_results[model] else " (Import only)"
        elif model == "ML Models (XGBoost/LGBM/LR)":
            if 'XGBoost' in functionality_results:
                func_status = " (Working)" if functionality_results['XGBoost'] else ""
        print(f"    {icon} {model}{func_status}")
    
    print("\n  For Streamlit app, these models will show:")
    if 'LSTM' in model_status and model_status['LSTM']:
        print("    ✅ LSTM Neural Network")
    if 'Prophet' in model_status and model_status['Prophet']:
        print("    ✅ Facebook Prophet")
    if 'ARIMA' in model_status and model_status['ARIMA']:
        print("    ✅ ARIMA")
    if model_status.get('ML Models (XGBoost/LGBM/LR)'):
        print("    ✅ XGBoost")
        print("    ✅ LightGBM")
        print("    ✅ Linear Regression")
    
    print("\n  ✅ Simple Moving Average is ALWAYS available (built-in)")
    
    return available == total_tests

if __name__ == "__main__":
    success = main()
    print("\n" + "="*60)
    if success:
        print("  🎉 ALL MODELS AVAILABLE!")
    else:
        print("  ⚠️  Some models are missing. Check suggestions above.")
    print("="*60 + "\n")
    sys.exit(0 if success else 1)