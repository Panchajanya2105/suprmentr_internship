"""
Utility functions for StockPro
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

def validate_ticker(ticker):
    """Validate if ticker symbol exists"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if len(info) > 0 and 'symbol' in info:
            return True
        return False
    except:
        return False

def get_available_indices():
    """Return dictionary of major indices"""
    return {
        'NIFTY 50': '^NSEI',
        'SENSEX': '^BSESN',
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'DOW JONES': '^DJI',
        'FTSE 100': '^FTSE',
        'NIKKEI 225': '^N225',
        'HANG SENG': '^HSI'
    }

def calculate_returns(data, period=1):
    """Calculate returns over given period"""
    return data['Close'].pct_change(periods=period)

def calculate_volatility(data, window=20):
    """Calculate rolling volatility"""
    returns = data['Close'].pct_change()
    return returns.rolling(window=window).std() * np.sqrt(252)

def calculate_sharpe_ratio(data, risk_free_rate=0.02):
    """Calculate Sharpe Ratio"""
    returns = data['Close'].pct_change()
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_beta(stock_data, market_data):
    """Calculate Beta relative to market"""
    stock_returns = stock_data['Close'].pct_change().dropna()
    market_returns = market_data['Close'].pct_change().dropna()
    
    # Align the data
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    stock_returns = aligned.iloc[:, 0]
    market_returns = aligned.iloc[:, 1]
    
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    
    return covariance / market_variance

def prepare_data_for_ml(df, target_col='Close', lookback=60):
    """Prepare data for ML models with features"""
    df = df.copy()
    
    # Price-based features
    df['Returns'] = df[target_col].pct_change()
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = df['High'] - df['Close'].shift(1)
    df['Low_Close'] = df['Low'] - df['Close'].shift(1)
    df['Close_Open'] = df['Close'] - df['Open']
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'SMA_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'Std_{window}'] = df[target_col].rolling(window=window).std()
    
    # RSI
    delta = df[target_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume features
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    
    # Target
    df['Target'] = df[target_col].shift(-1)
    
    return df.dropna()

def format_number(num):
    """Format large numbers for display"""
    if num >= 1e9:
        return f'{num/1e9:.2f}B'
    elif num >= 1e6:
        return f'{num/1e6:.2f}M'
    elif num >= 1e3:
        return f'{num/1e3:.2f}K'
    else:
        return f'{num:.2f}'