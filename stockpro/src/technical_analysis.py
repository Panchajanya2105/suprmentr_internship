"""
Technical Analysis module with various indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import ta

class TechnicalAnalyzer:
    """Class for performing technical analysis on stock data"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with stock data
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data.copy()
        self.calculate_all_indicators()
    
    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        self.calculate_moving_averages()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_stochastic()
        self.calculate_volume_indicators()
        self.calculate_support_resistance()
        self.identify_trend()
    
    
    def calculate_moving_averages(self):
        """Calculate SMA and EMA for multiple periods"""
        periods = [20, 50, 100, 200]
        
        for period in periods:
            # Simple Moving Average
            self.data[f'SMA_{period}'] = self.data['Close'].rolling(window=period).mean()
            # Exponential Moving Average
            self.data[f'EMA_{period}'] = self.data['Close'].ewm(span=period, adjust=False).mean()
    
    # def calculate_rsi(self, period: int = 14):
    #     """Calculate Relative Strength Index"""
    #     delta = self.data['Close'].diff()
    #     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    #     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    #     
    #     rs = gain / loss
    #     self.data['RSI'] = 100 - (100 / (1 + rs))
    #     
    #     # Add RSI signals
    #     self.data['RSI_Signal'] = 'Neutral'
    #     self.data.loc[self.data['RSI'] > 70, 'RSI_Signal'] = 'Overbought'
    #     self.data.loc[self.data['RSI'] < 30, 'RSI_Signal'] = 'Oversold'
    def calculate_rsi(self, period: int = 14):
        """Calculate Relative Strength Index"""
        self.data['RSI'] = ta.momentum.RSIIndicator(
            close=self.data['Close'], 
            window=period
        ).rsi()
        
        # Add RSI signals
        self.data['RSI_Signal'] = 'Neutral'
        self.data.loc[self.data['RSI'] > 70, 'RSI_Signal'] = 'Overbought'
        self.data.loc[self.data['RSI'] < 30, 'RSI_Signal'] = 'Oversold'
    
    # def calculate_macd(self):
    #     """Calculate MACD indicator"""
    #     # MACD Line
    #     exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
    #     exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
    #     self.data['MACD'] = exp1 - exp2
    #     
    #     # Signal Line
    #     self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
    #     
    #     # MACD Histogram
    #     self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']
    #     
    #     # MACD Crossover signals
    #     self.data['MACD_Crossover'] = 'Hold'
    #     self.data.loc[self.data['MACD'] > self.data['MACD_Signal'], 'MACD_Crossover'] = 'Bullish'
    #     self.data.loc[self.data['MACD'] < self.data['MACD_Signal'], 'MACD_Crossover'] = 'Bearish'
    def calculate_macd(self):
        """Calculate MACD indicator"""
        # MACD Line
        macd = ta.trend.MACD(close=self.data['Close'])
        self.data['MACD'] = macd.macd()
        self.data['MACD_Signal'] = macd.macd_signal()
        self.data['MACD_Histogram'] = macd.macd_diff()
        
        # MACD Crossover signals
        self.data['MACD_Crossover'] = 'Hold'
        self.data.loc[self.data['MACD'] > self.data['MACD_Signal'], 'MACD_Crossover'] = 'Bullish'
        self.data.loc[self.data['MACD'] < self.data['MACD_Signal'], 'MACD_Crossover'] = 'Bearish'
        
    # def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2):
    #     """Calculate Bollinger Bands"""
    #     # Middle Band (SMA)
    #     self.data['BB_Middle'] = self.data['Close'].rolling(window=period).mean()
    #     
    #     # Standard Deviation
    #     bb_std = self.data['Close'].rolling(window=period).std()
    #     
    #     # Upper and Lower Bands
    #     self.data['BB_Upper'] = self.data['BB_Middle'] + (bb_std * std_dev)
    #     self.data['BB_Lower'] = self.data['BB_Middle'] - (bb_std * std_dev)
    #     
    #     # Bandwidth
    #     self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / self.data['BB_Middle']
    #     
    #     # %B indicator
    #     self.data['BB_Pct'] = (self.data['Close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands"""
        bollinger = ta.volatility.BollingerBands(
            close=self.data['Close'],
            window=period,
            window_dev=std_dev
        )
        self.data['BB_Middle'] = bollinger.bollinger_mavg()
        self.data['BB_Upper'] = bollinger.bollinger_hband()
        self.data['BB_Lower'] = bollinger.bollinger_lband()
        self.data['BB_Width'] = bollinger.bollinger_wband()
        self.data['BB_Pct'] = bollinger.bollinger_pband()

    # def calculate_stochastic(self, k_period: int = 14, d_period: int = 3):
    #     """Calculate Stochastic Oscillator"""
    #     low_min = self.data['Low'].rolling(window=k_period).min()
    #     high_max = self.data['High'].rolling(window=k_period).max()
    #     
    #     self.data['Stoch_K'] = 100 * ((self.data['Close'] - low_min) / (high_max - low_min))
    #     self.data['Stoch_D'] = self.data['Stoch_K'].rolling(window=d_period).mean()
    #     
    #     # Stochastic signals
    #     self.data['Stoch_Signal'] = 'Neutral'
    #     self.data.loc[self.data['Stoch_K'] > 80, 'Stoch_Signal'] = 'Overbought'
    #     self.data.loc[self.data['Stoch_K'] < 20, 'Stoch_Signal'] = 'Oversold'
    # 
    def calculate_stochastic(self, k_period: int = 14, d_period: int = 3):
        """Calculate Stochastic Oscillator"""
        stoch = ta.momentum.StochasticOscillator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            window=k_period,
            smooth_window=d_period
        )
        self.data['Stoch_K'] = stoch.stoch()
        self.data['Stoch_D'] = stoch.stoch_signal()
        
        # Stochastic signals
        self.data['Stoch_Signal'] = 'Neutral'
        self.data.loc[self.data['Stoch_K'] > 80, 'Stoch_Signal'] = 'Overbought'
        self.data.loc[self.data['Stoch_K'] < 20, 'Stoch_Signal'] = 'Oversold'

    def calculate_volume_indicators(self):
        """Calculate volume-based indicators"""
        # Volume Moving Average
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
        
        # Volume Price Trend (VPT)
        self.data['VPT'] = (self.data['Volume'] * 
                           ((self.data['Close'] - self.data['Close'].shift(1)) / 
                            self.data['Close'].shift(1))).cumsum()
        
        # On-Balance Volume (OBV)
        self.data['OBV'] = (np.sign(self.data['Close'].diff()) * self.data['Volume']).fillna(0).cumsum()
        
        # Volume Relative Strength
        self.data['Volume_RS'] = self.data['Volume'] / self.data['Volume_SMA']
    
    def calculate_support_resistance(self, window: int = 20):
        """Detect support and resistance levels"""
        # Support levels (local minima)
        self.data['Support'] = self.data['Low'].rolling(window=window, center=True).min()
        
        # Resistance levels (local maxima)
        self.data['Resistance'] = self.data['High'].rolling(window=window, center=True).max()
        
        # Dynamic support and resistance
        self.data['Dynamic_Support'] = self.data['Low'].rolling(window=window).min()
        self.data['Dynamic_Resistance'] = self.data['High'].rolling(window=window).max()
    
    def identify_trend(self):
        """Identify the current trend (Uptrend, Downtrend, Sideways)"""
        # Using SMA crossovers and price action
        sma_20 = self.data['SMA_20']
        sma_50 = self.data['SMA_50']
        
        conditions = [
            (sma_20 > sma_50) & (self.data['Close'] > sma_20),
            (sma_20 < sma_50) & (self.data['Close'] < sma_20),
            True  # Default condition
        ]
        choices = ['Uptrend', 'Downtrend', 'Sideways']
        
        self.data['Trend'] = np.select(conditions[:-1], choices[:-1], default='Sideways')
    
    def get_trend_strength(self) -> float:
        """Calculate trend strength using ADX-like metric"""
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        # Calculate directional movement
        plus_dm = self.data['High'].diff()
        minus_dm = self.data['Low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        # Calculate DI
        tr_14 = true_range.rolling(window=14).sum()
        plus_di = 100 * (plus_dm.rolling(window=14).sum() / tr_14)
        minus_di = abs(100 * (minus_dm.rolling(window=14).sum() / tr_14))
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14).mean()
        
        return adx.iloc[-1] if not adx.empty else 0
    
    def get_summary_statistics(self) -> Dict:
        """Get summary of technical indicators"""
        latest = self.data.iloc[-1]
        
        return {
            'Close': latest['Close'],
            'RSI': latest['RSI'],
            'MACD': latest['MACD'],
            'MACD_Signal': latest['MACD_Signal'],
            'Trend': latest['Trend'],
            'BB_Position': 'Upper' if latest['Close'] > latest['BB_Upper'] else 
                          'Lower' if latest['Close'] < latest['BB_Lower'] else 'Middle',
            'Volume_Status': 'High' if latest['Volume'] > latest['Volume_SMA'] else 'Low',
            'Stochastic': latest['Stoch_K'],
            'Volatility': latest['BB_Width']
        }