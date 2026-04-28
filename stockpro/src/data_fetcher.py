"""
Data fetching module using yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import streamlit as st

class StockDataFetcher:
    """Class for fetching stock data from Yahoo Finance"""
    
    def __init__(self):
        self.cache = {}
        self.indices = {
            'NIFTY 50': '^NSEI',
            'SENSEX': '^BSESN',
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'DOW JONES': '^DJI',
            'FTSE 100': '^FTSE',
            'NIKKEI 225': '^N225',
            'HANG SENG': '^HSI'
        }
    
    def fetch_stock_data(self, ticker: str, period: str = "1y", 
                         interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical stock data
        
        Args:
            ticker: Stock ticker symbol
            period: Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,max
            interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        """
        # Use the cached static method
        return self._cached_fetch(ticker, period, interval)
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def _cached_fetch(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Cached static method to fetch stock data
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for ticker: {ticker}")
            
            # Add some basic indicators
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            
            return data
        
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_stocks(self, tickers: List[str], period: str = "1y",
                             interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks"""
        data_dict = {}
        for ticker in tickers:
            data = self.fetch_stock_data(ticker, period, interval)
            if not data.empty:
                data_dict[ticker] = data
        return data_dict
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_stock_info_static(ticker: str) -> Dict:
        """Static cached method to fetch stock information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key information
            key_info = {
                'Name': info.get('longName', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'Market Cap': info.get('marketCap', 'N/A'),
                'PE Ratio': info.get('trailingPE', 'N/A'),
                'EPS': info.get('trailingEps', 'N/A'),
                'Dividend Yield': info.get('dividendYield', 'N/A'),
                '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
                'Average Volume': info.get('averageVolume', 'N/A'),
                'Beta': info.get('beta', 'N/A'),
                'Currency': info.get('currency', 'N/A'),
                'Current Price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
            }
            
            return key_info
        
        except Exception as e:
            st.error(f"Error fetching info for {ticker}: {str(e)}")
            return {}
    
    def fetch_stock_info(self, ticker: str) -> Dict:
        """Fetch detailed stock information"""
        return self.fetch_stock_info_static(ticker)
    
    def fetch_index_data(self, index_name: str, period: str = "1y") -> pd.DataFrame:
        """Fetch data for major indices"""
        if index_name in self.indices:
            return self.fetch_stock_data(self.indices[index_name], period)
        else:
            raise ValueError(f"Unknown index: {index_name}")
    
    def get_available_periods(self) -> List[str]:
        """Return available time periods"""
        return ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']
    
    def get_available_intervals(self, period: str) -> List[str]:
        """Return valid intervals based on period"""
        if period in ['1d']:
            return ['1m', '2m', '5m', '15m', '30m', '60m', '90m']
        elif period in ['5d', '1mo']:
            return ['15m', '30m', '60m', '90m', '1h', '1d']
        elif period in ['3mo', '6mo']:
            return ['1h', '1d', '5d', '1wk']
        else:
            return ['1d', '5d', '1wk', '1mo', '3mo']
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def compare_stocks_static(tickers: tuple, period: str = "1y") -> pd.DataFrame:
        """Compare multiple stocks by normalizing their prices (cached version)"""
        data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                stock_data = stock.history(period=period)
                if not stock_data.empty:
                    # Normalize prices to start at 100
                    normalized = (stock_data['Close'] / stock_data['Close'].iloc[0] * 100)
                    data[ticker] = normalized
            except Exception as e:
                st.warning(f"Could not fetch data for {ticker}: {str(e)}")
        
        return pd.DataFrame(data)
    
    def compare_stocks(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """Compare multiple stocks by normalizing their prices"""
        # Convert list to tuple for caching
        return self.compare_stocks_static(tuple(tickers), period)