"""
StockPro - Smart Stock Analysis & Future Price Prediction
Core module initialization
"""

__version__ = "1.0.0"
__author__ = "StockPro Team"

from .data_fetcher import StockDataFetcher
from .technical_analysis import TechnicalAnalyzer
from .visualization import StockVisualizer
from .utils import (
    validate_ticker,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_beta,
    format_number,
    get_available_indices
)

__all__ = [
    'StockDataFetcher',
    'TechnicalAnalyzer',
    'StockVisualizer',
    'validate_ticker',
    'calculate_volatility',
    'calculate_sharpe_ratio',
    'calculate_beta',
    'format_number',
    'get_available_indices'
]