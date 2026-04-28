# 📈 StockPro - Smart Stock Analysis & Future Price Prediction

![StockPro Banner](https://img.shields.io/badge/StockPro-v1.0-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

StockPro is a comprehensive stock analysis and prediction application that combines technical analysis with state-of-the-art machine learning models.

## 🚀 Features

### 📊 Real-time Data
- Fetch historical stock data using Yahoo Finance API
- Support for global stocks and major indices
- Multiple timeframes (1d to 10y+) and intervals
- Indian market support (NSE, BSE) with .NS suffix

### 📈 Technical Analysis
- **Interactive Candlestick Charts** with Plotly
- **10+ Technical Indicators**:
  - Moving Averages (SMA, EMA - 20, 50, 100, 200)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Stochastic Oscillator
- **Support & Resistance Levels** detection
- **Trend Identification** (Uptrend, Downtrend, Sideways)
- **Volume Analysis** with OBV and VPT

### 🤖 AI-Powered Predictions
- **LSTM/GRU Neural Networks** (Deep Learning)
- **Facebook Prophet** (Time Series Forecasting)
- **ARIMA/SARIMA** (Statistical Models)
- **XGBoost & LightGBM** (Gradient Boosting)
- **Linear Regression** (Baseline)
- **Ensemble Predictions** with weighted averaging

### 📉 Risk Analysis
- Volatility Metrics
- Sharpe Ratio
- Beta Calculation
- Maximum Drawdown
- Value at Risk (VaR)

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- TA-Lib (optional, for advanced technical indicators)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/StockPro.git
cd StockPro

# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt


# On Windows
pip install TA-Lib-0.4.28-cp39-cp39-win_amd64.whl

# On macOS
brew install ta-lib
pip install ta-lib

# On Linux
sudo apt-get install ta-lib
pip install ta-lib


streamlit run app.py




















Usage Guide
Enter Stock Symbol

US Stocks: AAPL, GOOGL, MSFT

Indian Stocks: RELIANCE.NS, TCS.NS, INFY.NS

Indices: ^GSPC, ^NSEI, ^DJI

Select Time Period

Choose from preset periods (1mo, 3mo, 6mo, 1y, etc.)

Select data interval (daily, weekly, monthly)

Explore Tabs

Overview: Price charts and key statistics

Technical Analysis: Detailed technical indicators

Trend Analysis: Trend identification and signals

Price Predictor: AI-powered price predictions

Model Comparison: Compare different models

About: Information and documentation

📁 Project Structure
text
stockpro/
├── app.py                      # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py         # Yahoo Finance data fetching
│   ├── technical_analysis.py   # Technical indicators
│   ├── visualization.py        # Plotly charts
│   ├── utils.py                # Utility functions
│   └── models/
│       ├── __init__.py
│       ├── lstm_model.py       # LSTM/GRU Neural Network
│       ├── prophet_model.py    # Facebook Prophet
│       ├── arima_model.py      # ARIMA/SARIMA
│       └── ml_models.py        # XGBoost/LightGBM/Linear
├── data/                       # Data storage
├── saved_models/               # Trained model storage
├── notebooks/                  # Jupyter notebooks
├── requirements.txt
└── README.md
📊 Supported Models
Deep Learning Models
LSTM/GRU: Best for long-term dependencies and complex patterns

Sequence Length: 60 days (configurable)

Features: Price, Volume, Technical Indicators

Time Series Models
Prophet: Excellent for seasonal patterns and holidays

ARIMA: Traditional statistical approach

SARIMA: Seasonal variant for better accuracy

Machine Learning Models
XGBoost: Gradient boosting with feature engineering

LightGBM: Faster training with categorical features

Random Forest: Ensemble method with reduced overfitting

Linear Regression: Simple baseline model

🎯 Use Cases
Day Traders: Quick technical analysis and real-time signals

Long-term Investors: Trend analysis and future predictions

Portfolio Managers: Multi-stock comparison and risk analysis

Students/Researchers: Learn about financial markets and ML

Indian Market: Special support for NSE/BSE stocks

⚙️ Configuration
Customizing Models
You can adjust model parameters in the respective model files:

python
# Example: LSTM parameters
model = LSTMPredictor(
    sequence_length=60,  # Look-back period
    model_type='lstm'    # 'lstm' or 'gru'
)

# Example: Prophet parameters
model = ProphetPredictor()
model.train(data, 
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05
)
Adding New Stocks
Simply enter the ticker symbol in the sidebar:

US Stocks: Just the symbol (e.g., AAPL, TSLA)

Indian Stocks: Add .NS suffix (e.g., RELIANCE.NS)

Other Markets: Use appropriate suffix based on Yahoo Finance

📈 Performance Tips
Data Size: More historical data improves prediction accuracy

Model Selection: Use ensemble predictions for better results

Feature Engineering: ML models perform better with more features

Hyperparameter Tuning: Adjust model parameters for specific stocks

Regular Retraining: Models should be retrained periodically

🐛 Troubleshooting
Common Issues
Data Not Loading

Check internet connection

Verify ticker symbol (US stocks don't need suffix)

Indian stocks need .NS suffix

Some indices use ^ prefix (^NSEI, ^GSPC)

Model Training Errors

Ensure sufficient historical data (at least 100 data points)

Check for NaN values in data

Reduce sequence_length for shorter data

TA-Lib Installation

If TA-Lib fails, the app will use pandas_ta as fallback

Windows users can download pre-built wheels

Linux users need build-essential

📚 Dependencies
Core
Python 3.9+

pandas, numpy

yfinance

plotly, matplotlib, seaborn

streamlit

Machine Learning
scikit-learn

tensorflow, keras

xgboost, lightgbm

prophet

statsmodels, pmdarima

Technical Analysis
pandas_ta

ta-lib (optional)

🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request
