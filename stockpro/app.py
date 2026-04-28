"""
StockPro - Smart Stock Analysis & Future Price Prediction
Complete Application with All Features
"""

# ============================================
# MUST BE THE FIRST STREAMLIT COMMAND
# ============================================
import streamlit as st
st.set_page_config(
    page_title="StockPro - Smart Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SUPPRESS WARNINGS
# ============================================
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

# ============================================
# IMPORTS
# ============================================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots  # ADD THIS LINE
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================
# YFINANCE DATE PATCH
# ============================================
import yfinance as yf
_original_history = yf.Ticker.history

def _patched_history(self, *args, **kwargs):
    try:
        return _original_history(self, *args, **kwargs)
    except TypeError as e:
        if 'unsupported operand type' in str(e):
            period = kwargs.get('period', '1y')
            interval = kwargs.get('interval', '1d')
            end_date = datetime.now()
            period_map = {
                '1d': timedelta(days=1), '5d': timedelta(days=5),
                '1mo': timedelta(days=30), '3mo': timedelta(days=90),
                '6mo': timedelta(days=180), '1y': timedelta(days=365),
                '2y': timedelta(days=730), '5y': timedelta(days=1825),
                '10y': timedelta(days=3650),
            }
            start_date = end_date - period_map.get(period, timedelta(days=365))
            return _original_history(self, start=start_date, end=end_date, interval=interval)
        raise e

yf.Ticker.history = _patched_history

# ============================================
# IMPORT CUSTOM MODULES
# ============================================
from src.data_fetcher import StockDataFetcher
from src.technical_analysis import TechnicalAnalyzer
from src.visualization import StockVisualizer
from src.utils import calculate_volatility, calculate_sharpe_ratio, format_number

# ============================================
# IMPORT ALL MODELS
# ============================================
MODELS_STATUS = {
    'LSTM': False,
    'Prophet': False,
    'ARIMA': False,
    'XGBoost': False,
    'LightGBM': False,
    'Linear Regression': False
}

try:
    from src.models.lstm_model import LSTMPredictor
    MODELS_STATUS['LSTM'] = True
except Exception as e:
    pass

try:
    from src.models.prophet_model import ProphetPredictor
    MODELS_STATUS['Prophet'] = True
except Exception as e:
    pass

try:
    from src.models.arima_model import ARIMAPredictor
    MODELS_STATUS['ARIMA'] = True
except Exception as e:
    pass

try:
    from src.models.ml_models import MLPredictor
    MODELS_STATUS['XGBoost'] = True
    MODELS_STATUS['LightGBM'] = True
    MODELS_STATUS['Linear Regression'] = True
except Exception as e:
    pass

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .model-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1));
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(102,126,234,0.3);
        margin-bottom: 1rem;
    }
    .metric-positive {
        color: #00ff88;
        font-weight: bold;
    }
    .metric-negative {
        color: #ff4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'stock_info' not in st.session_state:
    st.session_state.stock_info = None
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'fetcher' not in st.session_state:
    st.session_state.fetcher = StockDataFetcher()

# ============================================
# INITIALIZE
# ============================================
fetcher = st.session_state.fetcher
visualizer = StockVisualizer()

# ============================================
# CACHED FUNCTIONS
# ============================================
@st.cache_data(ttl=3600)
def cached_fetch_data(ticker, period, interval):
    return fetcher.fetch_stock_data(ticker, period, interval)

@st.cache_data(ttl=3600)
def cached_fetch_info(ticker):
    return fetcher.fetch_stock_info(ticker)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("## 🎯 Stock Selection")
    
    stock_input = st.text_input(
        "Enter Stock Symbol",
        value="AAPL",
        help="US: AAPL, GOOGL | India: RELIANCE.NS, TCS.NS | Index: ^GSPC, ^NSEI"
    ).upper().strip()
    
    # Quick select
    st.markdown("### ⚡ Quick Select")
    indices = {
        '🇺🇸 S&P 500': '^GSPC',
        '🇺🇸 NASDAQ': '^IXIC', 
        '🇺🇸 DOW': '^DJI',
        '🇮🇳 NIFTY 50': '^NSEI',
        '🇮🇳 SENSEX': '^BSESN',
        '🇯🇵 NIKKEI': '^N225'
    }
    
    cols = st.columns(3)
    for i, (name, symbol) in enumerate(indices.items()):
        with cols[i % 3]:
            if st.button(name, key=f"idx_{i}", use_container_width=True):
                stock_input = symbol
                st.rerun()
    
    st.markdown("---")
    
    # Parameters
    st.markdown("### 📅 Parameters")
    period = st.selectbox("Time Period", 
        ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'], index=3)
    interval = st.selectbox("Interval", 
        ['1d', '1wk', '1mo'], index=0)
    
    # Fetch button
    if st.button("🔍 Fetch Data", use_container_width=True, type="primary"):
        if not stock_input:
            st.error("Please enter a stock symbol")
        else:
            with st.spinner(f"🔄 Fetching data for {stock_input}..."):
                try:
                    data = cached_fetch_data(stock_input, period, interval)
                    if data is not None and not data.empty and 'Close' in data.columns:
                        st.session_state.data = data
                        st.session_state.current_ticker = stock_input
                        st.session_state.analyzer = TechnicalAnalyzer(data)
                        st.session_state.stock_info = cached_fetch_info(stock_input)
                        st.session_state.predictions = {}
                        st.success(f"✅ Data loaded: {stock_input}")
                        st.rerun()
                    else:
                        st.error(f"❌ No data found for {stock_input}")
                        st.info("Check symbol. US: AAPL, Indian: RELIANCE.NS, Index: ^GSPC")
                except Exception as e:
                    st.error(f"Error: {str(e)[:100]}")
    
    st.markdown("---")
    
    # Navigation
    st.markdown("## 📋 Menu")
    
    if st.session_state.data is not None:
        tab = st.radio(
            "Go to",
            ["📊 Overview", "📈 Technical Analysis", "🎯 Trend Analysis",
             "🔮 Price Predictor", "📊 Model Comparison", "📉 Risk Analysis", "ℹ️ About"],
            label_visibility="collapsed"
        )
    else:
        tab = "ℹ️ About"

# ============================================
# MAIN HEADER
# ============================================
st.markdown('<h1 class="main-header">📈 StockPro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Smart Stock Analysis & Future Price Prediction</p>', unsafe_allow_html=True)

# ============================================
# WELCOME SCREEN (No Data)
# ============================================
if st.session_state.data is None:
    st.markdown("## 👋 Welcome to StockPro!")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("""
        ### 🚀 Get Started in 3 Steps
        
        1. **Enter Symbol** → Use sidebar to enter stock ticker
        2. **Fetch Data** → Click 'Fetch Data' button  
        3. **Explore** → Navigate through analysis tabs
        
        ### ✨ All Features
        
        📊 **Overview** - Price charts, volume, key statistics  
        📈 **Technical Analysis** - RSI, MACD, Bollinger Bands, Stochastic  
        🎯 **Trend Analysis** - Support/Resistance, ADX, trend direction  
        🔮 **Price Predictor** - 6 prediction models with ensemble  
        📊 **Model Comparison** - Compare predictions side-by-side  
        📉 **Risk Analysis** - Volatility, Sharpe ratio, VaR, Beta
        """)
    
    with col2:
        st.markdown("### 🔥 Popular Stocks")
        popular = pd.DataFrame({
            'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
            'Name': ['Apple', 'Alphabet', 'Microsoft', 'Tesla', 'Amazon']
        })
        st.dataframe(popular, hide_index=True, use_container_width=True)
        
    with col3:
        st.markdown("### 🇮🇳 Indian Stocks")
        indian = pd.DataFrame({
            'Symbol': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS'],
            'Name': ['Reliance', 'TCS', 'Infosys', 'HDFC Bank']
        })
        st.dataframe(indian, hide_index=True, use_container_width=True)
    
    # Model availability
    st.markdown("### 🤖 Available Models")
    model_df = pd.DataFrame([
        {'Model': k, 'Status': '✅ Available' if v else '❌ Not Installed'}
        for k, v in MODELS_STATUS.items()
    ])
    st.dataframe(model_df, hide_index=True, use_container_width=True)

# ============================================
# MAIN CONTENT (With Data)
# ============================================
else:
    data = st.session_state.data
    analyzer = st.session_state.analyzer
    stock = st.session_state.current_ticker
    
    if 'Close' not in data.columns:
        st.error("Invalid data format. Please fetch again.")
        st.stop()
    
    # ==================== OVERVIEW TAB ====================
    if tab == "📊 Overview":
        st.markdown("## 📊 Stock Overview")
        
        # Stock name and info
        info = st.session_state.stock_info or {}
        st.markdown(f"### {info.get('Name', stock)}")
        if info.get('Sector'):
            st.caption(f"{info.get('Sector')} | {info.get('Industry', '')}")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current = data['Close'].iloc[-1]
        prev = data['Close'].iloc[-2]
        change = current - prev
        change_pct = (change / prev) * 100
        
        col1.metric("Price", f"${current:.2f}", f"{change_pct:+.2f}%")
        
        high_52 = data['High'].max()
        low_52 = data['Low'].min()
        col2.metric("52W High", f"${high_52:.2f}")
        col3.metric("52W Low", f"${low_52:.2f}")
        
        if 'Volume' in data.columns:
            avg_vol = data['Volume'].mean()
            col4.metric("Avg Volume", format_number(avg_vol))
        
        volatility = calculate_volatility(data).iloc[-1] * 100 if len(data) > 20 else 0
        col5.metric("Volatility", f"{volatility:.1f}%")
        
        # Stock info grid
        st.markdown("### 📋 Company Info")
        if info:
            cols = st.columns(4)
            metrics = [
                ('Market Cap', format_number(info.get('Market Cap', 0)) if isinstance(info.get('Market Cap'), (int, float)) else 'N/A'),
                ('P/E Ratio', f"{info.get('PE Ratio', 'N/A')}"),
                ('EPS', f"{info.get('EPS', 'N/A')}"),
                ('Dividend Yield', f"{info.get('Dividend Yield', 'N/A')}"),
                ('Beta', f"{info.get('Beta', 'N/A')}"),
                ('Currency', f"{info.get('Currency', 'N/A')}"),
                ('Sector', f"{info.get('Sector', 'N/A')}"),
                ('Industry', f"{info.get('Industry', 'N/A')}")
            ]
            for i, (label, value) in enumerate(metrics):
                cols[i % 4].metric(label, value)
        
        # Price chart
        st.markdown("### 📈 Price Chart")
        chart_data = data.tail(100) if len(data) > 100 else data
        fig = visualizer.create_candlestick_chart(
            chart_data,
            title=f"{stock} - Candlestick Chart",
            show_volume=True,
            mas=[20, 50, 200],
            bollinger=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Returns Distribution")
            returns = data['Close'].pct_change().dropna()
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns, nbinsx=50,
                marker_color='#667eea',
                opacity=0.7,
                name='Daily Returns'
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="white")
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Cumulative Returns")
            cum_returns = (1 + returns).cumprod()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cum_returns.index, y=cum_returns,
                fill='tozeroy',
                line=dict(color='#00ff88', width=2),
                name='Cumulative Return'
            ))
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TECHNICAL ANALYSIS TAB ====================
    elif tab == "📈 Technical Analysis":
        st.markdown("## 📈 Technical Analysis")
        
        indicator = st.selectbox(
            "Select Indicator",
            ['RSI', 'MACD', 'Bollinger Bands', 'Stochastic Oscillator', 'Volume Analysis']
        )
        
        if indicator == 'Volume Analysis':
            st.markdown("### 📊 Volume Analysis")
            plot_data = data.tail(100) if len(data) > 100 else data
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.05, row_heights=[0.5, 0.5])
            
            # Price
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'],
                                     name='Close', line=dict(color='white', width=2)), row=1, col=1)
            
            # Volume
            colors = ['#00ff88' if plot_data['Close'].iloc[i] >= plot_data['Open'].iloc[i] 
                     else '#ff4444' for i in range(len(plot_data))]
            fig.add_trace(go.Bar(x=plot_data.index, y=plot_data['Volume'],
                                name='Volume', marker_color=colors, opacity=0.6), row=2, col=1)
            
            if 'Volume_SMA' in plot_data.columns:
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Volume_SMA'],
                                        name='Vol SMA', line=dict(color='yellow', width=1)), row=2, col=1)
            
            fig.update_layout(template='plotly_dark', height=500,
                             title='Volume Analysis', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        else:
            indicator_map = {'RSI': 'RSI', 'MACD': 'MACD', 
                           'Bollinger Bands': 'Bollinger', 'Stochastic Oscillator': 'Stochastic'}
            plot_data = data.tail(200) if len(data) > 200 else data
            fig = visualizer.create_technical_indicator_chart(plot_data, indicator_map[indicator])
            st.plotly_chart(fig, use_container_width=True)
        
        # Technical signals summary
        if analyzer:
            st.markdown("### 📡 Signal Summary")
            summary = analyzer.get_summary_statistics()
            
            cols = st.columns(5)
            with cols[0]:
                rsi = summary.get('RSI', 50)
                emoji = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
                st.metric(f"{emoji} RSI", f"{rsi:.1f}")
            with cols[1]:
                trend = summary.get('Trend', 'N/A')
                emoji = "🟢" if trend == 'Uptrend' else "🔴" if trend == 'Downtrend' else "🟡"
                st.metric(f"{emoji} Trend", trend)
            with cols[2]:
                bb = summary.get('BB_Position', 'Middle')
                emoji = "🔴" if bb == 'Upper' else "🟢" if bb == 'Lower' else "🟡"
                st.metric(f"{emoji} BB Position", bb)
            with cols[3]:
                vol_stat = summary.get('Volume_Status', 'Normal')
                st.metric("📊 Volume", vol_stat)
            with cols[4]:
                stoch = summary.get('Stochastic', 50)
                emoji = "🔴" if stoch > 80 else "🟢" if stoch < 20 else "🟡"
                st.metric(f"{emoji} Stochastic", f"{stoch:.1f}")
    
    # ==================== TREND ANALYSIS TAB ====================
    elif tab == "🎯 Trend Analysis":
        st.markdown("## 🎯 Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trend strength gauge
            trend_strength = analyzer.get_trend_strength() if analyzer else 0
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=trend_strength,
                title={'text': "ADX - Trend Strength"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray", 'name': 'Weak'},
                        {'range': [25, 50], 'color': "gray", 'name': 'Moderate'},
                        {'range': [50, 75], 'color': "darkgray", 'name': 'Strong'},
                        {'range': [75, 100], 'color': "#667eea", 'name': 'Very Strong'}
                    ]
                }
            ))
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Trend distribution
            if 'Trend' in data.columns:
                trend_counts = data['Trend'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=trend_counts.index,
                    values=trend_counts.values,
                    hole=0.5,
                    marker_colors=['#00ff88', '#ff4444', '#ffaa00'],
                    textinfo='label+percent'
                )])
                fig.update_layout(template='plotly_dark', height=350, title='Trend Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        # Support and Resistance
        st.markdown("### 🎯 Support & Resistance Levels")
        plot_data = data.tail(100) if len(data) > 100 else data
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=plot_data.index, open=plot_data['Open'], high=plot_data['High'],
            low=plot_data['Low'], close=plot_data['Close'], name='Price'
        ))
        
        if 'Dynamic_Support' in plot_data.columns:
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Dynamic_Support'],
                                    name='Support', line=dict(color='#00ff88', width=2, dash='dash')))
        if 'Dynamic_Resistance' in plot_data.columns:
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Dynamic_Resistance'],
                                    name='Resistance', line=dict(color='#ff4444', width=2, dash='dash')))
        
        fig.update_layout(template='plotly_dark', height=500,
                         title='Support & Resistance Levels', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== PRICE PREDICTOR TAB ====================
    elif tab == "🔮 Price Predictor":
        st.markdown("## 🔮 Price Predictor")
        
        # Build available models - ALWAYS include all possible models
        available_models = ['Simple Moving Average']  # Always available baseline
        
        if MODELS_STATUS['Linear Regression']:
            available_models.append('Linear Regression')
        if MODELS_STATUS['ARIMA']:
            available_models.append('ARIMA')
        if MODELS_STATUS['Prophet']:
            available_models.append('Facebook Prophet')
        if MODELS_STATUS['XGBoost']:
            available_models.append('XGBoost')
        if MODELS_STATUS['LightGBM']:
            available_models.append('LightGBM')
        if MODELS_STATUS['LSTM']:
            available_models.append('LSTM Neural Network')
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            model_type = st.selectbox("🤖 Choose Model", available_models)
        
        with col2:
            days_ahead = st.number_input("📅 Days", min_value=7, max_value=90, value=30)
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🚀 Predict", use_container_width=True, type="primary")
        
        # Model info
        model_descriptions = {
            'Simple Moving Average': '📊 Baseline: Simple trend projection using moving averages',
            'Linear Regression': '📈 Basic ML: Linear relationship between features and price',
            'ARIMA': '📉 Statistical: Auto-Regressive Integrated Moving Average',
            'Facebook Prophet': '🔮 Advanced: Facebook\'s time series forecasting model',
            'XGBoost': '🌲 Powerful: Gradient boosting with decision trees',
            'LightGBM': '⚡ Fast: Light gradient boosting machine',
            'LSTM Neural Network': '🧠 Deep Learning: Long Short-Term Memory neural network'
        }
        st.caption(model_descriptions.get(model_type, ''))
        
        if predict_btn:
            with st.spinner(f"🔄 Training {model_type}..."):
                try:
                    predictions = None
                    future_dates = pd.date_range(
                        start=data.index[-1] + timedelta(days=1),
                        periods=days_ahead,
                        freq='B'
                    )
                    
                    if model_type == 'Simple Moving Average':
                        close_prices = data['Close'].values
                        sma_short = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else np.mean(close_prices)
                        sma_long = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else np.mean(close_prices)
                        trend = np.polyfit(range(min(20, len(close_prices))), 
                                         close_prices[-min(20, len(close_prices)):], 1)[0]
                        predictions = np.array([close_prices[-1] + trend * (i+1) for i in range(days_ahead)])
                    
                    elif model_type == 'Linear Regression' and MODELS_STATUS['Linear Regression']:
                        model = MLPredictor(model_type='linear')
                        model.train(data)
                        predictions = model.predict(data, days_ahead)
                    
                    elif model_type == 'ARIMA' and MODELS_STATUS['ARIMA']:
                        from src.models.arima_model import ARIMAPredictor
                        model = ARIMAPredictor()
                        model.train(data['Close'].values)
                        predictions, _ = model.predict(data['Close'].values, days_ahead)
                    
                    elif model_type == 'Facebook Prophet' and MODELS_STATUS['Prophet']:
                        from src.models.prophet_model import ProphetPredictor
                        model = ProphetPredictor()
                        model.train(data)
                        pred_df, _, _ = model.predict(data, days_ahead)
                        predictions = pred_df['yhat'].values
                    
                    elif model_type == 'XGBoost' and MODELS_STATUS['XGBoost']:
                        model = MLPredictor(model_type='xgboost')
                        model.train(data)
                        predictions = model.predict(data, days_ahead)
                    
                    elif model_type == 'LightGBM' and MODELS_STATUS['LightGBM']:
                        model = MLPredictor(model_type='lightgbm')
                        model.train(data)
                        predictions = model.predict(data, days_ahead)
                    
                    elif model_type == 'LSTM Neural Network' and MODELS_STATUS['LSTM']:
                        from src.models.lstm_model import LSTMPredictor
                        model = LSTMPredictor(sequence_length=min(60, len(data)//2))
                        model.train(data, epochs=30, verbose=0)
                        predictions = model.predict(data, days_ahead)
                    
                    if predictions is not None and len(predictions) > 0:
                        st.session_state.predictions[model_type] = {
                            'predictions': predictions,
                            'dates': future_dates
                        }
                        st.success(f"✅ {model_type} predictions ready!")
                        st.rerun()
                    else:
                        st.error("❌ Prediction failed - no output generated")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:200]}")
                    st.info("Try a different model or check your data")
        
        # Display predictions
        if model_type in st.session_state.predictions:
            pred_data = st.session_state.predictions[model_type]
            
            st.markdown("---")
            st.markdown("### 📈 Prediction Results")
            
            # Prediction chart
            fig = visualizer.create_prediction_chart(
                data.tail(100) if len(data) > 100 else data,
                pred_data['predictions'],
                pred_data['dates'],
                f"{model_type} - {days_ahead} Day Forecast"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction stats
            current_price = data['Close'].iloc[-1]
            final_pred = pred_data['predictions'][-1]
            pred_change = ((final_pred - current_price) / current_price) * 100
            
            cols = st.columns(5)
            cols[0].metric("Current Price", f"${current_price:.2f}")
            cols[1].metric("Predicted", f"${final_pred:.2f}", f"{pred_change:+.2f}%")
            cols[2].metric("Min", f"${pred_data['predictions'].min():.2f}")
            cols[3].metric("Max", f"${pred_data['predictions'].max():.2f}")
            cols[4].metric("Avg", f"${pred_data['predictions'].mean():.2f}")
            
            # Prediction table
            st.markdown("### 📋 Day-by-Day Forecast")
            pred_table = pd.DataFrame({
                'Date': pred_data['dates'].strftime('%Y-%m-%d'),
                'Predicted Price': [f"${p:.2f}" for p in pred_data['predictions']],
                'Change': [f"{((p-current_price)/current_price)*100:+.2f}%" for p in pred_data['predictions']]
            })
            st.dataframe(pred_table, use_container_width=True, hide_index=True, height=400)
            
            # Download
            csv_data = pd.DataFrame({
                'Date': pred_data['dates'].strftime('%Y-%m-%d'),
                'Predicted_Price': pred_data['predictions'].round(2)
            })
            st.download_button(
                "📥 Download CSV",
                csv_data.to_csv(index=False),
                f"prediction_{stock}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
    
    # ==================== MODEL COMPARISON TAB ====================
    elif tab == "📊 Model Comparison":
        st.markdown("## 📊 Model Comparison")
        
        if len(st.session_state.predictions) < 2:
            st.info("👆 Generate predictions from at least 2 models in the Price Predictor tab to compare them here")
            
            if len(st.session_state.predictions) == 1:
                st.success(f"✅ 1 model ready: {list(st.session_state.predictions.keys())[0]}")
                st.write("Generate predictions from another model to enable comparison")
        else:
            st.success(f"✅ {len(st.session_state.predictions)} models ready for comparison")
            
            # Prepare comparison data
            predictions_dict = {}
            for name, pred in st.session_state.predictions.items():
                predictions_dict[name] = pred['predictions']
            
            first_model = list(st.session_state.predictions.keys())[0]
            dates = st.session_state.predictions[first_model]['dates']
            
            # Comparison chart
            fig = visualizer.create_model_comparison_chart(
                predictions_dict,
                title=f"Model Comparison - {len(predictions_dict)} Models"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison table
            st.markdown("### 📊 Prediction Comparison")
            current_price = data['Close'].iloc[-1]
            
            comparison_data = []
            for model_name, pred in st.session_state.predictions.items():
                preds = pred['predictions']
                comparison_data.append({
                    'Model': model_name,
                    'Final Price': f"${preds[-1]:.2f}",
                    'Expected Change': f"{((preds[-1]-current_price)/current_price)*100:+.2f}%",
                    'Min': f"${preds.min():.2f}",
                    'Max': f"${preds.max():.2f}",
                    'Range': f"${preds.max()-preds.min():.2f}",
                    'Volatility': f"{preds.std():.2f}"
                })
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    # ==================== RISK ANALYSIS TAB ====================
    elif tab == "📉 Risk Analysis":
        st.markdown("## 📉 Risk Analysis")
        
        returns = data['Close'].pct_change().dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Risk Metrics")
            
            # Calculate metrics
            volatility_annual = returns.std() * np.sqrt(252)
            sharpe = calculate_sharpe_ratio(data)
            
            metrics = {
                'Annualized Volatility': f"{volatility_annual*100:.2f}%",
                'Daily Volatility': f"{returns.std()*100:.2f}%",
                'Sharpe Ratio': f"{sharpe:.3f}",
                'Max Daily Gain': f"{returns.max()*100:.2f}%",
                'Max Daily Loss': f"{returns.min()*100:.2f}%",
                'Positive Days': f"{(returns > 0).sum()} ({(returns > 0).mean()*100:.1f}%)",
                'Negative Days': f"{(returns < 0).sum()} ({(returns < 0).mean()*100:.1f}%)"
            }
            
            for label, value in metrics.items():
                st.metric(label, value)
        
        with col2:
            st.markdown("### 📈 Rolling Volatility")
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_vol.index, y=rolling_vol,
                fill='tozeroy',
                line=dict(color='#ff8800', width=2),
                name='20-Day Volatility'
            ))
            fig.update_layout(template='plotly_dark', height=400,
                             yaxis_title='Volatility (%)', title='Rolling Volatility (Annualized)')
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== ABOUT TAB ====================
    elif tab == "ℹ️ About":
        st.markdown("## ℹ️ About StockPro")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🚀 StockPro v1.0
            
            **Smart Stock Analysis & Future Price Prediction**
            
            A comprehensive tool combining technical analysis with state-of-the-art machine learning models.
            
            #### 🎯 Features
            - 📊 **Interactive Charts** - Candlestick, line, and area charts
            - 📈 **Technical Indicators** - RSI, MACD, Bollinger Bands, Stochastic, Volume
            - 🎯 **Trend Detection** - Support/Resistance levels, ADX, trend strength
            - 🤖 **6 Prediction Models** - From simple MA to deep learning LSTM
            - 📉 **Risk Metrics** - Volatility, Sharpe ratio, drawdown analysis
            - 📊 **Model Comparison** - Side-by-side prediction comparison
            
            #### 🛠️ Tech Stack
            - **Frontend**: Streamlit, Plotly
            - **Data**: yfinance, pandas, numpy
            - **ML/DL**: scikit-learn, TensorFlow, XGBoost, LightGBM, Prophet
            - **Technical Analysis**: ta, custom indicators
            """)
        
        with col2:
            st.markdown("### 🤖 Model Status")
            for model, status in MODELS_STATUS.items():
                icon = "✅" if status else "❌"
                st.write(f"{icon} **{model}**")
            
            st.markdown("### ⚠️ Disclaimer")
            st.warning("""
            This tool is for **educational purposes only**. 
            Past performance does not guarantee future results.
            Always consult a financial advisor before investing.
            """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #888; padding: 1rem; font-size: 0.9rem;'>
        📈 <b>StockPro</b> v1.0 | Powered by Streamlit & Python | 
        <span style='color: #667eea;'>Models Available: {sum(MODELS_STATUS.values())}/6</span>
        <br>⚠️ For educational purposes only | Not financial advice
    </div>
    """,
    unsafe_allow_html=True
)