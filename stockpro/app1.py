"""
StockPro - Smart Stock Analysis & Future Price Prediction
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from src.data_fetcher import StockDataFetcher
from src.technical_analysis import TechnicalAnalyzer
from src.visualization import StockVisualizer
from src.utils import (
    calculate_volatility, 
    calculate_sharpe_ratio, 
    calculate_beta, 
    format_number,
    validate_ticker
)

# Import models (with error handling)
try:
    from src.models.lstm_model import LSTMPredictor
    LSTM_AVAILABLE = True
except Exception as e:
    LSTM_AVAILABLE = False
    st.warning(f"LSTM model not available: {e}")

try:
    from src.models.prophet_model import ProphetPredictor
    PROPHET_AVAILABLE = True
except Exception as e:
    PROPHET_AVAILABLE = False

try:
    from src.models.arima_model import ARIMAPredictor
    ARIMA_AVAILABLE = True
except Exception as e:
    ARIMA_AVAILABLE = False

try:
    from src.models.ml_models import MLPredictor
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="StockPro - Smart Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
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

# Initialize core objects
fetcher = StockDataFetcher()
visualizer = StockVisualizer()

# Main header
st.markdown('<h1 class="main-header">📈 StockPro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Smart Stock Analysis & Future Price Prediction</p>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Stock Selection")
    
    # Stock input
    stock_input = st.text_input(
        "Enter Stock Symbol",
        value="AAPL",
        help="US Stocks: AAPL, GOOGL | Indian: RELIANCE.NS, TCS.NS | Indices: ^GSPC, ^NSEI"
    ).upper().strip()
    
    # Quick select indices
    st.markdown("### Quick Select")
    indices = {
        '🇺🇸 S&P 500': '^GSPC',
        '🇺🇸 NASDAQ': '^IXIC',
        '🇺🇸 DOW JONES': '^DJI',
        '🇮🇳 NIFTY 50': '^NSEI',
        '🇮🇳 SENSEX': '^BSESN'
    }
    
    cols = st.columns(2)
    for i, (name, symbol) in enumerate(indices.items()):
        with cols[i % 2]:
            if st.button(name, key=f"idx_{i}", use_container_width=True):
                stock_input = symbol
    
    # Time period selection
    st.markdown("### 📅 Parameters")
    period = st.selectbox(
        "Time Period",
        ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
        index=3,
        help="Select historical data period"
    )
    
    interval = st.selectbox(
        "Data Interval",
        ['1d', '1wk', '1mo'],
        index=0,
        help="Select data frequency"
    )
    
    # Fetch data button
    if st.button("🔍 Fetch Data", use_container_width=True, type="primary"):
        if not stock_input:
            st.error("Please enter a stock symbol")
        else:
            with st.spinner(f"Fetching data for {stock_input}..."):
                try:
                    # Fetch data
                    data = fetcher.fetch_stock_data(stock_input, period, interval)
                    
                    # Check if data is valid
                    if data is not None and not data.empty and 'Close' in data.columns:
                        st.session_state.data = data
                        st.session_state.current_ticker = stock_input
                        
                        # Perform technical analysis
                        st.session_state.analyzer = TechnicalAnalyzer(data)
                        
                        # Fetch stock info
                        try:
                            st.session_state.stock_info = fetcher.fetch_stock_info(stock_input)
                        except:
                            st.session_state.stock_info = {}
                        
                        st.success(f"✅ Successfully loaded data for {stock_input}")
                        st.rerun()
                    else:
                        st.error(f"❌ No data found for symbol: {stock_input}")
                        st.info("Tips:\n- US stocks: Use symbol directly (e.g., AAPL, GOOGL)\n- Indian stocks: Add .NS suffix (e.g., RELIANCE.NS)\n- Indices: Use ^ prefix (e.g., ^GSPC, ^NSEI)")
                        st.session_state.data = None
                        st.session_state.analyzer = None
                        
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
                    st.session_state.data = None
                    st.session_state.analyzer = None
    
    st.markdown("---")
    
    # Navigation
    st.markdown("## 📋 Navigation")
    
    if st.session_state.data is not None:
        tab = st.radio(
            "Select Section",
            ["📊 Overview", "📈 Technical Analysis", "🎯 Trend Analysis", 
             "🔮 Price Predictor", "📊 Model Comparison", "ℹ️ About"],
            label_visibility="collapsed"
        )
    else:
        tab = "ℹ️ About"

# Main content area
if st.session_state.data is None:
    # Welcome screen
    st.markdown("## 👋 Welcome to StockPro!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🚀 Getting Started:
        
        1. **Enter a stock symbol** in the sidebar
           - US Stocks: `AAPL`, `GOOGL`, `MSFT`, `TSLA`
           - Indian Stocks: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
           - Indices: `^GSPC`, `^NSEI`, `^DJI`
        
        2. **Select time period** and data interval
        
        3. **Click 'Fetch Data'** to load stock data
        
        4. **Explore various analysis tabs**:
           - 📊 **Overview**: Price charts and key statistics
           - 📈 **Technical Analysis**: RSI, MACD, Bollinger Bands
           - 🎯 **Trend Analysis**: Support/Resistance levels
           - 🔮 **Price Predictor**: AI-powered predictions
           - 📊 **Model Comparison**: Compare prediction models
        
        ### ✨ Features:
        - 📊 **Interactive Charts** with candlestick patterns
        - 📈 **10+ Technical Indicators**
        - 🤖 **Multiple ML Prediction Models**
        - 📉 **Risk Analysis & Metrics**
        - 📱 **Responsive Design**
        """)
    
    with col2:
        st.markdown("### 📈 Quick Market Overview")
        
        # Show sample market data
        try:
            sp500 = fetcher.fetch_stock_data('^GSPC', '5d')
            nifty = fetcher.fetch_stock_data('^NSEI', '5d')
            
            if not sp500.empty and 'Close' in sp500.columns:
                sp_change = ((sp500['Close'].iloc[-1] - sp500['Close'].iloc[-2]) / sp500['Close'].iloc[-2]) * 100
                st.metric("S&P 500", f"${sp500['Close'].iloc[-1]:,.2f}", f"{sp_change:.2f}%")
            
            if not nifty.empty and 'Close' in nifty.columns:
                nifty_change = ((nifty['Close'].iloc[-1] - nifty['Close'].iloc[-2]) / nifty['Close'].iloc[-2]) * 100
                st.metric("NIFTY 50", f"₹{nifty['Close'].iloc[-1]:,.2f}", f"{nifty_change:.2f}%")
        except:
            st.info("Market data will appear here when fetched")
        
        st.markdown("### 💡 Pro Tips")
        st.info("""
        - Use `.NS` suffix for NSE stocks
        - Use `.BO` suffix for BSE stocks
        - Check model comparison for best predictions
        - Longer time periods give better analysis
        """)
    
else:
    data = st.session_state.data.copy()  # Create a copy to avoid warnings
    analyzer = st.session_state.analyzer
    
    # Ensure data has required columns
    if 'Close' not in data.columns:
        st.error("Data does not contain required 'Close' column. Please fetch data again.")
        st.stop()
    
    # ==================== OVERVIEW TAB ====================
    if tab == "📊 Overview":
        st.markdown("## 📊 Stock Overview")
        
        # Stock info header
        if st.session_state.stock_info:
            info = st.session_state.stock_info
            st.markdown(f"### {info.get('Name', st.session_state.current_ticker)}")
            
            if info.get('Sector') and info.get('Sector') != 'N/A':
                st.markdown(f"*{info.get('Sector', '')} | {info.get('Industry', '')}*")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{price_change_pct:.2f}%",
                    delta_color="normal"
                )
            
            with col2:
                if 'Volume' in data.columns:
                    volume = data['Volume'].iloc[-1]
                    avg_volume = data['Volume'].mean()
                    vol_change = ((volume/avg_volume) - 1) * 100 if avg_volume > 0 else 0
                    st.metric(
                        "Volume",
                        format_number(volume),
                        f"{vol_change:.1f}% vs avg"
                    )
                else:
                    st.metric("Volume", "N/A")
            
            with col3:
                high_52 = data['High'].max()
                low_52 = data['Low'].min()
                st.metric("52W Range", f"${low_52:.2f} - ${high_52:.2f}")
            
            with col4:
                volatility = calculate_volatility(data).iloc[-1] * 100 if len(data) > 20 else 0
                st.metric("Volatility (Ann.)", f"{volatility:.2f}%")
        
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
        
        # Price chart
        st.markdown("### 📈 Price Chart")
        try:
            chart_data = data.tail(100) if len(data) > 100 else data
            fig = visualizer.create_candlestick_chart(
                chart_data,
                title=f"{st.session_state.current_ticker} - Price Chart",
                show_volume=True,
                mas=[20, 50]
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            # Fallback: simple line chart
            st.line_chart(data['Close'])
        
        # Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Price Statistics")
            try:
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max'],
                    'Value': [
                        f"${data['Close'].mean():.2f}",
                        f"${data['Close'].std():.2f}",
                        f"${data['Close'].min():.2f}",
                        f"${data['Close'].quantile(0.25):.2f}",
                        f"${data['Close'].median():.2f}",
                        f"${data['Close'].quantile(0.75):.2f}",
                        f"${data['Close'].max():.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error calculating statistics: {str(e)}")
        
        with col2:
            st.markdown("### 📊 Returns Analysis")
            try:
                returns = data['Close'].pct_change().dropna()
                
                if len(returns) > 0:
                    cumulative_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                    positive_days = (returns > 0).sum()
                    
                    returns_stats = pd.DataFrame({
                        'Metric': ['Daily Return Mean', 'Daily Return Std', 
                                  'Cumulative Return', 'Positive Days %'],
                        'Value': [
                            f"{returns.mean()*100:.3f}%",
                            f"{returns.std()*100:.3f}%",
                            f"{cumulative_return:.2f}%",
                            f"{(positive_days/len(returns))*100:.1f}%"
                        ]
                    })
                    st.dataframe(returns_stats, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error calculating returns: {str(e)}")
    
    # ==================== TECHNICAL ANALYSIS TAB ====================
    elif tab == "📈 Technical Analysis":
        st.markdown("## 📈 Technical Analysis")
        
        # Check if analyzer exists and has data
        if analyzer is None or data is None:
            st.error("No analysis data available. Please fetch stock data first.")
        else:
            # Indicator selection
            indicator = st.selectbox(
                "Select Technical Indicator",
                ['RSI', 'MACD', 'Bollinger Bands', 'Stochastic Oscillator']
            )
            
            try:
                # Create indicator chart
                indicator_map = {
                    'RSI': 'RSI',
                    'MACD': 'MACD',
                    'Bollinger Bands': 'Bollinger',
                    'Stochastic Oscillator': 'Stochastic'
                }
                
                plot_data = data.tail(200) if len(data) > 200 else data
                fig = visualizer.create_technical_indicator_chart(
                    plot_data, 
                    indicator_map[indicator]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Indicator explanation
                with st.expander("ℹ️ Understanding the Indicator"):
                    explanations = {
                        'RSI': """
                        **Relative Strength Index (RSI)** measures momentum.
                        - **Above 70**: Overbought (potential sell)
                        - **Below 30**: Oversold (potential buy)
                        - **Divergence**: Price vs RSI divergence signals reversal
                        """,
                        'MACD': """
                        **MACD** shows trend direction and momentum.
                        - **MACD > Signal**: Bullish momentum
                        - **MACD < Signal**: Bearish momentum
                        - **Histogram**: Shows strength of trend
                        """,
                        'Bollinger Bands': """
                        **Bollinger Bands** measure volatility.
                        - **Price at Upper Band**: Potentially overbought
                        - **Price at Lower Band**: Potentially oversold
                        - **Band Squeeze**: Low volatility, breakout imminent
                        """,
                        'Stochastic Oscillator': """
                        **Stochastic** compares closing price to price range.
                        - **Above 80**: Overbought
                        - **Below 20**: Oversold
                        - **%K crosses %D**: Trading signal
                        """
                    }
                    st.markdown(explanations[indicator])
                
                # Latest signals
                st.markdown("### 📡 Latest Technical Signals")
                
                summary = analyzer.get_summary_statistics()
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    rsi_value = summary.get('RSI', 50)
                    rsi_emoji = "🔴" if rsi_value > 70 else "🟢" if rsi_value < 30 else "🟡"
                    st.markdown(f"**{rsi_emoji} RSI:** {rsi_value:.1f}")
                
                with col2:
                    trend = summary.get('Trend', 'N/A')
                    trend_emoji = "🟢" if trend == 'Uptrend' else "🔴" if trend == 'Downtrend' else "🟡"
                    st.markdown(f"**{trend_emoji} Trend:** {trend}")
                
                with col3:
                    if 'MACD_Crossover' in data.columns:
                        macd_signal = data['MACD_Crossover'].iloc[-1]
                        macd_emoji = "🟢" if macd_signal == 'Bullish' else "🔴" if macd_signal == 'Bearish' else "🟡"
                        st.markdown(f"**{macd_emoji} MACD:** {macd_signal}")
                
                with col4:
                    bb_pos = summary.get('BB_Position', 'Middle')
                    bb_emoji = "🔴" if bb_pos == 'Upper' else "🟢" if bb_pos == 'Lower' else "🟡"
                    st.markdown(f"**{bb_emoji} Bollinger:** {bb_pos}")
                    
            except Exception as e:
                st.error(f"Error in technical analysis: {str(e)}")
                st.info("Try fetching more data or selecting a different indicator")
    
    # ==================== TREND ANALYSIS TAB ====================
    elif tab == "🎯 Trend Analysis":
        st.markdown("## 🎯 Trend Analysis")
        
        if analyzer is None:
            st.error("No analysis data available. Please fetch stock data first.")
        else:
            try:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Trend Direction")
                    if 'Trend' in data.columns:
                        trend_counts = data['Trend'].value_counts()
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=trend_counts.index,
                            values=trend_counts.values,
                            hole=0.4,
                            marker_colors=['#00ff88', '#ff4444', '#ffaa00']
                        )])
                        fig.update_layout(
                            template='plotly_dark',
                            title='Trend Distribution',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 📊 Trend Strength (ADX)")
                    trend_strength = analyzer.get_trend_strength()
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=trend_strength,
                        title={'text': "ADX"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#667eea"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 50], 'color': "gray"},
                                {'range': [50, 75], 'color': "darkgray"},
                                {'range': [75, 100], 'color': "#667eea"}
                            ]
                        }
                    ))
                    fig.update_layout(template='plotly_dark', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Support and Resistance
                st.markdown("### 🎯 Support & Resistance Levels")
                
                fig = go.Figure()
                
                plot_data = data.tail(100) if len(data) > 100 else data
                
                fig.add_trace(go.Candlestick(
                    x=plot_data.index,
                    open=plot_data['Open'],
                    high=plot_data['High'],
                    low=plot_data['Low'],
                    close=plot_data['Close'],
                    name='Price'
                ))
                
                if 'Dynamic_Support' in plot_data.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_data.index,
                        y=plot_data['Dynamic_Support'],
                        name='Support',
                        line=dict(color='#00ff88', width=2, dash='dash')
                    ))
                
                if 'Dynamic_Resistance' in plot_data.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_data.index,
                        y=plot_data['Dynamic_Resistance'],
                        name='Resistance',
                        line=dict(color='#ff4444', width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    template='plotly_dark',
                    title='Support & Resistance Levels',
                    yaxis_title='Price',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in trend analysis: {str(e)}")
    
    # ==================== PRICE PREDICTOR TAB ====================
    elif tab == "🔮 Price Predictor":
        st.markdown("## 🔮 Price Predictor")
        
        # Model selection
        models_available = []
        if LSTM_AVAILABLE:
            models_available.append('LSTM Neural Network')
        if PROPHET_AVAILABLE:
            models_available.append('Facebook Prophet')
        if ARIMA_AVAILABLE:
            models_available.append('ARIMA')
        if ML_AVAILABLE:
            models_available.extend(['XGBoost', 'LightGBM', 'Linear Regression'])
        
        if not models_available:
            st.error("No prediction models are available. Please check your installation.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                model_type = st.selectbox("Select Prediction Model", models_available)
            
            with col2:
                days_ahead = st.number_input(
                    "Prediction Days",
                    min_value=7,
                    max_value=90,
                    value=30,
                    step=1
                )
            
            if st.button("🚀 Generate Predictions", use_container_width=True, type="primary"):
                with st.spinner(f"Training {model_type} and generating predictions..."):
                    try:
                        predictions = None
                        future_dates = pd.date_range(
                            start=data.index[-1] + timedelta(days=1),
                            periods=days_ahead,
                            freq='B'  # Business days
                        )
                        
                        if model_type == 'LSTM Neural Network' and LSTM_AVAILABLE:
                            model = LSTMPredictor(sequence_length=min(60, len(data)//2))
                            model.train(data, epochs=20, verbose=0)
                            predictions = model.predict(data, days_ahead)
                            
                        elif model_type == 'Facebook Prophet' and PROPHET_AVAILABLE:
                            model = ProphetPredictor()
                            model.train(data)
                            predictions_df, _, _ = model.predict(data, days_ahead)
                            predictions = predictions_df['yhat'].values
                            
                        elif model_type == 'ARIMA' and ARIMA_AVAILABLE:
                            model = ARIMAPredictor()
                            close_prices = data['Close'].values
                            model.train(close_prices)
                            predictions, _ = model.predict(close_prices, days_ahead)
                            
                        elif model_type == 'XGBoost' and ML_AVAILABLE:
                            model = MLPredictor(model_type='xgboost')
                            model.train(data)
                            predictions = model.predict(data, days_ahead)
                            
                        elif model_type == 'LightGBM' and ML_AVAILABLE:
                            model = MLPredictor(model_type='lightgbm')
                            model.train(data)
                            predictions = model.predict(data, days_ahead)
                            
                        elif model_type == 'Linear Regression' and ML_AVAILABLE:
                            model = MLPredictor(model_type='linear')
                            model.train(data)
                            predictions = model.predict(data, days_ahead)
                        
                        if predictions is not None:
                            st.session_state.predictions[model_type] = {
                                'predictions': predictions,
                                'dates': future_dates
                            }
                            st.success(f"✅ {model_type} predictions generated!")
                            st.rerun()
                        else:
                            st.error("Failed to generate predictions")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Try a different model or adjust parameters")
            
            # Display predictions if available
            if model_type in st.session_state.predictions:
                pred_data = st.session_state.predictions[model_type]
                
                st.markdown("### 📈 Prediction Results")
                
                fig = visualizer.create_prediction_chart(
                    data.tail(100),
                    pred_data['predictions'],
                    pred_data['dates'],
                    f"{model_type} - {days_ahead} Days Prediction"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction metrics
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = data['Close'].iloc[-1]
                
                with col1:
                    final_pred = pred_data['predictions'][-1]
                    change = ((final_pred - current_price) / current_price) * 100
                    st.metric("Predicted Price", f"${final_pred:.2f}", f"{change:.2f}%")
                
                with col2:
                    st.metric("Min", f"${pred_data['predictions'].min():.2f}")
                
                with col3:
                    st.metric("Max", f"${pred_data['predictions'].max():.2f}")
                
                with col4:
                    st.metric("Avg", f"${pred_data['predictions'].mean():.2f}")
                
                # Download button
                pred_df = pd.DataFrame({
                    'Date': pred_data['dates'].strftime('%Y-%m-%d'),
                    'Predicted Price': pred_data['predictions']
                })
                
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Predictions",
                    csv,
                    f"predictions_{st.session_state.current_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
    
    # ==================== MODEL COMPARISON TAB ====================
    elif tab == "📊 Model Comparison":
        st.markdown("## 📊 Model Comparison")
        
        if len(st.session_state.predictions) < 1:
            st.info("Generate predictions in the 'Price Predictor' tab first")
        else:
            predictions_dict = {}
            for name, pred in st.session_state.predictions.items():
                predictions_dict[name] = pred['predictions']
            
            first_model = list(st.session_state.predictions.keys())[0]
            dates = st.session_state.predictions[first_model]['dates']
            
            fig = visualizer.create_model_comparison_chart(
                predictions_dict,
                title="Model Predictions Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== ABOUT TAB ====================
    elif tab == "ℹ️ About":
        st.markdown("## ℹ️ About StockPro")
        
        st.markdown("""
        ### 🚀 StockPro - Smart Stock Analysis
        
        A comprehensive stock analysis tool combining technical analysis with machine learning.
        
        #### Features:
        - Real-time stock data via Yahoo Finance
        - 10+ technical indicators
        - Multiple ML prediction models
        - Interactive visualizations
        - Model comparison tools
        
        #### Tech Stack:
        - **Frontend**: Streamlit
        - **Data**: yfinance, pandas
        - **Visualization**: Plotly
        - **ML**: scikit-learn, TensorFlow, XGBoost, Prophet
        
        ⚠️ **Disclaimer**: For educational purposes only. Not financial advice.
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 1rem; font-size: 0.9rem;'>
        📈 StockPro v1.0 | Built with Streamlit | 
        ⚠️ For educational purposes only
    </div>
    """,
    unsafe_allow_html=True
)