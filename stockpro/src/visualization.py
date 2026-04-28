"""
Visualization module for StockPro using Plotly
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

class StockVisualizer:
    """Class for creating interactive stock visualizations"""
    
    def __init__(self, theme='plotly_dark'):
        self.theme = theme
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'neutral': '#ffaa00',
            'volume': '#4488ff',
            'ma_colors': ['#ffaa00', '#ff8800', '#ff6600', '#ff4400'],
            'bb_band': 'rgba(128, 128, 128, 0.2)'
        }
    
    def create_candlestick_chart(self, data: pd.DataFrame, title: str = "Stock Price",
                                 show_volume: bool = True,
                                 mas: List[int] = [20, 50],
                                 bollinger: bool = False) -> go.Figure:
        """Create interactive candlestick chart with technical indicators"""
        
        # Calculate number of rows
        rows = 2 if show_volume else 1
        row_heights = [0.7, 0.3] if show_volume else [1.0]
        
        # Create subplots
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=(title, 'Volume') if show_volume else (title,)
        )
        
        # Convert index to string format to avoid timestamp issues
        x_dates = data.index
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=x_dates,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        for i, ma in enumerate(mas):
            if f'SMA_{ma}' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_dates,
                        y=data[f'SMA_{ma}'],
                        name=f'SMA {ma}',
                        line=dict(color=self.colors['ma_colors'][i % len(self.colors['ma_colors'])], width=1.5),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands
        if bollinger and 'BB_Upper' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_dates,
                    y=data['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    opacity=0.5
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x_dates,
                    y=data['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    opacity=0.5,
                    fill='tonexty',
                    fillcolor=self.colors['bb_band']
                ),
                row=1, col=1
            )
        
        # Add volume bars
        if show_volume and 'Volume' in data.columns:
            colors = ['#00ff88' if data['Close'].iloc[i] >= data['Open'].iloc[i] 
                     else '#ff4444' for i in range(len(data))]
            
            fig.add_trace(
                go.Bar(
                    x=x_dates,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.5
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            template=self.theme,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=rows, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        if show_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def create_technical_indicator_chart(self, data: pd.DataFrame, 
                                        indicator: str = 'RSI') -> go.Figure:
        """Create chart for specific technical indicator"""
        
        x_dates = data.index
        
        if indicator == 'RSI' and 'RSI' in data.columns:
            fig = go.Figure()
            
            # Add RSI line
            fig.add_trace(go.Scatter(
                x=x_dates,
                y=data['RSI'],
                name='RSI',
                line=dict(color='#4488ff', width=2)
            ))
            
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Overbought", opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         annotation_text="Oversold", opacity=0.5)
            
            fig.update_layout(
                template=self.theme,
                title='Relative Strength Index (RSI)',
                yaxis_title='RSI',
                yaxis_range=[0, 100]
            )
            
        elif indicator == 'MACD' and all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.05,
                              row_heights=[0.7, 0.3])
            
            # MACD Line and Signal
            fig.add_trace(
                go.Scatter(x=x_dates, y=data['MACD'], name='MACD',
                          line=dict(color='#4488ff', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=x_dates, y=data['MACD_Signal'], name='Signal',
                          line=dict(color='#ff8800', width=1.5)),
                row=1, col=1
            )
            
            # MACD Histogram
            colors = ['#00ff88' if val >= 0 else '#ff4444' 
                     for val in data['MACD_Histogram']]
            fig.add_trace(
                go.Bar(x=x_dates, y=data['MACD_Histogram'], name='Histogram',
                      marker_color=colors),
                row=2, col=1
            )
            
            fig.update_layout(
                template=self.theme,
                title='MACD Indicator',
                yaxis_title='MACD',
                yaxis2_title='Histogram'
            )
            
        elif indicator == 'Stochastic' and all(col in data.columns for col in ['Stoch_K', 'Stoch_D']):
            fig = go.Figure()
            
            # %K and %D lines
            fig.add_trace(go.Scatter(
                x=x_dates, y=data['Stoch_K'],
                name='%K', line=dict(color='#4488ff', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=x_dates, y=data['Stoch_D'],
                name='%D', line=dict(color='#ff8800', width=1.5)
            ))
            
            # Overbought/Oversold lines
            fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5)
            fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5)
            
            fig.update_layout(
                template=self.theme,
                title='Stochastic Oscillator',
                yaxis_title='Value',
                yaxis_range=[0, 100]
            )
        
        elif indicator == 'Bollinger' and all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            fig = go.Figure()
            
            # Price
            fig.add_trace(go.Scatter(
                x=x_dates, y=data['Close'],
                name='Close Price',
                line=dict(color='#ffffff', width=2)
            ))
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(
                x=x_dates, y=data['BB_Upper'],
                name='Upper Band',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=x_dates, y=data['BB_Middle'],
                name='Middle Band',
                line=dict(color='#4488ff', width=1),
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=x_dates, y=data['BB_Lower'],
                name='Lower Band',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.7,
                fill='tonexty',
                fillcolor=self.colors['bb_band']
            ))
            
            fig.update_layout(
                template=self.theme,
                title='Bollinger Bands',
                yaxis_title='Price'
            )
        else:
            # Fallback: simple line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_dates, y=data['Close'],
                name='Close Price',
                line=dict(color='#4488ff', width=2)
            ))
            fig.update_layout(
                template=self.theme,
                title='Price Chart',
                yaxis_title='Price'
            )
        
        return fig
    
    def create_prediction_chart(self, historical_data: pd.DataFrame,
                               predictions: np.ndarray,
                               future_dates: pd.DatetimeIndex,
                               title: str = "Price Prediction",
                               confidence_intervals: Optional[Dict] = None) -> go.Figure:
        """Create prediction visualization chart"""
        
        fig = go.Figure()
        
        # Convert dates to string format to avoid timestamp issues
        hist_dates = historical_data.index
        future_dates_str = future_dates
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=historical_data['Close'],
            name='Historical Price',
            line=dict(color='#4488ff', width=2),
            mode='lines'
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=future_dates_str,
            y=predictions,
            name='Predicted Price',
            line=dict(color='#00ff88', width=2, dash='dash'),
            mode='lines+markers'
        ))
        
        # Confidence intervals (if provided)
        if confidence_intervals:
            fig.add_trace(go.Scatter(
                x=future_dates_str,
                y=confidence_intervals['upper'],
                name='Upper Bound',
                line=dict(color='gray', width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates_str,
                y=confidence_intervals['lower'],
                name='Lower Bound',
                line=dict(color='gray', width=0),
                fill='tonexty',
                fillcolor='rgba(0, 255, 136, 0.2)',
                showlegend=False
            ))
        
        # Add vertical line separating historical and prediction
        # Use add_shape instead of add_vline to avoid timestamp issues
        if len(hist_dates) > 0:
            separation_date = hist_dates[-1]
            fig.add_shape(
                type="line",
                x0=separation_date,
                y0=0,
                x1=separation_date,
                y1=1,
                yref="paper",
                line=dict(
                    color="yellow",
                    width=2,
                    dash="dash",
                )
            )
            
            # Add annotation for the vertical line
            fig.add_annotation(
                x=separation_date,
                y=1,
                yref="paper",
                text="Prediction Start",
                showarrow=False,
                yshift=10,
                font=dict(color="yellow", size=10)
            )
        
        fig.update_layout(
            template=self.theme,
            title=title,
            yaxis_title='Price',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_model_comparison_chart(self, predictions_dict: Dict[str, np.ndarray],
                                     title: str = "Model Comparison",
                                     actual_values: np.ndarray = None) -> go.Figure:
        """Create chart comparing different model predictions"""
        
        fig = go.Figure()
        
        # Add actual values if provided
        if actual_values is not None:
            fig.add_trace(go.Scatter(
                x=list(range(len(actual_values))),
                y=actual_values,
                name='Actual',
                line=dict(color='white', width=3)
            ))
        
        # Add predictions from each model
        colors = ['#00ff88', '#4488ff', '#ff8800', '#ff4444', '#ff00ff']
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            fig.add_trace(go.Scatter(
                x=list(range(len(predictions))),
                y=predictions,
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                mode='lines+markers'
            ))
        
        fig.update_layout(
            template=self.theme,
            title=title,
            yaxis_title='Price',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_risk_metrics_chart(self, metrics: Dict) -> go.Figure:
        """Create risk metrics visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Volatility', 'Sharpe Ratio', 'Beta', 'Max Drawdown'],
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]]
        )
        
        # Volatility gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get('volatility', 0) * 100,
                title={'text': "Volatility (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#ff8800"},
                       'steps': [
                           {'range': [0, 20], 'color': "lightgray"},
                           {'range': [20, 50], 'color': "gray"},
                           {'range': [50, 100], 'color': "darkgray"}
                       ]}
            ),
            row=1, col=1
        )
        
        # Sharpe Ratio gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get('sharpe_ratio', 0),
                title={'text': "Sharpe Ratio"},
                gauge={'axis': {'range': [-2, 4]},
                       'bar': {'color': "#4488ff"},
                       'steps': [
                           {'range': [-2, 0], 'color': "lightgray"},
                           {'range': [0, 1], 'color': "gray"},
                           {'range': [1, 4], 'color': "darkgray"}
                       ]}
            ),
            row=1, col=2
        )
        
        # Beta indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=metrics.get('beta', 1),
                title={'text': "Beta"},
                number={'suffix': " β"},
                delta={'reference': 1, 'relative': False}
            ),
            row=2, col=1
        )
        
        # Max Drawdown indicator
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=metrics.get('max_drawdown', 0) * 100,
                title={'text': "Max Drawdown (%)"},
                number={'suffix': "%"}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            template=self.theme,
            height=500
        )
        
        return fig
    
    def create_stock_comparison_chart(self, normalized_data: pd.DataFrame,
                                    title: str = "Stock Comparison") -> go.Figure:
        """Create stock comparison chart with normalized prices"""
        
        fig = go.Figure()
        
        colors = ['#00ff88', '#4488ff', '#ff8800', '#ff4444', '#ff00ff']
        
        for i, column in enumerate(normalized_data.columns):
            fig.add_trace(go.Scatter(
                x=normalized_data.index,
                y=normalized_data[column],
                name=column,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            template=self.theme,
            title=title,
            yaxis_title='Normalized Price (Base=100)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_simple_line_chart(self, data: pd.Series, title: str = "Price Chart") -> go.Figure:
        """Create a simple line chart (fallback method)"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            name='Price',
            line=dict(color='#4488ff', width=2)
        ))
        
        fig.update_layout(
            template=self.theme,
            title=title,
            yaxis_title='Price',
            hovermode='x unified'
        )
        
        return fig