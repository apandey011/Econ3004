"""
Visualization Module
Creates charts and plots for analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
import os


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_equity_curve(
    portfolio_values: list,
    baseline_values: list = None,
    title: str = "Portfolio Equity Curve",
    save_path: str = None
):
    """
    Plot portfolio value over time.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(portfolio_values, label='Strategy', linewidth=2, color='#2E86AB')
    
    if baseline_values:
        ax.plot(baseline_values, label='Buy & Hold', linewidth=2, 
                color='#A23B72', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: str = None
):
    """
    Plot top feature importances.
    """
    df = importance_df.head(top_n).copy()
    df = df.sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    ax.barh(df['feature'], df['importance'], color=colors)
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_prediction_vs_actual(
    predictions: pd.DataFrame,
    ticker: str = None,
    n_points: int = 100,
    save_path: str = None
):
    """
    Plot predicted vs actual returns.
    """
    df = predictions.copy()
    
    if ticker:
        df = df[df['ticker'] == ticker]
    
    df = df.tail(n_points)
    
    if len(df) == 0:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Time series of predictions vs actuals
    ax1 = axes[0]
    ax1.plot(df.index, df['predicted_return'] * 100, label='Predicted', 
             linewidth=2, color='#2E86AB')
    
    if 'target_return' in df.columns:
        ax1.plot(df.index, df['target_return'] * 100, label='Actual',
                 linewidth=2, color='#A23B72', alpha=0.7)
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Return (%)', fontsize=12)
    ax1.set_title(f'Predicted vs Actual Returns{" - " + ticker if ticker else ""}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    ax2 = axes[1]
    if 'target_return' in df.columns:
        ax2.scatter(df['target_return'] * 100, df['predicted_return'] * 100,
                    alpha=0.5, c='#2E86AB')
        
        # Add perfect prediction line
        min_val = min(df['target_return'].min(), df['predicted_return'].min()) * 100
        max_val = max(df['target_return'].max(), df['predicted_return'].max()) * 100
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', 
                 label='Perfect Prediction', alpha=0.7)
        
        ax2.set_xlabel('Actual Return (%)', fontsize=12)
        ax2.set_ylabel('Predicted Return (%)', fontsize=12)
        ax2.set_title('Prediction Accuracy Scatter', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_confidence_distribution(
    predictions: pd.DataFrame,
    save_path: str = None
):
    """
    Plot distribution of prediction confidences.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(predictions['confidence'], bins=50, color='#2E86AB', 
             edgecolor='white', alpha=0.8)
    ax1.axvline(x=0.55, color='red', linestyle='--', label='Trade Threshold (55%)')
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot by direction
    ax2 = axes[1]
    predictions['direction_label'] = predictions['predicted_direction'].map({0: 'Down', 1: 'Up'})
    sns.boxplot(data=predictions, x='direction_label', y='confidence', ax=ax2,
                palette=['#A23B72', '#2E86AB'])
    ax2.set_xlabel('Predicted Direction', fontsize=12)
    ax2.set_ylabel('Confidence', fontsize=12)
    ax2.set_title('Confidence by Prediction Direction', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_trade_analysis(
    trade_history: pd.DataFrame,
    save_path: str = None
):
    """
    Analyze and visualize trade performance.
    """
    if len(trade_history) == 0:
        print("No trades to analyze")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. P&L Distribution
    ax1 = axes[0, 0]
    colors = ['#2E86AB' if x > 0 else '#A23B72' for x in trade_history['pnl_pct']]
    ax1.hist(trade_history['pnl_pct'] * 100, bins=50, color='#2E86AB', 
             edgecolor='white', alpha=0.8)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('P&L (%)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Win rate by direction
    ax2 = axes[0, 1]
    trade_history['win'] = trade_history['pnl_pct'] > 0
    win_rate = trade_history.groupby('direction')['win'].mean() * 100
    win_rate.plot(kind='bar', ax=ax2, color=['#A23B72', '#2E86AB'], edgecolor='white')
    ax2.set_xlabel('Direction', fontsize=12)
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2.set_title('Win Rate by Direction', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Cumulative P&L
    ax3 = axes[1, 0]
    cumulative_pnl = (1 + trade_history['pnl_pct']).cumprod() - 1
    ax3.plot(cumulative_pnl.values * 100, linewidth=2, color='#2E86AB')
    ax3.fill_between(range(len(cumulative_pnl)), cumulative_pnl.values * 100, 
                     alpha=0.3, color='#2E86AB')
    ax3.set_xlabel('Trade Number', fontsize=12)
    ax3.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax3.set_title('Cumulative Trade Returns', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Confidence vs P&L
    ax4 = axes[1, 1]
    ax4.scatter(trade_history['confidence'], trade_history['pnl_pct'] * 100,
                alpha=0.5, c='#2E86AB')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Confidence', fontsize=12)
    ax4.set_ylabel('P&L (%)', fontsize=12)
    ax4.set_title('Confidence vs Trade P&L', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_stock_with_signals(
    df: pd.DataFrame,
    ticker: str,
    predictions: pd.DataFrame = None,
    last_n_days: int = 60,
    save_path: str = None
):
    """
    Plot stock price with buy/sell signals.
    """
    stock_data = df[df['ticker'] == ticker].sort_values('date').tail(last_n_days)
    
    if len(stock_data) == 0:
        print(f"No data for {ticker}")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Price with moving averages
    ax1 = axes[0]
    ax1.plot(stock_data['date'], stock_data['close'], label='Close', 
             linewidth=2, color='#2E86AB')
    
    if 'sma_20' in stock_data.columns:
        ax1.plot(stock_data['date'], stock_data['sma_20'], label='SMA 20', 
                 linestyle='--', alpha=0.7, color='#F18F01')
    if 'sma_50' in stock_data.columns:
        ax1.plot(stock_data['date'], stock_data['sma_50'], label='SMA 50', 
                 linestyle='--', alpha=0.7, color='#A23B72')
    
    # Add signals if predictions available
    if predictions is not None:
        pred_data = predictions[predictions['ticker'] == ticker]
        pred_data = pred_data[pred_data['date'].isin(stock_data['date'])]
        
        # Buy signals (high confidence UP)
        buys = pred_data[(pred_data['predicted_direction'] == 1) & 
                         (pred_data['confidence'] >= 0.6)]
        if len(buys) > 0:
            buy_prices = stock_data[stock_data['date'].isin(buys['date'])]['close']
            ax1.scatter(buys['date'], buy_prices, marker='^', s=100, 
                       color='green', label='Buy Signal', zorder=5)
        
        # Sell signals (high confidence DOWN)
        sells = pred_data[(pred_data['predicted_direction'] == 0) & 
                          (pred_data['confidence'] >= 0.6)]
        if len(sells) > 0:
            sell_prices = stock_data[stock_data['date'].isin(sells['date'])]['close']
            ax1.scatter(sells['date'], sell_prices, marker='v', s=100, 
                       color='red', label='Sell Signal', zorder=5)
    
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'{ticker} - Price Action with Signals', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. RSI
    ax2 = axes[1]
    if 'rsi_14' in stock_data.columns:
        ax2.plot(stock_data['date'], stock_data['rsi_14'], linewidth=2, color='#2E86AB')
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax2.fill_between(stock_data['date'], 30, 70, alpha=0.1, color='gray')
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_title('RSI (14)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # 3. Volume
    ax3 = axes[2]
    colors = ['#2E86AB' if c > o else '#A23B72' 
              for c, o in zip(stock_data['close'], stock_data['open'])]
    ax3.bar(stock_data['date'], stock_data['volume'], color=colors, alpha=0.7)
    
    if 'volume_sma_20' in stock_data.columns:
        ax3.plot(stock_data['date'], stock_data['volume_sma_20'], 
                 linestyle='--', color='orange', label='20-day Avg', linewidth=2)
    
    ax3.set_ylabel('Volume', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Trading Volume', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Format y-axis
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def create_all_visualizations(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    trade_history: pd.DataFrame,
    feature_importance: pd.DataFrame,
    portfolio_values: list,
    output_dir: str = "plots"
):
    """
    Create and save all visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating visualizations...")
    
    # Equity curve
    plot_equity_curve(
        portfolio_values,
        save_path=os.path.join(output_dir, "equity_curve.png")
    )
    
    # Feature importance
    if len(feature_importance) > 0:
        plot_feature_importance(
            feature_importance,
            save_path=os.path.join(output_dir, "feature_importance.png")
        )
    
    # Confidence distribution
    if 'confidence' in predictions.columns:
        plot_confidence_distribution(
            predictions,
            save_path=os.path.join(output_dir, "confidence_dist.png")
        )
    
    # Trade analysis
    if len(trade_history) > 0:
        plot_trade_analysis(
            trade_history,
            save_path=os.path.join(output_dir, "trade_analysis.png")
        )
    
    # Sample stock charts
    for ticker in ['AAPL', 'MSFT', 'GOOGL']:
        if ticker in df['ticker'].values:
            plot_stock_with_signals(
                df, ticker, predictions,
                save_path=os.path.join(output_dir, f"{ticker}_signals.png")
            )
    
    print(f"All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    print("Visualization module loaded successfully")

