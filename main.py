#!/usr/bin/env python3
"""
S&P 500 Stock Price Prediction Model
=====================================
A machine learning system for predicting short-term stock price movements.

Usage:
    python main.py                    # Run full pipeline
    python main.py --fetch-only       # Only fetch data
    python main.py --train-only       # Train on existing data
    python main.py --predict AAPL     # Predict for specific stock

Author: Econ 3004 Final Project
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import (
    fetch_multiple_stocks, 
    clean_data, 
    calculate_returns,
    save_data, 
    load_data,
    fetch_stock_data,
    get_sp500_tickers
)
from features import create_all_features, get_feature_columns
from models import StockPredictor, cross_validate_model
from backtester import Backtester, run_full_backtest, print_backtest_results


# Configuration
DATA_DIR = "data"
MODEL_DIR = "models"
DATA_FILE = os.path.join(DATA_DIR, "stock_data.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "stock_predictor.joblib")


def ensure_dirs():
    """Create necessary directories."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def fetch_data(tickers: list = None, period: str = "2y"):
    """Fetch and prepare stock data."""
    print("\n" + "="*60)
    print("STEP 1: FETCHING DATA")
    print("="*60)
    
    df = fetch_multiple_stocks(tickers=tickers, period=period)
    df = clean_data(df)
    df = calculate_returns(df)
    
    save_data(df, DATA_FILE)
    return df


def engineer_features(df):
    """Create all features."""
    print("\n" + "="*60)
    print("STEP 2: ENGINEERING FEATURES")
    print("="*60)
    
    df = create_all_features(df)
    return df


def train_model(df, model_type: str = "xgboost"):
    """Train the prediction model."""
    print("\n" + "="*60)
    print("STEP 3: TRAINING MODEL")
    print("="*60)
    
    feature_columns = get_feature_columns(df)
    print(f"Using {len(feature_columns)} features")
    
    # Cross-validate first
    cv_results = cross_validate_model(df, feature_columns, model_type=model_type)
    
    # Train final model on all data
    model = StockPredictor(model_type=model_type)
    model.fit(df, feature_columns)
    model.save(MODEL_FILE)
    
    return model, feature_columns


def backtest(model, df, feature_columns):
    """Run backtest."""
    print("\n" + "="*60)
    print("STEP 4: BACKTESTING")
    print("="*60)
    
    results = run_full_backtest(
        model=StockPredictor(model_type=model.model_type),  # Fresh model for fair test
        df=df,
        feature_columns=feature_columns,
        train_ratio=0.8,
        confidence_threshold=0.55
    )
    
    print_backtest_results(results)
    return results


def generate_prediction_with_explanation(
    model: StockPredictor,
    df_recent,
    ticker: str,
    feature_columns: list
):
    """
    Generate a prediction with confidence score and explanation.
    This is the key output - actionable trading advice.
    """
    # Get latest data for this ticker
    ticker_data = df_recent[df_recent['ticker'] == ticker].sort_values('date')
    
    if len(ticker_data) == 0:
        print(f"No data available for {ticker}")
        return None
    
    latest = ticker_data.iloc[-1]
    
    # Get prediction
    features = {col: latest[col] for col in feature_columns if col in latest.index}
    pred = model.predict_single(features)
    
    # Get feature importance for explanation
    importance = model.get_feature_importance()
    top_features = importance.head(10)
    
    # Build explanation
    print("\n" + "="*60)
    print(f"PREDICTION FOR {ticker}")
    print("="*60)
    print(f"Date: {latest['date'].strftime('%Y-%m-%d')}")
    print(f"Current Price: ${latest['close']:.2f}")
    
    print(f"\n🎯 PREDICTION: {pred['predicted_direction']}")
    print(f"   Expected Return: {pred['predicted_return']*100:.3f}%")
    print(f"   Confidence: {pred['confidence']*100:.1f}%")
    print(f"   Probability Up: {pred['prob_up']*100:.1f}%")
    print(f"   Probability Down: {pred['prob_down']*100:.1f}%")
    
    # Trading signal
    confidence = pred['confidence']
    direction = pred['predicted_direction']
    
    print(f"\n📊 TRADING SIGNAL:")
    if confidence >= 0.65:
        if direction == "UP":
            signal = "STRONG BUY"
            action = "Consider going LONG"
        else:
            signal = "STRONG SELL"
            action = "Consider going SHORT"
    elif confidence >= 0.55:
        if direction == "UP":
            signal = "MODERATE BUY"
            action = "Consider a small LONG position"
        else:
            signal = "MODERATE SELL"
            action = "Consider a small SHORT position"
    else:
        signal = "HOLD / NO TRADE"
        action = "Confidence too low - stay out"
    
    print(f"   Signal: {signal}")
    print(f"   Action: {action}")
    
    # Explanation based on key features
    print(f"\n📈 KEY FACTORS:")
    
    # RSI
    if 'rsi_14' in latest.index:
        rsi = latest['rsi_14']
        if rsi > 70:
            print(f"   • RSI ({rsi:.1f}): Overbought - potential pullback")
        elif rsi < 30:
            print(f"   • RSI ({rsi:.1f}): Oversold - potential bounce")
        else:
            print(f"   • RSI ({rsi:.1f}): Neutral momentum")
    
    # Moving averages
    if 'price_vs_sma_20' in latest.index:
        sma_pos = latest['price_vs_sma_20'] * 100
        print(f"   • Price vs 20-day SMA: {sma_pos:+.2f}%")
    
    # Volatility
    if 'volatility_20' in latest.index:
        vol = latest['volatility_20'] * 100
        print(f"   • 20-day Volatility: {vol:.2f}%")
    
    # Volume
    if 'relative_volume' in latest.index:
        rel_vol = latest['relative_volume']
        if rel_vol > 1.5:
            print(f"   • Volume: {rel_vol:.2f}x average (HIGH - strong interest)")
        elif rel_vol < 0.5:
            print(f"   • Volume: {rel_vol:.2f}x average (LOW - weak interest)")
        else:
            print(f"   • Volume: {rel_vol:.2f}x average (normal)")
    
    # MACD
    if 'macd_histogram' in latest.index:
        macd_hist = latest['macd_histogram']
        if macd_hist > 0:
            print(f"   • MACD: Bullish (histogram = {macd_hist:.3f})")
        else:
            print(f"   • MACD: Bearish (histogram = {macd_hist:.3f})")
    
    print("\n⚠️  DISCLAIMER: This is a model prediction, not financial advice.")
    print("    Always do your own research and manage risk appropriately.")
    
    return {
        'ticker': ticker,
        'date': latest['date'],
        'price': latest['close'],
        'prediction': pred,
        'signal': signal,
        'action': action
    }


def run_full_pipeline(
    tickers: list = None,
    period: str = "2y",
    model_type: str = "xgboost",
    skip_fetch: bool = False
):
    """
    Run the complete prediction pipeline.
    """
    ensure_dirs()
    
    print("\n" + "="*60)
    print("S&P 500 STOCK PREDICTION MODEL")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Step 1: Fetch or load data
    if skip_fetch and os.path.exists(DATA_FILE):
        print(f"\nLoading existing data from {DATA_FILE}")
        df = load_data(DATA_FILE)
        df = calculate_returns(df)
    else:
        df = fetch_data(tickers=tickers, period=period)
    
    # Step 2: Engineer features
    df = engineer_features(df)
    feature_columns = get_feature_columns(df)
    
    # Step 3: Train model
    model, feature_columns = train_model(df, model_type=model_type)
    
    # Step 4: Backtest
    backtest_results = backtest(model, df, feature_columns)
    
    # Step 5: Generate sample predictions
    print("\n" + "="*60)
    print("STEP 5: SAMPLE PREDICTIONS")
    print("="*60)
    
    # Get predictions for a few stocks
    sample_tickers = ['AAPL', 'MSFT', 'GOOGL']
    for ticker in sample_tickers:
        if ticker in df['ticker'].values:
            generate_prediction_with_explanation(model, df, ticker, feature_columns)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    
    return {
        'model': model,
        'data': df,
        'feature_columns': feature_columns,
        'backtest_results': backtest_results
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="S&P 500 Stock Price Prediction Model"
    )
    parser.add_argument(
        '--fetch-only',
        action='store_true',
        help='Only fetch and save data'
    )
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Use existing data file'
    )
    parser.add_argument(
        '--predict',
        type=str,
        help='Generate prediction for a specific ticker'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Specific tickers to use (default: SP500 sample)'
    )
    parser.add_argument(
        '--period',
        type=str,
        default='2y',
        help='Data period: 1y, 2y, 5y, max (default: 2y)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['ridge', 'xgboost', 'random_forest'],
        help='Model type (default: xgboost)'
    )
    
    args = parser.parse_args()
    
    if args.fetch_only:
        ensure_dirs()
        fetch_data(tickers=args.tickers, period=args.period)
        print("\nData fetching complete!")
        return
    
    if args.predict:
        # Quick prediction mode
        ensure_dirs()
        
        if os.path.exists(MODEL_FILE) and os.path.exists(DATA_FILE):
            model = StockPredictor.load(MODEL_FILE)
            df = load_data(DATA_FILE)
            df = calculate_returns(df)
            df = engineer_features(df)
            feature_columns = get_feature_columns(df)
            generate_prediction_with_explanation(model, df, args.predict.upper(), feature_columns)
        else:
            print("Model or data not found. Running full pipeline first...")
            results = run_full_pipeline(
                tickers=[args.predict.upper()],
                period=args.period,
                model_type=args.model
            )
            generate_prediction_with_explanation(
                results['model'],
                results['data'],
                args.predict.upper(),
                results['feature_columns']
            )
        return
    
    # Run full pipeline
    run_full_pipeline(
        tickers=args.tickers,
        period=args.period,
        model_type=args.model,
        skip_fetch=args.skip_fetch
    )


if __name__ == "__main__":
    main()

