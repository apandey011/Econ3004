#!/usr/bin/env python3
"""
S&P 500 Stock Price Prediction Model
=====================================
Predicts short-term price movements using machine learning.

Features:
- Ensemble of XGBoost, Random Forest, and Linear models
- Market regime detection (trades WITH the trend, not against it)
- 70+ technical indicators
- Confidence-based position sizing

Usage:
    python main.py                    # Full pipeline
    python main.py --no-regime        # Disable regime filter
    python main.py --skip-fetch       # Use cached data
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_fetcher import fetch_multiple_stocks, clean_data, calculate_returns, save_data, load_data
from src.features import create_all_features, get_feature_columns
from src.model import StockPredictor
from src.backtester import run_backtest, print_results, Backtester


DATA_DIR = "data"
MODEL_DIR = "models"
DATA_FILE = os.path.join(DATA_DIR, "stock_data.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "model.joblib")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def run_pipeline(skip_fetch: bool = False, use_regime_filter: bool = True):
    """Run the full prediction pipeline."""
    ensure_dirs()
    
    print("\n" + "="*70)
    print("S&P 500 STOCK PREDICTION MODEL")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Regime Filter: {'ON (trade WITH the trend)' if use_regime_filter else 'OFF'}")
    print("="*70)
    
    # Step 1: Data
    print("\n" + "="*70)
    print("STEP 1: DATA")
    print("="*70)
    
    if skip_fetch and os.path.exists(DATA_FILE):
        print(f"Loading cached data...")
        df = load_data(DATA_FILE)
        df = calculate_returns(df)
    else:
        print("Fetching fresh data...")
        df = fetch_multiple_stocks(period="2y")
        df = clean_data(df)
        df = calculate_returns(df)
        save_data(df, DATA_FILE)
    
    print(f"Data: {len(df)} rows, {df['ticker'].nunique()} stocks")
    print(f"Period: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Step 2: Features
    print("\n" + "="*70)
    print("STEP 2: FEATURES")
    print("="*70)
    
    df = create_all_features(df)
    feature_columns = get_feature_columns(df)
    
    # Step 3: Train Model
    print("\n" + "="*70)
    print("STEP 3: TRAINING MODEL")
    print("="*70)
    
    model = StockPredictor()
    results = run_backtest(
        model=model,
        df=df,
        feature_columns=feature_columns,
        train_ratio=0.80,
        use_regime_filter=use_regime_filter
    )
    model.save(MODEL_FILE)
    
    # Step 4: Top Features
    print("\n" + "="*70)
    print("STEP 4: TOP PREDICTIVE FEATURES")
    print("="*70)
    
    importance = model.get_feature_importance()
    print("\nMost important features for prediction:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Step 5: Trading Signals
    print("\n" + "="*70)
    print("STEP 5: TODAY'S TRADING SIGNALS")
    print("="*70)
    
    latest_date = df['date'].max()
    latest_data = df[df['date'] == latest_date].copy()
    
    if len(latest_data) > 0:
        predictions = model.predict(latest_data)
        predictions = predictions.merge(
            latest_data[['ticker', 'price_vs_sma_20', 'market_regime', 'rsi_14']],
            on='ticker', how='left'
        )
        
        strong_buys = predictions[
            (predictions['predicted_direction'] == 1) & 
            (predictions['confidence'] >= 0.54) &
            (predictions['price_vs_sma_20'] > 0.01)
        ].nlargest(3, 'confidence')
        
        strong_sells = predictions[
            (predictions['predicted_direction'] == 0) & 
            (predictions['confidence'] >= 0.54) &
            (predictions['price_vs_sma_20'] < -0.01)
        ].nlargest(2, 'confidence')
        
        if len(strong_buys) > 0:
            print("\n🟢 STRONG BUY SIGNALS:")
            for _, row in strong_buys.iterrows():
                print(f"\n  {row['ticker']}: ${row['close']:.2f}")
                print(f"    Confidence: {row['confidence']*100:.1f}%")
                print(f"    Expected Return: {row['predicted_return']*100:+.3f}%")
                print(f"    Trend: {row['price_vs_sma_20']*100:+.1f}% above 20-day MA")
                print(f"    Signal: ✅ BUY - Strong upward momentum")
        
        if len(strong_sells) > 0:
            print("\n🔴 STRONG SELL/SHORT SIGNALS:")
            for _, row in strong_sells.iterrows():
                print(f"\n  {row['ticker']}: ${row['close']:.2f}")
                print(f"    Confidence: {row['confidence']*100:.1f}%")
                print(f"    Expected Return: {row['predicted_return']*100:+.3f}%")
                print(f"    Trend: {row['price_vs_sma_20']*100:.1f}% below 20-day MA")
                print(f"    Signal: 🔻 SHORT - Downward momentum")
        
        if len(strong_buys) == 0 and len(strong_sells) == 0:
            print("\n  No strong signals today - market conditions unclear")
    
    # Step 6: Backtest Results (at the end)
    print_results(results)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    
    return results


def show_prediction(model, df, ticker: str, feature_columns: list):
    """Show prediction for a single stock."""
    ticker_data = df[df['ticker'] == ticker].sort_values('date')
    if len(ticker_data) == 0:
        return
    
    latest = ticker_data.iloc[-1]
    features = {col: latest[col] for col in feature_columns if col in latest.index}
    pred = model.predict_single(features)
    
    regime = "BULL 📈" if latest.get('market_regime', 1) == 1 else "BEAR 📉"
    
    print(f"\n{ticker}: ${latest['close']:.2f}")
    print(f"  Prediction: {pred['predicted_direction']} ({pred['confidence']*100:.1f}% confidence)")
    print(f"  Expected Return: {pred['predicted_return']*100:+.3f}%")
    print(f"  Market Regime: {regime}")
    
    # Trading signal based on regime
    if pred['confidence'] >= 0.58:
        if latest.get('market_regime', 1) == 1 and pred['predicted_direction'] == 'UP':
            print(f"  Signal: ✅ BUY (high confidence + bull market)")
        elif latest.get('market_regime', 1) == 0 and pred['predicted_direction'] == 'DOWN':
            print(f"  Signal: ✅ SHORT (high confidence + bear market)")
        else:
            print(f"  Signal: ⚠️ HOLD (prediction against market trend)")
    else:
        print(f"  Signal: ⏸️ HOLD (confidence too low)")


def main():
    parser = argparse.ArgumentParser(description="S&P 500 Stock Prediction")
    parser.add_argument('--skip-fetch', action='store_true', help='Use cached data')
    parser.add_argument('--no-regime', action='store_true', help='Disable market regime filter')
    args = parser.parse_args()
    
    run_pipeline(
        skip_fetch=args.skip_fetch,
        use_regime_filter=not args.no_regime
    )


if __name__ == "__main__":
    main()

