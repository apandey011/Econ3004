#!/usr/bin/env python3
"""
Quick Stock Prediction Script
Get predictions for any stock.

Usage:
    python predict.py AAPL
    python predict.py AAPL MSFT GOOGL NVDA
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import fetch_stock_data, clean_data, calculate_returns
from features import create_all_features, get_feature_columns
from model import StockPredictor
import pandas as pd


def predict_stock(ticker: str):
    """Get prediction for a single stock."""
    ticker = ticker.upper()
    
    print(f"\n{'='*60}")
    print(f"Fetching data for {ticker}...")
    
    df = fetch_stock_data(ticker, period="6mo")
    if df is None or len(df) == 0:
        print(f"❌ Could not fetch data for {ticker}")
        return
    
    df = clean_data(df)
    df = calculate_returns(df)
    df = create_all_features(df)
    
    model_path = "models/model.joblib"
    if not os.path.exists(model_path):
        print("❌ Model not found. Run 'python main.py' first.")
        return
    
    model = StockPredictor.load(model_path)
    latest = df.iloc[-1]
    
    features = {col: latest[col] for col in model.feature_columns if col in latest.index}
    pred = model.predict_single(features)
    
    print(f"\n{'='*60}")
    print(f"📊 {ticker} PREDICTION")
    print('='*60)
    print(f"Date: {latest['date'].strftime('%Y-%m-%d')}")
    print(f"Price: ${latest['close']:.2f}")
    
    print(f"\n🎯 PREDICTION: {pred['predicted_direction']}")
    print(f"   Confidence: {pred['confidence']*100:.1f}%")
    print(f"   Expected Return: {pred['predicted_return']*100:+.3f}%")
    
    # Market regime
    regime = "BULL 📈" if latest.get('market_regime', 1) == 1 else "BEAR 📉"
    print(f"\n🌍 Market Regime: {regime}")
    
    # Trading signal
    conf = pred['confidence']
    direction = pred['predicted_direction']
    market_bull = latest.get('market_regime', 1) == 1
    
    print(f"\n📈 SIGNAL:")
    if conf >= 0.60:
        if (market_bull and direction == 'UP') or (not market_bull and direction == 'DOWN'):
            signal = "STRONG " + ("BUY" if direction == 'UP' else "SHORT")
            print(f"   ✅ {signal} - High confidence, aligned with market")
        else:
            print(f"   ⚠️ CAUTION - Prediction against market trend")
    elif conf >= 0.55:
        if (market_bull and direction == 'UP') or (not market_bull and direction == 'DOWN'):
            print(f"   📊 MODERATE {direction} - Consider small position")
        else:
            print(f"   ⏸️ HOLD - Against market trend")
    else:
        print(f"   ⏸️ HOLD - Confidence too low")
    
    # Key indicators
    print(f"\n📉 KEY INDICATORS:")
    if 'rsi_14' in latest.index and pd.notna(latest['rsi_14']):
        rsi = latest['rsi_14']
        status = "⚠️ OVERBOUGHT" if rsi > 70 else "⚠️ OVERSOLD" if rsi < 30 else "Neutral"
        print(f"   RSI(14): {rsi:.1f} ({status})")
    
    if 'price_vs_sma_20' in latest.index and pd.notna(latest['price_vs_sma_20']):
        print(f"   vs 20-day MA: {latest['price_vs_sma_20']*100:+.2f}%")
    
    if 'macd_histogram' in latest.index and pd.notna(latest['macd_histogram']):
        macd_signal = "Bullish" if latest['macd_histogram'] > 0 else "Bearish"
        print(f"   MACD: {macd_signal}")
    
    if 'adx' in latest.index and pd.notna(latest['adx']):
        trend = "Strong Trend" if latest['adx'] > 25 else "Weak Trend"
        print(f"   ADX: {latest['adx']:.1f} ({trend})")
    
    print(f"\n⚠️ DISCLAIMER: Not financial advice. Do your own research.")
    print('='*60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py TICKER [TICKER2 ...]")
        print("Example: python predict.py AAPL MSFT NVDA")
        sys.exit(1)
    
    for ticker in sys.argv[1:]:
        predict_stock(ticker)


if __name__ == "__main__":
    main()
