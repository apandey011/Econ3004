# S&P 500 Stock Price Prediction Model

A machine learning system for predicting short-term stock price movements.

## 🎯 Key Features

- **Ensemble Model**: XGBoost + Random Forest + Linear models
- **Market Regime Detection**: Only trades WITH the trend (long in bull markets, short in bear)
- **70+ Technical Indicators**: RSI, MACD, Bollinger Bands, ADX, etc.
- **Confidence Scoring**: Only takes high-conviction trades

## 🚀 Quick Start

```bash
# Install dependencies
cd /Users/apandey/Downloads/Econ3004
source venv/bin/activate

# Run full pipeline (fetch data, train, backtest)
python main.py

# Use cached data (faster)
python main.py --skip-fetch

# Predict any stock
python predict.py AAPL
python predict.py AAPL MSFT GOOGL NVDA
```

## 📁 Project Structure

```
Econ3004/
├── main.py              # Main pipeline
├── predict.py           # Quick predictions for any stock
├── requirements.txt     # Dependencies
├── README.md
├── src/
│   ├── data_fetcher.py  # Download stock data (Yahoo Finance)
│   ├── features.py      # Technical indicators & features
│   ├── model.py         # Ensemble prediction model
│   └── backtester.py    # Backtesting with regime filter
├── data/                # Cached stock data
└── models/              # Trained model files
```

## 📊 How It Works

### 1. Data
- Fetches 2 years of daily OHLCV data for 50 S&P 500 stocks
- Source: Yahoo Finance (free, no API key needed)

### 2. Features (70+)
- **Moving Averages**: SMA, EMA (5, 10, 20, 50 day)
- **Momentum**: RSI, MACD, Stochastic, Williams %R, CCI, ADX
- **Volatility**: ATR, Bollinger Bands, Historical Vol
- **Volume**: Relative Volume, OBV
- **Market-Wide**: Market breadth, regime detection

### 3. Model
- **Ensemble**: Combines 3 model types for robust predictions
- **Regularization**: Prevents overfitting to training data
- **Feature Selection**: Keeps only the most predictive features

### 4. Trading Strategy
- **Regime Filter**: Only long in bull markets, short in bear markets
- **Confidence Threshold**: Only trades above 55% confidence
- **Position Sizing**: Limits risk per trade

## 🔧 Command Options

```bash
# Full pipeline with regime filter (recommended)
python main.py --skip-fetch

# Disable regime filter (trade both directions always)
python main.py --skip-fetch --no-regime
```

## 📈 Performance Metrics

| Metric | Description |
|--------|-------------|
| **Return** | Total portfolio return |
| **Alpha** | Return vs buy-and-hold baseline |
| **Sharpe Ratio** | Risk-adjusted return (>1 is good) |
| **Sortino Ratio** | Downside risk-adjusted return |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | % of profitable trades |
| **Direction Accuracy** | % of correct up/down predictions |

## ⚠️ Important Notes

1. **Not Financial Advice**: This is an educational project
2. **Past ≠ Future**: Historical performance doesn't guarantee results
3. **Market Efficiency**: Stock prices are hard to predict (~51% is good!)
4. **Risk Management**: Never invest more than you can afford to lose

## 🎓 For Econ 3004

This project demonstrates:
- Machine learning for financial prediction
- Time series analysis
- Backtesting methodology
- Risk management principles
