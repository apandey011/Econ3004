# S&P 500 Stock Price Prediction Model

A machine learning system for predicting short-term stock price movements for S&P 500 stocks.

## 📋 Project Overview

This model predicts next-day price movements (up/down) and expected returns for stocks in the S&P 500. It uses:

- **Historical price data**: Open, High, Low, Close, Volume
- **Technical indicators**: Moving averages, RSI, MACD, Bollinger Bands, etc.
- **Volume analysis**: Relative volume, OBV
- **Volatility measures**: ATR, historical volatility

### Key Outputs

1. **Prediction**: Will the stock go UP or DOWN tomorrow?
2. **Confidence Score**: How sure is the model? (0-100%)
3. **Expected Return**: Predicted percentage change
4. **Trading Signal**: STRONG BUY, MODERATE BUY, HOLD, MODERATE SELL, STRONG SELL
5. **Explanation**: Key factors driving the prediction

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/apandey/Downloads/Econ3004
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python main.py
```

This will:
1. Fetch 2 years of data for 50 S&P 500 stocks
2. Engineer 70+ technical features
3. Train an XGBoost model
4. Run backtesting with performance metrics
5. Generate sample predictions with explanations

### 3. Get a Prediction for a Specific Stock

```bash
python main.py --predict AAPL
```

## 📁 Project Structure

```
Econ3004/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   ├── data_fetcher.py    # Data downloading & cleaning
│   ├── features.py        # Feature engineering
│   ├── models.py          # ML models (XGBoost, Ridge, Random Forest)
│   ├── lstm_model.py      # LSTM neural network
│   ├── backtester.py      # Backtesting engine
│   └── visualize.py       # Charts and plots
├── data/                  # Downloaded stock data
├── models/                # Saved trained models
└── plots/                 # Generated visualizations
```

## 📊 Features Created

### Moving Averages
- SMA (5, 10, 20, 50 day)
- EMA (5, 10, 20, 50 day)
- Price position relative to each MA

### Momentum Indicators
- RSI (7 and 14 period)
- MACD (line, signal, histogram)
- Stochastic Oscillator
- Rate of Change (5, 10, 20 day)

### Volatility Indicators
- ATR (Average True Range)
- Bollinger Bands (upper, lower, width, position)
- Historical volatility (5, 10, 20 day)
- Daily range

### Volume Indicators
- Relative volume
- Volume moving averages
- On-Balance Volume (OBV)
- Volume change

### Price Patterns
- Candlestick body size
- Upper/lower shadows
- Gap up/down
- Consecutive up/down days

### Lagged Features
- Lagged returns (1-5 days)
- Lagged RSI
- Lagged MACD
- Lagged volume

## 🎯 Models

### 1. XGBoost (Default)
- Gradient boosted trees
- Best for tabular/structured data
- Fast training and inference

### 2. Random Forest
- Ensemble of decision trees
- Good for understanding feature importance
- Robust to overfitting

### 3. Ridge Regression
- Linear model with L2 regularization
- Simple baseline
- Interpretable coefficients

### 4. LSTM (Advanced)
- Recurrent neural network
- Learns sequential patterns
- Captures temporal dependencies

## 📈 Success Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Direction Accuracy** | % of correct up/down predictions | > 52% |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 |
| **Alpha** | Excess return vs buy & hold | > 0% |
| **Max Drawdown** | Largest peak-to-trough decline | < 20% |
| **MAE** | Average prediction error | < 2% |

## 🔧 Usage Options

```bash
# Fetch data only (no training)
python main.py --fetch-only

# Use existing data (skip download)
python main.py --skip-fetch

# Specific tickers
python main.py --tickers AAPL MSFT GOOGL AMZN

# Different time period
python main.py --period 5y

# Different model
python main.py --model random_forest
```

## ⚠️ Important Notes

1. **This is not financial advice** - This is an educational project for Econ 3004
2. **Past performance ≠ future results** - Markets are unpredictable
3. **Paper trade first** - Test with fake money before risking real capital
4. **Manage risk** - Never invest more than you can afford to lose

## 🔄 Continuous Improvement

The model can be improved by:

1. **More data sources**: News sentiment, social media, earnings data
2. **Alternative data**: Satellite imagery, web traffic, credit card data
3. **Ensemble methods**: Combining multiple models
4. **Hyperparameter tuning**: Optimize model parameters
5. **Walk-forward validation**: More rigorous backtesting
6. **Live paper trading**: Test in real-time market conditions

## 📝 Example Output

```
============================================================
PREDICTION FOR AAPL
============================================================
Date: 2024-12-06
Current Price: $243.12

🎯 PREDICTION: UP
   Expected Return: 0.234%
   Confidence: 62.4%
   Probability Up: 62.4%
   Probability Down: 37.6%

📊 TRADING SIGNAL:
   Signal: MODERATE BUY
   Action: Consider a small LONG position

📈 KEY FACTORS:
   • RSI (54.2): Neutral momentum
   • Price vs 20-day SMA: +1.23%
   • 20-day Volatility: 1.42%
   • Volume: 1.12x average (normal)
   • MACD: Bullish (histogram = 0.234)
```

## 👨‍💻 Author

Econ 3004 Final Project

---

*Remember: The best prediction is one that helps you make better decisions, not perfect decisions.*

