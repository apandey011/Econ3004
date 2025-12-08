"""
Feature Engineering Module
Creates technical indicators and features for the prediction model.

Features:
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, MACD)
- Volatility measures (ATR, Bollinger Bands)
- Volume indicators (relative volume, OBV)
"""
import pandas as pd
import numpy as np
from typing import List
import ta


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Simple and Exponential Moving Averages.
    
    Features created:
    - SMA 5, 10, 20, 50 day
    - EMA 5, 10, 20, 50 day
    - Price relative to each MA (above/below)
    """
    df = df.copy()
    
    for window in [5, 10, 20, 50]:
        # Simple Moving Average
        df[f'sma_{window}'] = df.groupby('ticker')['close'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # Exponential Moving Average
        df[f'ema_{window}'] = df.groupby('ticker')['close'].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )
        
        # Price relative to MA (percentage above/below)
        df[f'price_vs_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']
        df[f'price_vs_ema_{window}'] = (df['close'] - df[f'ema_{window}']) / df[f'ema_{window}']
    
    # MA crossover signals
    df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
    df['ema_5_20_cross'] = (df['ema_5'] > df['ema_20']).astype(int)
    
    return df


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum-based technical indicators.
    
    Features:
    - RSI (Relative Strength Index) - overbought/oversold
    - MACD (Moving Average Convergence Divergence)
    - Rate of Change (ROC)
    - Stochastic Oscillator
    """
    df = df.copy()
    
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df.loc[mask].copy()
        
        # RSI - 14 period (standard)
        df.loc[mask, 'rsi_14'] = ta.momentum.RSIIndicator(
            ticker_data['close'], window=14
        ).rsi()
        
        # RSI - 7 period (faster)
        df.loc[mask, 'rsi_7'] = ta.momentum.RSIIndicator(
            ticker_data['close'], window=7
        ).rsi()
        
        # MACD
        macd = ta.trend.MACD(ticker_data['close'])
        df.loc[mask, 'macd'] = macd.macd()
        df.loc[mask, 'macd_signal'] = macd.macd_signal()
        df.loc[mask, 'macd_histogram'] = macd.macd_diff()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            ticker_data['high'], ticker_data['low'], ticker_data['close']
        )
        df.loc[mask, 'stoch_k'] = stoch.stoch()
        df.loc[mask, 'stoch_d'] = stoch.stoch_signal()
    
    # Rate of Change (simple momentum)
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = df.groupby('ticker')['close'].transform(
            lambda x: x.pct_change(periods=period)
        )
    
    return df


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility-based features.
    
    Features:
    - ATR (Average True Range)
    - Bollinger Bands (and position within bands)
    - Historical volatility (rolling std of returns)
    """
    df = df.copy()
    
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df.loc[mask].copy()
        
        # ATR - Average True Range
        df.loc[mask, 'atr_14'] = ta.volatility.AverageTrueRange(
            ticker_data['high'], ticker_data['low'], ticker_data['close'], window=14
        ).average_true_range()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(ticker_data['close'], window=20, window_dev=2)
        df.loc[mask, 'bb_upper'] = bb.bollinger_hband()
        df.loc[mask, 'bb_lower'] = bb.bollinger_lband()
        df.loc[mask, 'bb_middle'] = bb.bollinger_mavg()
        df.loc[mask, 'bb_width'] = bb.bollinger_wband()
        df.loc[mask, 'bb_position'] = bb.bollinger_pband()  # % position within bands
    
    # Historical volatility (rolling std of daily returns)
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df.groupby('ticker')['daily_return'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # Daily range as % of close
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    
    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based features.
    
    Features:
    - Relative volume (vs average)
    - Volume moving averages
    - On-Balance Volume (OBV)
    - Volume-price trend
    """
    df = df.copy()
    
    # Volume moving averages
    for window in [5, 10, 20]:
        df[f'volume_sma_{window}'] = df.groupby('ticker')['volume'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    # Relative volume (today's volume vs 20-day average)
    df['relative_volume'] = df['volume'] / df['volume_sma_20']
    
    # Volume change
    df['volume_change'] = df.groupby('ticker')['volume'].pct_change()
    
    # On-Balance Volume (OBV)
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df.loc[mask].copy()
        
        df.loc[mask, 'obv'] = ta.volume.OnBalanceVolumeIndicator(
            ticker_data['close'], ticker_data['volume']
        ).on_balance_volume()
    
    # Volume-weighted price
    df['vwap_proxy'] = (df['high'] + df['low'] + df['close']) / 3
    
    return df


def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price pattern features.
    
    Features:
    - Candlestick patterns (simple versions)
    - Gap up/down
    - Higher highs, lower lows
    """
    df = df.copy()
    
    # Candlestick body size
    df['candle_body'] = (df['close'] - df['open']) / df['open']
    df['candle_body_abs'] = abs(df['candle_body'])
    
    # Upper/lower shadows
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    # Gap (today's open vs yesterday's close)
    df['gap'] = df.groupby('ticker').apply(
        lambda x: (x['open'] - x['close'].shift(1)) / x['close'].shift(1)
    ).reset_index(level=0, drop=True)
    
    # Consecutive up/down days
    df['up_day'] = (df['close'] > df['open']).astype(int)
    df['consecutive_up'] = df.groupby('ticker')['up_day'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    ) * df['up_day']
    df['consecutive_down'] = df.groupby('ticker')['up_day'].transform(
        lambda x: (1 - x).groupby(((1-x) != (1-x).shift()).cumsum()).cumcount() + 1
    ) * (1 - df['up_day'])
    
    return df


def add_lagged_features(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
    """
    Add lagged versions of key features.
    The model can learn from recent history patterns.
    """
    df = df.copy()
    
    # Key features to lag
    features_to_lag = ['daily_return', 'relative_volume', 'rsi_14', 'macd_histogram']
    
    for feature in features_to_lag:
        if feature in df.columns:
            for lag in lags:
                df[f'{feature}_lag_{lag}'] = df.groupby('ticker')[feature].shift(lag)
    
    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to create all features.
    Call this after calculating returns.
    """
    print("Engineering features...")
    
    # Add all feature groups
    df = add_moving_averages(df)
    print("  ✓ Moving averages")
    
    df = add_momentum_indicators(df)
    print("  ✓ Momentum indicators")
    
    df = add_volatility_indicators(df)
    print("  ✓ Volatility indicators")
    
    df = add_volume_indicators(df)
    print("  ✓ Volume indicators")
    
    df = add_price_patterns(df)
    print("  ✓ Price patterns")
    
    df = add_lagged_features(df)
    print("  ✓ Lagged features")
    
    # Drop rows with NaN targets (last row of each stock)
    initial = len(df)
    df = df.dropna(subset=['target_return'])
    
    print(f"\nTotal features created: {len([c for c in df.columns if c not in ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']])}")
    print(f"Rows with valid targets: {len(df)} (dropped {initial - len(df)} without targets)")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns (exclude metadata and targets).
    """
    exclude = [
        'date', 'ticker', 'open', 'high', 'low', 'close', 'volume',
        'dividends', 'stock_splits', 'adj_close',
        'target_return', 'target_direction', 'daily_return'
    ]
    
    return [col for col in df.columns if col not in exclude]


if __name__ == "__main__":
    # Test feature engineering
    from data_fetcher import fetch_stock_data, clean_data, calculate_returns
    
    print("Testing feature engineering...")
    df = fetch_stock_data("AAPL", period="6mo")
    if df is not None:
        df = clean_data(df)
        df = calculate_returns(df)
        df = create_all_features(df)
        
        print(f"\nFinal shape: {df.shape}")
        print(f"\nFeature columns:\n{get_feature_columns(df)}")

