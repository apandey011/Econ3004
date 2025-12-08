"""
Feature Engineering Module
Creates technical indicators and features for stock prediction.

Features include:
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, MACD, Williams %R, CCI, ADX)
- Volatility measures (ATR, Bollinger Bands, historical volatility)
- Volume indicators (relative volume, OBV)
- Market-wide features (breadth, regime detection)
- Cross-sectional features (stock vs market performance)
"""
import pandas as pd
import numpy as np
from typing import List
import ta


# =============================================================================
# BASIC TECHNICAL INDICATORS
# =============================================================================

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add Simple and Exponential Moving Averages."""
    df = df.copy()
    
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = df.groupby('ticker')['close'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'ema_{window}'] = df.groupby('ticker')['close'].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )
        df[f'price_vs_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']
        df[f'price_vs_ema_{window}'] = (df['close'] - df[f'ema_{window}']) / df[f'ema_{window}']
    
    # MA crossover signals
    df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
    df['ema_5_20_cross'] = (df['ema_5'] > df['ema_20']).astype(int)
    
    return df


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum-based technical indicators."""
    df = df.copy()
    
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df.loc[mask].copy()
        
        # RSI
        df.loc[mask, 'rsi_14'] = ta.momentum.RSIIndicator(ticker_data['close'], window=14).rsi()
        df.loc[mask, 'rsi_7'] = ta.momentum.RSIIndicator(ticker_data['close'], window=7).rsi()
        
        # MACD
        macd = ta.trend.MACD(ticker_data['close'])
        df.loc[mask, 'macd'] = macd.macd()
        df.loc[mask, 'macd_signal'] = macd.macd_signal()
        df.loc[mask, 'macd_histogram'] = macd.macd_diff()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(ticker_data['high'], ticker_data['low'], ticker_data['close'])
        df.loc[mask, 'stoch_k'] = stoch.stoch()
        df.loc[mask, 'stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df.loc[mask, 'williams_r'] = ta.momentum.WilliamsRIndicator(
            ticker_data['high'], ticker_data['low'], ticker_data['close'], lbp=14
        ).williams_r()
        
        # CCI
        df.loc[mask, 'cci'] = ta.trend.CCIIndicator(
            ticker_data['high'], ticker_data['low'], ticker_data['close'], window=20
        ).cci()
        
        # ADX - trend strength
        adx = ta.trend.ADXIndicator(ticker_data['high'], ticker_data['low'], ticker_data['close'], window=14)
        df.loc[mask, 'adx'] = adx.adx()
        df.loc[mask, 'adx_pos'] = adx.adx_pos()
        df.loc[mask, 'adx_neg'] = adx.adx_neg()
    
    # Rate of Change
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(periods=period))
    
    return df


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility-based features."""
    df = df.copy()
    
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df.loc[mask].copy()
        
        # ATR
        df.loc[mask, 'atr_14'] = ta.volatility.AverageTrueRange(
            ticker_data['high'], ticker_data['low'], ticker_data['close'], window=14
        ).average_true_range()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(ticker_data['close'], window=20, window_dev=2)
        df.loc[mask, 'bb_upper'] = bb.bollinger_hband()
        df.loc[mask, 'bb_lower'] = bb.bollinger_lband()
        df.loc[mask, 'bb_width'] = bb.bollinger_wband()
        df.loc[mask, 'bb_position'] = bb.bollinger_pband()
    
    # Historical volatility
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df.groupby('ticker')['daily_return'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    
    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features."""
    df = df.copy()
    
    for window in [5, 10, 20]:
        df[f'volume_sma_{window}'] = df.groupby('ticker')['volume'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    df['relative_volume'] = df['volume'] / df['volume_sma_20']
    df['volume_change'] = df.groupby('ticker')['volume'].pct_change()
    
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df.loc[mask].copy()
        df.loc[mask, 'obv'] = ta.volume.OnBalanceVolumeIndicator(
            ticker_data['close'], ticker_data['volume']
        ).on_balance_volume()
    
    return df


def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add price pattern features."""
    df = df.copy()
    
    df['candle_body'] = (df['close'] - df['open']) / df['open']
    df['candle_body_abs'] = abs(df['candle_body'])
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    # Gap
    df['prev_close'] = df.groupby('ticker')['close'].shift(1)
    df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    df = df.drop(columns=['prev_close'])
    
    # Consecutive days
    df['up_day'] = (df['close'] > df['open']).astype(int)
    
    return df


# =============================================================================
# MARKET-WIDE FEATURES (KEY FOR REGIME DETECTION)
# =============================================================================

def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market-wide features for regime detection."""
    df = df.copy()
    
    # Market return (average of all stocks)
    market_return = df.groupby('date')['daily_return'].mean().reset_index()
    market_return.columns = ['date', 'market_return']
    
    # Market volatility
    market_vol = df.groupby('date')['daily_return'].std().reset_index()
    market_vol.columns = ['date', 'market_volatility']
    
    # Market breadth (% of stocks up)
    market_breadth = df.groupby('date').apply(
        lambda x: (x['daily_return'] > 0).mean(), include_groups=False
    ).reset_index()
    market_breadth.columns = ['date', 'market_breadth']
    
    df = df.merge(market_return, on='date', how='left')
    df = df.merge(market_vol, on='date', how='left')
    df = df.merge(market_breadth, on='date', how='left')
    
    # Rolling market features
    market_daily = df.groupby('date')[['market_return']].first().reset_index()
    market_daily['market_return_5d'] = market_daily['market_return'].rolling(5).mean()
    market_daily['market_return_20d'] = market_daily['market_return'].rolling(20).mean()
    market_daily['market_momentum'] = market_daily['market_return_5d'] - market_daily['market_return_20d']
    
    # MARKET REGIME: Bull if 20d return positive, Bear if negative
    market_daily['market_regime'] = (market_daily['market_return_20d'] > 0).astype(int)
    
    df = df.merge(
        market_daily[['date', 'market_return_5d', 'market_return_20d', 'market_momentum', 'market_regime']], 
        on='date', how='left'
    )
    
    return df


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional features (stock vs market)."""
    df = df.copy()
    
    df['return_vs_market'] = df['daily_return'] - df['market_return']
    df['return_rank'] = df.groupby('date')['daily_return'].rank(pct=True)
    df['volume_rank'] = df.groupby('date')['relative_volume'].rank(pct=True)
    
    if 'rsi_14' in df.columns:
        df['rsi_rank'] = df.groupby('date')['rsi_14'].rank(pct=True)
    
    # Momentum rank (is this stock one of the stronger or weaker performers?)
    if 'roc_20' in df.columns:
        df['momentum_rank'] = df.groupby('date')['roc_20'].rank(pct=True)
    
    return df


# =============================================================================
# LAGGED FEATURES
# =============================================================================

def add_lagged_features(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
    """Add lagged versions of key features."""
    df = df.copy()
    
    features_to_lag = ['daily_return', 'relative_volume', 'rsi_14', 'macd_histogram']
    
    for feature in features_to_lag:
        if feature in df.columns:
            for lag in lags:
                df[f'{feature}_lag_{lag}'] = df.groupby('ticker')[feature].shift(lag)
    
    return df


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Z-score normalize features within rolling windows."""
    df = df.copy()
    
    skip_patterns = ['direction', 'regime', 'signal', 'rank', 'cross', 'ticker', 'date']
    
    for col in feature_columns:
        if any(pattern in col.lower() for pattern in skip_patterns):
            continue
        if col not in df.columns:
            continue
            
        df[f'{col}_zscore'] = df.groupby('ticker')[col].transform(
            lambda x: (x - x.rolling(60, min_periods=20).mean()) / (x.rolling(60, min_periods=20).std() + 1e-8)
        )
    
    return df


# =============================================================================
# MASTER FUNCTION
# =============================================================================

def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all features for the prediction model."""
    print("Engineering features...")
    
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
    
    df = add_market_features(df)
    print("  ✓ Market-wide features & regime detection")
    
    df = add_relative_features(df)
    print("  ✓ Cross-sectional features")
    
    df = add_lagged_features(df)
    print("  ✓ Lagged features")
    
    # Drop rows without targets
    initial = len(df)
    df = df.dropna(subset=['target_return'])
    
    feature_cols = get_feature_columns(df)
    df = normalize_features(df, feature_cols)
    print("  ✓ Feature normalization")
    
    print(f"\nTotal features: {len(get_feature_columns(df))}")
    print(f"Rows: {len(df)} (dropped {initial - len(df)} without targets)")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns."""
    exclude = [
        'date', 'ticker', 'open', 'high', 'low', 'close', 'volume',
        'dividends', 'stock_splits', 'adj_close',
        'target_return', 'target_direction', 'daily_return'
    ]
    return [col for col in df.columns if col not in exclude and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
