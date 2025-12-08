"""
Data Fetcher Module
Pulls historical S&P 500 stock data using Yahoo Finance API (free, no key required).
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from tqdm import tqdm
import time


# Top S&P 500 stocks by market cap (representative sample)
SP500_SAMPLE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV', 'MRK', 'PEP',
    'KO', 'COST', 'AVGO', 'LLY', 'WMT', 'MCD', 'CSCO', 'ACN', 'ABT', 'TMO',
    'DHR', 'NEE', 'NKE', 'DIS', 'VZ', 'ADBE', 'TXN', 'PM', 'CRM', 'RTX',
    'CMCSA', 'WFC', 'BMY', 'COP', 'ORCL', 'INTC', 'AMD', 'QCOM', 'HON', 'UPS'
]


def get_sp500_tickers() -> List[str]:
    """
    Get list of S&P 500 ticker symbols.
    Returns a curated sample of top S&P 500 stocks.
    """
    return SP500_SAMPLE.copy()


def fetch_stock_data(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    period: str = "2y"
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data for a single stock.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        period: Alternative to dates - '1y', '2y', '5y', 'max'
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
    """
    try:
        stock = yf.Ticker(ticker)
        
        if start_date and end_date:
            df = stock.history(start=start_date, end=end_date)
        else:
            df = stock.history(period=period)
        
        if df.empty:
            print(f"Warning: No data returned for {ticker}")
            return None
        
        # Clean column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Add ticker column
        df['ticker'] = ticker
        
        # Reset index to make date a column
        df = df.reset_index()
        df.rename(columns={'Date': 'date', 'index': 'date'}, inplace=True)
        
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        
        return df
        
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def fetch_multiple_stocks(
    tickers: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    period: str = "2y",
    delay: float = 0.1
) -> pd.DataFrame:
    """
    Fetch historical data for multiple stocks.
    
    Args:
        tickers: List of stock symbols. If None, uses SP500 sample.
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        period: Alternative to dates - '1y', '2y', '5y', 'max'
        delay: Delay between requests (be nice to the API)
    
    Returns:
        Combined DataFrame with all stock data
    """
    if tickers is None:
        tickers = get_sp500_tickers()
    
    all_data = []
    
    print(f"Fetching data for {len(tickers)} stocks...")
    for ticker in tqdm(tickers, desc="Downloading"):
        df = fetch_stock_data(ticker, start_date, end_date, period)
        if df is not None:
            all_data.append(df)
        time.sleep(delay)  # Rate limiting
    
    if not all_data:
        raise ValueError("No data was fetched successfully")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Fetched {len(combined)} total records for {len(all_data)} stocks")
    
    return combined


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate stock data.
    
    - Remove rows with missing OHLCV values
    - Remove obvious outliers (price <= 0)
    - Sort by ticker and date
    """
    print("Cleaning data...")
    initial_rows = len(df)
    
    # Required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    
    # Drop rows with missing required values
    df = df.dropna(subset=required)
    
    # Remove invalid prices
    for col in ['open', 'high', 'low', 'close']:
        df = df[df[col] > 0]
    
    # Remove zero/negative volume
    df = df[df['volume'] > 0]
    
    # Sort by ticker and date
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    removed = initial_rows - len(df)
    if removed > 0:
        print(f"Removed {removed} invalid rows ({removed/initial_rows*100:.2f}%)")
    
    print(f"Clean data: {len(df)} rows, {df['ticker'].nunique()} stocks")
    
    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns and target variable (next day return).
    
    This is what we're predicting: will the stock go up or down tomorrow?
    """
    df = df.copy()
    
    # Daily return (today's close vs yesterday's close)
    df['daily_return'] = df.groupby('ticker')['close'].pct_change()
    
    # Target: Next day's return (what we want to predict)
    df['target_return'] = df.groupby('ticker')['daily_return'].shift(-1)
    
    # Binary target: 1 if stock goes up, 0 if down
    df['target_direction'] = (df['target_return'] > 0).astype(int)
    
    return df


def save_data(df: pd.DataFrame, filepath: str = "data/stock_data.csv"):
    """Save processed data to CSV."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def load_data(filepath: str = "data/stock_data.csv") -> pd.DataFrame:
    """Load processed data from CSV."""
    df = pd.read_csv(filepath, parse_dates=['date'])
    print(f"Loaded {len(df)} rows from {filepath}")
    return df


if __name__ == "__main__":
    # Quick test
    print("Testing data fetcher...")
    df = fetch_stock_data("AAPL", period="1mo")
    if df is not None:
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

