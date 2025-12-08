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


# Full S&P 500 stocks list
SP500_FULL = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'NVDA', 'META', 'AVGO', 'ADBE', 'CRM', 'CSCO',
    'ACN', 'ORCL', 'AMD', 'INTC', 'IBM', 'QCOM', 'TXN', 'NOW', 'INTU', 'AMAT',
    'ADI', 'LRCX', 'MU', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'APH', 'MSI', 'TEL',
    'FTNT', 'PANW', 'CRWD', 'IT', 'ANSS', 'KEYS', 'CDW', 'FSLR', 'ENPH', 'ON',
    'MPWR', 'TER', 'SWKS', 'AKAM', 'JNPR', 'NTAP', 'WDC', 'STX', 'HPQ', 'HPE',
    # Financials
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SPGI', 'C',
    'AXP', 'SCHW', 'CB', 'MMC', 'PGR', 'AON', 'CME', 'ICE', 'MCO', 'USB',
    'PNC', 'TFC', 'AIG', 'MET', 'PRU', 'AFL', 'TRV', 'ALL', 'COF', 'BK',
    'FITB', 'STT', 'HBAN', 'RF', 'CFG', 'KEY', 'NTRS', 'CINF', 'L', 'RE',
    'GL', 'AIZ', 'CBOE', 'NDAQ', 'MSCI', 'FDS', 'MKTX', 'IVZ', 'BEN', 'TROW',
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'MDT', 'GILD', 'CVS', 'ELV', 'CI', 'ISRG', 'VRTX', 'REGN', 'SYK',
    'BSX', 'ZBH', 'BDX', 'HUM', 'CNC', 'MCK', 'CAH', 'DXCM', 'IQV', 'MTD',
    'A', 'IDXX', 'EW', 'RMD', 'HOLX', 'ALGN', 'BAX', 'BIIB', 'MRNA', 'MOH',
    'LH', 'DGX', 'CRL', 'TECH', 'HSIC', 'XRAY', 'DVA', 'INCY', 'VTRS', 'OGN',
    # Consumer
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'CMG',
    'ORLY', 'AZO', 'ROST', 'MAR', 'HLT', 'DHI', 'LEN', 'GM', 'F', 'APTV',
    'GRMN', 'BBY', 'DRI', 'YUM', 'EBAY', 'ETSY', 'ULTA', 'POOL', 'DPZ', 'MGM',
    'WYNN', 'CZR', 'LVS', 'RCL', 'CCL', 'NCLH', 'HAS', 'NWL', 'WHR', 'MHK',
    'KMX', 'AN', 'GPC', 'AAP', 'BWA', 'LEA', 'LKQ', 'GNRC', 'PHM', 'NVR',
    # Consumer Staples
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
    'GIS', 'K', 'HSY', 'SJM', 'CAG', 'CPB', 'HRL', 'MKC', 'TSN', 'KHC',
    'STZ', 'BF-B', 'TAP', 'ADM', 'BG', 'KR', 'SYY', 'WBA', 'EL', 'CLX',
    'CHD', 'MNST', 'KDP', 'CTRA', 'KVUE',
    # Energy
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'PXD', 'OXY',
    'HES', 'DVN', 'HAL', 'BKR', 'FANG', 'KMI', 'WMB', 'OKE', 'TRGP', 'EQT',
    'APA', 'MRO', 'CTRA',
    # Industrials
    'UPS', 'HON', 'RTX', 'CAT', 'GE', 'BA', 'DE', 'LMT', 'UNP', 'MMM',
    'NOC', 'GD', 'CSX', 'NSC', 'WM', 'ITW', 'EMR', 'ETN', 'PH', 'ROK',
    'FDX', 'JCI', 'CMI', 'PCAR', 'CARR', 'OTIS', 'TT', 'AME', 'FAST', 'VRSK',
    'IR', 'DOV', 'CTAS', 'PAYX', 'CPRT', 'RSG', 'GWW', 'ODFL', 'J', 'PWR',
    'HII', 'LHX', 'TDG', 'WAB', 'SWK', 'XYL', 'IEX', 'ROP', 'NDSN', 'PNR',
    # Materials
    'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'VMC', 'MLM',
    'DOW', 'PPG', 'IP', 'PKG', 'AVY', 'CE', 'ALB', 'EMN', 'FMC', 'CF',
    'MOS', 'IFF', 'BALL', 'SEE', 'WRK', 'AMCR',
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'ED',
    'WEC', 'ES', 'AWK', 'DTE', 'EIX', 'FE', 'AEE', 'CMS', 'CNP', 'EVRG',
    'ATO', 'NI', 'LNT', 'PPL', 'NRG', 'AES', 'CEG', 'VST', 'PNW', 'MGEE',
    # Real Estate
    'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'O', 'WELL', 'SPG', 'DLR', 'VICI',
    'AVB', 'EQR', 'VTR', 'ARE', 'MAA', 'UDR', 'ESS', 'EXR', 'INVH', 'KIM',
    'REG', 'HST', 'PEAK', 'CPT', 'BXP', 'SLG', 'VNO', 'AIV',
    # Communication
    'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS', 'CHTR', 'WBD', 'PARA', 'FOX',
    'FOXA', 'NWS', 'NWSA', 'LYV', 'MTCH', 'IPG', 'OMC',
]

# Top 50 stocks sample (default - faster)
SP500_TOP50 = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV', 'MRK', 'PEP',
    'KO', 'COST', 'AVGO', 'LLY', 'WMT', 'MCD', 'CSCO', 'ACN', 'ABT', 'TMO',
    'DHR', 'NEE', 'NKE', 'DIS', 'VZ', 'ADBE', 'TXN', 'PM', 'CRM', 'RTX',
    'CMCSA', 'WFC', 'BMY', 'COP', 'ORCL', 'INTC', 'AMD', 'QCOM', 'HON', 'UPS'
]

# Default to top 50 (use --500 flag for full list)
SP500_SAMPLE = SP500_TOP50


def get_sp500_tickers(full: bool = False) -> List[str]:
    """
    Get S&P 500 ticker symbols.
    
    Args:
        full: If True, return all ~400 stocks. If False, return top 50.
    """
    if full:
        return SP500_FULL.copy()
    return SP500_TOP50.copy()




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
    delay: float = 0.1,
    full: bool = False
) -> pd.DataFrame:
    """
    Fetch historical data for multiple stocks.
    
    Args:
        tickers: List of stock symbols. If None, uses SP500 sample.
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        period: Alternative to dates - '1y', '2y', '5y', 'max'
        delay: Delay between requests (be nice to the API)
        full: If True, fetch all ~400 S&P 500 stocks (slower)
    
    Returns:
        Combined DataFrame with all stock data
    """
    if tickers is None:
        tickers = get_sp500_tickers(full=full)
    
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

