"""
Backtesting Engine
Evaluates the model's performance on historical data.

Metrics:
- Total return vs baseline (buy & hold)
- Sharpe ratio
- Max drawdown
- Win rate
- Average trade returns
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """Represents a single trade."""
    ticker: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    direction: str  # 'long' or 'short'
    predicted_return: float
    confidence: float
    actual_return: float
    
    @property
    def pnl_pct(self) -> float:
        """Profit/loss percentage."""
        if self.direction == 'long':
            return (self.exit_price - self.entry_price) / self.entry_price
        else:  # short
            return (self.entry_price - self.exit_price) / self.entry_price


class Backtester:
    """
    Backtesting engine for stock predictions.
    
    Strategy:
    - If model predicts UP with high confidence: go LONG
    - If model predicts DOWN with high confidence: go SHORT
    - If confidence is low: stay OUT
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.55,
        transaction_cost: float = 0.001,  # 0.1% per trade
        max_positions: int = 10,
        position_size: float = 0.1  # 10% of portfolio per position
    ):
        """
        Initialize backtester.
        
        Args:
            confidence_threshold: Minimum confidence to take a trade
            transaction_cost: Cost per trade as decimal
            max_positions: Maximum simultaneous positions
            position_size: Fraction of portfolio per position
        """
        self.confidence_threshold = confidence_threshold
        self.transaction_cost = transaction_cost
        self.max_positions = max_positions
        self.position_size = position_size
        
        self.trades: List[Trade] = []
        self.daily_returns: List[float] = []
        self.portfolio_values: List[float] = []
    
    def run(
        self,
        predictions: pd.DataFrame,
        actual_data: pd.DataFrame,
        initial_capital: float = 100000
    ) -> Dict:
        """
        Run backtest on predictions.
        
        Args:
            predictions: DataFrame with model predictions
                Required columns: date, ticker, predicted_return, 
                predicted_direction, confidence
            actual_data: DataFrame with actual prices and returns
                Required columns: date, ticker, close, target_return
            initial_capital: Starting portfolio value
            
        Returns:
            Dictionary with backtest results
        """
        self.trades = []
        self.daily_returns = []
        self.portfolio_values = [initial_capital]
        
        # Merge predictions with actual data
        df = predictions.merge(
            actual_data[['date', 'ticker', 'close', 'target_return']],
            on=['date', 'ticker'],
            how='inner',
            suffixes=('', '_actual')
        )
        
        if 'close_actual' in df.columns:
            df['close'] = df['close_actual']
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Get unique dates
        dates = df['date'].unique()
        
        portfolio_value = initial_capital
        
        for i, date in enumerate(dates[:-1]):  # Exclude last date (no next day)
            day_data = df[df['date'] == date]
            next_date = dates[i + 1]
            
            # Select trades based on confidence
            high_conf = day_data[day_data['confidence'] >= self.confidence_threshold]
            
            if len(high_conf) == 0:
                self.daily_returns.append(0)
                self.portfolio_values.append(portfolio_value)
                continue
            
            # Sort by confidence, take top positions
            high_conf = high_conf.nlargest(self.max_positions, 'confidence')
            
            day_return = 0
            n_trades = len(high_conf)
            trade_size = self.position_size * portfolio_value
            
            for _, row in high_conf.iterrows():
                direction = 'long' if row['predicted_direction'] == 1 else 'short'
                actual_return = row['target_return'] if not pd.isna(row['target_return']) else 0
                
                # Adjust for direction
                if direction == 'short':
                    trade_return = -actual_return  # Short profits when stock falls
                else:
                    trade_return = actual_return
                
                # Apply transaction cost
                trade_return -= self.transaction_cost
                
                day_return += trade_return / n_trades
                
                # Record trade
                self.trades.append(Trade(
                    ticker=row['ticker'],
                    entry_date=date,
                    exit_date=next_date,
                    entry_price=row['close'],
                    exit_price=row['close'] * (1 + actual_return),
                    direction=direction,
                    predicted_return=row['predicted_return'],
                    confidence=row['confidence'],
                    actual_return=actual_return
                ))
            
            # Update portfolio
            portfolio_value *= (1 + day_return * self.position_size * n_trades)
            self.daily_returns.append(day_return)
            self.portfolio_values.append(portfolio_value)
        
        return self._calculate_metrics(initial_capital, actual_data)
    
    def _calculate_metrics(
        self,
        initial_capital: float,
        actual_data: pd.DataFrame
    ) -> Dict:
        """Calculate performance metrics."""
        
        if not self.trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'error': 'No trades executed'
            }
        
        # Returns
        total_return = (self.portfolio_values[-1] - initial_capital) / initial_capital
        daily_returns = np.array(self.daily_returns)
        
        # Sharpe Ratio (annualized, assuming 252 trading days)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe_ratio = 0
        
        # Max Drawdown
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Win Rate
        winning_trades = sum(1 for t in self.trades if t.pnl_pct > 0)
        win_rate = winning_trades / len(self.trades) if self.trades else 0
        
        # Trade Statistics
        trade_returns = [t.pnl_pct for t in self.trades]
        avg_trade_return = np.mean(trade_returns)
        
        # Direction Accuracy
        correct_direction = sum(
            1 for t in self.trades 
            if (t.direction == 'long' and t.actual_return > 0) or
               (t.direction == 'short' and t.actual_return < 0)
        )
        direction_accuracy = correct_direction / len(self.trades) if self.trades else 0
        
        # Baseline: Buy and hold S&P (using average of all stocks)
        baseline_return = actual_data.groupby('date')['target_return'].mean().sum()
        
        # Long-only trades performance
        long_trades = [t for t in self.trades if t.direction == 'long']
        short_trades = [t for t in self.trades if t.direction == 'short']
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annual_return': total_return * (252 / max(len(self.daily_returns), 1)),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'direction_accuracy': direction_accuracy,
            'direction_accuracy_pct': direction_accuracy * 100,
            'total_trades': len(self.trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'avg_trade_return': avg_trade_return,
            'avg_trade_return_pct': avg_trade_return * 100,
            'baseline_return': baseline_return,
            'baseline_return_pct': baseline_return * 100,
            'alpha': total_return - baseline_return,
            'alpha_pct': (total_return - baseline_return) * 100,
            'final_portfolio_value': self.portfolio_values[-1],
            'prediction_mae': np.mean([
                abs(t.predicted_return - t.actual_return) for t in self.trades
            ])
        }
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get detailed trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            'ticker': t.ticker,
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'direction': t.direction,
            'predicted_return': t.predicted_return,
            'actual_return': t.actual_return,
            'confidence': t.confidence,
            'pnl_pct': t.pnl_pct
        } for t in self.trades])
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get portfolio value over time."""
        return pd.DataFrame({
            'portfolio_value': self.portfolio_values
        })


def run_full_backtest(
    model,
    df: pd.DataFrame,
    feature_columns: list,
    train_ratio: float = 0.8,
    confidence_threshold: float = 0.55
) -> Dict:
    """
    Run a full train-test backtest.
    
    Args:
        model: Trained StockPredictor model
        df: Full dataset with features
        feature_columns: List of feature columns
        train_ratio: Fraction of data for training
        confidence_threshold: Min confidence for trades
        
    Returns:
        Backtest results dictionary
    """
    # Sort by date
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    # Split data
    split_idx = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    print(f"Training on {len(train_df)} samples, testing on {len(test_df)} samples")
    print(f"Train period: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    
    # Train model
    model.fit(train_df, feature_columns)
    
    # Get predictions on test set
    predictions = model.predict(test_df)
    
    # Run backtest
    backtester = Backtester(confidence_threshold=confidence_threshold)
    results = backtester.run(predictions, test_df)
    
    return {
        'metrics': results,
        'trade_history': backtester.get_trade_history(),
        'equity_curve': backtester.get_equity_curve(),
        'predictions': predictions
    }


def print_backtest_results(results: Dict):
    """Pretty print backtest results."""
    metrics = results['metrics']
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    print(f"\n📈 RETURNS")
    print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"  Baseline (Buy & Hold): {metrics['baseline_return_pct']:.2f}%")
    print(f"  Alpha (Excess Return): {metrics['alpha_pct']:.2f}%")
    
    print(f"\n📊 RISK METRICS")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    
    print(f"\n🎯 ACCURACY")
    print(f"  Direction Accuracy: {metrics['direction_accuracy_pct']:.1f}%")
    print(f"  Win Rate: {metrics['win_rate_pct']:.1f}%")
    print(f"  Prediction MAE: {metrics['prediction_mae']:.6f}")
    
    print(f"\n📝 TRADE STATISTICS")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Long Trades: {metrics['long_trades']}")
    print(f"  Short Trades: {metrics['short_trades']}")
    print(f"  Avg Trade Return: {metrics['avg_trade_return_pct']:.3f}%")
    
    print(f"\n💰 FINAL PORTFOLIO")
    print(f"  Value: ${metrics['final_portfolio_value']:,.2f}")
    print("="*60)


if __name__ == "__main__":
    print("Backtester module loaded successfully")

