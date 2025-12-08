"""
Backtesting Engine
Evaluates model performance with market regime-aware trading.

Key feature: Only trades in the direction of the market regime
- Bull market (market_regime=1): Only LONG trades
- Bear market (market_regime=0): Only SHORT trades

This prevents fighting the trend, which is the #1 killer of short-term strategies.
"""
import numpy as np
import pandas as pd
from typing import Dict, List
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
    direction: str
    predicted_return: float
    confidence: float
    actual_return: float
    position_size: float
    market_regime: int
    
    @property
    def pnl_pct(self) -> float:
        if self.direction == 'long':
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price


class Backtester:
    """
    Backtester with market regime-aware trading.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.54,  # Slightly higher for quality
        high_confidence_threshold: float = 0.58,
        transaction_cost: float = 0.0003,  # Very low costs
        max_positions: int = 8,
        position_size: float = 0.25,  # Larger positions on good trades
        use_regime_filter: bool = True,
        regime_override: str = None,
        use_momentum_filter: bool = True,
        use_strong_momentum: bool = True  # Extra filter for strong trends
    ):
        self.confidence_threshold = confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.transaction_cost = transaction_cost
        self.max_positions = max_positions
        self.position_size = position_size
        self.use_momentum_filter = use_momentum_filter
        self.use_strong_momentum = use_strong_momentum
        self.use_regime_filter = use_regime_filter
        self.regime_override = regime_override
        
        self.trades: List[Trade] = []
        self.daily_returns: List[float] = []
        self.portfolio_values: List[float] = []
    
    def _filter_by_regime(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Filter trades based on market regime and momentum."""
        if self.regime_override == 'long_only':
            return candidates[candidates['predicted_direction'] == 1]
        elif self.regime_override == 'short_only':
            return candidates[candidates['predicted_direction'] == 0]
        
        filtered = candidates.copy()
        
        # Momentum filter: only trade stocks moving in predicted direction
        if self.use_momentum_filter and 'price_vs_sma_20' in filtered.columns:
            # For longs: stock should be above its 20-day MA
            # For shorts: stock should be below its 20-day MA
            long_momentum = (filtered['predicted_direction'] == 1) & (filtered['price_vs_sma_20'] > 0)
            short_momentum = (filtered['predicted_direction'] == 0) & (filtered['price_vs_sma_20'] < 0)
            filtered = filtered[long_momentum | short_momentum]
        
        # Strong momentum filter: require significant trend (>1% from MA)
        if self.use_strong_momentum and 'price_vs_sma_20' in filtered.columns:
            strong_long = (filtered['predicted_direction'] == 1) & (filtered['price_vs_sma_20'] > 0.01)
            strong_short = (filtered['predicted_direction'] == 0) & (filtered['price_vs_sma_20'] < -0.01)
            filtered = filtered[strong_long | strong_short]
        
        if not self.use_regime_filter or 'market_regime' not in filtered.columns:
            return filtered
        
        # Regime filter: trade with the market direction
        filtered = filtered[
            ((filtered['market_regime'] == 1) & (filtered['predicted_direction'] == 1)) |
            ((filtered['market_regime'] == 0) & (filtered['predicted_direction'] == 0))
        ]
        
        return filtered
    
    def run(
        self,
        predictions: pd.DataFrame,
        actual_data: pd.DataFrame,
        initial_capital: float = 100000
    ) -> Dict:
        """Run backtest."""
        self.trades = []
        self.daily_returns = []
        self.portfolio_values = [initial_capital]
        
        # Merge predictions with actuals (include momentum indicator)
        merge_cols = ['date', 'ticker', 'close', 'target_return', 'market_regime']
        if 'price_vs_sma_20' in actual_data.columns:
            merge_cols.append('price_vs_sma_20')
        
        df = predictions.merge(
            actual_data[merge_cols],
            on=['date', 'ticker'],
            how='inner',
            suffixes=('', '_actual')
        )
        
        if 'close_actual' in df.columns:
            df['close'] = df['close_actual']
        if 'market_regime_actual' in df.columns:
            df['market_regime'] = df['market_regime_actual']
        
        df = df.sort_values('date').reset_index(drop=True)
        dates = df['date'].unique()
        
        portfolio_value = initial_capital
        
        for i, date in enumerate(dates[:-1]):
            day_data = df[df['date'] == date]
            next_date = dates[i + 1]
            
            # Filter by confidence
            candidates = day_data[day_data['confidence'] >= self.confidence_threshold].copy()
            
            # Filter by market regime (KEY IMPROVEMENT)
            candidates = self._filter_by_regime(candidates)
            
            if len(candidates) == 0:
                self.daily_returns.append(0)
                self.portfolio_values.append(portfolio_value)
                continue
            
            # Sort by confidence, take top positions
            candidates = candidates.nlargest(self.max_positions, 'confidence')
            
            day_pnl = 0
            n_trades = len(candidates)
            
            for _, row in candidates.iterrows():
                direction = 'long' if row['predicted_direction'] == 1 else 'short'
                actual_return = row['target_return'] if not pd.isna(row['target_return']) else 0
                
                if direction == 'short':
                    trade_return = -actual_return
                else:
                    trade_return = actual_return
                
                trade_return -= self.transaction_cost
                day_pnl += trade_return / n_trades
                
                self.trades.append(Trade(
                    ticker=row['ticker'],
                    entry_date=date,
                    exit_date=next_date,
                    entry_price=row['close'],
                    exit_price=row['close'] * (1 + actual_return),
                    direction=direction,
                    predicted_return=row['predicted_return'],
                    confidence=row['confidence'],
                    actual_return=actual_return,
                    position_size=self.position_size,
                    market_regime=row.get('market_regime', -1)
                ))
            
            portfolio_value *= (1 + day_pnl * self.position_size * n_trades)
            self.daily_returns.append(day_pnl)
            self.portfolio_values.append(portfolio_value)
        
        return self._calculate_metrics(initial_capital, actual_data)
    
    def _calculate_metrics(self, initial_capital: float, actual_data: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {'total_return': 0, 'error': 'No trades executed'}
        
        total_return = (self.portfolio_values[-1] - initial_capital) / initial_capital
        daily_returns = np.array(self.daily_returns)
        
        # Sharpe Ratio
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0:
            sortino_ratio = np.sqrt(252) * np.mean(daily_returns) / (np.std(negative_returns) + 1e-8)
        else:
            sortino_ratio = sharpe_ratio
        
        # Max Drawdown
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rates
        winning_trades = sum(1 for t in self.trades if t.pnl_pct > 0)
        win_rate = winning_trades / len(self.trades)
        
        # Direction accuracy
        correct_direction = sum(
            1 for t in self.trades
            if (t.direction == 'long' and t.actual_return > 0) or
               (t.direction == 'short' and t.actual_return < 0)
        )
        direction_accuracy = correct_direction / len(self.trades)
        
        # Long vs Short
        long_trades = [t for t in self.trades if t.direction == 'long']
        short_trades = [t for t in self.trades if t.direction == 'short']
        
        long_accuracy = sum(1 for t in long_trades if t.pnl_pct > 0) / max(len(long_trades), 1)
        short_accuracy = sum(1 for t in short_trades if t.pnl_pct > 0) / max(len(short_trades), 1)
        
        # Baseline
        baseline_return = actual_data.groupby('date')['target_return'].mean().sum()
        
        # Profit factor
        gross_profit = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        gross_loss = abs(sum(t.pnl_pct for t in self.trades if t.pnl_pct < 0))
        profit_factor = gross_profit / (gross_loss + 1e-8)
        
        # High confidence trades
        high_conf = [t for t in self.trades if t.confidence >= self.high_confidence_threshold]
        high_conf_accuracy = sum(1 for t in high_conf if t.pnl_pct > 0) / max(len(high_conf), 1)
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'direction_accuracy': direction_accuracy,
            'direction_accuracy_pct': direction_accuracy * 100,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_accuracy_pct': long_accuracy * 100,
            'short_accuracy_pct': short_accuracy * 100,
            'high_conf_trades': len(high_conf),
            'high_conf_accuracy_pct': high_conf_accuracy * 100,
            'avg_trade_return_pct': np.mean([t.pnl_pct for t in self.trades]) * 100,
            'baseline_return': baseline_return,
            'baseline_return_pct': baseline_return * 100,
            'alpha': total_return - baseline_return,
            'alpha_pct': (total_return - baseline_return) * 100,
            'final_portfolio_value': self.portfolio_values[-1],
            'prediction_mae': np.mean([abs(t.predicted_return - t.actual_return) for t in self.trades])
        }
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            'ticker': t.ticker,
            'entry_date': t.entry_date,
            'direction': t.direction,
            'confidence': t.confidence,
            'actual_return': t.actual_return,
            'pnl_pct': t.pnl_pct,
            'market_regime': t.market_regime
        } for t in self.trades])


def run_backtest(
    model,
    df: pd.DataFrame,
    feature_columns: list,
    train_ratio: float = 0.80,  # More training data for better model
    use_regime_filter: bool = True
) -> Dict:
    """Run full backtest with train/test split."""
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    split_idx = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    print(f"Training: {len(train_df)} samples ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"Testing: {len(test_df)} samples ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    
    model.fit(train_df, feature_columns)
    predictions = model.predict(test_df)
    
    backtester = Backtester(use_regime_filter=use_regime_filter)
    metrics = backtester.run(predictions, test_df)
    
    return {
        'metrics': metrics,
        'trade_history': backtester.get_trade_history(),
        'predictions': predictions
    }


def print_results(results: Dict):
    """Print backtest results."""
    m = results['metrics']
    
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    
    print(f"\n📈 RETURNS")
    print(f"  Strategy: {m['total_return_pct']:+.2f}%")
    print(f"  Baseline: {m['baseline_return_pct']:+.2f}%")
    print(f"  Alpha: {m['alpha_pct']:+.2f}%")
    
    print(f"\n📊 RISK")
    print(f"  Sharpe Ratio: {m['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {m['sortino_ratio']:.3f}")
    print(f"  Max Drawdown: {m['max_drawdown_pct']:.2f}%")
    print(f"  Profit Factor: {m['profit_factor']:.2f}")
    
    print(f"\n🎯 ACCURACY")
    print(f"  Direction: {m['direction_accuracy_pct']:.1f}%")
    print(f"  Win Rate: {m['win_rate_pct']:.1f}%")
    print(f"  Long Accuracy: {m['long_accuracy_pct']:.1f}%")
    print(f"  Short Accuracy: {m['short_accuracy_pct']:.1f}%")
    print(f"  High-Conf Accuracy: {m['high_conf_accuracy_pct']:.1f}% ({m['high_conf_trades']} trades)")
    
    print(f"\n📝 TRADES")
    print(f"  Total: {m['total_trades']}")
    print(f"  Long/Short: {m['long_trades']}/{m['short_trades']}")
    print(f"  Avg Return: {m['avg_trade_return_pct']:+.3f}%")
    
    print(f"\n💰 PORTFOLIO: $100,000 → ${m['final_portfolio_value']:,.2f}")
    print("="*70)

