"""
Prediction Models Module
Contains the ML models for stock price prediction.

Models:
1. Ridge Regression (baseline)
2. XGBoost (gradient boosting - strong for tabular data)
3. LSTM (sequence learning for time series)
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import xgboost as xgb
import joblib
import os


class StockPredictor:
    """
    Base stock prediction model with both regression (predict return %)
    and classification (predict direction) capabilities.
    """
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize predictor.
        
        Args:
            model_type: 'ridge', 'xgboost', or 'random_forest'
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.regression_model = None
        self.classification_model = None
        self.feature_columns = None
        self.is_fitted = False
        
        self._init_models()
    
    def _init_models(self):
        """Initialize the underlying ML models."""
        if self.model_type == "ridge":
            self.regression_model = Ridge(alpha=1.0)
            self.classification_model = LogisticRegression(max_iter=1000, C=0.1)
            
        elif self.model_type == "xgboost":
            self.regression_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            self.classification_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
        elif self.model_type == "random_forest":
            self.regression_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.classification_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: list
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features and targets from dataframe.
        
        Returns:
            X: Feature matrix
            y_reg: Regression target (return %)
            y_clf: Classification target (0/1 direction)
        """
        # Remove rows with NaN in features or targets
        df_clean = df.dropna(subset=feature_columns + ['target_return', 'target_direction'])
        
        X = df_clean[feature_columns].values
        y_reg = df_clean['target_return'].values
        y_clf = df_clean['target_direction'].values
        
        return X, y_reg, y_clf
    
    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: list
    ) -> Dict[str, float]:
        """
        Train the model on the provided data.
        
        Args:
            df: DataFrame with features and targets
            feature_columns: List of feature column names
            
        Returns:
            Dictionary of training metrics
        """
        self.feature_columns = feature_columns
        
        X, y_reg, y_clf = self.prepare_data(df, feature_columns)
        
        # Handle any remaining NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train both models
        print(f"Training {self.model_type} models on {len(X)} samples...")
        
        self.regression_model.fit(X_scaled, y_reg)
        self.classification_model.fit(X_scaled, y_clf)
        
        self.is_fitted = True
        
        # Calculate training metrics
        reg_pred = self.regression_model.predict(X_scaled)
        clf_pred = self.classification_model.predict(X_scaled)
        
        metrics = {
            'mae': mean_absolute_error(y_reg, reg_pred),
            'rmse': np.sqrt(mean_squared_error(y_reg, reg_pred)),
            'direction_accuracy': accuracy_score(y_clf, clf_pred),
            'f1_score': f1_score(y_clf, clf_pred)
        }
        
        print(f"Training complete:")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
        
        return metrics
    
    def predict(
        self,
        df: pd.DataFrame,
        return_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with features
            return_confidence: Include confidence scores
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        df_pred = df.dropna(subset=self.feature_columns)
        X = df_pred[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        # Predictions
        results = df_pred[['date', 'ticker', 'close']].copy()
        results['predicted_return'] = self.regression_model.predict(X_scaled)
        results['predicted_direction'] = self.classification_model.predict(X_scaled)
        
        if return_confidence:
            # Get probability estimates for direction
            proba = self.classification_model.predict_proba(X_scaled)
            results['confidence'] = np.max(proba, axis=1)
            results['prob_up'] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            results['prob_down'] = 1 - results['prob_up']
        
        return results
    
    def predict_single(
        self,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Make prediction for a single observation.
        Useful for real-time predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Create feature vector
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        pred_return = self.regression_model.predict(X_scaled)[0]
        pred_direction = self.classification_model.predict(X_scaled)[0]
        proba = self.classification_model.predict_proba(X_scaled)[0]
        
        return {
            'predicted_return': pred_return,
            'predicted_direction': 'UP' if pred_direction == 1 else 'DOWN',
            'confidence': max(proba),
            'prob_up': proba[1] if len(proba) > 1 else proba[0],
            'prob_down': proba[0] if len(proba) > 1 else 1 - proba[0]
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        Works best with XGBoost and Random Forest.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.regression_model, 'feature_importances_'):
            importance = self.regression_model.feature_importances_
        elif hasattr(self.regression_model, 'coef_'):
            importance = np.abs(self.regression_model.coef_)
        else:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, filepath: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model_type': self.model_type,
            'scaler': self.scaler,
            'regression_model': self.regression_model,
            'classification_model': self.classification_model,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'StockPredictor':
        """Load model from disk."""
        data = joblib.load(filepath)
        model = cls(model_type=data['model_type'])
        model.scaler = data['scaler']
        model.regression_model = data['regression_model']
        model.classification_model = data['classification_model']
        model.feature_columns = data['feature_columns']
        model.is_fitted = data['is_fitted']
        print(f"Model loaded from {filepath}")
        return model


def cross_validate_model(
    df: pd.DataFrame,
    feature_columns: list,
    model_type: str = "xgboost",
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Perform time-series cross-validation.
    
    Uses TimeSeriesSplit to ensure we always train on past data
    and test on future data (no look-ahead bias).
    """
    print(f"\nCross-validating {model_type} model with {n_splits} splits...")
    
    # Sort by date
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    # Prepare data
    model = StockPredictor(model_type)
    X, y_reg, y_clf = model.prepare_data(df_sorted, feature_columns)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = {
        'mae': [],
        'rmse': [],
        'direction_accuracy': [],
        'f1': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]
        y_clf_train, y_clf_test = y_clf[train_idx], y_clf[test_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model._init_models()
        model.regression_model.fit(X_train_scaled, y_reg_train)
        model.classification_model.fit(X_train_scaled, y_clf_train)
        
        # Predict
        reg_pred = model.regression_model.predict(X_test_scaled)
        clf_pred = model.classification_model.predict(X_test_scaled)
        
        # Metrics
        results['mae'].append(mean_absolute_error(y_reg_test, reg_pred))
        results['rmse'].append(np.sqrt(mean_squared_error(y_reg_test, reg_pred)))
        results['direction_accuracy'].append(accuracy_score(y_clf_test, clf_pred))
        results['f1'].append(f1_score(y_clf_test, clf_pred))
        
        print(f"  Fold {fold+1}: Accuracy={results['direction_accuracy'][-1]:.2%}, MAE={results['mae'][-1]:.6f}")
    
    # Aggregate results
    summary = {
        'mae_mean': np.mean(results['mae']),
        'mae_std': np.std(results['mae']),
        'rmse_mean': np.mean(results['rmse']),
        'rmse_std': np.std(results['rmse']),
        'accuracy_mean': np.mean(results['direction_accuracy']),
        'accuracy_std': np.std(results['direction_accuracy']),
        'f1_mean': np.mean(results['f1']),
        'f1_std': np.std(results['f1']),
        'fold_results': results
    }
    
    print(f"\nCross-validation summary:")
    print(f"  MAE: {summary['mae_mean']:.6f} ± {summary['mae_std']:.6f}")
    print(f"  Direction Accuracy: {summary['accuracy_mean']:.2%} ± {summary['accuracy_std']:.2%}")
    
    return summary


if __name__ == "__main__":
    # Test model
    print("Testing model module...")
    
    from data_fetcher import fetch_stock_data, clean_data, calculate_returns
    from features import create_all_features, get_feature_columns
    
    # Get sample data
    df = fetch_stock_data("AAPL", period="1y")
    df = clean_data(df)
    df = calculate_returns(df)
    df = create_all_features(df)
    
    feature_cols = get_feature_columns(df)
    
    # Test model training
    model = StockPredictor("xgboost")
    metrics = model.fit(df.iloc[:-20], feature_cols)  # Hold out last 20 days
    
    # Test prediction
    predictions = model.predict(df.iloc[-20:])
    print("\nSample predictions:")
    print(predictions.head())

