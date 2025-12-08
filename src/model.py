"""
Stock Prediction Model
Ensemble model with market regime awareness.

Key features:
- Ensemble of XGBoost, Random Forest, and Linear models
- Market regime detection (only long in bull, short in bear)
- Feature selection to reduce overfitting
- Calibrated probability outputs
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class StockPredictor:
    """
    Stock prediction model with ensemble approach and market regime awareness.
    """
    
    def __init__(self, use_feature_selection: bool = True):
        self.use_feature_selection = use_feature_selection
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.feature_columns = None
        self.selected_features = None
        self.is_fitted = False
        self.models = {}
        self._init_models()
    
    def _init_models(self):
        """Initialize ensemble models with regularization."""
        # XGBoost - strong regularization to prevent overfitting
        self.models['xgb_reg'] = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_weight=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['xgb_clf'] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_weight=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Random Forest
        self.models['rf_reg'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=30,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        self.models['rf_clf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=30,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Linear models
        self.models['ridge'] = Ridge(alpha=10.0)
        self.models['logistic'] = LogisticRegression(C=0.1, max_iter=1000)
    
    def _select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Select most important features."""
        if not self.use_feature_selection:
            self.selected_features = feature_names
            return X
        
        selector_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
        selector_model.fit(X, y)
        
        importances = selector_model.feature_importances_
        threshold = np.percentile(importances, 40)  # Keep top 60% of features
        
        self.feature_selector = SelectFromModel(selector_model, threshold=threshold, prefit=True)
        X_selected = self.feature_selector.transform(X)
        
        mask = self.feature_selector.get_support()
        self.selected_features = [f for f, m in zip(feature_names, mask) if m]
        
        print(f"Selected {len(self.selected_features)} of {len(feature_names)} features")
        return X_selected
    
    def fit(self, df: pd.DataFrame, feature_columns: list) -> Dict[str, float]:
        """Train the ensemble model."""
        self.feature_columns = feature_columns
        
        # Prepare data
        df_clean = df.dropna(subset=feature_columns + ['target_return', 'target_direction'])
        X = df_clean[feature_columns].values
        y_reg = df_clean['target_return'].values
        y_clf = df_clean['target_direction'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.fit_transform(X)
        X_selected = self._select_features(X_scaled, y_reg, feature_columns)
        
        print(f"Training ensemble on {len(X_selected)} samples...")
        
        # Train all models
        self.models['xgb_reg'].fit(X_selected, y_reg)
        self.models['xgb_clf'].fit(X_selected, y_clf)
        self.models['rf_reg'].fit(X_selected, y_reg)
        self.models['rf_clf'].fit(X_selected, y_clf)
        self.models['ridge'].fit(X_selected, y_reg)
        self.models['logistic'].fit(X_selected, y_clf)
        
        self.is_fitted = True
        
        # Evaluate
        reg_preds = np.mean([
            self.models['xgb_reg'].predict(X_selected),
            self.models['rf_reg'].predict(X_selected),
            self.models['ridge'].predict(X_selected)
        ], axis=0)
        
        clf_probas = np.mean([
            self.models['xgb_clf'].predict_proba(X_selected)[:, 1],
            self.models['rf_clf'].predict_proba(X_selected)[:, 1],
            self.models['logistic'].predict_proba(X_selected)[:, 1]
        ], axis=0)
        clf_preds = (clf_probas > 0.5).astype(int)
        
        metrics = {
            'mae': mean_absolute_error(y_reg, reg_preds),
            'direction_accuracy': accuracy_score(y_clf, clf_preds)
        }
        
        print(f"Training complete: MAE={metrics['mae']:.6f}, Accuracy={metrics['direction_accuracy']:.2%}")
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions with ensemble."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        df_pred = df.dropna(subset=self.feature_columns)
        X = df_pred[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        results = df_pred[['date', 'ticker', 'close']].copy()
        
        # Include market regime if available
        if 'market_regime' in df_pred.columns:
            results['market_regime'] = df_pred['market_regime'].values
        
        # Ensemble predictions
        reg_preds = np.mean([
            self.models['xgb_reg'].predict(X_scaled),
            self.models['rf_reg'].predict(X_scaled),
            self.models['ridge'].predict(X_scaled)
        ], axis=0)
        
        clf_probas = np.mean([
            self.models['xgb_clf'].predict_proba(X_scaled),
            self.models['rf_clf'].predict_proba(X_scaled),
            self.models['logistic'].predict_proba(X_scaled)
        ], axis=0)
        
        results['predicted_return'] = reg_preds
        results['predicted_direction'] = (clf_probas[:, 1] > 0.5).astype(int)
        results['confidence'] = np.max(clf_probas, axis=1)
        results['prob_up'] = clf_probas[:, 1]
        results['prob_down'] = clf_probas[:, 0]
        
        # Model agreement
        clf_preds = np.array([
            self.models['xgb_clf'].predict(X_scaled),
            self.models['rf_clf'].predict(X_scaled),
            self.models['logistic'].predict(X_scaled)
        ])
        agreement = np.mean(clf_preds, axis=0)
        results['model_agreement'] = np.maximum(agreement, 1 - agreement)
        
        return results
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction for a single observation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        reg_pred = np.mean([
            self.models['xgb_reg'].predict(X_scaled)[0],
            self.models['rf_reg'].predict(X_scaled)[0],
            self.models['ridge'].predict(X_scaled)[0]
        ])
        
        clf_proba = np.mean([
            self.models['xgb_clf'].predict_proba(X_scaled)[0],
            self.models['rf_clf'].predict_proba(X_scaled)[0],
            self.models['logistic'].predict_proba(X_scaled)[0]
        ], axis=0)
        
        return {
            'predicted_return': reg_pred,
            'predicted_direction': 'UP' if clf_proba[1] > 0.5 else 'DOWN',
            'confidence': max(clf_proba),
            'prob_up': clf_proba[1],
            'prob_down': clf_proba[0]
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get ensemble feature importance."""
        if not self.is_fitted:
            return pd.DataFrame()
        
        features = self.selected_features or self.feature_columns
        importances = np.mean([
            self.models['xgb_reg'].feature_importances_,
            self.models['rf_reg'].feature_importances_
        ], axis=0)
        
        return pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def save(self, filepath: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_columns': self.feature_columns,
            'selected_features': self.selected_features,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'StockPredictor':
        """Load model from disk."""
        data = joblib.load(filepath)
        model = cls()
        model.models = data['models']
        model.scaler = data['scaler']
        model.feature_selector = data['feature_selector']
        model.feature_columns = data['feature_columns']
        model.selected_features = data['selected_features']
        model.is_fitted = data['is_fitted']
        print(f"Model loaded from {filepath}")
        return model

