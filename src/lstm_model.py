"""
LSTM Model for Sequence Prediction
Learns temporal patterns in stock price movements.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
import os


class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences."""
    
    def __init__(self, X: np.ndarray, y_reg: np.ndarray, y_clf: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y_reg = torch.FloatTensor(y_reg)
        self.y_clf = torch.LongTensor(y_clf)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_clf[idx]


class LSTMModel(nn.Module):
    """
    LSTM network for stock prediction.
    
    Architecture:
    - LSTM layers for sequence learning
    - Fully connected layers for prediction
    - Dual output: regression (return %) and classification (direction)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Regression head (predict return %)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Classification head (predict direction)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # 2 classes: down, up
        )
    
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Predictions
        reg_output = self.regression_head(last_hidden).squeeze(-1)
        clf_output = self.classification_head(last_hidden)
        
        return reg_output, clf_output


class LSTMPredictor:
    """
    LSTM-based stock predictor.
    Wraps the PyTorch model with scikit-learn-like interface.
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        device: str = None
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
    
    def _create_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        For each prediction, we use the last `sequence_length` days of data.
        """
        sequences = []
        y_reg = []
        y_clf = []
        
        # Process each ticker separately
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].sort_values('date')
            
            if len(ticker_data) < self.sequence_length + 1:
                continue
            
            features = ticker_data[feature_columns].values
            targets_reg = ticker_data['target_return'].values
            targets_clf = ticker_data['target_direction'].values
            
            # Create sliding window sequences
            for i in range(self.sequence_length, len(ticker_data)):
                seq = features[i - self.sequence_length:i]
                sequences.append(seq)
                y_reg.append(targets_reg[i - 1])  # Target for last day in sequence
                y_clf.append(targets_clf[i - 1])
        
        return np.array(sequences), np.array(y_reg), np.array(y_clf)
    
    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train the LSTM model.
        """
        self.feature_columns = feature_columns
        
        # Create sequences
        X, y_reg, y_clf = self._create_sequences(df, feature_columns)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_reg = np.nan_to_num(y_reg, nan=0.0)
        
        # Scale features
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_flat)
        X = X_scaled.reshape(n_samples, seq_len, n_features)
        
        # Create dataset and dataloader
        dataset = StockDataset(X, y_reg, y_clf)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = LSTMModel(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        # Loss functions and optimizer
        reg_criterion = nn.MSELoss()
        clf_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        if verbose:
            print(f"Training LSTM on {len(X)} sequences...")
            print(f"Device: {self.device}")
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y_reg, batch_y_clf in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y_reg = batch_y_reg.to(self.device)
                batch_y_clf = batch_y_clf.to(self.device)
                
                optimizer.zero_grad()
                
                reg_pred, clf_pred = self.model(batch_X)
                
                # Combined loss
                loss = reg_criterion(reg_pred, batch_y_reg) + clf_criterion(clf_pred, batch_y_clf)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.6f}")
        
        self.is_fitted = True
        
        # Calculate training metrics
        return self._evaluate(X, y_reg, y_clf)
    
    def _evaluate(
        self,
        X: np.ndarray,
        y_reg: np.ndarray,
        y_clf: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reg_pred, clf_pred = self.model(X_tensor)
            
            reg_pred = reg_pred.cpu().numpy()
            clf_pred = torch.argmax(clf_pred, dim=1).cpu().numpy()
        
        mae = np.mean(np.abs(y_reg - reg_pred))
        accuracy = np.mean(y_clf == clf_pred)
        
        return {
            'mae': mae,
            'direction_accuracy': accuracy
        }
    
    def predict(
        self,
        df: pd.DataFrame,
        return_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions on new data.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        results = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].sort_values('date')
            
            if len(ticker_data) < self.sequence_length:
                continue
            
            features = ticker_data[self.feature_columns].values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Create sequences and predict
            for i in range(self.sequence_length, len(ticker_data) + 1):
                seq = features_scaled[i - self.sequence_length:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    reg_pred, clf_pred = self.model(seq_tensor)
                    proba = torch.softmax(clf_pred, dim=1)
                
                row_idx = i - 1
                result = {
                    'date': ticker_data.iloc[row_idx]['date'],
                    'ticker': ticker,
                    'close': ticker_data.iloc[row_idx]['close'],
                    'predicted_return': reg_pred.cpu().item(),
                    'predicted_direction': torch.argmax(clf_pred, dim=1).cpu().item()
                }
                
                if return_confidence:
                    result['confidence'] = proba.max().cpu().item()
                    result['prob_up'] = proba[0, 1].cpu().item()
                    result['prob_down'] = proba[0, 0].cpu().item()
                
                results.append(result)
        
        return pd.DataFrame(results)
    
    def save(self, filepath: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'config': {
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'input_size': len(self.feature_columns)
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LSTMPredictor':
        """Load model from disk."""
        checkpoint = torch.load(filepath)
        
        predictor = cls(
            sequence_length=checkpoint['config']['sequence_length'],
            hidden_size=checkpoint['config']['hidden_size'],
            num_layers=checkpoint['config']['num_layers']
        )
        
        predictor.scaler = checkpoint['scaler']
        predictor.feature_columns = checkpoint['feature_columns']
        
        predictor.model = LSTMModel(
            input_size=checkpoint['config']['input_size'],
            hidden_size=checkpoint['config']['hidden_size'],
            num_layers=checkpoint['config']['num_layers']
        ).to(predictor.device)
        
        predictor.model.load_state_dict(checkpoint['model_state_dict'])
        predictor.is_fitted = True
        
        print(f"Model loaded from {filepath}")
        return predictor


if __name__ == "__main__":
    print("LSTM model module loaded successfully")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

