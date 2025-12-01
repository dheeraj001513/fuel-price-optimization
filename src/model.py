"""
Machine Learning Model Module
Trains and evaluates XGBoost model for demand prediction.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemandModel:
    """XGBoost model for predicting fuel demand (volume)."""
    
    def __init__(self, model_params: Optional[Dict] = None):
        """
        Initialize demand prediction model.
        
        Args:
            model_params: XGBoost hyperparameters
        """
        self.model_params = model_params or self._default_params()
        self.model = XGBRegressor(**self.model_params)
        self.feature_importance_ = None
        self.is_trained = False
    
    def _default_params(self) -> Dict:
        """Default XGBoost parameters."""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series,
             validation_split: float = 0.2,
             early_stopping_rounds: int = 10) -> Dict:
        """
        Train the demand prediction model.
        
        Args:
            X: Feature DataFrame
            y: Target (volume)
            validation_split: Fraction of data for validation
            early_stopping_rounds: Early stopping rounds
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training demand prediction model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, shuffle=False
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        # Feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics = {
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'n_features': len(X.columns),
            'n_samples_train': len(X_train),
            'n_samples_val': len(X_val)
        }
        
        self.is_trained = True
        
        logger.info(f"Training complete. Val MAE: {val_mae:.2f}, Val R²: {val_r2:.3f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict demand (volume) for given features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predicted volumes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv: int = 5) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target
            cv: Number of folds
            
        Returns:
            Dictionary with CV metrics
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        scores_mae = cross_val_score(self.model, X, y, cv=cv, 
                                     scoring='neg_mean_absolute_error', n_jobs=-1)
        scores_rmse = cross_val_score(self.model, X, y, cv=cv,
                                      scoring='neg_mean_squared_error', n_jobs=-1)
        
        cv_mae = -scores_mae.mean()
        cv_mae_std = scores_mae.std()
        cv_rmse = np.sqrt(-scores_rmse.mean())
        cv_rmse_std = np.sqrt(scores_rmse.std())
        
        return {
            'cv_mae_mean': cv_mae,
            'cv_mae_std': cv_mae_std,
            'cv_rmse_mean': cv_rmse,
            'cv_rmse_std': cv_rmse_std
        }
    
    def save(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_importance': self.feature_importance_,
                'model_params': self.model_params
            }, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load trained model from file.
        
        Args:
            filepath: Path to saved model
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.feature_importance_ = data['feature_importance']
        self.model_params = data.get('model_params', {})
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    print("DemandModel module - use with training pipeline")

