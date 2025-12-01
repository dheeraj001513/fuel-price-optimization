"""
Feature Engineering Module
Computes derived features such as price differentials, lag features, moving averages, etc.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for price optimization model."""
    
    def __init__(self, lag_periods: List[int] = [1, 7, 30], 
                 ma_periods: List[int] = [7, 30, 90]):
        """
        Initialize feature engineer.
        
        Args:
            lag_periods: List of periods for lag features
            ma_periods: List of periods for moving averages
        """
        self.lag_periods = lag_periods
        self.ma_periods = ma_periods
    
    def compute_price_differentials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute price differentials relative to competitors.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with price differential features
        """
        df = df.copy()
        
        # Price difference from each competitor
        for i in [1, 2, 3]:
            comp_col = f'comp{i}_price'
            if comp_col in df.columns:
                df[f'price_diff_comp{i}'] = df['price'] - df[comp_col]
                df[f'price_ratio_comp{i}'] = df['price'] / (df[comp_col] + 1e-6)
        
        # Average competitor price
        comp_cols = [col for col in df.columns if col.startswith('comp') and col.endswith('_price')]
        if comp_cols:
            df['avg_comp_price'] = df[comp_cols].mean(axis=1)
            df['min_comp_price'] = df[comp_cols].min(axis=1)
            df['max_comp_price'] = df[comp_cols].max(axis=1)
            df['price_diff_avg_comp'] = df['price'] - df['avg_comp_price']
            df['price_ratio_avg_comp'] = df['price'] / (df['avg_comp_price'] + 1e-6)
        
        # Profit margin
        if 'cost' in df.columns and 'price' in df.columns:
            df['profit_margin'] = df['price'] - df['cost']
            df['profit_margin_pct'] = (df['price'] - df['cost']) / (df['cost'] + 1e-6) * 100
        
        return df
    
    def compute_lag_features(self, df: pd.DataFrame, 
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute lag features for specified columns.
        
        Args:
            df: Input DataFrame (must be sorted by date)
            columns: List of columns to create lags for. If None, uses price and volume.
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        if columns is None:
            columns = ['price', 'volume']
            if 'avg_comp_price' in df.columns:
                columns.append('avg_comp_price')
        
        # Ensure sorted by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for lag in self.lag_periods:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        return df
    
    def compute_moving_averages(self, df: pd.DataFrame,
                               columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute moving averages for specified columns.
        
        Args:
            df: Input DataFrame (must be sorted by date)
            columns: List of columns to compute MAs for. If None, uses price and volume.
            
        Returns:
            DataFrame with moving average features
        """
        df = df.copy()
        
        if columns is None:
            columns = ['price', 'volume']
            if 'avg_comp_price' in df.columns:
                columns.append('avg_comp_price')
        
        # Ensure sorted by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for period in self.ma_periods:
                df[f'{col}_ma{period}'] = df[col].rolling(window=period, min_periods=1).mean()
                df[f'{col}_std{period}'] = df[col].rolling(window=period, min_periods=1).std()
        
        return df
    
    def compute_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute temporal features (day of week, month, etc.).
        
        Args:
            df: Input DataFrame with 'date' column
            
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        if 'date' not in df.columns:
            return df
        
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def compute_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute demand-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with demand features
        """
        df = df.copy()
        
        if 'volume' in df.columns and 'price' in df.columns:
            # Price elasticity proxy (volume/price)
            df['volume_price_ratio'] = df['volume'] / (df['price'] + 1e-6)
            
            # Revenue
            df['revenue'] = df['price'] * df['volume']
            
            # Profit
            if 'cost' in df.columns:
                df['profit'] = (df['price'] - df['cost']) * df['volume']
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all features
        """
        logger.info("Computing all features...")
        
        # Ensure sorted by date first
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        # Step 1: Price differentials
        df = self.compute_price_differentials(df)
        
        # Step 2: Temporal features
        df = self.compute_temporal_features(df)
        
        # Step 3: Lag features
        df = self.compute_lag_features(df)
        
        # Step 4: Moving averages
        df = self.compute_moving_averages(df)
        
        # Step 5: Demand features
        df = self.compute_demand_features(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df
    
    def prepare_for_training(self, df: pd.DataFrame, 
                           target_col: str = 'volume') -> tuple:
        """
        Prepare data for model training.
        
        Args:
            df: DataFrame with all features
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Remove non-feature columns
        exclude_cols = ['date', target_col, 'revenue', 'profit']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        # Handle missing values
        X = X.fillna(X.median())
        
        return X, y


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range('2022-01-01', periods=100, freq='D')
    sample_df = pd.DataFrame({
        'date': dates,
        'price': np.random.uniform(1.0, 2.0, 100),
        'cost': np.random.uniform(0.8, 1.2, 100),
        'comp1_price': np.random.uniform(1.0, 2.0, 100),
        'comp2_price': np.random.uniform(1.0, 2.0, 100),
        'comp3_price': np.random.uniform(1.0, 2.0, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    
    fe = FeatureEngineer()
    df_features = fe.create_all_features(sample_df)
    print(f"Original columns: {len(sample_df.columns)}")
    print(f"Features after engineering: {len(df_features.columns)}")
    print("\nFeature columns:", df_features.columns.tolist())

