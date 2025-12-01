"""
Data Ingestion and Transformation Pipeline
Handles reading, cleaning, validating, and transforming daily fuel price data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Pipeline for data ingestion, validation, and transformation."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data pipeline.
        
        Args:
            config: Configuration dictionary with validation rules
        """
        self.config = config or self._default_config()
        self.processed_data = None
        
    def _default_config(self) -> Dict:
        """Default configuration for data validation."""
        return {
            'min_price': 0.5,
            'max_price': 3.0,
            'min_volume': 0,
            'max_volume': 100000,
            'required_columns': ['date', 'price', 'cost', 'comp1_price', 'comp2_price', 'comp3_price', 'volume'],
            'date_format': '%Y-%m-%d'
        }
    
    def ingest_csv(self, file_path: str) -> pd.DataFrame:
        """
        Ingest historical data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with ingested data
        """
        logger.info(f"Ingesting data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error ingesting CSV: {e}")
            raise
    
    def ingest_json(self, file_path: str) -> Dict:
        """
        Ingest daily data from JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with daily data
        """
        logger.info(f"Ingesting daily data from {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info("Successfully loaded daily data")
            return data
        except Exception as e:
            logger.error(f"Error ingesting JSON: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Validate and clean the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, list of validation errors)
        """
        errors = []
        df_clean = df.copy()
        
        # Check required columns
        missing_cols = set(self.config['required_columns']) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date column
        if 'date' in df_clean.columns:
            try:
                df_clean['date'] = pd.to_datetime(df_clean['date'])
            except Exception as e:
                errors.append(f"Error parsing dates: {e}")
        
        # Validate price columns
        price_cols = ['price', 'cost', 'comp1_price', 'comp2_price', 'comp3_price']
        for col in price_cols:
            if col in df_clean.columns:
                # Remove invalid prices
                invalid_mask = (df_clean[col] < self.config['min_price']) | \
                             (df_clean[col] > self.config['max_price'])
                if invalid_mask.any():
                    errors.append(f"Found {invalid_mask.sum()} invalid values in {col}")
                    df_clean.loc[invalid_mask, col] = np.nan
        
        # Validate volume
        if 'volume' in df_clean.columns:
            invalid_volume = (df_clean['volume'] < self.config['min_volume']) | \
                           (df_clean['volume'] > self.config['max_volume'])
            if invalid_volume.any():
                errors.append(f"Found {invalid_volume.sum()} invalid volume values")
                df_clean.loc[invalid_volume, 'volume'] = np.nan
        
        # Remove rows with critical missing values
        critical_cols = ['date', 'price', 'cost', 'volume']
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=critical_cols)
        removed_rows = initial_rows - len(df_clean)
        if removed_rows > 0:
            errors.append(f"Removed {removed_rows} rows with missing critical values")
        
        # Sort by date
        if 'date' in df_clean.columns:
            df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Validation complete. {len(errors)} warnings. Final dataset: {len(df_clean)} rows")
        return df_clean, errors
    
    def transform_daily_data(self, daily_data: Dict) -> pd.DataFrame:
        """
        Transform daily JSON data into DataFrame format compatible with historical data.
        
        Args:
            daily_data: Dictionary with daily data
            
        Returns:
            DataFrame with single row
        """
        df = pd.DataFrame([daily_data])
        
        # Convert date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def process(self, file_path: str, data_type: str = 'csv') -> pd.DataFrame:
        """
        Complete pipeline: ingest, validate, and return processed data.
        
        Args:
            file_path: Path to data file
            data_type: Type of data file ('csv' or 'json')
            
        Returns:
            Processed DataFrame
        """
        if data_type == 'csv':
            df = self.ingest_csv(file_path)
        elif data_type == 'json':
            data = self.ingest_json(file_path)
            df = self.transform_daily_data(data)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")
        
        df_clean, errors = self.validate_data(df)
        self.processed_data = df_clean
        
        if errors:
            logger.warning(f"Validation warnings: {errors}")
        
        return df_clean


if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline()
    
    # Test with sample data if available
    if Path("data/oil_retail_history.csv").exists():
        df = pipeline.process("data/oil_retail_history.csv", data_type='csv')
        print(f"Processed {len(df)} records")
        print(df.head())

