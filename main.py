"""
Main execution script for Fuel Price Optimization System
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path (current directory)
sys.path.append(str(Path(__file__).parent / 'src'))

from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from model import DemandModel
from business_rules import BusinessRules
from optimization import PriceOptimizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(data_path: str = "data/oil_retail_history.csv",
                model_save_path: str = "models/demand_model.pkl"):
    """Train the demand prediction model."""
    logger.info("=" * 60)
    logger.info("TRAINING DEMAND PREDICTION MODEL")
    logger.info("=" * 60)
    
    # Data pipeline
    pipeline = DataPipeline()
    df = pipeline.process(data_path, data_type='csv')
    
    # Feature engineering
    fe = FeatureEngineer()
    df_features = fe.create_all_features(df)
    
    # Prepare for training
    X, y = fe.prepare_for_training(df_features, target_col='volume')
    
    # Remove rows with NaN in target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    logger.info(f"Training data: {len(X)} samples, {len(X.columns)} features")
    
    # Train model
    model = DemandModel()
    metrics = model.train(X, y)
    
    # Print metrics
    print("\n" + "=" * 60)
    print("TRAINING METRICS")
    print("=" * 60)
    print(f"Training MAE: {metrics['train_mae']:.2f}")
    print(f"Validation MAE: {metrics['val_mae']:.2f}")
    print(f"Training RMSE: {metrics['train_rmse']:.2f}")
    print(f"Validation RMSE: {metrics['val_rmse']:.2f}")
    print(f"Training R²: {metrics['train_r2']:.3f}")
    print(f"Validation R²: {metrics['val_r2']:.3f}")
    
    # Cross-validation
    cv_metrics = model.cross_validate(X, y, cv=5)
    print(f"\nCross-Validation MAE: {cv_metrics['cv_mae_mean']:.2f} ± {cv_metrics['cv_mae_std']:.2f}")
    print(f"Cross-Validation RMSE: {cv_metrics['cv_rmse_mean']:.2f} ± {cv_metrics['cv_rmse_std']:.2f}")
    
    # Feature importance
    print("\n" + "=" * 60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 60)
    print(model.feature_importance_.head(10).to_string(index=False))
    
    # Save model
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path)
    
    logger.info(f"\nModel saved to {model_save_path}")
    
    return model, fe


def predict_price(json_path: str = "data/today_example.json",
                 model_path: str = "models/demand_model.pkl",
                 historical_data_path: str = "data/oil_retail_history.csv"):
    """Predict optimal price for given daily input."""
    logger.info("=" * 60)
    logger.info("PREDICTING OPTIMAL PRICE")
    logger.info("=" * 60)
    
    # Load model
    model = DemandModel()
    model.load(model_path)
    
    # Load historical data for feature engineering context
    pipeline = DataPipeline()
    df_historical = pipeline.process(historical_data_path, data_type='csv')
    
    # Load daily input
    with open(json_path, 'r') as f:
        daily_data = json.load(f)
    
    logger.info(f"Daily input: {daily_data}")
    
    # Feature engineering
    fe = FeatureEngineer()
    
    # Combine historical + today for feature computation
    df_today = pipeline.transform_daily_data(daily_data)
    df_combined = pd.concat([df_historical, df_today], ignore_index=True)
    df_features = fe.create_all_features(df_combined)
    
    # Get features for today (last row)
    today_features = df_features.iloc[[-1]].copy()
    
    # Prepare features
    exclude_cols = ['date', 'volume', 'revenue', 'profit']
    feature_cols = [col for col in today_features.columns 
                   if col not in exclude_cols and today_features[col].dtype in ['float64', 'int64']]
    
    X_today = today_features[feature_cols].fillna(today_features[feature_cols].median())
    
    # Business rules
    rules = BusinessRules()
    
    # Price optimizer
    optimizer = PriceOptimizer(model, rules, fe)
    
    # Extract inputs
    current_price = daily_data['price']
    cost = daily_data['cost']
    competitor_prices = [
        daily_data['comp1_price'],
        daily_data['comp2_price'],
        daily_data['comp3_price']
    ]
    
    # Optimize
    result = optimizer.optimize_price(
        current_price=current_price,
        cost=cost,
        competitor_prices=competitor_prices,
        features=today_features
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("PRICE RECOMMENDATION")
    print("=" * 60)
    print(f"Current Price: ${current_price:.3f}/L")
    print(f"Cost: ${cost:.3f}/L")
    print(f"Average Competitor Price: ${np.mean(competitor_prices):.3f}/L")
    print(f"\nRecommended Price: ${result['recommended_price']:.3f}/L")
    print(f"Price Change: {result['price_change_pct']:.2f}%")
    print(f"\nExpected Volume: {result['predicted_volume']:.2f} L")
    print(f"Expected Revenue: ${result['expected_revenue']:.2f}")
    print(f"Expected Profit: ${result['expected_profit']:.2f}")
    print(f"Profit Margin: {result['profit_margin']:.2f}%")
    
    # Save results
    output_path = "output/recommendation.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'input': daily_data,
            'recommendation': result
        }, f, indent=2)
    
    logger.info(f"\nRecommendation saved to {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Fuel Price Optimization System')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Path to today_example.json for prediction')
    parser.add_argument('--data', type=str, default='data/oil_retail_history.csv',
                       help='Path to historical data CSV')
    parser.add_argument('--model', type=str, default='models/demand_model.pkl',
                       help='Path to saved model')
    
    args = parser.parse_args()
    
    if args.train:
        train_model(args.data, args.model)
    elif args.predict:
        predict_price(args.predict, args.model, args.data)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

