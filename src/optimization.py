"""
Price Optimization Module
Finds the optimal price that maximizes profit using demand prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
try:
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, using simple grid search")
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceOptimizer:
    """Optimize price to maximize profit."""
    
    def __init__(self, demand_model, business_rules, feature_engineer):
        """
        Initialize price optimizer.
        
        Args:
            demand_model: Trained model that predicts volume given price
            business_rules: BusinessRules instance
            feature_engineer: FeatureEngineer instance
        """
        self.demand_model = demand_model
        self.business_rules = business_rules
        self.feature_engineer = feature_engineer
    
    def predict_demand(self, price: float, features: pd.DataFrame) -> float:
        """
        Predict demand (volume) for a given price.
        
        Args:
            price: Proposed price
            features: Feature DataFrame (will be updated with new price)
            
        Returns:
            Predicted volume
        """
        # Create a copy and update price
        features_copy = features.copy()
        features_copy['price'] = price
        
        # Recompute price-dependent features
        if 'avg_comp_price' in features_copy.columns:
            features_copy['price_diff_avg_comp'] = price - features_copy['avg_comp_price']
            features_copy['price_ratio_avg_comp'] = price / (features_copy['avg_comp_price'] + 1e-6)
        
        # Recompute profit margin
        if 'cost' in features_copy.columns:
            features_copy['profit_margin'] = price - features_copy['cost']
            features_copy['profit_margin_pct'] = (price - features_copy['cost']) / (features_copy['cost'] + 1e-6) * 100
        
        # Prepare features for prediction
        exclude_cols = ['date', 'volume', 'revenue', 'profit']
        feature_cols = [col for col in features_copy.columns 
                       if col not in exclude_cols and features_copy[col].dtype in ['float64', 'int64']]
        
        X = features_copy[feature_cols].fillna(features_copy[feature_cols].median())
        
        # Predict volume
        volume = self.demand_model.predict(X)[0]
        return max(0, volume)  # Ensure non-negative
    
    def calculate_profit(self, price: float, cost: float, 
                        features: pd.DataFrame) -> float:
        """
        Calculate expected profit for a given price.
        
        Args:
            price: Proposed price
            cost: Cost per liter
            features: Feature DataFrame
            
        Returns:
            Expected profit (price - cost) * predicted_volume
        """
        volume = self.predict_demand(price, features)
        profit = (price - cost) * volume
        return profit
    
    def optimize_price(self, current_price: float,
                      cost: float,
                      competitor_prices: list,
                      features: pd.DataFrame,
                      price_range: Tuple[float, float] = (0.5, 3.0)) -> Dict:
        """
        Find optimal price that maximizes profit.
        
        Args:
            current_price: Current/last observed price
            cost: Cost per liter
            competitor_prices: List of competitor prices
            features: Feature DataFrame
            price_range: (min_price, max_price) for optimization
            
        Returns:
            Dictionary with optimal price, predicted volume, and expected profit
        """
        logger.info("Optimizing price to maximize profit...")
        
        # Objective function (negative profit for minimization)
        def objective(price):
            profit = self.calculate_profit(price, cost, features)
            return -profit  # Negative because we're minimizing
        
        # Optimize
        if SCIPY_AVAILABLE:
            result = minimize_scalar(
                objective,
                bounds=price_range,
                method='bounded'
            )
            optimal_price = result.x
        else:
            # Simple grid search fallback
            prices = np.linspace(price_range[0], price_range[1], 100)
            profits = [-objective(p) for p in prices]
            optimal_price = prices[np.argmax(profits)]
        optimal_volume = self.predict_demand(optimal_price, features)
        optimal_profit = self.calculate_profit(optimal_price, cost, features)
        
        # Apply business rules
        final_price = self.business_rules.apply_all_rules(
            optimal_price,
            current_price,
            cost,
            competitor_prices
        )
        
        final_volume = self.predict_demand(final_price, features)
        final_profit = self.calculate_profit(final_price, cost, features)
        
        return {
            'optimal_price': optimal_price,
            'recommended_price': final_price,
            'predicted_volume': final_volume,
            'expected_profit': final_profit,
            'expected_revenue': final_price * final_volume,
            'profit_margin': (final_price - cost) / cost * 100,
            'price_change_pct': (final_price - current_price) / current_price * 100 if current_price > 0 else 0
        }


if __name__ == "__main__":
    # Example usage would require trained model
    print("PriceOptimizer module - requires trained demand model")

