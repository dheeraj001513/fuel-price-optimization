"""
Business Rules and Constraints Module
Implements business guardrails for price optimization.
"""

from typing import Dict, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BusinessRules:
    """Apply business constraints to price recommendations."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize business rules.
        
        Args:
            config: Configuration dictionary with business rules
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        """Default business rules configuration."""
        return {
            'max_daily_price_change_pct': 5.0,  # Maximum 5% price change per day
            'min_profit_margin_pct': 2.0,       # Minimum 2% profit margin
            'max_price_vs_competitors_pct': 10.0,  # Max 10% above average competitor
            'min_price_vs_competitors_pct': -5.0,  # Min 5% below average competitor
            'min_price': 0.5,
            'max_price': 3.0
        }
    
    def apply_max_price_change(self, recommended_price: float, 
                               current_price: float) -> float:
        """
        Limit daily price change to maximum allowed percentage.
        
        Args:
            recommended_price: Recommended price from model
            current_price: Current/last observed price
            
        Returns:
            Adjusted price respecting max change constraint
        """
        max_change = self.config['max_daily_price_change_pct'] / 100.0
        
        if current_price == 0:
            return recommended_price
        
        price_change = (recommended_price - current_price) / current_price
        
        if abs(price_change) > max_change:
            if price_change > 0:
                adjusted_price = current_price * (1 + max_change)
            else:
                adjusted_price = current_price * (1 - max_change)
            
            logger.info(f"Price change limited: {price_change*100:.2f}% -> {max_change*100:.2f}%")
            return adjusted_price
        
        return recommended_price
    
    def apply_min_profit_margin(self, recommended_price: float, 
                               cost: float) -> float:
        """
        Ensure minimum profit margin.
        
        Args:
            recommended_price: Recommended price
            cost: Cost per liter
            
        Returns:
            Adjusted price ensuring minimum margin
        """
        min_margin_pct = self.config['min_profit_margin_pct'] / 100.0
        min_price = cost * (1 + min_margin_pct)
        
        if recommended_price < min_price:
            logger.info(f"Price adjusted to meet minimum margin: {recommended_price:.3f} -> {min_price:.3f}")
            return min_price
        
        return recommended_price
    
    def apply_competitor_alignment(self, recommended_price: float,
                                  competitor_prices: list) -> float:
        """
        Ensure price stays within competitive range.
        
        Args:
            recommended_price: Recommended price
            competitor_prices: List of competitor prices
            
        Returns:
            Adjusted price within competitive range
        """
        if not competitor_prices or len(competitor_prices) == 0:
            return recommended_price
        
        avg_comp_price = np.mean(competitor_prices)
        max_above = self.config['max_price_vs_competitors_pct'] / 100.0
        min_below = self.config['min_price_vs_competitors_pct'] / 100.0
        
        max_price = avg_comp_price * (1 + max_above)
        min_price = avg_comp_price * (1 + min_below)
        
        if recommended_price > max_price:
            logger.info(f"Price adjusted to stay competitive (above max): {recommended_price:.3f} -> {max_price:.3f}")
            return max_price
        elif recommended_price < min_price:
            logger.info(f"Price adjusted to stay competitive (below min): {recommended_price:.3f} -> {min_price:.3f}")
            return min_price
        
        return recommended_price
    
    def apply_price_bounds(self, recommended_price: float) -> float:
        """
        Apply absolute price bounds.
        
        Args:
            recommended_price: Recommended price
            
        Returns:
            Price within absolute bounds
        """
        min_price = self.config['min_price']
        max_price = self.config['max_price']
        
        if recommended_price < min_price:
            logger.info(f"Price adjusted to minimum: {recommended_price:.3f} -> {min_price:.3f}")
            return min_price
        elif recommended_price > max_price:
            logger.info(f"Price adjusted to maximum: {recommended_price:.3f} -> {max_price:.3f}")
            return max_price
        
        return recommended_price
    
    def apply_all_rules(self, recommended_price: float,
                       current_price: float,
                       cost: float,
                       competitor_prices: list) -> float:
        """
        Apply all business rules in sequence.
        
        Args:
            recommended_price: Initial recommended price from model
            current_price: Current/last observed price
            cost: Cost per liter
            competitor_prices: List of competitor prices
            
        Returns:
            Final price after applying all constraints
        """
        price = recommended_price
        
        # Apply rules in order
        price = self.apply_price_bounds(price)
        price = self.apply_min_profit_margin(price, cost)
        price = self.apply_competitor_alignment(price, competitor_prices)
        price = self.apply_max_price_change(price, current_price)
        
        # Final bounds check
        price = self.apply_price_bounds(price)
        
        return price


if __name__ == "__main__":
    # Example usage
    rules = BusinessRules()
    
    # Test scenario
    recommended = 1.85
    current = 1.80
    cost = 1.50
    competitors = [1.75, 1.82, 1.78]
    
    final_price = rules.apply_all_rules(recommended, current, cost, competitors)
    print(f"Recommended: {recommended:.3f}")
    print(f"Final (after rules): {final_price:.3f}")
    print(f"Profit margin: {(final_price - cost) / cost * 100:.2f}%")

