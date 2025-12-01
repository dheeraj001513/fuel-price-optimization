"""
Generate sample historical data for testing if real data is not available.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def generate_historical_data(n_days=730, output_path="data/oil_retail_history.csv"):
    """Generate sample historical fuel price data."""
    
    # Start date (2 years ago)
    start_date = datetime.now() - timedelta(days=n_days)
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    
    # Base prices
    base_price = 1.80
    base_cost = 1.50
    
    # Generate realistic price data with trends and seasonality
    np.random.seed(42)
    
    # Trend component
    trend = np.linspace(0, 0.3, n_days)
    
    # Seasonal component (higher in summer)
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    seasonal = 0.1 * np.sin(2 * np.pi * day_of_year / 365.25)
    
    # Random walk for prices
    price_changes = np.random.normal(0, 0.02, n_days)
    prices = base_price + trend + seasonal + np.cumsum(price_changes)
    prices = np.clip(prices, 1.0, 2.5)  # Keep in reasonable range
    
    # Cost (slightly less volatile)
    cost_changes = np.random.normal(0, 0.015, n_days)
    costs = base_cost + trend * 0.8 + np.cumsum(cost_changes)
    costs = np.clip(costs, 1.0, 2.0)
    
    # Competitor prices (correlated but with variation)
    comp1_prices = prices + np.random.normal(0, 0.05, n_days)
    comp2_prices = prices + np.random.normal(0, 0.05, n_days)
    comp3_prices = prices + np.random.normal(0, 0.05, n_days)
    
    comp1_prices = np.clip(comp1_prices, 1.0, 2.5)
    comp2_prices = np.clip(comp2_prices, 1.0, 2.5)
    comp3_prices = np.clip(comp3_prices, 1.0, 2.5)
    
    # Volume (demand) - inversely related to price, with some randomness
    # Price elasticity: volume decreases as price increases
    avg_comp_price = (comp1_prices + comp2_prices + comp3_prices) / 3
    price_diff = prices - avg_comp_price
    
    # Base volume
    base_volume = 3000
    
    # Volume decreases with price and price difference
    volumes = base_volume - 500 * (prices - base_price) - 300 * price_diff
    volumes += np.random.normal(0, 200, n_days)  # Random variation
    volumes = np.clip(volumes, 500, 5000)  # Keep in reasonable range
    
    # Weekend effect (lower volume)
    day_of_week = np.array([d.weekday() for d in dates])
    volumes[day_of_week >= 5] *= 0.8
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'price': np.round(prices, 3),
        'cost': np.round(costs, 3),
        'comp1_price': np.round(comp1_prices, 3),
        'comp2_price': np.round(comp2_prices, 3),
        'comp3_price': np.round(comp3_prices, 3),
        'volume': np.round(volumes, 2)
    })
    
    # Save
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {n_days} days of historical data")
    print(f"Saved to {output_path}")
    print(f"\nSample data:")
    print(df.head(10))
    print(f"\nStatistics:")
    print(df.describe())
    
    return df


def generate_today_example(output_path="data/today_example.json"):
    """Generate example daily input JSON."""
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    example = {
        "date": today,
        "price": 1.85,
        "cost": 1.52,
        "comp1_price": 1.82,
        "comp2_price": 1.88,
        "comp3_price": 1.79
    }
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(example, f, indent=2)
    
    print(f"\nGenerated today_example.json")
    print(f"Saved to {output_path}")
    print(f"\nContent:")
    print(json.dumps(example, indent=2))
    
    return example


if __name__ == "__main__":
    print("Generating sample data...")
    generate_historical_data()
    generate_today_example()
    print("\nSample data generation complete!")

