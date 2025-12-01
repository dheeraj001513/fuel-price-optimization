"""
Data Exploration Script
Analyzes fuel price data to understand demand dynamics, seasonality, and price relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer

def explore_data(data_path: str = "../data/oil_retail_history.csv"):
    """Explore the dataset."""
    
    print("=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    
    # Load data
    pipeline = DataPipeline()
    df = pipeline.process(data_path, data_type='csv')
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total Days: {len(df)}")
    
    # Basic statistics
    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    print(df.describe())
    
    # Missing values
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())
    
    # Price relationships
    print("\n" + "=" * 60)
    print("PRICE RELATIONSHIPS")
    print("=" * 60)
    
    # Correlation with volume
    price_cols = ['price', 'cost', 'comp1_price', 'comp2_price', 'comp3_price']
    correlations = df[price_cols + ['volume']].corr()['volume'].sort_values(ascending=False)
    print("\nCorrelation with Volume:")
    print(correlations)
    
    # Price vs Competitors
    df['avg_comp_price'] = df[['comp1_price', 'comp2_price', 'comp3_price']].mean(axis=1)
    df['price_diff'] = df['price'] - df['avg_comp_price']
    
    print("\nPrice vs Average Competitor Price:")
    print(f"Mean difference: ${df['price_diff'].mean():.3f}")
    print(f"Std difference: ${df['price_diff'].std():.3f}")
    
    # Demand analysis
    print("\n" + "=" * 60)
    print("DEMAND ANALYSIS")
    print("=" * 60)
    
    # Price elasticity (approximate)
    df['price_change'] = df['price'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    elasticity = (df['volume_change'] / df['price_change']).mean()
    print(f"Approximate Price Elasticity: {elasticity:.3f}")
    print("(Negative values indicate demand decreases with price increase)")
    
    # Profit analysis
    df['profit_margin'] = df['price'] - df['cost']
    df['profit'] = df['profit_margin'] * df['volume']
    df['revenue'] = df['price'] * df['volume']
    
    print(f"\nAverage Profit Margin: ${df['profit_margin'].mean():.3f}")
    print(f"Average Daily Profit: ${df['profit'].mean():.2f}")
    print(f"Average Daily Revenue: ${df['revenue'].mean():.2f}")
    
    # Seasonality
    print("\n" + "=" * 60)
    print("SEASONALITY ANALYSIS")
    print("=" * 60)
    
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    
    monthly_avg = df.groupby('month').agg({
        'volume': 'mean',
        'price': 'mean',
        'profit': 'mean'
    })
    
    print("\nMonthly Averages:")
    print(monthly_avg)
    
    weekday_avg = df.groupby('day_of_week').agg({
        'volume': 'mean',
        'price': 'mean'
    })
    
    print("\nDay of Week Averages:")
    print(weekday_avg)
    
    # Visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Price over time
    df['date_dt'] = pd.to_datetime(df['date'])
    axes[0, 0].plot(df['date_dt'], df['price'], label='Price', alpha=0.7)
    axes[0, 0].plot(df['date_dt'], df['avg_comp_price'], label='Avg Competitor', alpha=0.7)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price ($/L)')
    axes[0, 0].set_title('Price Over Time')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Volume over time
    axes[0, 1].plot(df['date_dt'], df['volume'], alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Volume (L)')
    axes[0, 1].set_title('Volume Over Time')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Price vs Volume scatter
    axes[1, 0].scatter(df['price'], df['volume'], alpha=0.5)
    axes[1, 0].set_xlabel('Price ($/L)')
    axes[1, 0].set_ylabel('Volume (L)')
    axes[1, 0].set_title('Price vs Volume')
    
    # Profit over time
    axes[1, 1].plot(df['date_dt'], df['profit'], alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Profit ($)')
    axes[1, 1].set_title('Daily Profit Over Time')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = Path("../output/exploration_plots.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualizations saved to {output_path}")
    
    plt.close()
    
    # Feature engineering preview
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING PREVIEW")
    print("=" * 60)
    
    fe = FeatureEngineer()
    df_features = fe.create_all_features(df)
    
    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {len(df_features.columns)}")
    print(f"\nNew features: {set(df_features.columns) - set(df.columns)}")
    
    return df, df_features


if __name__ == "__main__":
    # Check if data exists
    data_path = Path("../data/oil_retail_history.csv")
    
    if not data_path.exists():
        print("Data file not found. Please generate sample data first:")
        print("python generate_sample_data.py")
    else:
        df, df_features = explore_data(str(data_path))
        print("\nExploration complete!")

