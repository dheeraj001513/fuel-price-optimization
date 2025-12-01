"""
FastAPI endpoint for Fuel Price Optimization System
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path
import json

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from model import DemandModel
from business_rules import BusinessRules
from optimization import PriceOptimizer
import pandas as pd
import numpy as np

app = FastAPI(
    title="Fuel Price Optimization API",
    description="API for recommending optimal daily fuel prices",
    version="1.0.0"
)

# Global variables for loaded model
model = None
feature_engineer = None
business_rules = None
historical_data = None


class DailyInput(BaseModel):
    """Daily input schema."""
    date: str
    price: float
    cost: float
    comp1_price: float
    comp2_price: float
    comp3_price: float


class PriceRecommendation(BaseModel):
    """Price recommendation response schema."""
    recommended_price: float
    predicted_volume: float
    expected_profit: float
    expected_revenue: float
    profit_margin: float
    price_change_pct: float
    optimal_price: Optional[float] = None


@app.on_event("startup")
async def load_model():
    """Load model and data on startup."""
    global model, feature_engineer, business_rules, historical_data
    
    model_path = Path("models/demand_model.pkl")
    data_path = Path("data/oil_retail_history.csv")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Historical data not found at {data_path}")
    
    # Load model
    model = DemandModel()
    model.load(str(model_path))
    
    # Load historical data
    pipeline = DataPipeline()
    historical_data = pipeline.process(str(data_path), data_type='csv')
    
    # Initialize components
    feature_engineer = FeatureEngineer()
    business_rules = BusinessRules()
    
    print("Model and data loaded successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Fuel Price Optimization API",
        "version": "1.0.0",
        "endpoints": {
            "/recommend": "POST - Get price recommendation",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None and model.is_trained,
        "data_loaded": historical_data is not None
    }


@app.post("/recommend", response_model=PriceRecommendation)
async def recommend_price(input_data: DailyInput):
    """
    Get price recommendation for given daily input.
    
    Args:
        input_data: Daily input with current price, cost, and competitor prices
        
    Returns:
        Price recommendation with expected volume and profit
    """
    if model is None or not model.is_trained:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if historical_data is None:
        raise HTTPException(status_code=500, detail="Historical data not loaded")
    
    try:
        # Convert input to dict
        daily_dict = input_data.dict()
        
        # Combine historical + today for feature computation
        pipeline = DataPipeline()
        df_today = pipeline.transform_daily_data(daily_dict)
        df_combined = pd.concat([historical_data, df_today], ignore_index=True)
        df_features = feature_engineer.create_all_features(df_combined)
        
        # Get features for today (last row)
        today_features = df_features.iloc[[-1]].copy()
        
        # Price optimizer
        optimizer = PriceOptimizer(model, business_rules, feature_engineer)
        
        # Extract inputs
        current_price = daily_dict['price']
        cost = daily_dict['cost']
        competitor_prices = [
            daily_dict['comp1_price'],
            daily_dict['comp2_price'],
            daily_dict['comp3_price']
        ]
        
        # Optimize
        result = optimizer.optimize_price(
            current_price=current_price,
            cost=cost,
            competitor_prices=competitor_prices,
            features=today_features
        )
        
        return PriceRecommendation(
            recommended_price=result['recommended_price'],
            predicted_volume=result['predicted_volume'],
            expected_profit=result['expected_profit'],
            expected_revenue=result['expected_revenue'],
            profit_margin=result['profit_margin'],
            price_change_pct=result['price_change_pct'],
            optimal_price=result.get('optimal_price')
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

