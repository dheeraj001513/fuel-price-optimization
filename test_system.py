"""
Test script to verify the Fuel Price Optimization System is working correctly.
"""

import sys
from pathlib import Path
import subprocess

def check_dependencies():
    """Check if all required packages are installed."""
    print("=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 
        'matplotlib', 'seaborn', 'fastapi', 'scipy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'scipy':
                __import__('scipy')
            else:
                __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True

def check_files():
    """Check if required files and directories exist."""
    print("\n" + "=" * 60)
    print("CHECKING FILES AND DIRECTORIES")
    print("=" * 60)
    
    required_files = [
        'main.py',
        'requirements.txt',
        'src/data_pipeline.py',
        'src/feature_engineering.py',
        'src/model.py',
        'src/business_rules.py',
        'src/optimization.py',
        'api/app.py',
        'generate_sample_data.py'
    ]
    
    required_dirs = [
        'data',
        'src',
        'api'
    ]
    
    all_good = True
    
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} MISSING")
            all_good = False
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/ directory")
        else:
            print(f"✗ {dir_path}/ directory MISSING")
            all_good = False
    
    return all_good

def check_data():
    """Check if data files exist."""
    print("\n" + "=" * 60)
    print("CHECKING DATA FILES")
    print("=" * 60)
    
    data_file = Path("data/oil_retail_history.csv")
    example_file = Path("data/today_example.json")
    
    if data_file.exists():
        print(f"✓ Historical data found: {data_file}")
        # Check file size
        size = data_file.stat().st_size / 1024  # KB
        print(f"  File size: {size:.1f} KB")
    else:
        print(f"✗ Historical data NOT found: {data_file}")
        print("  Run: python generate_sample_data.py")
        return False
    
    if example_file.exists():
        print(f"✓ Example JSON found: {example_file}")
    else:
        print(f"✗ Example JSON NOT found: {example_file}")
        print("  Run: python generate_sample_data.py")
        return False
    
    return True

def test_imports():
    """Test if modules can be imported."""
    print("\n" + "=" * 60)
    print("TESTING MODULE IMPORTS")
    print("=" * 60)
    
    sys.path.append(str(Path('src')))
    
    modules = [
        ('data_pipeline', 'DataPipeline'),
        ('feature_engineering', 'FeatureEngineer'),
        ('model', 'DemandModel'),
        ('business_rules', 'BusinessRules'),
        ('optimization', 'PriceOptimizer')
    ]
    
    all_good = True
    for module_name, class_name in modules:
        try:
            mod = __import__(module_name)
            cls = getattr(mod, class_name)
            print(f"✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"✗ {module_name}.{class_name} - Error: {e}")
            all_good = False
    
    return all_good

def test_data_pipeline():
    """Test data pipeline."""
    print("\n" + "=" * 60)
    print("TESTING DATA PIPELINE")
    print("=" * 60)
    
    if not Path("data/oil_retail_history.csv").exists():
        print("⚠ Skipping - data file not found")
        return False
    
    try:
        sys.path.append(str(Path('src')))
        from data_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        df = pipeline.process("data/oil_retail_history.csv", data_type='csv')
        
        print(f"✓ Data loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering."""
    print("\n" + "=" * 60)
    print("TESTING FEATURE ENGINEERING")
    print("=" * 60)
    
    if not Path("data/oil_retail_history.csv").exists():
        print("⚠ Skipping - data file not found")
        return False
    
    try:
        sys.path.append(str(Path('src')))
        from data_pipeline import DataPipeline
        from feature_engineering import FeatureEngineer
        
        pipeline = DataPipeline()
        df = pipeline.process("data/oil_retail_history.csv", data_type='csv')
        
        fe = FeatureEngineer()
        df_features = fe.create_all_features(df)
        
        print(f"✓ Features created: {len(df.columns)} -> {len(df_features.columns)} features")
        print(f"  New features: {len(df_features.columns) - len(df.columns)}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_business_rules():
    """Test business rules."""
    print("\n" + "=" * 60)
    print("TESTING BUSINESS RULES")
    print("=" * 60)
    
    try:
        sys.path.append(str(Path('src')))
        from business_rules import BusinessRules
        
        rules = BusinessRules()
        
        # Test scenario
        recommended = 1.85
        current = 1.80
        cost = 1.50
        competitors = [1.75, 1.82, 1.78]
        
        final_price = rules.apply_all_rules(recommended, current, cost, competitors)
        
        print(f"✓ Business rules working")
        print(f"  Input: ${recommended:.3f}, Output: ${final_price:.3f}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FUEL PRICE OPTIMIZATION SYSTEM - SYSTEM CHECK")
    print("=" * 60)
    
    results = []
    
    # Check dependencies
    results.append(("Dependencies", check_dependencies()))
    
    # Check files
    results.append(("Files", check_files()))
    
    # Check data
    results.append(("Data Files", check_data()))
    
    # Test imports
    results.append(("Module Imports", test_imports()))
    
    # Test data pipeline
    results.append(("Data Pipeline", test_data_pipeline()))
    
    # Test feature engineering
    results.append(("Feature Engineering", test_feature_engineering()))
    
    # Test business rules
    results.append(("Business Rules", test_business_rules()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All checks passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Train model: python main.py --train")
        print("2. Get recommendation: python main.py --predict data/today_example.json")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Generate sample data: python generate_sample_data.py")

if __name__ == "__main__":
    main()

