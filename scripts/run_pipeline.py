"""
Master Pipeline Runner
Runs the complete DCF data pipeline for a specified ticker
"""

import sys
import argparse
import os
from datetime import datetime

# Import your modules with actual file names
# Note: Python import uses the file name without .py extension
# If files are named with dots, we need to handle differently
try:
    # Try importing with underscores (Python-friendly names)
    from data_collector import DataCollector
    from cleaning import DataCleaner
    from dcf_feature import FeatureEngineer
except ImportError:
    # If that doesn't work, use importlib for files with dots
    import importlib.util
    
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load Script 1
    script1_path = os.path.join(current_dir, "01_data_collector.py")
    spec = importlib.util.spec_from_file_location("data_collector", script1_path)
    if spec is not None and spec.loader is not None:
        data_collector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_collector_module)
        DataCollector = data_collector_module.DataCollector
    else:
        raise ImportError("Could not load 01_data_collector.py")
    
    # Load Script 2
    script2_path = os.path.join(current_dir, "02_cleaning.py")
    spec = importlib.util.spec_from_file_location("cleaning", script2_path)
    if spec is not None and spec.loader is not None:
        cleaner_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cleaner_module)
        DataCleaner = cleaner_module.DataCleaner
    else:
        raise ImportError("Could not load 02_cleaning   .py")
    
    # Load Script 3
    script3_path = os.path.join(current_dir, "03_dcf_feature.py")
    spec = importlib.util.spec_from_file_location("dcf_feature", script3_path)
    if spec is not None and spec.loader is not None:
        features_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(features_module)
        FeatureEngineer = features_module.FeatureEngineer
    else:
        raise ImportError("Could not load 03_dcf_feature.py")


def run_pipeline(ticker: str, years_back: int = 10, skip_collection: bool = False):
    """
    Run the complete pipeline for a given ticker
    
    Args:
        ticker: Stock ticker symbol
        years_back: Years of historical data to collect
        skip_collection: Skip data collection if data already exists
    """
    
    print(f"\n{'='*60}")
    print(f"Starting DCF Data Pipeline for {ticker}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Step 1: Data Collection
    if not skip_collection:
        print("Step 1: Collecting Data...")
        print("-" * 40)
        try:
            collector = DataCollector(ticker, years_back=years_back)
            data = collector.collect_all_data()
            print("✓ Data collection complete!")
            
            # Print summary
            for key, df in data.items():
                if df is not None and not df.empty:
                    print(f"  - {key}: {df.shape[0]} records")
        except Exception as e:
            print(f"✗ Error in data collection: {e}")
            return False
    else:
        print("Skipping data collection (using existing data)")
    
    # Step 2: Data Cleaning
    print("\nStep 2: Cleaning Data...")
    print("-" * 40)
    try:
        cleaner = DataCleaner(ticker)
        cleaned_data = cleaner.clean_all_data()
        print("✓ Data cleaning complete!")
        
        # Print summary
        for key, df in cleaned_data.items():
            if not df.empty:
                print(f"  - {key}: {df.shape[0]} records after cleaning")
    except Exception as e:
        print(f"✗ Error in data cleaning: {e}")
        return False
    
    # Step 3: Feature Engineering
    print("\nStep 3: Engineering Features...")
    print("-" * 40)
    try:
        engineer = FeatureEngineer(ticker)
        features, targets = engineer.engineer_all_features()
        print("✓ Feature engineering complete!")
        print(f"  - Features created: {features.shape[1]}")
        print(f"  - Target variables: {targets.shape[1]}")
        print(f"  - Total observations: {features.shape[0]}")
    except Exception as e:
        print(f"✗ Error in feature engineering: {e}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed successfully for {ticker}!")
    print(f"All data saved to: dcf_data/{ticker}/")
    print(f"{'='*60}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Run DCF data pipeline for a stock ticker')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., CMG, NVDA)')
    parser.add_argument('--years', type=int, default=10, help='Years of historical data (default: 10)')
    parser.add_argument('--skip-collection', action='store_true', 
                       help='Skip data collection and use existing data')
    
    args = parser.parse_args()
    
    # Run the pipeline
    success = run_pipeline(args.ticker.upper(), args.years, args.skip_collection)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    # If no command line arguments, prompt for ticker
    if len(sys.argv) == 1:
        print("\n=== DCF Data Pipeline Runner ===")
        ticker = input("Enter ticker symbol: ").upper()
        years = input("Years of data (default 10): ")
        years = int(years) if years else 10
        
        skip = input("Skip data collection? (y/N): ").lower()
        skip_collection = skip == 'y'
        
        run_pipeline(ticker, years, skip_collection)
    else:
        main()