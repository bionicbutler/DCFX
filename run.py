"""
Master Pipeline Runner
Runs the complete DCF data pipeline for a specified ticker
Updated to include enhanced Script 1 and new Script 4
"""

import sys
import os
import argparse
from datetime import datetime
import importlib.util
import pandas as pd
import numpy as np

# Add the scripts directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(current_dir, 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Import modules using importlib to handle custom file names
def import_module_from_file(module_name, file_path):
    """Import a module from a specific file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        raise ImportError(f"Could not load {file_path}")

# Import all scripts
script1_path = os.path.join(scripts_dir, "01_data_collectcomplete.py")
data_collector_module = import_module_from_file("data_collector", script1_path)
DataCollector = data_collector_module.DataCollector

script2_path = os.path.join(scripts_dir, "02_cleaning.py")
cleaner_module = import_module_from_file("cleaning", script2_path)
DataCleaner = cleaner_module.DataCleaner

script3_path = os.path.join(scripts_dir, "03_dcf_feature.py")
features_module = import_module_from_file("feature", script3_path)
FeatureEngineer = features_module.FeatureEngineer

script4_path = os.path.join(scripts_dir, "04_dcfcalc.py")
dcf_module = import_module_from_file("dcf_calc", script4_path)
DCFCalculator = dcf_module.DCFCalculator
DCFInputs = dcf_module.DCFInputs


def run_dcf_example(ticker: str, dcf_data: pd.DataFrame):
    """
    Run example DCF calculation with collected data
    
    Args:
        ticker: Stock ticker
        dcf_data: DCF-ready dataset from Script 1
    """
    print("\nStep 4a: Running Example DCF Calculation...")
    print("-" * 40)
    
    try:
        # Get latest available data
        latest_data = dcf_data.iloc[-1]
        
        # Extract DCF inputs from collected data
        # Base values
        base_revenue = latest_data['Revenue'] * 4 if 'Revenue' in latest_data else 1000  # Annualize
        shares_outstanding = latest_data['latest_shares_outstanding'] if 'latest_shares_outstanding' in latest_data else 100e6
        
        # Beta
        beta = latest_data['latest_beta'] if 'latest_beta' in latest_data else 1.0
        
        # Tax rate - get most recent non-null value
        if 'effective_tax_rate' in dcf_data.columns:
            tax_rate_series = dcf_data['effective_tax_rate'].dropna()
            tax_rate = tax_rate_series.iloc[-1] if len(tax_rate_series) > 0 else 0.21
        else:
            tax_rate = 0.21  # Default corporate tax rate
        
        # Cost of debt
        if 'implied_cost_of_debt' in dcf_data.columns:
            cod_series = dcf_data['implied_cost_of_debt'].dropna()
            cost_of_debt = cod_series.iloc[-1] if len(cod_series) > 0 else 0.04
        else:
            cost_of_debt = 0.04  # Default 4%
        
        # Working capital percentage
        if 'wc_pct_revenue' in dcf_data.columns:
            wc_series = dcf_data['wc_pct_revenue'].dropna()
            wc_pct = wc_series.iloc[-1] if len(wc_series) > 0 else 0.05
        else:
            wc_pct = 0.05  # Default 5%
        
        # Debt to equity
        debt_to_equity = latest_data['latest_debt_to_equity'] if 'latest_debt_to_equity' in latest_data else 0.3
        
        # Historical metrics for assumptions
        # Calculate historical revenue growth
        if 'revenue_growth_1y' in dcf_data.columns:
            growth_series = dcf_data['revenue_growth_1y'].dropna()
            hist_growth = growth_series.mean() if len(growth_series) > 0 else 0.10
        else:
            hist_growth = 0.10
        
        # Calculate historical EBIT margin
        if 'historical_ebit_margin' in dcf_data.columns:
            margin_series = dcf_data['historical_ebit_margin'].dropna()
            hist_margin = margin_series.mean() if len(margin_series) > 0 else 0.15
        else:
            hist_margin = 0.15
        
        # Calculate historical CapEx intensity
        if 'historical_capex_intensity' in dcf_data.columns:
            capex_series = dcf_data['historical_capex_intensity'].dropna()
            capex_intensity = capex_series.mean() if len(capex_series) > 0 else 0.05
        else:
            capex_intensity = 0.05
        
        # Get current market data
        current_price = latest_data['latest_current_price'] if 'latest_current_price' in latest_data else None
        
        # Create example DCF inputs (these would normally come from ML predictions)
        dcf_inputs = DCFInputs(
            # Revenue growth - declining from historical rate
            revenue_growth_y1=min(0.30, max(-0.20, hist_growth)),
            revenue_growth_y2=min(0.25, max(-0.15, hist_growth * 0.9)),
            revenue_growth_y3=min(0.20, max(-0.10, hist_growth * 0.8)),
            revenue_growth_y4=min(0.15, max(-0.05, hist_growth * 0.7)),
            revenue_growth_y5=min(0.10, max(0, hist_growth * 0.6)),
            
            # EBIT margins - gradual improvement
            ebit_margin_y1=hist_margin,
            ebit_margin_y2=hist_margin * 1.02,
            ebit_margin_y3=hist_margin * 1.04,
            ebit_margin_y4=hist_margin * 1.05,
            ebit_margin_y5=hist_margin * 1.05,
            
            # Other assumptions
            capex_pct_revenue=capex_intensity,
            tax_rate=tax_rate,
            working_capital_pct_revenue=wc_pct,
            terminal_growth_rate=0.025,  # 2.5% terminal growth
            
            # WACC components
            risk_free_rate=0.04,  # Could get from macro data
            equity_risk_premium=0.065,  # Market assumption
            beta=beta,
            cost_of_debt=cost_of_debt,
            debt_to_equity=debt_to_equity,
            
            # Base values
            base_revenue=base_revenue,
            base_shares_outstanding=shares_outstanding
        )
        
        # Initialize DCF calculator
        calculator = DCFCalculator(ticker)
        
        # Run DCF calculation
        dcf_result = calculator.calculate_dcf_value(dcf_inputs)
        
        print(f"✓ DCF calculation complete!")
        print(f"\nDCF Results:")
        print(f"  Fair Value per Share: ${dcf_result['equity_value_per_share']:.2f}")
        print(f"  Enterprise Value: ${dcf_result['enterprise_value']:,.0f}")
        print(f"  WACC: {dcf_result['wacc']:.2%}")
        print(f"  Terminal Value %: {dcf_result['terminal_value_pct']:.1%}")
        
        if current_price:
            diff_pct = (dcf_result['equity_value_per_share'] - current_price) / current_price
            print(f"\n  Current Price: ${current_price:.2f}")
            print(f"  Implied Return: {diff_pct:.1%}")
            print(f"  Valuation: {'Undervalued' if diff_pct > 0 else 'Overvalued'}")
        
        # Run sensitivity analysis
        print("\n  Running quick sensitivity analysis...")
        sensitivity_ranges = {
            'terminal_growth_rate': (-0.20, 0.20),
            'revenue_growth_y1': (-0.30, 0.30),
            'beta': (-0.20, 0.20)
        }
        
        # Create tornado chart
        fig = calculator.create_tornado_chart(dcf_inputs, sensitivity_ranges)
        tornado_path = os.path.join(calculator.reports_dir, f"tornado_chart_{ticker}_pipeline.png")
        fig.savefig(tornado_path, dpi=300, bbox_inches='tight')
        print(f"  Sensitivity chart saved to: {tornado_path}")
        
        # Generate report
        calculator.create_dcf_report(dcf_result)
        
        return True
        
    except Exception as e:
        print(f"✗ Error in DCF calculation: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pipeline(ticker: str, years_back: int = 10, skip_collection: bool = False, 
                 skip_dcf: bool = False):
    """
    Run the complete pipeline for a given ticker
    
    Args:
        ticker: Stock ticker symbol
        years_back: Years of historical data to collect
        skip_collection: Skip data collection if data already exists
        skip_dcf: Skip DCF calculation example
    """
    
    print(f"\n{'='*60}")
    print(f"Starting Enhanced DCF Data Pipeline for {ticker}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Step 1: Enhanced Data Collection
    if not skip_collection:
        print("Step 1: Collecting Data (Enhanced with DCF inputs)...")
        print("-" * 40)
        try:
            collector = DataCollector(ticker, years_back=years_back)
            data = collector.collect_all_data()
            
            # Create DCF-ready dataset
            dcf_dataset = collector.create_dcf_ready_dataset()
            
            print("✓ Enhanced data collection complete!")
            
            # Print detailed summary
            for key, df in data.items():
                if df is not None and not df.empty:
                    print(f"  - {key}: {df.shape[0]} records")
                    if key == 'dcf_additional':
                        # Show DCF-specific values
                        for col in df.columns:
                            if not df[col].isna().all():
                                value = df[col].iloc[0]
                                if isinstance(value, float):
                                    print(f"    • {col}: {value:.2f}")
                                else:
                                    print(f"    • {col}: {value}")
            
            print(f"\n  DCF-ready dataset: {dcf_dataset.shape}")
            
        except Exception as e:
            print(f"✗ Error in data collection: {e}")
            return False
    else:
        print("Skipping data collection (using existing data)")
        # Load existing DCF dataset
        dcf_path = os.path.join("dcf_data", ticker, "dcf_ready_dataset.csv")
        if os.path.exists(dcf_path):
            dcf_dataset = pd.read_csv(dcf_path, index_col=0, parse_dates=True)
        else:
            print("✗ No existing DCF dataset found. Run without --skip-collection first.")
            return False
    
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
    
    # Step 4: DCF Calculation Example
    if not skip_dcf:
        success = run_dcf_example(ticker, dcf_dataset)
        if not success:
            print("⚠ DCF example failed, but pipeline continued")
    else:
        print("\nSkipping DCF calculation example")
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed successfully for {ticker}!")
    print(f"All data saved to: dcf_data/{ticker}/")
    print(f"\nNext steps:")
    print(f"  1. Run ML models (Scripts 6-7) to predict DCF inputs")
    print(f"  2. Run Monte Carlo simulation (Scripts 8-9)")
    print(f"  3. Generate final analysis (Scripts 10-11)")
    print(f"{'='*60}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run enhanced DCF data pipeline with integrated DCF calculator'
    )
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., CMG, NVDA)')
    parser.add_argument('--years', type=int, default=10, 
                       help='Years of historical data (default: 10)')
    parser.add_argument('--skip-collection', action='store_true', 
                       help='Skip data collection and use existing data')
    parser.add_argument('--skip-dcf', action='store_true',
                       help='Skip DCF calculation example')
    
    args = parser.parse_args()
    
    # Run the pipeline
    success = run_pipeline(
        args.ticker.upper(), 
        args.years, 
        args.skip_collection,
        args.skip_dcf
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    # If no command line arguments, prompt for ticker
    if len(sys.argv) == 1:
        print("\n=== Enhanced DCF Data Pipeline Runner ===")
        print("Now includes DCF-specific data collection and example DCF calculation\n")
        
        ticker = input("Enter ticker symbol: ").upper()
        years = input("Years of data (default 10): ")
        years = int(years) if years else 10
        
        skip_collection = input("Skip data collection? (y/N): ").lower() == 'y'
        skip_dcf = input("Skip DCF example? (y/N): ").lower() == 'y'
        
        run_pipeline(ticker, years, skip_collection, skip_dcf)
    else:
        main()