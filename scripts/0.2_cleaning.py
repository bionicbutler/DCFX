"""
Script 2: Data Cleaning & Validation
Cleans and validates financial data for DCF analysis
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Cleans and validates financial data for DCF analysis"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.base_dir = "dcf_data"
        self.ticker_dir = os.path.join(self.base_dir, self.ticker)
        self.raw_dir = os.path.join(self.ticker_dir, "raw")
        self.clean_dir = os.path.join(self.ticker_dir, "clean")
        self.reports_dir = os.path.join(self.ticker_dir, "quality_reports")
        self._create_directories()
        
        # Data quality thresholds
        self.outlier_threshold = 3  # Z-score threshold
        self.max_missing_pct = 0.3  # Maximum 30% missing data
        self.min_quarters = 12      # Minimum quarters needed
        
        # Financial data validation rules
        self.validation_rules = {
            'Revenue': {'min': 0, 'max_growth': 5.0},  # Max 500% QoQ growth
            'EBIT': {'min': None, 'max_growth': 10.0},  # Can be negative
            'CapEx': {'min': 0, 'max_pct_revenue': 1.0},  # Max 100% of revenue
            'TotalAssets': {'min': 0, 'max_growth': 3.0},
            'TotalDebt': {'min': 0, 'max_growth': 5.0},
            'Price': {'min': 0.01, 'max_growth': 2.0},
            'MarketCap': {'min': 1e6, 'max': 5e12}  # $1M to $5T
        }
        
    def _create_directories(self):
        """Create folder structure"""
        for directory in [self.clean_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw data from data collector output"""
        data = {}
        
        # Load fundamental data
        fundamental_path = os.path.join(self.raw_dir, f"combined_fundamentals_{self.ticker}.csv")
        if os.path.exists(fundamental_path):
            data['fundamental'] = pd.read_csv(fundamental_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded fundamental data: {data['fundamental'].shape}")
        else:
            logger.warning("No fundamental data found")
            data['fundamental'] = pd.DataFrame()
        
        # Load macro data
        macro_path = os.path.join(self.raw_dir, "macro_data.csv")
        if os.path.exists(macro_path):
            data['macro'] = pd.read_csv(macro_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded macro data: {data['macro'].shape}")
        else:
            logger.warning("No macro data found")
            data['macro'] = pd.DataFrame()
        
        # Load price data
        price_path = os.path.join(self.raw_dir, f"price_data_{self.ticker}.csv")
        if os.path.exists(price_path):
            data['price'] = pd.read_csv(price_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded price data: {data['price'].shape}")
        else:
            logger.warning("No price data found")
            data['price'] = pd.DataFrame()
        
        return data
    
    def validate_data_quality(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Validate data quality and flag issues"""
        quality_report = []
        
        for col in df.columns:
            col_report = {
                'column': col,
                'data_type': data_type,
                'total_records': len(df),
                'missing_count': df[col].isna().sum(),
                'missing_pct': df[col].isna().sum() / len(df),
                'zeros_count': (df[col] == 0).sum() if pd.api.types.is_numeric_dtype(df[col]) else 0,
                'negative_count': (df[col] < 0).sum() if pd.api.types.is_numeric_dtype(df[col]) else 0,
                'outliers_count': 0,
                'validation_errors': []
            }
            
            if pd.api.types.is_numeric_dtype(df[col]) and col_report['missing_pct'] < 1.0:
                # Check for outliers using Z-score
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                col_report['outliers_count'] = (z_scores > self.outlier_threshold).sum()
                
                # Validate against rules
                if col in self.validation_rules:
                    rules = self.validation_rules[col]
                    
                    # Check minimum values
                    if rules.get('min') is not None:
                        invalid_count = (df[col] < rules['min']).sum()
                        if invalid_count > 0:
                            col_report['validation_errors'].append(
                                f"{invalid_count} values below minimum {rules['min']}"
                            )
                    
                    # Check maximum values
                    if rules.get('max') is not None:
                        invalid_count = (df[col] > rules['max']).sum()
                        if invalid_count > 0:
                            col_report['validation_errors'].append(
                                f"{invalid_count} values above maximum {rules['max']}"
                            )
                    
                    # Check growth rates
                    if rules.get('max_growth') is not None:
                        growth = df[col].pct_change()
                        extreme_growth = (growth.abs() > rules['max_growth']).sum()
                        if extreme_growth > 0:
                            col_report['validation_errors'].append(
                                f"{extreme_growth} extreme growth rates (>{rules['max_growth']*100}%)"
                            )
            
            quality_report.append(col_report)
        
        return pd.DataFrame(quality_report)
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'smart') -> pd.DataFrame:
        """Handle missing values with appropriate methods"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            missing_count = df_clean[col].isna().sum()
            if missing_count == 0:
                continue
            
            missing_pct = missing_count / len(df_clean)
            logger.info(f"Handling {missing_count} ({missing_pct:.1%}) missing values in {col}")
            
            if missing_pct > self.max_missing_pct:
                logger.warning(f"Too many missing values in {col} ({missing_pct:.1%})")
                if method == 'drop':
                    df_clean = df_clean.drop(columns=[col])
                    continue
            
            if method == 'smart':
                # For financial metrics, use forward fill then interpolation
                if col in ['Revenue', 'EBIT', 'TotalAssets', 'TotalDebt', 'CapEx']:
                    # First forward fill (assume previous quarter value holds)
                    df_clean[col] = df_clean[col].fillna(method='ffill', limit=1)
                    # Then interpolate remaining gaps
                    df_clean[col] = df_clean[col].interpolate(method='linear', limit=2)
                
                # For rates and indices, use interpolation
                elif col in ['10Y_Treasury', 'VIX', 'Sector_Index', 'GDP_Growth']:
                    df_clean[col] = df_clean[col].interpolate(method='linear')
                
                # For price data, use forward fill
                elif col in ['Price', 'Volume', 'MarketCap']:
                    df_clean[col] = df_clean[col].fillna(method='ffill')
                
                else:
                    # Default: interpolation
                    df_clean[col] = df_clean[col].interpolate(method='linear')
            
            elif method == 'forward_fill':
                df_clean[col] = df_clean[col].fillna(method='ffill')
            
            elif method == 'interpolate':
                df_clean[col] = df_clean[col].interpolate(method='linear')
            
            elif method == 'mean':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            
            # Log remaining missing values
            remaining_missing = df_clean[col].isna().sum()
            if remaining_missing > 0:
                logger.warning(f"Still {remaining_missing} missing values in {col} after cleaning")
        
        return df_clean
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'cap') -> pd.DataFrame:
        """Handle outliers in the data"""
        df_clean = df.copy()
        outlier_report = []
        
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            # Skip if too many missing values
            if df_clean[col].isna().sum() / len(df_clean) > 0.5:
                continue
            
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            outliers = z_scores > self.outlier_threshold
            outlier_indices = df_clean[col].dropna().index[outliers]
            
            if len(outlier_indices) > 0:
                logger.info(f"Found {len(outlier_indices)} outliers in {col}")
                
                if method == 'cap':
                    # Cap at 99th and 1st percentile
                    lower = df_clean[col].quantile(0.01)
                    upper = df_clean[col].quantile(0.99)
                    df_clean.loc[df_clean[col] < lower, col] = lower
                    df_clean.loc[df_clean[col] > upper, col] = upper
                
                elif method == 'remove':
                    df_clean.loc[outlier_indices, col] = np.nan
                
                elif method == 'winsorize':
                    # Winsorize at 5% and 95%
                    df_clean[col] = stats.mstats.winsorize(df_clean[col], limits=[0.05, 0.05])
                
                outlier_report.append({
                    'column': col,
                    'outliers_found': len(outlier_indices),
                    'method_applied': method,
                    'outlier_values': df[col].loc[outlier_indices].tolist()[:5]  # Sample
                })
        
        return df_clean, outlier_report
    
    def align_frequencies(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align all data to quarterly frequency"""
        aligned_data = {}
        
        # Get the date range from fundamental data
        if not data['fundamental'].empty:
            start_date = data['fundamental'].index.min()
            end_date = data['fundamental'].index.max()
            
            # Create quarterly date range
            quarterly_dates = pd.date_range(
                start=start_date, 
                end=end_date, 
                freq='QE'
            )
            
            # Align each dataset
            for key, df in data.items():
                if df.empty:
                    aligned_data[key] = df
                    continue
                
                # Resample to quarterly if needed
                if key == 'macro':
                    # Macro data might be at different frequencies
                    df_quarterly = pd.DataFrame(index=quarterly_dates)
                    
                    for col in df.columns:
                        # Use appropriate method for each macro variable
                        if col in ['10Y_Treasury', 'VIX']:
                            # Use end of quarter value for rates
                            resampled = df[col].resample('QE').last()
                        elif col in ['GDP_Growth']:
                            # Use mean for growth rates
                            resampled = df[col].resample('QE').mean()
                        else:
                            # Default to mean
                            resampled = df[col].resample('QE').mean()
                        
                        # Align to our quarterly dates
                        df_quarterly[col] = resampled.reindex(quarterly_dates, method='nearest')
                    
                    aligned_data[key] = df_quarterly
                
                else:
                    # For fundamental and price data, reindex to quarterly dates
                    aligned_data[key] = df.reindex(quarterly_dates, method='nearest')
                
                logger.info(f"Aligned {key} data to {len(aligned_data[key])} quarterly periods")
        
        else:
            logger.error("No fundamental data to determine date range")
            aligned_data = data
        
        return aligned_data
    
    def create_rolling_averages(self, df: pd.DataFrame, windows: List[int] = [4, 8]) -> pd.DataFrame:
        """Create rolling averages for noisy data"""
        df_smooth = df.copy()
        
        # Define which columns need smoothing
        smooth_cols = {
            'VIX': [4],  # 1-year rolling average
            'GDP_Growth': [4, 8],  # 1 and 2-year
            '10Y_Treasury': [2, 4],  # 6-month and 1-year
            'Sector_Index': [4]  # 1-year
        }
        
        for col in df.columns:
            if col in smooth_cols:
                for window in smooth_cols[col]:
                    new_col = f"{col}_MA{window}Q"
                    df_smooth[new_col] = df[col].rolling(window=window, min_periods=1).mean()
                    logger.info(f"Created {new_col} rolling average")
        
        return df_smooth
    
    def validate_financial_relationships(self, df: pd.DataFrame) -> List[Dict]:
        """Validate logical relationships in financial data"""
        issues = []
        
        # Check if CapEx exceeds Revenue (unlikely)
        if 'CapEx' in df.columns and 'Revenue' in df.columns:
            capex_pct = df['CapEx'] / df['Revenue']
            high_capex = capex_pct > 0.5  # CapEx > 50% of revenue
            if high_capex.any():
                issues.append({
                    'type': 'High CapEx/Revenue',
                    'periods': df.index[high_capex].tolist(),
                    'values': capex_pct[high_capex].tolist()
                })
        
        # Check if TotalDebt exceeds TotalAssets (concerning)
        if 'TotalDebt' in df.columns and 'TotalAssets' in df.columns:
            debt_ratio = df['TotalDebt'] / df['TotalAssets']
            high_debt = debt_ratio > 0.9  # Debt > 90% of assets
            if high_debt.any():
                issues.append({
                    'type': 'High Debt/Assets',
                    'periods': df.index[high_debt].tolist(),
                    'values': debt_ratio[high_debt].tolist()
                })
        
        # Check for negative revenue (should not happen)
        if 'Revenue' in df.columns:
            neg_revenue = df['Revenue'] < 0
            if neg_revenue.any():
                issues.append({
                    'type': 'Negative Revenue',
                    'periods': df.index[neg_revenue].tolist(),
                    'values': df.loc[neg_revenue, 'Revenue'].tolist()
                })
        
        # Check EBIT margin reasonability
        if 'EBIT' in df.columns and 'Revenue' in df.columns:
            ebit_margin = df['EBIT'] / df['Revenue']
            extreme_margin = (ebit_margin < -1) | (ebit_margin > 1)  # Beyond -100% to 100%
            if extreme_margin.any():
                issues.append({
                    'type': 'Extreme EBIT Margin',
                    'periods': df.index[extreme_margin].tolist(),
                    'values': ebit_margin[extreme_margin].tolist()
                })
        
        return issues
    
    def generate_quality_report(self, data: Dict[str, pd.DataFrame], 
                              quality_stats: Dict[str, pd.DataFrame],
                              validation_issues: List[Dict]):
        """Generate comprehensive data quality report"""
        report_path = os.path.join(self.reports_dir, f"data_quality_report_{self.ticker}.html")
        
        html_content = f"""
        <html>
        <head>
            <title>Data Quality Report - {self.ticker}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .warning {{ color: #ff6b6b; }}
                .good {{ color: #51cf66; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>Data Quality Report - {self.ticker}</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        # Summary section
        html_content += "<div class='section'><h2>Summary</h2><table>"
        html_content += "<tr><th>Dataset</th><th>Records</th><th>Columns</th><th>Date Range</th></tr>"
        
        for key, df in data.items():
            if not df.empty:
                html_content += f"""
                <tr>
                    <td>{key.capitalize()}</td>
                    <td>{len(df)}</td>
                    <td>{len(df.columns)}</td>
                    <td>{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}</td>
                </tr>
                """
        
        html_content += "</table></div>"
        
        # Data quality details
        html_content += "<div class='section'><h2>Data Quality Details</h2>"
        
        for dataset_name, quality_df in quality_stats.items():
            html_content += f"<h3>{dataset_name}</h3>"
            html_content += quality_df.to_html(classes='quality-table', index=False)
        
        html_content += "</div>"
        
        # Validation issues
        if validation_issues:
            html_content += "<div class='section'><h2>Validation Issues</h2>"
            for issue in validation_issues:
                html_content += f"<h3 class='warning'>{issue['type']}</h3>"
                html_content += f"<p>Found in {len(issue['periods'])} periods</p>"
                html_content += f"<p>Sample values: {issue['values'][:5]}</p>"
            html_content += "</div>"
        
        html_content += "</body></html>"
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Quality report saved to {report_path}")
    
    def create_visualizations(self, data: Dict[str, pd.DataFrame]):
        """Create data quality visualizations"""
        # Setup figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Data Quality Visualization - {self.ticker}', fontsize=16)
        
        # 1. Missing data heatmap
        ax = axes[0, 0]
        if not data['fundamental'].empty:
            missing_data = data['fundamental'].isna()
            sns.heatmap(missing_data.T, cbar=True, cmap='Reds', ax=ax)
            ax.set_title('Missing Data Pattern - Fundamentals')
            ax.set_xlabel('Time')
            ax.set_ylabel('Variables')
        
        # 2. Data coverage timeline
        ax = axes[0, 1]
        coverage_data = []
        for key, df in data.items():
            if not df.empty:
                for col in df.columns:
                    non_missing = df[col].notna().sum() / len(df)
                    coverage_data.append({
                        'Dataset': key,
                        'Variable': col,
                        'Coverage': non_missing * 100
                    })
        
        if coverage_data:
            coverage_df = pd.DataFrame(coverage_data)
            coverage_pivot = coverage_df.pivot(index='Variable', columns='Dataset', values='Coverage')
            coverage_pivot.plot(kind='barh', ax=ax)
            ax.set_title('Data Coverage by Variable (%)')
            ax.set_xlabel('Coverage %')
            ax.legend(loc='lower right')
        
        # 3. Distribution of key metrics
        ax = axes[1, 0]
        if 'fundamental' in data and not data['fundamental'].empty:
            metrics_to_plot = ['Revenue', 'EBIT', 'CapEx']
            available_metrics = [m for m in metrics_to_plot if m in data['fundamental'].columns]
            
            if available_metrics:
                for i, metric in enumerate(available_metrics):
                    values = data['fundamental'][metric].dropna()
                    if len(values) > 0:
                        ax.hist(values, bins=20, alpha=0.7, label=metric)
                
                ax.set_title('Distribution of Key Financial Metrics')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.legend()
        
        # 4. Time series of data availability
        ax = axes[1, 1]
        availability_over_time = {}
        
        for key, df in data.items():
            if not df.empty:
                availability_over_time[key] = df.notna().sum(axis=1)
        
        if availability_over_time:
            availability_df = pd.DataFrame(availability_over_time)
            availability_df.plot(ax=ax)
            ax.set_title('Data Availability Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Available Variables')
            ax.legend()
        
        plt.tight_layout()
        viz_path = os.path.join(self.reports_dir, f"data_quality_viz_{self.ticker}.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {viz_path}")
    
    def clean_all_data(self) -> Dict[str, pd.DataFrame]:
        """Main cleaning pipeline"""
        logger.info(f"Starting data cleaning for {self.ticker}")
        
        # Load raw data
        raw_data = self.load_raw_data()
        
        # Track quality statistics
        quality_stats = {}
        
        # Clean each dataset
        cleaned_data = {}
        
        # Clean fundamental data
        if not raw_data['fundamental'].empty:
            logger.info("Cleaning fundamental data...")
            
            # Validate quality
            quality_stats['Fundamental'] = self.validate_data_quality(
                raw_data['fundamental'], 'fundamental'
            )
            
            # Handle missing values
            fundamental_clean = self.handle_missing_values(
                raw_data['fundamental'], method='smart'
            )
            
            # Handle outliers
            fundamental_clean, outlier_report = self.handle_outliers(
                fundamental_clean, method='cap'
            )
            
            cleaned_data['fundamental'] = fundamental_clean
        else:
            cleaned_data['fundamental'] = pd.DataFrame()
        
        # Clean macro data
        if not raw_data['macro'].empty:
            logger.info("Cleaning macro data...")
            
            # Validate quality
            quality_stats['Macro'] = self.validate_data_quality(
                raw_data['macro'], 'macro'
            )
            
            # Handle missing values
            macro_clean = self.handle_missing_values(
                raw_data['macro'], method='interpolate'
            )
            
            # Create rolling averages for noisy macro data
            macro_clean = self.create_rolling_averages(macro_clean)
            
            cleaned_data['macro'] = macro_clean
        else:
            cleaned_data['macro'] = pd.DataFrame()
        
        # Clean price data
        if not raw_data['price'].empty:
            logger.info("Cleaning price data...")
            
            # Validate quality
            quality_stats['Price'] = self.validate_data_quality(
                raw_data['price'], 'price'
            )
            
            # Handle missing values
            price_clean = self.handle_missing_values(
                raw_data['price'], method='forward_fill'
            )
            
            cleaned_data['price'] = price_clean
        else:
            cleaned_data['price'] = pd.DataFrame()
        
        # Align all data to quarterly frequency
        logger.info("Aligning data frequencies...")
        aligned_data = self.align_frequencies(cleaned_data)
        
        # Validate financial relationships
        validation_issues = []
        if not aligned_data['fundamental'].empty:
            issues = self.validate_financial_relationships(aligned_data['fundamental'])
            validation_issues.extend(issues)
        
        # Generate quality report
        self.generate_quality_report(aligned_data, quality_stats, validation_issues)
        
        # Create visualizations
        self.create_visualizations(aligned_data)
        
        # Save cleaned data
        for key, df in aligned_data.items():
            if not df.empty:
                output_path = os.path.join(self.clean_dir, f"{key}_cleaned.csv")
                df.to_csv(output_path)
                logger.info(f"Saved cleaned {key} data to {output_path}")
        
        # Save cleaning summary
        summary = {
            'ticker': self.ticker,
            'cleaning_date': datetime.now().isoformat(),
            'records_before': {k: len(v) for k, v in raw_data.items()},
            'records_after': {k: len(v) for k, v in aligned_data.items()},
            'validation_issues': len(validation_issues),
            'quality_thresholds': {
                'outlier_threshold': self.outlier_threshold,
                'max_missing_pct': self.max_missing_pct,
                'min_quarters': self.min_quarters
            }
        }
        
        summary_path = os.path.join(self.ticker_dir, 'cleaning_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Data cleaning complete for {self.ticker}")
        
        return aligned_data


def main():
    """Example usage"""
    # Clean data for CMG
    cleaner = DataCleaner('CMG')
    cleaned_data = cleaner.clean_all_data()
    
    # Display summary
    print("\n=== Data Cleaning Summary ===")
    for key, df in cleaned_data.items():
        if not df.empty:
            print(f"\n{key}:")
            print(f"  Shape: {df.shape}")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Missing values: {df.isna().sum().sum()}")
            print(f"  Sample data:")
            print(df.head())
        else:
            print(f"\n{key}: No data available")


if __name__ == "__main__":
    main()