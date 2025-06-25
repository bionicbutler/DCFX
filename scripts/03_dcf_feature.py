"""
Script 3: Feature Engineering
Creates features for ML models to predict DCF inputs
"""

import os
import json
import logging
from datetime import datetime
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

class FeatureEngineer:
    """Creates features for DCF component prediction"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.base_dir = "dcf_data"
        self.ticker_dir = os.path.join(self.base_dir, self.ticker)
        self.clean_dir = os.path.join(self.ticker_dir, "clean")
        self.features_dir = os.path.join(self.ticker_dir, "features")
        self.reports_dir = os.path.join(self.ticker_dir, "feature_reports")
        self._create_directories()
        
        # Feature engineering parameters
        self.lag_periods = [1, 2, 4]  # 1Q, 2Q, 1Y lags
        self.rolling_windows = [4, 8]  # 1Y, 2Y rolling windows
        self.correlation_window = 8  # 2Y for rolling correlations
        
        # Target variable definitions
        self.target_definitions = {
            'revenue_growth_1y': 'Forward 1-year revenue growth',
            'revenue_growth_2y': 'Forward 2-year revenue growth',
            'ebit_margin': 'EBIT as % of Revenue',
            'capex_pct_revenue': 'CapEx as % of Revenue',
            'terminal_growth_proxy': 'Long-term growth approximation'
        }
        
    def _create_directories(self):
        """Create folder structure"""
        for directory in [self.features_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def load_clean_data(self) -> Dict[str, pd.DataFrame]:
        """Load cleaned data from previous step"""
        data = {}
        
        # Load fundamental data
        fundamental_path = os.path.join(self.clean_dir, "fundamental_cleaned.csv")
        if os.path.exists(fundamental_path):
            data['fundamental'] = pd.read_csv(fundamental_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded fundamental data: {data['fundamental'].shape}")
        else:
            raise FileNotFoundError("No cleaned fundamental data found. Run data cleaning first.")
        
        # Load macro data
        macro_path = os.path.join(self.clean_dir, "macro_cleaned.csv")
        if os.path.exists(macro_path):
            data['macro'] = pd.read_csv(macro_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded macro data: {data['macro'].shape}")
        
        # Load price data
        price_path = os.path.join(self.clean_dir, "price_cleaned.csv")
        if os.path.exists(price_path):
            data['price'] = pd.read_csv(price_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded price data: {data['price'].shape}")
        
        return data
    
    def create_dcf_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for DCF components"""
        targets = pd.DataFrame(index=df.index)
        
        # Revenue growth targets
        if 'Revenue' in df.columns:
            # 1-year forward revenue growth
            targets['revenue_growth_1y'] = df['Revenue'].pct_change(4).shift(-4)
            # 2-year forward revenue growth (annualized)
            targets['revenue_growth_2y'] = ((df['Revenue'].shift(-8) / df['Revenue']) ** 0.5 - 1).fillna(0)
            
            # Clip extreme values
            targets['revenue_growth_1y'] = targets['revenue_growth_1y'].clip(-0.5, 2.0)
            targets['revenue_growth_2y'] = targets['revenue_growth_2y'].clip(-0.5, 2.0)
        
        # EBIT margin target
        if 'EBIT' in df.columns and 'Revenue' in df.columns:
            targets['ebit_margin'] = (df['EBIT'] / df['Revenue']).clip(-0.5, 0.5)
        
        # CapEx as % of Revenue target
        if 'CapEx' in df.columns and 'Revenue' in df.columns:
            targets['capex_pct_revenue'] = (df['CapEx'] / df['Revenue']).clip(0, 0.5)
        
        # Terminal growth proxy (use GDP growth or sector growth as proxy)
        # This will be enhanced with macro data later
        if 'Revenue' in df.columns:
            # Use long-term average revenue growth as initial proxy
            targets['terminal_growth_proxy'] = df['Revenue'].pct_change(4).rolling(12).mean().clip(0, 0.1)
        
        logger.info(f"Created {len(targets.columns)} target variables")
        return targets
    
    def create_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from fundamental data"""
        features = pd.DataFrame(index=df.index)
        
        # Growth rates
        for col in ['Revenue', 'EBIT', 'TotalAssets']:
            if col in df.columns:
                # QoQ growth
                features[f'{col}_growth_qoq'] = df[col].pct_change(1).clip(-1, 2)
                # YoY growth
                features[f'{col}_growth_yoy'] = df[col].pct_change(4).clip(-1, 2)
        
        # Margins and ratios
        if 'EBIT' in df.columns and 'Revenue' in df.columns:
            features['ebit_margin'] = (df['EBIT'] / df['Revenue']).clip(-0.5, 0.5)
            features['ebit_margin_change'] = features['ebit_margin'].diff()
        
        if 'CapEx' in df.columns and 'Revenue' in df.columns:
            features['capex_intensity'] = (df['CapEx'] / df['Revenue']).clip(0, 0.5)
        
        if 'TotalDebt' in df.columns and 'TotalAssets' in df.columns:
            features['debt_to_assets'] = (df['TotalDebt'] / df['TotalAssets']).clip(0, 1)
        
        # Efficiency metrics
        if 'Revenue' in df.columns and 'TotalAssets' in df.columns:
            features['asset_turnover'] = (df['Revenue'] * 4 / df['TotalAssets']).clip(0, 10)  # Annualized
        
        # Volatility measures
        for col in ['Revenue', 'EBIT']:
            if col in df.columns:
                features[f'{col}_volatility'] = df[col].pct_change().rolling(8).std()
        
        # Trend features
        for col in ['Revenue', 'EBIT']:
            if col in df.columns:
                # Linear trend over past year
                for i in range(len(df)):
                    if i >= 4:
                        y = df[col].iloc[i-4:i].values
                        x = np.arange(4)
                        if len(y) == 4 and not np.isnan(y).any():
                            slope, _ = np.polyfit(x, y, 1)
                            features.loc[df.index[i], f'{col}_trend'] = slope / np.mean(y) if np.mean(y) != 0 else 0
        
        logger.info(f"Created {len(features.columns)} fundamental features")
        return features
    
    def create_lagged_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create lagged features"""
        lagged_features = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col in df.columns:
                for lag in self.lag_periods:
                    lagged_features[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        logger.info(f"Created {len(lagged_features.columns)} lagged features")
        return lagged_features
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create rolling window features"""
        rolling_features = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col in df.columns:
                for window in self.rolling_windows:
                    # Rolling mean
                    rolling_features[f'{col}_ma{window}'] = df[col].rolling(window, min_periods=1).mean()
                    # Rolling std
                    rolling_features[f'{col}_std{window}'] = df[col].rolling(window, min_periods=1).std()
                    # Rolling min/max
                    rolling_features[f'{col}_min{window}'] = df[col].rolling(window, min_periods=1).min()
                    rolling_features[f'{col}_max{window}'] = df[col].rolling(window, min_periods=1).max()
        
        logger.info(f"Created {len(rolling_features.columns)} rolling features")
        return rolling_features
    
    def create_macro_features(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from macro data"""
        features = pd.DataFrame(index=macro_df.index)
        
        # Rate changes
        if '10Y_Treasury' in macro_df.columns:
            features['treasury_change_1q'] = macro_df['10Y_Treasury'].diff(1)
            features['treasury_change_1y'] = macro_df['10Y_Treasury'].diff(4)
            features['treasury_level'] = macro_df['10Y_Treasury']
        
        # Volatility regime
        if 'VIX' in macro_df.columns:
            features['vix_level'] = macro_df['VIX']
            features['vix_percentile'] = macro_df['VIX'].rank(pct=True)
            features['high_volatility'] = (macro_df['VIX'] > macro_df['VIX'].quantile(0.75)).astype(int)
        
        # Economic momentum
        if 'GDP_Growth' in macro_df.columns:
            features['gdp_growth'] = macro_df['GDP_Growth']
            features['gdp_acceleration'] = macro_df['GDP_Growth'].diff(1)
        
        # Use MA features if available
        for col in macro_df.columns:
            if '_MA' in col:
                features[f'macro_{col}'] = macro_df[col]
        
        # Sector performance
        if 'Sector_Index' in macro_df.columns:
            features['sector_return_1q'] = macro_df['Sector_Index'].pct_change(1)
            features['sector_return_1y'] = macro_df['Sector_Index'].pct_change(4)
            features['sector_momentum'] = features['sector_return_1q'] - features['sector_return_1y']
        
        logger.info(f"Created {len(features.columns)} macro features")
        return features
    
    def create_interaction_features(self, fundamental_df: pd.DataFrame, 
                                  macro_df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between fundamental and macro data"""
        interactions = pd.DataFrame(index=fundamental_df.index)
        
        # Margin behavior during high volatility
        if 'ebit_margin' in fundamental_df.columns and 'VIX' in macro_df.columns:
            high_vix = macro_df['VIX'] > macro_df['VIX'].quantile(0.75)
            interactions['margin_x_high_volatility'] = fundamental_df['ebit_margin'] * high_vix
        
        # Growth during different rate environments
        if 'Revenue_growth_yoy' in fundamental_df.columns and '10Y_Treasury' in macro_df.columns:
            high_rates = macro_df['10Y_Treasury'] > macro_df['10Y_Treasury'].median()
            interactions['growth_x_high_rates'] = fundamental_df['Revenue_growth_yoy'] * high_rates
        
        # Debt servicing cost proxy
        if 'debt_to_assets' in fundamental_df.columns and '10Y_Treasury' in macro_df.columns:
            interactions['debt_service_proxy'] = (
                fundamental_df['debt_to_assets'] * macro_df['10Y_Treasury'] / 100
            )
        
        logger.info(f"Created {len(interactions.columns)} interaction features")
        return interactions
    
    def create_rolling_correlations(self, fundamental_df: pd.DataFrame, 
                                  macro_df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling correlation features"""
        correlations = pd.DataFrame(index=fundamental_df.index)
        
        # Company metrics vs macro factors
        pairs = [
            ('Revenue_growth_yoy', 'GDP_Growth'),
            ('Revenue_growth_yoy', 'Sector_Index'),
            ('ebit_margin', 'VIX'),
            ('ebit_margin', '10Y_Treasury')
        ]
        
        for fund_col, macro_col in pairs:
            if fund_col in fundamental_df.columns and macro_col in macro_df.columns:
                # Calculate rolling correlation
                for i in range(len(fundamental_df)):
                    if i >= self.correlation_window:
                        window_slice = slice(i - self.correlation_window, i)
                        fund_values = fundamental_df[fund_col].iloc[window_slice]
                        macro_values = macro_df[macro_col].iloc[window_slice]
                        
                        # Only calculate if we have enough non-null values
                        if fund_values.notna().sum() >= 4 and macro_values.notna().sum() >= 4:
                            corr = fund_values.corr(macro_values)
                            correlations.loc[fundamental_df.index[i], f'corr_{fund_col}_{macro_col}'] = corr
        
        logger.info(f"Created {len(correlations.columns)} correlation features")
        return correlations
    
    def handle_extreme_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle extreme values in features"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                # Use 1st and 99th percentile for capping
                lower = df_clean[col].quantile(0.01)
                upper = df_clean[col].quantile(0.99)
                
                # Cap extreme values
                df_clean[col] = df_clean[col].clip(lower, upper)
                
                # Fill any infinities that might have been created
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        
        return df_clean
    
    def create_feature_report(self, features_df: pd.DataFrame, targets_df: pd.DataFrame):
        """Generate feature engineering report"""
        report_path = os.path.join(self.reports_dir, f"feature_report_{self.ticker}.html")
        
        html_content = f"""
        <html>
        <head>
            <title>Feature Engineering Report - {self.ticker}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .metric {{ font-weight: bold; color: #2ecc71; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>Feature Engineering Report - {self.ticker}</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class='section'>
                <h2>Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Features Created</td><td class='metric'>{len(features_df.columns)}</td></tr>
                    <tr><td>Target Variables</td><td class='metric'>{len(targets_df.columns)}</td></tr>
                    <tr><td>Total Observations</td><td>{len(features_df)}</td></tr>
                    <tr><td>Date Range</td><td>{features_df.index.min()} to {features_df.index.max()}</td></tr>
                </table>
            </div>
            
            <div class='section'>
                <h2>Target Variables</h2>
                <table>
                    <tr><th>Target</th><th>Description</th><th>Non-null Count</th><th>Mean</th><th>Std</th></tr>
        """
        
        for col in targets_df.columns:
            desc = self.target_definitions.get(col, col)
            html_content += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{desc}</td>
                        <td>{targets_df[col].notna().sum()}</td>
                        <td>{targets_df[col].mean():.4f}</td>
                        <td>{targets_df[col].std():.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class='section'>
                <h2>Feature Categories</h2>
                <table>
                    <tr><th>Category</th><th>Count</th><th>Examples</th></tr>
        """
        
        # Categorize features
        categories = {
            'Fundamental': [col for col in features_df.columns if any(x in col for x in ['Revenue', 'EBIT', 'margin', 'debt'])],
            'Lagged': [col for col in features_df.columns if 'lag' in col],
            'Rolling': [col for col in features_df.columns if any(x in col for x in ['ma', 'std', 'min', 'max'])],
            'Macro': [col for col in features_df.columns if any(x in col for x in ['treasury', 'vix', 'gdp', 'sector'])],
            'Interaction': [col for col in features_df.columns if '_x_' in col],
            'Correlation': [col for col in features_df.columns if 'corr_' in col]
        }
        
        for cat, cols in categories.items():
            if cols:
                html_content += f"""
                    <tr>
                        <td>{cat}</td>
                        <td>{len(cols)}</td>
                        <td>{', '.join(cols[:3])}{'...' if len(cols) > 3 else ''}</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Feature report saved to {report_path}")
    
    def create_feature_importance_plot(self, features_df: pd.DataFrame, targets_df: pd.DataFrame):
        """Create feature importance visualization"""
        # Calculate correlations with targets
        correlations = {}
        
        for target in targets_df.columns:
            if target in ['revenue_growth_1y', 'ebit_margin']:  # Focus on key targets
                target_corrs = []
                for feature in features_df.columns:
                    if features_df[feature].notna().sum() > 10:  # Need enough data
                        corr = features_df[feature].corr(targets_df[target])
                        if not np.isnan(corr):
                            target_corrs.append({
                                'feature': feature,
                                'correlation': abs(corr),
                                'sign': 'positive' if corr > 0 else 'negative'
                            })
                
                # Sort by absolute correlation
                target_corrs.sort(key=lambda x: x['correlation'], reverse=True)
                correlations[target] = target_corrs[:20]  # Top 20 features
        
        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, (target, corr_list) in enumerate(correlations.items()):
            ax = axes[idx]
            
            features = [item['feature'] for item in corr_list]
            corrs = [item['correlation'] for item in corr_list]
            colors = ['green' if item['sign'] == 'positive' else 'red' for item in corr_list]
            
            bars = ax.barh(features, corrs, color=colors, alpha=0.7)
            ax.set_xlabel('Absolute Correlation')
            ax.set_title(f'Top Features for {target}')
            ax.set_xlim(0, 1)
            
            # Add value labels
            for bar, corr in zip(bars, corrs):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{corr:.3f}', va='center')
        
        plt.tight_layout()
        plot_path = os.path.join(self.reports_dir, f"feature_importance_{self.ticker}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_path}")
    
    def engineer_all_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main feature engineering pipeline"""
        logger.info(f"Starting feature engineering for {self.ticker}")
        
        # Load clean data
        data = self.load_clean_data()
        
        # Create target variables first
        targets = self.create_dcf_targets(data['fundamental'])
        
        # Create fundamental features
        fundamental_features = self.create_fundamental_features(data['fundamental'])
        
        # Create lagged features for key metrics
        lag_columns = ['Revenue', 'EBIT', 'CapEx', 'TotalDebt', 'TotalAssets']
        lagged_features = self.create_lagged_features(data['fundamental'], lag_columns)
        
        # Create rolling features
        rolling_columns = ['Revenue', 'EBIT', 'ebit_margin', 'capex_intensity']
        rolling_features = self.create_rolling_features(
            pd.concat([data['fundamental'], fundamental_features], axis=1),
            rolling_columns
        )
        
        # Combine all fundamental-based features
        all_features = pd.concat([
            fundamental_features,
            lagged_features,
            rolling_features
        ], axis=1)
        
        # Add macro features if available
        if 'macro' in data and not data['macro'].empty:
            macro_features = self.create_macro_features(data['macro'])
            
            # Align macro features with fundamental data index
            macro_aligned = macro_features.reindex(all_features.index, method='nearest')
            
            # Create interaction features
            interaction_features = self.create_interaction_features(
                fundamental_features, macro_aligned
            )
            
            # Create correlation features
            correlation_features = self.create_rolling_correlations(
                fundamental_features, data['macro']
            )
            
            # Add to all features
            all_features = pd.concat([
                all_features,
                macro_aligned,
                interaction_features,
                correlation_features
            ], axis=1)
        
        # Add price-based features if available
        if 'price' in data and not data['price'].empty:
            price_features = pd.DataFrame(index=all_features.index)
            
            if 'Price' in data['price'].columns:
                price_features['price_return_1q'] = data['price']['Price'].pct_change(1)
                price_features['price_return_1y'] = data['price']['Price'].pct_change(4)
                price_features['price_volatility'] = data['price']['Price'].pct_change().rolling(8).std()
            
            if 'MarketCap' in data['price'].columns:
                price_features['log_market_cap'] = np.log(data['price']['MarketCap'])
            
            # Align and add price features
            price_aligned = price_features.reindex(all_features.index, method='nearest')
            all_features = pd.concat([all_features, price_aligned], axis=1)
        
        # Handle extreme values
        all_features = self.handle_extreme_values(all_features)
        
        # Remove features with too many missing values
        missing_threshold = 0.5
        valid_features = all_features.columns[all_features.isna().mean() < missing_threshold]
        all_features = all_features[valid_features]
        
        logger.info(f"Created {len(all_features.columns)} total features")
        logger.info(f"Created {len(targets.columns)} target variables")
        
        # Generate reports
        self.create_feature_report(all_features, targets)
        self.create_feature_importance_plot(all_features, targets)
        
        # Save features and targets
        features_path = os.path.join(self.features_dir, "all_features.csv")
        all_features.to_csv(features_path)
        logger.info(f"Saved features to {features_path}")
        
        targets_path = os.path.join(self.features_dir, "targets.csv")
        targets.to_csv(targets_path)
        logger.info(f"Saved targets to {targets_path}")
        
        # Save feature metadata
        metadata = {
            'ticker': self.ticker,
            'engineering_date': datetime.now().isoformat(),
            'n_features': len(all_features.columns),
            'n_targets': len(targets.columns),
            'n_observations': len(all_features),
            'date_range': {
                'start': str(all_features.index.min()),
                'end': str(all_features.index.max())
            },
            'feature_categories': {
                'fundamental': len([col for col in all_features.columns if any(x in col for x in ['Revenue', 'EBIT', 'margin', 'debt'])]),
                'lagged': len([col for col in all_features.columns if 'lag' in col]),
                'rolling': len([col for col in all_features.columns if any(x in col for x in ['ma', 'std', 'min', 'max'])]),
                'macro': len([col for col in all_features.columns if any(x in col for x in ['treasury', 'vix', 'gdp', 'sector'])]),
                'interaction': len([col for col in all_features.columns if '_x_' in col]),
                'correlation': len([col for col in all_features.columns if 'corr_' in col])
            },
            'parameters': {
                'lag_periods': self.lag_periods,
                'rolling_windows': self.rolling_windows,
                'correlation_window': self.correlation_window
            }
        }
        
        metadata_path = os.path.join(self.ticker_dir, 'feature_engineering_summary.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Feature engineering complete for {self.ticker}")
        
        return all_features, targets


def main():
    """Example usage"""
    # Engineer features for CMG
    engineer = FeatureEngineer('CMG')
    features, targets = engineer.engineer_all_features()
    
    # Display summary
    print("\n=== Feature Engineering Summary ===")
    print(f"\nFeatures shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Date range: {features.index.min()} to {features.index.max()}")
    
    print("\n=== Target Variables ===")
    for col in targets.columns:
        print(f"{col}: {targets[col].notna().sum()} non-null values")
    
    print("\n=== Feature Categories ===")
    categories = {
        'Fundamental': len([col for col in features.columns if any(x in col for x in ['Revenue', 'EBIT', 'margin', 'debt'])]),
        'Lagged': len([col for col in features.columns if 'lag' in col]),
        'Rolling': len([col for col in features.columns if any(x in col for x in ['ma', 'std', 'min', 'max'])]),
        'Macro': len([col for col in features.columns if any(x in col for x in ['treasury', 'vix', 'gdp', 'sector'])]),
        'Interaction': len([col for col in features.columns if '_x_' in col]),
        'Correlation': len([col for col in features.columns if 'corr_' in col])
    }
    
    for cat, count in categories.items():
        print(f"{cat}: {count} features")
    
    print("\n=== Sample Features ===")
    print(features.head())


if __name__ == "__main__":
    main()