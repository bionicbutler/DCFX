"""        
Script 1: Data Collection & API Setup
Collects fundamental and macro data for DCF analysis
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    """Modular data collector for DCF analysis"""
    
    def __init__(self, ticker: str, years_back: int = 10):
        self.ticker = ticker.upper()
        self.years_back = years_back
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * years_back)
        
        # Setup folder structure
        self.base_dir = "dcf_data"
        self.ticker_dir = os.path.join(self.base_dir, self.ticker)
        self.raw_dir = os.path.join(self.ticker_dir, "raw")
        self.cache_dir = os.path.join(self.ticker_dir, "cache")
        self._create_directories()
        
        # Setup session with retry logic
        self.session = self._create_session()
        
    def _create_directories(self):
        """Create folder structure for data storage"""
        for directory in [self.base_dir, self.ticker_dir, self.raw_dir, self.cache_dir]:
            os.makedirs(directory, exist_ok=True)
            
    def _create_session(self) -> requests.Session:
        """Create session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _check_cache(self, cache_name: str, max_age_days: int = 1) -> Optional[pd.DataFrame]:
        """Check if cached data exists and is fresh"""
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.csv")
        if os.path.exists(cache_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - mod_time < timedelta(days=max_age_days):
                logger.info(f"Loading {cache_name} from cache")
                return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return None
    
    def _save_cache(self, data: pd.DataFrame, cache_name: str):
        """Save data to cache"""
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.csv")
        data.to_csv(cache_path)
        logger.info(f"Saved {cache_name} to cache")
    
    def fetch_sec_fundamental_data(self) -> Optional[pd.DataFrame]:
        """Fetch quarterly fundamental data from SEC EDGAR"""
        cache_name = f"sec_fundamentals_{self.ticker}"
        cached_data = self._check_cache(cache_name, max_age_days=7)
        if cached_data is not None:
            return cached_data
        
        try:
            # SEC EDGAR headers - MUST include proper User-Agent
            headers = {
                'User-Agent': 'DCF Analysis Tool (corbin.footitt@verecan.com)',  # UPDATE WITH YOUR EMAIL!
                'Accept': 'application/json',
                'Host': 'data.sec.gov'
            }
            
            # Get company CIK
            cik_url = "https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(cik_url, headers={'User-Agent': 'DCF Analysis Tool (corbin.footitt@verecan.com)'})
            response.raise_for_status()
            company_data = response.json()
            
            # Find CIK for ticker
            cik = None
            for key, company in company_data.items():
                if company['ticker'] == self.ticker:
                    cik = str(company['cik_str']).zfill(10)
                    break
            
            if not cik:
                logger.error(f"CIK not found for {self.ticker}")
                return None
            
            logger.info(f"Found CIK {cik} for {self.ticker}")
            
            # Fetch company facts
            facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            time.sleep(0.1)  # SEC rate limiting
            response = self.session.get(facts_url, headers=headers)
            response.raise_for_status()
            facts = response.json()
            
            # Debug: Print available metrics
            if 'facts' in facts and 'us-gaap' in facts['facts']:
                available_metrics = list(facts['facts']['us-gaap'].keys())
                logger.info(f"Available metrics count: {len(available_metrics)}")
                
                # Look for debt-related metrics
                debt_metrics = [m for m in available_metrics if 'debt' in m.lower() or 'borrowing' in m.lower()]
                logger.info(f"Debt-related metrics found: {debt_metrics[:10]}")  # Show first 10
            
            # Enhanced metrics map with more debt options
            metrics_map = {
                'Revenue': [
                    'Revenues', 
                    'RevenueFromContractWithCustomerExcludingAssessedTax', 
                    'SalesRevenueNet',
                    'RevenueFromContractWithCustomerIncludingAssessedTax'
                ],
                'EBIT': [
                    'OperatingIncomeLoss', 
                    'IncomeLossFromContinuingOperationsBeforeIncomeTaxes',
                    'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'
                ],
                'CapEx': [
                    'PaymentsToAcquirePropertyPlantAndEquipment', 
                    'CapitalExpenditures',
                    'PaymentsForCapitalImprovements'
                ],
                'TotalAssets': [
                    'Assets'
                ],
                'TotalDebt': [
                    'LongTermDebt',
                    'LongTermDebtCurrent', 
                    'DebtCurrent',
                    'LongTermDebtAndCapitalLeaseObligations',
                    'LongTermDebtNoncurrent',
                    'DebtAndCapitalLeaseObligations',
                    'LongTermDebtAndCapitalLeaseObligationsCurrent',
                    'ShortTermBorrowings',
                    'LongTermDebtAndShortTermBorrowings',
                    'DebtLongtermAndShorttermCombinedAmount'
                ]
            }
            
            quarterly_data = {}
            metrics_found = {}
            
            for metric_name, possible_keys in metrics_map.items():
                for key in possible_keys:
                    try:
                        if key in facts['facts']['us-gaap']:
                            metric_data = facts['facts']['us-gaap'][key]['units']
                            
                            # Handle USD units
                            if 'USD' in metric_data:
                                usd_data = metric_data['USD']
                            else:
                                continue
                            
                            # Filter for 10-Q and 10-K forms
                            quarterly_values = []
                            for item in usd_data:
                                if item.get('form') in ['10-Q', '10-K']:
                                    quarterly_values.append({
                                        'date': item['end'],
                                        'value': item['val'],
                                        'form': item['form']
                                    })
                            
                            if quarterly_values:
                                df_metric = pd.DataFrame(quarterly_values)
                                df_metric['date'] = pd.to_datetime(df_metric['date'])
                                df_metric = df_metric.sort_values('date')
                                df_metric = df_metric.drop_duplicates(subset=['date'], keep='last')
                                quarterly_data[metric_name] = df_metric.set_index('date')['value']
                                metrics_found[metric_name] = key
                                logger.info(f"Found {metric_name} using tag: {key}")
                                break
                    except KeyError:
                        continue
                
                if metric_name not in metrics_found:
                    logger.warning(f"Could not find {metric_name} in SEC data")
            
            # If we still don't have TotalDebt, try to calculate it
            if 'TotalDebt' not in quarterly_data and len(quarterly_data) > 0:
                logger.info("Attempting to calculate TotalDebt from components...")
                
                # Try to find current and non-current portions
                debt_components = []
                component_tags = [
                    'LongTermDebtNoncurrent',
                    'LongTermDebtCurrent',
                    'DebtCurrent',
                    'ShortTermBorrowings',
                    'CapitalLeaseObligationsCurrent',
                    'CapitalLeaseObligationsNoncurrent'
                ]
                
                for tag in component_tags:
                    if tag in facts['facts']['us-gaap']:
                        try:
                            metric_data = facts['facts']['us-gaap'][tag]['units']['USD']
                            quarterly_values = []
                            for item in metric_data:
                                if item.get('form') in ['10-Q', '10-K']:
                                    quarterly_values.append({
                                        'date': item['end'],
                                        'value': item['val']
                                    })
                            
                            if quarterly_values:
                                df_temp = pd.DataFrame(quarterly_values)
                                df_temp['date'] = pd.to_datetime(df_temp['date'])
                                df_temp = df_temp.drop_duplicates(subset=['date'], keep='last')
                                debt_components.append(df_temp.set_index('date')['value'])
                                logger.info(f"Found debt component: {tag}")
                        except:
                            continue
                
                # Sum components if found
                if debt_components:
                    combined_debt = pd.concat(debt_components, axis=1).sum(axis=1)
                    quarterly_data['TotalDebt'] = combined_debt
                    logger.info("Calculated TotalDebt from components")
            
            if quarterly_data:
                df_fundamentals = pd.DataFrame(quarterly_data)
                df_fundamentals = df_fundamentals[df_fundamentals.index >= self.start_date]
                
                # Save raw data
                raw_path = os.path.join(self.raw_dir, f"sec_fundamentals_{self.ticker}.csv")
                df_fundamentals.to_csv(raw_path)
                # Save to cache
                if not df_fundamentals.empty and isinstance(df_fundamentals, pd.DataFrame):
                    self._save_cache(df_fundamentals, cache_name)
                    logger.info(f"Successfully fetched SEC data for {self.ticker}")
                    logger.info(f"Metrics found: {list(metrics_found.keys())}")
                    return df_fundamentals
                    logger.error(f"SEC data DataFrame for {self.ticker} is empty.")
                    return None
        except Exception as e:
            logger.error(f"Error fetching SEC data: {e}")
            
        return None
    
    def fetch_yfinance_fundamental_data(self) -> Optional[pd.DataFrame]:
        """Fallback to yfinance for fundamental data"""
        cache_name = f"yf_fundamentals_{self.ticker}"
        cached_data = self._check_cache(cache_name, max_age_days=1)
        if cached_data is not None:
            return cached_data
        
        try:
            stock = yf.Ticker(self.ticker)
            
            # Get quarterly financials
            income_stmt = stock.quarterly_income_stmt
            balance_sheet = stock.quarterly_balance_sheet
            cash_flow = stock.quarterly_cash_flow
            
            if income_stmt.empty or balance_sheet.empty:
                logger.error(f"No yfinance data available for {self.ticker}")
                return None
            
            # Initialize dictionary to collect data
            data_dict = {}
            
            # Extract dates from income statement columns
            dates = income_stmt.columns
            
            # Revenue
            if 'Total Revenue' in income_stmt.index:
                data_dict['Revenue'] = income_stmt.loc['Total Revenue'].to_dict()
            
            # EBIT (Operating Income)
            if 'Operating Income' in income_stmt.index:
                data_dict['EBIT'] = income_stmt.loc['Operating Income'].to_dict()
            elif 'EBIT' in income_stmt.index:
                data_dict['EBIT'] = income_stmt.loc['EBIT'].to_dict()
            
            # Total Assets
            if 'Total Assets' in balance_sheet.index:
                data_dict['TotalAssets'] = balance_sheet.loc['Total Assets'].to_dict()
            
            # Total Debt
            debt_items = ['Total Debt', 'Long Term Debt', 'Short Long Term Debt']
            for item in debt_items:
                if item in balance_sheet.index:
                    data_dict['TotalDebt'] = balance_sheet.loc[item].to_dict()
                    break
            
            # CapEx
            if 'Capital Expenditure' in cash_flow.index:
                capex_data = cash_flow.loc['Capital Expenditure'].to_dict()
                # Make positive (usually negative in cash flow)
                data_dict['CapEx'] = {k: -v if v is not None else None for k, v in capex_data.items()}
            
            # Create DataFrame from dictionary
            if data_dict:
                fundamentals = pd.DataFrame.from_dict(data_dict, orient='index').T
                fundamentals.index = pd.to_datetime(fundamentals.index)
                fundamentals = fundamentals.sort_index()
                fundamentals = fundamentals[fundamentals.index >= self.start_date]
                
                # Save raw data
                raw_path = os.path.join(self.raw_dir, f"yf_fundamentals_{self.ticker}.csv")
                fundamentals.to_csv(raw_path)
                # Save to cache
                if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty:
                    self._save_cache(fundamentals, cache_name)
                    logger.info(f"Successfully fetched yfinance data for {self.ticker}")
                    return fundamentals
                else:
                    logger.error(f"Fundamentals DataFrame is empty or None for {self.ticker}")
                    return None
                logger.error(f"No fundamental data extracted from yfinance for {self.ticker}")
                return None
            
        except Exception as e:
            logger.error(f"Error fetching yfinance data: {e}")
            return None
    
    def combine_fundamental_data(self, sec_data: Optional[pd.DataFrame], 
                               yf_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Combine SEC and yfinance data, using yfinance to fill missing values"""
        
        # If no SEC data, return yfinance data
        if sec_data is None or sec_data.empty:
            logger.info("No SEC data available, using only yfinance data")
            return yf_data if yf_data is not None else pd.DataFrame()
        
        # If no yfinance data, return SEC data
        if yf_data is None or yf_data.empty:
            logger.info("No yfinance data available, using only SEC data")
            return sec_data
        
        # Both sources available - combine them
        logger.info("Combining SEC and yfinance data...")
        
        # Start with SEC data as the base
        combined_data = sec_data.copy()
        
        # Align yfinance data to nearest SEC dates
        for col in combined_data.columns:
            if col in yf_data.columns:
                # Count missing values before
                missing_before = combined_data[col].isna().sum()
                
                # For each missing value in SEC data, try to find nearest yfinance value
                for idx in combined_data[combined_data[col].isna()].index:
                    # Find nearest date in yfinance data
                    yf_dates = yf_data.index[~yf_data[col].isna()]
                    if len(yf_dates) > 0:
                        # Find closest date
                        time_diffs = abs(yf_dates - idx)
                        nearest_idx = time_diffs.argmin()
                        nearest_date = yf_dates[nearest_idx]
                        
                        # If within 45 days, use the value
                        if time_diffs[nearest_idx].days <= 45:
                            combined_data.loc[idx, col] = yf_data.loc[nearest_date, col]
                
                missing_after = combined_data[col].isna().sum()
                if missing_before > missing_after:
                    logger.info(f"Filled {missing_before - missing_after} missing {col} values from yfinance")
        
        # Add any columns from yfinance that aren't in SEC data
        for col in yf_data.columns:
            if col not in combined_data.columns:
                logger.info(f"Adding {col} from yfinance (not in SEC data)")
                # Align yfinance data to SEC dates
                aligned_data = pd.Series(index=combined_data.index, dtype='float64')
                for idx in combined_data.index:
                    yf_dates = yf_data.index[~yf_data[col].isna()]
                    if len(yf_dates) > 0:
                        time_diffs = abs(yf_dates - idx)
                        nearest_idx = time_diffs.argmin()
                        nearest_date = yf_dates[nearest_idx]
                        if time_diffs[nearest_idx].days <= 45:
                            aligned_data.loc[idx] = yf_data.loc[nearest_date, col]
                
                combined_data[col] = aligned_data
        
        # Log final data quality
        logger.info("\nFinal data quality:")
        for col in combined_data.columns:
            missing = combined_data[col].isna().sum()
            total = len(combined_data)
            logger.info(f"  {col}: {total - missing}/{total} records ({(total-missing)/total*100:.1f}% complete)")
        
        return combined_data
    
    def fetch_macro_data(self) -> pd.DataFrame:
        """Fetch macroeconomic data from FRED and other sources"""
        cache_name = "macro_data"
        cached_data = self._check_cache(cache_name, max_age_days=1)
        if cached_data is not None:
            return cached_data
        
        macro_series_list = []
        
        # FRED data
        fred_series = {
            'DGS10': '10Y_Treasury',  # 10-Year Treasury Rate
            'GDP': 'GDP_Growth',       # GDP
            'VIXCLS': 'VIX'           # VIX
        }
        
        for series_id, name in fred_series.items():
            try:
                data = pdr.get_data_fred(series_id, start=self.start_date, end=self.end_date)
                # Force to Series and ensure it's 1D
                if isinstance(data, pd.DataFrame):
                    data = data.squeeze()  # This converts single column DataFrame to Series
                # Resample to quarterly
                data_quarterly = data.resample('QE').mean()
                # Rename the series
                data_quarterly.name = name
                macro_series_list.append(data_quarterly)
                logger.info(f"Successfully fetched {name} from FRED")
            except Exception as e:
                logger.error(f"Error fetching {series_id} from FRED: {e}")
        
        # Sector index
        try:
            if self.ticker in ['NVDA', 'AMD', 'INTC', 'TSM']:
                sector_ticker = '^SOX'  # Semiconductor index
            elif self.ticker in ['CMG', 'MCD', 'SBUX', 'YUM']:
                sector_ticker = 'XLY'   # Consumer Discretionary ETF
            else:
                sector_ticker = 'SPY'   # Default to S&P 500
            
            sector_data = yf.download(
                sector_ticker, 
                start=self.start_date, 
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )
            if not sector_data.empty:
                sector_quarterly = sector_data['Close'].resample('QE').last()
                sector_quarterly.name = 'Sector_Index'
                macro_series_list.append(sector_quarterly)
                logger.info(f"Successfully fetched sector index {sector_ticker}")
        except Exception as e:
            logger.error(f"Error fetching sector index: {e}")
        
        # Combine all series into DataFrame
        if macro_series_list:
            # Use concat to properly align all series
            df_macro = pd.concat(macro_series_list, axis=1)
            df_macro = df_macro.dropna(how='all')
            
            # Save raw data
            raw_path = os.path.join(self.raw_dir, "macro_data.csv")
            df_macro.to_csv(raw_path)
            
            # Save to cache
            self._save_cache(df_macro, cache_name)
            
            return df_macro
        else:
            return pd.DataFrame()
    
    def fetch_price_data(self) -> pd.DataFrame:
        """Fetch stock price data"""
        cache_name = f"price_data_{self.ticker}"
        cached_data = self._check_cache(cache_name, max_age_days=1)
        if cached_data is not None:
            return cached_data
        
        try:
            stock_data = yf.download(
                self.ticker, 
                start=self.start_date, 
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )
            
            if not stock_data.empty:
                # Resample to quarterly
                price_quarterly = stock_data['Close'].resample('QE').last()
                volume_quarterly = stock_data['Volume'].resample('QE').mean()
                
                price_data = pd.DataFrame({
                    'Price': price_quarterly,
                    'Volume': volume_quarterly
                })
                
                # Calculate market cap if shares outstanding available
                ticker_obj = yf.Ticker(self.ticker)
                info = ticker_obj.info
                if 'sharesOutstanding' in info and info['sharesOutstanding']:
                    price_data['MarketCap'] = price_data['Price'] * info['sharesOutstanding']
                
                # Save raw data
                raw_path = os.path.join(self.raw_dir, f"price_data_{self.ticker}.csv")
                price_data.to_csv(raw_path)
                
                # Save to cache
                self._save_cache(price_data, cache_name)
                
                logger.info(f"Successfully fetched price data for {self.ticker}")
                return price_data
            else:
                logger.error(f"No price data available for {self.ticker}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return pd.DataFrame()
    
    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """Main method to collect all data"""
        logger.info(f"Starting data collection for {self.ticker}")
        
        results = {
            'fundamental_data': None,
            'macro_data': None,
            'price_data': None
        }
        
        # Fetch from both sources
        sec_data = self.fetch_sec_fundamental_data()
        yf_data = self.fetch_yfinance_fundamental_data()
        
        # Combine fundamental data
        combined_fundamentals = self.combine_fundamental_data(sec_data, yf_data)
        
        # Save combined fundamentals
        if not combined_fundamentals.empty:
            combined_path = os.path.join(self.raw_dir, f"combined_fundamentals_{self.ticker}.csv")
            combined_fundamentals.to_csv(combined_path)
            logger.info(f"Saved combined fundamentals to {combined_path}")
        
        results['fundamental_data'] = combined_fundamentals
        
        # Fetch macro data
        results['macro_data'] = self.fetch_macro_data()
        
        # Fetch price data
        results['price_data'] = self.fetch_price_data()
        
        # Save summary
        summary = {
            'ticker': self.ticker,
            'collection_date': datetime.now().isoformat(),
            'data_start': self.start_date.isoformat(),
            'data_end': self.end_date.isoformat(),
            'fundamental_records': len(combined_fundamentals) if combined_fundamentals is not None else 0,
            'sec_records': len(sec_data) if sec_data is not None else 0,
            'yf_records': len(yf_data) if yf_data is not None else 0,
            'macro_records': len(results['macro_data']) if results['macro_data'] is not None else 0,
            'price_records': len(results['price_data']) if results['price_data'] is not None else 0
        }
        
        summary_path = os.path.join(self.ticker_dir, 'collection_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Data collection complete for {self.ticker}")
        logger.info(f"Summary saved to {summary_path}")
        
        return results


def main():
    """Example usage"""
    # IMPORTANT: Update the User-Agent email in fetch_sec_fundamental_data() before running!
    
    # Collect data for CMG
    collector = DataCollector('AMAT', years_back=10)
    data = collector.collect_all_data()
    
    # Display summary
    print("\n=== Data Collection Summary ===")
    for key, df in data.items():
        if df is not None and not df.empty:
            print(f"\n{key}:")
            print(f"  Shape: {df.shape}")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Sample data:")
            print(df.head())
        else:
            print(f"\n{key}: No data collected")


if __name__ == "__main__":
    main()