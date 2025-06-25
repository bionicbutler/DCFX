"""        
Script 1: Data Collection & API Setup (Enhanced for DCF)
Collects fundamental, macro, and DCF-specific data for analysis
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
    """Modular data collector for DCF analysis with enhanced DCF-specific data"""
    
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
                'User-Agent': 'DCF Analysis Tool (your.email@example.com)',  # UPDATE WITH YOUR EMAIL!
                'Accept': 'application/json',
                'Host': 'data.sec.gov'
            }
            
            # Get company CIK
            cik_url = "https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(cik_url, headers={'User-Agent': 'DCF Analysis Tool (your.email@example.com)'})
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
            
            # Enhanced metrics map with more options
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
                ],
                # Add new metrics for DCF
                'PretaxIncome': [
                    'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
                    'IncomeLossFromContinuingOperationsBeforeIncomeTaxes',
                    'PretaxIncomeLoss'
                ],
                'TaxProvision': [
                    'IncomeTaxExpenseBenefit',
                    'IncomeTaxExpenseContinuingOperations'
                ],
                'InterestExpense': [
                    'InterestExpense',
                    'InterestExpenseDebt',
                    'InterestExpenseNet'
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
            
            if quarterly_data:
                df_fundamentals = pd.DataFrame(quarterly_data)
                df_fundamentals = df_fundamentals[df_fundamentals.index >= self.start_date]
                
                # Save raw data
                raw_path = os.path.join(self.raw_dir, f"sec_fundamentals_{self.ticker}.csv")
                df_fundamentals.to_csv(raw_path)
                # Save to cache
                if not df_fundamentals.empty:
                    self._save_cache(df_fundamentals, cache_name)
                    logger.info(f"Successfully fetched SEC data for {self.ticker}")
                    return df_fundamentals
                    
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
                if not fundamentals.empty:
                    self._save_cache(fundamentals, cache_name)
                    logger.info(f"Successfully fetched yfinance data for {self.ticker}")
                    return fundamentals
                    
        except Exception as e:
            logger.error(f"Error fetching yfinance data: {e}")
            return None
    
    def fetch_additional_dcf_data(self) -> pd.DataFrame:
        """Fetch additional data needed for DCF calculations"""
        cache_name = f"dcf_additional_{self.ticker}"
        cached_data = self._check_cache(cache_name, max_age_days=1)
        if cached_data is not None:
            return cached_data
        
        dcf_data = {}
        
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            
            # 1. Shares Outstanding
            if 'sharesOutstanding' in info:
                dcf_data['shares_outstanding'] = info['sharesOutstanding']
            elif 'impliedSharesOutstanding' in info:
                dcf_data['shares_outstanding'] = info['impliedSharesOutstanding']
            elif 'marketCap' in info and 'currentPrice' in info and info['currentPrice'] > 0:
                # Fallback: calculate from market cap
                dcf_data['shares_outstanding'] = info['marketCap'] / info['currentPrice']
            else:
                logger.warning("Could not determine shares outstanding")
                dcf_data['shares_outstanding'] = None
            
            # 2. Beta
            if 'beta' in info:
                dcf_data['beta'] = info['beta']
            else:
                # Calculate beta from returns if not available
                logger.info("Beta not in info, calculating from returns...")
                dcf_data['beta'] = self._calculate_beta()
            
            # 3. Current Price (for validation)
            if 'currentPrice' in info:
                dcf_data['current_price'] = info['currentPrice']
            elif 'regularMarketPrice' in info:
                dcf_data['current_price'] = info['regularMarketPrice']
            
            # 4. Financial Ratios
            if 'returnOnEquity' in info:
                dcf_data['roe'] = info['returnOnEquity']
            
            if 'profitMargins' in info:
                dcf_data['net_margin'] = info['profitMargins']
            
            # 5. Enterprise Value (for validation)
            if 'enterpriseValue' in info:
                dcf_data['enterprise_value'] = info['enterpriseValue']
            
            # 6. Market Cap
            if 'marketCap' in info:
                dcf_data['market_cap'] = info['marketCap']
                
            # 5. Save DCF-specific data
            dcf_df = pd.DataFrame([dcf_data], index=[datetime.now()])
            
            # Save to cache
            self._save_cache(dcf_df, cache_name)
            
            return dcf_df
            
        except Exception as e:
            logger.error(f"Error fetching additional DCF data: {e}")
            return pd.DataFrame()
    
    def _calculate_beta(self, market_ticker: str = '^GSPC') -> float:
        """Calculate beta using 2-year monthly returns vs S&P 500"""
        try:
            # Get 2 years of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            # Download stock data
            stock_data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            market_data = yf.download(market_ticker, start=start_date, end=end_date, progress=False)
            
            if stock_data.empty or market_data.empty:
                logger.warning("Could not download data for beta calculation")
                return 1.0  # Default beta
            
            # Calculate monthly returns
            stock_returns = stock_data['Adj Close'].resample('M').last().pct_change().dropna()
            market_returns = market_data['Adj Close'].resample('M').last().pct_change().dropna()
            
            # Align indices
            aligned = pd.DataFrame({
                'stock': stock_returns,
                'market': market_returns
            }).dropna()
            
            if len(aligned) < 12:
                logger.warning("Insufficient data for beta calculation")
                return 1.0
            
            # Calculate beta (covariance / variance)
            covariance = aligned['stock'].cov(aligned['market'])
            market_variance = aligned['market'].var()
            
            beta = covariance / market_variance if market_variance > 0 else 1.0
            
            # Sanity check
            beta = max(0.2, min(3.0, beta))  # Cap between 0.2 and 3.0
            
            logger.info(f"Calculated beta: {beta:.2f}")
            return beta
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 1.0  # Default market beta
    
    def fetch_working_capital_data(self) -> pd.DataFrame:
        """Fetch working capital components from financial statements"""
        cache_name = f"working_capital_{self.ticker}"
        cached_data = self._check_cache(cache_name, max_age_days=7)
        if cached_data is not None:
            return cached_data
        
        try:
            stock = yf.Ticker(self.ticker)
            
            # Get quarterly balance sheet
            balance_sheet = stock.quarterly_balance_sheet
            
            if balance_sheet.empty:
                logger.warning("No balance sheet data available")
                return pd.DataFrame()
            
            wc_data = {}
            dates = balance_sheet.columns
            
            # Extract working capital components
            for date in dates:
                wc_components = {}
                
                # Current Assets (excluding cash)
                if 'Total Current Assets' in balance_sheet.index:
                    current_assets = balance_sheet.loc['Total Current Assets', date]
                    # Subtract cash if available
                    if 'Cash And Cash Equivalents' in balance_sheet.index:
                        cash = balance_sheet.loc['Cash And Cash Equivalents', date]
                        wc_components['current_assets_ex_cash'] = current_assets - cash
                    else:
                        wc_components['current_assets_ex_cash'] = current_assets
                
                # Current Liabilities (excluding debt)
                if 'Total Current Liabilities' in balance_sheet.index:
                    current_liab = balance_sheet.loc['Total Current Liabilities', date]
                    # Subtract short-term debt if available
                    if 'Short Term Debt' in balance_sheet.index:
                        st_debt = balance_sheet.loc['Short Term Debt', date]
                        wc_components['current_liab_ex_debt'] = current_liab - st_debt
                    else:
                        wc_components['current_liab_ex_debt'] = current_liab
                
                # Calculate operating working capital
                if 'current_assets_ex_cash' in wc_components and 'current_liab_ex_debt' in wc_components:
                    wc_components['operating_working_capital'] = (
                        wc_components['current_assets_ex_cash'] - 
                        wc_components['current_liab_ex_debt']
                    )
                
                # Individual components if available
                for item in ['Inventory', 'Accounts Receivable', 'Accounts Payable']:
                    if item in balance_sheet.index:
                        wc_components[item.lower().replace(' ', '_')] = balance_sheet.loc[item, date]
                
                wc_data[date] = wc_components
            
            # Convert to DataFrame
            wc_df = pd.DataFrame.from_dict(wc_data, orient='index')
            wc_df.index = pd.to_datetime(wc_df.index)
            wc_df = wc_df.sort_index()
            
            # Save to cache
            self._save_cache(wc_df, cache_name)
            
            return wc_df
            
        except Exception as e:
            logger.error(f"Error fetching working capital data: {e}")
            return pd.DataFrame()
    
    def calculate_historical_metrics(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate historical metrics needed for DCF assumptions"""
        if fundamental_data is None or fundamental_data.empty:
            logger.warning("No fundamental data to calculate metrics")
            return pd.DataFrame()
        
        metrics = pd.DataFrame(index=fundamental_data.index)
        
        try:
            stock = yf.Ticker(self.ticker)
            
            # 1. Historical Tax Rate
            income_stmt = stock.quarterly_income_stmt
            if not income_stmt.empty:
                if 'Pretax Income' in income_stmt.index and 'Tax Provision' in income_stmt.index:
                    pretax = income_stmt.loc['Pretax Income']
                    tax = income_stmt.loc['Tax Provision']
                    # Calculate effective tax rate where pretax income is positive
                    tax_rate = pd.Series(index=pretax.index, dtype=float)
                    positive_pretax = pretax > 0
                    tax_rate[positive_pretax] = tax[positive_pretax] / pretax[positive_pretax]
                    tax_rate = tax_rate.clip(0, 0.5)  # Cap between 0-50%
                    # Align with fundamental data index
                    metrics['effective_tax_rate'] = tax_rate.reindex(fundamental_data.index, method='nearest')
            
            # 2. Cost of Debt (interest expense / total debt)
            if 'Interest Expense' in income_stmt.index and 'TotalDebt' in fundamental_data.columns:
                interest = income_stmt.loc['Interest Expense']
                # Make interest positive (usually negative in statements)
                interest = interest.abs()
                # Annualize interest expense
                annual_interest = interest * 4
                # Align with fundamental data index
                interest_aligned = annual_interest.reindex(fundamental_data.index, method='nearest')
                # Calculate cost of debt where debt > 0
                debt_positive = fundamental_data['TotalDebt'] > 0
                cost_of_debt = pd.Series(index=fundamental_data.index, dtype=float)
                cost_of_debt[debt_positive] = interest_aligned[debt_positive] / fundamental_data['TotalDebt'][debt_positive]
                metrics['implied_cost_of_debt'] = cost_of_debt.clip(0, 0.15)  # Cap at 15%
            
            # 3. Working Capital as % of Revenue
            wc_data = self.fetch_working_capital_data()
            if not wc_data.empty and 'operating_working_capital' in wc_data.columns and 'Revenue' in fundamental_data.columns:
                # Align indices
                wc_aligned = wc_data['operating_working_capital'].reindex(fundamental_data.index, method='nearest')
                # Calculate WC as % of annualized revenue
                revenue_positive = fundamental_data['Revenue'] > 0
                wc_pct = pd.Series(index=fundamental_data.index, dtype=float)
                wc_pct[revenue_positive] = wc_aligned[revenue_positive] / (fundamental_data['Revenue'][revenue_positive] * 4)
                metrics['wc_pct_revenue'] = wc_pct.clip(-0.5, 0.5)  # Cap between -50% and 50%
            
            # 4. Historical growth rates
            if 'Revenue' in fundamental_data.columns:
                metrics['revenue_growth_1y'] = fundamental_data['Revenue'].pct_change(4)
                # 3-year CAGR (if we have enough data)
                if len(fundamental_data) >= 12:
                    revenue_3y_ago = fundamental_data['Revenue'].shift(12)
                    revenue_current = fundamental_data['Revenue']
                    valid_growth = (revenue_3y_ago > 0) & (revenue_current > 0)
                    growth_3y = pd.Series(index=fundamental_data.index, dtype=float)
                    growth_3y[valid_growth] = (revenue_current[valid_growth] / revenue_3y_ago[valid_growth]) ** (1/3) - 1
                    metrics['revenue_growth_3y'] = growth_3y
            
            # 5. Historical margins
            if 'EBIT' in fundamental_data.columns and 'Revenue' in fundamental_data.columns:
                revenue_positive = fundamental_data['Revenue'] > 0
                ebit_margin = pd.Series(index=fundamental_data.index, dtype=float)
                ebit_margin[revenue_positive] = fundamental_data['EBIT'][revenue_positive] / fundamental_data['Revenue'][revenue_positive]
                metrics['historical_ebit_margin'] = ebit_margin.clip(-1, 1)
            
            # 6. CapEx intensity
            if 'CapEx' in fundamental_data.columns and 'Revenue' in fundamental_data.columns:
                revenue_positive = fundamental_data['Revenue'] > 0
                capex_intensity = pd.Series(index=fundamental_data.index, dtype=float)
                capex_intensity[revenue_positive] = fundamental_data['CapEx'][revenue_positive] / fundamental_data['Revenue'][revenue_positive]
                metrics['historical_capex_intensity'] = capex_intensity.clip(0, 1)
            
        except Exception as e:
            logger.error(f"Error calculating historical metrics: {e}")
        
        return metrics
    
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
        """Main method to collect all data including DCF-specific data"""
        logger.info(f"Starting enhanced data collection for {self.ticker}")
        
        results = {
            'fundamental_data': None,
            'macro_data': None,
            'price_data': None,
            'dcf_additional': None,
            'working_capital': None,
            'historical_metrics': None
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
        
        # Fetch DCF-specific data
        logger.info("Collecting additional DCF data...")
        results['dcf_additional'] = self.fetch_additional_dcf_data()
        results['working_capital'] = self.fetch_working_capital_data()
        
        # Calculate historical metrics
        if not combined_fundamentals.empty:
            results['historical_metrics'] = self.calculate_historical_metrics(combined_fundamentals)
        
        # Calculate debt-to-equity ratio if we have the data
        if (not results['fundamental_data'].empty and 
            'TotalDebt' in results['fundamental_data'].columns and
            not results['dcf_additional'].empty):
            
            latest_debt = results['fundamental_data']['TotalDebt'].iloc[-1]
            
            if 'shares_outstanding' in results['dcf_additional'].columns:
                shares = results['dcf_additional']['shares_outstanding'].iloc[0]
                current_price = results['dcf_additional']['current_price'].iloc[0] if 'current_price' in results['dcf_additional'].columns else None
                
                if shares and current_price and not pd.isna(shares) and not pd.isna(current_price):
                    market_cap = shares * current_price
                    debt_to_equity = latest_debt / market_cap if market_cap > 0 else 0
                    results['dcf_additional'].loc[results['dcf_additional'].index[0], 'debt_to_equity'] = debt_to_equity
                    logger.info(f"Calculated debt-to-equity ratio: {debt_to_equity:.2f}")
        
        # Create summary with DCF data
        summary = {
            'ticker': self.ticker,
            'collection_date': datetime.now().isoformat(),
            'data_start': self.start_date.isoformat(),
            'data_end': self.end_date.isoformat(),
            'fundamental_records': len(combined_fundamentals) if combined_fundamentals is not None else 0,
            'sec_records': len(sec_data) if sec_data is not None else 0,
            'yf_records': len(yf_data) if yf_data is not None else 0,
            'macro_records': len(results['macro_data']) if results['macro_data'] is not None else 0,
            'price_records': len(results['price_data']) if results['price_data'] is not None else 0,
            'dcf_data_available': {
                'shares_outstanding': not results['dcf_additional'].empty and 'shares_outstanding' in results['dcf_additional'].columns,
                'beta': not results['dcf_additional'].empty and 'beta' in results['dcf_additional'].columns,
                'current_price': not results['dcf_additional'].empty and 'current_price' in results['dcf_additional'].columns,
                'working_capital': not results['working_capital'].empty,
                'historical_metrics': not results['historical_metrics'].empty
            }
        }
        
        # Add DCF data quality to summary
        if not results['historical_metrics'].empty:
            summary['dcf_metrics_available'] = list(results['historical_metrics'].columns)
        
        summary_path = os.path.join(self.ticker_dir, 'collection_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Enhanced data collection complete for {self.ticker}")
        logger.info(f"Summary saved to {summary_path}")
        
        return results
    
    def create_dcf_ready_dataset(self) -> pd.DataFrame:
        """Create a consolidated dataset ready for DCF analysis"""
        # Collect all data
        all_data = self.collect_all_data()
        
        if all_data['fundamental_data'].empty:
            logger.error("No fundamental data available for DCF dataset")
            return pd.DataFrame()
        
        # Start with fundamental data as base
        dcf_dataset = all_data['fundamental_data'].copy()
        
        # Add historical metrics
        if not all_data['historical_metrics'].empty:
            for col in all_data['historical_metrics'].columns:
                dcf_dataset[col] = all_data['historical_metrics'][col]
        
        # Add latest DCF additional data as columns (not time series)
        if not all_data['dcf_additional'].empty:
            for col in all_data['dcf_additional'].columns:
                # Use the latest value for each metric
                latest_value = all_data['dcf_additional'][col].iloc[0]
                dcf_dataset[f'latest_{col}'] = latest_value
        
        # Add working capital metrics
        if not all_data['working_capital'].empty and 'operating_working_capital' in all_data['working_capital'].columns:
            # Align working capital with fundamental data dates
            wc_aligned = all_data['working_capital']['operating_working_capital'].reindex(
                dcf_dataset.index, method='nearest'
            )
            dcf_dataset['working_capital'] = wc_aligned
        
        # Calculate additional DCF metrics
        if 'Revenue' in dcf_dataset.columns:
            # TTM Revenue (Trailing Twelve Months)
            dcf_dataset['revenue_ttm'] = dcf_dataset['Revenue'].rolling(4, min_periods=1).sum()
        
        if 'EBIT' in dcf_dataset.columns:
            # TTM EBIT
            dcf_dataset['ebit_ttm'] = dcf_dataset['EBIT'].rolling(4, min_periods=1).sum()
        
        # Save DCF-ready dataset
        dcf_ready_path = os.path.join(self.ticker_dir, 'dcf_ready_dataset.csv')
        dcf_dataset.to_csv(dcf_ready_path)
        logger.info(f"Created DCF-ready dataset with {len(dcf_dataset.columns)} columns")
        logger.info(f"Saved to {dcf_ready_path}")
        
        return dcf_dataset


def main():
    """Example usage with enhanced DCF data collection"""
    # IMPORTANT: Update the User-Agent email in fetch_sec_fundamental_data() before running!
    
    # Collect data for CMG
    collector = DataCollector('CMG', years_back=10)
    data = collector.collect_all_data()
    
    # Display enhanced summary
    print("\n=== Enhanced Data Collection Summary ===")
    for key, df in data.items():
        if df is not None and not df.empty:
            print(f"\n{key}:")
            print(f"  Shape: {df.shape}")
            if hasattr(df.index, 'min'):
                print(f"  Date range: {df.index.min()} to {df.index.max()}")
            print(f"  Columns: {list(df.columns)}")
            
            # Special handling for DCF additional data
            if key == 'dcf_additional':
                print(f"  DCF Data Points:")
                for col in df.columns:
                    value = df[col].iloc[0] if not df[col].isna().all() else 'N/A'
                    print(f"    - {col}: {value}")
            else:
                print(f"  Sample data:")
                print(df.head())
        else:
            print(f"\n{key}: No data collected")
    
    # Create and display DCF-ready dataset
    print("\n=== Creating DCF-Ready Dataset ===")
    dcf_dataset = collector.create_dcf_ready_dataset()
    
    if not dcf_dataset.empty:
        print(f"\nDCF Dataset Shape: {dcf_dataset.shape}")
        print(f"Available DCF Inputs:")
        dcf_columns = [col for col in dcf_dataset.columns if any(
            x in col for x in ['beta', 'tax_rate', 'cost_of_debt', 'shares', 'working_capital']
        )]
        for col in dcf_columns:
            print(f"  - {col}")
        
        # Check latest values for key DCF inputs
        latest_idx = dcf_dataset.index[-1]
        print(f"\nLatest DCF Input Values ({latest_idx.date()}):")
        if 'latest_beta' in dcf_dataset.columns:
            print(f"  Beta: {dcf_dataset.loc[latest_idx, 'latest_beta']:.2f}")
        if 'effective_tax_rate' in dcf_dataset.columns:
            latest_tax = dcf_dataset['effective_tax_rate'].dropna().iloc[-1] if not dcf_dataset['effective_tax_rate'].isna().all() else None
            if latest_tax:
                print(f"  Effective Tax Rate: {latest_tax:.1%}")
        if 'implied_cost_of_debt' in dcf_dataset.columns:
            latest_cod = dcf_dataset['implied_cost_of_debt'].dropna().iloc[-1] if not dcf_dataset['implied_cost_of_debt'].isna().all() else None
            if latest_cod:
                print(f"  Implied Cost of Debt: {latest_cod:.1%}")
        if 'wc_pct_revenue' in dcf_dataset.columns:
            latest_wc = dcf_dataset['wc_pct_revenue'].dropna().iloc[-1] if not dcf_dataset['wc_pct_revenue'].isna().all() else None
            if latest_wc:
                print(f"  Working Capital % Revenue: {latest_wc:.1%}")


if __name__ == "__main__":
    main()