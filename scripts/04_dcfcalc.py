"""
Script 4: DCF Calculator Core
Modular DCF calculator with sensitivity testing capabilities
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DCFInputs:
    """Data class to hold DCF inputs with validation"""
    # Revenue growth rates (years 1-5)
    revenue_growth_y1: float
    revenue_growth_y2: float
    revenue_growth_y3: float
    revenue_growth_y4: float
    revenue_growth_y5: float
    
    # EBIT margin progression (years 1-5)
    ebit_margin_y1: float
    ebit_margin_y2: float
    ebit_margin_y3: float
    ebit_margin_y4: float
    ebit_margin_y5: float
    
    # Other assumptions
    capex_pct_revenue: float  # Constant across forecast period
    tax_rate: float
    working_capital_pct_revenue: float
    terminal_growth_rate: float
    
    # WACC components
    risk_free_rate: float
    equity_risk_premium: float
    beta: float
    cost_of_debt: float
    debt_to_equity: float
    
    # Base year values
    base_revenue: float
    base_shares_outstanding: float
    
    def __post_init__(self):
        """Validate inputs after initialization"""
        # Validate growth rates
        for i in range(1, 6):
            growth = getattr(self, f'revenue_growth_y{i}')
            if not -0.5 <= growth <= 2.0:
                raise ValueError(f"Revenue growth year {i} ({growth}) outside reasonable range [-50%, 200%]")
        
        # Validate margins
        for i in range(1, 6):
            margin = getattr(self, f'ebit_margin_y{i}')
            if not -0.5 <= margin <= 0.5:
                raise ValueError(f"EBIT margin year {i} ({margin}) outside reasonable range [-50%, 50%]")
        
        # Validate other ratios
        if not 0 <= self.capex_pct_revenue <= 0.5:
            raise ValueError(f"CapEx % of revenue ({self.capex_pct_revenue}) outside range [0%, 50%]")
        
        if not 0 <= self.tax_rate <= 0.5:
            raise ValueError(f"Tax rate ({self.tax_rate}) outside range [0%, 50%]")
        
        if not -0.02 <= self.terminal_growth_rate <= 0.05:
            raise ValueError(f"Terminal growth rate ({self.terminal_growth_rate}) outside range [-2%, 5%]")
        
        # Validate WACC components
        if not 0 <= self.risk_free_rate <= 0.1:
            raise ValueError(f"Risk-free rate ({self.risk_free_rate}) outside range [0%, 10%]")
        
        if not 0.03 <= self.equity_risk_premium <= 0.15:
            raise ValueError(f"Equity risk premium ({self.equity_risk_premium}) outside range [3%, 15%]")
        
        if not 0.2 <= self.beta <= 3.0:
            raise ValueError(f"Beta ({self.beta}) outside range [0.2, 3.0]")


class DCFCalculator:
    """Modular DCF calculator for valuation analysis"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.base_dir = "dcf_data"
        self.ticker_dir = os.path.join(self.base_dir, self.ticker)
        self.dcf_dir = os.path.join(self.ticker_dir, "dcf_analysis")
        self.reports_dir = os.path.join(self.ticker_dir, "dcf_reports")
        self._create_directories()
        
        # DCF parameters
        self.forecast_years = 5
        self.mid_year_convention = True  # Discount cash flows from mid-year
        
    def _create_directories(self):
        """Create folder structure"""
        for directory in [self.dcf_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def calculate_wacc(self, inputs: DCFInputs) -> float:
        """Calculate Weighted Average Cost of Capital"""
        # Cost of equity using CAPM
        cost_of_equity = inputs.risk_free_rate + inputs.beta * inputs.equity_risk_premium
        
        # Weights
        total_value = 1 + inputs.debt_to_equity
        equity_weight = 1 / total_value
        debt_weight = inputs.debt_to_equity / total_value
        
        # After-tax cost of debt
        after_tax_cost_of_debt = inputs.cost_of_debt * (1 - inputs.tax_rate)
        
        # WACC
        wacc = (equity_weight * cost_of_equity) + (debt_weight * after_tax_cost_of_debt)
        
        logger.info(f"WACC calculated: {wacc:.2%}")
        logger.info(f"  Cost of Equity: {cost_of_equity:.2%}")
        logger.info(f"  After-tax Cost of Debt: {after_tax_cost_of_debt:.2%}")
        logger.info(f"  Equity Weight: {equity_weight:.2%}")
        
        return wacc
    
    def project_financials(self, inputs: DCFInputs) -> pd.DataFrame:
        """Project financial statements for DCF"""
        projections = {
            'Revenue': [inputs.base_revenue],
            'EBIT': [],
            'Tax': [],
            'NOPAT': [],
            'CapEx': [],
            'Working_Capital_Change': [],
            'FCFF': []
        }
        
        # Track working capital
        prev_working_capital = inputs.base_revenue * inputs.working_capital_pct_revenue
        
        # Project each year
        for year in range(1, self.forecast_years + 1):
            # Revenue
            revenue_growth = getattr(inputs, f'revenue_growth_y{year}')
            revenue = projections['Revenue'][-1] * (1 + revenue_growth)
            projections['Revenue'].append(revenue)
            
            # EBIT
            ebit_margin = getattr(inputs, f'ebit_margin_y{year}')
            ebit = revenue * ebit_margin
            projections['EBIT'].append(ebit)
            
            # Tax
            tax = ebit * inputs.tax_rate if ebit > 0 else 0
            projections['Tax'].append(tax)
            
            # NOPAT (Net Operating Profit After Tax)
            nopat = ebit - tax
            projections['NOPAT'].append(nopat)
            
            # CapEx
            capex = revenue * inputs.capex_pct_revenue
            projections['CapEx'].append(capex)
            
            # Working Capital Change
            working_capital = revenue * inputs.working_capital_pct_revenue
            wc_change = working_capital - prev_working_capital
            projections['Working_Capital_Change'].append(wc_change)
            prev_working_capital = working_capital
            
            # Free Cash Flow to Firm
            fcff = nopat - capex - wc_change
            projections['FCFF'].append(fcff)
        
        # Remove base revenue (year 0) and create DataFrame
        projections['Revenue'] = projections['Revenue'][1:]
        
        df_projections = pd.DataFrame(projections)
        df_projections.index = [f'Year_{i}' for i in range(1, self.forecast_years + 1)]
        
        return df_projections
    
    def calculate_terminal_value(self, final_fcff: float, terminal_growth: float, wacc: float) -> float:
        """Calculate terminal value using Gordon Growth Model"""
        if wacc <= terminal_growth:
            raise ValueError(f"WACC ({wacc:.2%}) must be greater than terminal growth ({terminal_growth:.2%})")
        
        # Terminal value at end of forecast period
        terminal_value = final_fcff * (1 + terminal_growth) / (wacc - terminal_growth)
        
        return terminal_value
    
    def calculate_dcf_value(self, inputs: DCFInputs) -> Dict[str, Any]:
        """Calculate DCF value with detailed breakdown"""
        # Calculate WACC
        wacc = self.calculate_wacc(inputs)
        
        # Project financials
        projections = self.project_financials(inputs)
        
        # Calculate present value of cash flows
        pv_fcff = []
        for year in range(1, self.forecast_years + 1):
            fcff = projections.loc[f'Year_{year}', 'FCFF']
            
            # Discount factor (with mid-year convention if enabled)
            if self.mid_year_convention:
                discount_factor = 1 / ((1 + wacc) ** (year - 0.5))
            else:
                discount_factor = 1 / ((1 + wacc) ** year)
            
            pv = fcff * discount_factor
            pv_fcff.append(pv)
        
        # Terminal value
        terminal_value = self.calculate_terminal_value(
            projections.loc[f'Year_{self.forecast_years}', 'FCFF'],
            inputs.terminal_growth_rate,
            wacc
        )
        
        # Present value of terminal value
        if self.mid_year_convention:
            terminal_discount_factor = 1 / ((1 + wacc) ** (self.forecast_years - 0.5))
        else:
            terminal_discount_factor = 1 / ((1 + wacc) ** self.forecast_years)
        
        pv_terminal_value = terminal_value * terminal_discount_factor
        
        # Enterprise value
        enterprise_value = sum(pv_fcff) + pv_terminal_value
        
        # Equity value per share (simplified - assuming net debt is handled externally)
        equity_value_per_share = enterprise_value / inputs.base_shares_outstanding
        
        # Prepare detailed results
        results = {
            'inputs': asdict(inputs),
            'wacc': wacc,
            'projections': projections.to_dict(),
            'present_values': {
                f'Year_{i+1}': pv for i, pv in enumerate(pv_fcff)
            },
            'terminal_value': terminal_value,
            'pv_terminal_value': pv_terminal_value,
            'pv_forecast_period': sum(pv_fcff),
            'enterprise_value': enterprise_value,
            'equity_value_per_share': equity_value_per_share,
            'terminal_value_pct': pv_terminal_value / enterprise_value,
            'calculation_date': datetime.now().isoformat()
        }
        
        return results
    
    def sensitivity_analysis(self, base_inputs: DCFInputs, 
                           sensitivity_params: Dict[str, List[float]]) -> pd.DataFrame:
        """Perform sensitivity analysis on DCF inputs"""
        results = []
        
        for param_name, param_values in sensitivity_params.items():
            base_value = getattr(base_inputs, param_name)
            
            for test_value in param_values:
                # Create modified inputs
                test_inputs_dict = asdict(base_inputs)
                test_inputs_dict[param_name] = test_value
                
                try:
                    test_inputs = DCFInputs(**test_inputs_dict)
                    dcf_result = self.calculate_dcf_value(test_inputs)
                    
                    results.append({
                        'parameter': param_name,
                        'base_value': base_value,
                        'test_value': test_value,
                        'value_per_share': dcf_result['equity_value_per_share'],
                        'enterprise_value': dcf_result['enterprise_value'],
                        'change_pct': (test_value - base_value) / base_value if base_value != 0 else 0
                    })
                except Exception as e:
                    logger.warning(f"Sensitivity test failed for {param_name}={test_value}: {e}")
        
        sensitivity_df = pd.DataFrame(results)
        return sensitivity_df
    
    def create_tornado_chart(self, base_inputs: DCFInputs, 
                           sensitivity_ranges: Dict[str, Tuple[float, float]]) -> plt.Figure:
        """Create tornado chart showing sensitivity of key parameters"""
        base_result = self.calculate_dcf_value(base_inputs)
        base_value = base_result['equity_value_per_share']
        
        tornado_data = []
        
        for param_name, (low_pct, high_pct) in sensitivity_ranges.items():
            base_param_value = getattr(base_inputs, param_name)
            
            # Calculate low and high values
            low_value = base_param_value * (1 + low_pct)
            high_value = base_param_value * (1 + high_pct)
            
            # Calculate DCF with low value
            low_inputs_dict = asdict(base_inputs)
            low_inputs_dict[param_name] = low_value
            try:
                low_inputs = DCFInputs(**low_inputs_dict)
                low_result = self.calculate_dcf_value(low_inputs)
                low_dcf = low_result['equity_value_per_share']
            except:
                low_dcf = base_value
            
            # Calculate DCF with high value
            high_inputs_dict = asdict(base_inputs)
            high_inputs_dict[param_name] = high_value
            try:
                high_inputs = DCFInputs(**high_inputs_dict)
                high_result = self.calculate_dcf_value(high_inputs)
                high_dcf = high_result['equity_value_per_share']
            except:
                high_dcf = base_value
            
            tornado_data.append({
                'parameter': param_name,
                'low_value': low_dcf,
                'high_value': high_dcf,
                'impact': abs(high_dcf - low_dcf)
            })
        
        # Sort by impact
        tornado_df = pd.DataFrame(tornado_data).sort_values('impact', ascending=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(tornado_df))
        
        # Plot bars
        for i, row in tornado_df.iterrows():
            low = row['low_value'] - base_value
            high = row['high_value'] - base_value
            
            # Plot from low to high
            if low < 0:
                ax.barh(i, -low, left=low, color='red', alpha=0.7)
            if high > 0:
                ax.barh(i, high, left=0, color='green', alpha=0.7)
        
        # Add base line
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tornado_df['parameter'])
        ax.set_xlabel(f'Change in Value per Share from Base (${base_value:.2f})')
        ax.set_title(f'Tornado Chart - DCF Sensitivity Analysis for {self.ticker}')
        
        # Add grid
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def scenario_analysis(self, scenarios: Dict[str, DCFInputs]) -> pd.DataFrame:
        """Run DCF for multiple scenarios"""
        results = []
        
        for scenario_name, inputs in scenarios.items():
            try:
                dcf_result = self.calculate_dcf_value(inputs)
                
                results.append({
                    'scenario': scenario_name,
                    'equity_value_per_share': dcf_result['equity_value_per_share'],
                    'enterprise_value': dcf_result['enterprise_value'],
                    'wacc': dcf_result['wacc'],
                    'terminal_value_pct': dcf_result['terminal_value_pct'],
                    'revenue_cagr': self._calculate_revenue_cagr(inputs),
                    'avg_ebit_margin': self._calculate_avg_margin(inputs),
                    'terminal_growth': inputs.terminal_growth_rate
                })
            except Exception as e:
                logger.error(f"Scenario {scenario_name} failed: {e}")
        
        scenario_df = pd.DataFrame(results)
        return scenario_df
    
    def _calculate_revenue_cagr(self, inputs: DCFInputs) -> float:
        """Calculate revenue CAGR over forecast period"""
        growth_rates = [getattr(inputs, f'revenue_growth_y{i}') for i in range(1, 6)]
        final_revenue_multiple = np.prod([1 + g for g in growth_rates])
        cagr = final_revenue_multiple ** (1/5) - 1
        return cagr
    
    def _calculate_avg_margin(self, inputs: DCFInputs) -> float:
        """Calculate average EBIT margin over forecast period"""
        margins = [getattr(inputs, f'ebit_margin_y{i}') for i in range(1, 6)]
        return np.mean(margins)
    
    def create_dcf_report(self, dcf_result: Dict[str, Any], 
                         sensitivity_df: Optional[pd.DataFrame] = None,
                         scenario_df: Optional[pd.DataFrame] = None):
        """Generate comprehensive DCF report"""
        report_path = os.path.join(self.reports_dir, f"dcf_report_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        html_content = f"""
        <html>
        <head>
            <title>DCF Analysis Report - {self.ticker}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .metric {{ font-weight: bold; color: #2ecc71; }}
                .section {{ margin: 30px 0; }}
                .key-result {{ font-size: 24px; font-weight: bold; color: #3498db; }}
            </style>
        </head>
        <body>
            <h1>DCF Analysis Report - {self.ticker}</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class='section'>
                <h2>Valuation Summary</h2>
                <p class='key-result'>Fair Value per Share: ${dcf_result['equity_value_per_share']:.2f}</p>
                <p>Enterprise Value: ${dcf_result['enterprise_value']:,.0f}</p>
                <p>WACC: {dcf_result['wacc']:.2%}</p>
                <p>Terminal Value as % of Total: {dcf_result['terminal_value_pct']:.1%}</p>
            </div>
            
            <div class='section'>
                <h2>Key Assumptions</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Revenue CAGR (5Y)</td><td>{self._calculate_revenue_cagr(DCFInputs(**dcf_result['inputs'])):.1%}</td></tr>
                    <tr><td>Average EBIT Margin</td><td>{self._calculate_avg_margin(DCFInputs(**dcf_result['inputs'])):.1%}</td></tr>
                    <tr><td>CapEx % of Revenue</td><td>{dcf_result['inputs']['capex_pct_revenue']:.1%}</td></tr>
                    <tr><td>Tax Rate</td><td>{dcf_result['inputs']['tax_rate']:.1%}</td></tr>
                    <tr><td>Terminal Growth Rate</td><td>{dcf_result['inputs']['terminal_growth_rate']:.1%}</td></tr>
                    <tr><td>Beta</td><td>{dcf_result['inputs']['beta']:.2f}</td></tr>
                    <tr><td>Risk-Free Rate</td><td>{dcf_result['inputs']['risk_free_rate']:.2%}</td></tr>
                    <tr><td>Equity Risk Premium</td><td>{dcf_result['inputs']['equity_risk_premium']:.2%}</td></tr>
                </table>
            </div>
            
            <div class='section'>
                <h2>Financial Projections</h2>
                <table>
                    <tr>
                        <th>Year</th>
        """
        
        # Add projection years
        for year in range(1, 6):
            html_content += f"<th>Year {year}</th>"
        html_content += "</tr>"
        
        # Add projection data
        projections = pd.DataFrame(dcf_result['projections'])
        for metric in ['Revenue', 'EBIT', 'FCFF']:
            html_content += f"<tr><td>{metric}</td>"
            for year in range(5):
                value = projections.iloc[year][metric]
                html_content += f"<td>${value:,.0f}</td>"
            html_content += "</tr>"
        
        html_content += "</table></div>"
        
        # Add scenario analysis if available
        if scenario_df is not None and not scenario_df.empty:
            html_content += """
            <div class='section'>
                <h2>Scenario Analysis</h2>
                <table>
                    <tr>
                        <th>Scenario</th>
                        <th>Value per Share</th>
                        <th>vs Base</th>
                        <th>Revenue CAGR</th>
                        <th>Avg EBIT Margin</th>
                    </tr>
            """
            
            base_value = scenario_df.loc[scenario_df['scenario'] == 'Base', 'equity_value_per_share'].iloc[0] if 'Base' in scenario_df['scenario'].values else scenario_df['equity_value_per_share'].iloc[0]
            
            for _, row in scenario_df.iterrows():
                change_pct = (row['equity_value_per_share'] - base_value) / base_value
                html_content += f"""
                    <tr>
                        <td>{row['scenario']}</td>
                        <td>${row['equity_value_per_share']:.2f}</td>
                        <td>{change_pct:+.1%}</td>
                        <td>{row['revenue_cagr']:.1%}</td>
                        <td>{row['avg_ebit_margin']:.1%}</td>
                    </tr>
                """
            
            html_content += "</table></div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"DCF report saved to {report_path}")
    
    def validate_against_current_price(self, dcf_value: float, current_price: float) -> Dict[str, float]:
        """Compare DCF value against current market price"""
        difference = dcf_value - current_price
        difference_pct = difference / current_price
        
        validation = {
            'dcf_value': dcf_value,
            'current_price': current_price,
            'difference': difference,
            'difference_pct': difference_pct,
            'implied_return': difference_pct,
            'valuation_status': 'Undervalued' if difference > 0 else 'Overvalued'
        }
        
        logger.info(f"DCF Value: ${dcf_value:.2f}")
        logger.info(f"Current Price: ${current_price:.2f}")
        logger.info(f"Difference: {difference_pct:.1%} ({validation['valuation_status']})")
        
        return validation


def create_example_inputs(ticker: str) -> DCFInputs:
    """Create example DCF inputs for testing"""
    # This would typically come from your ML models or manual input
    example_inputs = DCFInputs(
        # Revenue growth rates
        revenue_growth_y1=0.15,  # 15%
        revenue_growth_y2=0.12,  # 12%
        revenue_growth_y3=0.10,  # 10%
        revenue_growth_y4=0.08,  # 8%
        revenue_growth_y5=0.06,  # 6%
        
        # EBIT margins
        ebit_margin_y1=0.18,  # 18%
        ebit_margin_y2=0.19,  # 19%
        ebit_margin_y3=0.20,  # 20%
        ebit_margin_y4=0.20,  # 20%
        ebit_margin_y5=0.20,  # 20%
        
        # Other assumptions
        capex_pct_revenue=0.05,  # 5%
        tax_rate=0.21,  # 21%
        working_capital_pct_revenue=0.02,  # 2%
        terminal_growth_rate=0.025,  # 2.5%
        
        # WACC components
        risk_free_rate=0.04,  # 4%
        equity_risk_premium=0.065,  # 6.5%
        beta=1.1,
        cost_of_debt=0.035,  # 3.5%
        debt_to_equity=0.3,  # 30% D/E
        
        # Base values (these would come from actual data)
        base_revenue=3.5e9,  # $3.5B
        base_shares_outstanding=27.5e6  # 27.5M shares
    )
    
    return example_inputs


def main():
    """Example usage"""
    # Initialize calculator
    calculator = DCFCalculator('CMG')
    
    # Create example inputs
    base_inputs = create_example_inputs('CMG')
    
    # Run base case DCF
    logger.info("=== Running Base Case DCF ===")
    dcf_result = calculator.calculate_dcf_value(base_inputs)
    print(f"\nFair Value per Share: ${dcf_result['equity_value_per_share']:.2f}")
    print(f"Enterprise Value: ${dcf_result['enterprise_value']:,.0f}")
    print(f"WACC: {dcf_result['wacc']:.2%}")
    print(f"Terminal Value %: {dcf_result['terminal_value_pct']:.1%}")
    
    # Run sensitivity analysis
    logger.info("\n=== Running Sensitivity Analysis ===")
    sensitivity_params = {
        'terminal_growth_rate': [0.015, 0.02, 0.025, 0.03, 0.035],
        'wacc': [0.08, 0.09, 0.10, 0.11, 0.12],
        'revenue_growth_y1': [0.05, 0.10, 0.15, 0.20, 0.25]
    }
    
    # Create sensitivity ranges for tornado chart
    sensitivity_ranges = {
        'terminal_growth_rate': (-0.20, 0.20),  # ±20%
        'beta': (-0.20, 0.20),
        'revenue_growth_y1': (-0.30, 0.30),
        'ebit_margin_y3': (-0.20, 0.20),
        'capex_pct_revenue': (-0.30, 0.30),
        'equity_risk_premium': (-0.20, 0.20)
    }
    
    # Create tornado chart
    fig = calculator.create_tornado_chart(base_inputs, sensitivity_ranges)
    tornado_path = os.path.join(calculator.reports_dir, f"tornado_chart_{calculator.ticker}.png")
    fig.savefig(tornado_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Tornado chart saved to {tornado_path}")
    
    # Run scenario analysis
    logger.info("\n=== Running Scenario Analysis ===")
    
    # Create scenarios
    scenarios = {
        'Base': base_inputs,
        'Conservative': DCFInputs(
            revenue_growth_y1=0.08,
            revenue_growth_y2=0.06,
            revenue_growth_y3=0.05,
            revenue_growth_y4=0.04,
            revenue_growth_y5=0.03,
            ebit_margin_y1=0.15,
            ebit_margin_y2=0.15,
            ebit_margin_y3=0.16,
            ebit_margin_y4=0.16,
            ebit_margin_y5=0.16,
            capex_pct_revenue=0.06,
            tax_rate=0.25,
            working_capital_pct_revenue=0.03,
            terminal_growth_rate=0.02,
            risk_free_rate=0.045,
            equity_risk_premium=0.07,
            beta=1.2,
            cost_of_debt=0.04,
            debt_to_equity=0.4,
            base_revenue=3.5e9,
            base_shares_outstanding=27.5e6
        ),
        'Optimistic': DCFInputs(
            revenue_growth_y1=0.20,
            revenue_growth_y2=0.18,
            revenue_growth_y3=0.15,
            revenue_growth_y4=0.12,
            revenue_growth_y5=0.10,
            ebit_margin_y1=0.20,
            ebit_margin_y2=0.22,
            ebit_margin_y3=0.23,
            ebit_margin_y4=0.23,
            ebit_margin_y5=0.24,
            capex_pct_revenue=0.04,
            tax_rate=0.19,
            working_capital_pct_revenue=0.015,
            terminal_growth_rate=0.03,
            risk_free_rate=0.035,
            equity_risk_premium=0.06,
            beta=1.0,
            cost_of_debt=0.03,
            debt_to_equity=0.2,
            base_revenue=3.5e9,
            base_shares_outstanding=27.5e6
        )
    }
    
    scenario_df = calculator.scenario_analysis(scenarios)
    print("\n=== Scenario Analysis Results ===")
    print(scenario_df)
    
    # Generate comprehensive report
    calculator.create_dcf_report(dcf_result, scenario_df=scenario_df)
    
    # Save DCF results
    results_path = os.path.join(calculator.dcf_dir, f"dcf_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump(dcf_result, f, indent=2)
    logger.info(f"DCF results saved to {results_path}")
    
    # Validate against hypothetical current price
    current_price = 55.0  # Example current price
    validation = calculator.validate_against_current_price(
        dcf_result['equity_value_per_share'], 
        current_price
    )
    
    print(f"\n=== Valuation vs Market ===")
    print(f"DCF Value: ${validation['dcf_value']:.2f}")
    print(f"Current Price: ${validation['current_price']:.2f}")
    print(f"Implied Return: {validation['implied_return']:.1%}")
    print(f"Status: {validation['valuation_status']}")


if __name__ == "__main__":
    main()