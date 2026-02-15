
# Nifty 50 VaR, Expected Shortfall & Volatility Analysis

A comprehensive Streamlit application for analyzing Value at Risk (VaR), Expected Shortfall (ES), and volatility for Nifty 50 index and constituent stocks using advanced risk management models.

## Features

### 1. Extreme Value Theory (EVT) Analysis
- **Generalized Pareto Distribution (GPD)** fitting for tail risk modeling
- **VaR and Expected Shortfall** estimation at user-defined confidence levels
- **Threshold selection** with adjustable quantile parameter
- Visual analysis of loss distributions and tail behavior

### 2. Volatility Models
- **EWMA (Exponentially Weighted Moving Average)** - Simple but effective volatility estimation
- **ARCH(1)** - Autoregressive Conditional Heteroskedasticity model
- **GARCH(1,1)** - Generalized ARCH with mean reversion
- **EGARCH(1,1)** - Exponential GARCH capturing leverage effects

### 3. Risk Metrics
- **Conditional VaR** from each volatility model
- **Expected Shortfall (CVaR)** for coherent risk measurement
- **Multi-step forecasting** (1-10 days ahead)
- **Annualized volatility** projections

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

### Sidebar Controls

#### Stock/Index Selection
- Choose from Nifty 50 Index or any of the 47 constituent stocks
- Data is fetched automatically from Yahoo Finance

#### Historical Data Period
- Adjust the slider to select 1-5 years of historical data
- More data provides better statistical estimates but may include outdated market regimes

#### EVT Settings
- **Confidence Level (α)**: Set the confidence level for VaR/ES (90%-99%)
  - 95% means you expect losses to exceed VaR only 5% of the time
- **Threshold Quantile**: Controls how much of the tail to model with GPD (85%-95%)
  - Higher values = fewer exceedances but better focus on extreme events

#### Volatility Model Settings
- **EWMA λ (Lambda)**: Decay parameter for EWMA (0.90-0.98)
  - Higher λ = more weight on recent observations
  - RiskMetrics standard: 0.94
- **Forecast Horizon**: Number of days to forecast ahead (1-10)
- **VaR/ES Confidence**: Confidence level for volatility-based risk metrics (90%-99%)

### Tabs

#### Tab 1: EVT VaR & Expected Shortfall
**Displays:**
- **Key Metrics Cards:**
  - VaR (% loss) at specified confidence level
  - Expected Shortfall (average loss beyond VaR)
  - GPD shape parameter (ξ) - indicates tail heaviness
  - Threshold value and number of exceedances

- **Visualizations:**
  - **Loss Distribution**: Histogram with threshold, VaR, and ES markers
  - **Tail Analysis**: Exceedances with fitted GPD overlay

**Interpretation:**
- **ξ > 0**: Heavy-tailed distribution (common for financial assets)
- **ξ = 0**: Exponential tail (moderate risk)
- **ξ < 0**: Finite tail (bounded risk)

#### Tab 2: Volatility & VaR/ES
**Displays:**
- **Volatility Forecast Table**: Annualized volatility predictions for each model
- **VaR Forecast Table**: Predicted VaR values for forecast horizon

- **Charts:**
  - **Returns & Conditional Volatility**: Time series showing:
    - Daily returns
    - In-sample volatility from all four models
    - Out-of-sample volatility forecasts
  
  - **VaR & ES Analysis**: Comprehensive view of:
    - Historical VaR from each model
    - Historical ES from each model
    - Forecasted VaR (1-10 days ahead)
    - Forecasted ES (1-10 days ahead)

## Model Details

### Extreme Value Theory (EVT)
EVT focuses on modeling the tail of the return distribution, which is crucial for risk management:

1. **Threshold Selection**: Choose a high quantile (e.g., 90th percentile)
2. **Exceedances**: Extract losses exceeding the threshold
3. **GPD Fitting**: Fit Generalized Pareto Distribution to exceedances
4. **VaR Calculation**: 
   - VaR_α = u + (σ/ξ) * [(1/(1-α_adj))^ξ - 1]
   - where u = threshold, σ = scale, ξ = shape
5. **ES Calculation**: 
   - ES_α = (VaR_α + σ - ξ*u) / (1 - ξ)

### Volatility Models

#### EWMA
σ²_t = λ * σ²_(t-1) + (1-λ) * r²_(t-1)

- Simple recursive formula
- No parameter estimation required
- Exponentially declining weights on past returns

#### ARCH(1)
σ²_t = ω + α₁ * r²_(t-1)

- Captures volatility clustering
- One lagged squared return

#### GARCH(1,1)
σ²_t = ω + α₁ * r²_(t-1) + β₁ * σ²_(t-1)

- Includes lagged variance (mean reversion)
- Most widely used volatility model
- Parsimonious yet effective

#### EGARCH(1,1)
log(σ²_t) = ω + β₁ * log(σ²_(t-1)) + α₁ * |z_(t-1)| + γ * z_(t-1)

- Logarithmic form ensures positive variance
- Captures leverage effect (γ < 0)
- Asymmetric response to positive/negative shocks

## Key Issues Fixed

### Issues in Original Code
1. **Metric Help Parameter**: The third metric (GPD shape) incorrectly had `None` for delta parameter
2. **Axes Sharing**: Removed unnecessary `sharex` that caused layout issues
3. **Error Handling**: Improved exception handling with traceback for debugging

### Corrections Made
1. Removed `None` parameter from `col3.metric()` call
2. Removed `axes1[1].sharex(axes1[0])` line that was redundant
3. Removed `axes2[1].sharex(axes2[0])` line for cleaner subplot layout
4. Added comprehensive error handling throughout

## Technical Notes

### Data Source
- **Yahoo Finance** via `yfinance` library
- Adjusted close prices for corporate actions
- Daily frequency (252 trading days per year)

### Assumptions
- **Log Returns**: ln(P_t / P_(t-1))
- **Normal Distribution**: For VaR/ES from volatility models
- **Stationarity**: Returns assumed stationary over sample period
- **No Transaction Costs**: Risk metrics are before costs

### Limitations
- Models assume no structural breaks
- Historical data may not predict future crises
- GPD assumes tail behavior is stable
- Normal distribution assumption may underestimate tail risk

## Example Use Cases

### 1. Portfolio Risk Manager
- Monitor daily VaR for position limits
- Compare multiple volatility models
- Stress test with different confidence levels

### 2. Academic Research
- Compare EVT vs. parametric VaR methods
- Study volatility clustering in Indian markets
- Analyze tail behavior of sectoral indices

### 3. Regulatory Compliance
- Calculate VaR for Basel capital requirements
- Backtesting risk models
- Scenario analysis for stress testing

## Contributing

Feel free to submit issues, feature requests, or pull requests.

## License

This project is provided for educational and research purposes.

## Author

Prof. V. Ravichandran  
28+ Years Corporate Finance & Banking Experience  
10+ Years Academic Excellence  

**The Mountain Path - World of Finance**

## References

1. McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management*
2. Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity
3. Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity
4. Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: EGARCH
5. RiskMetrics Technical Document (1996)
