
"""
Data utilities for fetching Nifty 50 stock prices.

Note: Tata Motors underwent a demerger in 2023-2024:
- TATAMOTORS.NS: Original combined entity (may have limited recent data)
- TMPV.NS: Tata Motors Passenger Vehicles Limited (post-demerger)
- TMCV.NS: Tata Motors Commercial Vehicles (post-demerger)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Nifty 50 stocks (as of recent composition)
# Note: Post-demerger, TMPV.NS replaced TATAMOTORS.NS in some indices
NIFTY50_LIST = [
    ("ADANIPORTS.NS", "Adani Ports"),
    ("ASIANPAINT.NS", "Asian Paints"),
    ("AXISBANK.NS", "Axis Bank"),
    ("BAJAJ-AUTO.NS", "Bajaj Auto"),
    ("BAJFINANCE.NS", "Bajaj Finance"),
    ("BAJAJFINSV.NS", "Bajaj Finserv"),
    ("BPCL.NS", "BPCL"),
    ("BHARTIARTL.NS", "Bharti Airtel"),
    ("BRITANNIA.NS", "Britannia"),
    ("CIPLA.NS", "Cipla"),
    ("COALINDIA.NS", "Coal India"),
    ("DIVISLAB.NS", "Divi's Labs"),
    ("DRREDDY.NS", "Dr. Reddy's"),
    ("EICHERMOT.NS", "Eicher Motors"),
    ("GRASIM.NS", "Grasim"),
    ("HCLTECH.NS", "HCL Tech"),
    ("HDFCBANK.NS", "HDFC Bank"),
    ("HDFCLIFE.NS", "HDFC Life"),
    ("HEROMOTOCO.NS", "Hero MotoCorp"),
    ("HINDALCO.NS", "Hindalco"),
    ("HINDUNILVR.NS", "Hindustan Unilever"),
    ("ICICIBANK.NS", "ICICI Bank"),
    ("ITC.NS", "ITC"),
    ("INDUSINDBK.NS", "IndusInd Bank"),
    ("INFY.NS", "Infosys"),
    ("JSWSTEEL.NS", "JSW Steel"),
    ("KOTAKBANK.NS", "Kotak Bank"),
    ("LT.NS", "L&T"),
    ("M&M.NS", "M&M"),
    ("MARUTI.NS", "Maruti Suzuki"),
    ("NTPC.NS", "NTPC"),
    ("NESTLEIND.NS", "Nestle India"),
    ("ONGC.NS", "ONGC"),
    ("POWERGRID.NS", "Power Grid"),
    ("RELIANCE.NS", "Reliance"),
    ("SBILIFE.NS", "SBI Life"),
    ("SBIN.NS", "SBI"),
    ("SUNPHARMA.NS", "Sun Pharma"),
    ("TCS.NS", "TCS"),
    ("TATACONSUM.NS", "Tata Consumer"),
    ("TMPV.NS", "Tata Motors PV"),  # Post-demerger entity
    ("TATASTEEL.NS", "Tata Steel"),
    ("TECHM.NS", "Tech Mahindra"),
    ("TITAN.NS", "Titan"),
    ("ULTRACEMCO.NS", "UltraTech Cement"),
    ("UPL.NS", "UPL"),
    ("WIPRO.NS", "Wipro"),
]


def fetch_prices(ticker: str, years: float = 3.0) -> pd.Series:
    """
    Fetch historical adjusted close prices for a given ticker.
    
    Parameters:
    -----------
    ticker : str
        Yahoo Finance ticker symbol (e.g., '^NSEI', 'RELIANCE.NS')
    years : float
        Number of years of historical data to fetch
    
    Returns:
    --------
    pd.Series
        Time series of adjusted close prices with DatetimeIndex
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(years * 365))
    
    # Try multiple approaches
    attempts = [
        {'auto_adjust': True, 'actions': False},
        {'auto_adjust': False, 'actions': False},
        {'auto_adjust': True, 'actions': True},
    ]
    
    for attempt_num, params in enumerate(attempts, 1):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                **params
            )
            
            if data.empty:
                if attempt_num < len(attempts):
                    continue  # Try next approach
                else:
                    # Try with shorter period
                    shorter_start = end_date - timedelta(days=int(years * 365 / 2))
                    data = yf.download(
                        ticker,
                        start=shorter_start,
                        end=end_date,
                        progress=False,
                        auto_adjust=True
                    )
                    if data.empty:
                        raise ValueError(f"No data available for {ticker}. Please try a different stock or check the ticker symbol.")
            
            # Successfully got data, now extract prices
            # Handle both single and multi-column DataFrames
            if 'Close' in data.columns:
                prices = data['Close']
            elif 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif len(data.columns) == 1:
                prices = data.iloc[:, 0]
            else:
                # Last resort - take first column
                prices = data.iloc[:, 0]
            
            # Ensure it's a Series with proper name
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]
            
            prices = prices.dropna()
            
            # Validate we have enough data
            if len(prices) < 50:
                raise ValueError(f"Insufficient data for {ticker}. Only {len(prices)} data points available. Need at least 50.")
            
            prices.name = ticker
            return prices
            
        except Exception as e:
            if attempt_num == len(attempts):
                # Last attempt failed
                error_msg = str(e)
                if "No data" in error_msg or "insufficient" in error_msg.lower():
                    raise Exception(f"Could not fetch data for {ticker}. This may be due to:\n"
                                  f"• Ticker symbol may be delisted or incorrect\n"
                                  f"• Yahoo Finance API temporary issues\n"
                                  f"• Data not available for the selected time period\n"
                                  f"Please try: (1) Different stock, (2) Shorter time period, (3) Try again later")
                else:
                    raise Exception(f"Error fetching data for {ticker}: {error_msg}")
            # Try next attempt
            continue
    
    # Should never reach here, but just in case
    raise Exception(f"Failed to fetch data for {ticker} after {len(attempts)} attempts")
