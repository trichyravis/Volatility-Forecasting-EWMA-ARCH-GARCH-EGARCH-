
"""
Data utilities for fetching Nifty 50 stock prices.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Nifty 50 stocks (as of recent composition)
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
    ("TATAMOTORS.NS", "Tata Motors"),
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
    
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker}")
        
        # Handle both single and multi-column DataFrames
        if 'Close' in data.columns:
            prices = data['Close']
        elif len(data.columns) == 1:
            prices = data.iloc[:, 0]
        else:
            prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
        # Ensure it's a Series with proper name
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        
        prices.name = ticker
        return prices.dropna()
    
    except Exception as e:
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")
