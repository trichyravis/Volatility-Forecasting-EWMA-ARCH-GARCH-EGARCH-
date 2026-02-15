"""Shared data: Nifty 50 list and price fetch for Streamlit app."""

from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# Nifty 50 constituents (NSE; Yahoo Finance: .NS). Format: (ticker, company_name)
NIFTY50_LIST = [
    ("ADANIPORTS.NS", "Adani Ports and SEZ"),
    ("ASIANPAINT.NS", "Asian Paints"),
    ("AXISBANK.NS", "Axis Bank"),
    ("BAJAJ-AUTO.NS", "Bajaj Auto"),
    ("BAJFINANCE.NS", "Bajaj Finance"),
    ("BAJAJFINSV.NS", "Bajaj Finserv"),
    ("BHARTIARTL.NS", "Bharti Airtel"),
    ("BPCL.NS", "Bharat Petroleum"),
    ("BRITANNIA.NS", "Britannia Industries"),
    ("CIPLA.NS", "Cipla"),
    ("COALINDIA.NS", "Coal India"),
    ("DIVISLAB.NS", "Divi's Laboratories"),
    ("DRREDDY.NS", "Dr. Reddy's Laboratories"),
    ("EICHERMOT.NS", "Eicher Motors"),
    ("GRASIM.NS", "Grasim Industries"),
    ("HCLTECH.NS", "HCL Technologies"),
    ("HDFCBANK.NS", "HDFC Bank"),
    ("HDFCLIFE.NS", "HDFC Life Insurance"),
    ("HEROMOTOCO.NS", "Hero MotoCorp"),
    ("HINDALCO.NS", "Hindalco Industries"),
    ("HINDUNILVR.NS", "Hindustan Unilever"),
    ("ICICIBANK.NS", "ICICI Bank"),
    ("INDUSINDBK.NS", "IndusInd Bank"),
    ("INFY.NS", "Infosys"),
    ("ITC.NS", "ITC"),
    ("JSWSTEEL.NS", "JSW Steel"),
    ("KOTAKBANK.NS", "Kotak Mahindra Bank"),
    ("LT.NS", "Larsen & Toubro"),
    ("M&M.NS", "Mahindra & Mahindra"),
    ("MARUTI.NS", "Maruti Suzuki"),
    ("NESTLEIND.NS", "Nestle India"),
    ("NTPC.NS", "NTPC"),
    ("ONGC.NS", "Oil and Natural Gas"),
    ("POWERGRID.NS", "Power Grid Corporation"),
    ("RELIANCE.NS", "Reliance Industries"),
    ("SBILIFE.NS", "SBI Life Insurance"),
    ("SBIN.NS", "State Bank of India"),
    ("SUNPHARMA.NS", "Sun Pharmaceutical"),
    ("TATAMOTORS.NS", "Tata Motors"),
    ("TATASTEEL.NS", "Tata Steel"),
    ("TCS.NS", "Tata Consultancy Services"),
    ("TECHM.NS", "Tech Mahindra"),
    ("TITAN.NS", "Titan Company"),
    ("ULTRACEMCO.NS", "UltraTech Cement"),
    ("WIPRO.NS", "Wipro"),
    ("APOLLOHOSP.NS", "Apollo Hospitals"),
    ("LTIM.NS", "LTIMindtree"),
    ("ADANIENT.NS", "Adani Enterprises"),
    ("SBICARD.NS", "SBI Cards"),
    ("UPL.NS", "UPL"),
]


def fetch_prices(ticker: str, years: float) -> pd.Series:
    """Download adjusted close prices from Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError(f"No data for ticker {ticker!r}.")
    return hist["Close"]
