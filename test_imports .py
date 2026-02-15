
"""
Debug script to verify data caching and refresh behavior
"""

import streamlit as st
import pandas as pd
from data_utils import NIFTY50_LIST, fetch_prices
from evt_model import get_returns, run_evt_var_es

st.title("🔍 Cache Behavior Test")

# Sidebar
ticker_options = ["^NSEI (Nifty 50 Index)"] + [
    f"{name} ({ticker})" for ticker, name in NIFTY50_LIST[:5]  # Only first 5 for testing
]
ticker_display = st.sidebar.selectbox("Select stock", ticker_options, index=0)
if ticker_display.startswith("^"):
    TICKER = "^NSEI"
else:
    TICKER = ticker_display.split("(")[-1].rstrip(")")

years = st.sidebar.slider("Years", 1.0, 3.0, 2.0, 0.5)
evt_alpha = st.sidebar.slider("Confidence", 0.90, 0.99, 0.95, 0.01)
evt_threshold = st.sidebar.slider("Threshold", 0.85, 0.95, 0.90, 0.01)

# Show current parameters
st.write("### Current Parameters")
st.write(f"- **Ticker:** {TICKER}")
st.write(f"- **Years:** {years}")
st.write(f"- **Alpha:** {evt_alpha}")
st.write(f"- **Threshold:** {evt_threshold}")

# Load data WITHOUT underscore prefix (should cache based on parameters)
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_data(ticker: str, years: float):
    st.write(f"🔄 **CACHE MISS** - Fetching fresh data for {ticker}")
    prices = fetch_prices(ticker, years)
    returns = get_returns(prices)
    return prices, returns.dropna()

# Load and display
try:
    prices, returns = load_data(TICKER, years)
    n_returns = len(returns)
    
    st.success(f"✅ Data loaded: {n_returns:,} returns")
    
    # Run EVT
    with st.spinner("Calculating EVT..."):
        result = run_evt_var_es(returns, alpha=evt_alpha, threshold_quantile=evt_threshold)
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("VaR", f"{result['var']*100:.2f}%")
    col2.metric("ES", f"{result['es']*100:.2f}%")
    col3.metric("GPD ξ", f"{result['xi']:.4f}")
    col4.metric("Exceedances", result['n_exceedances'])
    
    st.write("### Debug Info")
    st.write(f"- Returns shape: {returns.shape}")
    st.write(f"- First date: {returns.index[0]}")
    st.write(f"- Last date: {returns.index[-1]}")
    
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())

st.write("---")
st.info("""
**Test Instructions:**
1. Change the ticker → Should see 'CACHE MISS' and new VaR values
2. Change years → Should see 'CACHE MISS' and new VaR values
3. Change only alpha/threshold → Should NOT see 'CACHE MISS', but VaR should update
4. Change back to same ticker/years → Should use cached data (no 'CACHE MISS')
""")
