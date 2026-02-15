
"""
Nifty 50 VaR, ES & Volatility Analysis Platform
The Mountain Path - World of Finance
Prof. V. Ravichandran
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from data_utils import NIFTY50_LIST, fetch_prices
from evt_model import get_returns, run_evt_var_es, losses_from_returns
from vol_models import run_volatility_pipeline

# ============================================================================
# BRANDING & STYLING
# ============================================================================
COLORS = {
    'dark_blue': '#003366',
    'medium_blue': '#004d80',
    'light_blue': '#ADD8E6',
    'accent_gold': '#FFD700',
    'bg_dark': '#0a1628',
    'card_bg': '#112240',
    'text_primary': '#e6f1ff',
    'text_secondary': '#8892b0',
    'success': '#28a745',
    'danger': '#dc3545',
}

BRANDING = {
    'name': 'The Mountain Path - World of Finance',
    'instructor': 'Prof. V. Ravichandran',
    'credentials': '28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence',
    'icon': '🏔️',
}

PAGE_CONFIG = {
    'page_title': 'Nifty 50 VaR & Volatility | Mountain Path',
    'page_icon': '🏔️',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
}


def apply_styles():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

        .stApp {{
            background: linear-gradient(135deg, {COLORS['bg_dark']} 0%, {COLORS['dark_blue']} 50%, #0d2137 100%);
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS['bg_dark']} 0%, {COLORS['dark_blue']} 100%);
            border-right: 1px solid rgba(255,215,0,0.2);
        }}

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stNumberInput label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label {{
            color: {COLORS['text_primary']} !important;
        }}

        section[data-testid="stSidebar"] input {{
            color: #1a1a2e !important;
            background-color: #ffffff !important;
        }}

        .header-container {{
            background: linear-gradient(135deg, {COLORS['dark_blue']}, {COLORS['medium_blue']});
            border: 2px solid {COLORS['accent_gold']};
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }}
        .header-container h1 {{
            font-family: 'Playfair Display', serif;
            color: {COLORS['accent_gold']};
            margin: 0; font-size: 2rem;
        }}
        .header-container p {{
            color: {COLORS['text_primary']};
            font-family: 'Source Sans Pro', sans-serif;
            margin: 0.3rem 0 0; font-size: 0.9rem;
        }}

        .metric-card {{
            background: {COLORS['card_bg']};
            border: 1px solid rgba(255,215,0,0.3);
            border-radius: 10px;
            padding: 1.2rem; text-align: center;
            margin-bottom: 0.8rem;
        }}
        .metric-card .label {{
            color: {COLORS['text_secondary']};
            font-size: 0.8rem; text-transform: uppercase;
            letter-spacing: 1px;
            font-family: 'Source Sans Pro', sans-serif;
        }}
        .metric-card .value {{
            color: {COLORS['accent_gold']};
            font-size: 1.6rem; font-weight: 700;
            font-family: 'Playfair Display', serif;
            margin-top: 0.3rem;
        }}

        .info-box {{
            background: rgba(0,51,102,0.5);
            border: 1px solid {COLORS['accent_gold']};
            border-radius: 8px;
            padding: 1rem 1.5rem;
            font-family: 'Source Sans Pro', sans-serif;
            color: {COLORS['text_primary']};
            margin: 0.8rem 0;
        }}

        .section-title {{
            font-family: 'Playfair Display', serif;
            color: {COLORS['accent_gold']};
            font-size: 1.3rem;
            border-bottom: 2px solid rgba(255,215,0,0.3);
            padding-bottom: 0.5rem;
            margin: 1.5rem 0 1rem;
        }}

        .formula-box {{
            background: rgba(0,51,102,0.5);
            border: 1px solid {COLORS['accent_gold']};
            border-radius: 8px;
            padding: 1rem 1.5rem;
            font-family: 'Source Sans Pro', monospace;
            color: {COLORS['text_primary']};
            margin: 0.8rem 0;
        }}

        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
        .stTabs [data-baseweb="tab"] {{
            background: {COLORS['card_bg']};
            border: 1px solid rgba(255,215,0,0.3);
            border-radius: 8px;
            color: {COLORS['text_primary']};
            font-family: 'Source Sans Pro', sans-serif;
            padding: 0.5rem 1rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: {COLORS['dark_blue']};
            border: 2px solid {COLORS['accent_gold']};
            color: {COLORS['accent_gold']};
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(255,215,0,0.2);
            border-radius: 8px;
        }}

        footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
st.set_page_config(**PAGE_CONFIG)
apply_styles()

# Header
st.markdown(f"""
<div class="header-container">
    <h1>{BRANDING['icon']} Nifty 50 VaR, ES & Volatility Analysis</h1>
    <p>{BRANDING['name']}</p>
    <p style="font-size:0.8rem; color:{COLORS['text_secondary']};">
        {BRANDING['instructor']} | {BRANDING['credentials']}</p>
</div>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
st.sidebar.markdown(f"""
<div style="text-align:center; padding:1.2rem; background:rgba(255,215,0,0.08);
     border-radius:10px; margin-bottom:1.5rem; border:2px solid {COLORS['accent_gold']};">
    <h3 style="color:{COLORS['accent_gold']}; margin:0;">{BRANDING['icon']} RISK ANALYTICS</h3>
    <p style="color:{COLORS['text_secondary']}; font-size:0.75rem; margin:5px 0 0;">
        Advanced Financial Risk Models</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"<p style='color:{COLORS['accent_gold']}; font-weight:700;'>📊 Stock Selection</p>",
                    unsafe_allow_html=True)

ticker_options = ["^NSEI (Nifty 50 Index)"] + [
    f"{name} ({ticker})" for ticker, name in NIFTY50_LIST
]
ticker_display = st.sidebar.selectbox("Select stock or index", ticker_options, index=0)
if ticker_display.startswith("^"):
    TICKER = "^NSEI"
else:
    TICKER = ticker_display.split("(")[-1].rstrip(")")

years = st.sidebar.slider("Years of history", 1.0, 5.0, 3.0, 0.5)

st.sidebar.markdown(f"<p style='color:{COLORS['accent_gold']}; font-weight:700;'>📈 EVT Parameters</p>",
                    unsafe_allow_html=True)
evt_alpha = st.sidebar.slider("EVT confidence level (α)", 0.90, 0.99, 0.95, 0.01)
evt_threshold_q = st.sidebar.slider("EVT threshold quantile", 0.85, 0.95, 0.90, 0.01)

st.sidebar.markdown(f"<p style='color:{COLORS['accent_gold']}; font-weight:700;'>⚡ Volatility Models</p>",
                    unsafe_allow_html=True)
ewma_lambda = st.sidebar.slider("EWMA λ", 0.90, 0.98, 0.94, 0.01)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 1, 10, 5)
var_es_alpha = st.sidebar.slider("VaR/ES confidence (vol models)", 0.90, 0.99, 0.95, 0.01)


@st.cache_data(ttl=3600)
def load_data(ticker: str, years: float):
    """Load and cache price data - updates when ticker or years change"""
    prices = fetch_prices(ticker, years)
    returns = get_returns(prices)
    return prices, returns.dropna()


# Load data - will refresh when TICKER or years change
with st.spinner(f'Loading data for {TICKER}...'):
    try:
        prices, returns = load_data(TICKER, years)
        n_returns = len(returns)
        st.success(f"✅ Loaded {n_returns:,} returns for {TICKER}", icon="✅")
    except Exception as e:
        st.error(f"Could not load data for {TICKER}: {e}")
        st.stop()

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "ℹ️ About Platform",
    "📊 EVT VaR & ES",
    "⚡ Volatility & VaR/ES",
    "📚 Educational Materials",
    "🎓 Model Details"
])

# ========== TAB 1: ABOUT PLATFORM ==========
with tab1:
    st.markdown('<div class="section-title">🏔️ Welcome to The Mountain Path</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
        <h3 style="color:{COLORS['accent_gold']}; margin-top:0;">Advanced Financial Risk Management Platform</h3>
        <p>This comprehensive platform provides institutional-grade Value at Risk (VaR) and Expected Shortfall (ES) 
        analysis for Nifty 50 stocks and index using cutting-edge statistical models.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Platform Capabilities</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Extreme Value Theory (EVT)</div>
            <div class="value" style="font-size:1.2rem;">GPD-based VaR & ES</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <strong>🎯 What it does:</strong>
            <ul>
                <li>Models tail risk using Generalized Pareto Distribution</li>
                <li>Focuses specifically on extreme losses</li>
                <li>Provides VaR and Expected Shortfall estimates</li>
                <li>Adjustable threshold and confidence levels</li>
            </ul>
            <strong>💡 Key Features:</strong>
            <ul>
                <li>GPD shape parameter (ξ) estimation</li>
                <li>Threshold selection via quantiles</li>
                <li>Visual tail analysis with fitted distributions</li>
                <li>Exceedance counting and validation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Volatility Models</div>
            <div class="value" style="font-size:1.2rem;">EWMA | ARCH | GARCH | EGARCH</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <strong>🎯 What it does:</strong>
            <ul>
                <li>Estimates conditional volatility using 4 models</li>
                <li>Multi-day ahead volatility forecasting</li>
                <li>VaR and ES from volatility estimates</li>
                <li>Model comparison and validation</li>
            </ul>
            <strong>💡 Key Features:</strong>
            <ul>
                <li><strong>EWMA:</strong> Exponentially weighted moving average</li>
                <li><strong>ARCH(1):</strong> Autoregressive conditional heteroskedasticity</li>
                <li><strong>GARCH(1,1):</strong> Generalized ARCH with mean reversion</li>
                <li><strong>EGARCH(1,1):</strong> Exponential GARCH (leverage effects)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 Use Cases</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h4 style="color:{COLORS['accent_gold']};">📊 Portfolio Managers</h4>
            <ul>
                <li>Daily VaR monitoring</li>
                <li>Position limit setting</li>
                <li>Risk budgeting</li>
                <li>Stress testing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h4 style="color:{COLORS['accent_gold']};">🏦 Risk Managers</h4>
            <ul>
                <li>Regulatory compliance (Basel)</li>
                <li>Capital allocation</li>
                <li>Model backtesting</li>
                <li>Risk reporting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="info-box">
            <h4 style="color:{COLORS['accent_gold']};">🎓 Researchers</h4>
            <ul>
                <li>Model comparison studies</li>
                <li>Tail behavior analysis</li>
                <li>Indian market research</li>
                <li>Academic publications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">⚙️ Current Configuration</div>', unsafe_allow_html=True)
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Selected Asset</div>
            <div class="value" style="font-size:1.2rem;">{TICKER}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with config_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Data Points</div>
            <div class="value" style="font-size:1.2rem;">{n_returns:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with config_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Time Period</div>
            <div class="value" style="font-size:1.2rem;">{years:.1f} Years</div>
        </div>
        """, unsafe_allow_html=True)


# ========== TAB 2: EVT VaR & ES ==========
with tab2:
    st.markdown(f'<div class="section-title">📈 Extreme Value Theory Analysis for {TICKER}</div>', unsafe_allow_html=True)
    st.caption(f"Using {n_returns:,} daily returns ({years:.1f} years) | Confidence: {evt_alpha*100:.0f}% | Threshold: {evt_threshold_q*100:.0f}%")
    
    try:
        # Run EVT analysis with current parameters
        with st.spinner('Calculating EVT VaR & ES...'):
            result = run_evt_var_es(
                returns,
                alpha=evt_alpha,
                threshold_quantile=evt_threshold_q,
            )
        pct = evt_alpha * 100
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="label">VaR (% loss)</div>'
                        f'<div class="value">{result["var"]*100:.2f}%</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="label">Expected Shortfall</div>'
                        f'<div class="value">{result["es"]*100:.2f}%</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="label">GPD Shape (ξ)</div>'
                        f'<div class="value">{result["xi"]:.4f}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="label">Exceedances</div>'
                        f'<div class="value">{result["n_exceedances"]}</div></div>', unsafe_allow_html=True)

        losses = losses_from_returns(returns)
        u = result["threshold"]
        var_val, es_val = result["var"], result["es"]
        xi, sigma = result["xi"], result["sigma"]
        exceedances = losses[losses > u] - u

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax1, ax2 = axes[0], axes[1]
        ax1.hist(losses, bins=50, density=True, color="steelblue", alpha=0.7, edgecolor="white", label="Losses")
        ax1.axvline(u, color="gray", linestyle="--", linewidth=2, label=f"Threshold u = {u:.3f}")
        ax1.axvline(var_val, color="darkred", linestyle="-", linewidth=2, label=f"VaR = {var_val:.3f}")
        ax1.axvline(es_val, color="crimson", linestyle=":", linewidth=2, label=f"ES = {es_val:.3f}")
        ax1.set_xlabel("Loss (positive = loss)")
        ax1.set_ylabel("Density")
        ax1.set_title(f"Loss distribution — {TICKER}")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.hist(exceedances, bins=min(40, max(15, len(exceedances)//5)), density=True,
                 color="steelblue", alpha=0.7, edgecolor="white", label="Exceedances (L − u)")
        if len(exceedances) >= 10 and not np.isnan(xi):
            x_gpd = np.linspace(0, exceedances.max() * 1.05, 200)
            gpd_pdf = stats.genpareto.pdf(x_gpd, xi, loc=0, scale=sigma)
            ax2.plot(x_gpd, gpd_pdf, "r-", linewidth=2, label=f"GPD fit (ξ={xi:.3f})")
        ax2.axvline(var_val - u, color="darkred", linestyle="-", linewidth=1.5, label=f"VaR − u = {var_val - u:.3f}")
        ax2.set_xlabel("Exceedance (loss − threshold)")
        ax2.set_ylabel("Density")
        ax2.set_title("Tail: exceedances & GPD fit")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"EVT failed: {e}")


# ========== TAB 3: VOLATILITY & VaR/ES ==========
with tab3:
    st.markdown('<div class="section-title">⚡ Volatility Models & Risk Metrics</div>', unsafe_allow_html=True)
    
    try:
        vol = run_volatility_pipeline(
            returns,
            ewma_lambda=ewma_lambda,
            forecast_horizon=forecast_horizon,
            var_es_alpha=var_es_alpha,
        )
        pct = vol["var_es_alpha"] * 100
        H = vol["forecast_horizon"]

        st.subheader("Volatility forecast (annualized %)")
        df_vol = pd.DataFrame({
            "EWMA": [vol["ewma_forecast"][h] * np.sqrt(252) * 100 for h in range(H)],
            "ARCH": [vol["arch_forecast"][h] * np.sqrt(252) * 100 if not np.isnan(vol["arch_forecast"][h]) else np.nan for h in range(H)],
            "GARCH": [vol["garch_forecast"][h] * np.sqrt(252) * 100 if not np.isnan(vol["garch_forecast"][h]) else np.nan for h in range(H)],
            "EGARCH": [vol["egarch_forecast"][h] * np.sqrt(252) * 100 if not np.isnan(vol["egarch_forecast"][h]) else np.nan for h in range(H)],
        }, index=[f"Day {h+1}" for h in range(H)])
        st.dataframe(df_vol.round(2), use_container_width=True)

        st.subheader(f"VaR & Expected Shortfall ({pct:.0f}%) from forecasted vol")
        df_var = pd.DataFrame({
            "VaR_EWMA": [x*100 for x in vol["var_f_ewma"]],
            "VaR_ARCH": [x*100 if not np.isnan(x) else np.nan for x in vol["var_f_arch"]],
            "VaR_GARCH": [x*100 if not np.isnan(x) else np.nan for x in vol["var_f_garch"]],
            "VaR_EGARCH": [x*100 if not np.isnan(x) else np.nan for x in vol["var_f_egarch"]],
        }, index=[f"Day {h+1}" for h in range(H)])
        st.dataframe(df_var.round(2), use_container_width=True)

        # Figure 1: Returns + Volatility + Vol forecast
        fig1, axes1 = plt.subplots(3, 1, figsize=(11, 8))
        ax1, ax2, ax3 = axes1[0], axes1[1], axes1[2]
        ax1.plot(vol["returns"].index, vol["returns"].values, color="steelblue", alpha=0.8, linewidth=0.6, label="Daily returns")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax1.set_ylabel("Return")
        ax1.set_title(f"{TICKER} — Returns and conditional volatility")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(vol["vol_ewma"].index, vol["vol_ewma"].values, color="green", alpha=0.9, linewidth=0.8, label="EWMA")
        ax2.plot(vol["vol_arch"].index, vol["vol_arch"].values, color="darkorange", alpha=0.9, linewidth=0.8, label="ARCH(1)")
        ax2.plot(vol["vol_garch"].index, vol["vol_garch"].values, color="purple", alpha=0.9, linewidth=0.8, label="GARCH(1,1)")
        ax2.plot(vol["vol_egarch"].index, vol["vol_egarch"].values, color="brown", alpha=0.9, linewidth=0.8, label="EGARCH(1,1)")
        ax2.set_ylabel("Conditional volatility")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)

        x_fore = np.arange(1, H + 1)
        ax3.plot(x_fore, [x * np.sqrt(252) * 100 for x in vol["ewma_forecast"]], "o-", color="green", label="EWMA", linewidth=2)
        ax3.plot(x_fore, [x * np.sqrt(252) * 100 for x in vol["arch_forecast"]], "s-", color="darkorange", label="ARCH", linewidth=2)
        ax3.plot(x_fore, [x * np.sqrt(252) * 100 for x in vol["garch_forecast"]], "^-", color="purple", label="GARCH(1,1)", linewidth=2)
        ax3.plot(x_fore, [x * np.sqrt(252) * 100 for x in vol["egarch_forecast"]], "d-", color="brown", label="EGARCH(1,1)", linewidth=2)
        ax3.set_xlabel("Forecast step (days ahead)")
        ax3.set_ylabel("Volatility (% ann.)")
        ax3.set_title("Volatility forecast (annualized %)")
        ax3.legend(loc="upper right", fontsize=9)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        # Figure 2: VaR & ES (in-sample + forecast)
        fig2, axes2 = plt.subplots(4, 1, figsize=(11, 10))
        ax1, ax2, ax3, ax4 = axes2[0], axes2[1], axes2[2], axes2[3]
        ax1.plot(vol["var_ewma"].index, vol["var_ewma"].values * 100, color="green", alpha=0.9, linewidth=0.8, label="EWMA")
        ax1.plot(vol["var_arch"].index, vol["var_arch"].values * 100, color="darkorange", alpha=0.9, linewidth=0.8, label="ARCH(1)")
        ax1.plot(vol["var_garch"].index, vol["var_garch"].values * 100, color="purple", alpha=0.9, linewidth=0.8, label="GARCH(1,1)")
        ax1.plot(vol["var_egarch"].index, vol["var_egarch"].values * 100, color="brown", alpha=0.9, linewidth=0.8, label="EGARCH(1,1)")
        ax1.set_ylabel("VaR (% loss)")
        ax1.set_title(f"VaR ({pct:.0f}%) from conditional volatility (normal)")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.plot(vol["es_ewma"].index, vol["es_ewma"].values * 100, color="green", alpha=0.9, linewidth=0.8, label="EWMA")
        ax2.plot(vol["es_arch"].index, vol["es_arch"].values * 100, color="darkorange", alpha=0.9, linewidth=0.8, label="ARCH(1)")
        ax2.plot(vol["es_garch"].index, vol["es_garch"].values * 100, color="purple", alpha=0.9, linewidth=0.8, label="GARCH(1,1)")
        ax2.plot(vol["es_egarch"].index, vol["es_egarch"].values * 100, color="brown", alpha=0.9, linewidth=0.8, label="EGARCH(1,1)")
        ax2.set_ylabel("Expected Shortfall (% loss)")
        ax2.set_title(f"Expected Shortfall ({pct:.0f}%) from conditional volatility (normal)")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)

        ax3.plot(x_fore, [x*100 for x in vol["var_f_ewma"]], "o-", color="green", linewidth=2, label="EWMA")
        ax3.plot(x_fore, [x*100 for x in vol["var_f_arch"]], "s-", color="darkorange", linewidth=2, label="ARCH")
        ax3.plot(x_fore, [x*100 for x in vol["var_f_garch"]], "^-", color="purple", linewidth=2, label="GARCH(1,1)")
        ax3.plot(x_fore, [x*100 for x in vol["var_f_egarch"]], "d-", color="brown", linewidth=2, label="EGARCH(1,1)")
        ax3.set_xticks(x_fore)
        ax3.set_xlabel("Forecast step (days ahead)")
        ax3.set_ylabel("VaR (% loss)")
        ax3.set_title(f"Forecast VaR ({pct:.0f}%)")
        ax3.legend(loc="upper right", fontsize=9)
        ax3.grid(True, alpha=0.3)

        ax4.plot(x_fore, [x*100 for x in vol["es_f_ewma"]], "o-", color="green", linewidth=2, label="EWMA")
        ax4.plot(x_fore, [x*100 for x in vol["es_f_arch"]], "s-", color="darkorange", linewidth=2, label="ARCH")
        ax4.plot(x_fore, [x*100 for x in vol["es_f_garch"]], "^-", color="purple", linewidth=2, label="GARCH(1,1)")
        ax4.plot(x_fore, [x*100 for x in vol["es_f_egarch"]], "d-", color="brown", linewidth=2, label="EGARCH(1,1)")
        ax4.set_xticks(x_fore)
        ax4.set_xlabel("Forecast step (days ahead)")
        ax4.set_ylabel("ES (% loss)")
        ax4.set_title(f"Forecast Expected Shortfall ({pct:.0f}%)")
        ax4.legend(loc="upper right", fontsize=9)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    except Exception as e:
        st.error(f"Volatility pipeline failed: {e}")
        import traceback
        st.code(traceback.format_exc())


# ========== TAB 4: EDUCATIONAL MATERIALS ==========
with tab4:
    st.markdown('<div class="section-title">📚 Educational Materials</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
        <h3 style="color:{COLORS['accent_gold']}; margin-top:0;">Comprehensive Risk Management Education</h3>
        <p>Learn the theoretical foundations and practical applications of advanced risk measurement techniques 
        used in institutional finance and banking.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sub-tabs for different topics
    edu_tab1, edu_tab2, edu_tab3, edu_tab4 = st.tabs([
        "📊 Extreme Value Theory",
        "⚡ Volatility Models",
        "📈 VaR & Expected Shortfall",
        "🎯 Practical Applications"
    ])

    with edu_tab1:
        st.markdown('<div class="section-title">📊 Extreme Value Theory (EVT)</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <h4 style="color:{COLORS['accent_gold']};">What is EVT?</h4>
            <p>Extreme Value Theory is a branch of statistics dealing with extreme deviations from the median 
            of probability distributions. In finance, it's used to model tail risk—the risk of extreme losses.</p>
            
            <h4 style="color:{COLORS['accent_gold']}; margin-top:1rem;">Key Concepts:</h4>
            <ul>
                <li><strong>Threshold (u):</strong> A high quantile (e.g., 90th percentile) above which we model the tail</li>
                <li><strong>Exceedances:</strong> Observations that exceed the threshold</li>
                <li><strong>GPD (Generalized Pareto Distribution):</strong> The limiting distribution of exceedances</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">GPD Formula</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="formula-box">
            <strong>Generalized Pareto Distribution CDF:</strong><br><br>
            F(x) = 1 − (1 + ξx/σ)<sup>−1/ξ</sup>  for ξ ≠ 0<br><br>
            where:<br>
            • ξ (xi) = shape parameter (determines tail heaviness)<br>
            • σ (sigma) = scale parameter<br>
            • x = exceedance (loss − threshold)
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <h4 style="color:{COLORS['accent_gold']};">Interpreting the Shape Parameter (ξ):</h4>
            <ul>
                <li><strong>ξ &gt; 0:</strong> Heavy-tailed distribution (Pareto-type, common for financial assets)</li>
                <li><strong>ξ = 0:</strong> Exponential tail (medium risk)</li>
                <li><strong>ξ &lt; 0:</strong> Short-tailed distribution (bounded, rare in finance)</li>
            </ul>
            
            <p style="margin-top:1rem;">For most stock returns, ξ is positive, indicating fat tails with higher 
            probability of extreme events than normal distribution predicts.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">EVT-based VaR Formula</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="formula-box">
            <strong>Value at Risk using GPD:</strong><br><br>
            VaR<sub>α</sub> = u + (σ/ξ) × [(n/N<sub>u</sub> × (1−α))<sup>−ξ</sup> − 1]<br><br>
            where:<br>
            • u = threshold<br>
            • n = total observations<br>
            • N<sub>u</sub> = number of exceedances<br>
            • α = confidence level (e.g., 0.95)
        </div>
        """, unsafe_allow_html=True)

    with edu_tab2:
        st.markdown('<div class="section-title">⚡ Volatility Models</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <h4 style="color:{COLORS['accent_gold']};">1. EWMA (Exponentially Weighted Moving Average)</h4>
                <strong>Formula:</strong><br>
                σ²<sub>t</sub> = λ × σ²<sub>t-1</sub> + (1−λ) × r²<sub>t-1</sub><br><br>
                
                <strong>Key Features:</strong>
                <ul>
                    <li>Simple recursive formula</li>
                    <li>λ typically 0.94 (RiskMetrics standard)</li>
                    <li>Recent returns weighted more heavily</li>
                    <li>No parameter estimation needed</li>
                </ul>
                
                <strong>Advantages:</strong>
                <ul>
                    <li>Easy to implement</li>
                    <li>Responds quickly to market changes</li>
                    <li>Industry standard baseline</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box" style="margin-top:1rem;">
                <h4 style="color:{COLORS['accent_gold']};">3. GARCH(1,1)</h4>
                <strong>Formula:</strong><br>
                σ²<sub>t</sub> = ω + α₁ × r²<sub>t-1</sub> + β₁ × σ²<sub>t-1</sub><br><br>
                
                <strong>Key Features:</strong>
                <ul>
                    <li>Includes lagged variance (mean reversion)</li>
                    <li>Most widely used volatility model</li>
                    <li>Persistence: α₁ + β₁ (usually ~0.99)</li>
                    <li>Long-run variance: ω/(1−α₁−β₁)</li>
                </ul>
                
                <strong>Advantages:</strong>
                <ul>
                    <li>Captures volatility clustering</li>
                    <li>Mean-reverting to long-run level</li>
                    <li>Parsimonious (only 3 parameters)</li>
                    <li>Good forecasting performance</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="info-box">
                <h4 style="color:{COLORS['accent_gold']};">2. ARCH(1)</h4>
                <strong>Formula:</strong><br>
                σ²<sub>t</sub> = ω + α₁ × r²<sub>t-1</sub><br><br>
                
                <strong>Key Features:</strong>
                <ul>
                    <li>Autoregressive conditional heteroskedasticity</li>
                    <li>Variance depends on past squared returns</li>
                    <li>Captures volatility clustering</li>
                    <li>Foundation for GARCH family</li>
                </ul>
                
                <strong>Limitations:</strong>
                <ul>
                    <li>May need many lags for adequate fit</li>
                    <li>No mean reversion</li>
                    <li>GARCH usually preferred</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box" style="margin-top:1rem;">
                <h4 style="color:{COLORS['accent_gold']};">4. EGARCH(1,1)</h4>
                <strong>Formula:</strong><br>
                log(σ²<sub>t</sub>) = ω + β₁ × log(σ²<sub>t-1</sub>) + α₁ × |z<sub>t-1</sub>| + γ × z<sub>t-1</sub><br><br>
                
                <strong>Key Features:</strong>
                <ul>
                    <li>Logarithmic form ensures σ²<sub>t</sub> &gt; 0</li>
                    <li>γ captures leverage effect</li>
                    <li>Negative shocks (γ &lt; 0) increase volatility more</li>
                    <li>Asymmetric response to news</li>
                </ul>
                
                <strong>Advantages:</strong>
                <ul>
                    <li>Captures leverage effect in equity returns</li>
                    <li>No parameter constraints needed</li>
                    <li>Better for stocks (asymmetric volatility)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with edu_tab3:
        st.markdown('<div class="section-title">📈 VaR & Expected Shortfall</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <h4 style="color:{COLORS['accent_gold']};">Value at Risk (VaR)</h4>
            <p><strong>Definition:</strong> The maximum expected loss over a given time horizon at a specified 
            confidence level.</p>
            
            <p style="margin-top:1rem;"><strong>Example:</strong> 1-day VaR of $1 million at 95% confidence means:
            "We are 95% confident that our losses will not exceed $1 million tomorrow."</p>
            
            <h4 style="color:{COLORS['accent_gold']}; margin-top:1rem;">Calculation Methods:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <strong>1. Parametric (Variance-Covariance):</strong> Assumes normal distribution
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="formula-box">
            VaR<sub>α</sub> = μ + σ × z<sub>α</sub><br>
            where z<sub>α</sub> is the α-quantile of standard normal
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <strong>2. Historical Simulation:</strong> Uses empirical distribution
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="formula-box">
            VaR<sub>α</sub> = α-th quantile of historical returns
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <strong>3. Monte Carlo:</strong> Simulates future scenarios
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="formula-box">
            Generate N scenarios → Sort → Take α-th percentile
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:{COLORS['accent_gold']};">Expected Shortfall (ES / CVaR)</h4>
            <p><strong>Definition:</strong> The average loss given that the loss exceeds VaR.</p>
            
            <p style="margin-top:1rem;"><strong>Why ES?</strong></p>
            <ul>
                <li>VaR doesn't tell us about losses beyond the threshold</li>
                <li>ES is a <strong>coherent risk measure</strong> (VaR is not)</li>
                <li>ES satisfies subadditivity: ES(X+Y) ≤ ES(X) + ES(Y)</li>
                <li>Preferred by Basel III for market risk capital</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="formula-box">
            <strong>For Normal Distribution:</strong><br><br>
            ES<sub>α</sub> = μ + σ × φ(z<sub>α</sub>) / (1−α)<br><br>
            where φ is the standard normal PDF<br><br>
            
            <strong>For GPD (EVT):</strong><br><br>
            ES<sub>α</sub> = (VaR<sub>α</sub> + σ − ξ×u) / (1−ξ)  for ξ &lt; 1
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:{COLORS['accent_gold']};">Comparison: VaR vs ES</h4>
        </div>
        """, unsafe_allow_html=True)
        
        comparison = pd.DataFrame({
            'Aspect': ['Definition', 'Information', 'Coherence', 'Optimization', 'Regulatory', 'Interpretation'],
            'VaR': [
                'Loss threshold at α confidence',
                'Single quantile',
                'Not coherent',
                'Harder to optimize',
                'Basel II (market risk)',
                'Easier to explain'
            ],
            'Expected Shortfall': [
                'Average loss beyond VaR',
                'Entire tail distribution',
                'Coherent risk measure',
                'Easier to optimize (convex)',
                'Basel III (preferred)',
                'More technical'
            ]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    with edu_tab4:
        st.markdown('<div class="section-title">🎯 Practical Applications</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <h4 style="color:{COLORS['accent_gold']};">1. Portfolio Management</h4>
            <ul>
                <li><strong>Position Sizing:</strong> Limit position size so portfolio VaR stays within risk budget</li>
                <li><strong>Risk Budgeting:</strong> Allocate risk capital across strategies based on ES</li>
                <li><strong>Performance Attribution:</strong> Compare returns to risk-adjusted metrics (Sharpe, Sortino)</li>
                <li><strong>Stress Testing:</strong> Use EVT to model extreme scenarios</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:{COLORS['accent_gold']};">2. Regulatory Compliance</h4>
            <ul>
                <li><strong>Basel III:</strong> Banks must calculate ES for market risk capital requirements</li>
                <li><strong>Backtesting:</strong> Regulators require validation that VaR/ES models are accurate</li>
                <li><strong>Documentation:</strong> Must document model assumptions, parameters, and validation</li>
                <li><strong>Model Review:</strong> Annual or more frequent review of risk models</li>
            </ul>
            
            <div class="formula-box" style="margin-top:1rem;">
                <strong>Market Risk Capital (Basel III):</strong><br><br>
                K = max(VaR_t-1, mc × VaR_avg) + SRC<br><br>
                where:<br>
                • mc = multiplier (≥3, typically 3-4 based on backtesting)<br>
                • VaR_avg = average VaR over last 60 days<br>
                • SRC = stressed capital add-on
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:{COLORS['accent_gold']};">3. Model Selection Guidelines</h4>
        </div>
        """, unsafe_allow_html=True)
        
        model_guide = pd.DataFrame({
            'Scenario': [
                'Quick daily risk estimates',
                'Capturing volatility clustering',
                'Long-term forecasting',
                'Equity portfolios (leverage effect)',
                'Extreme events / tail risk',
                'Regulatory compliance',
            ],
            'Recommended Model': [
                'EWMA',
                'GARCH(1,1)',
                'GARCH(1,1)',
                'EGARCH(1,1)',
                'EVT (GPD)',
                'ES from GARCH + EVT',
            ],
            'Rationale': [
                'Fast, responsive, no estimation',
                'Best captures vol. clustering',
                'Mean reversion to long-run vol.',
                'Captures asymmetric volatility',
                'Directly models tail distribution',
                'Coherent, tail-focused',
            ]
        })
        st.dataframe(model_guide, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:{COLORS['accent_gold']};">4. Backtesting Framework</h4>
            <p>All risk models must be validated against actual losses:</p>
            
            <strong>Traffic Light Approach (Basel):</strong>
            <ul>
                <li><strong>Green Zone (0-4 exceptions):</strong> Model adequate, multiplier = 3.0</li>
                <li><strong>Yellow Zone (5-9 exceptions):</strong> Review needed, multiplier = 3.4-3.85</li>
                <li><strong>Red Zone (10+ exceptions):</strong> Model inadequate, multiplier = 4.0</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">Kupiec Test Formula</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="formula-box">
            <strong>Kupiec Test (Unconditional Coverage):</strong><br><br>
            
            LR<sub>uc</sub> = -2 × ln[(1−α)<sup>T−N</sup> × α<sup>N</sup>] + 2 × ln[(1−N/T)<sup>T−N</sup> × (N/T)<sup>N</sup>]<br><br>
            
            where:<br>
            • T = number of observations<br>
            • N = number of VaR exceptions<br>
            • α = VaR confidence level<br><br>
            
            LR<sub>uc</sub> ~ χ²(1) under null hypothesis
        </div>
        """, unsafe_allow_html=True)


# ========== TAB 5: MODEL DETAILS ==========
with tab5:
    st.markdown('<div class="section-title">🎓 Technical Model Specifications</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
        <h4 style="color:{COLORS['accent_gold']};">Implementation Details</h4>
        <p>This platform implements industry-standard risk models with the following specifications:</p>
    </div>
    """, unsafe_allow_html=True)

    spec_tab1, spec_tab2, spec_tab3 = st.tabs([
        "🔢 Model Parameters",
        "📊 Assumptions",
        "⚙️ Computational Methods"
    ])

    with spec_tab1:
        st.markdown('<div class="section-title">EVT Model Parameters</div>', unsafe_allow_html=True)
        evt_params = pd.DataFrame({
            'Parameter': [
                'Threshold Selection',
                'GPD Estimation',
                'Minimum Exceedances',
                'VaR Confidence Levels',
                'Time Horizon',
            ],
            'Value/Method': [
                'User-adjustable quantile (85%-95%)',
                'Maximum Likelihood Estimation (MLE)',
                '10 exceedances minimum',
                '90% - 99% (user-adjustable)',
                '1 day',
            ],
            'Notes': [
                'Higher threshold = fewer but more extreme observations',
                'Optimization with bounded parameters',
                'Falls back to empirical quantile if insufficient',
                'Typical: 95% for daily, 99% for regulatory',
                'Based on daily return data',
            ]
        })
        st.dataframe(evt_params, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Volatility Model Parameters</div>', unsafe_allow_html=True)
        vol_params = pd.DataFrame({
            'Model': ['EWMA', 'ARCH(1)', 'GARCH(1,1)', 'EGARCH(1,1)'],
            'Parameters': [
                'λ (decay): 0.90-0.98',
                'ω, α₁',
                'ω, α₁, β₁',
                'ω, α₁, β₁, γ',
            ],
            'Estimation': [
                'User-specified (no estimation)',
                'MLE via ARCH package',
                'MLE via ARCH package',
                'MLE via ARCH package',
            ],
            'Forecast Horizon': [
                '1-10 days',
                '1-10 days',
                '1-10 days',
                '1-10 days',
            ],
            'Typical Values': [
                'λ = 0.94 (RiskMetrics)',
                'α₁ ≈ 0.1-0.3',
                'α₁+β₁ ≈ 0.98-0.99',
                'γ < 0 (leverage)',
            ]
        })
        st.dataframe(vol_params, use_container_width=True, hide_index=True)

    with spec_tab2:
        st.markdown(f"""
        <div class="info-box">
            <h4 style="color:{COLORS['accent_gold']};">Key Assumptions</h4>
            
            <strong>1. Return Distribution:</strong>
            <ul>
                <li>Log returns calculated as: r_t = ln(P_t / P_(t-1))</li>
                <li>Losses defined as negative returns (positive = loss)</li>
                <li>Annualization: multiply by √252 for volatility</li>
            </ul>
            
            <strong>2. EVT Assumptions:</strong>
            <ul>
                <li>Exceedances over high threshold follow GPD</li>
                <li>Threshold sufficiently high but with adequate exceedances</li>
                <li>Independence of exceedances (may not hold with clustering)</li>
                <li>Stationarity over sample period</li>
            </ul>
            
            <strong>3. GARCH Model Assumptions:</strong>
            <ul>
                <li>Conditional volatility time-varying</li>
                <li>Standardized residuals i.i.d. (typically normal)</li>
                <li>Model parameters stable over estimation period</li>
                <li>No structural breaks in variance process</li>
            </ul>
            
            <strong>4. VaR/ES Calculation:</strong>
            <ul>
                <li>Normal distribution for volatility-based VaR/ES</li>
                <li>GPD for EVT-based VaR/ES</li>
                <li>Zero mean assumption for short horizon (1-day)</li>
                <li>Losses are i.i.d. (or conditionally independent given volatility)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:{COLORS['accent_gold']};">Limitations</h4>
            
            <strong>General:</strong>
            <ul>
                <li>Historical data may not predict future crises</li>
                <li>Model assumes no regime changes</li>
                <li>Single asset analysis (no portfolio correlations)</li>
                <li>No transaction costs or liquidity effects</li>
            </ul>
            
            <strong>EVT-Specific:</strong>
            <ul>
                <li>Requires sufficient tail observations</li>
                <li>Threshold selection is somewhat subjective</li>
                <li>Assumes tail behavior is stable</li>
                <li>May underestimate risk during structural breaks</li>
            </ul>
            
            <strong>GARCH-Specific:</strong>
            <ul>
                <li>Normal assumption may underestimate tail risk</li>
                <li>Forecasts degrade with longer horizons</li>
                <li>Estimation can fail for some stocks</li>
                <li>Does not capture jumps or discontinuities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with spec_tab3:
        st.markdown(f"""
        <div class="info-box">
            <h4 style="color:{COLORS['accent_gold']};">Computational Methods</h4>
            
            <strong>Data Processing:</strong>
            <ul>
                <li><strong>Source:</strong> Yahoo Finance via yfinance library</li>
                <li><strong>Frequency:</strong> Daily adjusted close prices</li>
                <li><strong>Missing Data:</strong> Dropped (not interpolated)</li>
                <li><strong>Caching:</strong> 1-hour TTL for price data</li>
            </ul>
            
            <strong>GPD Estimation:</strong>
            <ul>
                <li><strong>Method:</strong> Maximum Likelihood Estimation</li>
                <li><strong>Optimizer:</strong> L-BFGS-B with bounded parameters</li>
                <li><strong>Bounds:</strong> ξ ∈ [-0.5, 0.5], σ > 0</li>
                <li><strong>Fallback:</strong> SciPy's genpareto.fit() if MLE fails</li>
            </ul>
            
            <strong>GARCH Estimation:</strong>
            <ul>
                <li><strong>Package:</strong> arch (Python ARCH/GARCH library)</li>
                <li><strong>Method:</strong> Maximum Likelihood via numerical optimization</li>
                <li><strong>Scaling:</strong> Returns multiplied by 100 for numerical stability</li>
                <li><strong>Convergence:</strong> Default tolerances, warnings suppressed</li>
            </ul>
            
            <strong>Visualization:</strong>
            <ul>
                <li><strong>Library:</strong> Matplotlib</li>
                <li><strong>Charts:</strong> Histograms, line plots, bar charts</li>
                <li><strong>Resolution:</strong> Default DPI for web display</li>
                <li><strong>Memory:</strong> Figures closed after display to prevent leaks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="formula-box" style="margin-top:1rem;">
            <strong>Performance Considerations:</strong><br><br>
            
            • <strong>Data Loading:</strong> ~2-5 seconds (cached after first load)<br>
            • <strong>EVT Calculation:</strong> ~0.1-0.5 seconds<br>
            • <strong>GARCH Estimation:</strong> ~2-5 seconds per model<br>
            • <strong>Total Pipeline:</strong> ~10-15 seconds for complete analysis<br><br>
            
            <strong>Optimization:</strong> Streamlit caching reduces subsequent loads to <1 second
        </div>
        """, unsafe_allow_html=True)


# Footer
st.divider()
st.markdown(f"""
<div style="text-align:center; padding:1rem;">
    <p style="color:{COLORS['accent_gold']}; font-family:'Playfair Display', serif; font-weight:700;">
        {BRANDING['icon']} {BRANDING['name']}</p>
    <p style="color:{COLORS['text_secondary']}; font-size:0.8rem;">
        {BRANDING['instructor']} | {BRANDING['credentials']}</p>
</div>
""", unsafe_allow_html=True)
