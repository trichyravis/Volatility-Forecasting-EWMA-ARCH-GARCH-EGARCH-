
"""
Streamlit app: EVT VaR/ES and Volatility (EWMA, ARCH, GARCH, EGARCH) for Nifty 50 stocks.
Robust production-safe version.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from data_utils import NIFTY50_LIST, fetch_prices
from evt_model import get_returns, run_evt_var_es, losses_from_returns
from vol_models import run_volatility_pipeline

st.set_page_config(page_title="Nifty 50 — VaR, ES & Volatility", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")

ticker_options = ["^NSEI (Nifty 50 Index)"] + [
    f"{name} ({ticker})" for ticker, name in NIFTY50_LIST
]

ticker_display = st.sidebar.selectbox("Select stock or index", ticker_options, index=0)

if ticker_display.startswith("^"):
    TICKER = "^NSEI"
else:
    TICKER = ticker_display.split("(")[-1].rstrip(")")

years = st.sidebar.slider("Years of history", 1.0, 5.0, 3.0, 0.5)

st.sidebar.subheader("EVT (VaR & Expected Shortfall)")
evt_alpha = st.sidebar.slider("EVT confidence level (α)", 0.90, 0.99, 0.95, 0.01)
evt_threshold_q = st.sidebar.slider("EVT threshold quantile", 0.85, 0.95, 0.90, 0.01)

st.sidebar.subheader("Volatility models")
ewma_lambda = st.sidebar.slider("EWMA λ", 0.90, 0.98, 0.94, 0.01)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 1, 10, 5)
var_es_alpha = st.sidebar.slider("VaR/ES confidence (vol models)", 0.90, 0.99, 0.95, 0.01)


# ---------------- Utility Helpers ----------------
def safe_get(arr, h):
    try:
        return arr[h]
    except Exception:
        return np.nan


def safe_array(arr):
    if arr is None:
        return np.array([])
    return np.array(arr).flatten()


@st.cache_data(ttl=3600)
def load_data(_ticker: str, _years: float):
    prices = fetch_prices(_ticker, _years)
    returns = get_returns(prices)
    return prices, returns.dropna()


# ---------------- Load Data ----------------
try:
    prices, returns = load_data(TICKER, years)
    n_returns = len(returns)

    if n_returns < 100:
        st.error("Not enough return observations for volatility modelling (need at least 100).")
        st.stop()

except Exception as e:
    st.error(f"Could not load data for {TICKER}: {e}")
    st.stop()

st.title(f"{TICKER} — VaR, Expected Shortfall & Volatility")
st.caption(f"Using {n_returns} daily returns ({years} years)")

tab1, tab2 = st.tabs(["EVT VaR & Expected Shortfall", "Volatility & VaR/ES"])


# ==========================================================
# ======================= TAB 1 =============================
# ==========================================================
with tab1:

    try:
        result = run_evt_var_es(
            returns,
            alpha=evt_alpha,
            threshold_quantile=evt_threshold_q,
        )

        pct = evt_alpha * 100

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("VaR (% loss)", f"{result['var']*100:.2f}%", help=f"{pct:.0f}% confidence")
        col2.metric("Expected Shortfall (% loss)", f"{result['es']*100:.2f}%", help=f"{pct:.0f}%")
        col3.metric("GPD shape (ξ)", f"{result['xi']:.4f}")
        col4.metric("Threshold (u)", f"{result['threshold']:.4f}",
                    f"{result['n_exceedances']} exceedances")

        losses = losses_from_returns(returns)
        u = result["threshold"]
        var_val, es_val = result["var"], result["es"]
        xi, sigma = result["xi"], result["sigma"]

        exceedances = losses[losses > u] - u

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax1, ax2 = axes

        # Histogram
        ax1.hist(losses, bins=50, density=True,
                 color="steelblue", alpha=0.7, edgecolor="white")
        ax1.axvline(u, linestyle="--")
        ax1.axvline(var_val)
        ax1.axvline(es_val)
        ax1.set_title("Loss Distribution")

        # Tail fit
        if len(exceedances) > 10 and not np.isnan(xi):
            x_gpd = np.linspace(0, exceedances.max() * 1.05, 200)
            gpd_pdf = stats.genpareto.pdf(x_gpd, xi, loc=0, scale=sigma)
            ax2.hist(exceedances, bins=30, density=True, alpha=0.6)
            ax2.plot(x_gpd, gpd_pdf)

        ax2.set_title("GPD Tail Fit")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"EVT failed: {e}")


# ==========================================================
# ======================= TAB 2 =============================
# ==========================================================
with tab2:

    try:
        vol = run_volatility_pipeline(
            returns,
            ewma_lambda=ewma_lambda,
            forecast_horizon=forecast_horizon,
            var_es_alpha=var_es_alpha,
        )

        required_keys = [
            "ewma_forecast", "arch_forecast", "garch_forecast", "egarch_forecast",
            "var_f_ewma", "var_f_arch", "var_f_garch", "var_f_egarch",
            "es_f_ewma", "es_f_arch", "es_f_garch", "es_f_egarch"
        ]

        for key in required_keys:
            if key not in vol:
                st.error(f"Missing key in volatility output: {key}")
                st.stop()

        H = vol["forecast_horizon"]
        pct = vol["var_es_alpha"] * 100

        # Convert forecasts safely
        ewma_f = safe_array(vol["ewma_forecast"])
        arch_f = safe_array(vol["arch_forecast"])
        garch_f = safe_array(vol["garch_forecast"])
        egarch_f = safe_array(vol["egarch_forecast"])

        # ---------------- Volatility Table ----------------
        df_vol = pd.DataFrame({
            "EWMA": [safe_get(ewma_f, h)*np.sqrt(252)*100 for h in range(H)],
            "ARCH": [safe_get(arch_f, h)*np.sqrt(252)*100 for h in range(H)],
            "GARCH": [safe_get(garch_f, h)*np.sqrt(252)*100 for h in range(H)],
            "EGARCH": [safe_get(egarch_f, h)*np.sqrt(252)*100 for h in range(H)],
        }, index=[f"Day {h+1}" for h in range(H)])

        st.subheader("Volatility Forecast (Annualized %)")
        st.dataframe(df_vol.round(2), use_container_width=True)

        # ---------------- VaR Forecast Table ----------------
        df_var = pd.DataFrame({
            "VaR_EWMA": [safe_get(vol["var_f_ewma"], h)*100 for h in range(H)],
            "VaR_ARCH": [safe_get(vol["var_f_arch"], h)*100 for h in range(H)],
            "VaR_GARCH": [safe_get(vol["var_f_garch"], h)*100 for h in range(H)],
            "VaR_EGARCH": [safe_get(vol["var_f_egarch"], h)*100 for h in range(H)],
        }, index=[f"Day {h+1}" for h in range(H)])

        st.subheader(f"VaR Forecast ({pct:.0f}%)")
        st.dataframe(df_var.round(2), use_container_width=True)

        # ---------------- Plot Forecast ----------------
        x_fore = np.arange(1, H + 1)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_fore, df_vol["EWMA"])
        ax.plot(x_fore, df_vol["ARCH"])
        ax.plot(x_fore, df_vol["GARCH"])
        ax.plot(x_fore, df_vol["EGARCH"])
        ax.set_xlabel("Days Ahead")
        ax.set_ylabel("Volatility (% ann.)")
        ax.set_title("Volatility Forecast")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Volatility pipeline failed: {e}")
        import traceback
        st.code(traceback.format_exc())
