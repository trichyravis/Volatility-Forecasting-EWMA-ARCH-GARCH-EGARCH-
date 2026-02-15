
"""
Streamlit app: EVT VaR/ES and Volatility (EWMA, ARCH, GARCH, EGARCH) for Nifty 50 stocks.
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

# ---------- Sidebar ----------
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


@st.cache_data(ttl=3600)
def load_data(_ticker: str, _years: float):
    prices = fetch_prices(_ticker, _years)
    returns = get_returns(prices)
    return prices, returns.dropna()


# Load data once
try:
    prices, returns = load_data(TICKER, years)
    n_returns = len(returns)
except Exception as e:
    st.error(f"Could not load data for {TICKER}: {e}")
    st.stop()

st.title(f"{TICKER} — VaR, Expected Shortfall & Volatility")
st.caption(f"Using {n_returns} daily returns ({years} years)")

tab1, tab2 = st.tabs(["EVT VaR & Expected Shortfall", "Volatility & VaR/ES"])

# ---------- Tab 1: EVT ----------
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
        col4.metric("Threshold (u)", f"{result['threshold']:.4f}", f"{result['n_exceedances']} exceedances")

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

# ---------- Tab 2: Volatility & VaR/ES ----------
with tab2:
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
