# Nifty 50 — VaR, Expected Shortfall & Volatility (Streamlit)

Streamlit app for **EVT VaR/ES** and **volatility forecasting (EWMA, ARCH, GARCH, EGARCH)** with **VaR & Expected Shortfall** from forecasted volatilities, using Nifty 50 stocks or the index.

## Files

| File | Purpose |
|------|--------|
| `app.py` | Main Streamlit application (run this) |
| `data_utils.py` | Nifty 50 list and `fetch_prices()` |
| `evt_model.py` | EVT (POT-GPD) VaR & Expected Shortfall |
| `vol_models.py` | EWMA, ARCH, GARCH, EGARCH and VaR/ES from vol |
| `requirements_streamlit.txt` | Python dependencies for the app |

## Setup

```bash
cd evt_var_es
pip install -r requirements_streamlit.txt
```

## Run the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## Usage

1. **Sidebar**
   - **Select stock or index** — Nifty 50 Index (^NSEI) or any of the 50 Nifty stocks.
   - **Years of history** — 1–5 years (slider).
   - **EVT** — Confidence level (α) and threshold quantile for the EVT tab.
   - **Volatility** — EWMA λ, forecast horizon (days), and VaR/ES confidence for the Volatility tab.

2. **Tab: EVT VaR & Expected Shortfall**
   - VaR and Expected Shortfall from the POT-GPD model.
   - Metrics (VaR, ES, GPD ξ, threshold).
   - Two plots: loss distribution with VaR/ES/threshold, and tail exceedances with GPD fit.

3. **Tab: Volatility & VaR/ES**
   - Volatility forecasts (EWMA, ARCH, GARCH, EGARCH) and tables.
   - VaR/ES from forecasted volatilities (normal assumption).
   - Plots: returns + conditional volatility + vol forecast; in-sample and forecast VaR/ES.

Data is cached for 1 hour so changing only the sidebar and rerunning is fast.
