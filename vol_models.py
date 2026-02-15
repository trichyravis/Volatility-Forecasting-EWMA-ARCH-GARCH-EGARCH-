"""
Volatility models (EWMA, ARCH, GARCH, EGARCH) and VaR/ES from forecasted volatilities.
Used by the Streamlit app.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from arch import arch_model


def get_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    r = np.asarray(returns)
    n = len(r)
    sigma2 = np.full(n, np.nan)
    sigma2[0] = np.var(r[: min(30, n)])
    for t in range(1, n):
        sigma2[t] = lambda_ * sigma2[t - 1] + (1 - lambda_) * (r[t - 1] ** 2)
    return pd.Series(np.sqrt(sigma2), index=returns.index)


def fit_arch_volatility(returns: pd.Series) -> tuple:
    model = arch_model(returns * 100, mean="Zero", vol="ARCH", p=1, rescale=False)
    fit = model.fit(disp="off")
    vol = fit.conditional_volatility / 100.0
    return vol, fit


def fit_garch_volatility(returns: pd.Series) -> tuple:
    model = arch_model(returns * 100, mean="Zero", vol="Garch", p=1, q=1, rescale=False)
    fit = model.fit(disp="off")
    vol = fit.conditional_volatility / 100.0
    return vol, fit


def fit_egarch_volatility(returns: pd.Series) -> tuple:
    model = arch_model(returns * 100, mean="Zero", vol="EGARCH", p=1, q=1, rescale=False)
    fit = model.fit(disp="off")
    vol = fit.conditional_volatility / 100.0
    return vol, fit


def _egarch_forecast(egarch_fit, horizon: int) -> list:
    if horizon <= 1:
        f = egarch_fit.forecast(horizon=horizon)
        raw = getattr(f, "variance", None)
        v = np.asarray(raw.values if hasattr(raw, "values") else raw).reshape(-1)
    else:
        f = egarch_fit.forecast(horizon=horizon, method="simulation", simulations=1000)
        raw = getattr(f, "variance", None)
        if raw is None:
            raw = getattr(f, "variances", None)
        v_arr = np.asarray(raw.values if hasattr(raw, "values") else raw)
        if v_arr.ndim == 3:
            v = np.mean(v_arr[-1], axis=1)
        elif v_arr.ndim == 2:
            v = v_arr[-1, :] if v_arr.shape[0] >= horizon else v_arr[0, :]
        else:
            v = v_arr.reshape(-1)
    out = [float(np.sqrt(v[h]) / 100.0) for h in range(min(horizon, len(v)))]
    while len(out) < horizon:
        out.append(out[-1] if out else np.nan)
    return out


def run_volatility_pipeline(
    returns: pd.Series,
    ewma_lambda: float = 0.94,
    forecast_horizon: int = 5,
    var_es_alpha: float = 0.95,
) -> dict:
    """
    Run EWMA, ARCH, GARCH, EGARCH; compute in-sample and forecast VaR/ES.
    Returns dict with keys: vol_ewma, vol_arch, vol_garch, vol_egarch,
    arch_ok, garch_ok, egarch_ok, arch_fit, garch_fit, egarch_fit,
    ewma_forecast, arch_forecast, garch_forecast, egarch_forecast,
    var_ewma, es_ewma, var_arch, es_arch, ... (in-sample),
    var_f_ewma, es_f_ewma, ... (forecast lists), z_alpha, es_scale, last_ewma.
    """
    returns = returns.dropna()
    vol_ewma = ewma_volatility(returns, lambda_=ewma_lambda)

    try:
        vol_arch, arch_fit = fit_arch_volatility(returns)
        arch_ok = True
    except Exception:
        vol_arch = pd.Series(np.nan, index=returns.index)
        arch_fit = None
        arch_ok = False

    try:
        vol_garch, garch_fit = fit_garch_volatility(returns)
        garch_ok = True
    except Exception:
        vol_garch = pd.Series(np.nan, index=returns.index)
        garch_fit = None
        garch_ok = False

    try:
        vol_egarch, egarch_fit = fit_egarch_volatility(returns)
        egarch_ok = True
    except Exception:
        vol_egarch = pd.Series(np.nan, index=returns.index)
        egarch_fit = None
        egarch_ok = False

    last_ewma = float(vol_ewma.iloc[-1])
    ewma_forecast = [last_ewma] * forecast_horizon

    arch_forecast = []
    if arch_ok:
        f = arch_fit.forecast(horizon=forecast_horizon)
        for h in range(forecast_horizon):
            arch_forecast.append(float(np.sqrt(f.variance.values[-1, h]) / 100.0))
    else:
        arch_forecast = [np.nan] * forecast_horizon

    garch_forecast = []
    if garch_ok:
        f = garch_fit.forecast(horizon=forecast_horizon)
        for h in range(forecast_horizon):
            garch_forecast.append(float(np.sqrt(f.variance.values[-1, h]) / 100.0))
    else:
        garch_forecast = [np.nan] * forecast_horizon

    egarch_forecast = _egarch_forecast(egarch_fit, forecast_horizon) if egarch_ok else [np.nan] * forecast_horizon

    z_alpha = float(scipy_stats.norm.ppf(var_es_alpha))
    phi_z = float(scipy_stats.norm.pdf(z_alpha))
    es_scale = phi_z / (1 - var_es_alpha)

    var_ewma = z_alpha * vol_ewma
    es_ewma = vol_ewma * es_scale
    var_arch = z_alpha * vol_arch if arch_ok else pd.Series(dtype=float)
    es_arch = vol_arch * es_scale if arch_ok else pd.Series(dtype=float)
    var_garch = z_alpha * vol_garch if garch_ok else pd.Series(dtype=float)
    es_garch = vol_garch * es_scale if garch_ok else pd.Series(dtype=float)
    var_egarch = z_alpha * vol_egarch if egarch_ok else pd.Series(dtype=float)
    es_egarch = vol_egarch * es_scale if egarch_ok else pd.Series(dtype=float)

    var_f_ewma = [z_alpha * x for x in ewma_forecast]
    es_f_ewma = [es_scale * x for x in ewma_forecast]
    var_f_arch = [z_alpha * x if not np.isnan(x) else np.nan for x in arch_forecast]
    es_f_arch = [es_scale * x if not np.isnan(x) else np.nan for x in arch_forecast]
    var_f_garch = [z_alpha * x if not np.isnan(x) else np.nan for x in garch_forecast]
    es_f_garch = [es_scale * x if not np.isnan(x) else np.nan for x in garch_forecast]
    var_f_egarch = [z_alpha * x if not np.isnan(x) else np.nan for x in egarch_forecast]
    es_f_egarch = [es_scale * x if not np.isnan(x) else np.nan for x in egarch_forecast]

    return {
        "returns": returns,
        "vol_ewma": vol_ewma,
        "vol_arch": vol_arch,
        "vol_garch": vol_garch,
        "vol_egarch": vol_egarch,
        "arch_ok": arch_ok,
        "garch_ok": garch_ok,
        "egarch_ok": egarch_ok,
        "arch_fit": arch_fit,
        "garch_fit": garch_fit,
        "egarch_fit": egarch_fit,
        "last_ewma": last_ewma,
        "ewma_forecast": ewma_forecast,
        "arch_forecast": arch_forecast,
        "garch_forecast": garch_forecast,
        "egarch_forecast": egarch_forecast,
        "var_ewma": var_ewma,
        "es_ewma": es_ewma,
        "var_arch": var_arch,
        "es_arch": es_arch,
        "var_garch": var_garch,
        "es_garch": es_garch,
        "var_egarch": var_egarch,
        "es_egarch": es_egarch,
        "var_f_ewma": var_f_ewma,
        "es_f_ewma": es_f_ewma,
        "var_f_arch": var_f_arch,
        "es_f_arch": es_f_arch,
        "var_f_garch": var_f_garch,
        "es_f_garch": es_f_garch,
        "var_f_egarch": var_f_egarch,
        "es_f_egarch": es_f_egarch,
        "z_alpha": z_alpha,
        "es_scale": es_scale,
        "var_es_alpha": var_es_alpha,
        "forecast_horizon": forecast_horizon,
    }
