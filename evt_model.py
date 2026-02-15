"""
Extreme Value Theory (EVT) model using Peaks Over Threshold (POT)
with Generalized Pareto Distribution (GPD) for VaR and Expected Shortfall.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional


def get_returns(prices: pd.Series, log_returns: bool = True) -> pd.Series:
    """Compute returns from price series."""
    if log_returns:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def losses_from_returns(returns: pd.Series) -> np.ndarray:
    """Convert returns to losses (positive = loss). L = -R."""
    return -np.asarray(returns, dtype=float)


def fit_gpd(exceedances: np.ndarray) -> Tuple[float, float]:
    """
    Fit Generalized Pareto Distribution to exceedances (values above threshold).
    GPD has shape xi (ξ) and scale sigma (σ). scipy uses c = xi, scale = sigma.
    Returns (xi, sigma).
    """
    if len(exceedances) < 10:
        raise ValueError("Need at least 10 exceedances to fit GPD.")
    # scipy: genpareto(c) with c = xi; fit to exceedances
    xi, loc, sigma = stats.genpareto.fit(exceedances, floc=0)
    return float(xi), float(sigma)


def pot_var_es(
    losses: np.ndarray,
    threshold: float,
    alpha: float,
    xi: Optional[float] = None,
    sigma: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """
    Compute VaR and Expected Shortfall using POT-GPD.

    Parameters
    ----------
    losses : array of losses (positive values)
    threshold : threshold u for exceedances
    alpha : confidence level for VaR/ES (e.g. 0.95 or 0.99)
    xi, sigma : GPD parameters; if None, estimated from exceedances

    Returns
    -------
    var : Value at Risk at level alpha (loss quantile)
    es : Expected Shortfall (average loss given loss > VaR)
    xi_hat : fitted shape
    sigma_hat : fitted scale
    """
    exceedances = losses[losses > threshold] - threshold
    n = len(losses)
    n_u = len(exceedances)

    if n_u < 5:
        # Fallback: empirical VaR and ES
        var = np.quantile(losses, alpha)
        tail = losses[losses >= var]
        es = tail.mean() if len(tail) > 0 else var
        return var, es, np.nan, np.nan

    if xi is None or sigma is None:
        xi, sigma = fit_gpd(exceedances)

    # VaR: quantile of loss distribution
    # P(L > VaR) = 1 - alpha => (n_u/n) * (1 - F_gpd(VaR - u)) = 1 - alpha
    # F_gpd(x) = 1 - (1 + xi*x/sigma)^(-1/xi) for xi != 0
    # VaR_alpha = u + (sigma/xi) * ( ((n_u/n)/(1-alpha))^xi - 1 )
    zeta = n_u / n
    if abs(xi) < 1e-8:  # xi ≈ 0: exponential limit
        var = threshold + sigma * np.log(zeta / (1 - alpha))
    else:
        var = threshold + (sigma / xi) * (((zeta / (1 - alpha)) ** xi) - 1)

    # Expected Shortfall: E[L | L > VaR] = (VaR + sigma - xi*threshold) / (1 - xi) for xi < 1
    if xi >= 1:
        es = np.nan  # ES not defined for xi >= 1
    else:
        es = (var + sigma - xi * threshold) / (1 - xi)

    return float(var), float(es), float(xi), float(sigma)


def select_threshold(
    losses: np.ndarray,
    quantile: float = 0.90,
) -> float:
    """Choose threshold as a quantile of the loss distribution (e.g. 90% or 95%)."""
    return float(np.quantile(losses, quantile))


def run_evt_var_es(
    returns: pd.Series,
    alpha: float = 0.95,
    threshold_quantile: float = 0.90,
    threshold: Optional[float] = None,
) -> dict:
    """
    Run full EVT pipeline: losses -> threshold -> GPD -> VaR & ES.

    Parameters
    ----------
    returns : time series of returns (e.g. log returns)
    alpha : VaR/ES confidence level (e.g. 0.95 for 95% VaR)
    threshold_quantile : quantile for threshold if threshold is None
    threshold : optional fixed threshold; if None, use quantile-based

    Returns
    -------
    dict with keys: var, es, xi, sigma, threshold, n_exceedances, alpha
    """
    losses = losses_from_returns(returns)
    if threshold is None:
        threshold = select_threshold(losses, quantile=threshold_quantile)

    var, es, xi, sigma = pot_var_es(losses, threshold, alpha)
    n_exceedances = np.sum(losses > threshold)

    return {
        "var": var,
        "es": es,
        "xi": xi,
        "sigma": sigma,
        "threshold": threshold,
        "n_exceedances": int(n_exceedances),
        "n_obs": len(losses),
        "alpha": alpha,
    }
