
"""
Extreme Value Theory (EVT) model for VaR and Expected Shortfall estimation.
Uses Generalized Pareto Distribution (GPD) for tail modeling.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


def get_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from prices.
    
    Parameters:
    -----------
    prices : pd.Series
        Time series of prices
    
    Returns:
    --------
    pd.Series
        Log returns
    """
    returns = np.log(prices / prices.shift(1))
    returns.name = 'returns'
    return returns


def losses_from_returns(returns: pd.Series) -> np.ndarray:
    """
    Convert returns to losses (negative returns).
    
    Parameters:
    -----------
    returns : pd.Series
        Time series of returns
    
    Returns:
    --------
    np.ndarray
        Array of losses (positive values represent losses)
    """
    return -returns.values


def fit_gpd(exceedances: np.ndarray) -> tuple:
    """
    Fit Generalized Pareto Distribution to exceedances.
    
    Parameters:
    -----------
    exceedances : np.ndarray
        Exceedances over threshold (loss - threshold)
    
    Returns:
    --------
    tuple
        (xi, sigma) - shape and scale parameters
    """
    if len(exceedances) < 10:
        return np.nan, np.nan
    
    try:
        # Method of moments initial estimates
        mean_exc = np.mean(exceedances)
        var_exc = np.var(exceedances)
        
        # Initial parameter estimates
        xi_init = 0.5 * (1 - (mean_exc**2) / var_exc)
        sigma_init = 0.5 * mean_exc * (1 + (mean_exc**2) / var_exc)
        
        # Bounds to ensure valid GPD parameters
        bounds = [(-0.5, 0.5), (1e-6, None)]
        
        # Negative log-likelihood function
        def neg_log_likelihood(params):
            xi, sigma = params
            if sigma <= 0:
                return 1e10
            
            if abs(xi) < 1e-6:
                # Exponential case (xi ≈ 0)
                ll = -len(exceedances) * np.log(sigma) - np.sum(exceedances) / sigma
            else:
                z = 1 + xi * exceedances / sigma
                if np.any(z <= 0):
                    return 1e10
                ll = -len(exceedances) * np.log(sigma) - (1 + 1/xi) * np.sum(np.log(z))
            
            return -ll
        
        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0=[xi_init, sigma_init],
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if result.success:
            xi, sigma = result.x
            return xi, sigma
        else:
            # Fallback to scipy's built-in fit
            params = stats.genpareto.fit(exceedances, floc=0)
            xi, sigma = params[0], params[2]
            return xi, sigma
            
    except Exception as e:
        print(f"GPD fitting error: {e}")
        # Fallback to scipy's built-in fit
        try:
            params = stats.genpareto.fit(exceedances, floc=0)
            xi, sigma = params[0], params[2]
            return xi, sigma
        except:
            return np.nan, np.nan


def calculate_var_es_evt(losses: np.ndarray, threshold: float, xi: float, sigma: float, 
                         n_exceedances: int, n_total: int, alpha: float) -> tuple:
    """
    Calculate VaR and ES using EVT approach.
    
    Parameters:
    -----------
    losses : np.ndarray
        Array of all losses
    threshold : float
        Threshold value (u)
    xi : float
        GPD shape parameter
    sigma : float
        GPD scale parameter
    n_exceedances : int
        Number of exceedances over threshold
    n_total : int
        Total number of observations
    alpha : float
        Confidence level (e.g., 0.95)
    
    Returns:
    --------
    tuple
        (VaR, ES) at the specified confidence level
    """
    # Probability of exceeding threshold
    p_u = n_exceedances / n_total
    
    # Adjust alpha for the conditional distribution
    alpha_adj = (alpha - (1 - p_u)) / p_u
    
    if alpha_adj <= 0 or alpha_adj >= 1:
        # If alpha is below the threshold, use empirical quantile
        var = np.quantile(losses, alpha)
        es = np.mean(losses[losses >= var]) if np.any(losses >= var) else var
        return var, es
    
    # VaR using GPD
    if abs(xi) < 1e-6:
        # Exponential case
        var = threshold + sigma * (-np.log(1 - alpha_adj))
    else:
        var = threshold + (sigma / xi) * ((1 / (1 - alpha_adj))**xi - 1)
    
    # Expected Shortfall using GPD
    if xi < 1:
        if abs(xi) < 1e-6:
            # Exponential case
            es = var + sigma
        else:
            es = (var + sigma - xi * threshold) / (1 - xi)
    else:
        # For xi >= 1, ES is undefined; use a large multiplier of VaR
        es = var * 1.5
    
    return var, es


def run_evt_var_es(returns: pd.Series, alpha: float = 0.95, 
                   threshold_quantile: float = 0.90) -> dict:
    """
    Run complete EVT analysis for VaR and Expected Shortfall.
    
    Parameters:
    -----------
    returns : pd.Series
        Time series of returns
    alpha : float
        Confidence level for VaR/ES (default: 0.95)
    threshold_quantile : float
        Quantile for threshold selection (default: 0.90)
    
    Returns:
    --------
    dict
        Results containing:
        - var: Value at Risk
        - es: Expected Shortfall
        - threshold: Threshold value (u)
        - xi: GPD shape parameter
        - sigma: GPD scale parameter
        - n_exceedances: Number of exceedances
    """
    # Convert to losses
    losses = losses_from_returns(returns)
    
    # Determine threshold
    threshold = np.quantile(losses, threshold_quantile)
    
    # Get exceedances
    exceedances = losses[losses > threshold] - threshold
    n_exceedances = len(exceedances)
    n_total = len(losses)
    
    if n_exceedances < 10:
        # Not enough exceedances - fall back to empirical quantile
        var = np.quantile(losses, alpha)
        es = np.mean(losses[losses >= var]) if np.any(losses >= var) else var
        
        return {
            'var': var,
            'es': es,
            'threshold': threshold,
            'xi': np.nan,
            'sigma': np.nan,
            'n_exceedances': n_exceedances
        }
    
    # Fit GPD to exceedances
    xi, sigma = fit_gpd(exceedances)
    
    if np.isnan(xi) or np.isnan(sigma):
        # Fitting failed - use empirical quantile
        var = np.quantile(losses, alpha)
        es = np.mean(losses[losses >= var]) if np.any(losses >= var) else var
    else:
        # Calculate VaR and ES using EVT
        var, es = calculate_var_es_evt(losses, threshold, xi, sigma, 
                                       n_exceedances, n_total, alpha)
    
    return {
        'var': var,
        'es': es,
        'threshold': threshold,
        'xi': xi,
        'sigma': sigma,
        'n_exceedances': n_exceedances
    }
