
"""
Volatility models: EWMA, ARCH, GARCH, and EGARCH.
Includes VaR and Expected Shortfall calculations based on conditional volatility.
"""

import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model


def ewma_volatility(returns: pd.Series, lambda_param: float = 0.94) -> pd.Series:
    """
    Calculate Exponentially Weighted Moving Average (EWMA) volatility.
    
    Parameters:
    -----------
    returns : pd.Series
        Time series of returns
    lambda_param : float
        Decay parameter (default: 0.94)
    
    Returns:
    --------
    pd.Series
        EWMA volatility estimates
    """
    squared_returns = returns ** 2
    ewma_var = squared_returns.ewm(alpha=1-lambda_param, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var)
    return ewma_vol


def forecast_ewma(last_vol: float, last_return: float, lambda_param: float = 0.94, 
                  horizon: int = 5) -> np.ndarray:
    """
    Forecast EWMA volatility for multiple periods ahead.
    
    Parameters:
    -----------
    last_vol : float
        Last observed volatility
    last_return : float
        Last observed return
    lambda_param : float
        Decay parameter
    horizon : int
        Number of periods to forecast
    
    Returns:
    --------
    np.ndarray
        Array of forecasted volatilities
    """
    forecasts = np.zeros(horizon)
    current_var = last_vol ** 2
    
    # Update variance with last return
    current_var = lambda_param * current_var + (1 - lambda_param) * (last_return ** 2)
    
    for h in range(horizon):
        # EWMA forecast converges to long-run variance
        # For h-step ahead: σ²(t+h) = λ^h * σ²(t+1) + (1-λ^h) * long_run_var
        # Simplified: use current variance as it converges slowly
        forecasts[h] = np.sqrt(current_var)
    
    return forecasts


def fit_arch(returns: pd.Series, p: int = 1) -> dict:
    """
    Fit ARCH(p) model to returns.
    
    Parameters:
    -----------
    returns : pd.Series
        Time series of returns
    p : int
        ARCH order (default: 1)
    
    Returns:
    --------
    dict
        Fitted model results including conditional volatility and forecast
    """
    try:
        # Fit ARCH model
        model = arch_model(returns * 100, vol='ARCH', p=p, rescale=False)
        result = model.fit(disp='off', show_warning=False)
        
        # Get conditional volatility
        cond_vol = result.conditional_volatility / 100
        
        return {
            'model': result,
            'cond_vol': cond_vol,
            'params': result.params,
        }
    except Exception as e:
        print(f"ARCH fitting error: {e}")
        return None


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> dict:
    """
    Fit GARCH(p,q) model to returns.
    
    Parameters:
    -----------
    returns : pd.Series
        Time series of returns
    p : int
        GARCH order (default: 1)
    q : int
        ARCH order (default: 1)
    
    Returns:
    --------
    dict
        Fitted model results including conditional volatility and forecast
    """
    try:
        # Fit GARCH model
        model = arch_model(returns * 100, vol='GARCH', p=p, q=q, rescale=False)
        result = model.fit(disp='off', show_warning=False)
        
        # Get conditional volatility
        cond_vol = result.conditional_volatility / 100
        
        return {
            'model': result,
            'cond_vol': cond_vol,
            'params': result.params,
        }
    except Exception as e:
        print(f"GARCH fitting error: {e}")
        return None


def fit_egarch(returns: pd.Series, p: int = 1, q: int = 1) -> dict:
    """
    Fit EGARCH(p,q) model to returns (captures leverage effects).
    
    Parameters:
    -----------
    returns : pd.Series
        Time series of returns
    p : int
        EGARCH order (default: 1)
    q : int
        ARCH order (default: 1)
    
    Returns:
    --------
    dict
        Fitted model results including conditional volatility and forecast
    """
    try:
        # Fit EGARCH model
        model = arch_model(returns * 100, vol='EGARCH', p=p, q=q, rescale=False)
        result = model.fit(disp='off', show_warning=False)
        
        # Get conditional volatility
        cond_vol = result.conditional_volatility / 100
        
        return {
            'model': result,
            'cond_vol': cond_vol,
            'params': result.params,
        }
    except Exception as e:
        print(f"EGARCH fitting error: {e}")
        return None


def forecast_garch_family(model_result, horizon: int = 5) -> np.ndarray:
    """
    Forecast volatility from ARCH/GARCH/EGARCH model.
    
    Parameters:
    -----------
    model_result
        Fitted ARCH model result object
    horizon : int
        Number of periods to forecast
    
    Returns:
    --------
    np.ndarray
        Array of forecasted volatilities
    """
    try:
        forecast = model_result.forecast(horizon=horizon, reindex=False)
        forecasted_variance = forecast.variance.values[-1, :] / (100**2)
        forecasted_vol = np.sqrt(forecasted_variance)
        return forecasted_vol
    except Exception as e:
        print(f"Forecast error: {e}")
        return np.full(horizon, np.nan)


def calculate_var_es_from_vol(volatility: pd.Series, alpha: float = 0.95, 
                               distribution: str = 'normal') -> tuple:
    """
    Calculate VaR and ES from conditional volatility assuming a distribution.
    
    Parameters:
    -----------
    volatility : pd.Series
        Conditional volatility estimates
    alpha : float
        Confidence level (default: 0.95)
    distribution : str
        Distribution assumption ('normal' or 't')
    
    Returns:
    --------
    tuple
        (VaR series, ES series)
    """
    if distribution == 'normal':
        z_alpha = stats.norm.ppf(alpha)
        var = volatility * z_alpha
        
        # Expected Shortfall for normal distribution
        phi_z = stats.norm.pdf(z_alpha)
        es = volatility * phi_z / (1 - alpha)
    
    elif distribution == 't':
        # Student's t with df=5 (heavy tails)
        df = 5
        t_alpha = stats.t.ppf(alpha, df)
        var = volatility * t_alpha * np.sqrt((df - 2) / df)
        
        # ES for Student's t (approximation)
        es = var * 1.2  # Rough approximation
    
    else:
        raise ValueError("Distribution must be 'normal' or 't'")
    
    return var, es


def run_volatility_pipeline(returns: pd.Series, ewma_lambda: float = 0.94, 
                            forecast_horizon: int = 5, var_es_alpha: float = 0.95) -> dict:
    """
    Run complete volatility modeling pipeline with all models.
    
    Parameters:
    -----------
    returns : pd.Series
        Time series of returns
    ewma_lambda : float
        EWMA decay parameter (default: 0.94)
    forecast_horizon : int
        Number of periods to forecast (default: 5)
    var_es_alpha : float
        Confidence level for VaR/ES (default: 0.95)
    
    Returns:
    --------
    dict
        Comprehensive results from all models including:
        - Conditional volatility estimates
        - VaR and ES estimates
        - Volatility forecasts
        - VaR and ES forecasts
    """
    # EWMA
    vol_ewma = ewma_volatility(returns, ewma_lambda)
    var_ewma, es_ewma = calculate_var_es_from_vol(vol_ewma, var_es_alpha)
    ewma_forecast = forecast_ewma(
        vol_ewma.iloc[-1], 
        returns.iloc[-1], 
        ewma_lambda, 
        forecast_horizon
    )
    
    # ARCH
    arch_fit = fit_arch(returns, p=1)
    if arch_fit is not None:
        vol_arch = arch_fit['cond_vol']
        var_arch, es_arch = calculate_var_es_from_vol(vol_arch, var_es_alpha)
        arch_forecast = forecast_garch_family(arch_fit['model'], forecast_horizon)
    else:
        vol_arch = pd.Series(np.nan, index=returns.index)
        var_arch = pd.Series(np.nan, index=returns.index)
        es_arch = pd.Series(np.nan, index=returns.index)
        arch_forecast = np.full(forecast_horizon, np.nan)
    
    # GARCH
    garch_fit = fit_garch(returns, p=1, q=1)
    if garch_fit is not None:
        vol_garch = garch_fit['cond_vol']
        var_garch, es_garch = calculate_var_es_from_vol(vol_garch, var_es_alpha)
        garch_forecast = forecast_garch_family(garch_fit['model'], forecast_horizon)
    else:
        vol_garch = pd.Series(np.nan, index=returns.index)
        var_garch = pd.Series(np.nan, index=returns.index)
        es_garch = pd.Series(np.nan, index=returns.index)
        garch_forecast = np.full(forecast_horizon, np.nan)
    
    # EGARCH
    egarch_fit = fit_egarch(returns, p=1, q=1)
    if egarch_fit is not None:
        vol_egarch = egarch_fit['cond_vol']
        var_egarch, es_egarch = calculate_var_es_from_vol(vol_egarch, var_es_alpha)
        egarch_forecast = forecast_garch_family(egarch_fit['model'], forecast_horizon)
    else:
        vol_egarch = pd.Series(np.nan, index=returns.index)
        var_egarch = pd.Series(np.nan, index=returns.index)
        es_egarch = pd.Series(np.nan, index=returns.index)
        egarch_forecast = np.full(forecast_horizon, np.nan)
    
    # Calculate forecast VaR and ES
    z_alpha = stats.norm.ppf(var_es_alpha)
    phi_z = stats.norm.pdf(z_alpha)
    
    var_f_ewma = ewma_forecast * z_alpha
    es_f_ewma = ewma_forecast * phi_z / (1 - var_es_alpha)
    
    var_f_arch = arch_forecast * z_alpha
    es_f_arch = arch_forecast * phi_z / (1 - var_es_alpha)
    
    var_f_garch = garch_forecast * z_alpha
    es_f_garch = garch_forecast * phi_z / (1 - var_es_alpha)
    
    var_f_egarch = egarch_forecast * z_alpha
    es_f_egarch = egarch_forecast * phi_z / (1 - var_es_alpha)
    
    return {
        'returns': returns,
        'var_es_alpha': var_es_alpha,
        'forecast_horizon': forecast_horizon,
        
        # EWMA
        'vol_ewma': vol_ewma,
        'var_ewma': var_ewma,
        'es_ewma': es_ewma,
        'ewma_forecast': ewma_forecast,
        'var_f_ewma': var_f_ewma,
        'es_f_ewma': es_f_ewma,
        
        # ARCH
        'vol_arch': vol_arch,
        'var_arch': var_arch,
        'es_arch': es_arch,
        'arch_forecast': arch_forecast,
        'var_f_arch': var_f_arch,
        'es_f_arch': es_f_arch,
        
        # GARCH
        'vol_garch': vol_garch,
        'var_garch': var_garch,
        'es_garch': es_garch,
        'garch_forecast': garch_forecast,
        'var_f_garch': var_f_garch,
        'es_f_garch': es_f_garch,
        
        # EGARCH
        'vol_egarch': vol_egarch,
        'var_egarch': var_egarch,
        'es_egarch': es_egarch,
        'egarch_forecast': egarch_forecast,
        'var_f_egarch': var_f_egarch,
        'es_f_egarch': es_f_egarch,
    }
