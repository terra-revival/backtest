import numpy as np


def calibrate_gbm_params(series):
    log_mu = calibrate_mu(series)
    sigma = calibrate_sigma(series)
    return log_mu + 0.5 * sigma**2, sigma


def calibrate_mu(prices, dt, n):
    T = dt * n
    ts = np.linspace(dt, T, n)
    log_prices = np.log(prices)
    total = (1.0 / dt) * (ts**2).sum()
    return (1.0 / total) * (1.0 / dt) * (ts * log_prices).sum()


def calibrate_sigma(prices, dt, n):
    return np.sqrt((np.diff(prices) ** 2).sum() / (n * dt))
