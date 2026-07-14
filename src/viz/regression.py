"""
Generic scatter-plot regression annotation helper (OLS / Theil-Sen / RANSAC),
used across case-study scatter panels to add a fit line plus CC/MAE/RMSE/NRMSE
text annotation without repeating the same ~15 lines at every call site.
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, LinearRegression


def add_fit_metrics(ax, x, y, show=('nrmse', 'cc'), ret=('cc', 'nrmse'), nrmse='std',
                    pos=(0.03, 0.97), fs=13, line=True, use_xlim=False, fit='ols',
                    fit_use_xlim=False, trim_q=None, lw=1):
    """
    Fit y ~ x on ax (OLS/Theil-Sen/RANSAC), draw the fit line, and annotate
    CC/MAE/RMSE/NRMSE/N in the corner.

    fit      : 'ols' | 'theilsen' | 'ransac'
    show/ret : which metrics to display / return -- subset of
               {'cc','mae','rmse','nrmse','n'}, or a single string, or None.
    nrmse    : normalization for NRMSE -- 'std' | 'range' | 'mean'.
    trim_q   : optional (low, high) percentile pair to trim x before fitting.
    use_xlim / fit_use_xlim : restrict the drawn line / the fit itself to the
               axes' current x-limits (e.g. after set_xlim()).

    Returns the requested metric(s) per `ret`, or None if ret is None.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)

    if trim_q is not None:
        lo, hi = np.percentile(x[m], trim_q)
        m = m & (x >= lo) & (x <= hi)
    if fit_use_xlim:
        x0, x1 = ax.get_xlim()
        m = m & (x >= x0) & (x <= x1)

    x, y = x[m], y[m]

    if fit == 'ols':
        a, b = np.polyfit(x, y, 1)
    elif fit in ('theilsen', 'ransac'):
        X = x.reshape(-1, 1)
        model = (TheilSenRegressor(random_state=0) if fit == 'theilsen'
                else RANSACRegressor(estimator=LinearRegression(), random_state=0)).fit(X, y)
        a = float(model.estimator_.coef_[0]) if fit == 'ransac' else float(model.coef_[0])
        b = float(model.estimator_.intercept_) if fit == 'ransac' else float(model.intercept_)
    else:
        raise ValueError("fit must be one of: 'ols', 'theilsen', 'ransac'")

    yhat = a * x + b
    cc, _ = pearsonr(x, y)
    mae = mean_absolute_error(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    denom = {'std': np.std(y), 'range': (y.max() - y.min()), 'mean': np.mean(y)}[nrmse]
    nrmse_v = rmse / denom if denom != 0 else np.nan

    metrics = {'cc': cc, 'mae': mae, 'rmse': rmse, 'nrmse': nrmse_v, 'a': a, 'b': b, 'n': len(x)}

    if line:
        if use_xlim:
            x0p, x1p = ax.get_xlim()
            pad = 0.03 * (x1p - x0p)
            x0p, x1p = x0p + pad, x1p - pad
        else:
            x0p, x1p = x.min(), x.max()
        xx = np.linspace(x0p, x1p, 200)
        ax.plot(xx, a * xx + b, color='red', lw=lw)

    if show:
        fmt = {'cc': 'CC$={:.2f}$', 'mae': 'MAE$={:.1f}$', 'rmse': 'RMSE$={:.1f}$',
              'nrmse': 'NRMSE$={:.3f}$', 'n': 'N$={:d}$'}
        ax.text(pos[0], pos[1], '\n'.join(fmt[k].format(metrics[k]) for k in show),
               transform=ax.transAxes, va='top', ha='left', fontsize=fs)

    if ret is None:
        return None
    if isinstance(ret, str):
        return metrics[ret]
    return {k: metrics[k] for k in ret}
