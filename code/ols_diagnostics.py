# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 22:37:51 2025

@author: epicx

OLS diagnostics helpers.

Usage:
    from ols_diagnostics import compute_diagnostics, save_diagnostic_report
"""


import os
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor


def compute_diagnostics(model, nlags: int = 4) -> dict:
    """
    Compute core OLS diagnostics for a fitted statsmodels OLS result.

    Returns a dict with:
      - n, k
      - adj_r2
      - rmse
      - jb_stat, jb_p
      - bp_stat, bp_p
      - bg_stat, bg_p
      - dw
      - max_cooks_d, num_high_cooks
      - vif: {var_name: vif_value}
    """
    resid = model.resid
    fitted = model.fittedvalues
    y = model.model.endog
    X = model.model.exog
    exog_names = model.model.exog_names

    n = len(resid)
    k = X.shape[1]

    # basic fit metrics
    rmse = float(np.sqrt(np.mean(resid**2)))
    adj_r2 = float(model.rsquared_adj)

    # normality (Jarque-Bera)
    jb_stat, jb_p, _, _ = jarque_bera(resid)

    # heteroskedasticity (Breusch-Pagan)
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)

    # autocorrelation (Breusch-Godfrey, up to nlags)
    bg_res = acorr_breusch_godfrey(model, nlags=nlags)
    bg_stat, bg_p = bg_res[0], bg_res[1]

    # Durbin-Watson
    from statsmodels.stats.stattools import durbin_watson
    dw = float(durbin_watson(resid))

    # influence / Cook's distance
    infl = OLSInfluence(model)
    cooks_d, _ = infl.cooks_distance
    max_cooks = float(np.max(cooks_d))
    # simple rule-of-thumb threshold
    thr = 4.0 / n
    num_high_cooks = int(np.sum(cooks_d > thr))

    # VIF (skip constant, assumed to be first column)
    vif = {}
    for i in range(1, X.shape[1]):
        vif_val = variance_inflation_factor(X, i)
        vif[exog_names[i]] = float(vif_val)

    return {
        "n": n,
        "k": k,
        "adj_r2": adj_r2,
        "rmse": rmse,
        "jb_stat": float(jb_stat),
        "jb_p": float(jb_p),
        "bp_stat": float(bp_stat),
        "bp_p": float(bp_p),
        "bg_stat": float(bg_stat),
        "bg_p": float(bg_p),
        "dw": dw,
        "max_cooks_d": max_cooks,
        "num_high_cooks": num_high_cooks,
        "cooks_threshold": thr,
        "vif": vif,
    }


def save_diagnostic_report(model, diag: dict, model_name: str, out_dir: str):
    """
    Save a text summary + a couple of basic plots for diagnostics:

      - residuals vs fitted
      - Q-Q plot of residuals
      - ACF of residuals

    Files:
      {model_name}_diagnostics.txt
      {model_name}_resid_vs_fitted.png
      {model_name}_qq.png
      {model_name}_acf.png
    """
    import statsmodels.api as sm

    os.makedirs(out_dir, exist_ok=True)

    # ---------- text report ----------
    txt_path = os.path.join(out_dir, f"{model_name}_diagnostics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(model.summary().as_text())
        f.write("\n\n")
        f.write("Diagnostic summary:\n")
        for k, v in diag.items():
            if k == "vif":
                continue
            f.write(f"  {k}: {v}\n")
        f.write("\nVIF:\n")
        for var, vif_val in diag["vif"].items():
            f.write(f"  {var}: {vif_val:0.3f}\n")
    print(f"[Diagnostics] Wrote text report: {txt_path}")

    resid = model.resid
    fitted = model.fittedvalues

    # ---------- residuals vs fitted ----------
    plt.figure()
    plt.scatter(fitted, resid, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Fitted ({model_name})")
    png1 = os.path.join(out_dir, f"{model_name}_resid_vs_fitted.png")
    plt.tight_layout()
    plt.savefig(png1, dpi=150)
    plt.close()
    print(f"[Diagnostics] Saved: {png1}")

    # ---------- Q-Q plot ----------
    plt.figure()
    sm.ProbPlot(resid).qqplot(line="45")
    plt.title(f"Q-Q Plot of Residuals ({model_name})")
    png2 = os.path.join(out_dir, f"{model_name}_qq.png")
    plt.tight_layout()
    plt.savefig(png2, dpi=150)
    plt.close()
    print(f"[Diagnostics] Saved: {png2}")

    # ---------- ACF of residuals ----------
    from statsmodels.graphics.tsaplots import plot_acf

    plt.figure()
    plot_acf(resid, lags=40)
    plt.title(f"ACF of Residuals ({model_name})")
    png3 = os.path.join(out_dir, f"{model_name}_acf.png")
    plt.tight_layout()
    plt.savefig(png3, dpi=150)
    plt.close()
    print(f"[Diagnostics] Saved: {png3}")

