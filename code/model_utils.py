# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 10:39:27 2025

@author: epicx
"""

# model_utils.py
import re
import pandas as pd

def coeff_table(model, drop_const=False):
    params = model.params
    bse = model.bse
    tvals = model.tvalues
    pvals = model.pvalues

    df_coef = pd.DataFrame({
        "param": params.index,
        "coef": params.values,
        "std_err": bse.values,
        "t": tvals.values,
        "pvalue": pvals.values,
    })

    if drop_const:
        df_coef = df_coef[df_coef["param"] != "const"]

    return df_coef


def sig_code(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "."
    else:
        return ""


def add_to_summary(summary_rows, model, dep_var, label):
    coef_df = coeff_table(model, drop_const=True)

    row = {
        "model_label": label,
        "dep_var": dep_var,
        "r2": model.rsquared,
        "r2_adj": model.rsquared_adj,
        "n_obs": int(model.nobs),
    }

    for i, (_, r) in enumerate(coef_df.iterrows(), start=1):
        row[f"coef_{i}"] = r["coef"]
        row[f"var_{i}"] = r["param"]
        row[f"sig_{i}"] = sig_code(r["pvalue"])
        row[f"p_{i}"] = r["pvalue"]

    summary_rows.append(row)


def clean_sheet_name(label: str) -> str:
    name = re.sub(r"[\[\]\:\*\?\/\\]", "_", label)
    return name[:31]
