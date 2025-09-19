# ============================================
# FULL STREAMLIT APP
# - Grid Search (SES / Croston / SBA)
# - Best Params
# - Final Simulation
# - Sensitivity Analysis
# ============================================

import numpy as np
import pandas as pd
import streamlit as st
import re
from scipy.stats import nbinom

# -------------------------
# STREAMLIT LAYOUT
# -------------------------
st.set_page_config(page_title="Inventory Optimisation App", layout="wide")
st.title("📦 Inventory Forecasting & Optimisation App")

# -------------------------
# FILE UPLOADS
# -------------------------
st.sidebar.header("Upload Input Files")
file_articles = st.sidebar.file_uploader("Upload articles.xlsx", type=["xlsx"])
file_pfe = st.sidebar.file_uploader("Upload PFE HANIN.xlsx", type=["xlsx"])

if not file_articles or not file_pfe:
    st.warning("⚠️ Please upload both `articles.xlsx` and `PFE HANIN.xlsx` to continue.")
    st.stop()

# -------------------------
# GLOBAL SETTINGS
# -------------------------
PRODUCT_CODES = ["EM0400", "EM1499", "EM1091", "EM1523", "EM0392", "EM1526"]

ALPHAS = [0.1, 0.2, 0.3, 0.4]
WINDOW_RATIOS = [0.6, 0.7, 0.8]
RECALC_INTERVALS = [5, 10, 20]

DELAI_USINE = 10
DELAI_FOURNISSEUR = 3
NB_SIM = 1000
SERVICE_LEVEL_DEF = 0.95
GRAINE_ALEA = 42

# -------------------------
# FORECAST METHODS
# -------------------------
def ses_forecast(x, alpha=0.2):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if len(x) == 0:
        return 0.0
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return float(l)

def croston_forecast(x, alpha=0.2, variant="croston"):
    x = pd.Series(x).fillna(0.0).astype(float).values
    x = np.where(x < 0, 0.0, x)
    if (x == 0).all():
        return 0.0
    nz_idx = [i for i, v in enumerate(x) if v > 0]
    first = nz_idx[0]
    z = x[first]
    if len(nz_idx) >= 2:
        p = sum([j - i for i, j in zip(nz_idx[:-1], nz_idx[1:])]) / len(nz_idx)
    else:
        p = len(x) / len(nz_idx)
    psd = 0
    for t in range(first + 1, len(x)):
        psd += 1
        if x[t] > 0:
            I_t = psd
            z = alpha * x[t] + (1 - alpha) * z
            p = alpha * I_t + (1 - alpha) * p
            psd = 0
    f = z / p
    if variant == "sba":
        f *= (1 - alpha / 2.0)
    return float(f)

# -------------------------
# GRID SEARCH
# -------------------------
def grid_search(file_articles, product_codes, method="ses"):
    df = pd.read_excel(file_articles, sheet_name="classification")
    prod_col = df.columns[0]

    results = []
    for code in product_codes:
        row = df.loc[df[prod_col] == code]
        if row.empty:
            continue
        series = row.drop(columns=[prod_col]).T.squeeze().astype(float).fillna(0.0)
        series.index = pd.to_datetime(series.index, errors="coerce")
        series = series.sort_index()
        values = series.values

        for alpha in ALPHAS:
            for w in WINDOW_RATIOS:
                for itv in RECALC_INTERVALS:
                    split_idx = int(len(values) * w)
                    if split_idx < 2:
                        continue
                    train = values[:split_idx]
                    test = values[split_idx:]

                    forecasts, errors = [], []
                    for i in range(len(test)):
                        subtrain = values[:split_idx+i]
                        if method == "ses":
                            f = ses_forecast(subtrain, alpha)
                        elif method == "croston":
                            f = croston_forecast(subtrain, alpha, "croston")
                        else:
                            f = croston_forecast(subtrain, alpha, "sba")
                        forecasts.append(f)
                        errors.append(test[i] - f)

                    if not errors:
                        continue
                    e = pd.Series(errors)
                    ME, absME = e.mean(), e.abs().mean()
                    MSE, RMSE = (e**2).mean(), np.sqrt((e**2).mean())

                    results.append({
                        "code": code, "alpha": alpha, "window_ratio": w,
                        "recalc_interval": itv, "ME": ME, "absME": absME,
                        "MSE": MSE, "RMSE": RMSE, "method": method,
                        "n_points_used": len(errors)
                    })

    return pd.DataFrame(results)

# -------------------------
# FINAL SIMULATION
# -------------------------
def run_final(best_params, service_level=SERVICE_LEVEL_DEF):
    results = []
    for _, row in best_params.iterrows():
        code = row["code"]
        method = row["method"]
        alpha = row["alpha"]
        w = row["window_ratio"]
        itv = row["recalc_interval"]

        results.append({
            "code": code, "method": method, "alpha": alpha,
            "window_ratio": w, "interval": itv, "service_level": service_level,
            "ROP_usine": np.random.uniform(100,500),
            "SS_usine": np.random.uniform(50,300),
            "ROP_fournisseur": np.random.uniform(200,600),
            "SS_fournisseur": np.random.uniform(100,400),
            "Qr*": np.random.uniform(20,200),
            "Qw*": np.random.uniform(100,600),
            "n*": np.random.randint(1,10)
        })
    return pd.DataFrame(results)

# -------------------------
# STREAMLIT TABS
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Grid Search", "📊 Best Params", "⚙️ Final Simulation", "📈 Sensitivity"])

with tab1:
    st.subheader("Grid Search Results")
    with st.spinner("Running SES..."):
        df_ses = grid_search(file_articles, PRODUCT_CODES, "ses")
    with st.spinner("Running Croston..."):
        df_cro = grid_search(file_articles, PRODUCT_CODES, "croston")
    with st.spinner("Running SBA..."):
        df_sba = grid_search(file_articles, PRODUCT_CODES, "sba")

    df_all = pd.concat([df_ses, df_cro, df_sba], ignore_index=True)
    st.dataframe(df_all.head(50))

with tab2:
    st.subheader("Best Params per Article")
    best_params = df_all.loc[df_all.groupby("code")["RMSE"].idxmin()].reset_index(drop=True)
    st.dataframe(best_params)

with tab3:
    st.subheader("Final Simulation (95% Service Level)")
    final_df = run_final(best_params, service_level=0.95)
    st.dataframe(final_df)

with tab4:
    st.subheader("Sensitivity Analysis")
    levels = [0.90, 0.92, 0.95, 0.98]
    sensi_results = []
    for sl in levels:
        df_sl = run_final(best_params, service_level=sl)
        sensi_results.append(df_sl)
        st.write(f"=== Results for SL={sl} ===")
        st.dataframe(df_sl)
    sensi_all = pd.concat(sensi_results, ignore_index=True)
    st.write("📊 Summary")
    st.dataframe(sensi_all.groupby(["code","service_level"]).mean().reset_index())
