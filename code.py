# ============================================
# Streamlit App – Grid Search + Best Params + Final Simulation + Sensitivity
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import nbinom
import re

# ============================================
# SECTION 0: UPLOAD FILES
# ============================================
st.set_page_config(page_title="Forecasting App", layout="wide")

st.title("📦 Inventory Forecasting App")

st.sidebar.header("Upload Data")
file_articles = st.sidebar.file_uploader("Upload articles.xlsx", type=["xlsx"])
file_pfe = st.sidebar.file_uploader("Upload PFE HANIN.xlsx", type=["xlsx"])

if not file_articles or not file_pfe:
    st.warning("⚠️ Please upload both Excel files to continue.")
    st.stop()

# Save file paths in memory
EXCEL_PATH = file_articles
PFE_PATH = file_pfe

# Product codes (can be expanded)
CODES_PRODUITS = ["EM0400", "EM1499", "EM1091", "EM1523", "EM0392", "EM1526"]

# Global params
LEAD_TIME = 1
LEAD_TIME_SUPPLIER = 3
SERVICE_LEVEL = 0.95
NB_SIM = 1000
RNG_SEED = 42

# Storage for Grid Search results
if "results_SES" not in st.session_state:
    st.session_state.results_SES = None
if "results_CRO" not in st.session_state:
    st.session_state.results_CRO = None
if "results_SBA" not in st.session_state:
    st.session_state.results_SBA = None
if "best_params" not in st.session_state:
    st.session_state.best_params = None

# ============================================
# UTIL FUNCTIONS (SES, CROSTON, SBA)
# ============================================
def ses_forecast_array(x, alpha):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if len(x) == 0:
        return {"forecast_per_period": 0.0}
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return {"forecast_per_period": float(l)}

def croston_or_sba_forecast_array(x, alpha, variant="croston"):
    x = pd.Series(x).fillna(0.0).astype(float).values
    x = np.where(x < 0, 0.0, x)
    if (x == 0).all():
        return {"forecast_per_period": 0.0}
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
    if variant.lower() == "sba":
        f *= (1 - alpha / 2.0)
    return {"forecast_per_period": float(f)}

def compute_metrics(df_run):
    if df_run.empty or "forecast_error" not in df_run:
        return np.nan, np.nan, np.nan, np.nan
    e = df_run["forecast_error"].astype(float)
    ME = e.mean()
    absME = e.abs().mean()
    MSE = (e**2).mean()
    RMSE = np.sqrt(MSE)
    return ME, absME, MSE, RMSE

# ============================================
# TAB 1: GRID SEARCH
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["Grid Search", "Best Params", "Final Simulation", "Sensitivity"])

with tab1:
    st.header("🔍 Grid Search")

    if st.button("▶️ Run Grid Search (SES, Croston, SBA)"):
        # Load matrix
        df = pd.read_excel(EXCEL_PATH, sheet_name="classification")
        prod_col = df.columns[0]

        ALPHAS = [0.1, 0.2, 0.3]
        WINDOW_RATIOS = [0.6, 0.8]
        RECALC_INTERVALS = [5, 10]

        results_SES, results_CRO, results_SBA = [], [], []

        for code in CODES_PRODUITS:
            row = df.loc[df[prod_col] == code]
            if row.empty:
                continue
            series = row.drop(columns=[prod_col]).T.squeeze()
            series.index = pd.to_datetime(series.index)
            daily = series.sort_index()
            values = daily.values

            for a in ALPHAS:
                for w in WINDOW_RATIOS:
                    for itv in RECALC_INTERVALS:
                        split_index = int(len(values) * w)
                        if split_index < 2: 
                            continue
                        train = values[:split_index]
                        real_demand = values[split_index] if split_index < len(values) else 0
                        
                        # SES
                        f_ses = ses_forecast_array(train, a)["forecast_per_period"]
                        err_ses = real_demand - f_ses
                        results_SES.append({"code": code, "alpha": a, "window": w, "interval": itv, "forecast_error": err_ses})

                        # Croston
                        f_cro = croston_or_sba_forecast_array(train, a, "croston")["forecast_per_period"]
                        err_cro = real_demand - f_cro
                        results_CRO.append({"code": code, "alpha": a, "window": w, "interval": itv, "forecast_error": err_cro})

                        # SBA
                        f_sba = croston_or_sba_forecast_array(train, a, "sba")["forecast_per_period"]
                        err_sba = real_demand - f_sba
                        results_SBA.append({"code": code, "alpha": a, "window": w, "interval": itv, "forecast_error": err_sba})

        st.session_state.results_SES = pd.DataFrame(results_SES)
        st.session_state.results_CRO = pd.DataFrame(results_CRO)
        st.session_state.results_SBA = pd.DataFrame(results_SBA)

        st.success("✅ Grid Search Completed")

    if st.session_state.results_SES is not None:
        st.subheader("SES Results (sample)")
        st.dataframe(st.session_state.results_SES.head(20))

# ============================================
# TAB 2: BEST PARAMS
# ============================================
with tab2:
    st.header("🏆 Best Parameters per Product")

    if st.session_state.results_SES is None:
        st.info("Run Grid Search first.")
    else:
        def pick_best(df):
            return df.groupby("code")["forecast_error"].mean().reset_index()

        best_ses = pick_best(st.session_state.results_SES)
        best_cro = pick_best(st.session_state.results_CRO)
        best_sba = pick_best(st.session_state.results_SBA)

        st.write("### SES")
        st.dataframe(best_ses)
        st.write("### Croston")
        st.dataframe(best_cro)
        st.write("### SBA")
        st.dataframe(best_sba)

        best_all = pd.concat([best_ses.assign(method="SES"),
                              best_cro.assign(method="Croston"),
                              best_sba.assign(method="SBA")])
        st.session_state.best_params = best_all
        st.write("### Combined Best")
        st.dataframe(best_all)

# ============================================
# TAB 3: FINAL SIMULATION
# ============================================
with tab3:
    st.header("📊 Final Simulation")

    if st.session_state.best_params is None:
        st.info("Run Grid Search + Best Params first.")
    else:
        st.write("Using best parameters to run simulation (dummy logic here).")
        st.dataframe(st.session_state.best_params)

# ============================================
# TAB 4: SENSITIVITY
# ============================================
with tab4:
    st.header("📈 Sensitivity Analysis")

    if st.session_state.best_params is None:
        st.info("Run Grid Search first.")
    else:
        for sl in [0.90, 0.95, 0.98]:
            st.subheader(f"Service Level {sl*100:.0f}%")
            df = st.session_state.best_params.copy()
            df["service_level"] = sl
            st.dataframe(df)
