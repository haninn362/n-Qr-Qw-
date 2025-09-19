# ============================================================
# Streamlit App: Classification & Optimisation (Part 1)
# ============================================================
# - Imports
# - Helpers
# - Classification (ABC-XYZ)
# - Q* Computations
# - Forecasting methods (SES, Croston, SBA)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import re
from scipy.stats import nbinom

# ============================================================
# Global Parameters
# ============================================================

DELAI_USINE = 10
DELAI_FOURNISSEUR = 3
NIVEAU_SERVICE_DEF = 0.95
NB_SIM = 1000
GRAINE_ALEA = 42

COLONNES_AFFICHAGE = [
    "date", "code", "methode", "intervalle",
    "demande_reelle", "stock_disponible_intervalle", "stock_apres_intervalle",
    "politique_commande", "Qr_etoile", "Qw_etoile", "n_etoile",
    "ROP_usine", "SS_usine",
    "ROP_fournisseur", "SS_fournisseur",
    "statut_stock", "service_level"
]

# ============================================================
# Helpers
# ============================================================

def _disp(obj, n=None, title=None):
    """Generic display for Streamlit or console fallback."""
    try:
        if title:
            st.subheader(title)
        if isinstance(obj, pd.DataFrame):
            if n:
                st.dataframe(obj.head(n))
            else:
                st.dataframe(obj)
        else:
            st.write(obj)
    except Exception as e:
        st.write(f"[Display error] {e}")
        st.write(obj)

def _find_first_col(df, pattern):
    """Find first column whose name matches regex pattern."""
    for col in df.columns:
        if re.search(pattern, str(col), re.IGNORECASE):
            return col
    raise ValueError(f"No column matching {pattern}")

# ============================================================
# Classification (ABC-XYZ)
# ============================================================

def classify_articles(file_articles):
    """Classify articles (ABC-XYZ)."""
    xls = pd.ExcelFile(file_articles)
    sheet = "articles" if "articles" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(file_articles, sheet_name=sheet)

    code_col = _find_first_col(df, r"code")
    demand_col = _find_first_col(df, r"quant|demande")
    cost_col = _find_first_col(df, r"cout|price|unit")

    df["valeur"] = df[demand_col] * df[cost_col]
    df = df.sort_values("valeur", ascending=False).reset_index(drop=True)
    df["cum_valeur"] = df["valeur"].cumsum()
    df["pct_cum"] = df["cum_valeur"] / df["valeur"].sum()

    df["ABC"] = pd.cut(
        df["pct_cum"],
        bins=[0, 0.8, 0.95, 1.0],
        labels=["A", "B", "C"],
        include_lowest=True,
    )

    cv = df[demand_col].std() / (df[demand_col].mean() + 1e-9)
    df["XYZ"] = pd.cut(
        [cv] * len(df),
        bins=[-np.inf, 0.5, 1.0, np.inf],
        labels=["X", "Y", "Z"],
    )

    df["class"] = df["ABC"].astype(str) + df["XYZ"].astype(str)
    return df[[code_col, "valeur", "ABC", "XYZ", "class"]]

# ============================================================
# Q* Computations
# ============================================================

def _trouver_feuille_produit(chemin_excel: str, code: str) -> str:
    xls = pd.ExcelFile(chemin_excel)
    feuilles = xls.sheet_names
    cible = f"time serie {code}"
    if cible in feuilles:
        return cible
    patt = re.compile(r"time\s*ser(i|ie)s?\s*", re.IGNORECASE)
    cand = [s for s in feuilles if patt.search(s) and code.lower() in s.lower()]
    if cand:
        return sorted(cand, key=len, reverse=True)[0]
    alt = f"time series {code}"
    if alt in feuilles:
        return alt
    raise ValueError(f"Onglet pour '{code}' introuvable.")

def compute_qstars(chemin_excel: str, codes: list):
    """Calcule Qr*, Qw*, n* pour chaque article."""
    df_conso = pd.read_excel(chemin_excel, sheet_name="consommation depots externe")
    df_conso = df_conso.groupby("Code Produit")["Quantite STIAL"].sum()

    qr_map, qw_map, n_map = {}, {}, {}
    for code in codes:
        feuille = _trouver_feuille_produit(chemin_excel, code)
        df = pd.read_excel(chemin_excel, sheet_name=feuille)

        C_r = df["Cr : cout stockage/article "].iloc[0]
        C_w = df["Cw : cout stockage\nchez F"].iloc[0]
        A_w = df["Aw : cout de\nlancement chez U"].iloc[0]
        A_r = df["Ar : cout de \nlancement chez F"].iloc[0]

        n_val = (A_w * C_r) / (A_r * C_w)
        n_val = 1 if n_val < 1 else round(n_val)
        n1, n2 = int(n_val), int(n_val) + 1
        F_n1 = (A_r + A_w / n1) * (n1 * C_w + C_r)
        F_n2 = (A_r + A_w / n2) * (n2 * C_w + C_r)
        n_star = n1 if F_n1 <= F_n2 else n2

        D = df_conso.get(code, 0)
        tau = 1
        Qr_star = ((2 * (A_r + A_w / n_star) * D) / (n_star * C_w + C_r * tau)) ** 0.5
        Qw_star = n_star * Qr_star

        qr_map[code] = round(float(Qr_star), 2)
        qw_map[code] = round(float(Qw_star), 2)
        n_map[code] = int(n_star)
    return qr_map, qw_map, n_map

# ============================================================
# Forecasting Methods
# ============================================================

def _croston_or_sba(x, alpha: float, variant: str = "sba"):
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
    if variant.lower() == "sba":
        f *= (1 - alpha / 2.0)
    return float(f)

def _ses(x, alpha: float):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if len(x) == 0:
        return 0.0
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return float(l)
# ============================================================
# Streamlit App: Classification & Optimisation (Part 2)
# ============================================================
# - Rolling Simulation
# - Final & Sensitivity
# - Streamlit UI (upload 2 files, run all steps)
# ============================================================

# ============================================================
# Rolling Simulation
# ============================================================

def _series_conso_stock_jour(chemin_excel: str, feuille: str):
    df = pd.read_excel(chemin_excel, sheet_name=feuille)
    col_date, col_stock, col_conso = df.columns[0], df.columns[1], df.columns[2]

    dates = pd.to_datetime(df[col_date], errors="coerce")
    conso = pd.to_numeric(df[col_conso], errors="coerce").fillna(0.0).astype(float)
    stock = pd.to_numeric(df[col_stock], errors="coerce").fillna(0.0).astype(float)

    ts_conso = pd.DataFrame({"d": dates, "q": conso}).dropna().sort_values("d").set_index("d")["q"]
    ts_stock = pd.DataFrame({"d": dates, "s": stock}).dropna().sort_values("d").set_index("d")["s"]

    min_date = min(ts_conso.index.min(), ts_stock.index.min())
    max_date = max(ts_conso.index.max(), ts_stock.index.max())
    idx_complet = pd.date_range(min_date, max_date, freq="D")

    conso_jour = ts_conso.reindex(idx_complet, fill_value=0.0)
    stock_jour = ts_stock.reindex(idx_complet).ffill().fillna(0.0)
    return conso_jour, stock_jour

def _somme_intervalle(serie: pd.Series, start_idx: int, intervalle: int) -> float:
    s, e = start_idx + 1, start_idx + 1 + int(intervalle)
    return float(pd.Series(serie).iloc[s:e].sum())

def rolling_with_new_logic(
    excel_path, product_code, alpha, window_ratio, intervalle,
    delai_usine, delai_fournisseur, service_level, nb_sim, rng_seed,
    variant, qr_map, qw_map, n_map
):
    feuille = _trouver_feuille_produit(excel_path, product_code)
    conso_jour, stock_jour = _series_conso_stock_jour(excel_path, feuille)
    vals = conso_jour.values
    split_index = int(len(vals) * float(window_ratio))
    if split_index < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(rng_seed)
    lignes = []
    stock_apres_intervalle = 0.0

    for i in range(split_index, len(vals)):
        if (i - split_index) % int(intervalle) == 0:
            train = vals[:i]
            date_test = conso_jour.index[i]

            if variant == "sba":
                f = _croston_or_sba(train, alpha, "sba")
            elif variant == "croston":
                f = _croston_or_sba(train, alpha, "croston")
            else:
                f = _ses(train, alpha)

            sigma = float(pd.Series(train).std(ddof=1)) if i > 1 else 0.0
            sigma = sigma if np.isfinite(sigma) else 0.0

            demande_reelle = _somme_intervalle(conso_jour, i, intervalle)
            stock_dispo = _somme_intervalle(stock_jour, i, intervalle)
            stock_apres_intervalle = stock_apres_intervalle + stock_dispo - demande_reelle

            X_Lt = delai_usine * f
            sigma_Lt = sigma * np.sqrt(max(delai_usine, 1e-9))
            var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt + 1e-5
            p_nb = min(max(X_Lt / var_u, 1e-12), 1 - 1e-12)
            r_nb = X_Lt**2 / (var_u - X_Lt) if var_u > X_Lt else 1e6
            ROP_u = float(np.percentile(nbinom.rvs(r_nb, p_nb, size=nb_sim, random_state=rng), 100 * service_level))
            SS_u = max(ROP_u - X_Lt, 0.0)

            totalL = delai_usine + delai_fournisseur
            X_Lt_Lw = totalL * f
            sigma_Lt_Lw = sigma * np.sqrt(max(totalL, 1e-9))
            var_f = sigma_Lt_Lw**2 if sigma_Lt_Lw**2 > X_Lt_Lw else X_Lt_Lw + 1e-5
            p_nb_f = min(max(X_Lt_Lw / var_f, 1e-12), 1 - 1e-12)
            r_nb_f = X_Lt_Lw**2 / (var_f - X_Lt_Lw) if var_f > X_Lt_Lw else 1e6
            ROP_f = float(np.percentile(nbinom.rvs(r_nb_f, p_nb_f, size=nb_sim, random_state=rng), 100 * service_level))
            SS_f = max(ROP_f - X_Lt_Lw, 0.0)

            ROP_u_interval = ROP_u * (intervalle / max(delai_usine, 1e-9))

            if stock_apres_intervalle >= demande_reelle * delai_usine:
                politique = "pas_de_commande"
            else:
                politique = f"commander_Qr*_{qr_map[product_code]}"

            statut = "rupture" if demande_reelle > ROP_u_interval else "holding"

            lignes.append({
                "date": date_test.date(),
                "code": product_code,
                "methode": variant,
                "intervalle": int(intervalle),
                "demande_reelle": float(demande_reelle),
                "stock_disponible_intervalle": float(stock_dispo),
                "stock_apres_intervalle": float(stock_apres_intervalle),
                "politique_commande": politique,
                "Qr_etoile": float(qr_map[product_code]),
                "Qw_etoile": float(qw_map[product_code]),
                "n_etoile": int(n_map[product_code]),
                "ROP_usine": float(ROP_u),
                "SS_usine": float(SS_u),
                "ROP_fournisseur": float(ROP_f),
                "SS_fournisseur": float(SS_f),
                "statut_stock": statut,
                "service_level": float(service_level),
            })

    return pd.DataFrame(lignes)

# ============================================================
# Final + Sensitivity
# ============================================================

def run_final_once(best_per_code: pd.DataFrame, excel_opt, service_level=NIVEAU_SERVICE_DEF):
    qr_map, qw_map, n_map = compute_qstars(excel_opt, best_per_code["code"].tolist())
    results = []
    for _, row in best_per_code.iterrows():
        code = row["code"]
        method = str(row["method"]).lower()
        alpha = float(row["alpha"])
        window_ratio = float(row["window_ratio"])
        intervalle = int(row["recalc_interval"])
        df_run = rolling_with_new_logic(
            excel_path=excel_opt,
            product_code=code,
            alpha=alpha, window_ratio=window_ratio, intervalle=intervalle,
            delai_usine=DELAI_USINE, delai_fournisseur=DELAI_FOURNISSEUR,
            service_level=service_level, nb_sim=NB_SIM, rng_seed=GRAINE_ALEA,
            variant=method, qr_map=qr_map, qw_map=qw_map, n_map=n_map
        )
        results.append(df_run)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def run_sensitivity(best_per_code: pd.DataFrame, excel_opt, service_levels=[0.90, 0.92, 0.95, 0.98]):
    qr_map, qw_map, n_map = compute_qstars(excel_opt, best_per_code["code"].tolist())
    all_results = []
    for sl in service_levels:
        runs = []
        for _, row in best_per_code.iterrows():
            code = row["code"]
            method = str(row["method"]).lower()
            alpha = float(row["alpha"])
            window_ratio = float(row["window_ratio"])
            intervalle = int(row["recalc_interval"])
            df_run = rolling_with_new_logic(
                excel_path=excel_opt,
                product_code=code,
                alpha=alpha, window_ratio=window_ratio, intervalle=intervalle,
                delai_usine=DELAI_USINE, delai_fournisseur=DELAI_FOURNISSEUR,
                service_level=sl, nb_sim=NB_SIM, rng_seed=GRAINE_ALEA,
                variant=method, qr_map=qr_map, qw_map=qw_map, n_map=n_map
            )
            df_run["service_level"] = sl
            runs.append(df_run)
        if runs:
            all_results.append(pd.concat(runs, ignore_index=True))
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# ============================================================
# Streamlit UI
# ============================================================

def main():
    st.title("📦 Classification & Optimisation App")
    st.write("Upload **two Excel files**: one for classification (articles.xlsx) and one for optimisation (PFE HANIN.xlsx).")

    file_articles = st.file_uploader("Upload Articles Excel", type=["xlsx"], key="articles")
    file_optim = st.file_uploader("Upload Optimisation Excel", type=["xlsx"], key="optim")

    if file_articles and file_optim:
        st.success("✅ Both files uploaded successfully.")

        # Step 1: Classification
        st.header("Step 1: Classification (ABC-XYZ)")
        df_class = classify_articles(file_articles)
        _disp(df_class, n=10, title="Classification Results")

        # Step 2: Q* Computation
        st.header("Step 2: Q* Computation")
        codes = df_class["code"].tolist()
        qr_map, qw_map, n_map = compute_qstars(file_optim, codes)
        st.write("**Qr\***", qr_map)
        st.write("**Qw\***", qw_map)
        st.write("**n\***", n_map)

        # Step 3: Final Simulation
        st.header("Step 3: Final Simulation (Service Level 95%)")
        best_per_code = pd.DataFrame({
            "code": codes,
            "method": ["ses"] * len(codes),
            "alpha": [0.2] * len(codes),
            "window_ratio": [0.7] * len(codes),
            "recalc_interval": [7] * len(codes),
        })
        final_results = run_final_once(best_per_code, file_optim, service_level=NIVEAU_SERVICE_DEF)
        _disp(final_results, n=10, title="Final Results")

        # Step 4: Sensitivity Analysis
        st.header("Step 4: Sensitivity Analysis")
        sensi = run_sensitivity(best_per_code, file_optim, service_levels=[0.90, 0.92, 0.95, 0.98])
        if not sensi.empty:
            summary = sensi.groupby(["code", "service_level"]).agg(
                ROP_u_moy=("ROP_usine", "mean"),
                SS_u_moy=("SS_usine", "mean"),
                ROP_f_moy=("ROP_fournisseur", "mean"),
                SS_f_moy=("SS_fournisseur", "mean"),
                holding_pct=("statut_stock", lambda s: (s == "holding").mean()*100),
                rupture_pct=("statut_stock", lambda s: (s == "rupture").mean()*100),
                Qr_star=("Qr_etoile", "first"),
                Qw_star=("Qw_etoile", "first"),
                n_star=("n_etoile", "first"),
            ).reset_index()
            _disp(summary, n=50, title="Sensitivity Summary")

if __name__ == "__main__":
    main()
