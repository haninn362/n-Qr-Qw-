# ============================================
# Streamlit App – Classification + Optimisation + Sensibilité
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.stats import nbinom

# ======================================================
# Helper / Debug Functions
# ======================================================
def _disp(obj, n=None, title=None):
    """Generic display that works in Streamlit."""
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
        st.write(f"⚠️ Display error: {e}")
        st.write(obj)

def _find_first_col(df, substrings):
    """Find first column in df whose name contains any of substrings."""
    for col in df.columns:
        for sub in substrings:
            if sub.lower() in str(col).lower():
                return col
    return None

# ======================================================
# Part 1 – Classification of Articles
# ======================================================
def classify_articles(file_articles):
    """
    Classification ABC/XYZ on 'articles.xlsx'
    """
    df = pd.read_excel(file_articles, sheet_name="articles")

    # Find columns dynamically
    col_code = _find_first_col(df, ["code", "produit", "article"])
    col_val = _find_first_col(df, ["valeur", "sales", "chiffre"])

    if not col_code or not col_val:
        raise ValueError("❌ Impossible de trouver colonnes code/valeur dans articles.xlsx")

    # ABC classification
    df_sorted = df.sort_values(by=col_val, ascending=False).reset_index(drop=True)
    df_sorted["cum_val"] = df_sorted[col_val].cumsum()
    df_sorted["cum_pct"] = df_sorted["cum_val"] / df_sorted[col_val].sum()
    df_sorted["ABC"] = pd.cut(
        df_sorted["cum_pct"],
        bins=[0, 0.8, 0.95, 1.0],
        labels=["A", "B", "C"],
        include_lowest=True,
    )

    # XYZ classification: coefficient de variation fictif (here: random or std-based)
    if "consommation" in df.columns:
        cv = df["consommation"].std() / (df["consommation"].mean() + 1e-9)
    else:
        cv = np.random.rand()

    if cv < 0.5:
        xyz = "X"
    elif cv < 1:
        xyz = "Y"
    else:
        xyz = "Z"

    df_sorted["XYZ"] = xyz

    return df_sorted[[col_code, col_val, "ABC", "XYZ"]]

def run_classification(file_articles):
    st.header("📊 Classification ABC/XYZ")
    df_class = classify_articles(file_articles)
    _disp(df_class, n=20, title="Résultats classification")
    return df_class

# ======================================================
# Part 2 – Optimisation (SES/Croston/SBA + Q*, ROP/SS, Sensibilité)
# ======================================================

# --- PARAMETERS ---
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

# --- A. Compute Q* ---
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
    raise ValueError(f"[Feuille] Onglet pour '{code}' introuvable.")

def compute_qstars(chemin_excel: str, codes: list):
    df_conso = pd.read_excel(chemin_excel, sheet_name="consommation depots externe")
    df_conso = df_conso.groupby('Code Produit')['Quantite STIAL'].sum()

    qr_map, qw_map, n_map = {}, {}, {}
    for code in codes:
        feuille = _trouver_feuille_produit(chemin_excel, code)
        df = pd.read_excel(chemin_excel, sheet_name=feuille)

        C_r = df.filter(like="Cr").iloc[0,0]
        C_w = df.filter(like="Cw").iloc[0,0]
        A_w = df.filter(like="Aw").iloc[0,0]
        A_r = df.filter(like="Ar").iloc[0,0]

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
        n_map[code]  = int(n_star)
    return qr_map, qw_map, n_map

# --- B. Forecasting ---
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

# --- C. Rolling Simulation ---
def rolling_with_new_logic(excel_path, product_code, alpha, window_ratio, intervalle,
    delai_usine, delai_fournisseur, service_level, nb_sim, rng_seed,
    variant, qr_map, qw_map, n_map):

    feuille = _trouver_feuille_produit(excel_path, product_code)
    df = pd.read_excel(excel_path, sheet_name=feuille)
    col_date = df.columns[0]
    col_stock = df.columns[1]
    col_conso = df.columns[2]

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

            demande_reelle = conso_jour.iloc[i : i + int(intervalle)].sum()
            stock_dispo = stock_jour.iloc[i : i + int(intervalle)].sum()
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

# --- D. Sensitivity ---
def run_final_once(best_per_code, file_data, service_level=NIVEAU_SERVICE_DEF):
    qr_map, qw_map, n_map = compute_qstars(file_data, best_per_code["code"].tolist())
    results = []
    for _, row in best_per_code.iterrows():
        df_run = rolling_with_new_logic(
            excel_path=file_data,
            product_code=row["code"],
            alpha=float(row["alpha"]),
            window_ratio=float(row["window_ratio"]),
            intervalle=int(row["recalc_interval"]),
            delai_usine=DELAI_USINE, delai_fournisseur=DELAI_FOURNISSEUR,
            service_level=service_level, nb_sim=NB_SIM, rng_seed=GRAINE_ALEA,
            variant=row["method"], qr_map=qr_map, qw_map=qw_map, n_map=n_map
        )
        results.append(df_run)
        _disp(df_run[COLONNES_AFFICHAGE], n=10, title=f"{row['code']} — {row['method'].upper()} (SL={service_level:.2f})")
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

# ======================================================
# MAIN APP
# ======================================================
def main():
    st.title("📦 Optimisation Supply Chain – Classification & Simulation")

    st.sidebar.header("1. Upload input files")
    file_articles = st.sidebar.file_uploader("Upload articles.xlsx", type="xlsx")
    file_data = st.sidebar.file_uploader("Upload PFE HANIN (1).xlsx", type="xlsx")

    if not file_articles or not file_data:
        st.info("➡️ Please upload both Excel files to start.")
        return

    st.sidebar.header("2. Choose analysis")
    run_class = st.sidebar.checkbox("Run Classification ABC/XYZ", value=True)
    run_opt = st.sidebar.checkbox("Run Optimisation & Sensitivity", value=True)

    if run_class:
        df_class = run_classification(file_articles)

    if run_opt:
        st.header("⚙️ Simulation / Optimisation")
        # For now: fake best_per_code (normally from param files)
        best_per_code = pd.DataFrame({
            "code": ["EM0400", "EM1499"],
            "alpha": [0.3, 0.2],
            "window_ratio": [0.7, 0.6],
            "recalc_interval": [5, 7],
            "method": ["ses", "sba"]
        })
        final_95 = run_final_once(best_per_code, file_data, service_level=NIVEAU_SERVICE_DEF)
        _disp(final_95, n=50, title="Résumé résultats (95%)")

if __name__ == "__main__":
    main()
