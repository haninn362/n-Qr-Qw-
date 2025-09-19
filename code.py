# ======================================================
# PART 1 – Classification + Optimisation (Streamlit-ready)
# ======================================================

import numpy as np
import pandas as pd
import re
from scipy.stats import nbinom
import streamlit as st
# ----------------------------------------
# Debug helper (Streamlit + fallback)
# ----------------------------------------
def _disp(obj, n=None, title=None):
    """Safe display wrapper for Streamlit and console."""
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
        print(f"[DEBUG] {_disp.__name__} error:", e)
        if title:
            print(title)
        print(obj.head(n) if isinstance(obj, pd.DataFrame) and n else obj)

# ----------------------------------------
# Load and classify articles
# ----------------------------------------
def classify_articles(file_articles):
    """
    Load articles.xlsx (single-sheet) and classify articles.
    """
    try:
        df = pd.read_excel(file_articles)  # no sheet_name anymore
    except Exception as e:
        st.error(f"❌ Could not read articles.xlsx: {e}")
        return pd.DataFrame()

    # Basic cleaning
    df.columns = [str(c).strip() for c in df.columns]

    # Ensure key columns exist (fallback for French column variants)
    col_map = {}
    for col in df.columns:
        c = col.lower()
        if "code" in c:
            col_map["code"] = col
        elif "quantite" in c or "qty" in c:
            col_map["quantite"] = col
        elif "famille" in c or "class" in c:
            col_map["famille"] = col

    # Standardise names
    df = df.rename(columns=col_map)

    # Drop rows missing product code
    if "code" not in df.columns:
        st.error("❌ 'Code' column not found in articles.xlsx")
        return pd.DataFrame()
    df = df.dropna(subset=["code"])

    # Example classification (ABC by quantite)
    if "quantite" in df.columns:
        df["categorie"] = pd.qcut(df["quantite"], 3, labels=["C", "B", "A"])
    else:
        df["categorie"] = "C"

    _disp(df.head(20), title="✅ Articles Classifiés")
    return df

# ----------------------------------------
# Optimisation placeholder (integrates later with Part 2)
# ----------------------------------------
def optimise_articles(df_classified):
    """
    Example optimisation step on classified articles.
    Placeholder: integrate with forecasting (Part 2).
    """
    if df_classified.empty:
        st.warning("⚠️ No classified articles available for optimisation.")
        return pd.DataFrame()

    # Fake optimisation: assign reorder point based on quantite
    if "quantite" in df_classified.columns:
        df_classified["ROP"] = df_classified["quantite"] * 0.2
    else:
        df_classified["ROP"] = 10

    _disp(df_classified.head(20), title="📊 Articles Optimisés")
    return df_classified

# ----------------------------------------
# Streamlit workflow (Part 1)
# ----------------------------------------
def run_classification_optimisation(file_articles):
    st.header("Step 1: Classification & Optimisation")

    df_class = classify_articles(file_articles)
    if df_class.empty:
        return pd.DataFrame()

    df_opt = optimise_articles(df_class)
    return df_opt
# ======================================================
# PART 2 – Forecasting + Sensitivity (Streamlit-ready)
# ======================================================



# --------- PARAMÈTRES SUPPLY / ROP ---------
DELAI_USINE = 10            # jours
DELAI_FOURNISSEUR = 3       # jours
NIVEAU_SERVICE_DEF = 0.95
NB_SIM = 1000
GRAINE_ALEA = 42

# --------- COLONNES D'AFFICHAGE ---------
COLONNES_AFFICHAGE = [
    "date", "code", "methode", "intervalle",
    "demande_reelle", "stock_disponible_intervalle", "stock_apres_intervalle",
    "politique_commande", "Qr_etoile", "Qw_etoile", "n_etoile",
    "ROP_usine", "SS_usine",
    "ROP_fournisseur", "SS_fournisseur",
    "statut_stock", "service_level"
]

# --------- OUTILS AFFICHAGE ---------
def _disp(obj, n=None, title=None):
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
        print("[DEBUG]", e)
        if title:
            print(title)
        print(obj.head(n) if isinstance(obj, pd.DataFrame) and n else obj)

# ======================================================
# PARTIE A : Q* (Qr*, Qw*, n*) depuis PFE HANIN
# ======================================================
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
    raise ValueError(f"[Feuille] Onglet pour '{code}' introuvable dans {chemin_excel}.")

def compute_qstars(chemin_excel: str, codes: list):
    df_conso = pd.read_excel(chemin_excel, sheet_name="consommation depots externe")
    df_conso = df_conso.groupby('Code Produit')['Quantite STIAL'].sum()

    qr_map, qw_map, n_map = {}, {}, {}
    for code in codes:
        feuille = _trouver_feuille_produit(chemin_excel, code)
        df = pd.read_excel(chemin_excel, sheet_name=feuille)

        C_r = df.filter(like="Cr").iloc[0, 0]
        C_w = df.filter(like="Cw").iloc[0, 0]
        A_w = df.filter(like="Aw").iloc[0, 0]
        A_r = df.filter(like="Ar").iloc[0, 0]

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

# ======================================================
# PARTIE B : Séries conso/stock journalières
# ======================================================
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

# ======================================================
# PARTIE C : Méthodes de prévision
# ======================================================
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

# ======================================================
# PARTIE F : Final (SL unique) + Sensibilité
# ======================================================
def run_final_once(best_per_code: pd.DataFrame, excel_path, service_level=NIVEAU_SERVICE_DEF):
    qr_map, qw_map, n_map = compute_qstars(excel_path, best_per_code["code"].tolist())
    results = []
    for _, row in best_per_code.iterrows():
        code = row["code"]
        method = str(row.get("method", "ses")).lower()
        alpha = float(row.get("alpha", 0.2))
        window_ratio = float(row.get("window_ratio", 0.8))
        intervalle = int(row.get("recalc_interval", 7))
        df_run = pd.DataFrame()  # placeholder rolling
        results.append(df_run)
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

def run_sensitivity(best_per_code: pd.DataFrame, excel_path, service_levels=[0.90, 0.92, 0.95, 0.98]):
    qr_map, qw_map, n_map = compute_qstars(excel_path, best_per_code["code"].tolist())
    all_results = []
    for sl in service_levels:
        runs = []
        for _, row in best_per_code.iterrows():
            df_run = pd.DataFrame()
            runs.append(df_run)
        df_concat = pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()
        all_results.append(df_concat)
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# ======================================================
# Streamlit workflow (Part 2)
# ======================================================
def run_forecasting_sensitivity(file_forecast):
    st.header("Step 2: Forecasting & Sensitivity")

    try:
        xl = pd.ExcelFile(file_forecast)
        st.write("📑 Sheets disponibles:", xl.sheet_names[:10])
    except Exception as e:
        st.error(f"❌ Could not open {file_forecast}: {e}")
        return pd.DataFrame()

    # Fake best params (for demo)
    best_per_code = pd.DataFrame({
        "code": ["EM0400", "EM1499"],
        "method": ["ses", "croston"],
        "alpha": [0.2, 0.3],
        "window_ratio": [0.8, 0.7],
        "recalc_interval": [7, 7]
    })

    _disp(best_per_code, title="✅ Best Method per Product")

    final_95 = run_final_once(best_per_code, file_forecast, service_level=NIVEAU_SERVICE_DEF)
    sensi = run_sensitivity(best_per_code, file_forecast)

    if not sensi.empty:
        summary = sensi.groupby(["code", "service_level"]).size().reset_index(name="n_obs")
        _disp(summary, title="📊 Résumé Global (Sensibilité)")
    else:
        st.warning("⚠️ Aucun résultat de sensibilité (vérifier les entrées).")

    return sensi
