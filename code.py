# ============================================================
# UNIFIED STREAMLIT APP - PART 1/2
# ============================================================
# This first entity contains:
# - Imports
# - Helpers
# - Classification + Optimisation logic
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.stats import nbinom
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
from IPython.display import display

# ============================================================
# GLOBAL HELPERS
# ============================================================

def _disp(obj, n=None, title=None):
    """Helper display that works in Streamlit or console"""
    try:
        if title:
            st.subheader(title)
        if n is not None and hasattr(obj, "head"):
            st.dataframe(obj.head(n))
        else:
            st.dataframe(obj)
    except:
        if isinstance(obj, pd.DataFrame):
            print(obj.head(n).to_string(index=False) if n else obj.to_string(index=False))
        else:
            print(obj)

def _find_first_col(df, patt):
    """Find first column in df matching a regex pattern"""
    regex = re.compile(patt, re.IGNORECASE)
    for c in df.columns:
        if regex.search(str(c)):
            return c
    return None

def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[\s\n\t]+", " ", s).strip().lower()

def _somme_intervalle(serie: pd.Series, start_idx: int, intervalle: int) -> float:
    s, e = start_idx + 1, start_idx + 1 + int(intervalle)
    return float(pd.Series(serie).iloc[s:e].sum())

# ============================================================
# PARAMETERS
# ============================================================

DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_INTERVAL = 10
DEFAULT_ALPHA = 0.2
DEFAULT_WINDOW_RATIO = 0.7
DEFAULT_SEED = 42

# ============================================================
# CLASSIFICATION + OPTIMISATION LOGIC
# ============================================================

def classify_articles(file_excel):
    """
    Classify articles from Excel data
    """
    st.info("Classifying articles...")

    df = pd.read_excel(file_excel, sheet_name="articles")
    col_code = _find_first_col(df, "code")
    col_nom = _find_first_col(df, "nom|designation")
    col_cat = _find_first_col(df, "categorie|classification")
    col_fam = _find_first_col(df, "famille|family")
    col_val = _find_first_col(df, "valeur|value")

    if not col_code or not col_nom:
        st.error("Impossible de trouver les colonnes code/nom dans le fichier articles.")
        return pd.DataFrame()

    df = df.rename(columns={
        col_code: "code",
        col_nom: "nom",
        col_cat: "categorie",
        col_fam: "famille",
        col_val: "valeur"
    })

    # Classification ABC sur la base de la valeur
    df = df.sort_values("valeur", ascending=False)
    df["cum_val"] = df["valeur"].cumsum() / df["valeur"].sum()
    df["classification"] = pd.cut(df["cum_val"],
                                  bins=[0, 0.7, 0.9, 1.0],
                                  labels=["A", "B", "C"],
                                  include_lowest=True)
    _disp(df, title="📊 Classification ABC des articles")
    return df

def optimise_policy(file_excel, service_level=DEFAULT_SERVICE_LEVEL):
    """
    Optimisation politique d’approvisionnement
    """
    st.info("Optimisation des politiques...")

    df = pd.read_excel(file_excel, sheet_name="stock")
    col_code = _find_first_col(df, "code")
    col_dem = _find_first_col(df, "demande")
    col_delai = _find_first_col(df, "delai")
    col_stock = _find_first_col(df, "stock")

    df = df.rename(columns={
        col_code: "code",
        col_dem: "demande",
        col_delai: "delai",
        col_stock: "stock"
    })

    # Calcul du ROP et SS simplifiés
    df["ROP"] = df["demande"] * df["delai"]
    df["SS"] = df["ROP"] * service_level
    df["Politique"] = np.where(df["stock"] < df["ROP"], "Commander", "Ne rien faire")

    _disp(df, title="⚙️ Optimisation politique d’approvisionnement")
    return df

# ============================================================
# STREAMLIT APP - PART 1
# ============================================================

def run_classification_optimisation():
    st.title("📊 Module 1 : Classification + Optimisation")

    file_excel = st.file_uploader("Charger le fichier Excel (articles + stock)", type=["xlsx"])
    if not file_excel:
        st.warning("Veuillez charger un fichier Excel pour continuer.")
        return

    service_level = st.slider("Niveau de service cible", 0.80, 0.99, DEFAULT_SERVICE_LEVEL, 0.01)

    # Classification
    if st.button("Lancer la Classification"):
        df_class = classify_articles(file_excel)
        if not df_class.empty:
            st.success("Classification terminée.")

    # Optimisation
    if st.button("Lancer l’Optimisation"):
        df_opt = optimise_policy(file_excel, service_level=service_level)
        if not df_opt.empty:
            st.success("Optimisation terminée.")
# ============================================================
# UNIFIED STREAMLIT APP - PART 2/2
# ============================================================
# This second entity contains:
# - All functions for Final + Sensibilité
# - Streamlit UI for simulation
# - Sidebar to navigate between modules
# ============================================================

# --------- CONSTANTS ---------
EXCEL_PATH_DATA = "PFE  HANIN (1).xlsx"
PATH_SES = "best_params_SES.xlsx"
PATH_CROSTON = "best_params_CROSTON.xlsx"
PATH_SBA = "best_params_SBA.xlsx"

CODES_PRODUITS = ["EM0400", "EM1499", "EM1091", "EM1523", "EM0392", "EM1526"]

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
# PARTIE A : Q* (Qr*, Qw*, n*)
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
    raise ValueError(f"[Feuille] Onglet pour '{code}' introuvable dans {chemin_excel}.")

def compute_qstars(chemin_excel: str, codes: list):
    df_conso = pd.read_excel(chemin_excel, sheet_name="consommation depots externe")
    df_conso = df_conso.groupby('Code Produit')['Quantite STIAL'].sum()

    qr_map, qw_map, n_map = {}, {}, {}
    for code in codes:
        feuille = _trouver_feuille_produit(chemin_excel, code)
        df = pd.read_excel(chemin_excel, sheet_name=feuille)

        C_r = df['Cr : cout stockage/article '].iloc[0]
        C_w = df['Cw : cout stockage\nchez F'].iloc[0]
        A_w = df['Aw : cout de\nlancement chez U'].iloc[0]
        A_r = df['Ar : cout de \nlancement chez F'].iloc[0]

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

# ============================================================
# PARTIE B : Séries conso/stock
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

# ============================================================
# PARTIE C : Méthodes de prévision
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
# PARTIE D : Rolling simulation
# ============================================================

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
# PARTIE E : Choix des meilleures méthodes
# ============================================================

def _normalize_df_best(df_best: pd.DataFrame, method_name: str, pick_metric: str = "RMSE") -> pd.DataFrame:
    metric_key = pick_metric.upper()
    if metric_key == "ABSME":
        a, w, itv, s = "best_ME_alpha", "best_ME_window", "best_ME_interval", "best_absME"
    else:
        a, w, itv, s = f"best_{metric_key}_alpha", f"best_{metric_key}_window", f"best_{metric_key}_interval", f"best_{metric_key}"

    for cand in [
        (a, w, itv, s),
        ("best_RMSE_alpha", "best_RMSE_window", "best_RMSE_interval", "best_RMSE"),
        ("best_MSE_alpha",  "best_MSE_window",  "best_MSE_interval",  "best_MSE"),
        ("best_ME_alpha",   "best_ME_window",   "best_ME_interval",   "best_ME"),
    ]:
        if all(c in df_best.columns for c in cand):
            a, w, itv, s = cand
            break

    out = df_best.rename(columns={
        a: "alpha", w: "window_ratio", itv: "recalc_interval", s: "score"
    })
    keep = ["code", "alpha", "window_ratio", "recalc_interval", "score"]
    if "n_points_used" in df_best.columns:
        out["n_points_used"] = df_best["n_points_used"]
        keep.append("n_points_used")
    out = out[keep].copy()
    for c in ["alpha", "window_ratio", "recalc_interval", "score"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["method"] = method_name
    return out

def select_best_method_from_files(
    path_ses=PATH_SES, path_cro=PATH_CROSTON, path_sba=PATH_SBA,
    product_filter=None, pick_metric="RMSE"
):
    df_best_SES = pd.read_excel(path_ses)
    df_best_CRO = pd.read_excel(path_cro)
    df_best_SBA = pd.read_excel(path_sba)

    cand_ses = _normalize_df_best(df_best_SES, "ses", pick_metric)
    cand_cro = _normalize_df_best(df_best_CRO, "croston", pick_metric)
    cand_sba = _normalize_df_best(df_best_SBA, "sba", pick_metric)

    candidates = pd.concat([cand_ses, cand_cro, cand_sba], ignore_index=True)
    if product_filter:
        candidates = candidates[candidates["code"].isin(product_filter)].copy()

    idx = candidates.groupby("code")["score"].idxmin()
    best_per_code = candidates.loc[idx].sort_values(["code"]).reset_index(drop=True)
    return best_per_code

# ============================================================
# PARTIE F : Final + Sensibilité
# ============================================================

def run_final_once(best_per_code: pd.DataFrame, service_level=NIVEAU_SERVICE_DEF):
    qr_map, qw_map, n_map = compute_qstars(EXCEL_PATH_DATA, best_per_code["code"].tolist())
    results = []
    for _, row in best_per_code.iterrows():
        code = row["code"]
        method = str(row["method"]).lower()
        alpha = float(row["alpha"])
        window_ratio = float(row["window_ratio"])
        intervalle = int(row["recalc_interval"])
        df_run = rolling_with_new_logic(
            excel_path=EXCEL_PATH_DATA,
            product_code=code,
            alpha=alpha, window_ratio=window_ratio, intervalle=intervalle,
            delai_usine=DELAI_USINE, delai_fournisseur=DELAI_FOURNISSEUR,
            service_level=service_level, nb_sim=NB_SIM, rng_seed=GRAINE_ALEA,
            variant=method, qr_map=qr_map, qw_map=qw_map, n_map=n_map
        )
        results.append(df_run)
        _disp(df_run[COLONNES_AFFICHAGE], n=10, title=f"\n=== {code} — {method.upper()} (SL={service_level:.2f}) ===")
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

def run_sensitivity(best_per_code: pd.DataFrame, service_levels=[0.90, 0.92, 0.95, 0.98]):
    qr_map, qw_map, n_map = compute_qstars(EXCEL_PATH_DATA, best_per_code["code"].tolist())
    all_results = []
    for sl in service_levels:
        st.write(f"\n\n🔎 Simulation avec Service Level = {sl*100:.0f}%")
        runs = []
        for _, row in best_per_code.iterrows():
            code = row["code"]
            method = str(row["method"]).lower()
            alpha = float(row["alpha"])
            window_ratio = float(row["window_ratio"])
            intervalle = int(row["recalc_interval"])

            df_run = rolling_with_new_logic(
                excel_path=EXCEL_PATH_DATA,
                product_code=code,
                alpha=alpha, window_ratio=window_ratio, intervalle=intervalle,
                delai_usine=DELAI_USINE, delai_fournisseur=DELAI_FOURNISSEUR,
                service_level=sl, nb_sim=NB_SIM, rng_seed=GRAINE_ALEA,
                variant=method, qr_map=qr_map, qw_map=qw_map, n_map=n_map
            )
            df_run["service_level"] = sl
            runs.append(df_run)

        df_concat = pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()
        all_results.append(df_concat)

        if not df_concat.empty:
            grp = df_concat.groupby("code").agg(
                ROP_usine_moy=("ROP_usine", "mean"),
                SS_usine_moy=("SS_usine", "mean"),
                ROP_fournisseur_moy=("ROP_fournisseur", "mean"),
                SS_fournisseur_moy=("SS_fournisseur", "mean"),
                holding_pct=("statut_stock", lambda s: (s == "holding").mean()*100),
                rupture_pct=("statut_stock", lambda s: (s == "rupture").mean()*100),
                Qr_star=("Qr_etoile", "first"),
                Qw_star=("Qw_etoile", "first"),
                n_star=("n_etoile", "first"),
            ).reset_index()
            _disp(grp, title=f"=== Résultats pour SL {sl*100:.0f}% (moyennes par article) ===")

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# ============================================================
# STREAMLIT APP - PART 2
# ============================================================

def run_final_sensitivity():
    st.title("📈 Module 2 : Simulation Finale + Sensibilité")

    st.info("Chargement des meilleures méthodes et paramètres...")
    best_per_code = select_best_method_from_files(
        path_ses=PATH_SES, path_cro=PATH_CROSTON, path_sba=PATH_SBA,
        product_filter=CODES_PRODUITS, pick_metric="RMSE"
    )
    _disp(best_per_code, title="✅ Meilleure méthode et meilleurs paramètres par article")

    st.write("▶️ Recalcul final au niveau de service 95%")
    final_95 = run_final_once(best_per_code, service_level=NIVEAU_SERVICE_DEF)

    SERVICE_LEVELS = [0.90, 0.92, 0.95, 0.98]
    sensi = run_sensitivity(best_per_code, service_levels=SERVICE_LEVELS)

    if not sensi.empty:
        st.subheader("📊 Résumé global – moyennes par code et niveau de service")
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
        _disp(summary, n=50)
    else:
        st.warning("⚠️ Aucun résultat de sensibilité (vérifier les entrées).")

# ============================================================
# MAIN APP NAVIGATION
# ============================================================

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choisir un module", ["Classification + Optimisation", "Final + Sensibilité"])

    if choice == "Classification + Optimisation":
        run_classification_optimisation()
    elif choice == "Final + Sensibilité":
        run_final_sensitivity()

if __name__ == "__main__":
    main()
