"""
Page 3 — Analyse croisée (tableau pivot)
Croisez deux dimensions au choix et visualisez le résultat en tableau ou heatmap.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

from utils.db import PARQUET, requete

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Analyse croisée — EDN1 v2",
    page_icon="📐",
    layout="wide",
)

st.title("Analyse croisée")
st.caption(
    "Croisez deux dimensions pour compter les saisines à chaque intersection. "
    "Exemple : combien de saisines 'examens' viennent de Paris ?"
)

# ---------------------------------------------------------------------------
# Sélection des axes
# ---------------------------------------------------------------------------

DIMENSIONS = {
    "Catégorie (label)": "label",
    "Sous-catégorie (sous_label)": "sous_label",
    "Pôle académique": "pole",
    "Année": "annee",
    "Mois": "mois",
}

col1, col2 = st.columns(2)
with col1:
    axe_lignes = st.selectbox("Dimension — Lignes", list(DIMENSIONS.keys()), index=2)
with col2:
    axe_colonnes = st.selectbox(
        "Dimension — Colonnes", list(DIMENSIONS.keys()), index=0
    )

col_lignes = DIMENSIONS[axe_lignes]
col_colonnes = DIMENSIONS[axe_colonnes]

if col_lignes == col_colonnes:
    st.warning("Veuillez sélectionner deux dimensions différentes.")
    st.stop()

# ---------------------------------------------------------------------------
# Requête
# ---------------------------------------------------------------------------

df_raw = requete(
    f"""
    SELECT
        CAST({col_lignes}  AS VARCHAR) AS {col_lignes},
        CAST({col_colonnes} AS VARCHAR) AS {col_colonnes},
        COUNT(*) AS total
    FROM read_parquet('{PARQUET}')
    WHERE {col_lignes} IS NOT NULL
      AND {col_colonnes} IS NOT NULL
    GROUP BY {col_lignes}, {col_colonnes}
    """
)

if df_raw.empty:
    st.warning("Aucune donnée pour cette combinaison de dimensions.")
    st.stop()

pivot = (
    df_raw.pivot(index=col_lignes, columns=col_colonnes, values="total")
    .fillna(0)
    .astype(int)
)

# Ajouter une colonne Total
pivot["TOTAL"] = pivot.sum(axis=1)
pivot = pivot.sort_values("TOTAL", ascending=False)

st.markdown(
    f"**{len(pivot)} lignes × {len(pivot.columns) - 1} colonnes** "
    f"({axe_lignes} × {axe_colonnes})"
)

# ---------------------------------------------------------------------------
# Affichage : tableau et heatmap
# ---------------------------------------------------------------------------

onglet1, onglet2 = st.tabs(["Tableau", "Heatmap"])

with onglet1:
    st.dataframe(
        pivot.style.background_gradient(cmap="Blues", axis=None, subset=pivot.columns[:-1]),
        use_container_width=True,
        height=520,
    )

with onglet2:
    # Pour la heatmap, on exclut la colonne TOTAL
    pivot_heatmap = pivot.drop(columns=["TOTAL"])

    fig = px.imshow(
        pivot_heatmap,
        color_continuous_scale="Blues",
        labels={"color": "Saisines", "x": axe_colonnes, "y": axe_lignes},
        aspect="auto",
        text_auto=True,
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        height=max(400, len(pivot_heatmap) * 30),
        coloraxis_showscale=True,
    )
    st.plotly_chart(fig, use_container_width=True)
