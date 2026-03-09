"""
Page d'accueil — Médiateur de l'Éducation Nationale
Affiche les indicateurs clés et la distribution des saisines par catégorie.
"""

import sys
from pathlib import Path

# Ajouter app/ au path pour que "from utils.db import ..." fonctionne
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotly.express as px
import streamlit as st

from utils.db import PARQUET, requete

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EDN1 v2 — Médiateur Éducation Nationale",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Données globales
# ---------------------------------------------------------------------------

stats = requete(
    f"""
    SELECT
        COUNT(*)              AS total,
        COUNT(DISTINCT pole)  AS n_poles,
        COUNT(DISTINCT label) AS n_labels,
        MIN(annee)            AS annee_min,
        MAX(annee)            AS annee_max
    FROM read_parquet('{PARQUET}')
    """
).iloc[0]

df_labels = requete(
    f"""
    SELECT label, COUNT(*) AS total
    FROM read_parquet('{PARQUET}')
    GROUP BY label
    ORDER BY total DESC
    """
)

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------

st.title("Médiateur de l'Éducation Nationale")
st.subheader("Analyse automatique des saisines — EDN1 v2")
st.divider()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Total saisines",
    value=f"{int(stats['total']):,}".replace(",", "\u202f"),
)
col2.metric(
    label="Années couvertes",
    value=f"{int(stats['annee_min'])} – {int(stats['annee_max'])}",
)
col3.metric(
    label="Pôles académiques",
    value=int(stats["n_poles"]),
)
col4.metric(
    label="Catégories NLP",
    value=int(stats["n_labels"]),
)

st.divider()

# ---------------------------------------------------------------------------
# Graphique + tableau
# ---------------------------------------------------------------------------

col_gauche, col_droite = st.columns([3, 1])

with col_gauche:
    st.subheader("Répartition par catégorie")

    total_global = int(df_labels["total"].sum())
    df_labels["pct"] = (df_labels["total"] / total_global * 100).round(1)

    fig = px.bar(
        df_labels,
        x="total",
        y="label",
        orientation="h",
        color="total",
        color_continuous_scale="Blues",
        labels={"total": "Nombre de saisines", "label": "Catégorie"},
        text=df_labels["pct"].astype(str) + "%",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
        margin={"l": 0, "r": 80, "t": 10, "b": 0},
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_droite:
    st.subheader("Détail")
    st.dataframe(
        df_labels[["label", "total", "pct"]].rename(
            columns={"label": "Catégorie", "total": "Nb", "pct": "%"}
        ),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

st.divider()
st.markdown("### Pages disponibles")

col1, col2, col3 = st.columns(3)
with col1:
    st.info(
        "**Exploration** \n\n"
        "Parcourez et filtrez les saisines par pôle, catégorie, année "
        "ou mot-clé. Export CSV inclus."
    )
with col2:
    st.info(
        "**Statistiques** \n\n"
        "Distribution des catégories, évolution mensuelle/annuelle, "
        "et répartition par pôle académique."
    )
with col3:
    st.info(
        "**Analyse croisée** \n\n"
        "Croisez deux dimensions au choix : label × pôle, "
        "année × catégorie, etc. Tableau + heatmap."
    )
