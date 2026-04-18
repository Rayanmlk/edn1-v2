"""
Page 2 — Statistiques
Trois onglets : distribution par catégorie, évolution temporelle, répartition par pôle.
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
    page_title="Statistiques — EDN1 v2",
    page_icon="📊",
    layout="wide",
)

st.title("Statistiques")
st.caption("Visualisez la répartition et l'évolution des saisines.")

onglet1, onglet2, onglet3 = st.tabs(
    ["Distribution par catégorie", "Evolution temporelle", "Par pôle académique"]
)

# ---------------------------------------------------------------------------
# Onglet 1 — Distribution par catégorie
# ---------------------------------------------------------------------------

with onglet1:
    df_labels = requete(
        f"""
        SELECT label, COUNT(*) AS total
        FROM read_parquet('{PARQUET}')
        GROUP BY label
        ORDER BY total DESC
        """
    )

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
        text=df_labels["pct"].astype(str) + " %",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
        margin={"l": 0, "r": 80, "t": 20, "b": 0},
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sous-catégories pour un label sélectionné
    st.markdown("---")
    st.markdown("**Zoom sur les sous-catégories**")
    tous_labels = df_labels["label"].tolist()
    label_zoom = st.selectbox("Choisir une catégorie", tous_labels)

    df_sous = requete(
        f"""
        SELECT sous_label, COUNT(*) AS total
        FROM read_parquet('{PARQUET}')
        WHERE label = '{label_zoom}'
        GROUP BY sous_label
        ORDER BY total DESC
        """
    )
    fig2 = px.bar(
        df_sous,
        x="total",
        y="sous_label",
        orientation="h",
        color="total",
        color_continuous_scale="Purples",
        labels={"total": "Saisines", "sous_label": "Sous-catégorie"},
        text="total",
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
        margin={"l": 0, "r": 60, "t": 10, "b": 0},
        height=max(300, len(df_sous) * 28),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Onglet 2 — Evolution temporelle
# ---------------------------------------------------------------------------

with onglet2:
    col1, col2 = st.columns([1, 2])
    with col1:
        granularite = st.radio(
            "Granularité", ["Par année", "Par mois"], horizontal=False
        )
    with col2:
        tous_labels_evo = requete(
            f"SELECT DISTINCT label FROM read_parquet('{PARQUET}') ORDER BY label"
        )["label"].tolist()
        labels_sel = st.multiselect(
            "Catégories à afficher",
            tous_labels_evo,
            default=tous_labels_evo[:5],
        )

    if not labels_sel:
        st.info("Sélectionnez au moins une catégorie pour afficher le graphique.")
        st.stop()

    labels_sql = ", ".join(f"'{l}'" for l in labels_sel)

    if granularite == "Par année":
        df_temps = requete(
            f"""
            SELECT
                CAST(annee AS VARCHAR)  AS periode,
                annee                   AS tri,
                label,
                COUNT(*)                AS total
            FROM read_parquet('{PARQUET}')
            WHERE annee IS NOT NULL
              AND label IN ({labels_sql})
            GROUP BY annee, label
            ORDER BY annee
            """
        )
    else:
        df_temps = requete(
            f"""
            SELECT
                CAST(annee AS VARCHAR) || '-' ||
                LPAD(CAST(mois AS VARCHAR), 2, '0') AS periode,
                annee * 100 + mois                  AS tri,
                label,
                COUNT(*)                            AS total
            FROM read_parquet('{PARQUET}')
            WHERE annee IS NOT NULL AND mois IS NOT NULL
              AND label IN ({labels_sql})
            GROUP BY annee, mois, label
            ORDER BY annee, mois
            """
        )

    # Trier les périodes chronologiquement
    ordre_periodes = (
        df_temps.drop_duplicates("periode")
        .sort_values("tri")["periode"]
        .tolist()
    )

    fig = px.line(
        df_temps,
        x="periode",
        y="total",
        color="label",
        markers=True,
        labels={
            "periode": "Période",
            "total": "Nombre de saisines",
            "label": "Catégorie",
        },
        category_orders={"periode": ordre_periodes},
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        height=420,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Onglet 3 — Par pôle
# ---------------------------------------------------------------------------

with onglet3:
    n_poles = st.slider("Nombre de pôles à afficher", min_value=5, max_value=32, value=15)

    df_poles = requete(
        f"""
        SELECT pole, COUNT(*) AS total
        FROM read_parquet('{PARQUET}')
        WHERE pole IS NOT NULL
        GROUP BY pole
        ORDER BY total DESC
        LIMIT {n_poles}
        """
    )

    fig = px.bar(
        df_poles,
        x="total",
        y="pole",
        orientation="h",
        color="total",
        color_continuous_scale="Greens",
        labels={"total": "Nombre de saisines", "pole": "Pôle académique"},
        text="total",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
        margin={"l": 0, "r": 60, "t": 20, "b": 0},
        height=max(350, n_poles * 28),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribution des catégories dans un pôle sélectionné
    st.markdown("---")
    st.markdown("**Zoom sur un pôle**")
    tous_poles = requete(
        f"""
        SELECT pole FROM read_parquet('{PARQUET}')
        WHERE pole IS NOT NULL
        GROUP BY pole ORDER BY COUNT(*) DESC
        """
    )["pole"].tolist()

    pole_zoom = st.selectbox("Choisir un pôle", tous_poles)

    df_pole_labels = requete(
        f"""
        SELECT label, COUNT(*) AS total
        FROM read_parquet('{PARQUET}')
        WHERE pole = '{pole_zoom.replace("'", "''")}'
        GROUP BY label
        ORDER BY total DESC
        """
    )
    fig3 = px.pie(
        df_pole_labels,
        names="label",
        values="total",
        title=f"Répartition des catégories — {pole_zoom}",
        hole=0.4,
    )
    fig3.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0}, height=380)
    st.plotly_chart(fig3, use_container_width=True)
