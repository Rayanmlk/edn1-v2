"""
Page 1 — Exploration des saisines
Table filtrable avec sidebar. Export CSV.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from utils.db import PARQUET, in_sql, requete, valeurs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Exploration — EDN1 v2",
    page_icon="🔍",
    layout="wide",
)

st.title("Exploration des saisines")
st.caption("Filtrez, parcourez et exportez les saisines.")

# ---------------------------------------------------------------------------
# Sidebar — Filtres
# ---------------------------------------------------------------------------

st.sidebar.header("Filtres")

tous_poles = valeurs("pole")
tous_labels = valeurs("label")
toutes_annees = [int(a) for a in valeurs("annee")]

poles_sel = st.sidebar.multiselect("Pôle académique", tous_poles)
labels_sel = st.sidebar.multiselect("Catégorie", tous_labels)
annees_sel = st.sidebar.multiselect("Année", toutes_annees)
recherche = st.sidebar.text_input(
    "Recherche dans l'analyse",
    placeholder="Ex: baccalauréat",
)

# ---------------------------------------------------------------------------
# Construction de la requête SQL dynamique
# ---------------------------------------------------------------------------

conditions = []

if poles_sel:
    conditions.append(f"pole IN {in_sql(poles_sel)}")

if labels_sel:
    conditions.append(f"label IN {in_sql(labels_sel)}")

if annees_sel:
    annees_int = ", ".join(str(a) for a in annees_sel)
    conditions.append(f"annee IN ({annees_int})")

if recherche:
    # Échappe les apostrophes pour éviter les erreurs SQL
    terme = recherche.replace("'", "''").lower()
    conditions.append(f'LOWER("analyse") LIKE \'%{terme}%\'')

where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

sql = f"""
    SELECT
        id,
        date_arrivee,
        annee,
        pole,
        label,
        sous_label,
        lieu,
        key_word,
        "analyse"
    FROM read_parquet('{PARQUET}')
    {where}
    ORDER BY date_arrivee DESC
"""

df = requete(sql)

# ---------------------------------------------------------------------------
# Affichage du tableau
# ---------------------------------------------------------------------------

total_filtre = len(df)
total_global = int(requete(f"SELECT COUNT(*) AS n FROM read_parquet('{PARQUET}')").iloc[0]["n"])

if conditions:
    st.markdown(
        f"**{total_filtre:,}** saisines sur {total_global:,} correspondent aux filtres sélectionnés."
    )
else:
    st.markdown(f"**{total_global:,}** saisines au total (aucun filtre actif).")

st.dataframe(
    df,
    use_container_width=True,
    height=520,
    column_config={
        "id": st.column_config.TextColumn("ID", width="small"),
        "date_arrivee": st.column_config.DateColumn(
            "Date", format="DD/MM/YYYY", width="small"
        ),
        "annee": st.column_config.NumberColumn("Année", width="small", format="%d"),
        "pole": st.column_config.TextColumn("Pôle", width="medium"),
        "label": st.column_config.TextColumn("Catégorie", width="medium"),
        "sous_label": st.column_config.TextColumn("Sous-catégorie", width="medium"),
        "lieu": st.column_config.TextColumn("Lieu", width="small"),
        "key_word": st.column_config.TextColumn("Mots-clés", width="medium"),
        "analyse": st.column_config.TextColumn(
            "Analyse", width="large", max_chars=200
        ),
    },
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------

csv = df.to_csv(index=False, encoding="utf-8-sig")  # utf-8-sig : lisible par Excel

st.download_button(
    label="Exporter en CSV",
    data=csv,
    file_name="saisines_export.csv",
    mime="text/csv",
    help="Télécharge les saisines affichées (avec filtres appliqués) au format CSV.",
)
