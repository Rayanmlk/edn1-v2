"""
Connexion DuckDB — module partagé entre toutes les pages du dashboard.

Fournit :
- PARQUET : chemin absolu du fichier Parquet (en slashs pour DuckDB)
- requete(sql) : exécute une requête SQL, résultat mis en cache 5 min
- valeurs(colonne) : valeurs distinctes d'une colonne (pour peupler les filtres)
- in_sql(liste) : convertit une liste Python en clause SQL  ['a','b'] -> "('a', 'b')"
"""

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Chemin du fichier Parquet
# ---------------------------------------------------------------------------

_PARQUET_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "processed"
    / "saisines.parquet"
)

# DuckDB sur Windows : les chemins doivent utiliser des slashs avant
PARQUET = str(_PARQUET_PATH).replace("\\", "/")


# ---------------------------------------------------------------------------
# Fonctions publiques
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def requete(sql: str) -> pd.DataFrame:
    """
    Exécute une requête SQL sur le fichier Parquet et retourne un DataFrame.

    Le résultat est mis en cache 5 minutes : si la même requête SQL est
    appelée plusieurs fois dans la même session (ex: l'utilisateur change
    d'onglet et revient), le résultat est retourné instantanément.

    Thread-safe : chaque appel sans cache crée une connexion DuckDB neuve.
    """
    if not _PARQUET_PATH.exists():
        st.error(
            "Le fichier saisines.parquet est introuvable. "
            "Lancez d'abord : `python pipeline/03_to_parquet.py`"
        )
        st.stop()

    with duckdb.connect() as con:
        return con.execute(sql).df()


def valeurs(colonne: str) -> list:
    """
    Retourne la liste triée des valeurs distinctes d'une colonne.
    Utile pour peupler les filtres multiselect de la sidebar.
    """
    df = requete(
        f"SELECT DISTINCT {colonne} "
        f"FROM read_parquet('{PARQUET}') "
        f"WHERE {colonne} IS NOT NULL "
        f"ORDER BY {colonne}"
    )
    return df[colonne].tolist()


def in_sql(valeurs_liste: list) -> str:
    """
    Convertit une liste Python en clause SQL IN.
    Exemple : ['Paris', 'Lyon'] -> "('Paris', 'Lyon')"
    Les apostrophes dans les valeurs sont automatiquement échappées.
    """
    items = ", ".join(f"'{str(v).replace(chr(39), chr(39)*2)}'" for v in valeurs_liste)
    return f"({items})"
