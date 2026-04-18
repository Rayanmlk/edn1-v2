"""
Etape 3 — Conversion JSON -> Parquet

Ce script prend le fichier saisines_classifiees.json (produit par l'étape 2)
et le convertit en format Parquet, un format de stockage colonnaıre ultra-rapide
utilisé pour les analyses.

Pourquoi Parquet ?
- Un fichier JSON de 6 000 saisines fait ~15 Mo et se lit lentement
- Le même fichier en Parquet fait ~1 Mo et se lit 10 à 50x plus vite
- DuckDB peut faire des requêtes SQL directement sur ce fichier, sans base de données
- Le dashboard Streamlit (étape 4) l'utilisera pour toutes ses statistiques

Usage :
    python pipeline/03_to_parquet.py
"""

import json
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent.parent
FICHIER_ENTREE = PROJECT_DIR / "data" / "processed" / "saisines_classifiees.json"
FICHIER_SORTIE = PROJECT_DIR / "data" / "processed" / "saisines.parquet"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colonnes attendues dans le fichier d'entrée
# ---------------------------------------------------------------------------

COLONNES_ATTENDUES = [
    "id",
    "date_arrivee",
    "label",
    "sous_label",
]


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def _charger_json(chemin: Path) -> list[dict]:
    """Charge le fichier JSON et retourne la liste des saisines."""
    if not chemin.exists():
        logger.error(
            f"Fichier introuvable : {chemin}\n"
            "Avez-vous bien lancé pipeline/02_classify.py avant ?"
        )
        sys.exit(1)

    with open(chemin, encoding="utf-8") as f:
        saisines = json.load(f)

    logger.info(f"Saisines chargées : {len(saisines)} (depuis {chemin.name})")
    return saisines


def _construire_dataframe(saisines: list[dict]) -> pd.DataFrame:
    """
    Convertit la liste de saisines en DataFrame pandas.

    Transformations appliquées :
    - key_word (liste Python) -> chaîne de caractères "mot1, mot2, mot3"
    - date_arrivee -> type date Python (YYYY-MM-DD)
    - annee, mois -> entiers (Int64, supporte les valeurs nulles)
    - Toutes les autres colonnes : conservées telles quelles
    """
    df = pd.DataFrame(saisines)

    # Vérification des colonnes minimales attendues
    manquantes = [c for c in COLONNES_ATTENDUES if c not in df.columns]
    if manquantes:
        logger.error(
            f"Colonnes manquantes dans le fichier d'entrée : {manquantes}\n"
            "Le fichier saisines_classifiees.json semble incomplet."
        )
        sys.exit(1)

    # key_word : liste -> chaîne "mot1, mot2, mot3"
    # Nécessaire car Parquet stocke les tableaux différemment selon les lecteurs.
    # Une chaîne de texte est universellement lisible.
    if "key_word" in df.columns:
        df["key_word"] = df["key_word"].apply(
            lambda kw: ", ".join(kw) if isinstance(kw, list) else (kw or "")
        )

    # date_arrivee : chaîne "YYYY-MM-DD" -> type datetime
    # On en dérive aussi annee et mois pour faciliter les filtres dans le dashboard
    if "date_arrivee" in df.columns:
        dates = pd.to_datetime(df["date_arrivee"], errors="coerce")
        df["date_arrivee"] = dates.dt.date
        df["annee"] = dates.dt.year.astype("Int64")
        df["mois"] = dates.dt.month.astype("Int64")

    logger.info(f"DataFrame construit : {len(df)} lignes x {len(df.columns)} colonnes")
    return df


def _sauvegarder_parquet(df: pd.DataFrame, chemin: Path) -> None:
    """Sauvegarde le DataFrame au format Parquet (compression snappy par défaut)."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(chemin, index=False, engine="pyarrow")

    taille_mo = chemin.stat().st_size / 1_000_000
    logger.info(f"Fichier Parquet sauvegardé : {chemin} ({taille_mo:.1f} Mo)")


def _valider_avec_duckdb(chemin: Path) -> None:
    """
    Utilise DuckDB pour lire le Parquet et vérifier son contenu.

    DuckDB permet de faire des requêtes SQL directement sur le fichier Parquet,
    sans créer de base de données. C'est ce qui sera utilisé dans le dashboard.
    """
    # DuckDB sur Windows : les chemins doivent utiliser des slashs
    chemin_sql = str(chemin).replace("\\", "/")

    con = duckdb.connect()

    # Statistiques globales
    total, n_labels, n_poles, annee_min, annee_max = con.execute(
        f"""
        SELECT
            COUNT(*)              AS total,
            COUNT(DISTINCT label) AS n_labels,
            COUNT(DISTINCT pole)  AS n_poles,
            MIN(annee)            AS annee_min,
            MAX(annee)            AS annee_max
        FROM read_parquet('{chemin_sql}')
        """
    ).fetchone()

    logger.info("Validation DuckDB :")
    logger.info(f"  Total lignes      : {total}")
    logger.info(f"  Labels distincts  : {n_labels}")
    logger.info(f"  Poles             : {n_poles}")
    logger.info(f"  Annees couvertes  : {annee_min} - {annee_max}")

    # Top 5 labels
    top_labels = con.execute(
        f"""
        SELECT label, COUNT(*) AS n
        FROM read_parquet('{chemin_sql}')
        GROUP BY label
        ORDER BY n DESC
        LIMIT 5
        """
    ).fetchall()

    logger.info("  Top 5 labels :")
    for label, count in top_labels:
        pct = count / total * 100
        logger.info(f"    {label:<35} {count:>5} ({pct:.1f}%)")

    # Top 5 pôles
    top_poles = con.execute(
        f"""
        SELECT pole, COUNT(*) AS n
        FROM read_parquet('{chemin_sql}')
        GROUP BY pole
        ORDER BY n DESC
        LIMIT 5
        """
    ).fetchall()

    logger.info("  Top 5 poles :")
    for pole, count in top_poles:
        pct = count / total * 100
        logger.info(f"    {str(pole):<35} {count:>5} ({pct:.1f}%)")

    con.close()


def convertir():
    """Orchestre la conversion JSON -> Parquet."""
    saisines = _charger_json(FICHIER_ENTREE)
    df = _construire_dataframe(saisines)
    _sauvegarder_parquet(df, FICHIER_SORTIE)
    _valider_avec_duckdb(FICHIER_SORTIE)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Demarrage de l'etape 3 - Conversion Parquet")
    logger.info("=" * 55)
    convertir()
    logger.info("=" * 55)
    logger.info(f"Etape 3 terminee. Fichier : {FICHIER_SORTIE}")
