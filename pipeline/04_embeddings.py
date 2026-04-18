"""
Étape 4 — Génération des embeddings sémantiques

Encode la colonne "analyse" de toutes les saisines avec un modèle multilingue
(sentence-transformers) et sauvegarde la matrice en .npy pour la recherche sémantique.

Usage :
    python pipeline/04_embeddings.py

Prérequis :
    pip install sentence-transformers
    python pipeline/03_to_parquet.py  (saisines.parquet doit exister)

Sortie :
    data/processed/embeddings.npy  — matrice float32 (n_saisines × 384)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging

import duckdb
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

PARQUET = ROOT / "data" / "processed" / "saisines.parquet"
SORTIE = ROOT / "data" / "processed" / "embeddings.npy"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def main():
    if not PARQUET.exists():
        log.error("saisines.parquet introuvable. Lancez d'abord 03_to_parquet.py")
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers non installé. Lancez : pip install sentence-transformers")
        sys.exit(1)

    log.info("Chargement des saisines depuis %s", PARQUET)
    parquet_path = str(PARQUET).replace("\\", "/")
    with duckdb.connect() as con:
        df = con.execute(
            f'SELECT id, "analyse" FROM read_parquet(\'{parquet_path}\')'
        ).df()

    log.info("%d saisines chargées", len(df))

    textes = df["analyse"].fillna("").tolist()

    log.info("Chargement du modèle %s …", MODEL_NAME)
    modele = SentenceTransformer(MODEL_NAME)

    log.info("Encodage en cours (peut prendre 1-2 minutes) …")
    embeddings = modele.encode(
        textes,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(SORTIE), embeddings.astype("float32"))

    log.info("Embeddings sauvegardés → %s  (shape: %s)", SORTIE, embeddings.shape)


if __name__ == "__main__":
    main()
