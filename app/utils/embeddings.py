"""
Recherche sémantique sur les saisines.

Utilise sentence-transformers pour encoder les textes et calculer
la similarité cosinus entre la requête et les saisines.

Prérequis :
- Avoir généré les embeddings : python pipeline/04_embeddings.py
- pip install sentence-transformers
"""

from pathlib import Path

import numpy as np
import pandas as pd

_EMBEDDINGS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "processed"
    / "embeddings.npy"
)

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None


def embeddings_disponibles() -> bool:
    return _EMBEDDINGS_PATH.exists()


def _charger_modele():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(e) from e
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def recherche_semantique(query: str, df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Retourne les k saisines les plus proches sémantiquement de la requête.

    Paramètres
    ----------
    query : question ou situation décrite en français
    df    : DataFrame complet des saisines (doit contenir la colonne "analyse")
    k     : nombre de résultats à retourner

    Retourne
    --------
    DataFrame trié par similarité décroissante, avec une colonne "similarité" (float 0-1).
    """
    if not _EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Fichier embeddings introuvable. "
            "Lancez : python pipeline/04_embeddings.py"
        )

    modele = _charger_modele()

    embeddings = np.load(str(_EMBEDDINGS_PATH))

    vecteur_query = modele.encode([query], normalize_embeddings=True)[0]

    scores = embeddings @ vecteur_query

    idx_top = np.argsort(scores)[::-1][:k]

    df_res = df.iloc[idx_top].copy()
    df_res["similarité"] = scores[idx_top]

    return df_res.reset_index(drop=True)
