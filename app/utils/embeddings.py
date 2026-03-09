"""
Module embeddings — Recherche sémantique sur les saisines.

Charge les embeddings précalculés (data/processed/embeddings.npy) et
permet de retrouver les saisines dont le sens est le plus proche d'une
requête en français, indépendamment des mots utilisés.

Fonctionnement :
  1. On encode la requête utilisateur en un vecteur de dimension 384
  2. On calcule la similarité cosinus entre ce vecteur et tous les
     embeddings des saisines (produit scalaire après normalisation)
  3. On retourne les k saisines avec les scores les plus élevés

Nécessite que pipeline/04_embeddings.py ait été exécuté au préalable.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

_RACINE = Path(__file__).resolve().parent.parent.parent
EMBEDDINGS_PATH = _RACINE / "data" / "processed" / "embeddings.npy"
IDS_PATH = _RACINE / "data" / "processed" / "embeddings_ids.json"


# ---------------------------------------------------------------------------
# Fonctions publiques
# ---------------------------------------------------------------------------


def embeddings_disponibles() -> bool:
    """Vérifie si les fichiers d'embeddings ont été générés."""
    return EMBEDDINGS_PATH.exists() and IDS_PATH.exists()


@st.cache_resource(show_spinner=False)
def _charger_ressources():
    """
    Charge les embeddings normalisés, les IDs et le modèle SentenceTransformer.
    Mis en cache avec st.cache_resource : chargé une seule fois par session.
    Retourne (embeddings_norm, ids, modele) ou (None, None, None) si absent.
    """
    if not embeddings_disponibles():
        return None, None, None

    from sentence_transformers import SentenceTransformer  # noqa

    embeddings = np.load(str(EMBEDDINGS_PATH))
    with open(IDS_PATH, "r", encoding="utf-8") as f:
        ids = json.load(f)

    # Normalisation pour la similarité cosinus (produit scalaire = cos sim)
    normes = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / np.maximum(normes, 1e-9)

    modele = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return embeddings_norm, ids, modele


def recherche_semantique(
    query: str, df_complet: pd.DataFrame, k: int = 10
) -> pd.DataFrame:
    """
    Retourne les k saisines sémantiquement les plus proches de la requête.

    Args:
        query       : question ou phrase en français
        df_complet  : DataFrame complet des saisines (id, label, analyse, ...)
        k           : nombre de résultats à retourner

    Returns:
        DataFrame des k saisines triées par score décroissant,
        avec une colonne supplémentaire 'similarité' (entre 0 et 1).

    Raises:
        FileNotFoundError : si les embeddings n'ont pas encore été générés.
        RuntimeError      : si sentence-transformers n'est pas installé.
    """
    embeddings_norm, ids, modele = _charger_ressources()

    if embeddings_norm is None:
        raise FileNotFoundError(
            "Les fichiers d'embeddings sont introuvables. "
            "Lancez d'abord : python pipeline/04_embeddings.py"
        )

    # Encoder la requête et la normaliser
    vecteur_query = modele.encode([query], convert_to_numpy=True)
    norme = np.linalg.norm(vecteur_query)
    vecteur_query_norm = vecteur_query / max(norme, 1e-9)

    # Similarité cosinus = produit scalaire (vecteurs déjà normalisés)
    scores = (embeddings_norm @ vecteur_query_norm.T).flatten()

    # Sélection des top-k
    indices_top = np.argsort(scores)[::-1][:k]
    ids_top = [ids[i] for i in indices_top]
    scores_top = [float(scores[i]) for i in indices_top]

    # Filtrage du DataFrame complet
    score_map = dict(zip(ids_top, scores_top))
    df_res = df_complet[df_complet["id"].isin(ids_top)].copy()
    df_res["similarité"] = df_res["id"].map(score_map)
    df_res = df_res.sort_values("similarité", ascending=False).reset_index(drop=True)

    return df_res
