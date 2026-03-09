"""
Étape 4 — Calcul des embeddings sémantiques

Génère un vecteur numérique (embedding) pour chaque saisine à partir du
texte de la colonne "analyse". Ces vecteurs capturent le sens du texte,
ce qui permet de retrouver des saisines similaires même quand elles
n'utilisent pas les mêmes mots.

Modèle utilisé : paraphrase-multilingual-MiniLM-L12-v2
  - Multilingue (français natif)
  - Léger (~400 Mo) et rapide sur CPU
  - Produit des vecteurs de dimension 384

Sortie :
  data/processed/embeddings.npy      — matrice numpy de forme (N, 384)
  data/processed/embeddings_ids.json — liste des IDs dans le même ordre

Durée estimée : 2 à 5 min sur CPU pour 6 000 saisines.

Usage :
    python pipeline/04_embeddings.py
"""

import json
from pathlib import Path

import duckdb
import numpy as np

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

RACINE = Path(__file__).resolve().parent.parent
PARQUET = str(RACINE / "data" / "processed" / "saisines.parquet").replace("\\", "/")
EMBEDDINGS = RACINE / "data" / "processed" / "embeddings.npy"
IDS_JSON = RACINE / "data" / "processed" / "embeddings_ids.json"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    # 1. Chargement des saisines depuis le Parquet
    print("Chargement des saisines depuis le Parquet...")
    df = duckdb.query(
        f'SELECT id, "analyse" FROM read_parquet(\'{PARQUET}\')'
    ).df()
    print(f"  → {len(df)} saisines chargées.")

    textes = df["analyse"].fillna("").tolist()
    ids = df["id"].tolist()

    # 2. Chargement du modèle
    print("Chargement du modèle paraphrase-multilingual-MiniLM-L12-v2...")
    print("  (premier lancement : téléchargement ~400 Mo, une seule fois)")
    from sentence_transformers import SentenceTransformer  # noqa

    modele = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("  → Modèle chargé.")

    # 3. Calcul des embeddings
    print(f"Calcul des embeddings pour {len(textes)} saisines...")
    embeddings = modele.encode(
        textes,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )
    print(f"  → Embeddings calculés : forme {embeddings.shape}")

    # 4. Sauvegarde
    np.save(str(EMBEDDINGS), embeddings)
    with open(IDS_JSON, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)

    print(f"\nFichiers sauvegardés :")
    print(f"  {EMBEDDINGS}  ({embeddings.nbytes / 1024 / 1024:.1f} Mo)")
    print(f"  {IDS_JSON}")
    print("\nRecherche sémantique prête. Lancez le dashboard : streamlit run app/Home.py")


if __name__ == "__main__":
    main()
