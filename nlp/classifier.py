"""
classifier.py — Moteur NLP de classification des saisines

Ce module fait tout le travail intellectuel de la classification.
Il est importé par pipeline/02_classify.py qui s'occupe uniquement
d'orchestrer (charger les données, appeler ce module, sauvegarder).

Principe de fonctionnement :
    1. On combine les champs textuels de chaque saisine en une seule chaîne
    2. On vectorise cette chaîne via TF-IDF (transforme les mots en scores numériques)
    3. Deux classifieurs prédisent le label et le sous_label
    4. On vérifie que le sous_label est cohérent avec le label (taxonomie)
    5. On extrait des mots-clés via les scores TF-IDF
    6. On détecte le lieu via des expressions régulières (regex)

Adaptation v2 : utilise les noms de colonnes snake_case (analyse, categorie...)
au lieu des noms originaux français (Analyse, Catégorie...).
Les modèles .joblib sont compatibles car ils traitent uniquement du texte —
ils ne voient jamais les noms des colonnes.

Note sur la classification hiérarchique :
    Une approche hiérarchique (1 modèle sous_label par label) a été testée
    et n'apporte que +3.4% sur le sous_label (49.6% → 53.0%). Le gain ne
    justifie pas la complexité supplémentaire. 
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from nlp.taxonomy import NATURE_PROBLEME

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stop words français
# ---------------------------------------------------------------------------
FRENCH_STOP_WORDS = [
    "a", "au", "aux", "avec", "ce", "ces", "cet", "cette", "dans", "de", "des",
    "du", "elle", "elles", "en", "et", "eu", "eux", "il", "ils", "je", "la",
    "le", "les", "leur", "leurs", "lui", "ma", "mais", "me", "même", "mes",
    "mon", "ni", "nos", "notre", "nous", "on", "ou", "où", "par", "pas", "pour",
    "qu", "que", "qui", "sa", "se", "ses", "si", "son", "sur", "ta", "te",
    "tes", "ton", "tu", "un", "une", "vos", "votre", "vous", "y",
    "été", "être", "avoir", "est", "sont", "était", "ont", "avait",
    "car", "donc", "or", "cela", "tout", "plus", "aussi", "bien", "très",
    "peut", "fait", "faire", "lors", "après", "avant", "pendant",
    "afin", "ainsi", "alors", "comme", "selon", "suite", "car",
    "d", "j", "m", "s", "n", "c", "l",
    "ça", "là", "ya", "non", "oui", "dès", "lors",
]

# ---------------------------------------------------------------------------
# Patterns de détection du lieu
# ---------------------------------------------------------------------------
LIEU_PATTERNS = [
    (r"\b(salle de classe|en classe|pendant le cours)\b", "salle de classe"),
    (r"\b(cantine|restaurant scolaire|repas)\b", "cantine"),
    (r"\b(cour de r[eé]cr[eé]ation|cour d[e'][eé]cole|dans la cour)\b", "cour de récréation"),
    (r"\b(internat|dortoir|r[eé]sidence)\b", "internat"),
    (r"\b(gymnase|cours d[' ]EPS|sport scolaire)\b", "gymnase"),
    (r"\b(couloir|hall)\b", "couloir"),
    (r"\b(salle d[' ]examen|centre d[' ]examen|lors de l[' ]examen)\b", "salle d'examen"),
    (r"\b(en ligne|internet|num[eé]rique|plateforme|r[eé]seau social|whatsapp|instagram|tiktok|snapchat)\b", "en ligne"),
    (r"\b(toilettes|sanitaires|WC)\b", "sanitaires"),
    (r"\b(bus scolaire|transport scolaire|dans le bus)\b", "transport scolaire"),
    (r"\b(biblioth[eè]que|CDI|centre de documentation)\b", "bibliothèque / CDI"),
    (r"\b(permanence|salle de permanence)\b", "permanence"),
    (r"\b(r[eé]sidence universitaire|logement [Cc][Rr][Oo][Uu][Ss]|cit[eé] [Uu])\b", "résidence universitaire"),
    (r"\b(salle de sport|piscine scolaire)\b", "salle de sport"),
]


# ---------------------------------------------------------------------------
# Construction de la feature textuelle
# ---------------------------------------------------------------------------

def construire_texte(saisine: dict) -> str:
    """
    Combine les champs textuels d'une saisine en une seule chaîne de caractères.

    Le champ 'analyse' est répété 3 fois pour lui donner plus de poids dans le
    calcul TF-IDF. Les autres champs (catégorie, domaine...) donnent du contexte.
    """
    parties = []
    analyse = saisine.get("analyse") or ""
    if isinstance(analyse, str) and analyse.strip():
        parties.extend([analyse.strip()] * 3)

    for champ in ["categorie", "sous_categorie", "domaine", "sous_domaine",
                  "nature_saisine", "aspect_contextuel", "position_mediateur"]:
        valeur = saisine.get(champ)
        if valeur and isinstance(valeur, str) and valeur.strip():
            parties.append(valeur.strip())

    return " ".join(parties)


# ---------------------------------------------------------------------------
# Détection du lieu
# ---------------------------------------------------------------------------

def detecter_lieu(texte_analyse: str) -> Optional[str]:
    """Cherche un lieu concret dans le texte via expressions régulières."""
    if not texte_analyse:
        return None
    for pattern, lieu in LIEU_PATTERNS:
        if re.search(pattern, texte_analyse, re.IGNORECASE):
            return lieu
    return None


# ---------------------------------------------------------------------------
# Extraction de mots-clés
# ---------------------------------------------------------------------------

def extraire_mots_cles(texte: str, vectoriseur: TfidfVectorizer, n: int = 5) -> List[str]:
    """
    Extrait les n mots les plus représentatifs d'un texte selon leurs scores TF-IDF.
    Les mots avec un score élevé sont ceux qui apparaissent souvent dans ce texte
    mais rarement dans les autres — ce sont les mots les plus distinctifs.
    """
    if not texte or not texte.strip():
        return []
    try:
        matrice = vectoriseur.transform([texte])
        noms_features = vectoriseur.get_feature_names_out()
        scores = matrice.toarray()[0]
        indices_top = np.argsort(scores)[::-1]
        mots_cles = []
        for idx in indices_top:
            if scores[idx] == 0:
                break
            mot = noms_features[idx]
            if len(mot) > 3 and len(mot.split()) <= 2:
                mots_cles.append(mot)
            if len(mots_cles) >= n:
                break
        return mots_cles
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Validation taxonomique
# ---------------------------------------------------------------------------

def valider_sous_label(label: str, sous_label: str) -> str:
    """
    Vérifie que le sous_label est cohérent avec le label.
    Si non (ex: label="examens" mais sous_label="refus_bourse"), retourne "autre".
    """
    if label not in NATURE_PROBLEME:
        return "autre"
    if sous_label in NATURE_PROBLEME[label]["sous_labels"]:
        return sous_label
    return "autre"


# ---------------------------------------------------------------------------
# Construction du pipeline sklearn
# ---------------------------------------------------------------------------

def _construire_pipeline() -> Pipeline:
    """
    Construit un pipeline TF-IDF + Régression Logistique.
    Utilisé pour le classifieur label et le classifieur sous_label.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=25000,
            sublinear_tf=True,
            stop_words=FRENCH_STOP_WORDS,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=5.0,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

def entrainer(
    donnees_etiquetees: List[dict],
    cv_folds: int = 3,
) -> Tuple[Pipeline, Pipeline, TfidfVectorizer]:
    """
    Entraîne les classifieurs label et sous_label à partir des données Gemini.

    Cette fonction n'est appelée que si les modèles .joblib n'existent pas encore,
    ou si on force un ré-entraînement avec --retrain.

    Paramètres
    ----------
    donnees_etiquetees : list[dict]
        Les saisines labélisées par Gemini (gemini_labels.json).
    cv_folds : int
        Folds pour la cross-validation interne. 0 = désactivé.

    Retourne
    --------
    (pipeline_label, pipeline_sous_label, vectoriseur_mots_cles)
    """
    # Adapter les noms de colonnes (les données Gemini utilisent l'ancien format)
    def _adapter(record: dict) -> dict:
        if "Analyse" in record and "analyse" not in record:
            return {
                **record,
                "analyse": record.get("Analyse"),
                "categorie": record.get("Catégorie"),
                "sous_categorie": record.get("Sous-catégorie"),
                "domaine": record.get("Domaine"),
                "sous_domaine": record.get("Sous-domaine"),
                "nature_saisine": record.get("Nature de la saisine"),
                "aspect_contextuel": record.get("Aspect contextuel"),
                "position_mediateur": record.get("Réclamation : position du médiateur"),
            }
        return record

    df = pd.DataFrame([_adapter(r) for r in donnees_etiquetees])
    df["_texte"] = df.apply(lambda row: construire_texte(row.to_dict()), axis=1)
    df = df[
        df["label"].notna() &
        df["sous_label"].notna() &
        (df["_texte"].str.strip() != "")
    ].copy()

    if len(df) < 20:
        raise ValueError(f"Pas assez de données ({len(df)} exemples, minimum 20).")

    logger.info(f"Entraînement sur {len(df)} exemples — {df['label'].nunique()} labels distincts.")

    # Adapter cv_folds à la taille minimale des classes
    min_label = df["label"].value_counts().min()
    min_sl = df["sous_label"].value_counts().min()
    cv_l = min(cv_folds, int(min_label)) if cv_folds >= 2 else 0
    cv_sl = min(cv_folds, int(min_sl)) if cv_folds >= 2 else 0

    # --- Classifieur label ---
    pipeline_label = _construire_pipeline()
    if cv_l >= 2:
        scores = cross_val_score(pipeline_label, df["_texte"], df["label"], cv=cv_l, scoring="f1_macro")
        logger.info(f"  Cross-val label — F1 macro : {scores.mean():.3f} ± {scores.std():.3f}")
    pipeline_label.fit(df["_texte"], df["label"])
    logger.info("  Classifieur 'label' entraîné.")

    # --- Classifieur sous_label ---
    pipeline_sous_label = _construire_pipeline()
    if cv_sl >= 2:
        scores_sl = cross_val_score(pipeline_sous_label, df["_texte"], df["sous_label"], cv=cv_sl, scoring="f1_macro")
        logger.info(f"  Cross-val sous_label — F1 macro : {scores_sl.mean():.3f} ± {scores_sl.std():.3f}")
    pipeline_sous_label.fit(df["_texte"], df["sous_label"])
    logger.info("  Classifieur 'sous_label' entraîné.")

    # --- Vectoriseur mots-clés ---
    vectoriseur_mots_cles = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
        stop_words=FRENCH_STOP_WORDS,
    )
    vectoriseur_mots_cles.fit(df["_texte"])
    logger.info("  Vectoriseur mots-clés entraîné.")

    return pipeline_label, pipeline_sous_label, vectoriseur_mots_cles


# ---------------------------------------------------------------------------
# Classification en lot
# ---------------------------------------------------------------------------

def classifier_lot(
    saisines: List[dict],
    pipeline_label: Pipeline,
    pipeline_sous_label: Pipeline,
    vectoriseur_mots_cles: TfidfVectorizer,
) -> List[dict]:
    """
    Classe un lot de saisines et retourne les saisines enrichies.

    Ajoute à chaque saisine : label, sous_label, lieu, key_word, source_label.
    Les saisines sans texte reçoivent null pour tous ces champs.
    """
    textes = [construire_texte(s) for s in saisines]
    indices_avec_texte = [i for i, t in enumerate(textes) if t.strip()]
    indices_sans_texte = [i for i, t in enumerate(textes) if not t.strip()]

    if indices_sans_texte:
        logger.warning(f"{len(indices_sans_texte)} saisines sans texte -> label null.")

    resultats: List[Optional[dict]] = [None] * len(saisines)

    if indices_avec_texte:
        textes_a_classifier = [textes[i] for i in indices_avec_texte]
        labels_predits = pipeline_label.predict(textes_a_classifier)
        sous_labels_predits = pipeline_sous_label.predict(textes_a_classifier)

        for pos, idx in enumerate(indices_avec_texte):
            saisine = saisines[idx]
            texte = textes[idx]
            label = labels_predits[pos]
            sous_label = valider_sous_label(label, sous_labels_predits[pos])
            lieu = detecter_lieu(str(saisine.get("analyse") or ""))
            mots_cles = extraire_mots_cles(texte, vectoriseur_mots_cles, n=5)

            resultats[idx] = {
                **saisine,
                "label": label,
                "sous_label": sous_label,
                "lieu": lieu,
                "key_word": mots_cles,
                "source_label": "nlp",
            }

    for idx in indices_sans_texte:
        resultats[idx] = {
            **saisines[idx],
            "label": None,
            "sous_label": None,
            "lieu": None,
            "key_word": [],
            "source_label": None,
        }

    return resultats


# ---------------------------------------------------------------------------
# Sauvegarde et chargement des modèles
# ---------------------------------------------------------------------------

def sauvegarder_modeles(
    pipeline_label: Pipeline,
    pipeline_sous_label: Pipeline,
    vectoriseur_mots_cles: TfidfVectorizer,
    dossier_modeles: Path,
) -> None:
    """Sauvegarde les 3 modèles dans le dossier spécifié."""
    dossier_modeles.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_label, dossier_modeles / "label_classifier.joblib")
    joblib.dump(pipeline_sous_label, dossier_modeles / "sous_label_classifier.joblib")
    joblib.dump(vectoriseur_mots_cles, dossier_modeles / "keyword_vectorizer.joblib")
    logger.info(f"Modèles sauvegardés dans : {dossier_modeles}")


def charger_modeles(
    dossier_modeles: Path,
) -> Tuple[Pipeline, Pipeline, TfidfVectorizer]:
    """Charge les 3 modèles depuis le dossier spécifié."""
    pipeline_label = joblib.load(dossier_modeles / "label_classifier.joblib")
    pipeline_sous_label = joblib.load(dossier_modeles / "sous_label_classifier.joblib")
    vectoriseur_mots_cles = joblib.load(dossier_modeles / "keyword_vectorizer.joblib")
    logger.info(f"Modèles chargés depuis : {dossier_modeles}")
    return pipeline_label, pipeline_sous_label, vectoriseur_mots_cles


def modeles_presents(dossier_modeles: Path) -> bool:
    """Vérifie si les 3 modèles nécessaires sont présents."""
    return (
        (dossier_modeles / "label_classifier.joblib").exists() and
        (dossier_modeles / "sous_label_classifier.joblib").exists() and
        (dossier_modeles / "keyword_vectorizer.joblib").exists()
    )
