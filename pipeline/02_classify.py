"""
02_classify.py — Etape 2 du pipeline EDN1_v2

Ce script charge les saisines extraites à l'étape 1, les classe avec le modèle NLP,
et produit un fichier JSON enrichi avec un label et un sous_label pour chaque saisine.

Comportement :
- Par défaut : charge les modèles déjà entraînés depuis data/models/ (rapide)
- Avec --retrain : ré-entraîne les modèles depuis les données Gemini avant de classifier

Usage :
    python pipeline/02_classify.py              # utilise les modèles existants
    python pipeline/02_classify.py --retrain    # ré-entraîne puis classe
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# On ajoute la racine du projet au path Python pour que "import nlp.classifier" fonctionne
# (nécessaire quand on lance le script en standalone depuis n'importe quel dossier)
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from nlp.classifier import (
    charger_modeles,
    classifier_lot,
    entrainer,
    modeles_presents,
    sauvegarder_modeles,
)

# ---------- CONFIGURATION DES LOGS ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------- CHEMINS ----------
ENTREE_JSON = PROJECT_DIR / "data" / "processed" / "saisines_brutes.json"
SORTIE_JSON = PROJECT_DIR / "data" / "processed" / "saisines_classifiees.json"
DOSSIER_MODELES = PROJECT_DIR / "data" / "models"

# Données d'entraînement Gemini (utilisées uniquement si --retrain ou si modèles absents)
# Situé dans data/input/ de CE projet — exclu de Git via .gitignore.
DONNEES_GEMINI = PROJECT_DIR / "data" / "input" / "gemini_labels.json"


# =============================================================================
# FONCTIONS
# =============================================================================

def charger_saisines(chemin: Path) -> list:
    """
    Charge le fichier JSON produit par 01_extract.py.

    Paramètres
    ----------
    chemin : Path
        Chemin vers saisines_brutes.json.

    Retourne
    --------
    list
        Liste de dictionnaires (une saisine par dict).
    """
    if not chemin.exists():
        logger.error(
            f"Fichier introuvable : {chemin}\n"
            f"Avez-vous bien lancé l'étape 1 (python pipeline/01_extract.py) ?"
        )
        sys.exit(1)
    with chemin.open("r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Saisines chargées depuis : {chemin} ({len(data)} saisines)")
    return data


def charger_donnees_gemini(chemin: Path) -> list:
    """
    Charge les 1 340 saisines labélisées par Gemini pour l'entraînement.

    Ces données ont été produites dans la v1 du projet et servent de "jeu
    d'entraînement" : c'est grâce à elles que le modèle local a appris à classifier.

    Paramètres
    ----------
    chemin : Path
        Chemin vers output_tri_structure2.json.

    Retourne
    --------
    list
        Liste des saisines labélisées (champs label et sous_label remplis).
    """
    if not chemin.exists():
        logger.error(
            f"Données Gemini introuvables : {chemin}\n"
            f"Ces données sont nécessaires pour le ré-entraînement."
        )
        sys.exit(1)
    with chemin.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # On ne garde que les saisines qui ont un label ET un sous_label
    etiquetees = [s for s in data if s.get("label") and s.get("sous_label")]
    logger.info(
        f"Données Gemini chargées : {len(etiquetees)} saisines étiquetées "
        f"(sur {len(data)} total)"
    )
    return etiquetees


def sauvegarder_json(data: list, chemin: Path) -> None:
    """
    Sauvegarde la liste de saisines classifiées en JSON.

    Paramètres
    ----------
    data : list
        Liste de saisines enrichies.
    chemin : Path
        Chemin de sortie.
    """
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Résultats sauvegardés : {chemin}")


def afficher_resume(saisines_classifiees: list) -> None:
    """
    Affiche un résumé de la classification : distribution des labels,
    nombre de saisines sans label, etc.

    Paramètres
    ----------
    saisines_classifiees : list
        Liste des saisines après classification.
    """
    total = len(saisines_classifiees)
    sans_label = sum(1 for s in saisines_classifiees if not s.get("label"))
    avec_label = total - sans_label

    # Compter les saisines par label
    comptage = {}
    for s in saisines_classifiees:
        label = s.get("label") or "(non classifié)"
        comptage[label] = comptage.get(label, 0) + 1

    labels_tries = sorted(comptage.items(), key=lambda x: x[1], reverse=True)

    logger.info("=" * 55)
    logger.info("RESUME DE LA CLASSIFICATION")
    logger.info(f"  Total saisines classifiées : {avec_label} / {total}")
    logger.info(f"  Non classifiées (sans texte) : {sans_label}")
    logger.info(f"  Distribution des labels :")
    for label, count in labels_tries:
        pct = count / total * 100
        logger.info(f"    {label:<35} {count:>4} ({pct:.1f}%)")
    logger.info("=" * 55)


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def classifier(retrain: bool = False) -> list:
    """
    Fonction principale. Orchestre toute l'étape 2.

    Paramètres
    ----------
    retrain : bool
        Si True, ré-entraîne les modèles depuis les données Gemini avant de classifier.
        Si False (défaut), utilise les modèles existants dans data/models/.

    Retourne
    --------
    list
        Liste des saisines classifiées (même structure que le JSON produit).
    """
    logger.info("Démarrage de l'étape 2 — Classification NLP")

    # --- Charger les saisines à classifier ---
    saisines = charger_saisines(ENTREE_JSON)

    # --- Charger ou entraîner les modèles ---
    if retrain or not modeles_presents(DOSSIER_MODELES):
        if not modeles_presents(DOSSIER_MODELES):
            logger.info("Aucun modèle trouvé — entraînement nécessaire.")
        else:
            logger.info("Option --retrain activée — ré-entraînement depuis les données Gemini.")

        donnees_gemini = charger_donnees_gemini(DONNEES_GEMINI)
        pipeline_label, pipeline_sous_label, vectoriseur = entrainer(donnees_gemini)
        sauvegarder_modeles(pipeline_label, pipeline_sous_label, vectoriseur, DOSSIER_MODELES)
    else:
        logger.info("Modèles trouvés — chargement direct (pas de ré-entraînement).")
        pipeline_label, pipeline_sous_label, vectoriseur = charger_modeles(DOSSIER_MODELES)

    # --- Classifier toutes les saisines ---
    logger.info(f"Classification de {len(saisines)} saisines en cours...")
    saisines_classifiees = classifier_lot(
        saisines,
        pipeline_label,
        pipeline_sous_label,
        vectoriseur,
    )

    # --- Sauvegarder le résultat ---
    sauvegarder_json(saisines_classifiees, SORTIE_JSON)

    # --- Afficher le résumé ---
    afficher_resume(saisines_classifiees)

    logger.info(f"Etape 2 terminée. Résultats dans : {SORTIE_JSON}")
    return saisines_classifiees


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Etape 2 — Classification NLP des saisines."
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Ré-entraîne les modèles depuis les données Gemini avant de classifier.",
    )
    args = parser.parse_args()
    classifier(retrain=args.retrain)
