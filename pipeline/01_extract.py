"""
01_extract.py — Etape 1 du pipeline EDN1_v2

Ce script lit le fichier source Excel (concatenation.xlsx) et produit un fichier
JSON propre et normalisé, prêt pour la classification NLP (étape 02).

Décisions documentées :
- Lignes sans "Analyse" : conservées avec null (valeur statistique préservée)
- Feuille Excel : première feuille détectée automatiquement (robustesse)
- Futurs fichiers : même format supposé, validation des colonnes à l'entrée
- Doublons : dédupliqués sur l'id (on garde la première occurrence)
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------- CONFIGURATION DES LOGS ----------
# Le logger affiche la date, le niveau (INFO / WARNING / ERROR) et le message.
# Cela remplace les print() pour un suivi plus professionnel.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------- CHEMINS ----------
# On calcule tout par rapport à l'emplacement de CE script,
# jamais en dur. Ainsi le projet reste portable.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Fichier source Excel (ne pas modifier ce fichier !)
# Situé dans data/input/ de CE projet — exclu de Git via .gitignore.
FICHIER_SOURCE = PROJECT_DIR / "data" / "input" / "concatenation.xlsx"

# Fichier de sortie JSON
SORTIE_JSON = PROJECT_DIR / "data" / "processed" / "saisines_brutes.json"

# ---------- CORRESPONDANCE CODES ACADÉMIQUES → NOMS ----------
# Source : extrait depuis EDN1/back/projet/extraction_tri_excel_to_python.py
# Décision : si un code est inconnu, on le conserve tel quel (pas d'erreur)
CODES_ACADEMIQUES = {
    "AMI": "Amiens",
    "AXM": "Aix-Marseille",
    "BES": "Besançon",
    "BOR": "Bordeaux",
    "CLE": "Clermont-Ferrand",
    "CND": "Caen",
    "COM": "Corse (Ajaccio / Bastia)",
    "CRE": "Creteil",
    "DIJ": "Dijon",
    "GUA": "Guadeloupe",
    "GRE": "Grenoble",
    "GUY": "Guyane",
    "LIL": "Lille",
    "LIM": "Limoges",
    "LYO": "Lyon",
    "MAR": "Martinique",
    "MON": "Montpellier",
    "NAN": "Nantes",
    "NAT": "Nationale",
    "NCY": "Nancy-Metz",
    "NIC": "Nice",
    "NOR": "Normandie",
    "ORL": "Orleans-Tours",
    "PAR": "Paris",
    "POI": "Poitiers",
    "REI": "Reims",
    "REN": "Rennes",
    "REU": "La Reunion",
    "STR": "Strasbourg",
    "TOU": "Toulouse",
    "VER": "Versailles",
}

# ---------- CORRESPONDANCE COLONNES EXCEL → NOMS NORMALISÉS ----------
# On renomme les colonnes avec des noms snake_case sans accents ni espaces.
# Cela rend le code plus simple à écrire partout dans le projet.
RENOMMAGE_COLONNES = {
    "id": "id",
    "Date arrivée": "date_arrivee",
    "Date clôture fiche": "date_cloture",
    "Pôle en charge": "pole",
    "Catégorie": "categorie",
    "Sous-catégorie": "sous_categorie",
    "Domaine": "domaine",
    "Sous-domaine": "sous_domaine",
    "Aspect contextuel": "aspect_contextuel",
    "Nature de la saisine": "nature_saisine",
    "Réclamation : position du médiateur": "position_mediateur",
    "Impact de l'appui du médiateur": "impact_appui",
    "Analyse": "analyse",
}

# Colonnes obligatoires que le fichier Excel DOIT contenir.
# Si l'une manque, le script s'arrête avec un message clair.
COLONNES_OBLIGATOIRES = list(RENOMMAGE_COLONNES.keys())


# =============================================================================
# FONCTIONS PRINCIPALES
# =============================================================================

def charger_excel(chemin: Path) -> pd.DataFrame:
    """
    Lit le fichier Excel et retourne un DataFrame brut.

    On détecte automatiquement la première feuille disponible.
    Cela rend le script compatible avec de futurs fichiers aux noms de feuilles
    différents, sans avoir à modifier le code.

    Paramètres
    ----------
    chemin : Path
        Chemin absolu vers le fichier .xlsx à lire.

    Retourne
    --------
    pd.DataFrame
        Contenu brut du fichier, sans transformation.
    """
    if not chemin.exists():
        logger.error(f"Fichier introuvable : {chemin}")
        sys.exit(1)

    # Lire les noms de feuilles disponibles
    xl = pd.ExcelFile(chemin, engine="openpyxl")
    nom_feuille = xl.sheet_names[0]
    logger.info(f"Feuille Excel sélectionnée : '{nom_feuille}'")

    df = pd.read_excel(chemin, sheet_name=nom_feuille, engine="openpyxl")
    logger.info(f"Lignes brutes lues : {len(df)}")
    logger.info(f"Colonnes trouvées : {list(df.columns)}")
    return df


def valider_colonnes(df: pd.DataFrame) -> None:
    """
    Vérifie que toutes les colonnes obligatoires sont présentes dans le DataFrame.

    Si une colonne manque, on arrête le script avec un message d'erreur clair
    indiquant exactement quelles colonnes sont absentes.
    Cela protège contre un futur fichier Excel mal formaté.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame à vérifier.
    """
    colonnes_manquantes = [c for c in COLONNES_OBLIGATOIRES if c not in df.columns]
    if colonnes_manquantes:
        logger.error(
            f"Colonnes manquantes dans le fichier Excel : {colonnes_manquantes}\n"
            f"Colonnes présentes : {list(df.columns)}"
        )
        sys.exit(1)
    logger.info("Validation des colonnes : OK")


def nettoyer_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique toutes les transformations nécessaires sur le DataFrame brut :

    1. Sélectionne uniquement les colonnes utiles (celles du mapping)
    2. Renomme les colonnes en snake_case
    3. Convertit la colonne 'id' en entier (les non-numériques deviennent NaN)
    4. Supprime les lignes sans id valide (entier non nul)
    5. Déduplique sur l'id (on garde la première occurrence)
    6. Convertit les codes académiques en noms de villes
    7. Normalise les dates au format YYYY-MM-DD

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame brut issu de charger_excel().

    Retourne
    --------
    pd.DataFrame
        DataFrame nettoyé et normalisé.
    """
    # --- Étape 1 : ne garder que les colonnes utiles ---
    df = df[list(RENOMMAGE_COLONNES.keys())].copy()

    # --- Étape 2 : renommer en snake_case ---
    df = df.rename(columns=RENOMMAGE_COLONNES)
    logger.info("Colonnes renommées en snake_case")

    # --- Étape 3 : convertir l'id en entier ---
    # pd.to_numeric avec errors='coerce' transforme les valeurs non numériques en NaN
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

    # --- Étape 4 : supprimer les lignes sans id valide ---
    nb_avant = len(df)
    df = df[df["id"].notna()].copy()
    nb_supprimees = nb_avant - len(df)
    if nb_supprimees > 0:
        logger.warning(f"Lignes supprimées (id manquant ou invalide) : {nb_supprimees}")
    else:
        logger.info("Aucune ligne supprimée pour id manquant")

    # --- Étape 5 : dédupliquer sur l'id ---
    nb_avant = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    nb_doublons = nb_avant - len(df)
    if nb_doublons > 0:
        logger.warning(f"Doublons supprimés sur l'id : {nb_doublons}")
    else:
        logger.info("Aucun doublon détecté sur l'id")

    # --- Étape 6 : décoder les codes académiques ---
    # Si le code est inconnu, on le laisse tel quel (pas d'erreur silencieuse)
    df["pole"] = df["pole"].apply(
        lambda x: CODES_ACADEMIQUES.get(str(x).strip(), x) if pd.notna(x) else x
    )
    logger.info("Codes académiques traduits en noms de villes")

    # --- Étape 7 : normaliser les dates ---
    # pd.to_datetime convertit les dates Excel en objets datetime Python.
    # On les formate ensuite en chaîne YYYY-MM-DD, et les valeurs manquantes
    # restent None (pas de chaîne vide, pas d'erreur).
    for col_date in ["date_arrivee", "date_cloture"]:
        df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
        df[col_date] = df[col_date].dt.strftime("%Y-%m-%d").where(
            df[col_date].notna(), other=None
        )
    logger.info("Dates normalisées au format YYYY-MM-DD")

    return df


def convertir_en_json(df: pd.DataFrame) -> list:
    """
    Convertit le DataFrame en une liste de dictionnaires prête à sérialiser en JSON.

    Les valeurs NaN / <NA> de pandas (qui ne sont pas du JSON valide) sont
    remplacées par None, ce qui devient 'null' en JSON.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame nettoyé.

    Retourne
    --------
    list
        Liste de dictionnaires (un par saisine).
    """
    # Convertir Int64 → int natif Python (sinon json.dump échoue)
    df = df.copy()
    df["id"] = df["id"].astype(object).where(df["id"].notna(), other=None)
    df["id"] = df["id"].apply(lambda x: int(x) if x is not None else None)

    # Remplacer pandas NA → None
    df = df.where(pd.notnull(df), other=None)

    return df.to_dict(orient="records")


def sauvegarder_json(data: list, chemin: Path) -> None:
    """
    Sauvegarde la liste de saisines dans un fichier JSON.

    Le fichier est encodé en UTF-8 avec indentation pour être lisible par un humain.
    ensure_ascii=False préserve les caractères français (accents, etc.).

    Paramètres
    ----------
    data : list
        Liste de dictionnaires à sérialiser.
    chemin : Path
        Chemin du fichier de sortie.
    """
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Fichier JSON sauvegardé : {chemin}")


def afficher_resume(data: list) -> None:
    """
    Affiche un résumé lisible du résultat de l'extraction.

    Permet de vérifier rapidement que tout s'est bien passé
    sans avoir à ouvrir le fichier JSON.

    Paramètres
    ----------
    data : list
        Liste de dictionnaires produite par convertir_en_json().
    """
    nb_total = len(data)
    nb_sans_analyse = sum(1 for s in data if not s.get("analyse"))
    nb_avec_analyse = nb_total - nb_sans_analyse

    poles = {}
    for s in data:
        pole = s.get("pole") or "Inconnu"
        poles[pole] = poles.get(pole, 0) + 1
    top_poles = sorted(poles.items(), key=lambda x: x[1], reverse=True)[:5]

    logger.info("=" * 50)
    logger.info(f"RESUME DE L'EXTRACTION")
    logger.info(f"  Total saisines extraites : {nb_total}")
    logger.info(f"  Avec texte 'Analyse'     : {nb_avec_analyse}")
    logger.info(f"  Sans texte 'Analyse'     : {nb_sans_analyse} (conservees, valeur null)")
    logger.info(f"  Top 5 poles academiques  :")
    for pole, count in top_poles:
        logger.info(f"    {pole:<30} {count} saisines")
    logger.info("=" * 50)


# =============================================================================
# FONCTION PRINCIPALE — peut aussi être appelée depuis run_all.py
# =============================================================================

def extraire(fichier_source: Path = FICHIER_SOURCE, sortie: Path = SORTIE_JSON) -> list:
    """
    Fonction principale d'extraction. Orchestre toutes les étapes.

    Peut être appelée directement depuis run_all.py ou en standalone.

    Paramètres
    ----------
    fichier_source : Path
        Chemin vers le fichier Excel à lire (par défaut : concatenation.xlsx).
    sortie : Path
        Chemin du fichier JSON à produire (par défaut : data/processed/saisines_brutes.json).

    Retourne
    --------
    list
        Liste des saisines extraites (même structure que le JSON produit).
    """
    logger.info(f"Démarrage de l'extraction depuis : {fichier_source}")

    df = charger_excel(fichier_source)
    valider_colonnes(df)
    df = nettoyer_dataframe(df)
    data = convertir_en_json(df)
    sauvegarder_json(data, sortie)
    afficher_resume(data)

    logger.info(f"Extraction terminée. {len(data)} saisines prêtes pour l'étape 02.")
    return data


# =============================================================================
# POINT D'ENTRÉE STANDALONE
# =============================================================================

if __name__ == "__main__":
    extraire()
