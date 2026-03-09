"""
eval.py — Evaluation du classifieur NLP

Ce script propose deux modes d'évaluation :

MODE 1 — Comparaison NLP vs Gemini (optimiste, rapide) :
    On compare les prédictions NLP avec les labels Gemini sur les 1 350 saisines communes.
    LIMITE : le modèle a été entraîné sur ces mêmes données -> score artificiel.

MODE 2 -- Evaluation honnête avec jeu de test caché (recommandé) :
    On sépare les 1 350 saisines en 80% entraînement / 20% test.
    On entraîne un modèle TEMPORAIRE sur les 80%, on le teste sur les 20% jamais vus.
    Ce score reflète ce que le modèle ferait sur de nouvelles saisines inconnues.
    Ce modèle temporaire n'écrase PAS les modèles en production dans data/models/.

Usage :
    # Mode 1 (rapide, optimiste) :
    C:\\ProgramData\\anaconda3\\python.exe pipeline/eval.py

    # Mode 2 (honnête, recommandé) :
    C:\\ProgramData\\anaconda3\\python.exe pipeline/eval.py --honnete
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# On ajoute la racine du projet au path pour importer le module nlp
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from nlp.classifier import _construire_pipeline, classifier_lot, construire_texte, entrainer

# ---------- CHEMINS ----------
NLP_JSON = PROJECT_DIR / "data" / "processed" / "saisines_classifiees.json"
GEMINI_JSON = PROJECT_DIR / "data" / "input" / "gemini_labels.json"


def charger_json(chemin: Path) -> list:
    if not chemin.exists():
        print(f"[ERREUR] Fichier introuvable : {chemin}")
        sys.exit(1)
    with chemin.open("r", encoding="utf-8") as f:
        return json.load(f)


def afficher_resultats(correct_label, correct_sous, total, confusion, erreurs_par_label, titre):
    """Affiche le rapport d'évaluation standard."""
    accord_label = correct_label / total * 100
    accord_sous = correct_sous / total * 100

    print("\n" + "=" * 65)
    print(titre)
    print("=" * 65)
    print(f"  Saisines evaluees  : {total}")
    print(f"  Accord label       : {correct_label}/{total}  ({accord_label:.1f}%)")
    print(f"  Accord sous_label  : {correct_sous}/{total}  ({accord_sous:.1f}%)")

    print("\n--- Accord par categorie ---")
    en_tete = f"  {'Categorie (label reel)':<35} {'Total':>6} {'Correct':>8} {'Accord':>7}"
    print(en_tete)
    print("  " + "-" * (len(en_tete) - 2))
    for vrai_label in sorted(confusion.keys()):
        comptage = confusion[vrai_label]
        total_label = sum(comptage.values())
        correct_label_i = comptage.get(vrai_label, 0)
        pct = correct_label_i / total_label * 100 if total_label else 0
        barre = "#" * int(pct / 10)
        print(f"  {vrai_label:<35} {total_label:>6} {correct_label_i:>8} {pct:>6.0f}% {barre}")

    if erreurs_par_label:
        print("\n--- Principales confusions ---")
        for vrai_label in sorted(erreurs_par_label.keys()):
            erreurs = erreurs_par_label[vrai_label]
            if not erreurs:
                continue
            top = Counter(erreurs).most_common(3)
            confusion_str = ", ".join([f"'{k}' ({v}x)" for k, v in top])
            print(f"  '{vrai_label}' -> confondu avec : {confusion_str}")

    if accord_label >= 95:
        qualite, commentaire = "EXCELLENT", "Le modele classe tres bien."
    elif accord_label >= 80:
        qualite, commentaire = "TRES BON", "Le modele est fiable."
    elif accord_label >= 65:
        qualite, commentaire = "ACCEPTABLE", "Globalement fiable, perfectible."
    else:
        qualite, commentaire = "INSUFFISANT", "Il faut plus de donnees d'entrainement."

    print(f"\n  label      : {accord_label:.1f}%  {qualite} -- {commentaire}")
    print(f"  sous_label : {accord_sous:.1f}%  (moins fiable : ~60 sous-categories)")

    return accord_label, accord_sous


def comparer_nlp_vs_gemini():
    """
    MODE 1 — Comparaison entre les prédictions NLP et les labels Gemini.

    IMPORTANT : le modèle a été entraîné sur ces mêmes données Gemini.
    Ce score est donc OPTIMISTE — ne pas le présenter comme une preuve de qualité absolue.
    """
    print("MODE 1 : Comparaison NLP vs Gemini (score optimiste)")
    print("  Le modele a ete entraine sur ces memes donnees -> score artificiel.\n")

    nlp_data = charger_json(NLP_JSON)
    nlp_par_id = {r["id"]: r for r in nlp_data if r.get("id") and r.get("label")}

    gemini_data = charger_json(GEMINI_JSON)
    gemini_par_id = {r["id"]: r for r in gemini_data if r.get("id") and r.get("label")}

    ids_communs = set(gemini_par_id.keys()) & set(nlp_par_id.keys())
    print(f"  Saisines Gemini : {len(gemini_par_id)} | Saisines NLP : {len(nlp_par_id)}")
    print(f"  Saisines communes pour evaluation : {len(ids_communs)}")

    correct_label = correct_sous = 0
    erreurs_par_label = defaultdict(list)
    confusion = defaultdict(Counter)

    for sid in ids_communs:
        g, n = gemini_par_id[sid], nlp_par_id[sid]
        vrai_label, pred_label = g["label"], n["label"]
        vrai_sous, pred_sous = g.get("sous_label", ""), n.get("sous_label", "")
        confusion[vrai_label][pred_label] += 1
        if pred_label == vrai_label:
            correct_label += 1
        else:
            erreurs_par_label[vrai_label].append(pred_label)
        if pred_sous == vrai_sous:
            correct_sous += 1

    afficher_resultats(
        correct_label, correct_sous, len(ids_communs),
        confusion, erreurs_par_label,
        "RESULTAT MODE 1 (optimiste — donnees d'entrainement)"
    )
    print("""
  RAPPEL : ce score est obtenu sur les donnees qui ont servi a entrainer le modele.
  Il sera toujours trop elevé. Lancez --honnete pour le vrai score.
""")


def evaluer_honnete(taille_test: float = 0.2):
    """
    MODE 2 — Evaluation honnête avec jeu de test caché (train/test split).

    On sépare les 1 350 saisines Gemini en :
    - 80% pour entraîner un modèle TEMPORAIRE
    - 20% pour tester ce modèle sur des données qu'il n'a JAMAIS vues

    Ce modèle temporaire n'écrase pas les modèles en production (data/models/).
    

    Paramètre
    ---------
    taille_test : float
        Proportion des données réservées pour le test (défaut : 20%).
    """
    import random
    print(f"MODE 2 : Evaluation honnete (train {int((1-taille_test)*100)}% / test {int(taille_test*100)}%)")
    print("  Un modele TEMPORAIRE est entraine sur 80% des donnees Gemini.")
    print("  Il est teste sur les 20% restants qu'il n'a JAMAIS vus.")
    print("  Ce modele temporaire n'ecrase pas les modeles de production.\n")

    # --- Charger les données Gemini ---
    gemini_data = charger_json(GEMINI_JSON)

    # Adapter les noms de colonnes (les données Gemini utilisent l'ancien format)
    def adapter_saisine(r):
        """Convertit les anciens noms de colonnes en snake_case."""
        if "Analyse" in r and "analyse" not in r:
            return {
                **r,
                "analyse": r.get("Analyse"),
                "categorie": r.get("Catégorie"),
                "sous_categorie": r.get("Sous-catégorie"),
                "domaine": r.get("Domaine"),
                "sous_domaine": r.get("Sous-domaine"),
                "nature_saisine": r.get("Nature de la saisine"),
                "aspect_contextuel": r.get("Aspect contextuel"),
                "position_mediateur": r.get("Réclamation : position du médiateur"),
            }
        return r

    # Ne garder que les saisines avec label, sous_label ET texte
    saisines_valides = []
    for r in gemini_data:
        r = adapter_saisine(r)
        texte = construire_texte(r)
        if r.get("label") and r.get("sous_label") and texte.strip():
            saisines_valides.append((r, texte))

    print(f"  Saisines Gemini utilisables : {len(saisines_valides)}")

    # --- Mélanger et séparer train / test ---
    random.seed(42)  # graine fixe pour reproductibilité (meme résultat à chaque run)
    random.shuffle(saisines_valides)
    seuil = int(len(saisines_valides) * (1 - taille_test))
    train_data = saisines_valides[:seuil]
    test_data = saisines_valides[seuil:]

    print(f"  Entrainement sur : {len(train_data)} saisines")
    print(f"  Test sur         : {len(test_data)} saisines (jamais vues)\n")

    # --- Séparer train / test (saisines complètes, pas juste textes) ---
    saisines_train = [r for r, _ in train_data]
    saisines_test = [r for r, _ in test_data]
    labels_test = [r["label"] for r in saisines_test]
    sous_labels_test = [r["sous_label"] for r in saisines_test]

    # --- Entraîner un modèle TEMPORAIRE ---
    print("  Entrainement du modele temporaire...")
    pipeline_label, pipeline_sous_label, vectoriseur = entrainer(saisines_train, cv_folds=0)
    print("  Modele temporaire entraine.\n")

    # --- Evaluer sur le jeu de test via classifier_lot ---
    resultats_test = classifier_lot(saisines_test, pipeline_label, pipeline_sous_label, vectoriseur)
    preds_label = [r["label"] or "" for r in resultats_test]
    preds_sous = [r["sous_label"] or "" for r in resultats_test]

    correct_label = correct_sous = 0
    erreurs_par_label = defaultdict(list)
    confusion = defaultdict(Counter)

    for vrai_label, pred_label, vrai_sous, pred_sous in zip(
        labels_test, preds_label, sous_labels_test, preds_sous
    ):
        confusion[vrai_label][pred_label] += 1
        if pred_label == vrai_label:
            correct_label += 1
        else:
            erreurs_par_label[vrai_label].append(pred_label)
        if pred_sous == vrai_sous:
            correct_sous += 1

    afficher_resultats(
        correct_label, correct_sous, len(test_data),
        confusion, erreurs_par_label,
        "RESULTAT MODE 2 (honnete -- donnees jamais vues)"
    )
    print("""
  CE SCORE EST FIABLE. 
  Il reflete la performance reelle sur de nouvelles saisines inconnues.
""")


# =============================================================================
# POINT D'ENTREE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation de la qualite du classifieur NLP."
    )
    parser.add_argument(
        "--honnete",
        action="store_true",
        help="Evaluation honnete : entraine sur 80%%, teste sur 20%% jamais vus.",
    )
    args = parser.parse_args()

    if args.honnete:
        evaluer_honnete()
    else:
        comparer_nlp_vs_gemini()
