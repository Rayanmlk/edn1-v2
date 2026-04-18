# Architecture cible — EDN1 v2

## Structure des dossiers

```
EDN1_v2/
│
├── data/
│   ├── input/              # Fichiers Excel bruts (ne pas modifier)
│   │   └── (lien vers C:\Users\rayan\Documents\PROJETS\EDN1\back\data\input\)
│   ├── processed/          # Données intermédiaires (JSON, Parquet)
│   │   ├── saisines_brutes.json       # Toutes les saisines extraites et nettoyées
│   │   ├── saisines_classifiees.json  # Après classification NLP
│   │   ├── saisines.parquet           # Format colonnaire pour DuckDB
│   │   └── keywords.parquet           # Mots-clés dénormalisés
│   └── models/             # Modèles NLP entraînés
│       ├── label_classifier.joblib
│       ├── sous_label_classifier.joblib
│       └── keyword_vectorizer.joblib
│
├── pipeline/               # Scripts ETL et classification (un script = une étape)
│   ├── 01_extract.py       # Excel → JSON unifié (tous les fichiers Excel)
│   ├── 02_classify.py      # JSON → JSON enrichi (NLP local)
│   ├── 03_to_parquet.py    # JSON → Parquet + DuckDB
│   └── run_all.py          # Lancer le pipeline complet en une commande
│
├── nlp/                    # Module NLP
│   ├── classifier.py       # Modèle TF-IDF + LogisticRegression
│   ├── taxonomy.py         # Taxonomie des labels (NATURE_PROBLEME)
│   ├── acronymes.py        # Dictionnaire des acronymes métier
│   └── train.py            # Script d'entraînement avec rapport de qualité
│
├── app/                    # Application Streamlit
│   ├── Home.py             # Page d'accueil
│   ├── pages/
│   │   ├── 1_Exploration.py      # Table filtrable + export CSV
│   │   ├── 2_Statistiques.py     # Graphiques, évolution temporelle, carte
│   │   ├── 3_Pivot.py            # Builder de tableaux croisés
│   │   └── 4_Chat.py             # Interface RAG (phase 4)
│   └── utils/
│       ├── db.py           # Connexion DuckDB (singleton)
│       ├── queries.py      # Requêtes SQL réutilisables
│       └── charts.py       # Fonctions de visualisation Altair/Plotly
│
├── docs/
│   ├── ARCHITECTURE.md     # Ce fichier
│   └── TODO.md             # Plan de travail priorisé
│
├── requirements.txt        # Dépendances Python
└── README.md               # Guide de démarrage rapide
```

---

## Pipeline de données — Vue d'ensemble

```
[Excel bruts]
    |
    v
[01_extract.py]
    - Lit TOUS les fichiers Excel (concatenation, DGESIP, MEDIA2, IEF)
    - Normalise les colonnes (noms, types, dates)
    - Déduplique sur l'id
    - Filtre les lignes sans Analyse
    - Produit : data/processed/saisines_brutes.json
    |
    v
[02_classify.py]
    - Charge les modèles NLP (ou les entraîne si absent)
    - Prédit label, sous_label, lieu, key_word
    - Produit : data/processed/saisines_classifiees.json
    |
    v
[03_to_parquet.py]
    - Convertit en Parquet (table principale + table keywords)
    - Met à jour la base DuckDB
    - Produit : data/processed/saisines.parquet + keywords.parquet
    |
    v
[App Streamlit]
    - Lit DuckDB
    - Affiche les tableaux de bord
```

---

## Module NLP — Fonctionnement

### Principe
Classifieur supervisé entraîné sur les 1 340 saisines déjà labellisées par Gemini.

### Features d'entrée (pour chaque saisine)
```
text = (Analyse * 3) + Catégorie + Sous-catégorie + Domaine + Sous-domaine + Nature
```
Le champ "Analyse" est répété 3 fois pour lui donner plus de poids dans le TF-IDF.

### Architecture du modèle
```
Pipeline sklearn :
    TfidfVectorizer(ngram_range=(1,2), max_features=25000, sublinear_tf=True)
    → LogisticRegression(C=5.0, class_weight="balanced", max_iter=2000)
```

Deux modèles indépendants :
1. `label_classifier` — prédit parmi 13 labels
2. `sous_label_classifier` — prédit parmi ~60 sous_labels
   - Post-traitement : si le sous_label ne correspond pas au label, retourne "autre"

### Performances mesurées (mars 2026)
- Accord label avec Gemini : **98.3%** sur 1 340 saisines
- Accord sous_label avec Gemini : **84.7%** sur 1 340 saisines
- Temps de classification : < 2 minutes pour 6 000 saisines

### Améliorations possibles
- Classification hiérarchique : entraîner un classifieur sous_label PAR label
  (ex: 1 modèle dédié aux sous_labels de "examens", 1 dédié à "harcelement", etc.)
- Augmenter le jeu d'entraînement (actuellement 1 340 exemples)

---

## Application Streamlit — Fonctionnalités cibles

### Page 1 — Exploration
- Table interactive avec toutes les saisines
- Filtres sidebar : académie, label, sous_label, période, lieu
- Recherche plein texte dans le champ Analyse
- Export CSV du résultat filtré
- Clic sur une ligne → fiche détaillée

### Page 2 — Statistiques
- **Onglet Général** : volumes, taux de clôture, durée médiane de traitement
- **Onglet Temporel** : évolution mensuelle/trimestrielle par label (courbe Altair)
- **Onglet Géographique** : carte de France des académies (choroplèthe)
  - Colorée par : volume, type dominant, évolution
- **Onglet Mots-clés** : nuage de mots par label sélectionné (wordcloud ou barres)

### Page 3 — Pivot / Analyse croisée
- Sélection de 2 dimensions à croiser
- Affichage : tableau de contingence + heatmap
- Exemple : label × académie, label × année

### Page 4 — Chat / RAG (phase ultérieure)
- Champ texte libre : "Montre-moi les saisines de harcèlement à Bordeaux en 2023"
- Le LLM génère une requête DuckDB SQL + une synthèse narrative
- Historique de conversation

---

## Stack technique

| Composant | Technologie | Justification |
|-----------|------------|---------------|
| Extraction | pandas, openpyxl | Standard pour Excel |
| NLP | scikit-learn, joblib | Léger, rapide, pas de GPU requis |
| Stockage | Parquet + DuckDB | Ultra rapide pour les analyses, fichiers plats |
| Dashboard | Streamlit | Rapide à développer, Python natif |
| Visualisation | Altair, Plotly | Graphiques interactifs modernes |
| RAG (futur) | ChromaDB + Claude API | Vectorisation + LLM |

---

## Conventions de code

- Tout commentaire et tout nom de variable : **en français**
- Chemins : toujours via `Path(__file__).resolve().parent` (jamais hardcodé)
- Chaque script doit fonctionner en standalone ET être importable
- Logs : utiliser `logging` (pas `print` dans les modules)
- Tests rapides : ajouter un bloc `if __name__ == "__main__"` avec un petit test
- Encodage : toujours `utf-8` explicitement
