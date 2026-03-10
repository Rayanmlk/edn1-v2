# EDN1 v2 — Analyse des saisines du Médiateur de l'Éducation Nationale

Plateforme d'analyse et de classification automatique des saisines reçues par le Médiateur de l'Éducation Nationale.

**6 012 saisines** classifiées automatiquement en **13 catégories** via un modèle NLP local — sans appel API à chaque utilisation.

---

## Ce que ça fait

- **Classification NLP** : un modèle TF-IDF + Régression Logistique, entraîné sur des données pré-labellisées, classifie chaque saisine en catégorie principale et sous-catégorie
- **Dashboard interactif** : 5 pages Streamlit pour explorer, filtrer, croiser et visualiser les données
- **Assistant IA** : questions chiffrées en langage naturel (text-to-SQL)
- **Stockage optimisé** : format Parquet via DuckDB (0.6 Mo au lieu de ~15 Mo JSON)

## Ce qui est mieux par rapport à la version précédente

La v1 faisait appel à l'API Gemini pour classer chaque saisine à la volée — ce qui coûtait des centaines d'appels API à chaque utilisation. Ici, le modèle NLP tourne entièrement en local après une phase d'entraînement unique. Résultat : **86% de précision sur la catégorie principale** (évaluation sur données jamais vues), zéro appel API pour la classification.

---

## Structure

```
EDN1_v2/
├── pipeline/
│   ├── 01_extract.py        # Lecture Excel, nettoyage, normalisation
│   ├── 02_classify.py       # Classification NLP (entraînement + inférence)
│   ├── 03_to_parquet.py     # Conversion JSON → Parquet
│   ├── eval.py              # Évaluation du modèle (score honnête / optimiste)
│   
├── nlp/
│   ├── classifier.py        # Modèle TF-IDF + Logistic Regression
│   ├── taxonomy.py          # 13 labels et leurs sous-catégories
│   └── acronymes.py         # Table de correspondance codes académiques
├── app/
│   ├── Home.py              # KPIs globaux + répartition
│   ├── pages/
│   │   ├── 1_Exploration.py # Table filtrable + export CSV
│   │   ├── 2_Statistiques.py # Distribution, évolution temporelle, par pôle
│   │   ├── 3_Pivot.py       # Tableau croisé + heatmap
│   │   └── 4_Chat.py        # Assistant IA (text-to-SQL + recherche sémantique)
│   └── utils/
│       ├── db.py            # Connexion DuckDB
│       ├── llm.py           # Abstraction LLM (Ollama / Gemini / Claude)
│       
├── data/
│   ├── input/               # Fichiers source — non versionnés
│   └── processed/           # Fichiers intermédiaires — non versionnés
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Pipeline — ordre d'exécution

```bash
# 1. Extraction et nettoyage
python pipeline/01_extract.py

# 2. Classification NLP (entraîne le modèle si absent, puis classifie)
python pipeline/02_classify.py

# 3. Conversion en Parquet
python pipeline/03_to_parquet.py

# 4. Embeddings sémantiques — optionnel, pour la recherche sémantique dans le Chat
pip install sentence-transformers
python pipeline/04_embeddings.py
```

---

## Lancer le dashboard

```bash
streamlit run app/Home.py
```

---

## Assistant IA — deux modes

### Analyse statistique (text-to-SQL)
Questions chiffrées : *"Combien de saisines de harcèlement par pôle ?"*, *"Quelle catégorie a le plus progressé ?"*

Le LLM génère automatiquement la requête SQL DuckDB, l'exécute, et rédige une synthèse narrative. Fonctionne avec **Ollama (local, gratuit)**, Gemini Flash ou Claude Haiku — configurable en une ligne dans `app/utils/llm.py`.

Pour Ollama :
```bash
# Installer depuis ollama.com, puis :
ollama pull mistral
```

### Recherche sémantique
Descriptions conceptuelles : *"problème de bourse non versée"*, *"refus d'inscription en master"*

Le modèle `paraphrase-multilingual-MiniLM-L12-v2` encode la requête et retrouve les saisines les plus proches par similarité cosinus — sans se limiter aux mots exacts.

Nécessite d'avoir exécuté `pipeline/04_embeddings.py`.

---

## Qualité de la classification

| Mesure | Catégorie principale (13) | Sous-catégorie (~60) |
|--------|--------------------------|----------------------|
| Score sur données d'entraînement | 98.1% | 85.0% |
| **Score sur données jamais vues** | **86.3%** | **49.6%** |

Le score honnête (80/20) est le seul représentatif. Le sous-label plus faible s'explique par le déséquilibre des données : certaines sous-catégories n'ont qu'un ou deux exemples d'entraînement.

```bash
# Évaluation honnête (recommandé)
python pipeline/eval.py --honnete
```

---

## Fournisseurs LLM

Modifier `LLM_PROVIDER` dans `app/utils/llm.py` :

```python
LLM_PROVIDER = "ollama"   # local, gratuit (défaut)
LLM_PROVIDER = "gemini"   # Google Gemini Flash, gratuit
LLM_PROVIDER = "claude"   # Anthropic Claude Haiku, payant
```
