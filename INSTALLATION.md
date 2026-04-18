# Installation — EDN1 v2

## Configuration minimale requise

| Composant | Minimum | Recommandé |
|-----------|---------|------------|
| OS | Windows 10 | Windows 10/11 |
| RAM | 8 Go | 16 Go |
| Stockage | 8 Go libres | 10 Go |
| Python | 3.10+ | 3.11/3.12 |

## Étapes d'installation (à faire une seule fois)

### 1. Python
Télécharger et installer Python 3.11+ : https://www.python.org/downloads/
> Cocher "Add Python to PATH" lors de l'installation

### 2. Ollama (pour l'assistant IA)
Télécharger et installer Ollama : https://ollama.com/download
Puis ouvrir un terminal et taper :
```
ollama pull mistral
```
> Téléchargement d'environ 4 Go — à faire une seule fois

### 3. Données
Copier le fichier `concatenation.xlsx` dans le dossier `data/input/`
Copier le fichier `gemini_labels.json` dans le dossier `data/input/`

## Lancement

Double-cliquer sur **`LANCER.bat`**

Le navigateur s'ouvre automatiquement sur le dashboard.
La première utilisation prend ~2 minutes (préparation des données).
Les lancements suivants sont immédiats.
