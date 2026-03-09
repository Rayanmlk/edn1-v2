"""
Module LLM — Abstraction pour l'appel au modèle de langage.

Pour changer de fournisseur, modifier LLM_PROVIDER (une seule ligne) :
  "ollama"  → modèle local, gratuit, tourne sur ton PC (GPU ou CPU)
  "gemini"  → API Google Gemini Flash, gratuit jusqu'à 15 req/min
  "claude"  → API Anthropic Claude Haiku, payant (~0.001€ par question)

Le reste du code n'a PAS besoin d'être modifié quand on change de fournisseur.
"""

import os
import re

# ---------------------------------------------------------------------------
# Configuration — modifier LLM_PROVIDER pour changer de fournisseur
# ---------------------------------------------------------------------------

LLM_PROVIDER = "ollama"     # "ollama" | "gemini" | "claude"
OLLAMA_MODEL = "mistral"    # modèle Ollama à utiliser

# Clés API (lire depuis les variables d'environnement)
# Pour Gemini  : set GEMINI_API_KEY=ta_clé  (dans le terminal avant de lancer streamlit)
# Pour Claude  : set ANTHROPIC_API_KEY=ta_clé
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ---------------------------------------------------------------------------
# Fonction publique — seule interface que le reste du code utilise
# ---------------------------------------------------------------------------


def appeler_llm(messages: list[dict], temperature: float = 0.1) -> str:
    """
    Envoie une liste de messages au LLM configuré et retourne la réponse texte.

    Format des messages (standard OpenAI, compatible avec tous les fournisseurs) :
        [
            {"role": "system", "content": "Tu es un assistant SQL..."},
            {"role": "user",   "content": "Combien de saisines en 2024 ?"},
        ]

    temperature=0.1 par défaut : réponses déterministes, importantes pour
    la génération SQL (on ne veut pas de créativité, on veut de la précision).
    """
    if LLM_PROVIDER == "ollama":
        return _ollama(messages, temperature)
    elif LLM_PROVIDER == "gemini":
        return _gemini(messages, temperature)
    elif LLM_PROVIDER == "claude":
        return _claude(messages, temperature)
    else:
        raise ValueError(
            f"LLM_PROVIDER '{LLM_PROVIDER}' inconnu. "
            "Valeurs acceptées : 'ollama', 'gemini', 'claude'."
        )


def fournisseur_actuel() -> str:
    """Retourne une description lisible du fournisseur configuré."""
    if LLM_PROVIDER == "ollama":
        return f"Ollama local ({OLLAMA_MODEL})"
    elif LLM_PROVIDER == "gemini":
        return "Google Gemini Flash (API gratuite)"
    elif LLM_PROVIDER == "claude":
        return "Anthropic Claude Haiku (API)"
    return LLM_PROVIDER


# ---------------------------------------------------------------------------
# Implémentations privées — une par fournisseur
# ---------------------------------------------------------------------------


def _ollama(messages: list[dict], temperature: float) -> str:
    """
    Appelle le modèle local via Ollama.
    Ollama doit être installé et lancé (service Windows, port 11434).
    """
    try:
        import ollama  # importé ici pour ne pas bloquer si non installé
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": temperature},
        )
        return response["message"]["content"].strip()
    except ImportError:
        raise RuntimeError(
            "La bibliothèque 'ollama' n'est pas installée. "
            "Lancez : pip install ollama"
        )
    except Exception as e:
        raise ConnectionError(
            f"Impossible de contacter Ollama (http://localhost:11434). "
            f"Vérifiez qu'Ollama est lancé (icône dans la barre des tâches). "
            f"Erreur technique : {e}"
        )


def _gemini(messages: list[dict], temperature: float) -> str:
    """
    Appelle Google Gemini Flash via l'API gratuite.
    Nécessite : pip install google-generativeai
    Clé API gratuite sur : aistudio.google.com
    """
    try:
        import google.generativeai as genai  # noqa
    except ImportError:
        raise RuntimeError(
            "La bibliothèque 'google-generativeai' n'est pas installée. "
            "Lancez : pip install google-generativeai"
        )

    if not GEMINI_API_KEY:
        raise ValueError(
            "Clé API Gemini manquante. "
            "Obtenez-en une gratuitement sur aistudio.google.com, "
            "puis : set GEMINI_API_KEY=votre_clé"
        )

    genai.configure(api_key=GEMINI_API_KEY)

    # Séparer le system prompt des autres messages
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    history_msgs = [m for m in messages if m["role"] != "system"]

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction=system or None,
        generation_config={"temperature": temperature},
    )

    # Convertir l'historique au format Gemini
    gemini_history = []
    for m in history_msgs[:-1]:
        role = "user" if m["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [m["content"]]})

    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(history_msgs[-1]["content"])
    return response.text.strip()


def _claude(messages: list[dict], temperature: float) -> str:
    """
    Appelle Claude Haiku via l'API Anthropic.
    Nécessite : pip install anthropic
    ~0.001€ par question.
    """
    try:
        import anthropic  # noqa
    except ImportError:
        raise RuntimeError(
            "La bibliothèque 'anthropic' n'est pas installée. "
            "Lancez : pip install anthropic"
        )

    if not CLAUDE_API_KEY:
        raise ValueError(
            "Clé API Anthropic manquante. "
            "Définissez : set ANTHROPIC_API_KEY=votre_clé"
        )

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_messages = [m for m in messages if m["role"] != "system"]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=user_messages,
        temperature=temperature,
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Utilitaire
# ---------------------------------------------------------------------------


def extraire_sql(texte: str) -> str:
    """
    Extrait la requête SQL d'une réponse LLM, même si le modèle a ajouté
    du texte explicatif avant ou après le SQL (comportement fréquent avec Mistral).

    Stratégies dans l'ordre de priorité :
    1. SQL dans un bloc markdown ```sql ... ``` ou ``` ... ```
    2. Cherche le premier mot-clé SQL (WITH, SELECT, INSERT...) dans le texte
    3. Retourne le texte brut en dernier recours
    """
    # Stratégie 1 : bloc markdown ``` ... ```
    match_bloc = re.search(r"```(?:sql)?\s*(.*?)\s*```", texte, re.DOTALL | re.IGNORECASE)
    if match_bloc:
        sql = match_bloc.group(1).strip()
        return sql.split(";")[0].strip() if ";" in sql else sql

    # Stratégie 2 : trouver le premier mot-clé SQL dans le texte
    # (Mistral écrit souvent "Pour répondre... voici la requête :\nSELECT ...")
    match_keyword = re.search(
        r"\b(WITH|SELECT|INSERT|UPDATE|DELETE|CREATE)\b",
        texte,
        re.IGNORECASE,
    )
    if match_keyword:
        sql = texte[match_keyword.start():].strip()
        return sql.split(";")[0].strip() if ";" in sql else sql

    # Stratégie 3 : fallback — retourner le texte nettoyé
    texte = texte.replace("```", "").strip()
    return texte.split(";")[0].strip() if ";" in texte else texte
