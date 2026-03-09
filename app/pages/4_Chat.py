"""
Page 4 — Assistant IA

Deux modes disponibles via les onglets :

  📊 Analyse statistique (text-to-SQL)
     L'utilisateur pose une question chiffrée, le LLM génère le SQL
     correspondant, on l'exécute sur le Parquet, et le LLM rédige
     une synthèse narrative du résultat.

  🔍 Recherche sémantique
     L'utilisateur décrit une situation en français naturel.
     Le modèle d'embeddings retrouve les saisines dont le sens
     est le plus proche, même si les mots exacts ne correspondent pas.

Fournisseur LLM configurable dans app/utils/llm.py :
  LLM_PROVIDER = "ollama"   → modèle local (défaut)
  LLM_PROVIDER = "gemini"   → Google Gemini Flash (gratuit)
  LLM_PROVIDER = "claude"   → Claude Haiku (payant, très peu)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from utils.db import PARQUET, requete, valeurs
from utils.llm import LLM_PROVIDER, appeler_llm, extraire_sql, fournisseur_actuel
from utils.embeddings import embeddings_disponibles, recherche_semantique

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Assistant IA — EDN1 v2",
    page_icon="💬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Système de prompt — description du schéma pour le LLM (mode SQL)
# ---------------------------------------------------------------------------

_labels = ", ".join(f"'{v}'" for v in valeurs("label"))
_poles_top = ", ".join(f"'{v}'" for v in valeurs("pole")[:10])

SYSTEM_SQL = f"""Tu es un expert SQL et DuckDB. Tu analyses les saisines du Médiateur de l'Éducation Nationale française.

TABLE DISPONIBLE : read_parquet('{PARQUET}')

COLONNES :
- id            : identifiant unique de la saisine (texte)
- date_arrivee  : date d'arrivée au médiateur (type date, format YYYY-MM-DD)
- annee         : année extraite de date_arrivee (entier : 2022, 2023, 2024, 2025)
- mois          : mois extrait (entier : 1 à 12)
- pole          : pôle académique (exemples : {_poles_top}, ...)
- label         : catégorie principale de la saisine ({_labels})
- sous_label    : sous-catégorie détaillée (texte)
- "analyse"     : texte complet de la saisine (ATTENTION : guillemets obligatoires car mot réservé SQL)
- key_word      : mots-clés extraits, séparés par des virgules (texte)
- lieu          : lieu de l'incident si détecté, souvent NULL

RÈGLES ABSOLUES — NE JAMAIS LES ENFREINDRE :
1. Utilise TOUJOURS read_parquet('{PARQUET}') comme source de données
2. La colonne "analyse" doit TOUJOURS être entourée de guillemets doubles
3. Retourne UNIQUEMENT la requête SQL brute. AUCUN texte avant. AUCUN texte après. AUCUN markdown. AUCUNE explication.
4. Ta réponse doit commencer directement par SELECT ou WITH. Jamais par un mot français.
5. Ajoute LIMIT 50 sur les requêtes qui retournent des lignes brutes (pas sur les COUNT/SUM/AVG)
6. Pour les comparaisons de texte sur label ou pole, utilise les valeurs exactes (respecte la casse)
7. Pour chercher dans le texte d'analyse, utilise : LOWER("analyse") LIKE '%mot%'

EXEMPLES :
Question : "Combien de saisines par label ?"
SQL : SELECT label, COUNT(*) AS total FROM read_parquet('{PARQUET}') GROUP BY label ORDER BY total DESC

Question : "Evolution des examens par année"
SQL : SELECT annee, COUNT(*) AS total FROM read_parquet('{PARQUET}') WHERE label = 'examens' GROUP BY annee ORDER BY annee

Question : "Saisines de harcèlement à Paris"
SQL : SELECT id, date_arrivee, sous_label, "analyse" FROM read_parquet('{PARQUET}') WHERE label = 'harcelement' AND pole = 'Paris' LIMIT 50"""

SYSTEM_NARRATIF = """Tu es un assistant analytique pour le Médiateur de l'Éducation Nationale.
Tu reçois des résultats d'une analyse statistique des saisines (plaintes) reçues par le médiateur.
Rédige une synthèse narrative en français, concise (3 à 5 phrases), accessible pour un non-technicien.
Mets en valeur les chiffres les plus importants. Ne répète pas la question, va directement à l'essentiel."""


# ---------------------------------------------------------------------------
# Fonctions — mode SQL
# ---------------------------------------------------------------------------


def generer_sql(question: str, erreur_precedente: str = "") -> str:
    """
    Demande au LLM de générer une requête SQL pour la question posée.
    Si erreur_precedente est fourni, le LLM reçoit le SQL raté + l'erreur
    et doit proposer une version corrigée.
    """
    if erreur_precedente:
        contenu = (
            f"Question : {question}\n\n"
            f"Ta requête précédente a produit cette erreur DuckDB :\n{erreur_precedente}\n\n"
            "Corrige la requête SQL. Retourne UNIQUEMENT le SQL corrigé, rien d'autre."
        )
    else:
        contenu = question

    messages = [
        {"role": "system", "content": SYSTEM_SQL},
        {"role": "user", "content": contenu},
    ]
    reponse_brute = appeler_llm(messages, temperature=0.05)
    return extraire_sql(reponse_brute)


def generer_narratif(question: str, sql: str, df_texte: str) -> str:
    """Demande au LLM de rédiger une synthèse narrative des résultats."""
    contenu = (
        f"Question posée : {question}\n\n"
        f"Requête exécutée :\n{sql}\n\n"
        f"Résultats :\n{df_texte}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_NARRATIF},
        {"role": "user", "content": contenu},
    ]
    return appeler_llm(messages, temperature=0.3)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.title("Assistant IA — Questions sur les saisines")
st.caption(f"Fournisseur LLM : **{fournisseur_actuel()}**")

if LLM_PROVIDER == "ollama":
    st.info(
        "Ollama doit être lancé sur votre PC (vérifiez l'icône dans la barre des tâches). "
        "Pour changer de fournisseur LLM, modifiez `LLM_PROVIDER` dans `app/utils/llm.py`.",
        icon="ℹ️",
    )

onglet_sql, onglet_semantique = st.tabs(
    ["📊 Analyse statistique", "🔍 Recherche sémantique"]
)


# ===========================================================================
# ONGLET 1 — Analyse statistique (text-to-SQL)
# ===========================================================================

with onglet_sql:
    st.markdown(
        "Posez une question **chiffrée** sur les saisines : comptages, évolutions, "
        "comparaisons par pôle ou catégorie. Le modèle génère automatiquement "
        "la requête SQL et vous présente une synthèse."
    )

    # Initialisation de l'historique
    if "historique_chat" not in st.session_state:
        st.session_state.historique_chat = []

    # Exemples cliquables
    with st.expander("Exemples de questions", expanded=len(st.session_state.historique_chat) == 0):
        exemples = [
            "Combien de saisines par catégorie ?",
            "Quelle est l'évolution du nombre de saisines par année ?",
            "Quels sont les 5 pôles académiques avec le plus de saisines ?",
            "Combien de saisines de harcèlement par pôle académique ?",
            "Quelle catégorie a le plus progressé entre 2022 et 2025 ?",
            "Combien de saisines concernent les bourses en 2024 ?",
        ]
        cols = st.columns(2)
        for i, exemple in enumerate(exemples):
            if cols[i % 2].button(exemple, use_container_width=True, key=f"ex_sql_{i}"):
                st.session_state.question_choisie = exemple

    if st.session_state.historique_chat:
        if st.button("Nouvelle conversation", type="secondary"):
            st.session_state.historique_chat = []
            st.rerun()

    st.divider()

    # Affichage de l'historique
    for echange in st.session_state.historique_chat:
        with st.chat_message("user"):
            st.markdown(echange["question"])

        with st.chat_message("assistant"):
            if "erreur" in echange:
                st.error(echange["erreur"])
            else:
                with st.expander("Requête SQL générée", expanded=False):
                    st.code(echange["sql"], language="sql")

                if echange["df"] is not None and not echange["df"].empty:
                    st.dataframe(echange["df"], use_container_width=True, hide_index=True)
                else:
                    st.warning("La requête n'a retourné aucun résultat.")

                if echange.get("narratif"):
                    st.markdown(echange["narratif"])

    # Zone de saisie
    question_defaut = st.session_state.pop("question_choisie", "")
    question = st.chat_input("Posez votre question sur les saisines...", key="chat_sql")

    if not question and question_defaut:
        question = question_defaut

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            echange = {"question": question}

            try:
                with st.spinner("Génération de la requête SQL..."):
                    sql = generer_sql(question)

                with st.spinner("Exécution de la requête..."):
                    try:
                        df = requete(sql)
                    except Exception as erreur_sql:
                        with st.spinner("Correction automatique du SQL..."):
                            sql = generer_sql(question, erreur_precedente=str(erreur_sql))
                        df = requete(sql)

                echange["sql"] = sql

                with st.expander("Requête SQL générée", expanded=True):
                    st.code(sql, language="sql")

                echange["df"] = df

                if df.empty:
                    st.warning("La requête n'a retourné aucun résultat.")
                    echange["narratif"] = None
                else:
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    with st.spinner("Rédaction de la synthèse..."):
                        df_texte = df.head(20).to_string(index=False)
                        if len(df) > 20:
                            df_texte += f"\n... ({len(df)} lignes au total, 20 affichées)"
                        narratif = generer_narratif(question, sql, df_texte)

                    echange["narratif"] = narratif
                    st.markdown(narratif)

            except ConnectionError as e:
                msg = str(e)
                st.error(msg)
                echange["erreur"] = msg
            except Exception as e:
                msg = (
                    f"Erreur lors du traitement : {e}\n\n"
                    "Essayez de reformuler votre question, ou vérifiez que le LLM est opérationnel."
                )
                st.error(msg)
                echange["erreur"] = msg

        st.session_state.historique_chat.append(echange)


# ===========================================================================
# ONGLET 2 — Recherche sémantique
# ===========================================================================

with onglet_semantique:
    st.markdown(
        "Décrivez une **situation** ou une **thématique** en français naturel. "
        "Le moteur retrouve les saisines dont le sens est le plus proche, "
        "même si les mots exacts ne correspondent pas."
    )
    st.caption(
        "Exemple : *« problème de bourse non versée »*, "
        "*« refus d'inscription en master »*, "
        "*« harcèlement dans un lycée »*"
    )

    if not embeddings_disponibles():
        st.warning(
            "Les embeddings sémantiques n'ont pas encore été générés. "
            "Lancez la commande suivante dans votre terminal, puis rechargez la page :",
            icon="⚠️",
        )
        st.code("python pipeline/04_embeddings.py", language="bash")
    else:
        # Initialisation de la session state pour persister les résultats
        if "sem_resultats" not in st.session_state:
            st.session_state["sem_resultats"] = None
        if "sem_query_affichee" not in st.session_state:
            st.session_state["sem_query_affichee"] = ""

        # Exemples cliquables — définis AVANT le text_input pour pouvoir
        # écrire dans son session_state sans déclencher d'erreur Streamlit
        exemples_sem = [
            "problème de bourse non versée",
            "refus d'inscription en master",
            "harcèlement entre élèves",
            "contestation d'une note d'examen",
            "accident scolaire",
        ]
        cols_sem = st.columns(len(exemples_sem))
        for i, ex in enumerate(exemples_sem):
            if cols_sem[i].button(ex, use_container_width=True, key=f"ex_sem_{i}"):
                # On écrit directement dans la clé du text_input avant qu'il
                # soit rendu — pas de st.rerun() pour éviter le changement d'onglet
                st.session_state["sem_input"] = ex
                st.session_state["sem_resultats"] = None  # force relance

        # Contrôles
        col_input, col_k = st.columns([4, 1])
        with col_input:
            query_sem = st.text_input(
                "Décrivez la situation recherchée :",
                placeholder="ex : problème de bourse non versée",
                key="sem_input",
            )
        with col_k:
            k = st.number_input("Résultats", min_value=3, max_value=50, value=10, step=1)

        # Lancer la recherche si la question a changé
        if query_sem and (
            query_sem != st.session_state["sem_query_affichee"]
            or st.session_state["sem_resultats"] is None
        ):
            try:
                with st.spinner("Recherche des saisines similaires..."):
                    df_all = requete(
                        f'SELECT id, date_arrivee, annee, pole, label, sous_label, "analyse" '
                        f"FROM read_parquet('{PARQUET}')"
                    )
                    df_res = recherche_semantique(query_sem, df_all, k=int(k))
                st.session_state["sem_resultats"] = df_res
                st.session_state["sem_query_affichee"] = query_sem
            except FileNotFoundError as e:
                st.error(str(e))
                st.session_state["sem_resultats"] = None
            except RuntimeError as e:
                st.error(
                    f"sentence-transformers n'est pas installé. "
                    f"Lancez : pip install sentence-transformers\n\nDétail : {e}"
                )
                st.session_state["sem_resultats"] = None
            except Exception as e:
                st.error(f"Erreur lors de la recherche : {e}")
                st.session_state["sem_resultats"] = None

        # Affichage des résultats stockés en session_state
        if st.session_state["sem_resultats"] is not None:
            df_res = st.session_state["sem_resultats"]
            st.success(f"{len(df_res)} saisines trouvées")

            df_affichage = df_res.copy()
            df_affichage["similarité"] = df_affichage["similarité"].apply(
                lambda x: f"{x:.0%}"
            )
            colonnes = ["similarité", "id", "date_arrivee", "pole", "label", "sous_label", "analyse"]
            colonnes_presentes = [c for c in colonnes if c in df_affichage.columns]
            st.dataframe(
                df_affichage[colonnes_presentes],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "analyse": st.column_config.TextColumn("analyse", width="large"),
                    "similarité": st.column_config.TextColumn("Score", width="small"),
                },
            )
