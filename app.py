
from pathlib import Path
import re

import streamlit as st

from graph_hybrid import graph
from rag_prepare import DOCUMENTS_DIR
from vector_store import (
    index_file,
    get_indexed_sources
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ResearchFlow AI",
    page_icon="🔬",
    layout="wide"
)

MAX_UPLOAD_MB = 20

st.title("🔬 ResearchFlow AI")

st.caption(
    "Multi-Agent RAG Research System | "
    "LangGraph · Groq · ChromaDB · Tavily"
)


# ==========================================
# 2. STATE INITIAL LANGGRAPH
# ==========================================

def create_initial_state(
    question: str,
    mode: str
) -> dict:

    return {
        "question": question,
        "requested_mode": mode,
        "mode": "",
        "routing_note": "",

        "document_matches": [],
        "web_sources": [],

        "draft_answer": "",
        "final_answer": "",

        "evaluation_status": "not_run",
        "evaluation_valid": None,
        "evaluation_score": None,
        "evaluation_issues": [],

        "critic_llm_valid": None,
        "critic_llm_issues": [],

        "numeric_checked_claims": 0,
        "numeric_issues": [],

        "revision_count": 0
    }


# ==========================================
# 3. VÉRIFIER LES CITATIONS
# ==========================================

def check_citations(
    answer: str,
    source_count: int
) -> list[str]:

    citations = re.findall(
        r"\[\s*([A-Za-z]\d+)\s*\]",
        answer
    )

    allowed = {
        f"S{i}"
        for i in range(1, source_count + 1)
    }

    return sorted({
        citation
        for citation in citations
        if citation not in allowed
    })


# ==========================================
# 4. IMPORTATION DES DOCUMENTS
# ==========================================

with st.sidebar:

    st.header("📁 Documents")

    st.write(
        "Importe un PDF ou un TXT pour "
        "l'ajouter à ta base documentaire."
    )

    uploaded_file = st.file_uploader(
        "Choisir un document",
        type=["pdf", "txt"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:

        st.caption(
            f"Fichier : {uploaded_file.name}"
        )

        st.caption(
            f"Taille : "
            f"{uploaded_file.size / 1024:.1f} Ko"
        )

    if st.button(
        "Indexer le document",
        disabled=uploaded_file is None,
        use_container_width=True
    ):

        file_name = Path(
            uploaded_file.name
        ).name

        target = DOCUMENTS_DIR / file_name

        if target.exists():

            st.warning(
                "Un fichier portant ce nom existe "
                "déjà dans documents/. "
                "Renomme ton nouveau fichier "
                "pour éviter tout écrasement."
            )

        elif (
            uploaded_file.size
            > MAX_UPLOAD_MB * 1024 * 1024
        ):

            st.error(
                f"Taille maximale : "
                f"{MAX_UPLOAD_MB} Mo."
            )

        elif target.suffix.lower() not in [
            ".pdf",
            ".txt"
        ]:

            st.error(
                "Seuls PDF et TXT sont acceptés."
            )

        else:

            DOCUMENTS_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            try:

                # Enregistrer le document
                target.write_bytes(
                    uploaded_file.getvalue()
                )

                with st.spinner(
                    "Extraction, embeddings "
                    "et indexation..."
                ):

                    chunk_count = index_file(
                        target
                    )

                st.success(
                    f"{file_name} indexé : "
                    f"{chunk_count} chunks."
                )

                # Ne pas afficher un résultat
                # ancien comme s'il concernait
                # le nouveau document.
                st.session_state.pop(
                    "result",
                    None
                )

            except Exception as error:

                # Nettoyer le fichier nouvellement
                # enregistré si l'import échoue.
                if target.exists():
                    target.unlink()

                st.error(
                    "L'indexation a échoué."
                )

                st.caption(str(error))

    st.divider()

    st.subheader("Documents indexés")

    try:

        indexed_sources = get_indexed_sources()

        if indexed_sources:

            for source in indexed_sources:
                st.write("📄", source)

            st.caption(
                f"{len(indexed_sources)} "
                "document(s) dans ChromaDB."
            )

        else:

            st.info(
                "Aucun document indexé."
            )

    except Exception as error:

        st.warning(
            "Impossible de lire la liste "
            "des documents indexés."
        )

        st.caption(str(error))


# ==========================================
# 5. FORMULAIRE DE RECHERCHE
# ==========================================

st.subheader("Nouvelle recherche")

with st.form("research_form"):

    question = st.text_area(
        "Ta question",
        placeholder=(
            "Résume les principaux résultats "
            "de mon document."
        ),
        height=110
    )

    mode = st.selectbox(
        "Mode de recherche",
        options=[
            "auto",
            "document",
            "web",
            "hybride"
        ],
        index=0
    )

    submitted = st.form_submit_button(
        "Lancer la recherche",
        type="primary"
    )


# ==========================================
# 6. EXÉCUTER LANGGRAPH
# ==========================================

if submitted:

    if not question.strip():

        st.warning(
            "Écris une question avant de lancer."
        )

    else:

        initial_state = create_initial_state(
            question.strip(),
            mode
        )

        with st.spinner(
            "Recherche, rédaction, "
            "évaluation et correction..."
        ):

            try:

                result = graph.invoke(
                    initial_state,
                    config={"recursion_limit": 20}
                )

                st.session_state[
                    "result"
                ] = result

            except Exception as error:

                st.error(
                    "Le workflow a rencontré une erreur."
                )

                st.code(str(error))

                st.session_state.pop(
                    "result",
                    None
                )


# ==========================================
# 7. AFFICHAGE DES RÉSULTATS
# ==========================================

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    st.subheader("Résultats du workflow")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Mode",
        result["mode"]
    )

    col2.metric(
        "Passages PDF",
        len(result["document_matches"])
    )

    col3.metric(
        "Sources Web",
        len(result["web_sources"])
    )

    col4.metric(
        "Corrections",
        result["revision_count"]
    )

    # --------------------------------------
    # A. RÉPONSE
    # --------------------------------------

    st.subheader("Réponse générée")

    st.markdown(
        result["final_answer"]
    )

    source_count = (
        len(result["document_matches"])
        + len(result["web_sources"])
    )

    invalid_citations = check_citations(
        result["draft_answer"],
        source_count
    )

    if invalid_citations:

        st.warning(
            "Citations non reconnues : "
            + ", ".join(invalid_citations)
        )

    # --------------------------------------
    # B. CRITIC
    # --------------------------------------

    st.subheader("Évaluation de la réponse")

    evaluation_status = result[
        "evaluation_status"
    ]

    evaluation_valid = result[
        "evaluation_valid"
    ]

    score = result[
        "evaluation_score"
    ]

    if evaluation_status == "not_run":

        st.info(
            "Critic non exécuté."
        )

    elif evaluation_status != "completed":

        st.error(
            "L'évaluation a échoué."
        )

    elif evaluation_valid is False:

        st.error(
            "La réponse n'a pas été validée "
            "par le Critic."
        )

    elif invalid_citations:

        st.warning(
            "Le Critic a accepté la réponse, "
            "mais certaines citations "
            "ne sont pas valides."
        )

    else:

        st.success(
            "Réponse acceptée par le Critic. "
            "Cette validation reste indicative."
        )

    if score is not None:

        st.metric(
            "Score du Critic LLM",
            f"{score:.2f} / 1"
        )

    if result["evaluation_issues"]:

        st.write("**Problèmes détectés :**")

        for issue in result[
            "evaluation_issues"
        ]:

            st.write("- " + issue)

    if result["critic_llm_issues"]:

        with st.expander(
            "Observations du Critic"
        ):

            for issue in result[
                "critic_llm_issues"
            ]:

                st.write("- " + issue)

    st.write(
        "**Contrôles numériques effectués :**",
        result["numeric_checked_claims"]
    )

    if result["numeric_checked_claims"] == 0:

        st.info(
            "Le Numeric Guard n'a vérifié "
            "aucun chiffre dans cette réponse."
        )

    # --------------------------------------
    # C. SOURCES RETROUVÉES
    # --------------------------------------

    with st.expander(
        "Voir les sources récupérées"
    ):

        index = 1

        for match in result[
            "document_matches"
        ]:

            st.markdown(
                f"**[S{index}] PDF/TXT — "
                f"{match['source']}, "
                f"page {match['page']}**"
            )

            st.write(
                match["content"]
            )

            index += 1

        for source in result[
            "web_sources"
        ]:

            st.markdown(
                f"**[S{index}] Web — "
                f"{source['title']}**"
            )

            st.write(source["url"])
            st.write(source["snippet"])

            index += 1

    # --------------------------------------
    # D. EXPORT MARKDOWN
    # --------------------------------------

    export_text = (
        f"# ResearchFlow AI\n\n"
        f"Question : {result['question']}\n\n"
        f"Mode : {result['mode']}\n\n"
        f"## Réponse\n\n"
        f"{result['final_answer']}\n\n"
        f"## Évaluation\n\n"
        f"Statut : {evaluation_status}\n\n"
        f"Valide : {evaluation_valid}\n\n"
        f"Score LLM : {score}\n\n"
        f"Corrections : {result['revision_count']}\n\n"
        f"Citations invalides : "
        f"{invalid_citations}\n\n"
        f"Problèmes : "
        f"{result['evaluation_issues']}\n"
    )

    st.download_button(
        "Télécharger le rapport Markdown",
        data=export_text,
        file_name="researchflow_result.md",
        mime="text/markdown"
    )
