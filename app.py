
from pathlib import Path
import re

import streamlit as st

from graph_hybrid import graph
from rag_prepare import DOCUMENTS_DIR
from vector_store import index_file, get_indexed_sources


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ResearchFlow AI",
    page_icon="🔬",
    layout="wide",
)

MAX_UPLOAD_MB = 20
ALL_DOCUMENTS = "Tous les documents"

st.title("🔬 ResearchFlow AI")
st.caption(
    "Multi-Agent RAG Research System | "
    "LangGraph · Groq · ChromaDB · Tavily"
)


# ==========================================
# 2. ETAT INITIAL LANGGRAPH
# ==========================================

def create_initial_state(
    question: str,
    mode: str,
    selected_source: str | None,
) -> dict:
    return {
        "question": question,
        "requested_mode": mode,
        "selected_source": selected_source,
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
        "revision_count": 0,
    }


# ==========================================
# 3. CITATIONS
# ==========================================

def check_citations(answer: str, source_count: int) -> list[str]:
    citations = re.findall(r"\[\s*([A-Za-z]\d+)\s*\]", answer)
    allowed = {f"S{i}" for i in range(1, source_count + 1)}
    return sorted({citation for citation in citations if citation not in allowed})


# ==========================================
# 4. DOCUMENTS INDEXES
# ==========================================

try:
    indexed_sources = get_indexed_sources()
except Exception as error:
    indexed_sources = []
    st.error(f"Impossible de lire ChromaDB : {error}")


# ==========================================
# 5. BARRE LATERALE : IMPORT DE FICHIERS
# ==========================================

with st.sidebar:
    st.header("📁 Documents")
    st.write("Importer un PDF ou TXT dans ChromaDB.")

    uploaded_file = st.file_uploader(
        "Choisir un document",
        type=["pdf", "txt"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        st.caption(f"Fichier : {uploaded_file.name}")
        st.caption(f"Taille : {uploaded_file.size / 1024:.1f} Ko")

    if st.button(
        "Indexer le document",
        disabled=uploaded_file is None,
        use_container_width=True,
    ):
        file_name = Path(uploaded_file.name).name
        target = DOCUMENTS_DIR / file_name

        if target.exists():
            st.warning(
                "Un fichier de ce nom existe deja dans documents/. "
                "Renomme ton nouveau fichier pour eviter l'ecrasement."
            )
        elif uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"Taille maximale : {MAX_UPLOAD_MB} Mo.")
        elif target.suffix.lower() not in [".pdf", ".txt"]:
            st.error("Seuls PDF et TXT sont acceptes.")
        else:
            DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            try:
                target.write_bytes(uploaded_file.getvalue())
                with st.spinner("Extraction et indexation..."):
                    chunk_count = index_file(target)
                st.success(f"{file_name} indexe : {chunk_count} chunks.")
                st.session_state.pop("result", None)
                # Actualiser la liste immediatement apres l'import.
                indexed_sources = get_indexed_sources()
            except Exception as error:
                if target.exists():
                    target.unlink()
                st.error("L'indexation a echoue.")
                st.caption(str(error))

    st.divider()
    st.subheader("Documents indexes")

    if indexed_sources:
        for source in indexed_sources:
            st.write("📄", source)
        st.caption(f"{len(indexed_sources)} document(s) dans ChromaDB")
    else:
        st.info("Aucun document indexe.")


# ==========================================
# 6. FORMULAIRE : SELECTION DU DOCUMENT
# ==========================================

st.subheader("Nouvelle recherche")

with st.form("research_form"):
    question = st.text_area(
        "Ta question",
        placeholder="Resume les resultats de mon document.",
        height=110,
    )

    # NOUVEAU : choisir un seul document, ou toute la collection.
    selected_label = st.selectbox(
        "Document a interroger",
        options=[ALL_DOCUMENTS] + indexed_sources,
        index=0,
        help=(
            "La recherche documentaire sera limitee au PDF/TXT choisi. "
            "Tous les documents conserve le fonctionnement precedent."
        ),
    )

    mode = st.selectbox(
        "Mode de recherche",
        options=["auto", "document", "web", "hybride"],
        index=0,
    )

    submitted = st.form_submit_button(
        "Lancer la recherche",
        type="primary",
    )


# ==========================================
# 7. EXECUTION LANGGRAPH
# ==========================================

if submitted:
    if not question.strip():
        st.warning("Ecris une question avant de lancer.")
    else:
        selected_source = (
            None if selected_label == ALL_DOCUMENTS else selected_label
        )

        initial_state = create_initial_state(
            question=question.strip(),
            mode=mode,
            selected_source=selected_source,
        )

        with st.spinner("Recherche, redaction et evaluation..."):
            try:
                result = graph.invoke(
                    initial_state,
                    config={"recursion_limit": 20},
                )
                st.session_state["result"] = result
            except Exception as error:
                st.error("Le workflow a rencontre une erreur.")
                st.code(str(error))
                st.session_state.pop("result", None)


# ==========================================
# 8. RESULTATS
# ==========================================

if "result" in st.session_state:
    result = st.session_state["result"]
    st.divider()
    st.subheader("Resultats du workflow")

    st.caption(
        "Document selectionne : "
        + (result.get("selected_source") or ALL_DOCUMENTS)
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mode", result["mode"])
    col2.metric("Passages documentaires", len(result["document_matches"]))
    col3.metric("Sources Web", len(result["web_sources"]))
    col4.metric("Corrections", result["revision_count"])

    st.subheader("Reponse generee")
    st.markdown(result["final_answer"])

    source_count = (
        len(result["document_matches"])
        + len(result["web_sources"])
    )
    invalid_citations = check_citations(result["draft_answer"], source_count)

    if invalid_citations:
        st.warning(
            "Citations non reconnues : " + ", ".join(invalid_citations)
        )

    st.subheader("Rapport du Critic")
    status = result["evaluation_status"]
    valid = result["evaluation_valid"]
    score = result["evaluation_score"]

    if status == "not_run":
        st.info("Critic non execute : pas de reponse documentaire a evaluer.")
    elif status != "completed":
        st.error("L'evaluation n'a pas pu etre terminee.")
    elif valid is False:
        st.error("La reponse n'a pas ete validee par le Critic.")
    elif invalid_citations:
        st.warning(
            "Le Critic a accepte la reponse, "
            "mais certaines citations sont invalides."
        )
    else:
        st.success(
            "Reponse acceptee par le Critic. "
            "Cette validation reste indicative."
        )

    if score is not None:
        st.metric("Score du Critic LLM", f"{score:.2f} / 1")

    if result["evaluation_issues"]:
        st.write("**Problemes signales :**")
        for issue in result["evaluation_issues"]:
            st.write("- " + issue)

    if result["critic_llm_issues"]:
        with st.expander("Observations du Critic"):
            for issue in result["critic_llm_issues"]:
                st.write("- " + issue)

    st.write(
        "**Affirmations numeriques controlees :**",
        result["numeric_checked_claims"],
    )
    if result["numeric_checked_claims"] == 0:
        st.info(
            "Le Numeric Guard n'a verifie aucun chiffre de cette reponse. "
            "Cela ne prouve pas que tous les chiffres sont exacts."
        )

    with st.expander("Voir les sources recuperees"):
        index = 1
        for match in result["document_matches"]:
            st.markdown(
                f"**[S{index}] PDF/TXT — {match['source']}, "
                f"page {match['page']}**"
            )
            st.write(match["content"])
            index += 1

        for source in result["web_sources"]:
            st.markdown(f"**[S{index}] Web — {source['title']}**")
            st.write(source["url"])
            st.write(source["snippet"])
            index += 1

    export_text = (
        "# ResearchFlow AI\n\n"
        f"Question : {result['question']}\n\n"
        f"Mode : {result['mode']}\n\n"
        f"Document : {result.get('selected_source') or ALL_DOCUMENTS}\n\n"
        "## Reponse\n\n"
        f"{result['final_answer']}\n\n"
        "## Evaluation\n\n"
        f"Statut : {status}\n\n"
        f"Valide : {valid}\n\n"
        f"Score LLM : {score}\n\n"
        f"Corrections : {result['revision_count']}\n\n"
        f"Citations invalides : {invalid_citations}\n\n"
        f"Problemes : {result['evaluation_issues']}\n"
    )
    st.download_button(
        "Telecharger le rapport Markdown",
        data=export_text,
        file_name="researchflow_result.md",
        mime="text/markdown",
    )
