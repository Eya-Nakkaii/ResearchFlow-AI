
from pathlib import Path
from hashlib import sha256
import sys

import chromadb

from rag_prepare import (
    load_documents,
    load_document,
    split_into_chunks
)

from embeddings import load_embedding_model


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "researchflow_documents"

TOP_K = 5
BATCH_SIZE = 100


# ==========================================
# 2. CONNEXION CHROMADB
# ==========================================

def get_client():

    return chromadb.PersistentClient(
        path=str(DB_PATH)
    )


def get_collection():

    client = get_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None
    )


# ==========================================
# 3. PRÉPARER LES DONNÉES
# ==========================================

def prepare_chunks(chunks: list[dict]):

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_number": chunk["chunk_number"]
        }
        for chunk in chunks
    ]

    model = load_embedding_model()

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return texts, metadatas, vectors


# ==========================================
# 4. AJOUTER LES CHUNKS À UNE COLLECTION
# ==========================================

def add_chunks(
    collection,
    ids,
    texts,
    metadatas,
    vectors
):

    for start in range(0, len(texts), BATCH_SIZE):

        end = start + BATCH_SIZE

        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=vectors[start:end].tolist()
        )


# ==========================================
# 5. INDEXER UN NOUVEAU FICHIER
# ==========================================

def index_file(file_path: Path) -> int:

    file_path = Path(file_path)

    # Lire uniquement ce document
    documents = load_document(file_path)

    chunks = split_into_chunks(documents)

    if not chunks:

        raise ValueError(
            "Aucun texte exploitable dans ce document. "
            "Un PDF scanné peut nécessiter un OCR."
        )

    print(
        f"\nIndexation de {file_path.name} : "
        f"{len(chunks)} chunks"
    )

    # Calculer tous les embeddings AVANT
    # de modifier la collection ChromaDB
    texts, metadatas, vectors = (
        prepare_chunks(chunks)
    )

    collection = get_collection()

    # Créer des identifiants spécifiques
    # au fichier, sans collision avec
    # les anciens IDs chunk_00001...
    file_hash = sha256(
        file_path.name.encode("utf-8")
    ).hexdigest()[:16]

    ids = [
        f"file_{file_hash}_{i:06d}"
        for i in range(len(chunks))
    ]

    # Si le fichier est déjà indexé,
    # remplacer uniquement SES chunks.
    # Les autres documents restent intacts.
    existing = collection.get(
        where={"source": file_path.name},
        include=["metadatas"]
    )

    if existing["ids"]:

        collection.delete(
            ids=existing["ids"]
        )

    add_chunks(
        collection,
        ids,
        texts,
        metadatas,
        vectors
    )

    print("\n=== FICHIER INDEXÉ ===")
    print("Document :", file_path.name)
    print("Chunks ajoutés :", len(chunks))
    print("Total ChromaDB :", collection.count())

    return len(chunks)


# ==========================================
# 6. RÉINDEXER TOUS LES DOCUMENTS
# ==========================================

def index_documents():

    documents = load_documents()
    chunks = split_into_chunks(documents)

    if not chunks:

        print("Aucun chunk disponible.")
        return

    print(f"Chunks à indexer : {len(chunks)}")

    texts, metadatas, vectors = (
        prepare_chunks(chunks)
    )

    ids = [
        f"chunk_{i:05d}"
        for i in range(len(chunks))
    ]

    client = get_client()

    # Cette commande reconstruit volontairement
    # la collection entière depuis documents/.
    try:

        client.delete_collection(
            name=COLLECTION_NAME
        )

    except ValueError:

        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None
    )

    add_chunks(
        collection,
        ids,
        texts,
        metadatas,
        vectors
    )

    print("\n=== INDEXATION TERMINÉE ===")
    print("Chunks stockés :", collection.count())


# ==========================================
# 7. LISTER LES DOCUMENTS INDEXÉS
# ==========================================

def get_indexed_sources() -> list[str]:

    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.get(
        include=["metadatas"]
    )

    sources = {
        metadata["source"]
        for metadata in results["metadatas"]
        if metadata and "source" in metadata
    }

    return sorted(sources)


# ==========================================
# 8. RECHERCHE DOCUMENTAIRE
# ==========================================

def search_documents(
    question: str,
    top_k: int = TOP_K
) -> list[dict]:

    collection = get_collection()

    if collection.count() == 0:

        print("ChromaDB est vide.")
        return []

    model = load_embedding_model()

    question_vector = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    candidate_count = min(
        max(top_k * 3, 10),
        collection.count()
    )

    results = collection.query(
        query_embeddings=[
            question_vector.tolist()
        ],
        n_results=candidate_count,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    matches = []
    page_counts = {}

    for i in range(len(results["ids"][0])):

        metadata = results["metadatas"][0][i]

        source = metadata["source"]
        page = metadata["page"]

        page_key = (source, page)

        current_count = page_counts.get(
            page_key,
            0
        )

        if current_count >= 2:
            continue

        matches.append({
            "source": source,
            "page": page,
            "chunk_number": metadata[
                "chunk_number"
            ],
            "content": results["documents"][0][i],
            "distance": results["distances"][0][i]
        })

        page_counts[page_key] = (
            current_count + 1
        )

        if len(matches) >= top_k:
            break

    return matches


# ==========================================
# 9. EXÉCUTION TERMINAL
# ==========================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Commandes :")
        print("python vector_store.py index")
        print("python vector_store.py search")
        print("python vector_store.py add documents/fichier.pdf")

    elif sys.argv[1] == "index":

        index_documents()

    elif sys.argv[1] == "add":

        if len(sys.argv) < 3:

            print(
                "Indique le chemin du fichier."
            )

        else:

            index_file(
                Path(sys.argv[2])
            )

    elif sys.argv[1] == "search":

        question = input(
            "\nPose ta question : "
        ).strip()

        if not question:

            print("Question vide.")
            raise SystemExit(1)

        matches = search_documents(question)

        print("\n=== PASSAGES RETROUVÉS ===")

        for i, match in enumerate(
            matches,
            start=1
        ):

            print(f"\n--- Résultat {i} ---")
            print("Source :", match["source"])
            print("Page :", match["page"])

            print(
                "Distance :",
                round(match["distance"], 3)
            )

            print("Contenu :")
            print(match["content"][:500])

    else:

        print("Commande inconnue.")
