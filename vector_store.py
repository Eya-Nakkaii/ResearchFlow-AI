
from pathlib import Path
from hashlib import sha256
import sys

import chromadb

from rag_prepare import load_documents, load_document, split_into_chunks
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
# 2. CHROMADB
# ==========================================

def get_client():
    return chromadb.PersistentClient(path=str(DB_PATH))


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


# ==========================================
# 3. PREPARATION DES CHUNKS
# ==========================================

def prepare_chunks(chunks: list[dict]):
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_number": chunk["chunk_number"],
        }
        for chunk in chunks
    ]

    model = load_embedding_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return texts, metadatas, vectors


def add_chunks(collection, ids, texts, metadatas, vectors):
    for start in range(0, len(texts), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=vectors[start:end].tolist(),
        )


# ==========================================
# 4. AJOUTER UN DOCUMENT SANS TOUT EFFACER
# ==========================================

def index_file(file_path: Path) -> int:
    file_path = Path(file_path)
    documents = load_document(file_path)
    chunks = split_into_chunks(documents)

    if not chunks:
        raise ValueError(
            "Aucun texte exploitable. Les PDF scannes peuvent necessiter un OCR."
        )

    print(f"\nIndexation de {file_path.name} : {len(chunks)} chunks")
    texts, metadatas, vectors = prepare_chunks(chunks)
    collection = get_collection()

    file_hash = sha256(file_path.name.encode("utf-8")).hexdigest()[:16]
    ids = [f"file_{file_hash}_{i:06d}" for i in range(len(chunks))]

    # Ne remplacer que les chunks du document portant le meme nom.
    existing = collection.get(
        where={"source": file_path.name},
        include=["metadatas"],
    )
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    add_chunks(collection, ids, texts, metadatas, vectors)

    print("\n=== FICHIER INDEXE ===")
    print("Document :", file_path.name)
    print("Chunks ajoutes :", len(chunks))
    print("Total ChromaDB :", collection.count())
    return len(chunks)


# ==========================================
# 5. RECONSTRUCTION COMPLETE (OPTIONNELLE)
# ==========================================

def index_documents():
    documents = load_documents()
    chunks = split_into_chunks(documents)

    if not chunks:
        print("Aucun chunk disponible.")
        return

    print(f"Chunks a indexer : {len(chunks)}")
    texts, metadatas, vectors = prepare_chunks(chunks)
    ids = [f"chunk_{i:05d}" for i in range(len(chunks))]
    client = get_client()

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    add_chunks(collection, ids, texts, metadatas, vectors)
    print("\n=== INDEXATION TERMINEE ===")
    print("Chunks stockes :", collection.count())


# ==========================================
# 6. LISTER LES DOCUMENTS INDEXES
# ==========================================

def get_indexed_sources() -> list[str]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    sources = {
        metadata["source"]
        for metadata in results["metadatas"]
        if metadata and "source" in metadata
    }
    return sorted(sources)


# ==========================================
# 7. RECHERCHE : TOUS OU UN SEUL DOCUMENT
# ==========================================

def search_documents(
    question: str,
    top_k: int = TOP_K,
    selected_source: str | None = None,
) -> list[dict]:
    """Recherche semantique sur tous les documents ou un document choisi.

    selected_source=None : l'ancienne recherche multi-documents.
    selected_source="rapport.pdf" : uniquement les chunks de ce PDF.
    """
    if top_k < 1:
        raise ValueError("top_k doit etre superieur ou egal a 1.")

    collection = get_collection()
    if collection.count() == 0:
        print("ChromaDB est vide.")
        return []

    where_filter = None
    available_count = collection.count()

    if selected_source is not None:
        # Verifier l'existence du document avant de calculer l'embedding.
        selected = collection.get(
            where={"source": selected_source},
            include=["metadatas"],
        )
        available_count = len(selected["ids"])
        if available_count == 0:
            print(f"Aucun passage indexe pour : {selected_source}")
            return []
        where_filter = {"source": selected_source}

    model = load_embedding_model()
    question_vector = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    candidate_count = min(
        max(top_k * 3, 10),
        available_count,
    )

    query_kwargs = {
        "query_embeddings": [question_vector.tolist()],
        "n_results": candidate_count,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter is not None:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    matches = []
    page_counts = {}
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        page_key = (metadata["source"], metadata["page"])

        if page_counts.get(page_key, 0) >= 2:
            continue

        matches.append(
            {
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_number": metadata["chunk_number"],
                "content": results["documents"][0][i],
                "distance": results["distances"][0][i],
            }
        )
        page_counts[page_key] = page_counts.get(page_key, 0) + 1

        if len(matches) >= top_k:
            break

    return matches


# ==========================================
# 8. TERMINAL
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
            print("Indique le chemin du fichier.")
        else:
            index_file(Path(sys.argv[2]))

    elif sys.argv[1] == "search":
        question = input("\nPose ta question : ").strip()
        if not question:
            print("Question vide.")
            raise SystemExit(1)

        print("Documents disponibles :", get_indexed_sources())
        selected = input("Document exact (Entree = tous) : ").strip() or None
        matches = search_documents(question, selected_source=selected)
        print("\n=== PASSAGES RETROUVES ===")
        for i, match in enumerate(matches, start=1):
            print(f"\n--- Resultat {i} ---")
            print("Source :", match["source"])
            print("Page :", match["page"])
            print("Distance :", round(match["distance"], 3))
            print("Contenu :", match["content"][:500])
    else:
        print("Commande inconnue.")
