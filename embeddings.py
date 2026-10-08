
from sentence_transformers import SentenceTransformer
import numpy as np

from rag_prepare import load_documents, split_into_chunks


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 3


# ==========================================
# 2. CHARGER LES DOCUMENTS ET LES CHUNKS
# ==========================================

def prepare_chunks():

    documents = load_documents()

    chunks = split_into_chunks(documents)

    print(f"\nDocuments chargés : {len(documents)}")
    print(f"Chunks créés : {len(chunks)}")

    return chunks


# ==========================================
# 3. CHARGER LE MODÈLE D'EMBEDDINGS
# ==========================================

def load_embedding_model():

    print("\nChargement du modèle d'embeddings...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    return model


# ==========================================
# 4. TRANSFORMER LES CHUNKS EN VECTEURS
# ==========================================

def create_embeddings(model, chunks):

    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(f"\nNombre de vecteurs : {len(embeddings)}")
    print(f"Dimensions : {embeddings.shape[1]}")

    return embeddings


# ==========================================
# 5. RECHERCHER LES CHUNKS PERTINENTS
# ==========================================

def search_chunks(model, chunks, embeddings, question):

    # Transformer la question en vecteur
    question_vector = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Calculer la similarité avec chaque chunk
    scores = embeddings @ question_vector

    # Trier les indices par score décroissant
    top_indices = np.argsort(-scores)[:TOP_K]

    results = []

    for index in top_indices:

        chunk = chunks[index]

        results.append({
            "source": chunk["source"],
            "content": chunk["content"],
            "score": float(scores[index])
        })

    return results


# ==========================================
# 6. EXÉCUTION
# ==========================================

if __name__ == "__main__":

    chunks = prepare_chunks()

    if not chunks:
        print("\nAucun chunk disponible.")
        raise SystemExit(1)

    model = load_embedding_model()

    embeddings = create_embeddings(model, chunks)

    question = input("\nPose ta question : ").strip()

    if not question:
        print("La question ne peut pas être vide.")
        raise SystemExit(1)

    results = search_chunks(
        model,
        chunks,
        embeddings,
        question
    )

    print("\n=== CHUNKS LES PLUS PERTINENTS ===")

    for i, result in enumerate(results, start=1):

        print(f"\n--- Résultat {i} ---")
        print(f"Source : {result['source']}")
        print(f"Similarité : {result['score']:.3f}")

        print("\nContenu :")
        print(result["content"][:500])
