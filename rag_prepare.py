
from pathlib import Path
from pypdf import PdfReader


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ==========================================
# 2. LIRE UN SEUL DOCUMENT
# ==========================================

def load_document(file_path: Path) -> list[dict]:

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}"
        )

    documents = []
    extension = file_path.suffix.lower()

    # ---------- PDF ----------

    if extension == ".pdf":

        print(f"\nLecture PDF : {file_path.name}")

        reader = PdfReader(str(file_path))

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text() or ""

            if text.strip():

                documents.append({
                    "source": file_path.name,
                    "page": page_number,
                    "text": text
                })

    # ---------- TXT ----------

    elif extension == ".txt":

        print(f"\nLecture TXT : {file_path.name}")

        text = file_path.read_text(
            encoding="utf-8-sig"
        )

        if text.strip():

            documents.append({
                "source": file_path.name,
                "page": 0,
                "text": text
            })

    else:

        raise ValueError(
            "Format non pris en charge. "
            "Formats acceptés : PDF et TXT."
        )

    return documents


# ==========================================
# 3. LIRE TOUS LES DOCUMENTS
# ==========================================

def load_documents() -> list[dict]:

    documents = []

    if not DOCUMENTS_DIR.exists():

        print("Le dossier documents/ est introuvable.")
        return documents

    for file_path in sorted(DOCUMENTS_DIR.iterdir()):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in [
            ".pdf",
            ".txt"
        ]:
            continue

        documents.extend(
            load_document(file_path)
        )

    return documents


# ==========================================
# 4. DÉCOUPER EN CHUNKS
# ==========================================

def split_into_chunks(
    documents: list[dict]
) -> list[dict]:

    chunks = []

    for document in documents:

        text = document["text"]
        start = 0
        chunk_number = 0

        while start < len(text):

            end = start + CHUNK_SIZE
            chunk_text = text[start:end]

            chunks.append({
                "source": document["source"],
                "page": document["page"],
                "chunk_number": chunk_number,
                "content": chunk_text
            })

            chunk_number += 1

            if end >= len(text):
                break

            start = end - CHUNK_OVERLAP

    return chunks


# ==========================================
# 5. TEST EN TERMINAL
# ==========================================

if __name__ == "__main__":

    documents = load_documents()
    chunks = split_into_chunks(documents)

    print("\n=== RÉSULTATS ===")

    print(
        "Unités de texte chargées :",
        len(documents)
    )

    print("Chunks créés :", len(chunks))

    if chunks:

        first_chunk = chunks[0]

        print("\n=== EXEMPLE DE CHUNK ===")
        print("Source :", first_chunk["source"])
        print("Page :", first_chunk["page"])
        print(
            "Numéro de chunk :",
            first_chunk["chunk_number"]
        )
        print("Contenu :")
        print(first_chunk["content"][:500])

    else:

        print("Aucun texte extrait.")
