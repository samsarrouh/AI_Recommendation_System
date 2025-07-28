from ingestors.pdf_ingestor import extract_text_from_pdf
from ingestors.csv_ingestor import extract_text_from_csv
from ingestors.web_ingestor import extract_text_from_url
import chromadb

# ✅ Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")

def ingest_document(source_path, source_type):
    """Ingest a document (PDF, CSV, or Web URL) into ChromaDB."""
    chunks = []
    
    if source_type == "pdf":
        chunks = extract_text_from_pdf(source_path)
    elif source_type == "csv":
        chunks = extract_text_from_csv(source_path)
    elif source_type == "url":
        chunks = extract_text_from_url(source_path)
    
    if chunks:
        collection.add(
            ids=[f"{source_path}-{i}" for i in range(len(chunks))],
            embeddings=[[0] * 768] * len(chunks),  # Placeholder embeddings
            metadatas=[{"source": source_path, "text": chunk} for chunk in chunks]
        )
        print(f"✅ Ingested {len(chunks)} chunks from {source_path}")
    else:
        print(f"⚠️ No text extracted from {source_path}")

# ✅ Example Usage:
if __name__ == "__main__":
    ingest_document("example.pdf", "pdf")
    ingest_document("example.csv", "csv")
    ingest_document("https://example.com", "url")