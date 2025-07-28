from document_ingestion import ingest_document

# ✅ List of sources to ingest
documents = [
    {"path": "example.pdf", "type": "pdf"},
    {"path": "example.csv", "type": "csv"},
    {"path": "https://example.com", "type": "url"}
]

for doc in documents:
    ingest_document(doc["path"], doc["type"])
