import os
import hashlib
import logging
import chromadb
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class VectorDBUtility:
    def __init__(self, db_path="./chroma_db", collection_name="products"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def compute_chunk_id(self, filename, chunk_text):
        """Compute a SHA-256 hash based on filename and chunk text."""
        h = hashlib.sha256()
        h.update(filename.encode('utf-8'))
        h.update(chunk_text.encode('utf-8'))
        return h.hexdigest()

    def get_existing_ids(self):
        """Retrieves all stored document IDs from the collection."""
        data = self.collection.get()
        return set(data.get("ids", [])) if data else set()

    def is_file_ingested(self, filename):
        """Checks if a document with the given filename exists in the database."""
        data = self.collection.get(include=["metadatas"])
        if not data:
            return False
        return any(meta.get("filename", "").strip().lower() == filename.strip().lower() 
                   for meta in data.get("metadatas", []))

    def delete_document_by_id(self, doc_id):
        """Deletes a document from the collection using its ID."""
        try:
            self.collection.delete([doc_id])
            logging.info(f"Deleted document ID: {doc_id}")
        except Exception as e:
            logging.error(f"Error deleting document ID {doc_id}: {e}")

    def delete_documents_by_filename(self, filename):
        """Deletes all documents with a specific filename."""
        try:
            # Get metadata only; ids are returned by default.
            data = self.collection.get(include=["metadatas"])
            if not data:
                logging.info("No data found in the database.")
                return
            
            to_delete = [
                doc_id 
                for doc_id, meta in zip(data.get("ids", []), data.get("metadatas", []))
                if meta.get("filename", "").strip().lower() == filename.strip().lower()
            ]
            
            if to_delete:
                self.collection.delete(to_delete)
                logging.info(f"Deleted {len(to_delete)} documents with filename: {filename}")
            else:
                logging.info(f"No documents found with filename: {filename}")
        except Exception as e:
            logging.error(f"Error deleting documents by filename {filename}: {e}")

    def get_document_by_id(self, doc_id):
        """Retrieves a document's content by its ID."""
        try:
            data = self.collection.get(ids=[doc_id], include=["documents"])
            return data.get("documents", [None])[0] if data else None
        except Exception as e:
            logging.error(f"Error retrieving document ID {doc_id}: {e}")
            return None

    def get_documents_by_filename(self, filename):
        """Retrieves all documents with a specific filename."""
        try:
            data = self.collection.get(include=["documents", "metadatas"])
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])
            result = []
            filename_query = filename.strip().lower()
            
            for i in range(len(documents)):
                metadata = metadatas[i] if i < len(metadatas) else {}
                meta_filename = metadata.get("filename", "").strip().lower()
                
                if filename_query in meta_filename:
                    result.append({
                        "filename": meta_filename,
                        "content": documents[i] if i < len(documents) else ""
                    })
            
            return result
        except Exception as e:
            logging.error(f"Error retrieving documents by filename {filename}: {e}")
            return []

    def document_exists(self, doc_id):
        """Checks if a document ID exists in the collection."""
        try:
            data = self.collection.get(ids=[doc_id])
            return bool(data.get("ids", []))
        except Exception as e:
            logging.error(f"Error checking document existence for ID {doc_id}: {e}")
            return False

    def list_documents(self, limit=10):
        """Retrieves a list of stored document metadata, IDs, and text."""
        try:
            data = self.collection.get(include=["documents", "metadatas"], limit=limit)
            ids = data.get("ids", [])
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])
            result = []
            for i in range(len(documents)):
                metadata = metadatas[i] if i < len(metadatas) else {}
                filename = metadata.get("filename", "Unknown Filename")
                result.append({
                    "id": ids[i] if i < len(ids) else "Unknown ID",
                    "filename": filename,
                    "content": documents[i] if i < len(documents) else ""
                })
            return result
        except Exception as e:
            logging.error(f"Error retrieving document list: {e}")
            return []

    def clear_database(self):
        """Deletes all documents in the collection."""
        try:
            # Call get() without include parameters so that 'ids' are returned by default.
            data = self.collection.get()
            ids = data.get("ids", [])
            if ids:
                self.collection.delete(ids)
                logging.info("Vector database cleared successfully.")
            else:
                logging.info("No documents found in the database.")
        except Exception as e:
            logging.error(f"Error clearing database: {e}")

    def remove_duplicates(self):
        """Removes duplicate documents by checking for identical content."""
        try:
            data = self.collection.get(include=["documents"])
            if not data:
                logging.info("No data found in the database.")
                return
            
            seen_texts = set()
            duplicate_ids = []
            for doc_id, doc_text in zip(data.get("ids", []), data.get("documents", [])):
                if doc_text in seen_texts:
                    duplicate_ids.append(doc_id)
                else:
                    seen_texts.add(doc_text)
            
            if duplicate_ids:
                self.collection.delete(duplicate_ids)
                logging.info(f"Removed {len(duplicate_ids)} duplicate documents.")
            else:
                logging.info("No duplicates found.")
        except Exception as e:
            logging.error(f"Error removing duplicates: {e}")

if __name__ == "__main__":
    db_util = VectorDBUtility()
    print(db_util.list_documents())
