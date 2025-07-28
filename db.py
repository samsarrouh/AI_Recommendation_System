# db.py
import chromadb

# Create a persistent client and a global collection
client = chromadb.PersistentClient(path="./chroma_db")
global_collection = client.get_or_create_collection(name="products")
