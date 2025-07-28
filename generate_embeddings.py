import os
import logging
import pandas as pd
import chromadb
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Load product data from CSV file
data_file = "products.csv"
if not os.path.exists(data_file):
    logging.error(f"❌ File {data_file} does not exist.")
    exit(1)

data = pd.read_csv(data_file)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="products")

# Embedding API URL for Ollama
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"

def get_embedding(text):
    """Generate an embedding for the given text using Ollama."""
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": "mxbai-embed-large", "prompt": text},
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        response_json = response.json()
        return response_json.get("embedding")
    except Exception as e:
        logging.error(f"❌ Error generating embedding for text '{text[:50]}': {e}")
        return None

# Clear existing collection before re-inserting new data
existing_data = collection.get()
if existing_data and "ids" in existing_data and existing_data["ids"]:
    collection.delete(ids=existing_data["ids"])
    logging.info("Cleared existing collection data.")

# Process each product and store embeddings
embeddings = []
for i, row in data.iterrows():
    product_name = row["product_name"]
    embedding = get_embedding(product_name)
    if embedding:
        collection.add(
            ids=[str(i)],  # Use product index as unique ID
            embeddings=[embedding],
            metadatas=[{"product_name": product_name}]
        )
        embeddings.append((product_name, embedding))
    else:
        logging.warning(f"Embedding not generated for product: {product_name}")

if embeddings:
    # Save the dataset with embeddings
    df_embeddings = pd.DataFrame({"product_name": [e[0] for e in embeddings]})
    df_embeddings.to_csv("products_with_embeddings.csv", index=False)
    logging.info(f"✅ Successfully stored {len(embeddings)} products in ChromaDB!")
else:
    logging.warning("No embeddings were generated, nothing was stored in ChromaDB.")
