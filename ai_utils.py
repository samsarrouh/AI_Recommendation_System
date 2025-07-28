# ai_utils.py
import requests
import logging

# Use your actual endpoints or environment variables
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

def get_embedding(text):
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": "mxbai-embed-large", "prompt": text},
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        logging.info(f"Generated embedding for text: {text[:50]}")
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None

def ollama_generate_response(prompt, context=""):
    try:
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": "mistral", "prompt": full_prompt, "stream": False},
            headers=headers,
            timeout=15
        )
        response.raise_for_status()
        result = response.json()
        resp = result.get("response", "").strip()
        logging.info(f"Generated response for prompt: {prompt[:50]}")
        return resp
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return None
