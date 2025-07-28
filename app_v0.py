# app.py
import os
import shutil
import hashlib
import logging
from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import tempfile
import pdfkit
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import logging
import traceback

# If you have other ingestors or custom modules
from ingestors.csv_ingestor import extract_text_from_csv
from ingestors.pdf_ingestor import extract_text_from_pdf
from ingestors.web_ingestor import extract_text_from_url
from utility import VectorDBUtility  

from db import global_collection  # We import your ChromaDB collection from db.py
from ai_utils import get_embedding, ollama_generate_response  # AI calls

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

app = Flask(__name__, template_folder="templates")
CORS(app)

# ✅ Initialize the database utility
db_util = VectorDBUtility()

# NLTK config
nltk.data.path.append("C:\\Users\\samer\\AppData\\Roaming\\nltk_data")
nltk.download('punkt')

def dynamic_extract_snippet(full_text, min_length=300):
    try:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(full_text)
    except Exception as e:
        logging.error(f"Error loading punkt tokenizer: {e}")
        sentences = sent_tokenize(full_text)
    snippet = ""
    for sentence in sentences:
        snippet += sentence + " "
        if len(snippet) >= min_length:
            break
    return snippet.strip()

# ---------- Helpers ----------

def compute_chunk_id(filename, chunk_text):
    h = hashlib.sha256()
    h.update(filename.encode('utf-8'))
    h.update(chunk_text.encode('utf-8'))
    return h.hexdigest()

def get_existing_ids():
    data = global_collection.get()
    return set(data.get("ids", [])) if data else set()

def is_file_ingested(filename):
    data = global_collection.get(include=["metadatas"])
    if not data:
        return False
    for meta in data.get("metadatas", []):
        if meta.get("filename", "").strip().lower() == filename.strip().lower():
            return True
    return False

# ---------- Routes ----------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ingest", methods=["POST"])
def ingest_document():
    try:
        if "file" in request.files:
            file = request.files["file"]
            filename = file.filename
            if is_file_ingested(filename):
                logging.info(f"File '{filename}' already ingested. Skipping ingestion.")
                return jsonify({"success": f"File '{filename}' already ingested."}), 200

            file_type = filename.split(".")[-1].lower()
            chunk_size = int(request.form.get("chunk_size", 500))
            chunk_overlap = int(request.form.get("chunk_overlap", 50))
            if file_type == "pdf":
                return ingest_pdf_endpoint(file, chunk_size, chunk_overlap)
            elif file_type == "csv":
                return ingest_csv_endpoint(file, chunk_size, chunk_overlap)
            else:
                return jsonify({"error": "Unsupported file type"}), 400

        elif request.is_json:
            data = request.get_json()
            chunk_size = data.get("chunk_size", 500)
            chunk_overlap = data.get("chunk_overlap", 50)
            if "url" in data:
                return ingest_url_endpoint(data["url"], chunk_size, chunk_overlap)
        return jsonify({"error": "No valid input provided"}), 400
    except Exception as e:
        logging.error(f"Error in /ingest: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

def ingest_pdf_endpoint(file, chunk_size=500, chunk_overlap=50): 
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)
    try:
        # Retrieve the 'use_dynamic_chunking' flag from form data; default to False if not provided.
        use_dynamic_chunking = request.form.get("use_dynamic_chunking", "false").lower() == "true"
        
        # Compute file hash and ingestion timestamp for detailed metadata.
        import hashlib
        import datetime
        def compute_file_hash(file_path):
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        
        file_hash = compute_file_hash(temp_path)
        ingestion_timestamp = datetime.datetime.now().isoformat()
        
        chunks = extract_text_from_pdf(temp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, use_dynamic_chunking=use_dynamic_chunking)
        
        if not chunks:
            return jsonify({"error": "No text extracted from PDF"}), 400
        existing_ids = get_existing_ids()
        for i, chunk_dict in enumerate(chunks):
            full_text = chunk_dict['text']
            embedding_id = compute_chunk_id(file.filename, full_text)
            if embedding_id in existing_ids:
                logging.info(f"Skipping duplicate embedding ID: {embedding_id}")
                continue
            snippet = dynamic_extract_snippet(full_text, min_length=300)
            product_name = chunk_dict.get('product_name', f"{file.filename}_chunk_{i}")
            embedding = get_embedding(full_text)
            if embedding:
                global_collection.add(
                    ids=[embedding_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": "pdf",
                        "filename": file.filename,
                        "snippet": snippet,
                        "product_name": product_name,
                        "ingestion_timestamp": ingestion_timestamp,
                        "file_hash": file_hash
                    }],
                    documents=[full_text]
                )
                existing_ids.add(embedding_id)
                logging.info(f"Added embedding ID: {embedding_id}")
        logging.info(f"Ingested PDF: {file.filename}")
        return jsonify({"success": f"Ingested {file.filename}"}), 200
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        return jsonify({"error": "Failed to process PDF"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def ingest_csv_endpoint(file, chunk_size=500, chunk_overlap=50):
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)
    try:
        chunks = extract_text_from_csv(temp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            return jsonify({"error": "No text extracted from CSV"}), 400
        existing_ids = get_existing_ids()
        for i, chunk_dict in enumerate(chunks):
            full_text = chunk_dict['text']
            embedding_id = compute_chunk_id(file.filename, full_text)
            if embedding_id in existing_ids:
                logging.info(f"Skipping duplicate embedding ID: {embedding_id}")
                continue
            snippet = dynamic_extract_snippet(full_text, min_length=300)
            product_name = chunk_dict.get('product_name', f"{file.filename}_chunk_{i}")
            embedding = get_embedding(full_text)
            if embedding:
                global_collection.add(
                    ids=[embedding_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": "csv",
                        "filename": file.filename,
                        "snippet": snippet,
                        "product_name": product_name
                    }],
                    documents=[full_text]
                )
                existing_ids.add(embedding_id)
                logging.info(f"Added embedding ID: {embedding_id}")
        logging.info(f"Ingested CSV: {file.filename}")
        return jsonify({"success": f"Ingested {file.filename}"}), 200
    except Exception as e:
        logging.error(f"Error processing CSV: {e}")
        return jsonify({"error": "Failed to process CSV"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def ingest_url_endpoint(url, chunk_size=500, chunk_overlap=50):
    try:
        data = request.get_json() or {}
        use_dynamic_chunking = data.get("use_dynamic_chunking", False)
        chunks = extract_text_from_url(url, chunk_size=chunk_size, chunk_overlap=chunk_overlap, use_dynamic_chunking=use_dynamic_chunking)
        if not chunks:
            return jsonify({"error": "No meaningful text found in URL"}), 400
        existing_ids = get_existing_ids()
        for i, chunk_dict in enumerate(chunks):
            full_text = chunk_dict['text']
            embedding_id = compute_chunk_id(url, full_text)
            if embedding_id in existing_ids:
                logging.info(f"Skipping duplicate embedding ID: {embedding_id}")
                continue
            snippet = dynamic_extract_snippet(full_text, min_length=300)
            product_name = chunk_dict.get('product_name', f"{url}_chunk_{i}")
            embedding = get_embedding(full_text)
            if embedding:
                global_collection.add(
                    ids=[embedding_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": "url",
                        "url": url,
                        "snippet": snippet,
                        "product_name": product_name
                    }],
                    documents=[full_text]
                )
                existing_ids.add(embedding_id)
                logging.info(f"Added embedding ID: {embedding_id}")
        logging.info(f"Ingested URL: {url}")
        return jsonify({"success": f"Ingested content from {url}"}), 200
    except Exception as e:
        logging.error(f"Error processing URL: {e}")
        return jsonify({"error": "Failed to process URL"}), 500

@app.route("/product-summary", methods=["POST"])
def product_summary():
    try:
        data_json = request.get_json()
        product_name = data_json.get("product", "").strip()
        if not product_name:
            return jsonify({"error": "Product name cannot be empty"}), 400

        snippet = get_product_snippet(product_name)
        if snippet:
            prompt = (
                f"Using the following information:\n\"{snippet}\"\n"
                f"provide a detailed summary of the product '{product_name}'. "
                "Include any additional context and ensure the summary is at least 150 words long."
            )
        else:
            prompt = (
                f"Provide a detailed summary of the product '{product_name}' based on all available data. "
                "Ensure the summary is comprehensive and at least 150 words long."
            )
        response = ollama_generate_response(prompt, "")
        # Fallback if response is too short
        if not response or len(response.split()) < 150:
            fallback_prompt = (
                f"Please provide a comprehensive summary of the product '{product_name}' using all available data. "
                "Ensure the summary is very detailed and at least 150 words long."
            )
            response = ollama_generate_response(fallback_prompt, snippet if snippet else "")
        logging.info(f"Generated summary for product '{product_name}': {response}")
        return jsonify({"summary": response})
    except Exception as e:
        logging.error(f"Error in /product-summary: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

def get_product_snippet(query_product):
    try:
        data = global_collection.get(include=["documents", "metadatas"])
        metadatas = data.get("metadatas", [])
        for meta in metadatas:
            stored_name = meta.get("product_name", "")
            if query_product.strip().lower() in stored_name.strip().lower():
                snippet = meta.get("snippet", None)
                if snippet:
                    return snippet
        return None
    except Exception as e:
        logging.error(f"Error retrieving product snippet: {e}")
        return None

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data_json = request.get_json()

        # 1) Read query and threshold from the POST body
        user_query = data_json.get("query", "").strip()
        threshold_raw = data_json.get("threshold", 200.0)  # default 200 if none provided
        # Make sure it's float
        threshold = float(threshold_raw)

        if not user_query:
            return jsonify({"error": "Query cannot be empty"}), 400

        # 2) Get the query embedding
        query_embedding = get_embedding(user_query)
        if not query_embedding:
            return jsonify({"error": "Failed to generate embedding"}), 500

        # 3) Query your vector DB for up to 5 results
        results = global_collection.query(query_embeddings=[query_embedding], n_results=5)

        # 4) Distances, plus optional +1 offset
        distances = [dist + 1 for dist in results.get("distances", [[]])[0]]
        metadatas = results.get("metadatas", [[]])[0]

        # Debug: see raw distances and chosen threshold
        print("Modified Distances:", distances)
        print("User-chosen threshold:", threshold)

        # 5) Filter out results above threshold
        filtered_results = [
            (dist, meta)
            for dist, meta in zip(distances, metadatas)
            if dist <= threshold
        ]

        if not filtered_results:
            return jsonify({
                "message": (
                    "No relevant matches were found in the uploaded data for your query. "
                    "Try refining your search or uploading more relevant documents."
                ),
                "recommendations": []
            })

        # 6) Decide message based on whether all distances > 170 (but below threshold)
        distances_only = [item[0] for item in filtered_results]
        if all(d > 170 for d in distances_only):
            message = "No highly relevant matches were found, but here are the closest available results."
        else:
            message = "Here are the most relevant results based on your query."

        # 7) Min-Max scale for a 0-100 range, with min_d => 100 and max_d => 0
        min_d = min(distances_only)
        max_d = max(distances_only)

        recommendations = []
        query_keywords = set(user_query.lower().split())

        for i, (distance, meta) in enumerate(filtered_results):
            prod_name = meta.get("product_name", f"Doc_{i}")
            snippet = meta.get("snippet", "").lower()
            combined_text = (prod_name + " " + snippet).lower()

            # Min-Max scaling
            if max_d == min_d:
                # edge case: all distances identical
                raw_score = 100
            else:
                # 1 - (distance - min_d) / (max_d - min_d)
                normalized = 1 - ((distance - min_d) / (max_d - min_d))
                raw_score = 100 * normalized

            score = round(max(0, min(100, raw_score)), 2)

            # Keyword + Score >= 30
            if any(keyword in combined_text for keyword in query_keywords) and score >= 30:
                recommendations.append({"product": prod_name, "score": score})

        return jsonify({"message": message, "recommendations": recommendations})

    except Exception as e:
        logging.error(f"Error in /recommend: {traceback.format_exc()}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data_json = request.get_json()
        user_query = data_json.get("query", "").strip()
        if not user_query:
            return jsonify({"error": "Query cannot be empty"}), 400

        query_embedding = get_embedding(user_query)
        if not query_embedding:
            return jsonify({"error": "Failed to generate embedding"}), 500

        # Query top N documents from your vector DB
        results = global_collection.query(query_embeddings=[query_embedding], n_results=5)
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]

        # Optional +1 offset if you're doing that in your setup
        # distances = [dist + 1 for dist in distances]
        # Debug: see raw distances and chosen threshold
        print("Modified Distances:", distances)
       
        # Static threshold = 150-270
        THRESHOLD = 270.0

        # Filter out any chunk above THRESHOLD  distance
        filtered_pairs = [
            (dist, doc) for dist, doc in zip(distances, documents)
            if dist <= THRESHOLD
        ]

        # If no chunks are below threshold, skip vector context
        if not filtered_pairs:
            # Just call your LLM with the user query alone
            response_text = ollama_generate_response(prompt=user_query, context="")
            return jsonify({"response": response_text})

        # Otherwise, build context from the relevant docs
        relevant_docs = [doc for _, doc in filtered_pairs if doc]
        vector_context = "\n\n".join(relevant_docs)

        # Optional dynamic snippet logic if you have it
        dynamic_context = get_product_snippet_dynamic(user_query)
        combined_context = ""
        if dynamic_context:
            combined_context += f"Dynamic snippet:\n{dynamic_context}\n\n"
        combined_context += f"Retrieved chunks:\n{vector_context}"

        # Build the final system prompt
        import datetime
        today_str = datetime.datetime.now().strftime("%B %d, %Y")
        system_message = f"You are a helpful AI assistant. Today is {today_str}."

        full_prompt = f"{system_message}\n\nContext:\n{combined_context}\n\nUser's question:\n{user_query}"
        response_text = ollama_generate_response(prompt=full_prompt, context="")

        # Fallback if the model returns an empty or "undefined" response
        if not response_text or response_text.strip().lower() == "undefined":
            response_text = ollama_generate_response(prompt=user_query, context="")

        return jsonify({"response": response_text})
    except Exception as e:
        logging.error(f"Error in /chat: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


def get_product_snippet_dynamic(query):
    try:
        data = global_collection.get(include=["metadatas"])
        if not data:
            return None
        query_words = set(query.lower().split())
        best_score = 0
        best_snippet = None
        for meta in data.get("metadatas", []):
            product_name = meta.get("product_name", "").lower()
            if not product_name:
                continue
            product_words = set(product_name.split())
            score = len(query_words & product_words) / len(query_words) if query_words else 0
            if score > best_score:
                best_score = score
                best_snippet = meta.get("snippet", None)
        return best_snippet if best_score >= 0.5 else None
    except Exception as e:
        logging.error(f"Error in dynamic snippet retrieval: {e}")
        return None

@app.route("/debug-chroma", methods=["GET"])
def debug_chroma():
    try:
        data = global_collection.get()
        ids = data.get("ids", [])
        return jsonify({"number_of_docs": len(ids), "sample_ids": ids[:5]})
    except Exception as e:
        logging.error(f"Error in /debug-chroma: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/debug-chroma-details/<int:index>", methods=["GET"])
def debug_chroma_details(index):
    try:
        data = global_collection.get()
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        if index < 0 or index >= len(documents):
            return jsonify({"error": "Index out of range"}), 404
        return jsonify({
            "chunk_id": index,
            "document_text": documents[index],
            "metadata": metadatas[index]
        })
    except Exception as e:
        logging.error(f"Error in /debug-chroma-details: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/report", methods=["GET"])
def report_builder():
    try:
        return render_template("report_builder.html")
    except Exception as e:
        logging.error(f"Error rendering report builder: {e}")
        return "Error loading report builder.", 500

@app.route("/report-preview", methods=["POST"])
def report_preview():
    try:
        report_config = request.get_json()
        # Now that reporting.py no longer imports from app.py, there's no circular import
        from reporting import generate_report_preview  
        html_report, _, _ = generate_report_preview(report_config)
        return jsonify({"html_report": html_report})
    except Exception as e:
        logging.error(f"Error generating report preview: {e}")
        return jsonify({"error": "Failed to generate report preview"}), 500
    
@app.route("/db_maintenance")
def db_maintenance():
    return render_template("db_maintenance.html")

@app.route("/db_maintenance/<action>", methods=["POST"])
def execute_db_action(action):
    try:
        if action == "clear_database":
            db_util.clear_database()
            return jsonify({"message": "Database cleared successfully."})
        elif action == "remove_duplicates":
            db_util.remove_duplicates()
            return jsonify({"message": "Duplicates removed successfully."})
        elif action == "list_documents":
            docs = db_util.list_documents()
            return jsonify({"message": "Documents listed.", "documents": docs})
        else:
            return jsonify({"message": "Invalid action."}), 400
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500
    
@app.route("/db_maintenance/delete_by_filename", methods=["POST"])
def delete_by_filename():
    """Deletes all documents with a specific filename."""
    try:
        data = request.get_json()
        filename = data.get("filename")
        if not filename:
            return jsonify({"message": "Filename is required."}), 400

        db_util.delete_documents_by_filename(filename)
        return jsonify({"message": f"Deleted documents with filename: {filename}"})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route("/db_maintenance/delete_by_id", methods=["POST"])
def delete_by_id():
    """Deletes a document by its ID."""
    try:
        data = request.get_json()
        doc_id = data.get("doc_id")
        if not doc_id:
            return jsonify({"message": "Document ID is required."}), 400

        db_util.delete_document_by_id(doc_id)
        return jsonify({"message": f"Deleted document ID: {doc_id}"})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route("/db_maintenance/is_file_ingested", methods=["POST"])
def check_file_ingested():
    """Checks if a file has already been ingested into the database."""
    try:
        data = request.get_json()
        filename = data.get("filename")
        if not filename:
            return jsonify({"message": "Filename is required."}), 400

        exists = db_util.is_file_ingested(filename)
        return jsonify({"message": "File ingestion status retrieved.", "exists": exists})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route("/db_maintenance/get_document_by_id", methods=["POST"])
def get_document_by_id():
    """Retrieves a document's content using its ID."""
    try:
        data = request.get_json()
        doc_id = data.get("doc_id")
        if not doc_id:
            return jsonify({"message": "Document ID is required."}), 400

        document = db_util.get_document_by_id(doc_id)
        if document:
            return jsonify({"message": "Document retrieved.", "document": document})
        else:
            return jsonify({"message": "Document not found."}), 404
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route("/db_maintenance/get_documents_by_filename", methods=["POST"])
def get_documents_by_filename():
    """Retrieves all documents stored under a specific filename."""
    try:
        data = request.get_json()
        filename = data.get("filename")
        if not filename:
            return jsonify({"message": "Filename is required."}), 400

        documents = db_util.get_documents_by_filename(filename)
        return jsonify({"message": "Documents retrieved.", "documents": documents})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500
    
@app.route("/db_maintenance/list_documents", methods=["POST"])
def list_documents():
    """Retrieves a list of stored documents."""
    try:
        data = db_util.list_documents()
        if not data:
            return jsonify({"message": "No documents found."}), 404
        return jsonify({"message": "Documents retrieved.", "documents": data})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route("/report-export", methods=["POST"])
def report_export():
    try:
        data = request.get_json()
        report_format = data.get("format", "pdf").lower()
        edited_html = data.get("edited_html")
        if report_format == "pdf":
            wkhtmltopdf_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
            config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
            if edited_html:
                pdf_data = pdfkit.from_string(edited_html, False, configuration=config)
            else:
                report_config = data.get("report_config", {})
                from reporting import generate_report_pdf
                pdf_data = generate_report_pdf(report_config)
            if pdf_data is None:
                return jsonify({"error": "Failed to generate PDF"}), 500
            response = make_response(pdf_data)
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = 'attachment; filename=report.pdf'
            return response
        elif report_format == "csv":
            report_config = data.get("report_config", {})
            from reporting import generate_report_csv
            csv_data = generate_report_csv(report_config)
            response = make_response(csv_data)
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = 'attachment; filename=report.csv'
            return response
        else:
            return jsonify({"error": "Unsupported format"}), 400
    except Exception as e:
        logging.error(f"Error exporting report: {e}")
        return jsonify({"error": "Failed to export report"}), 500
import inspect
print("Utility.py location:", inspect.getfile(VectorDBUtility))

if __name__ == "__main__":
    app.run(debug=True)
