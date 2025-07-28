# AI_Recommendation_System

An AI-powered recommendation system that ingests documents, generates embeddings, and provides intelligent retrieval via a Flask web interface.

---

## Project Structure

- **app.py**: Main Flask web application entry point.
- **app_v0.py**: Alternate or legacy version of the Flask app.
- **db.py**: Database connection and management logic.
- **document_ingestion.py**: Script for ingesting and processing documents into the system.
- **dynamic_chunker.py**: Handles dynamic chunking of documents for embedding.
- **generate_embeddings.py**: Utility for generating vector embeddings from ingested documents.
- **ingestion_manager.py**: Orchestrates the ingestion process.
- **reporting.py**: Reporting and analytics logic.
- **utility.py**: General utility functions.
- **ai_utils.py**: AI model configuration and helper functions.
- **requirements.txt**: Python dependencies for the project.
- **chroma_db/**: Vector database or storage for embeddings.
- **Docs/**: Project documentation and related resources.
- **ingestors/**: Contains specific ingestors (e.g., PDF, web) for different document types.
- **nltk_data/**: NLTK data files (e.g., punkt tokenizer).
- **share/**: Shared resources or files.
- **static/**: Static files (CSS, JS, images) for the web interface.
- **templates/**: HTML templates for Flask web pages.
- **tests/**: Unit tests for the project.

---

## Setup Instructions

### 1. Clone the repository

```sh
git clone <your-repo-url>
cd AI_Recommendation_System
```

### 2. Create and activate a virtual environment

#### On Windows

```sh
python -m venv venv
venv\Scripts\activate
```

#### On Mac/Linux

```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```sh
pip install -r requirements.txt
```

### 4. Run the Flask app

```sh
python app.py
```

### 5. Run document ingestion

```sh
python document_ingestion.py
```

### 6. Run unit tests

```sh
python -m unittest discover tests
```

---

## Notes

- **NLTK punkt requirement**:  
  The system requires the NLTK 'punkt' tokenizer. If not already downloaded, run:
  ```sh
  python -m nltk.downloader punkt
  ```

- **Model configuration**:  
  Model and embedding configuration can be found and adjusted in [`ai_utils.py`](ai_utils.py).

---

## License

This project is private and not licensed for redistribution.