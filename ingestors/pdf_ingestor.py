import fitz  # PyMuPDF 
import pytesseract
from PIL import Image
import os
import logging
import re
import hashlib
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter

# If you placed dynamic_chunk_text in a separate file, you can import it:
# from dynamic_chunker import dynamic_chunk_text

def clean_text(text):
    """
    Cleans the input text by removing extra whitespace, newlines, and optionally non-informative characters.
    """
    # Remove HTML tags if necessary (uncomment if needed)
    # text = re.sub(r'<[^>]+>', '', text)

    # Replace newlines and carriage returns with a single space
    text = re.sub(r'[\r\n]+', ' ', text)
    
    # Remove extra spaces (collapse multiple spaces into one)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def dynamic_chunk_text(text, chunk_size=500, chunk_overlap=50):
    import nltk
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if current_chunk:
            new_length = len(current_chunk) + 1 + len(sentence)
        else:
            new_length = len(sentence)
        if new_length <= chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                overlap_text = chunks[i-1][-chunk_overlap:]
                chunk = overlap_text + " " + chunk
            overlapped_chunks.append(chunk)
        chunks = overlapped_chunks
    return chunks

def compute_file_hash(file_path):
    """
    Computes a SHA-256 hash for the file at file_path.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_text_from_pdf(pdf_path, chunk_size=500, chunk_overlap=50, use_dynamic_chunking=False):
    """
    Extract and chunk text from a PDF file.
    Attempts to use machine-readable text first, falling back to OCR if necessary.
    If dynamic chunking is enabled, uses sentence boundaries to form chunks.
    Returns a list of dictionaries containing 'text', 'snippet', 'product_name', 
    and additional metadata: 'ingestion_timestamp' and 'file_hash'.
    """
    try:
        full_pdf_text = ""
        with fitz.open(pdf_path) as doc:
            for page_index, page in enumerate(doc):
                page_text = page.get_text().strip()
                if not page_text:
                    logging.info(f"Page {page_index} has no machine-readable text. Using OCR fallback.")
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    full_pdf_text += ocr_text + "\n"
                else:
                    full_pdf_text += page_text + "\n"
        
        # Pre-clean the extracted text
        full_pdf_text = clean_text(full_pdf_text)
        
        # Compute detailed metadata that applies to the whole file.
        file_hash = compute_file_hash(pdf_path)
        ingestion_timestamp = datetime.datetime.now().isoformat()
        
        # Choose dynamic chunking if enabled; otherwise use RecursiveCharacterTextSplitter.
        if use_dynamic_chunking:
            chunks = dynamic_chunk_text(full_pdf_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_text(full_pdf_text)
        
        result = []
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        for i, chunk in enumerate(chunks):
            snippet = chunk[:150].strip()
            product_name = f"{base_name} - Section {i}"
            result.append({
                'text': chunk,
                'snippet': snippet,
                'product_name': product_name,
                'ingestion_timestamp': ingestion_timestamp,
                'file_hash': file_hash
            })
        logging.info(f"Extracted {len(result)} chunks from PDF: {pdf_path}")
        return result

    except Exception as e:
        logging.error(f"❌ Error processing PDF {pdf_path}: {e}")
        return []
