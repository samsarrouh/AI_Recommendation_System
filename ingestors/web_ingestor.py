import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import re
from urllib.parse import urlparse
import nltk

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

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
    """
    Splits the text into chunks using sentence boundaries so that each chunk
    is approximately chunk_size characters long without breaking sentences.
    Optionally, adds a small overlap between chunks.
    """
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

def extract_text_from_url(url, chunk_size=500, chunk_overlap=50, use_dynamic_chunking=False):
    """
    Extract and chunk text from a web page.
    Returns a list of dictionaries, each containing:
      - 'text': the full text chunk,
      - 'snippet': the first 150 characters of the chunk,
      - 'product_name': a descriptive name derived from the URL's hostname and section index.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MyBot/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logging.error(f"❌ Failed to fetch URL {url}: Status code {response.status_code}")
            return []
        
        if "text/html" not in response.headers.get("Content-Type", ""):
            logging.error(f"❌ URL {url} did not return HTML content.")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(soup.get_text(separator=" ").split())
        
        text = clean_text(text)
        
        if use_dynamic_chunking:
            # Use dynamic chunking based on sentence boundaries.
            chunks = dynamic_chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            # Use the existing static chunking from langchain.
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_text(text)
        
        result = []
        parsed_url = urlparse(url)
        base_name = parsed_url.hostname or url
        for i, chunk in enumerate(chunks):
            snippet = chunk[:150].strip()
            product_name = f"{base_name} - Section {i}"
            result.append({
                'text': chunk,
                'snippet': snippet,
                'product_name': product_name
            })
        logging.info(f"Extracted {len(result)} chunks from URL: {url}")
        return result
    except Exception as e:
        logging.error(f"❌ Error processing URL {url}: {e}")
        return []
