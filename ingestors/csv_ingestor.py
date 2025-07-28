import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def extract_text_from_csv(csv_path, chunk_size=500, chunk_overlap=50):
    """
    Extract and chunk text from a CSV file.
    Returns a list of dictionaries, each containing:
      - 'text': the concatenated row text,
      - 'snippet': the first 150 characters of that text,
      - 'product_name': a descriptive name derived from the CSV file name and section index.
    """
    try:
        df = pd.read_csv(csv_path)
        text_data = "\n".join(df.astype(str).apply(lambda x: ' '.join(x), axis=1))
        text_data = " ".join(text_data.split())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text_data)
        result = []
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        for i, chunk in enumerate(chunks):
            snippet = chunk[:150].strip()
            product_name = f"{base_name} - Section {i}"
            result.append({
                'text': chunk,
                'snippet': snippet,
                'product_name': product_name
            })
        logging.info(f"Extracted {len(result)} chunks from CSV: {csv_path}")
        return result
    except Exception as e:
        logging.error(f"❌ Error processing CSV {csv_path}: {e}")
        return []
