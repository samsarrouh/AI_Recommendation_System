import nltk

def dynamic_chunk_text(text, chunk_size=500, chunk_overlap=50):
    """
    Splits the text into chunks using sentence boundaries so that each chunk
    is approximately chunk_size characters long without breaking sentences.
    Optionally, adds a small overlap between chunks.
    """
    # Ensure that NLTK's punkt tokenizer is available
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
    
    # Add overlap between chunks if desired.
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Take the last chunk_overlap characters from the previous chunk
                # (This is a simple approach; you might adjust to preserve whole words)
                overlap_text = chunks[i-1][-chunk_overlap:]
                chunk = overlap_text + " " + chunk
            overlapped_chunks.append(chunk)
        chunks = overlapped_chunks

    return chunks
