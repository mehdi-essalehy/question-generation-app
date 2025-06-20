from nltk.tokenize import sent_tokenize
from sentence_transformers import util

# STEP 1 — Clean text
def clean_text(text):
    # Minimal cleaning — customize as needed
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())
    return text

# STEP 2 — Chunk text into windows
def chunk_text(text, window_size=3, overlap=1):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), window_size - overlap):
        chunk = " ".join(sentences[i:i + window_size])
        chunks.append(chunk)
    return chunks

# STEP 3 — Semantic filtering using embeddings
def semantic_filter(chunks, embedding_model, topic, threshold):
    reference_embedding = embedding_model.encode(topic, convert_to_tensor=True)
    filtered_chunks = []
    for chunk in chunks:
        embedding = embedding_model.encode(chunk, convert_to_tensor=True)
        score = util.cos_sim(embedding, reference_embedding).item()
        if score > threshold:
            filtered_chunks.append(chunk)
    return filtered_chunks

# STEP 4 — Entity filtering (optional, more precision)
def entity_filter(chunks, nlp, target_entities=None):
    if target_entities is None:
        target_entities = ["PERSON", "ORG", "GPE", "EVENT", "DATE", "PRODUCT"]

    filtered_chunks = []
    for chunk in chunks:
        doc = nlp(chunk)
        if any(ent.label_ in target_entities for ent in doc.ents):
            filtered_chunks.append(chunk)
    return filtered_chunks

# FULL PIPELINE
def segment_for_question_generation(raw_text, embedding_model, nlp=None, topic="general", window_size=3, overlap=1, semantic_threshold=0.4, entity_filtering=True):
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned, window_size, overlap)
    semantically_relevant = semantic_filter(chunks, embedding_model, topic, semantic_threshold)
    if entity_filtering:
        final_chunks = entity_filter(semantically_relevant, nlp)
    else:
        final_chunks = semantically_relevant
    return final_chunks

