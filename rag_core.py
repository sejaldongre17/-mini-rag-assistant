from dotenv import load_dotenv
import os
from typing import List

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import logging
import requests

# -------------------------------------------
# Logging
# -------------------------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------------------------
# Embedding Model
# -------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# -------------------------------------------
# ChromaDB Client (Persistent)
# -------------------------------------------
client = PersistentClient(path="chroma_store")

# Load or create collection
collection = client.get_or_create_collection(name="pdf_knowledge_base")


# -------------------------------------------
# PDF TEXT EXTRACTION
# -------------------------------------------
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    all_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)
    return "\n".join(all_text)


# -------------------------------------------
# TEXT CHUNKING
# -------------------------------------------
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits long text into chunks with overlap.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


# -------------------------------------------
# ADD PDF TO VECTOR DATABASE
# -------------------------------------------
def add_pdf_to_vector_db(file_path: str, source_name: str):
    text = extract_text_from_pdf(file_path)
    chunks = split_text_into_chunks(text)

    embeddings = embedder.encode(chunks).tolist()

    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i}
                 for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )


# -------------------------------------------
# CLEAR DATABASE (THIS WAS MISSING)
# -------------------------------------------
def clear_vector_db():
    """
    Deletes the entire collection and recreates an empty one.
    Used for 'Clear Vector Database' button in Streamlit.
    """
    global collection
    try:
        client.delete_collection("pdf_knowledge_base")
    except Exception:
        pass  # ignore if collection doesn't exist

    collection = client.get_or_create_collection(name="pdf_knowledge_base")
    logging.info("Vector DB cleared and recreated.")


# -------------------------------------------
# RETRIEVE RELEVANT CHUNKS
# -------------------------------------------
def retrieve_relevant_chunks(query: str, top_k: int = 4) -> List[str]:
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]


# -------------------------------------------
# LLM CALL — GROQ
# -------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def call_llm(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers based only on the given context."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()

    if "choices" in response_json:
        return response_json["choices"][0]["message"]["content"]
    else:
        return f"LLM API Error: {response_json}"


# -------------------------------------------
# ANSWER QUESTION (RETURN ANSWER + CHUNKS)
# -------------------------------------------
def answer_question(query: str):
    chunks = retrieve_relevant_chunks(query, top_k=4)
    context = "\n\n".join(chunks)

    prompt = f"""
You are an AI Knowledge Assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {query}
Answer:
"""

    answer = call_llm(prompt)
    return answer, chunks
